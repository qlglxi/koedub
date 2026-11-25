# -*- coding: utf-8 -*-
"""参考音频自动选取。

GPT-SoVITS 零样本克隆的音色全靠"参考音频 + 它对应的台词"这一对锚点，
所以锚点选得好不好直接决定成品像不像原声。人工挑一句当然最准，但每次换
音源都要挑就太累了，这里做全自动：

    时长舒适度   5.5~9.5 秒最理想，太短音色学不稳，太长带上语气变化
    信噪比       候选段相对整条录音的噪声地板
    浊音比       有明确基频的帧占比——专门排除气声、喘息、纯耳语
    共振峰丰富度 500~2500Hz 相对 100~500Hz 的能量——声带闭合程度的代理指标，
                 防止挑到"低频轰鸣但声带几乎不闭合"的耳边轻语
    语速密度     每秒实义字符数，排除拖长音、慢速咏唱

思路来自 asmr-dubber 的 voice_reference.py（MIT），按 koedub 的结构重写：
不引入 soundfile，全部走 ffmpeg + numpy；不做人声分离，用浊音比和共振峰
两项把关"是不是干净人声"。
"""
from __future__ import annotations

import os
import re
import shutil

import numpy as np

from . import audiocore

# ---- 打分参数（与原实现一致的权重）----------------------------------------
_PICK_MAX_CANDIDATES = 48

# GPT-SoVITS 的服务端硬校验：参考音频必须在 3~10 秒之间，否则 /tts 直接
# 返回 400「参考音频在3~10秒范围外，请更换！」。这两个常量以前抄的是
# asmr-dubber 的口径（2~20 秒），但那个项目用的是 IndexTTS，不吃这条限制。
# 上限留 0.4 秒余量给容器时长与重采样误差，下限同理。
_MIN_SEG = 3.4
_MAX_SEG = 9.6
_SOVITS_MIN_REF = 3.0     # 服务端真正的下限，不能再低
_SOVITS_MAX_REF = 10.0    # 服务端真正的上限

# 语速硬门槛（实义字符 / 秒）。日语正常说话 4~8 字/秒，这里 0.7 已经非常宽松。
# 设它是为了挡住 ASR 只识别出半句的情况：实测踩过一次，8 秒的段里只写了
# 「ちょっと」，这种段拿去当参考音频会把音色锚点污染成气声。
_MIN_CPS = 0.7
_MIN_CPS_LOOSE = 0.35     # 候选全被滤掉时放宽一次，宁可差一点也别整个放弃
_MIN_REF_SECONDS = _MIN_SEG   # 修剪后再短下去就凑不满 3 秒，宁可换窗口

# 整段扫描兜底：ASR 分段太碎（比如降级到 openai-whisper 后每句固定 2 秒）时
# 一个合格候选都选不出来，就改成在整条录音上开滑动窗口硬挑一段。
_FALLBACK_WIN = 6.0
_FALLBACK_STEP = 2.0
_FALLBACK_MIN_CHARS = 6   # 窗口里至少要有这么多实义字符才算"在说话"

_SPOKEN_CONTENT = re.compile(r"[A-Za-z0-9\u3041-\u3096\u30a1-\u30fa\u4e00-\u9fff]")
_STRETCH_MARKS = re.compile(r"[・ー〜～…⋯]+")

# 纯语气词/感叹，不能当参考句
_FILLER_CHARS = set("うんあえおんはいいえうわー～ぁぃぅぇぉっッん〜、。,.!?！？ ")


def _duration_score(seconds: float) -> float:
    if 5.5 <= seconds <= 9.5:
        return 1.0
    if 4.0 <= seconds < 5.5:
        return 0.88 + 0.12 * (seconds - 4.0) / 1.5
    if seconds < 4.0:
        return max(0.0, 0.88 * (seconds - 2.0) / 2.0)
    if seconds <= 12.0:
        return 1.0 - 0.12 * (seconds - 9.5) / 2.5
    return max(0.0, 0.88 - 0.88 * (seconds - 12.0) / 8.0)


def _speech_density_score(text: str, seconds: float) -> float:
    """奖励正常说话语速，惩罚拖长音。"""
    spoken = len(_SPOKEN_CONTENT.findall(text or ""))
    if seconds <= 0.0 or spoken == 0:
        return 0.0
    cps = spoken / seconds
    latin = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    is_latin = latin > spoken * 0.5
    lo_cps, good_cps = (3.5, 9.0) if is_latin else (1.4, 4.2)
    rate_score = max(0.0, min(1.0, (cps - lo_cps) / (good_cps - lo_cps)))
    stretch_chars = sum(len(m.group(0)) for m in _STRETCH_MARKS.finditer(text or ""))
    stretch_ratio = stretch_chars / max(spoken, 1)
    stretch_pen = max(0.0, 1.0 - max(0.0, stretch_ratio - 0.12) * 2.5)
    return rate_score * stretch_pen


def _modal_voiced_score(vr: float) -> float:
    """正常说话的浊音比落在 0.65~0.93 是平台区；1.00 那种是咏唱，不是说话。"""
    if vr <= 0.25:
        return 0.0
    if vr < 0.65:
        return (vr - 0.25) / 0.40
    if vr <= 0.93:
        return 1.0
    return max(0.70, 1.0 - (vr - 0.93) * 4.0)


def _is_filler(text: str) -> bool:
    """「うん」「あー」「はい」这类不能当锚点。"""
    compact = re.sub(r"[\s、。,.!?！？…・ー〜～]", "", text or "")
    if not compact:
        return True
    if len(compact) > 6:
        return False
    return all(ch in _FILLER_CHARS for ch in compact)


# ---- 信号特征 --------------------------------------------------------------
def _frame_levels(clip: np.ndarray, rate: int, frame_seconds: float = 0.02) -> np.ndarray:
    """每帧取最大声道 RMS。"""
    hop = max(1, round(rate * frame_seconds))
    arr = np.asarray(clip, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    count = len(arr) // hop
    if count == 0:
        return np.zeros(0)
    frames = arr[: count * hop].reshape(count, hop, arr.shape[1])
    return np.max(np.sqrt(np.mean(frames ** 2, axis=1)), axis=1)


def _louder_ear_mono(clip: np.ndarray) -> np.ndarray:
    """取能量大的那只耳——参考音频要单一音源，不能混双耳。"""
    if clip.ndim == 1 or clip.shape[1] == 1:
        return np.asarray(clip).reshape(-1)
    ear_rms = np.sqrt(np.mean(np.asarray(clip, dtype=np.float64) ** 2, axis=0))
    return np.asarray(clip)[:, int(np.argmax(ear_rms))]


def _noise_floor(pcm: np.ndarray, rate: int) -> float:
    levels = _frame_levels(pcm, rate)
    valid = levels[levels > 0]
    return float(np.percentile(valid, 10)) if valid.size else 1e-9


def _clip_snr_db(clip: np.ndarray, rate: int, floor: float) -> float | None:
    levels = _frame_levels(clip, rate)
    if not levels.size or levels.max() <= 0:
        return None
    active = levels[levels >= levels.max() * 0.1]
    return 20.0 * np.log10(max(float(np.mean(active)), 1e-9) / max(floor, 1e-9))


def _voiced_ratio(signal: np.ndarray, rate: int) -> float:
    """有明确基频的帧占比。耳语接近 0，正常说话 0.6~0.9。"""
    mono = np.asarray(signal, dtype=np.float64).reshape(-1)
    step = max(1, round(rate / 16000))
    if step > 1:
        usable = len(mono) // step * step
        mono = mono[:usable].reshape(-1, step).mean(axis=1)
    reduced = rate / step
    size, hop = round(reduced * 0.040), round(reduced * 0.010)
    if len(mono) < size:
        return 0.0
    count = 1 + (len(mono) - size) // hop
    index = np.arange(size)[None, :] + hop * np.arange(count)[:, None]
    frames = mono[index]
    energy = np.sqrt(np.mean(frames ** 2, axis=1))
    active = energy > max(float(energy.max()) * 0.1, 1e-6)
    if not active.any():
        return 0.0
    frames = frames[active] - frames[active].mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(frames * np.hanning(size), 4 * size, axis=1)
    correlation = np.fft.irfft(np.abs(spectrum) ** 2, axis=1)
    correlation /= np.maximum(correlation[:, :1], 1e-20)
    lags = slice(round(reduced / 400), round(reduced / 70))
    if lags.stop <= lags.start:
        return 0.0
    return float(np.mean(correlation[:, lags].max(axis=1) > 0.45))


_SIZE = 2048
_HOP = 512


def _long_term_spectrum(signal: np.ndarray) -> np.ndarray:
    arr = np.asarray(signal, dtype=np.float64).reshape(-1)
    padded = np.pad(arr, (_SIZE // 2, _SIZE))
    count = 1 + (len(padded) - _SIZE) // _HOP
    if count <= 0:
        return np.full(_SIZE // 2 + 1, 1e-20)
    index = np.arange(_SIZE)[None, :] + _HOP * np.arange(count)[:, None]
    power = np.abs(np.fft.rfft(padded[index] * np.sqrt(np.hanning(_SIZE)), axis=1)) ** 2
    level = power.sum(axis=1)
    if not len(level) or level.max() <= 0:
        return np.full(_SIZE // 2 + 1, 1e-20)
    active = level >= level.max() * 1e-3
    return power[active].mean(axis=0) + 1e-20


def _formant_richness_score(mono: np.ndarray, rate: int, vr: float) -> float:
    """500~2500Hz 共振峰能量 vs 100~500Hz 基频能量。浊音门控，噪声拿不到分。"""
    if len(mono) < max(64, rate // 20):
        return 0.0
    spec = _long_term_spectrum(mono)
    hz = np.fft.rfftfreq(_SIZE, 1.0 / rate)
    p_lo = float(spec[(hz >= 100.0) & (hz < 500.0)].sum())
    p_mid = float(spec[(hz >= 500.0) & (hz < 2500.0)].sum())
    mid_lo_db = 10.0 * np.log10((p_mid + 1e-20) / (p_lo + 1e-20))
    raw = max(0.0, min(1.0, (mid_lo_db + 18.0) / 18.0))
    voiced_gate = max(0.0, min(1.0, (vr - 0.25) / 0.35))
    return raw * voiced_gate


# ---- 主流程 ----------------------------------------------------------------
def _scan_windows(segments: list[dict], total: float) -> list[dict]:
    """ASR 分段碎到没法用时，在整条录音上开窗口硬挑。

    GPT-SoVITS 只认 3~10 秒的参考音频。降级到 openai-whisper 后它常把每句
    切成固定 2.00 秒，一个合格候选都选不出来——这条兜底就是为了不让整个
    配音阶段挂掉：按固定步长扫全片，把窗口内重叠的几句台词拼起来当 prompt。
    台词不必和音频逐字对齐，GPT-SoVITS 只用它对齐韵律。
    """
    out: list[dict] = []
    pos = 0.0
    while pos + _SOVITS_MIN_REF <= total and len(out) < _PICK_MAX_CANDIDATES:
        end = min(total, pos + _FALLBACK_WIN)
        if end - pos < _SOVITS_MIN_REF:      # 片尾那截太短，凑不出 3 秒
            break
        texts = []
        for s in segments:
            try:
                s0 = float(s.get("start") or 0)
                s1 = float(s.get("end") or 0)
            except (TypeError, ValueError):
                continue
            if min(end, s1) - max(pos, s0) > 0.4:      # 有实质重叠才算
                txt = str(s.get("text") or "").strip()
                if txt:
                    texts.append(txt)
        text = "".join(texts)
        if len(_SPOKEN_CONTENT.findall(text)) >= _FALLBACK_MIN_CHARS:
            out.append({"start": pos, "end": end, "duration": end - pos,
                        "text": text, "synthetic": True})
        pos += _FALLBACK_STEP
    return out


def _fit_range(start: float, end: float, total: float) -> tuple[float, float]:
    """把区间夹到 GPT-SoVITS 能吃的 3~10 秒。

    修剪静音后经常掉到 3 秒以下（尾音收得快），这时向两侧扩，而不是直接
    放弃——放弃就等于整个配音阶段失败，用户看到的只有一句「配音连续失败」。
    """
    if total <= 0:
        return start, end
    if end > total:
        end = total
    if start < 0:
        start = 0.0
    if end - start > _SOVITS_MAX_REF:
        end = start + _SOVITS_MAX_REF
    dur = end - start
    if dur < _SOVITS_MIN_REF:
        need = _SOVITS_MIN_REF - dur
        # 往前多给一点：说话起点通常在句中，前面那段一般比后面安静
        back = min(need * 0.6, start)
        start -= back
        fwd = min(need - back, total - end)
        end += fwd
        if end - start < _SOVITS_MIN_REF:      # 还是不够，贴着能用的边界凑
            start = max(0.0, min(start, total - _SOVITS_MIN_REF))
            end = min(total, start + _SOVITS_MIN_REF)
    return start, end


def pick_reference(input_path: str, segments: list[dict], cfg: dict, on_log) -> dict | None:
    """从原音频里挑最干净的一句，导出成 GPT-SoVITS 参考音频。

    返回 {"path", "promptText", "start", "end", "score", "detail"}，挑不出来返回 None。
    """
    t = cfg.get("tts", {}) or {}
    if not segments:
        on_log("  参考音频自动选取：没有可用句子")
        return None

    pcm = audiocore.decode_pcm(input_path, sample_rate=16000, channels=2)
    rate = 16000
    floor = _noise_floor(pcm, rate)
    total = len(pcm)

    cands = [s for s in segments
             if str(s.get("text") or "").strip()
             and _MIN_SEG <= float(s.get("duration") or 0) <= _MAX_SEG
             and not _is_filler(s.get("text"))]

    def _cps(s: dict) -> float:
        n = len(_SPOKEN_CONTENT.findall(str(s.get("text") or "")))
        return n / max(0.01, float(s.get("duration") or 0))

    dense = [s for s in cands if _cps(s) >= _MIN_CPS]
    if dense:
        if len(dense) < len(cands):
            on_log("  参考音频自动选取：%d 句候选，%d 句语速达标（滤掉 %d 句疑似识别不全的）"
                   % (len(cands), len(dense), len(cands) - len(dense)))
        cands = dense
    else:
        loose = [s for s in cands if _cps(s) >= _MIN_CPS_LOOSE]
        if loose:
            on_log("  参考音频自动选取：没有语速完全达标的，放宽到 %.2f 字/秒，剩 %d 句"
                   % (_MIN_CPS_LOOSE, len(loose)))
            cands = loose
        else:
            cands = []

    if not cands:
        # 一句合格的都没选出来：ASR 很可能把整段切得太碎（降级到
        # openai-whisper 后每句固定 2 秒就是这样）。别直接放弃——改用整段
        # 滑动窗口，只要是"在说话"的位置就能凑出一段合规的参考音频。
        on_log("  参考音频自动选取：没有 %.1f~%.1f 秒的合格句子，改用整段扫描"
               % (_MIN_SEG, _MAX_SEG))
        cands = _scan_windows(segments, total / float(rate))
        if not cands:
            on_log("  参考音频自动选取：整段扫描也没找到在说话的位置，"
                   "沿用配置里的参考音频")
            return None
        on_log("  参考音频自动选取：整段扫描出 %d 个候选窗口" % len(cands))
    else:
        # 先按时长+密度粗排，只精算前 48 个，控制耗时
        cands.sort(key=lambda s: -(_duration_score(float(s["duration"]))
                                   + 0.8 * _speech_density_score(str(s["text"]),
                                                                 float(s["duration"]))))
        cands = cands[:_PICK_MAX_CANDIDATES]
        on_log("  参考音频自动选取：%d 句候选，精算 %d 个" % (len(segments), len(cands)))

    best = None
    for seg in cands:
        lo = max(0, int(float(seg["start"]) * rate))
        hi = min(total, int(float(seg["end"]) * rate))
        if hi - lo < rate // 2:
            continue
        clip = pcm[lo:hi]
        snr = _clip_snr_db(clip, rate, floor)
        if snr is None:
            continue
        snr_score = snr / (snr + 20.0) if snr > 0 else 0.0
        mono = _louder_ear_mono(clip)
        vr = _voiced_ratio(mono, rate)
        dur = float(seg["duration"])
        score = (0.8 * _duration_score(dur)
                 + 0.4 * snr_score
                 + 0.6 * _modal_voiced_score(vr)
                 + 0.8 * _formant_richness_score(mono, rate, vr)
                 + 0.6 * _speech_density_score(str(seg["text"]), dur))
        detail = {"snrDb": round(snr, 1), "voicedRatio": round(vr, 3),
                  "duration": round(dur, 2)}
        if best is None or score > best["score"]:
            best = {"score": score, "seg": seg, "detail": detail}

    if best is None:
        on_log("  参考音频自动选取：候选全部不可用")
        return None

    seg = best["seg"]
    start = float(seg["start"])
    end = float(seg["end"])
    start, end = _trim_silence(pcm, rate, start, end)
    start, end = _fit_range(start, end, total / float(rate))
    if end - start < _SOVITS_MIN_REF:
        on_log("  参考音频自动选取：能用的只有 %.2fs，凑不满 %.1fs，放弃"
               % (end - start, _SOVITS_MIN_REF))
        return None

    out_dir = t.get("sovitsRefCache") or cfgmod_cache_dir(cfg)
    os.makedirs(out_dir, exist_ok=True)
    ref_path = os.path.join(out_dir, "ref_auto_%05.2f-%05.2f.wav" % (start, end))
    _export_ref(input_path, start, end, ref_path, pcm, rate,
                sample_rate=int(t.get("sovitsRefSampleRate") or 32000))

    prompt_text = str(seg.get("text") or "").strip()
    _write_sidecar(ref_path, prompt_text)
    on_log("  参考音频已选定：%.2f~%.2fs（%.2fs，评分 %.2f，SNR %.1fdB，浊音比 %.2f）"
           % (start, end, end - start, best["score"],
              best["detail"]["snrDb"], best["detail"]["voicedRatio"]))
    on_log("  参考台词：%s" % prompt_text)
    return {"path": ref_path, "promptText": prompt_text,
            "start": round(start, 3), "end": round(end, 3),
            "score": round(best["score"], 3), "detail": best["detail"]}


def cfgmod_cache_dir(cfg: dict) -> str:
    from . import config as cfgmod
    return os.path.join(cfgmod.WORK_DIR, "_refcache")


def _trim_silence(pcm: np.ndarray, rate: int, start: float, end: float,
                  rel_db: float = -40.0) -> tuple[float, float]:
    """用 20ms 包络把首尾的静音削掉，避免 prompt 与音频对不齐。"""
    lo = max(0, int(start * rate))
    hi = min(len(pcm), int(end * rate))
    if hi <= lo:
        return start, end
    levels = _frame_levels(pcm[lo:hi], rate)
    if not levels.size:
        return start, end
    peak = float(levels.max())
    if peak <= 0:
        return start, end
    thr = peak * (10 ** (rel_db / 20.0))
    active = np.where(levels >= thr)[0]
    if not active.size:
        return start, end
    step = 0.02
    new_lo = lo / rate + active[0] * step
    new_hi = lo / rate + (active[-1] + 1) * step
    # 两侧各留 0.03s，别把字头吞掉
    return max(start, new_lo - 0.03), min(end, new_hi + 0.03)


def _export_ref(src: str, start: float, end: float, dst: str,
                pcm: np.ndarray, rate: int, sample_rate: int = 32000) -> str:
    """从原文件精确切一段，单声道、目标采样率、峰值归一到 -3dBFS。"""
    lo = max(0, int(start * rate))
    hi = min(len(pcm), int(end * rate))
    peak = float(np.max(np.abs(pcm[lo:hi]))) if hi > lo else 0.0
    gain = 1.0
    if peak > 0:
        gain = min(2.0, 10 ** (-3.0 / 20.0) / peak)

    args = [audiocore.ffmpeg_path(), "-v", "error", "-y",
            "-ss", "%.3f" % start, "-to", "%.3f" % end, "-i", src,
            "-ac", "1", "-ar", str(sample_rate)]
    if abs(gain - 1.0) > 0.01:
        args += ["-af", "volume=%.3fdB" % (20.0 * np.log10(gain))]
    args += ["-c:a", "pcm_s16le", dst]
    cp = audiocore._run(args)
    if cp.returncode != 0 or not os.path.isfile(dst):
        raise RuntimeError("导出参考音频失败：%s"
                           % cp.stderr.decode("utf-8", "replace")[:300])
    return dst


def _write_sidecar(ref_path: str, prompt_text: str) -> None:
    """把参考句的台词存到同名 .txt —— 自检 / 二次使用不用重新识别，
    GPT-SoVITS 的 prompt_text 也需要它才能贴合参考音频的语调。"""
    if not prompt_text:
        return
    side = os.path.splitext(ref_path)[0] + ".txt"
    try:
        with open(side, "w", encoding="utf-8", newline="\n") as f:
            f.write(prompt_text.strip() + "\n")
    except OSError:
        pass
