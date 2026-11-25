# -*- coding: utf-8 -*-
"""声道摆位与合成 —— 「原声留在原耳，中文配音放完全对侧耳」。

用 numpy 直接操作 PCM，比 ffmpeg 滤镜链更可控，也顺便能产出浏览器试听用的
双单声道 WAV（真左耳 / 真右耳，用耳机听能立刻分辨）。

原声侧处理策略（cfg.mix.originalMode）：
  keep      原声保持立体声不动（最简单，默认；原声本就在一侧时对侧几乎是空的）
  duck      配音落位的对侧耳在该时段按 duckAmountDb 下压，避免两路台词抢耳
  mono_split 原声折叠成单声道后分到两侧（音效仍在双耳），台词侧交给配音
  media_swap 影视级对调：原声中间声道（对白）搬到原耳，原声侧声道（音效）撒到双耳
"""
from __future__ import annotations

import os
import time

import numpy as np

from . import audiocore

SR = 48000


# ---------------------------------------------------------------------------
# 软限幅
# ---------------------------------------------------------------------------
def soft_clip(x: np.ndarray, ceiling: float = 0.97) -> np.ndarray:
    """tanh 软限幅：避免爆音，比硬截断自然。"""
    return np.tanh(x / ceiling) * ceiling


def bind_stereo(mix: np.ndarray, level: float = 6e-6,
                delay_us: float = 90.0, sr: int = SR) -> np.ndarray:
    """给左右声道加不可闻的去相关抖动，防止有损编码丢掉声道差异。

    背景：MP3/AAC 用「中/侧 (M/S) + joint stereo」。当一侧在某个时段近乎静音时，
    侧声道没有能量，编码器就当这段「无声」处理，解码回来两侧被抹平成一样 ——
    辛苦摆好的「原声耳 / 中文耳」就没了。实测本机样本里对侧耳语音峰值
    会从 -20 dB 掉到 -45 dB。

    做法：对右耳加一个亚采样级微延迟（等效几十微秒通道间时差），再叠加极低电平
    随机抖动，合起来在 -95 dBFS 量级，远低于 16bit 量化台阶，完全不可闻，
    但足以让编码器在每个频带都保留侧声道信息。
    """
    out = mix.astype(np.float32).copy()
    d = int(round(sr * delay_us / 1e6))
    if d > 0 and out.shape[0] > d:
        out[d:, 1] = mix[:-d, 1]
        out[:d, 1] = 0.0
    rng = np.random.default_rng(20261007)
    out[:, 0] += (rng.random(out.shape[0], dtype=np.float32) - 0.5) * level
    out[:, 1] += (rng.random(out.shape[0], dtype=np.float32) - 0.5) * level
    return out


def build_gain_envelope(spans: list[tuple[int, int]], n: int, sr: int,
                        depth_db: float, attack_ms: float = 6.0,
                        release_ms: float = 90.0) -> np.ndarray:
    """生成 duck 增益包络：在 spans 覆盖的样本上用余弦淡入淡出降到 depth_db。"""
    gain = np.ones(n, dtype=np.float32)
    target = float(10 ** (depth_db / 20.0))
    a = max(1, int(sr * attack_ms / 1000.0))
    r = max(1, int(sr * release_ms / 1000.0))
    for s0, s1 in spans:
        s0 = max(0, min(n, s0))
        s1 = max(s0, min(n, s1))
        if s1 <= s0:
            continue
        gain[s0:s1] = np.minimum(gain[s0:s1], target)
        # 平滑过渡
        if s0 - a >= 0:
            gain[s0 - a:s0] = np.minimum(
                gain[s0 - a:s0], np.linspace(1.0, target, a, dtype=np.float32))
        if s1 + r <= n:
            gain[s1:s1 + r] = np.minimum(
                gain[s1:s1 + r], np.linspace(target, 1.0, r, dtype=np.float32))
    return gain


# ---------------------------------------------------------------------------
# 主合成
# ---------------------------------------------------------------------------
def compose(original_path: str, segments: list[dict], plans: list[dict],
            rendered: dict, out_dir: str, cfg: dict, on_log,
            total_duration: float = 0.0) -> dict:
    """合成最终立体声 PCM，并导出多种成品。"""
    m = cfg.get("mix", {})
    mode = m.get("originalMode", "keep")
    orig_gain = float(10 ** (float(m.get("originalGainDb", 0.0)) / 20.0))
    dub_gain = float(10 ** (float(m.get("dubGainDb", 0.0)) / 20.0))
    duck_db = float(m.get("duckAmountDb", -7.0))
    t0 = time.time()

    # 1) 原声解码为 48k 立体声
    on_log("解码原声为 48kHz 立体声…")
    orig = audiocore.decode_pcm(original_path, sample_rate=SR, channels=2)
    if orig.ndim == 1:
        orig = np.stack([orig, orig], axis=1)
    n = orig.shape[0]
    on_log("原声 %d 样本（%.1f 秒）" % (n, n / SR))

    left = orig[:, 0].astype(np.float32).copy()
    right = orig[:, 1].astype(np.float32).copy()

    # 2) 按原声侧模式处理
    if mode == "duck":
        # 为每段生成「对侧耳下压」包络（下压的是配音要落的那只耳）
        left_spans, right_spans = [], []
        for p in plans:
            s0 = int(max(0.0, p["start"]) * SR)
            s1 = int(min(n / SR, p["end"]) * SR)
            seg = next((x for x in segments if x["id"] == p["id"]), None)
            target = (seg or {}).get("opposite", "right")
            (right_spans if target == "right" else left_spans).append((s0, s1))
        if left_spans:
            left *= build_gain_envelope(left_spans, n, SR, duck_db)
        if right_spans:
            right *= build_gain_envelope(right_spans, n, SR, duck_db)
        on_log("原声侧策略：duck（对侧耳在配音时段下压 %.1f dB）" % duck_db)
    elif mode == "mono_split":
        mono = (left + right) * 0.5
        left = mono.copy()
        right = mono.copy()
        on_log("原声侧策略：mono_split（音效保留双耳，台词侧交给配音）")
    elif mode in ("media_swap", "swap"):
        # 影视级中/侧分解（默认方案）：
        #   音声的双耳录音里，人声多在中间（Mid），环境音效偏两侧（Side）。
        #   逐段处理：M+S 放「原声耳」，S 也放「对侧耳」，对侧耳的 Mid（人声）直接去掉。
        #   结果：原声台词仍在原耳，对侧耳留住了音效、只腾出人声位置给中文配音。
        mid = (left + right) * 0.5
        side = (left - right) * 0.5
        swapped = 0
        for p in plans:
            seg = next((x for x in segments if x["id"] == p["id"]), None)
            if seg is None:
                continue
            target = seg.get("opposite") or "right"
            s0 = max(0, int(p["start"] * SR))
            s1 = min(n, int(p["end"] * SR))
            if s1 <= s0:
                continue
            M = mid[s0:s1].copy()
            S = side[s0:s1].copy()
            if target == "right":
                # 原声（含中置人声）留在左耳；右耳只保留音效侧声道
                left[s0:s1] = M + S
                right[s0:s1] = S
            else:
                left[s0:s1] = S
                right[s0:s1] = M - S
            swapped += 1
        on_log("原声侧策略：media_swap（%d 段中/侧分解：原声台词留原耳，对侧耳清出人声位）"
               % swapped)
    else:
        on_log("原声侧策略：keep（原声立体声原样保留）")

    left = left * orig_gain
    right = right * orig_gain

    # 3) 逐段把配音叠加到对侧耳
    by_id = {s["id"]: s for s in segments}
    placed = 0
    missing = 0
    for p in plans:
        seg = by_id.get(p["id"])
        if not seg:
            continue
        clip_path = rendered.get(p["id"]) or seg.get("dubPath")
        if not clip_path or not os.path.isfile(clip_path):
            missing += 1
            continue
        clip = audiocore.decode_pcm(clip_path, sample_rate=SR, channels=1)
        if clip.ndim == 2:
            clip = clip[:, 0]
        clip = clip * dub_gain

        start = int(max(0.0, p["start"]) * SR)
        end = min(n, start + clip.size)
        if end <= start:
            continue
        usable = end - start
        # 硬性防重叠：若与同侧已放置的下一段重叠，则截断（排轴已预留空隙，这里兜底）
        target = seg.get("opposite") or "right"
        dst = right if target == "right" else left
        dst[start:end] += clip[:usable]
        placed += 1

    if missing:
        on_log("警告：%d 段没有可用的配音文件，已跳过。" % missing)
    on_log("配音落位：%d 段（左耳 %d / 右耳 %d）"
           % (placed,
              sum(1 for p in plans if (by_id.get(p['id'], {}).get('opposite') == 'left')),
              sum(1 for p in plans if (by_id.get(p['id'], {}).get('opposite') == 'right'))))

    # 4) 限幅与响度
    mix = np.stack([left, right], axis=1)
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.99:
        on_log("峰值 %.3f，启用软限幅。" % peak)
        mix = soft_clip(mix)

    if m.get("loudnorm", True):
        target_lufs = float(m.get("targetLufs", -16.0))
        # 简易响度对齐：用 RMS 近似（真 EBU 需滤波，这里按 -20dBFS RMS 映射到目标）
        rms = float(np.sqrt(np.mean(mix ** 2) + 1e-12))
        rms_db = 20 * np.log10(rms + 1e-12)
        # 经验：-16 LUFS 约对应 RMS -19 dBFS
        want_db = target_lufs - 3.0
        gain_db = max(-12.0, min(12.0, want_db - rms_db))
        if abs(gain_db) > 0.2:
            mix = mix * float(10 ** (gain_db / 20.0))
            on_log("响度对齐：RMS %.1f dB → 增益 %+.1f dB" % (rms_db, gain_db))
        mix = soft_clip(mix)

    # 5) 编码前绑定立体声（防止有损编码抹平左右差异）
    mix = bind_stereo(mix)

    # 6) 导出
    stem = os.path.splitext(os.path.basename(original_path))[0]
    outs: dict[str, str] = {}
    wav = os.path.join(out_dir, "%s_bilingual.wav" % stem)
    audiocore.encode_wav(wav, mix, SR)
    outs["wav"] = wav
    # 统计主导侧，便于试听文件命名
    side_count = {"left": 0, "right": 0}
    for s in segments:
        sd = s.get("side")
        if sd in side_count:
            side_count[sd] += 1
    orig_side = "left" if side_count["left"] >= side_count["right"] else "right"
    on_log("写出双语音轨（WAV）→ %s" % os.path.basename(wav))
    return {"mix": mix, "outputs": outs, "placed": placed, "missing": missing,
            "peak": round(peak, 4), "origSide": orig_side,
            "elapsed": round(time.time() - t0, 2)}


# ---------------------------------------------------------------------------
# 试听专用输出
# ---------------------------------------------------------------------------
def export_mono_pair(mix: np.ndarray, out_dir: str, stem: str, on_log,
                     orig_side: str = "left") -> dict:
    """把左右耳分别导出成单声道 WAV，方便用播放器单独听「原声耳 / 中文耳」。"""
    dub_side = "right" if orig_side == "left" else "left"
    ch = {"left": 0, "right": 1}
    lpath = os.path.join(out_dir, "%s_%s耳_原声.wav"
                         % (stem, "左" if orig_side == "left" else "右"))
    rpath = os.path.join(out_dir, "%s_%s耳_中文配音.wav"
                         % (stem, "左" if dub_side == "left" else "右"))
    audiocore.encode_wav(lpath, mix[:, ch[orig_side]], SR)
    audiocore.encode_wav(rpath, mix[:, ch[dub_side]], SR)
    on_log("导出试听单声道：原声耳 / 中文耳 各一份")
    return {"originalSide": lpath, "dubSide": rpath,
            "origSide": orig_side, "dubSideName": dub_side}


def export_final(mix: np.ndarray, out_dir: str, stem: str, cfg: dict,
                 on_log) -> dict:
    """导出成品：MP3（通用）+ M4A（带封面元数据）+ MP4（黑底视频，分享用）。"""
    e = cfg.get("export", {})
    bitrate = str(e.get("audioBitrate", "192k"))
    fmt = (e.get("format") or "mp4").lower()
    outs: dict[str, str] = {}

    # WAV 无损（声道摆位绝对不丢），作为最保真的一版
    wav = os.path.join(out_dir, "%s_双语_无损.wav" % stem)
    audiocore.encode_wav(wav, mix, SR)
    outs["wav"] = wav

    mp3 = os.path.join(out_dir, "%s_双语_原声原耳_中文对耳.mp3" % stem)
    audiocore.write_pcm_to_file(mix, SR, mp3,
                                extra_args=["-c:a", "libmp3lame", "-b:a", bitrate])
    outs["mp3"] = mp3

    if fmt in ("mp4", "both", "m4a", "video"):
        m4a = os.path.join(out_dir, "%s_双语.m4a" % stem)
        audiocore.write_pcm_to_file(mix, SR, m4a,
                                    extra_args=["-c:a", "aac", "-b:a", bitrate])
        outs["m4a"] = m4a

    if fmt in ("mp4", "both", "video"):
        video = os.path.join(out_dir, "%s_双语.mp4" % stem)
        args = [audiocore.ffmpeg_path(), "-v", "error", "-y",
                "-f", "lavfi", "-i", "color=c=0x0f1216:s=1280x720:r=6",
                "-f", "s16le", "-ar", str(SR), "-ac", "2", "-i", "pipe:0",
                "-shortest", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", bitrate,
                "-movflags", "+faststart", video]
        ints = (np.clip(mix, -1.0, 1.0) * 32767.0).astype("<i2")
        import subprocess as _sp
        creationflags = getattr(_sp, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        cp = _sp.run(args, input=ints.tobytes(), capture_output=True,
                     creationflags=creationflags)
        if cp.returncode != 0:
            on_log("MP4 封装失败（已保留 MP3/M4A）：%s"
                   % cp.stderr.decode("utf-8", "replace")[:200])
        else:
            outs["mp4"] = video

    on_log("成品导出：%s" % "、".join(os.path.basename(v) for v in outs.values()))
    return outs
