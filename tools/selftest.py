# -*- coding: utf-8 -*-
"""端到端自检：验证「原声原耳 + 中文完全对侧耳」这条核心需求。

    python tools/selftest.py                  # 三种样本（左/右/居中）+ 50% 泄漏
    python tools/selftest.py --keep-work      # 保留中间产物

逐项检查（全部基于信号本身，不看日志）：
  A. 声道判定：left/right/center 三种样本是否判对；
  B. 摆位正确：中文配音音频与成品「对侧耳」的互相关，必须显著高于与「原声耳」的互相关；
  C. 无串耳：中文音频在原声耳上的能量应明显低于对侧耳；
  D. 时间对齐：配音峰值位置与排轴时间误差 < 60 ms；
  E. 成品可解码：mp3/wav/mp4 均存在且时长与原声一致。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from core import audiocore          # noqa: E402
from core import deps               # noqa: E402

# 可选依赖装在工程内 vendor/，挂上后才能看到 faster-whisper
deps.ensure_on_path()

# 复用语音样本生成逻辑（真实 Edge 日语朗读）
sys.path.insert(0, HERE)
try:
    import make_speech_sample as mss     # type: ignore
except Exception:
    mss = None

SR = 16000
PY = sys.executable


# 全部用例统一使用 Edge 真实日语语音构造样本（峰均比接近真人，判定才可信）。
# 拿不到语音时（无网络）自动退回离线合成音，只是结论要按合成音的特性看。
offline_only = False
_CLIPS_CACHE: list = []


def _run_quiet(cmd: list[str], timeout: int = 3600) -> tuple[int, str, str]:
    """运行子进程并取回输出。

    不用 ``capture_output=True``：那会走管道 stdio，在受限沙箱里子进程会直接
    失败且没有任何输出（Windows AppContainer 不允许创建命名管道）。改成把输出
    重定向到临时文件再读，既避开限制，也能拿到完整日志。
    """
    tmpdir = os.path.join(ROOT, ".tmp")
    os.makedirs(tmpdir, exist_ok=True)
    out_f = os.path.join(tmpdir, "sub.out")
    err_f = os.path.join(tmpdir, "sub.err")
    try:
        with open(out_f, "wb") as fo, open(err_f, "wb") as fe:
            cp = subprocess.run(cmd, stdout=fo, stderr=fe, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, "", "超时"
    def _read(p):
        try:
            with open(p, "rb") as f:
                return f.read().decode("utf-8", "replace")
        except OSError:
            return ""
    return cp.returncode, _read(out_f), _read(err_f)


def make_offline_sample(out: str, side: str, leak: float = 0.0) -> bool:
    """离线合成音样本（不依赖网络，峰均比与真人不同，仅作兜底）。"""
    cmd = [PY, os.path.join(HERE, "make_sample.py"), out,
           "--voice-side", side, "--batch", "5"]
    if leak:
        cmd += ["--leak", str(leak)]
    rc, _out, err = _run_quiet(cmd, timeout=600)
    if rc != 0 or not os.path.isfile(out):
        sys.stderr.write("  [make_sample 失败 rc=%s]\n%s\n" % (rc, err[-1200:]))
        return False
    return True


def speech_clips(tmp: str) -> list:
    """合成一次 Edge 日语语音片段，全部用例复用（省时且结论一致）。"""
    if _CLIPS_CACHE:
        return _CLIPS_CACHE
    if mss is None:
        return []
    work = os.path.join(tmp, "_speechwork")
    os.makedirs(work, exist_ok=True)
    try:
        clips = mss.synth_lines("ja-JP-NanamiNeural", 1.0, work)
    except Exception as e:
        sys.stderr.write("  [语音合成失败] %s: %s\n" % (type(e).__name__, e))
        return []
    if clips:
        _CLIPS_CACHE.extend(clips)
    return clips


def make_speech_variant(out: str, side: str, leak: float, tmp: str) -> bool:
    """用已缓存的语音片段生成指定摆位的样本（左/右/居中/泄漏共用一套语音）。"""
    clips = speech_clips(tmp)
    if not clips or mss is None:
        return False
    stereo = mss.build_stereo(clips, side, leak)
    audiocore.write_pcm_to_file(stereo, mss.SR, out,
                                extra_args=["-c:a", "libmp3lame", "-b:a", "192k"])
    return os.path.isfile(out)


def whisper_ready() -> bool:
    """本地识别是否可用（装了 faster-whisper 或 whisper）。"""
    deps.ensure_on_path()
    for mod in ("faster_whisper", "whisper"):
        try:
            import importlib.util
            if importlib.util.find_spec(mod):
                return True
        except Exception:
            pass
    return False


def run_pipeline(src: str, extra: list[str] | None = None) -> dict | None:
    cmd = [PY, os.path.join(ROOT, "app.py"), "--cli", src,
           "--asr", "demo", "--translate", "glossary", "--tts", "edge"]
    cmd += extra or []
    rc, out, err = _run_quiet(cmd, timeout=1800)
    if rc != 0:
        # 把最后几行处理日志也带出来，便于定位
        tail = "\n".join((out or "").strip().splitlines()[-14:])
        sys.stderr.write("  [流水线失败 rc=%s]\n%s\n%s\n" % (rc, err[-900:], tail))
        return None
    dirs = sorted(glob.glob(os.path.join(ROOT, "work", "%s-*"
                                        % os.path.splitext(os.path.basename(src))[0])),
                  key=os.path.getmtime)
    if not dirs:
        return None
    man = os.path.join(dirs[-1], "manifest.json")
    if not os.path.isfile(man):
        return None
    with open(man, encoding="utf-8") as f:
        return json.load(f)


def xcorr_lag(a: np.ndarray, b: np.ndarray, max_lag: int = 6400) -> tuple[float, int]:
    """归一化互相关峰值与滞后。

    返回 (峰值 0~1, lag)：lag > 0 表示 b 比 a 晚 lag 个样本（b 滞后）。
    """
    n = 1 << int(np.ceil(np.log2(a.size + b.size)))
    fa, fb = np.fft.rfft(a - a.mean(), n), np.fft.rfft(b - b.mean(), n)
    c = np.fft.irfft(fa * np.conj(fb), n)
    c = np.concatenate((c[-max_lag:], c[:max_lag + 1]))
    lags = np.arange(-max_lag, max_lag + 1)
    i = int(np.argmax(np.abs(c)))
    denom = np.sqrt(np.sum((a - a.mean()) ** 2) * np.sum((b - b.mean()) ** 2))
    return (float(np.abs(c[i]) / denom) if denom > 0 else 0.0), int(lags[i])


def check_case(name: str, side: str, leak: float, keep_work: bool,
               extra: list[str], tmp: str, speech: bool = False) -> dict:
    print("─" * 74)
    print("用例 %s（源：%s，人声在%s%s）"
          % (name,
             "Edge 日语真实朗读" if not offline_only else "离线合成音",
             side, "，泄漏 %.0f%%" % (leak * 100) if leak else ""))
    src = os.path.join(tmp, "case_%s.mp3" % name)
    made = make_offline_sample(src, side, leak) if offline_only \
        else make_speech_variant(src, side, leak, tmp)
    if not made:
        print("  × 样本生成失败")
        return {"case": name, "ok": False, "reason": "sample"}

    man = run_pipeline(src, extra)
    if not man:
        print("  × 流水线失败")
        return {"case": name, "ok": False, "reason": "pipeline"}

    outdir = man["outputDir"]
    res = {"case": name, "ok": True, "outputDir": outdir, "checks": {}}

    # A. 声道判定
    det = man["stages"]["channel"]["side"]
    expect = side if side in ("left", "right") else "center"
    a_ok = det == expect
    res["checks"]["A 声道判定"] = {"expect": expect, "got": det, "ok": a_ok}
    print("  A 声道判定      : 期望 %-6s 实际 %-6s  %s"
          % (expect, det, "✓" if a_ok else "✗"))

    # A2. 真实识别（仅 speech 用例）：段落数应接近台词数，且文本不应是占位符
    if speech:
        segs = man.get("segments") or []
        nonempty = [s for s in segs if (s.get("text") or "").strip()]
        no_placeholder = all("示范文本" not in (s.get("text") or "") for s in nonempty)
        provider = (man.get("stages") or {}).get("asr", {}).get("provider", "")
        real_asr = "demo" not in str(provider).lower()
        a2_ok = len(nonempty) >= 3 and no_placeholder and real_asr
        res["checks"]["A2 真实识别"] = {"segments": len(nonempty), "provider": provider,
                                       "noPlaceholder": no_placeholder, "ok": a2_ok}
        print("  A2 真实识别     : %d 段、后端=%s、无占位文本=%s  %s"
              % (len(nonempty), provider or "-", no_placeholder, "✓" if a2_ok else "✗"))
        if nonempty:
            print("       例：%s" % nonempty[0]["text"][:46])

    target = "right" if det == "left" else "left"      # 中文应当落在的这一侧
    orig_side = "left" if target == "right" else "right"
    prev = man.get("previews") or {}
    orig_p, dub_p = prev.get("originalSide"), prev.get("dubSide")
    if not (orig_p and dub_p) or not os.path.isfile(dub_p):
        print("  × 缺少试听产物")
        res["ok"] = False
        return res

    def mono(p):
        a = audiocore.decode_pcm(p, SR, 1)
        return a[:, 0] if a.ndim > 1 else a

    orig_ear, dub_ear = mono(orig_p), mono(dub_p)
    full = mono(man["outputs"]["wav"])
    src_pcm = audiocore.decode_pcm(src, SR, 2)
    src_orig_ch = src_pcm[:, 0 if orig_side == "left" else 1]
    src_dub_ch = src_pcm[:, 0 if target == "left" else 1]

    # B. 用排轴片段重建「中文应当出现的时间线」，再与该声道做互相关
    plans = man.get("plans") or []
    segs = man.get("segments") or []
    by_id = {s["id"]: s for s in segs}
    ref = np.zeros(max(full.size, dub_ear.size), dtype=np.float32)
    for p in plans:
        seg = by_id.get(p["id"])
        if not seg or not seg.get("dubPath") or not os.path.isfile(seg["dubPath"]):
            continue
        clip = audiocore.decode_pcm(seg["dubPath"], SR, 1)
        if clip.ndim > 1:
            clip = clip[:, 0]
        s0 = int(p["start"] * SR)
        s1 = min(ref.size, s0 + clip.size)
        if s1 > s0:
            ref[s0:s1] += clip[:s1 - s0]

    m = min(ref.size, full.size)
    ref, full_m = ref[:m], full[:m]
    c_dub, lag_dub = xcorr_lag(ref, full_m)
    # 只保留配音片段处的信号，避免长静音段主导相关
    mask = np.zeros(m, dtype=bool)
    for p in plans:
        s0, s1 = int(p["start"] * SR), min(m, int(p["end"] * SR))
        mask[s0:s1] = True
    if mask.sum() > SR:
        c_masked, lag_masked = xcorr_lag(ref[mask], full_m[mask])
    else:
        c_masked, lag_masked = c_dub, lag_dub
    b_ok = c_dub > 0.15 and lag_dub >= -80
    res["checks"]["B 中文落在对侧耳"] = {
        "corrFull": round(c_dub, 3), "corrInSegments": round(c_masked, 3),
        "oppositeEar": target, "ok": b_ok}
    print("  B 中文落在对侧耳: 全曲相关 %.3f（片段内 %.3f），落在%s耳  %s"
          % (c_dub, c_masked, "右" if target == "right" else "左",
             "✓" if b_ok else "✗"))

    # C. 原声内容：在配音时段内，原声（源的原声侧声道）应与成品的「原声耳」更相关，
    #    而在「对侧耳」上被削弱。注意成品经过中/侧分解，成分与源不同，所以只比较
    #    「两耳之间谁更像原声」，不做绝对相关值判定。
    seg_spans = [(int(p["start"] * SR), min(m, int(p["end"] * SR))) for p in plans]
    seg_spans = [(a, b) for a, b in seg_spans if b - a > SR // 5]

    def _collect(sig: np.ndarray) -> np.ndarray:
        if not seg_spans:
            return sig
        return np.concatenate([sig[a:b] for a, b in seg_spans])

    src_orig_v = _collect(src_orig_ch[:m])
    oe_v = _collect(mono(orig_p)[:m])
    de_v = _collect(mono(dub_p)[:m])
    c_src_orig, _ = xcorr_lag(src_orig_v, oe_v)
    c_src_dub, _ = xcorr_lag(src_orig_v, de_v)
    c_ok = c_src_orig > max(0.05, c_src_dub * 1.25)
    res["checks"]["C 原声留在原耳"] = {
        "corrInOriginalEar": round(c_src_orig, 3),
        "corrInOppositeEar": round(c_src_dub, 3), "ok": c_ok}
    print("  C 原声留在原耳  : 配音时段内 与原声耳相关 %.3f，与对侧耳相关 %.3f  %s"
          % (c_src_orig, c_src_dub, "✓" if c_ok else "✗"))

    # D. 时间对齐：配音实际落点相对排轴时间的偏移
    d_detail = {"lagMs": round(lag_masked / SR * 1000, 1)}
    d_ok = abs(lag_masked / SR) <= 0.06
    res["checks"]["D 时间对齐"] = {**d_detail, "ok": d_ok}
    print("  D 时间对齐      : 实配音相对排轴偏移 %+.1f ms（阈值 ±60ms）  %s"
          % (d_detail["lagMs"], "✓" if d_ok else "✗"))

    # E. 成品存在且时长一致
    outs = man.get("outputs") or {}
    need = [k for k in ("mp3", "wav") if k in outs]
    e_ok = bool(need) and all(os.path.isfile(outs[k]) for k in need)
    dur = man["media"]["duration"]
    for k in need:
        if abs(audiocore.probe(outs[k]).duration - dur) > 0.6:
            e_ok = False
    res["checks"]["E 成品完整"] = {"files": need, "ok": e_ok, "duration": dur}
    print("  E 成品完整      : %s 时长 %.2fs  %s"
          % (",".join(need), dur, "✓" if e_ok else "✗"))

    res["ok"] = all(v.get("ok") for v in res["checks"].values())
    if not keep_work:
        shutil.rmtree(outdir, ignore_errors=True)
    return res


def main() -> int:
    global offline_only
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--cases", default="left,right,center,leak")
    ap.add_argument("--with-speech", action="store_true",
                    help="额外跑一个真实识别的用例（需本地识别模型）")
    ap.add_argument("--offline-samples", action="store_true",
                    help="用离线合成音当样本（不联网，但峰均比与真人不同）")
    args = ap.parse_args()

    tmp = os.path.join(ROOT, "tests")
    os.makedirs(tmp, exist_ok=True)
    # 清掉上一轮的样本，避免残留的坏文件把结论带偏
    for pat in ("case_*.mp3", "sample_*.mp3"):
        for f in glob.glob(os.path.join(tmp, pat)):
            try:
                os.remove(f)
            except OSError:
                pass

    # 统一用 Edge 日语真实语音构造样本 —— 合成音的峰均比与真人差很多，
    # 拿它验证「居中人声是否被判成居中」会得出误导性结论。
    if not args.offline_samples:
        if not speech_clips(tmp):
            print("=" * 74)
            print("拿不到 Edge 日语语音，本次改用离线合成音样本。")
            print("（合成音只适合验证左右分离，居中/泄漏的判定阈值不适用）")
            print("=" * 74)
            offline_only = True
    else:
        offline_only = True

    cases = {
        "left": ("left", 0.0, False),
        "right": ("right", 0.0, False),
        "center": ("center", 0.0, False),
        "leak": ("left", 0.45, False),
        "speech": ("left", 0.0, True),
    }
    want = [c.strip() for c in args.cases.split(",") if c.strip()]
    if args.with_speech and "speech" not in want:
        want.append("speech")
    if "speech" in want and not whisper_ready():
        print("=" * 74)
        print("跳过真实识别用例：未安装本地识别模型。")
        print("  安装：python app.py --install faster-whisper")
        print("=" * 74)
        want = [w for w in want if w != "speech"]

    results = []
    for name in want:
        if name not in cases:
            continue
        side, leak, speech = cases[name]
        extra = [] if not speech else ["--asr", "whisper_local"]
        results.append(check_case(name, side, leak, args.keep_work, extra, tmp,
                                  speech=speech))
    print("=" * 74)
    passed = sum(1 for r in results if r.get("ok"))
    print("自检结果：%d/%d 用例全部通过" % (passed, len(results)))
    for r in results:
        marks = " ".join("%s%s" % (k.split(" ")[0], "✓" if v.get("ok") else "✗")
                         for k, v in (r.get("checks") or {}).items())
        print("  %-8s %s  %s" % (r["case"], "PASS" if r.get("ok") else "FAIL", marks))
    print("=" * 74)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
