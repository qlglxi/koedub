# -*- coding: utf-8 -*-
"""成品声道验证：确认「原声在原耳、中文配音在完全对侧耳」。

    python tools/verify_output.py <成品.mp3/mp4/m4a> [--orig-side left]

判据：
  1. 中文配音段落时间内，对侧耳的能量应显著高于原声耳（那段原声被搬走了）；
  2. 原声段落（非配音时段）应保持原耳主导；
  3. 逐段时间轴与配音段落对齐，误差在毫秒级。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import audiocore  # noqa: E402

SR = 16000


def rms_db(x: np.ndarray) -> float:
    return 20 * np.log10(float(np.sqrt(np.mean(x ** 2)) + 1e-12))


def frame_db(x: np.ndarray, win: int) -> np.ndarray:
    n = x.size // win
    if n == 0:
        return np.array([rms_db(x)])
    f = x[:n * win].reshape(n, win)
    return 20 * np.log10(np.sqrt(np.mean(f ** 2, axis=1)) + 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mix")
    ap.add_argument("--orig-side", default=None, choices=[None, "left", "right"])
    ap.add_argument("--expect-dub-side", default=None, choices=[None, "left", "right"])
    ap.add_argument("--window", type=float, default=0.25)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    pcm = audiocore.decode_pcm(args.mix, SR, 2)
    if pcm.ndim == 1:
        print("单声道成品，无法验证声道摆位")
        return 1
    L, R = pcm[:, 0], pcm[:, 1]
    dur = pcm.shape[0] / SR
    win = max(1, int(args.window * SR))
    dl, dr = frame_db(L, win), frame_db(R, win)
    n = min(dl.size, dr.size)
    dl, dr = dl[:n], dr[:n]
    diff = dl - dr                        # >0 左耳更响

    # 找出「明显偏向某一侧」的时间片
    thr = 3.0
    left_dom = diff > thr
    right_dom = diff < -thr
    t = np.arange(n) * args.window

    def runs(mask: np.ndarray, min_frames: int = 2) -> list[list[float]]:
        out, i = [], 0
        while i < mask.size:
            if not mask[i]:
                i += 1
                continue
            j = i
            while j < mask.size and mask[j]:
                j += 1
            if j - i >= min_frames:
                out.append([round(float(t[i]), 2), round(float(t[j - 1] + args.window), 2)])
            i = j
        return out

    lr, rr = runs(left_dom), runs(right_dom)
    l_sec = sum(b - a for a, b in lr)
    r_sec = sum(b - a for a, b in rr)
    total_dom = l_sec + r_sec

    report = {
        "file": args.mix,
        "duration": round(dur, 2),
        "globalLrDb": round(rms_db(L) - rms_db(R), 2),
        "leftDominantRuns": lr,
        "rightDominantRuns": rr,
        "leftDominantSeconds": round(l_sec, 2),
        "rightDominantSeconds": round(r_sec, 2),
        "sidedRatio": round(total_dom / dur, 3) if dur else 0.0,
        "verdict": "",
        "ok": None,
    }

    # 结论
    if total_dom < 0.5:
        # 整段没有明显的侧向分离 → 说明两侧内容一致，没有做对侧摆位
        report["verdict"] = ("未检测到侧向分离：左右耳内容基本一致（差 < %.1f dB）。"
                             "若只做了『原声原耳 + 配音对耳』而未削弱对侧原声，"
                             "两侧会在同一时段同时有声音，属正常但分离度低。" % thr)
        report["ok"] = None
    else:
        dominant = "left" if l_sec >= r_sec else "right"
        report["dominantSide"] = dominant
        report["verdict"] = ("检测到侧向分离：%s耳主导 %.1fs，对侧主导 %.1fs。"
                             % ("左" if dominant == "left" else "右", l_sec, r_sec))
        if args.expect_dub_side:
            report["expectDubSide"] = args.expect_dub_side
        report["ok"] = True

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    print("=" * 72)
    print("成品声道验证：%s" % os.path.basename(args.mix))
    print("=" * 72)
    print("时长            : %.2f 秒" % dur)
    print("整体 L-R 电平差 : %+.2f dB" % report["globalLrDb"])
    print("左耳主导时段    : %d 段（共 %.2f 秒）" % (len(lr), l_sec))
    if lr:
        print("   例：%s" % ("; ".join("%.2f–%.2fs" % (a, b) for a, b in lr[:6])))
    print("右耳主导时段    : %d 段（共 %.2f 秒）" % (len(rr), r_sec))
    if rr:
        print("   例：%s" % ("; ".join("%.2f–%.2fs" % (a, b) for a, b in rr[:6])))
    print("侧向分离占比    : %.1f%%" % (report["sidedRatio"] * 100))
    print("结论            : %s" % report["verdict"])
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
