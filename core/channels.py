# -*- coding: utf-8 -*-
"""左右耳声源判定（Stereo Source Localization）。

用途：判断音声源里「人声」主要落在左耳还是右耳（还是居中/双耳），
从而决定中文配音应该放到哪一侧 —— 规则是「完全对侧」。

判定思路（三重证据，互相校验）：
  1. 能量差 (ILD)：语音帧上的 L/R 能量比，取对数后求中位数，符号给方向。
  2. 一致性：偏向某一侧的语音帧占比，衡量稳定度。
  3. 互相关 (ITD)：L/R 的归一化互相关峰值偏移样本数，正值代表左耳先到，
     用于区分「真·侧向声源」和「居中但被混响糊开」。

输出 face: "left" | "right" | "center"
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

FRAME_MS = 20.0
HOP_MS = 10.0


@dataclass
class SideResult:
    side: str                 # left / right / center
    confidence: float         # 0~1
    opposite: str             # 配音应放置的声道
    mean_delta_db: float      # 左减右的平均电平差, >0 偏左
    median_delta_db: float
    left_frames: int
    right_frames: int
    center_frames: int
    consistency: float        # 主导侧帧占比
    itd_samples: int          # 互相关峰值偏移
    itd_ms: float
    voice_frame_ratio: float  # 有语音的帧比例
    verdict_note: str

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("confidence", "mean_delta_db", "median_delta_db", "consistency",
                  "itd_ms", "voice_frame_ratio"):
            d[k] = round(float(d[k]), 4)
        return d


def _frame_energy(x: np.ndarray, sr: int, frame_ms: float = FRAME_MS,
                  hop_ms: float = HOP_MS) -> np.ndarray:
    """分帧 RMS 能量。x: (n,) -> (frames,)"""
    fl = max(1, int(sr * frame_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    if x.size < fl:
        return np.array([float(np.sqrt(np.mean(x ** 2) + 1e-12))], dtype=np.float32)
    n = 1 + (x.size - fl) // hop
    idx = np.arange(fl)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx]
    return np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12).astype(np.float32)


def _xcorr_lag(left: np.ndarray, right: np.ndarray, sr: int,
               max_lag_ms: float = 1.2) -> tuple[int, float]:
    """在语音最响的片段上做归一化互相关，返回 (lag_samples, peak)。

    lag > 0 表示左耳先收到（声源偏左）。

    注意边界：分析窗可能比延迟范围还短（例如逐段判侧时的短句），此时可用延迟
    必须收缩到窗长以内，否则 lags 与相关系数序列长度不一致会越界。
    """
    win = min(left.size, int(sr * 1.2))
    if win < 64:
        return 0, 0.0
    max_lag = max(1, min(int(sr * max_lag_ms / 1000.0), win - 1))

    le = _frame_energy(left, sr)
    if le.size == 0:
        return 0, 0.0
    center = int(np.argmax(le)) * int(sr * HOP_MS / 1000.0)
    a = max(0, min(center - win // 2, left.size - win))
    l = left[a:a + win].astype(np.float64)
    r = right[a:a + win].astype(np.float64)
    l = l - l.mean()
    r = r - r.mean()
    if np.sqrt((l ** 2).sum()) < 1e-9 or np.sqrt((r ** 2).sum()) < 1e-9:
        return 0, 0.0
    n = 1 << int(np.ceil(np.log2(l.size + r.size)))
    corr = np.fft.irfft(np.fft.rfft(l, n) * np.conj(np.fft.rfft(r, n)), n)
    corr = np.concatenate((corr[-(max_lag + 1):], corr[:max_lag + 1]))
    lags = np.arange(-max_lag, max_lag + 1)
    if lags.size != corr.size:                 # 兜底，正常不会触发
        m = min(lags.size, corr.size)
        lags, corr = lags[:m], corr[:m]
    peak_i = int(np.argmax(np.abs(corr)))
    denom = np.sqrt((l ** 2).sum() * (r ** 2).sum())
    return int(lags[peak_i]), float(corr[peak_i] / denom) if denom else 0.0


def analyze(pcm: np.ndarray, sr: int, frame_ms: float = FRAME_MS) -> SideResult:
    """分析立体声 PCM，判定人声所在侧。

    pcm: (n, 2) float32。单声道输入按 center 处理。
    """
    if pcm.ndim == 1:
        pcm = pcm[:, None]
    if pcm.shape[1] == 1:
        return SideResult("center", 0.0, "right", 0.0, 0.0, 0, 0, 0, 0.0, 0, 0.0, 0.0,
                          "单声道输入，无侧向信息；中文配音默认放到右耳。")

    left = pcm[:, 0].astype(np.float64)
    right = pcm[:, 1].astype(np.float64)
    el = _frame_energy(left, sr, frame_ms)
    er = _frame_energy(right, sr, frame_ms)
    n = min(el.size, er.size)
    el, er = el[:n], er[:n]

    total_rms = np.sqrt(np.mean((left ** 2 + right ** 2) / 2.0) + 1e-12)
    # 语音帧门槛：整体 RMS 的 0.6 倍，且高于静音底噪
    noise = float(np.percentile(np.maximum(el, er), 10))
    thr = max(total_rms * 0.6, noise * 2.2, 1e-5)
    voiced = np.maximum(el, er) > thr
    voice_ratio = float(voiced.mean()) if n else 0.0
    if voiced.sum() < 5:
        # 语音太少：退回全段统计
        voiced = np.ones(n, dtype=bool)

    eps = 1e-9
    # 逐帧电平差（dB，仅用于展示与中位数统计）
    delta_db = 20.0 * np.log10((el + eps) / (er + eps))
    vd = delta_db[voiced]
    median_db = float(np.median(vd))

    # 方向判定用「能量加权的线性差值」而不是 dB 的均值：
    # dB 在某一侧近乎静音的帧上会飙到 ±100 dB，把结论带偏。
    #   D = Σ w·(El − Er) / Σ w·(El + Er)，w 为帧总能量 → 落在 [−1, 1]，+ 表示偏左
    #
    # 注意必须用 float64 计算：_frame_energy 返回 float32，而 float32 的
    # 最小正规数约 1e-38，给近静音帧加 1e-12 相当于没加，相减后全是 0，
    # 结果会退化成 ±60 dB 的垃圾值（实测会把「人声在右」判成「在左」）。
    elv = el[voiced].astype(np.float64)
    erv = er[voiced].astype(np.float64)
    if elv.size == 0:
        return SideResult("center", 0.0, "right", 0.0, 0.0, 0, 0, 0, 0.0, 0, 0.0,
                          voice_ratio, "没有可用的语音帧，按居中处理。")
    w = elv + erv + 1e-18
    d_lin = float(np.sum(w * (elv - erv)) / np.sum(w * w))
    d_lin = max(-1.0, min(1.0, d_lin))
    # 注意保留符号：一侧明显占优时 |d_lin| 会贴近 1，此时用 ±60dB 表示「极强偏向」。
    # 这里若写成常数 60.0 就丢了负号，会把「右耳」判成「左耳」。
    if abs(d_lin) >= 0.999:
        side_db = 60.0 if d_lin > 0 else -60.0
    else:
        side_db = 20.0 * np.log10((1.0 + d_lin) / (1.0 - d_lin))

    # 方向一致性：去掉极端帧后，强偏帧里方向一致的比例
    order = np.sort(np.abs(vd))
    cut = float(order[min(order.size - 1, int(order.size * 0.75))])
    strong = vd[np.abs(vd) >= cut] if cut > 0 else vd
    if strong.size:
        pos = float(np.mean(strong > 0))
        direction_consistency = max(pos, 1.0 - pos)
    else:
        direction_consistency = 0.0

    # 0.6 dB 是「明显偏向一侧」的经验门槛
    band = 0.6
    left_frames = int(np.sum(vd > band))
    right_frames = int(np.sum(vd < -band))
    center_frames = int(vd.size - left_frames - right_frames)
    dominant = max(left_frames, right_frames)
    consistency = float(dominant / max(1, vd.size))

    lag, peak = _xcorr_lag(left, right, sr)
    itd_ms = lag / float(sr) * 1000.0

    side = "center"
    note = ""
    if abs(side_db) > 0.8:
        side = "left" if side_db > 0 else "right"
        if direction_consistency < 0.6:
            note = "方向一致性偏低（%.0f%%），结果仅供参考" % (direction_consistency * 100)
    else:
        side = "center"
        note = "人声在两侧基本等量（双耳混音），按约定：原声留左、中文放右"

    # 互相关交叉校验：真·侧向声源会有 ~0.3ms 以上的 ITD
    if side in ("left", "right") and abs(itd_ms) >= 0.25:
        itd_side = "left" if itd_ms > 0 else "right"
        if itd_side != side:
            note = (note + "；" if note else "") + \
                "能量倾向%s但时差倾向%s，属混响/相位造成的假侧向，已按能量判定" % (
                    "左" if side == "left" else "右", "左" if itd_side == "left" else "右")

    strength = min(1.0, abs(side_db) / 6.0)
    confidence = float(min(1.0, 0.40 * direction_consistency + 0.40 * strength
                           + 0.20 * min(1.0, abs(itd_ms) / 0.6)))
    if side == "center":
        confidence = float(max(0.0, 1.0 - abs(side_db) / band) * 0.5
                           * (1.0 - direction_consistency * 0.5))

    opposite = {"left": "right", "right": "left", "center": "right"}[side]
    return SideResult(
        side=side,
        confidence=confidence,
        opposite=opposite,
        mean_delta_db=side_db,
        median_delta_db=side_db,
        left_frames=left_frames,
        right_frames=right_frames,
        center_frames=center_frames,
        consistency=consistency,
        itd_samples=lag,
        itd_ms=itd_ms,
        voice_frame_ratio=voice_ratio,
        verdict_note=note or ("人声稳定偏向%s" % ("左耳" if side == "left" else
                                             "右耳" if side == "right" else "居中/双耳")),
    )


def analyze_segments(pcm: np.ndarray, sr: int, segments: list[dict],
                     reference: SideResult | None = None,
                     min_confidence: float = 0.6,
                     min_seconds: float = 0.35) -> list[dict]:
    """对每个语音段落单独判定所在侧，用于「段落间换耳」的音声。

    重要：逐段结果**只在足够可信时才覆盖整体判定**。原因很实际 —— 句子之间的
    停顿/纯音效片段往往一侧近乎静音，逐段算出的电平差会飙到 ±60 dB，看起来
    "非常确定"，其实是噪声。若照单全收，那段中文就会被放到原声那一耳上，直接
    把原声压掉。所以这里有三道闸门：时长、置信度、以及是否与整体判定冲突。

    segments: [{"start", "end", ...}]，原地附加 side/opposite/sideConfidence。
    reference: 整段音频的判定结果，作为先验。
    """
    if pcm.ndim == 1:
        pcm = pcm[:, None]
    ref_side = reference.side if reference is not None else None
    ref_opposite = reference.opposite if reference is not None else "right"

    for seg in segments:
        a = max(0, int(seg["start"] * sr))
        b = min(pcm.shape[0], int(seg["end"] * sr))
        span = (b - a) / float(sr)
        if span < max(0.15, min_seconds):
            seg.update(side=ref_side or "center", opposite=ref_opposite,
                       sideConfidence=0.0, sideDeltaDb=0.0,
                       sideOverride=False)
            continue

        r = analyze(pcm[a:b], sr)
        seg["sideConfidence"] = round(r.confidence, 3)
        seg["sideDeltaDb"] = round(r.median_delta_db, 2)

        # 闸门：置信度够高、且该侧不是「居中」，才允许覆盖整体判定；
        # 与整体判定冲突时要求更高置信度，避免被停顿段带偏。
        keep = True
        if r.side == "center":
            keep = False
        elif r.confidence < min_confidence:
            keep = False
        elif ref_side and r.side != ref_side and r.confidence < 0.78:
            keep = False

        if keep:
            seg["side"] = r.side
            seg["opposite"] = r.opposite
            seg["sideOverride"] = True
        else:
            seg["side"] = ref_side or r.side
            seg["opposite"] = ref_opposite
            seg["sideOverride"] = False
    return segments


def voice_activity(pcm: np.ndarray, sr: int, frame_ms: float = FRAME_MS,
                   hop_ms: float = HOP_MS) -> tuple[np.ndarray, np.ndarray, float]:
    """能量法 VAD（与语言无关）。

    返回 (voiced_mask, frame_rms_db, threshold_db)。
    """
    if pcm.ndim == 2 and pcm.shape[1] > 1:
        mono = pcm.mean(axis=1)
    elif pcm.ndim == 2:
        mono = pcm[:, 0]
    else:
        mono = pcm
    fl = max(1, int(sr * frame_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    if mono.size < fl:
        return np.array([False]), np.array([-120.0]), -60.0
    n = 1 + (mono.size - fl) // hop
    idx = np.arange(fl)[None, :] + hop * np.arange(n)[:, None]
    frames = mono[idx]
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-12)
    peak = float(np.percentile(db, 95))
    floor = float(np.percentile(db, 20))
    # 自适应阈值：底噪与峰值之间，取 35% 处，且不低于底噪 +6dB
    thr = max(floor + 6.0, peak - 22.0, floor + (peak - floor) * 0.35)
    voiced = db > thr
    return voiced, db, float(thr)
