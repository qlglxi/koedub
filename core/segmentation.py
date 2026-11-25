# -*- coding: utf-8 -*-
"""语音分段（VAD）+ 时间轴骨架构建。

不依赖语言模型：用能量法 VAD 找出发声区间，合并/切分到适合排轴的粒度，
得到 [{"start", "end"}] 列表。ASR 的返回时间戳会与之合并，避免逐句切碎。
"""
from __future__ import annotations

import numpy as np

from . import channels


def detect_regions(pcm, sr: int, min_segment: float = 0.5,
                   max_segment: float = 12.0, merge_gap: float = 0.35,
                   pad: float = 0.08,
                   split_gap: float = 0.0) -> list[dict]:
    """从 PCM 里找出连续发声区间。

    - merge_gap: 小于该间隔的静音会被并进同一段（一句话内的换气）
    - split_gap: 大于该间隔的停顿一定切开（句与句之间）。
      默认取 merge_gap 的 1.4 倍并至少 0.45s —— 音声里「句子之间」的停顿
      通常明显长于「句子内部」的换气，这个双重阈值能避免把整段独白并成一块。
    - max_segment: 超过则强制切分（长独白拆成可排轴的块）
    """
    if split_gap <= 0:
        split_gap = max(0.45, merge_gap * 1.4)
    voiced, db, thr = channels.voice_activity(pcm, sr)
    hop = 0.01  # channels.HOP_MS

    # 先用 merge_gap 把「句内换气」粘起来
    regions: list[dict] = []
    i = 0
    n = voiced.size
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        j = i
        gap = 0
        last_voiced = i
        while j < n:
            if voiced[j]:
                last_voiced = j
                gap = 0
            else:
                gap += 1
                if gap * hop > merge_gap:
                    break
            j += 1
        start = max(0.0, i * hop - pad)
        end = min(n * hop, (last_voiced + 1) * hop + pad)
        regions.append({"start": start, "end": end})
        i = j + 1

    merged: list[dict] = []
    for r in regions:
        if merged and r["start"] - merged[-1]["end"] < merge_gap and \
                (r["end"] - merged[-1]["start"]) < max_segment:
            merged[-1]["end"] = r["end"]
        else:
            merged.append(dict(r))

    # 再用 split_gap 把「句间长停顿」切开
    split: list[dict] = []
    for r in merged:
        cur_start = r["start"]
        a = int(cur_start / hop)          # hop = 0.01s → 秒转帧
        b = min(n, int(r["end"] / hop))
        k = a
        while k < b:
            if voiced[k]:
                k += 1
                continue
            m = k
            while m < b and not voiced[m]:
                m += 1
            if (m - k) * hop >= split_gap and k > a and m < b:
                split.append({"start": cur_start, "end": k * hop})
                cur_start = m * hop
            k = m
        split.append({"start": cur_start, "end": r["end"]})

    # 过短片段并入前一段；过长片段在「能量谷」处切分
    final: list[dict] = []
    for r in split:
        dur = r["end"] - r["start"]
        if dur < min_segment:
            if final and (r["end"] - final[-1]["start"]) < max_segment:
                final[-1]["end"] = r["end"]
            continue
        if dur > max_segment:
            final.extend(_split_at_valleys(r, db, hop, max_segment, min_segment))
        else:
            final.append(r)
    # 去除时间重叠
    out: list[dict] = []
    for r in final:
        if out and r["start"] < out[-1]["end"]:
            r["start"] = out[-1]["end"]
        if r["end"] - r["start"] >= 0.05:
            out.append(r)
    return out


def _split_at_valleys(region: dict, db: np.ndarray, hop: float,
                      max_segment: float, min_segment: float) -> list[dict]:
    """把过长的发声区间切开：每次在「距离理想切点最近的能量谷」下刀。

    比固定静音阈值稳健：连读的独白没有真静音，但一定有能量低谷（换气处）。
    """
    pieces = [dict(region)]
    guard = 0
    while guard < 200:
        guard += 1
        over = [p for p in pieces if (p["end"] - p["start"]) > max_segment]
        if not over:
            break
        target = max(over, key=lambda p: p["end"] - p["start"])
        lo, hi = target["start"], target["end"]
        mid = (lo + hi) / 2.0
        # 在区间中段 ±35% 里找能量最低的一帧
        span = (hi - lo) * 0.35
        a = max(int((mid - span) / hop), int(lo / hop) + 1)
        b = min(int((mid + span) / hop), int(hi / hop) - 1)
        if b <= a:
            cut = mid
        else:
            seg = db[a:b]
            smooth = np.convolve(seg, np.ones(9, dtype=np.float32) / 9, mode="same")
            cut = (a + int(np.argmin(smooth))) * hop
        cut = min(max(cut, lo + min_segment), hi - min_segment)
        pieces.remove(target)
        pieces.append({"start": lo, "end": cut})
        pieces.append({"start": cut, "end": hi})
    pieces.sort(key=lambda p: p["start"])
    return pieces


def merge_asr_with_regions(asr_segments: list[dict],
                           regions: list[dict],
                           min_segment: float = 0.4) -> list[dict]:
    """把 ASR 结果与 VAD 骨架合并。

    优先信任 ASR 的时间戳（它更贴字），但用 VAD 修正明显漂移：
    ASR 段落若与相邻段落重叠或间隔过大，用 VAD 边界夹紧。
    """
    if not asr_segments:
        return [{"start": r["start"], "end": r["end"], "text": "",
                 "source": "vad"} for r in regions]
    segs = []
    for s in asr_segments:
        st = float(s.get("start", 0.0))
        en = float(s.get("end", st + 0.5))
        if en - st < 0.05:
            en = st + 0.05
        segs.append({"start": st, "end": en, "text": (s.get("text") or "").strip(),
                     "source": "asr",
                     "words": s.get("words") or []})
    segs.sort(key=lambda x: x["start"])

    # 夹紧到 VAD 区间范围内（若 ASR 整体漂移超过 0.6s 才动手）
    if regions:
        drift = abs(segs[0]["start"] - regions[0]["start"])
        if drift > 0.6:
            offset = regions[0]["start"] - segs[0]["start"]
            for s in segs:
                s["start"] += offset
                s["end"] += offset

    for s in segs:
        s["start"] = max(0.0, s["start"])
        s["end"] = max(s["start"] + 0.05, s["end"])

    # 合并被切碎的极短句
    merged: list[dict] = []
    for s in segs:
        if merged and (s["end"] - merged[-1]["start"]) < 1.0 and \
                not merged[-1]["text"].endswith(("。", "！", "？", "…")):
            merged[-1]["end"] = s["end"]
            merged[-1]["text"] = (merged[-1]["text"] + s["text"]).strip()
        else:
            merged.append(s)
    for i, s in enumerate(merged):
        s["id"] = i + 1
        s["duration"] = round(s["end"] - s["start"], 3)
    return merged
