# -*- coding: utf-8 -*-
"""排轴对齐：让每句中文配音严格吸附到原文对应的时间段。

规则（对应 cfg.align）：
  1. 每段可用空间 = min(原文时长, 到下一段开始前的空隙)，保证不与下一句抢时间。
  2. 配音比空间长 → 变速压缩（atempo，保持不变调），上限 maxTempo。
  3. 配音比空间短 → 补静音填充到空间长度，让下一句自然衔接。
  4. 超过 maxTempo 仍放不下 → 保留压缩后的内容并在末尾做短淡出，标记 overrun，
     同时把该段中文按语速预算二次裁剪，避免听感突兀。
  5. 支持全局位移与提前量（globalShiftMs / leadInMs）。
"""
from __future__ import annotations

import os
import time

import numpy as np

from . import audiocore

SR = 48000


def _fade_out(path: str, out_path: str, fade_ms: int = 120) -> str:
    return audiocore.convert(path, out_path, sample_rate=SR, channels=1,
                             filters="afade=t=out:st=%.3f:d=%.3f"
                                     % (0.0, fade_ms / 1000.0) + ",apad")


def plan(segments: list[dict], cfg: dict, total_duration: float = 0.0,
         on_log=None) -> list[dict]:
    """根据配音音频实际时长，计算每段的摆放时间与变速倍率。"""
    a = cfg.get("align", {})
    max_tempo = float(a.get("maxTempo", 1.45))
    min_gap = float(a.get("minGapMs", 60)) / 1000.0
    lead_in = float(a.get("leadInMs", 0)) / 1000.0
    shift = float(a.get("globalShiftMs", 0)) / 1000.0
    tail_pad = float(a.get("tailPaddingMs", 80)) / 1000.0

    ordered = [s for s in segments if s.get("dubPath")]
    ordered.sort(key=lambda s: s["start"])
    plans: list[dict] = []
    overrun = 0

    for i, s in enumerate(ordered):
        nxt = ordered[i + 1] if i + 1 < len(ordered) else None
        start = max(0.0, s["start"] + shift + lead_in)
        span_end = s["end"]
        if nxt is not None:
            span_end = min(span_end, nxt["start"] - min_gap)
        span = max(0.05, span_end - s["start"])

        src_dur = float(s.get("dubDuration") or 0.0)
        if src_dur <= 0:
            s["dubStatus"] = "empty"
            continue

        tempo = 1.0
        clipped = False
        if src_dur > span + 0.02:
            need = src_dur / span
            tempo = min(need, max_tempo)
            clipped = need > max_tempo

        final_dur = src_dur / tempo
        # 如果放得下，允许在末尾补静音到 span（让节奏贴原声）
        play_dur = min(final_dur, span)
        if final_dur < span - 0.02:
            play_dur = span          # 补静音填满，保持与原文同节奏
        if clipped:
            overrun += 1

        # 排轴不能伸到音频长度之外。伸出去的部分混音时会按总长裁掉，等于
        # 白合成一段，日志里还会留下一个骗人的 lastEnd（实测出现过 95.8s 排
        # 在 90s 音频上）。优先把起点往前挪；挪不开就把这句压进去。
        if total_duration > 0 and start + play_dur > total_duration + 0.05:
            prev_end = (plans[-1]["end"] + min_gap) if plans else 0.0
            start = max(0.0, total_duration - play_dur)
            if start < prev_end:
                room = total_duration - prev_end
                if room < 0.25:
                    if on_log:
                        on_log("  第 %d 段排到音频末尾之外，前面又没有位置，跳过" % s["id"])
                    overrun += 1
                    continue
                start = prev_end
                need = src_dur / room
                if need > tempo:
                    tempo = min(need, max_tempo)
                    clipped = clipped or need > max_tempo
                play_dur = min(src_dur / tempo, room)
        if total_duration > 0:
            play_dur = min(play_dur, max(0.05, total_duration - start))

        plans.append({
            "id": s["id"],
            "start": round(start, 4),
            "span": round(span, 4),
            "srcDuration": round(src_dur, 3),
            "tempo": round(tempo, 4),
            "playDuration": round(play_dur, 4),
            "end": round(start + play_dur, 4),
            "overrun": clipped,
            "delayMs": round(start * 1000.0, 1),
        })
    return plans


def render_clips(segments: list[dict], plans: list[dict], out_dir: str,
                 cfg: dict, on_log) -> dict:
    """按排轴把每段配音渲染成「变速 + 补静音」的成品片段。"""
    a = cfg.get("align", {})
    max_cps = cfg.get("translate", {}).get("maxCharsPerSecond", 7.5)
    ready_dir = os.path.join(out_dir, "aligned")
    os.makedirs(ready_dir, exist_ok=True)
    by_id = {s["id"]: s for s in segments}
    rendered: dict[int, str] = {}
    t0 = time.time()

    total_trimmed = 0
    for p in plans:
        seg = by_id.get(p["id"])
        if not seg:
            continue
        src = seg["dubPath"]
        dst = os.path.join(ready_dir, "aligned_%04d.mp3" % p["id"])
        filters = []
        if abs(p["tempo"] - 1.0) > 0.01:
            filters.append(audiocore.atempo_filter(p["tempo"]))
        # 补静音到 playDuration，让每段成品长度精确等于排轴长度
        filters.append("apad=whole_dur=%.4f" % max(0.02, p["playDuration"]))
        audiocore.convert(src, dst, sample_rate=SR, channels=1,
                          filters=",".join(filters),
                          extra=["-c:a", "libmp3lame", "-b:a", "160k"])
        rendered[p["id"]] = dst

        if p["overrun"]:
            # 二次裁剪文本预算，为下一轮/字幕提示提供依据
            budget = max(2, int(p["span"] * max_cps * 1.0))
            if len(seg.get("zh", "")) > budget:
                total_trimmed += 1
            p["suggestedMaxChars"] = budget

    on_log("排轴完成：%d 段摆放，%d 段需压缩到上限（建议精简文案），用时 %.1fs"
           % (len(rendered), total_trimmed, time.time() - t0))
    return {"rendered": rendered, "trimmed": total_trimmed,
            "elapsed": round(time.time() - t0, 2)}


def coverage_report(plans: list[dict], segments: list[dict],
                    total_duration: float) -> dict:
    """排轴质量报告：覆盖率、重叠、压缩分布。"""
    if not plans:
        return {"segments": 0}
    starts = [p["start"] for p in plans]
    ends = [p["end"] for p in plans]
    overlaps = 0
    for i in range(1, len(plans)):
        if plans[i]["start"] < plans[i - 1]["end"] - 0.01:
            overlaps += 1
    tempos = [p["tempo"] for p in plans]
    spans = [p["span"] for p in plans]
    played = sum(p["playDuration"] for p in plans)
    shift = min(starts) if starts else 0.0
    return {
        "segments": len(plans),
        "dubbedSeconds": round(played, 2),
        "audioSeconds": round(total_duration, 2),
        "coverageRatio": round(played / total_duration, 4) if total_duration else 0.0,
        "firstStart": round(shift, 3),
        "lastEnd": round(max(ends), 3),
        "overlaps": overlaps,
        "overrunSegments": sum(1 for p in plans if p["overrun"]),
        "maxTempo": round(max(tempos), 3),
        "meanTempo": round(float(np.mean(tempos)), 3),
        "meanSpanSeconds": round(float(np.mean(spans)), 3),
    }
