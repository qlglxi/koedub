# -*- coding: utf-8 -*-
"""字幕导出与校验（SRT / ASS，中日双语）。

排轴阶段产出的每段「摆放时间」直接变成字幕时间轴，因此字幕和配音天然一致。
"""
from __future__ import annotations

import os

from . import audiocore


def build_cues(segments: list[dict], plans: list[dict] | None = None) -> list[dict]:
    """合并原文/译文/排轴时间，得到字幕条目。"""
    by_id = {p["id"]: p for p in (plans or [])}
    cues: list[dict] = []
    for s in sorted(segments, key=lambda x: x["start"]):
        p = by_id.get(s["id"])
        start = p["start"] if p else s["start"]
        end = p["end"] if p else s["end"]
        if end <= start:
            end = start + max(1.0, float(s.get("duration") or 1.0))
        cues.append({
            "id": s["id"],
            "start": round(start, 3),
            "end": round(end, 3),
            "ja": (s.get("text") or "").strip(),
            "zh": (s.get("zh") or "").strip(),
            "side": s.get("side", "center"),
            "dubSide": s.get("opposite", "right"),
            "overrun": bool(p and p.get("overrun")),
        })
    return cues


def to_srt(cues: list[dict], bilingual: bool = True) -> str:
    lines = []
    for i, c in enumerate(cues, 1):
        lines.append(str(i))
        lines.append("%s --> %s" % (audiocore.srt_timestamp(c["start"]),
                                    audiocore.srt_timestamp(c["end"])))
        if bilingual and c["ja"]:
            lines.append(c["ja"])
        if c["zh"]:
            lines.append(c["zh"])
        elif c["ja"]:
            lines.append(c["ja"])
        lines.append("")
    return "\n".join(lines)


ASS_HEADER = """[Script Info]
Title: {title}
ScriptType: v4.00+
WrapStyle: 0
PlayResX: 1280
PlayResY: 720
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: JP,Microsoft YaHei,30,&H00C8C8C8,&H000000FF,&H00202020,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,60,60,70,134
Style: ZH,Microsoft YaHei,40,&H00FFFFFF,&H000000FF,&H00202020,&H80000000,1,0,0,0,100,100,0,0,1,2,1,2,60,60,30,134
Style: TAG,Microsoft YaHei,22,&H0080C0FF,&H000000FF,&H00202020,&H80000000,0,0,0,0,100,100,0,0,1,1,0,8,20,20,20,134

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def to_ass(cues: list[dict], bilingual: bool = True, title: str = "koedub",
           show_side_tag: bool = True) -> str:
    out = [ASS_HEADER.format(title=title)]
    for c in cues:
        st = audiocore.ass_timestamp(c["start"])
        en = audiocore.ass_timestamp(c["end"])
        if bilingual and c["ja"]:
            ja = c["ja"].replace("\n", "\\N")
            out.append("Dialogue: 0,%s,%s,JP,,0,0,0,,%s" % (st, en, ja))
        zh = (c["zh"] or c["ja"]).replace("\n", "\\N")
        if zh:
            out.append("Dialogue: 0,%s,%s,ZH,,0,0,0,,%s" % (st, en, zh))
        if show_side_tag:
            tag = "原声%s耳 · 中文%s耳%s" % (
                "左" if c["side"] == "left" else "右" if c["side"] == "right" else "居中",
                "左" if c["dubSide"] == "left" else "右",
                " · 已压缩" if c["overrun"] else "")
            out.append("Dialogue: 0,%s,%s,TAG,,0,0,0,,%s" % (st, en, tag))
    return "\n".join(out) + "\n"


def run(segments: list[dict], plans: list[dict], out_dir: str, stem: str,
        cfg: dict, on_log) -> dict:
    sub_cfg = cfg.get("subtitle", {})
    if not sub_cfg.get("enabled", True):
        on_log("字幕导出已关闭。")
        return {}
    cues = build_cues(segments, plans)
    bilingual = bool(sub_cfg.get("bilingual", True))
    paths: dict[str, str] = {}
    srt = os.path.join(out_dir, "%s.zh.srt" % stem)
    with open(srt, "w", encoding="utf-8") as f:
        f.write(to_srt(cues, bilingual=False))
    paths["srt_zh"] = srt
    if bilingual:
        srt2 = os.path.join(out_dir, "%s.bilingual.srt" % stem)
        with open(srt2, "w", encoding="utf-8") as f:
            f.write(to_srt(cues, bilingual=True))
        paths["srt_bi"] = srt2
    ass = os.path.join(out_dir, "%s.ass" % stem)
    with open(ass, "w", encoding="utf-8") as f:
        f.write(to_ass(cues, bilingual=bilingual, title=stem))
    paths["ass"] = ass
    on_log("字幕导出：%s" % "、".join(os.path.basename(p) for p in paths.values()))

    if sub_cfg.get("burnVideo"):
        wav = os.path.join(out_dir, "%s_bilingual.wav" % stem)
        if os.path.isfile(wav):
            vid = os.path.join(out_dir, "%s_字幕版.mp4" % stem)
            try:
                audiocore.render_subtitles(wav, ass, vid)
                paths["video"] = vid
                on_log("已烧录字幕视频 → %s" % os.path.basename(vid))
            except Exception as e:
                on_log("烧录字幕失败：%s" % str(e)[:160])
    return paths


def validate(cues: list[dict]) -> dict:
    """字幕质检：重叠、时长过短、空译文。"""
    issues = []
    prev_end = -1.0
    for c in cues:
        if c["start"] < prev_end - 0.01:
            issues.append({"id": c["id"], "type": "overlap",
                           "detail": "与上一条重叠 %.2fs" % (prev_end - c["start"])})
        if c["end"] - c["start"] < 0.4:
            issues.append({"id": c["id"], "type": "too_short",
                           "detail": "时长 %.2fs，可能看不清" % (c["end"] - c["start"])})
        if not c["zh"]:
            issues.append({"id": c["id"], "type": "empty_zh", "detail": "无译文"})
        prev_end = c["end"]
    return {"cues": len(cues), "issues": issues, "issueCount": len(issues)}
