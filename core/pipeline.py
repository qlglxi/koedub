# -*- coding: utf-8 -*-
"""主流水线：把音声源 MP3 变成「原声原耳 + 中文对耳」的双语成品。

        ①探测       ②声道判定      ③分段        ④识别
   MP3 ─────────────────────────────────────────────────► 带时间戳日文
                                                              │
                         ⑧合成 ◄── ⑦排轴 ◄── ⑥配音 ◄── ⑤自适应翻译
                    (原耳/对耳摆位)   (吸附时间)   (AI 中文)    (上下文译中)
                                                              │
                                                         ⑥.5 字幕 SRT/ASS

每一步都是独立 provider（见 core/providers.py），可单独替换。
"""
from __future__ import annotations

import json
import os
import time
import traceback

from . import align, asr, audiocore, channels, config as cfgmod, mix
from . import providers, refsel, segmentation, subtitles, translate, tts


class PipelineError(RuntimeError):
    pass


def is_sovits(cfg: dict) -> bool:
    """当前配音是不是走本地 GPT-SoVITS（决定要不要自动挑参考音频）。"""
    try:
        return tts.pick_provider(cfg) == "gpt_sovits"
    except Exception:
        return False


def run(input_path: str, cfg: dict, on_event, work_root: str | None = None,
        manual_text: str = "", manual_segments: list[dict] | None = None) -> dict:
    """执行完整流水线。

    on_event(stage: str, pct: float, message: str, data: dict|None)
    """
    t_start = time.time()
    logs: list[str] = []

    def log(msg: str) -> None:
        line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
        logs.append(line)
        on_event("log", -1, line, None)

    def stage(name: str, pct: float, msg: str, data=None) -> None:
        on_event(name, pct, msg, data)

    if not os.path.isfile(input_path):
        raise PipelineError("找不到输入文件：%s" % input_path)
    if not audiocore.have_ffmpeg():
        raise PipelineError("未检测到 ffmpeg/ffprobe，请先安装 FFmpeg 并确保在 PATH 中。")

    stem = os.path.splitext(os.path.basename(input_path))[0]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(work_root or cfgmod.WORK_DIR, "%s-%s" % (stem[:40], stamp))
    os.makedirs(out_dir, exist_ok=True)
    log("输出目录：%s" % out_dir)

    result: dict = {"input": input_path, "outputDir": out_dir, "stem": stem,
                    "stages": {}, "logs": logs}

    # ---------------- ① 媒体探测 ----------------
    stage("probe", 1, "探测媒体信息")
    info = audiocore.probe(input_path)
    cfg = dict(cfg)
    cfg["_duration"] = info.duration
    log("媒体：%s | %.2f 秒 | %d Hz | %d 声道 | %s | %s"
        % (os.path.basename(input_path), info.duration, info.sample_rate,
           info.channels, info.codec, info.format_name))
    if info.duration <= 0:
        raise PipelineError("无法读取时长，文件可能损坏。")
    result["media"] = info.to_dict()
    result["stages"]["probe"] = info.to_dict()
    stage("probe", 3, "媒体探测完成", info.to_dict())

    # ---------------- ② 声道判定 ----------------
    stage("channel", 4, "判定左右耳声源")
    pcm16 = audiocore.decode_pcm(input_path, sample_rate=16000, channels=2)
    if pcm16.ndim == 1:
        pcm16 = pcm16[:, None]
    side = channels.analyze(pcm16, 16000)
    log("声源判定：人声在 %s（置信度 %.0f%%，电平差 %+.1f dB，"
        "ITD %+.2f ms，语音帧 %d/%d/%d）"
        % ({"left": "左耳", "right": "右耳", "center": "居中/双耳"}[side.side],
           side.confidence * 100, side.median_delta_db, side.itd_ms,
           side.left_frames, side.center_frames, side.right_frames))
    if side.verdict_note:
        log("  " + side.verdict_note)
    manual_side = (cfg.get("channel") or {}).get("forceSide") or "auto"
    if manual_side in ("left", "right", "center"):
        log("用户手动指定声源侧：%s（覆盖自动判定）" % manual_side)
        side.side = manual_side
        side.opposite = {"left": "right", "right": "left", "center": "right"}[manual_side]
    result["stages"]["channel"] = side.to_dict()
    stage("channel", 7, "声道判定完成", side.to_dict())

    # ---------------- ③ 分段 ----------------
    stage("segment", 9, "VAD 分段")
    seg_cfg = cfg.get("asr", {})
    regions = segmentation.detect_regions(
        pcm16, 16000,
        min_segment=float(seg_cfg.get("minSegmentSeconds", 0.5)),
        max_segment=float(seg_cfg.get("maxSegmentSeconds", 12.0)),
        merge_gap=float(seg_cfg.get("mergeGapSeconds", 0.35)))
    log("VAD 得到 %d 个候选发声区间" % len(regions))
    stage("segment", 11, "分段完成", {"regions": len(regions)})

    # ---------------- ④ 识别 ----------------
    stage("asr", 13, "语音识别")
    # 送识别用单声道，减小体积；若人声明确偏一侧，取该侧更干净
    mono_path = os.path.join(out_dir, "asr_input.wav")
    use_ch = 0 if side.side == "left" else (1 if side.side == "right" else None)
    if use_ch is None:
        src = pcm16
    else:
        src = pcm16[:, use_ch:use_ch + 1]
    audiocore.encode_wav(mono_path, src, 16000)
    log("识别输入：%s声道"
        % ("左" if use_ch == 0 else "右" if use_ch == 1 else "混音"))
    asr_res = asr.run(mono_path, cfg, log, regions=regions,
                      manual_text=manual_text, manual_segments=manual_segments)
    segments = segmentation.merge_asr_with_regions(asr_res["segments"], regions)
    if not segments:
        raise PipelineError("识别结果为空，终止流程。")
    log("识别段落：%d 段" % len(segments))
    for s in segments:
        if not s.get("duration"):
            s["duration"] = round(s["end"] - s["start"], 3)
    result["stages"]["asr"] = {"provider": asr_res.get("provider"),
                               "language": asr_res.get("language"),
                               "segments": len(segments),
                               "elapsed": asr_res.get("elapsed")}
    result["asr"] = [{k: v for k, v in s.items() if k != "words"} for s in segments]
    stage("asr", 30, "识别完成（%d 段）" % len(segments))

    # 逐段判侧（音声可能在段间换耳），并确定每段配音落位。
    # 注意传入整体判定作为先验：停顿段会被误判成"极其确定"的一侧，必须用闸门挡住。
    if (cfg.get("channel") or {}).get("perSegment", True):
        channels.analyze_segments(pcm16, 16000, segments, reference=side)
        diff = [s for s in segments if s.get("side") and s["side"] != side.side]
        if diff:
            log("有 %d 段与整体判定不同侧（已过时长/置信度闸门），按段单独摆位。"
                % len(diff))
        else:
            log("各段侧向与整体判定一致，全部按同一侧处理。")
    for s in segments:
        s.setdefault("side", side.side)
        s.setdefault("opposite", {"left": "right", "right": "left",
                                  "center": "right"}[s["side"]])
    result["asr"] = [{k: v for k, v in s.items() if k != "words"} for s in segments]
    stage("channel", 33, "逐段声道分配完成", {"segments": len(segments)})

    # ---------------- ⑤ 自适应翻译 ----------------
    stage("translate", 35, "自适应翻译为中文")
    tr = translate.run(segments, cfg, log)
    segments = tr["segments"]
    result["stages"]["translate"] = {"provider": tr.get("provider"),
                                     "elapsed": tr.get("elapsed"),
                                     "segments": len(segments)}
    result["segments"] = segments
    stage("translate", 52, "翻译完成", {"provider": tr.get("provider")})

    # ---------------- ⑤.5 参考音频自动选取 ----------------
    # GPT-SoVITS 的音色由「参考音频 + 它对应的台词」这对锚点决定；写死一段
    # 参考音频就永远是那个人的音色，所以换音源时必须从这条录音里重新挑。
    if is_sovits(cfg):
        ref_mode = (cfg.get("tts", {}).get("sovitsRefMode") or "auto").lower()
        if ref_mode == "auto":
            stage("refsel", 53, "自动挑选参考音频")
            pick = None
            try:
                pick = refsel.pick_reference(input_path, segments, cfg, log)
            except Exception as e:  # 选不出来不该中断整条流水线
                log("  参考音频自动选取失败：%s: %s" % (type(e).__name__, str(e)[:200]))
            if pick:
                cfg = dict(cfg)
                cfg["tts"] = dict(cfg.get("tts", {}))
                cfg["tts"]["sovitsRef"] = pick["path"]
                cfg["tts"]["sovitsPromptText"] = pick["promptText"]
                result["reference"] = pick
                stage("refsel", 54, "参考音频就绪", pick)
            else:
                log("  沿用配置里的参考音频：%s"
                    % (cfg.get("tts", {}).get("sovitsRef") or "(空)"))
        else:
            log("参考音频按配置固定：%s" % (cfg.get("tts", {}).get("sovitsRef") or "(空)"))

    # ---------------- ⑥ AI 配音 ----------------
    stage("tts", 54, "AI 配音中")
    dub = tts.run(segments, cfg, out_dir, log)
    result["stages"]["tts"] = {"provider": dub.get("provider"),
                               "clips": len(dub.get("clips", {})),
                               "skipped": dub.get("skipped"),
                               "failed": dub.get("failed"),
                               "elapsed": dub.get("elapsed")}
    result["segments"] = segments
    stage("tts", 74, "配音完成（%d 段）" % len(dub.get("clips", {})))
    if not dub["clips"]:
        raise PipelineError("没有任何配音产物，请检查 TTS provider 与密钥设置。")

    # ---------------- ⑦ 排轴 ----------------
    stage("align", 76, "排轴对齐")
    plans = align.plan(segments, cfg, total_duration=info.duration, on_log=log)
    rendered = align.render_clips(segments, plans, out_dir, cfg, log)
    report = align.coverage_report(plans, segments, info.duration)
    log("排轴质量：覆盖 %.1f%% 音频，平均压缩 %.2fx，最大 %.2fx，"
        "超限 %d 段，重叠 %d 处"
        % (report.get("coverageRatio", 0) * 100, report.get("meanTempo", 1),
           report.get("maxTempo", 1), report.get("overrunSegments", 0),
           report.get("overlaps", 0)))
    result["stages"]["align"] = report
    result["plans"] = plans
    stage("align", 86, "排轴完成", report)

    # ---------------- ⑧ 声道合成 ----------------
    stage("mix", 88, "声道摆位与合成")
    comp = mix.compose(input_path, segments, plans, rendered["rendered"], out_dir,
                       cfg, log, total_duration=info.duration)
    result["stages"]["mix"] = {"placed": comp["placed"], "missing": comp["missing"],
                               "peak": comp["peak"], "origSide": comp["origSide"],
                               "elapsed": comp["elapsed"]}
    stage("mix", 93, "合成完成")

    # ---------------- 字幕 ----------------
    stage("subtitle", 94, "导出字幕")
    sub_paths = subtitles.run(segments, plans, out_dir, stem, cfg, log)
    cues = subtitles.build_cues(segments, plans)
    qc = subtitles.validate(cues)
    if qc["issueCount"]:
        log("字幕质检：%d 条提示（重叠/过短/空译文）" % qc["issueCount"])
    result["stages"]["subtitle"] = {"files": sub_paths, "qc": qc}
    stage("subtitle", 96, "字幕导出完成")

    # ---------------- 成品导出 ----------------
    stage("export", 97, "导出成品")
    outputs = mix.export_final(comp["mix"], out_dir, stem, cfg, log)
    previews = mix.export_mono_pair(comp["mix"], out_dir, stem, log,
                                    orig_side=comp["origSide"])
    result["outputs"] = outputs
    result["previews"] = previews

    # 落盘 JSON 报告，便于复盘与二次调整
    result["manifest"] = os.path.join(out_dir, "manifest.json")
    result["elapsedTotal"] = round(time.time() - t_start, 2)
    with open(result["manifest"], "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in result.items() if k != "logs"}, f,
                  ensure_ascii=False, indent=2, default=str)
    log("全部完成，用时 %.1f 秒。" % result["elapsedTotal"])
    stage("done", 100, "完成", {"outputs": outputs, "previews": previews,
                              "outputDir": out_dir})
    return result


def run_safe(input_path: str, cfg: dict, on_event, **kw) -> dict:
    try:
        return run(input_path, cfg, on_event, **kw)
    except Exception as e:
        on_event("error", -1, "%s: %s" % (type(e).__name__, e),
                 {"traceback": traceback.format_exc()[-2000:]})
        raise
