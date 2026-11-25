# -*- coding: utf-8 -*-
"""ASR（语音识别）provider 层。

统一接口：
    run(audio_path, cfg, on_log) -> {"segments": [{start, end, text}], "language": "ja",
                                     "provider": "..."}

provider 清单见 core/providers.py 的 asr 类别。
"""
from __future__ import annotations

import importlib.util
import os
import re
import time

from . import config as cfgmod
from . import httpx as hx

# ---------------------------------------------------------------------------


def _seg(start: float, end: float, text: str) -> dict:
    return {"start": round(float(start), 3), "end": round(float(end), 3), "text": text.strip()}


def _dedupe(segments: list[dict]) -> list[dict]:
    """whisper 在重复段上容易叠字，做一次去重与重叠裁剪。"""
    out: list[dict] = []
    for s in sorted(segments, key=lambda x: x["start"]):
        if not s["text"]:
            continue
        if out:
            if s["start"] < out[-1]["end"]:
                s["start"] = out[-1]["end"]
            if s["end"] <= s["start"]:
                continue
            if s["text"] == out[-1]["text"]:
                out[-1]["end"] = max(out[-1]["end"], s["end"])
                continue
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# 1) faster-whisper / openai-whisper 本地
# ---------------------------------------------------------------------------
def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


# ---------------------------------------------------------------------------
# 模型下载与国内镜像
# ---------------------------------------------------------------------------
# faster-whisper 的模型存放在 HuggingFace Hub。国内直连 huggingface.co 通常不通，
# 而 hf-mirror.com 可用（本机实测：官方超时 / 镜像 200），所以默认把端点指向镜像，
# 用户想用官方端点时把 HF_ENDPOINT 设成官方地址即可。
HF_MIRROR = "https://hf-mirror.com"
_MODEL_REPO = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
}


def setup_hf_mirror(on_log=None) -> str:
    """把 HuggingFace 端点切到国内镜像（除非用户已显式设置）。

    两个关键点，都必须在导入 huggingface_hub / faster_whisper **之前**生效：
      1. HF_ENDPOINT 指向 hf-mirror.com（官方 huggingface.co 国内常超时）；
      2. 禁用 Xet 存储后端。新版 huggingface_hub 默认走 Xet（cas-server.xethub.hf.co），
         镜像站不支持，会直接 401 Unauthorized 导致模型下载失败。
    """
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    cur = os.environ.get("HF_ENDPOINT")
    if cur and "hf-mirror" not in cur:
        return cur
    os.environ.setdefault("HF_ENDPOINT", HF_MIRROR)
    if on_log:
        on_log("模型下载端点：%s（国内镜像；可用 HF_ENDPOINT 覆盖。"
               "已禁用 Xet 后端，镜像站不支持它）" % os.environ["HF_ENDPOINT"])
    return os.environ["HF_ENDPOINT"]


def model_repo(model: str) -> str:
    return _MODEL_REPO.get(model, "Systran/faster-whisper-%s" % model)


def _snapshot_dir(repo_path: str) -> str:
    """在 HF 缓存仓库目录里找出真正能加载的快照目录（含 model.bin 的那层）。

    WhisperModel 要的是 snapshots/<hash>/ 这种目录，直接喂 models--xxx 会报缺文件。
    """
    snap = os.path.join(repo_path, "snapshots")
    if not os.path.isdir(snap):
        return ""
    for name in sorted(os.listdir(snap)):
        d = os.path.join(snap, name)
        if os.path.isfile(os.path.join(d, "model.bin")):
            return d
    return ""


def models_on_disk() -> list[dict]:
    """列出本地已经下好的模型（读 HF 缓存目录）。"""
    roots = []
    for env in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"):
        v = os.environ.get(env)
        if v:
            roots.append(os.path.join(v, "hub") if not v.endswith("hub") else v)
    roots.append(os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"))
    found: dict[str, dict] = {}
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            if not name.startswith("models--Systran--faster-whisper"):
                continue
            size = 0
            for dp, _d, fs in os.walk(os.path.join(root, name)):
                for f in fs:
                    try:
                        size += os.path.getsize(os.path.join(dp, f))
                    except OSError:
                        pass
            short = name.replace("models--Systran--faster-whisper-", "")
            found[short] = {"model": short, "repo": name.replace("models--", "").replace("--", "/"),
                            "path": os.path.join(root, name), "bytes": size}
    return sorted(found.values(), key=lambda x: -x["bytes"])


def run_whisper_local(audio_path: str, cfg: dict, on_log) -> dict:
    a = cfg.get("asr", {})
    model_name = a.get("model", "small")
    language = a.get("language", "ja") or None
    device = a.get("device", "auto")
    compute = a.get("computeType", "int8")
    if device == "auto":
        device = "cuda" if _cuda_available() else "cpu"
    if device == "cpu" and compute in ("float16", "int8_float16"):
        compute = "int8"

    if not _has("faster_whisper"):
        # 任何入口（服务 / CLI / 脚本）都先补上 vendor 路径。
        # 不补的话会静默退到 openai-whisper：90 秒音频实测 503 秒（快 24 倍差距），
        # 而且它自带的分段更碎、更容易出「www」「without」这类幻觉。
        try:
            from . import deps as _deps
            _deps.ensure_on_path()
        except Exception:
            pass

    if _has("faster_whisper"):
        # 先做版本一致性检查：vendor 里的 C 扩展（ctranslate2 / av / tokenizers）
        # 与 Python 版本强绑定，用错解释器会报莫名其妙的属性错误，不如直接说清楚。
        from . import deps
        mm = deps.version_mismatch()
        if mm:
            hint = ("可选依赖是为 %s 安装的，当前解释器是 %s，含 C 扩展的包无法通用。"
                    "请用同一个 Python 重新执行：python app.py --install faster-whisper"
                    % (mm["installedWith"], mm["running"]))
            on_log("⚠ " + hint)
            if (cfg.get("asr", {}) or {}).get("allowDemo", True):
                on_log("  本次改用 Demo 占位识别；也可选云端 ASR 或手工导入。")
                return run_demo(audio_path, cfg, on_log, regions=None)
            raise RuntimeError(hint)

        setup_hf_mirror(on_log)
        from faster_whisper import WhisperModel
        _patch_av_metadata_errors(on_log)
        # 已经下好的模型直接喂快照目录：不碰 huggingface_hub，也就不会被
        # 系统代理变量搞崩（实测过 env 里代理写坏时 httpx 直接抛
        # InvalidURL: Invalid port，用户只是加载本地模型却看到这种报错）。
        local_snap = ""
        for m in models_on_disk():
            if m["model"] == model_name:
                local_snap = _snapshot_dir(m["path"])
                break
        if local_snap:
            on_log("本地已缓存模型 %s，直接加载：%s" % (model_name, local_snap))
            model = WhisperModel(local_snap, device=device, compute_type=compute)
        else:
            on_log("加载 faster-whisper 模型 %s（%s/%s），首次需从 %s 下载…"
                   % (model_name, device, compute,
                      os.environ.get("HF_ENDPOINT", "huggingface.co")))
            model = WhisperModel(model_name, device=device, compute_type=compute)
        on_log("开始识别…")

        def _decode_once(source):
            segs, inf = model.transcribe(
                source,
                language=language,
                beam_size=int(a.get("beamSize", 5)),
                vad_filter=bool(a.get("vad", True)),
                initial_prompt=a.get("initialPrompt") or None,
                condition_on_previous_text=False,
            )
            return list(segs), inf

        # 优先走「ffmpeg 解码 + numpy 输入」，彻底避开 PyAV 兼容问题
        source = audio_path
        try:
            audio = _decode_with_ffmpeg(audio_path, 16000)
            on_log("已用 ffmpeg 解码 %d 个采样点（%.1f 秒），直接送入模型"
                   % (audio.size, audio.size / 16000.0))
            source = audio
        except Exception as e:
            on_log("ffmpeg 预解码失败（%s），退回由模型自行解码。" % str(e)[:80])

        try:
            seg_list, info = _decode_once(source)
        except TypeError as e:
            # 兜底：万一是 PyAV 版本问题，补丁后重试一次
            if _looks_like_av_signature_error(e) and _patch_av_metadata_errors(on_log):
                seg_list, info = _decode_once(source)
            else:
                raise
        out = [_seg(s.start, s.end, s.text) for s in seg_list]
        return {"segments": _dedupe(out),
                "language": getattr(info, "language", language or "ja"),
                "provider": "whisper_local(faster-whisper)",
                "model": model_name}

    if _has("whisper"):
        import whisper
        on_log("加载 openai-whisper 模型 %s…" % model_name)
        model = whisper.load_model(model_name, device=device)
        on_log("开始识别…")
        res = model.transcribe(audio_path, language=language,
                               initial_prompt=a.get("initialPrompt") or None)
        out = [_seg(s["start"], s["end"], s["text"]) for s in res.get("segments", [])]
        return {"segments": _dedupe(out),
                "language": res.get("language", language or "ja"),
                "provider": "whisper_local(openai-whisper)",
                "model": model_name}

    raise RuntimeError(
        "未安装本地识别模型。请执行：pip install faster-whisper\n"
        "或在「语音识别」里改选一个云端 provider / Demo。")


def _cuda_available() -> bool:
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _patch_av_metadata_errors(on_log=None) -> bool:
    """兼容性兜底：PyAV 不认识 metadata_errors 时自动忽略该参数。

    faster-whisper 1.2.x 调用 ``av.open(..., metadata_errors="ignore")``，但已发布的
    PyAV（≤19.0.1）还没有这个关键字，直接 TypeError 中断识别。

    faster_whisper/audio.py 里是 ``from av import open`` —— 函数对象被直接绑到模块
    命名空间，所以必须补丁到 **该模块的属性** 上，改 ``av.open`` 无效。
    """
    try:
        import av
        import inspect
    except Exception:
        return False
    try:
        if "metadata_errors" in inspect.signature(av.open).parameters:
            return False                # 该版本已支持，无需补丁
    except (TypeError, ValueError):
        return False
    original = av.open

    def patched(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        return original(*args, **kwargs)

    av.open = patched
    applied = False
    # 先尝试补丁已加载的模块，再把 audio 模块显式导入后补一次（它是延迟导入的）
    import importlib
    for modname in ("faster_whisper.audio", "faster_whisper.transcribe"):
        try:
            mod = sys.modules.get(modname) or importlib.import_module(modname)
        except Exception:
            continue
        if hasattr(mod, "open"):
            setattr(mod, "open", patched)
            applied = True
    if on_log and applied:
        on_log("已应用 PyAV 兼容层：该版本不支持 metadata_errors，已自动忽略")
    return applied


def _looks_like_av_signature_error(exc: BaseException) -> bool:
    text = "%s" % exc
    return "metadata_errors" in text and "unexpected keyword" in text


def _decode_with_ffmpeg(audio_path: str, sample_rate: int = 16000):
    """用 ffmpeg 解码成 numpy 数组，绕开 faster-whisper 内置的 PyAV 解码路径。

    为什么这么做：
      * faster-whisper 1.2.x 会调用 ``av.open(..., metadata_errors=...)``，而已发布的
        PyAV（含 19.0.1）没有这个关键字，识别会直接 TypeError；
      * 本工程本来就依赖 ffmpeg，用它解码结果一致、行为可控，也省掉一层依赖；
      * 传 numpy 给 transcribe() 比传路径更快（省一次重复解码）。
    返回 float32、单声道、16kHz，取值 [-1, 1]。
    """
    from . import audiocore          # 延迟导入，避免循环依赖
    pcm = audiocore.decode_pcm(audio_path, sample_rate=sample_rate, channels=1)
    if pcm.ndim > 1:
        pcm = pcm[:, 0]
    return pcm.astype("float32")


# ---------------------------------------------------------------------------
# 2) OpenAI 兼容 /audio/transcriptions（OpenAI / DeepSeek / Groq / 硅基流动）
# ---------------------------------------------------------------------------
ASR_ENDPOINTS = {
    "openai_asr": ("https://api.openai.com/v1/audio/transcriptions", "OPENAI_API_KEY"),
    "deepseek_asr": ("https://api.deepseek.com/v1/audio/transcriptions", "DEEPSEEK_API_KEY"),
    "groq_asr": ("https://api.groq.com/openai/v1/audio/transcriptions", "GROQ_API_KEY"),
    "siliconflow_asr": ("https://api.siliconflow.cn/v1/audio/transcriptions",
                        "SILICONFLOW_API_KEY"),
}
ASR_MODELS = {
    "openai_asr": "whisper-1",
    "deepseek_asr": "whisper-1",
    "groq_asr": "whisper-large-v3",
    "siliconflow_asr": "FunAudioLLM/SenseVoiceSmall",
}


def run_openai_compatible(audio_path: str, cfg: dict, on_log,
                          provider: str) -> dict:
    url, keyname = ASR_ENDPOINTS[provider]
    key = cfgmod.get_key(cfg, keyname)
    if not key:
        raise RuntimeError("缺少 %s，请在设置里填入密钥。" % keyname)
    with open(audio_path, "rb") as f:
        blob = f.read()
    fields = {
        "model": ASR_MODELS.get(provider, "whisper-1"),
        "response_format": "verbose_json",
        "language": cfg.get("asr", {}).get("language", "ja") or "ja",
        "temperature": "0",
    }
    body, ctype = hx.multipart(fields, {"file": (os.path.basename(audio_path), blob,
                                                  "audio/mpeg")})
    on_log("上传音频到 %s（%.1f MB）…" % (url.split("/")[2], len(blob) / 1048576.0))
    data = hx.request_json(url, method="POST", data=body,
                           headers={"Authorization": "Bearer " + key,
                                    "Content-Type": ctype},
                           timeout=600)
    segs = []
    for s in data.get("segments", []) or []:
        segs.append(_seg(s.get("start", 0), s.get("end", 0), s.get("text", "")))
    if not segs and data.get("text"):
        # 无时间戳的返回：按字符数按时长均分（粗排轴，后续对齐会修正）
        segs = [_seg(0.0, float(cfg.get("_duration") or 1.0), data["text"])]
    return {"segments": _dedupe(segs), "language": data.get("language", "ja"),
            "provider": provider}


# ---------------------------------------------------------------------------
# 3) Demo：内置识别器（无依赖、离线，用于跑通流程/自测）
# ---------------------------------------------------------------------------
DEMO_LINES = [
    "（示范文本）ねえ、聞こえてる？",
    "（示范文本）今日はゆっくり休んでね。",
    "（示范文本）耳元でささやくよ。",
    "（示范文本）こちらは左耳、中国語は右耳に流れるよ。",
]


def run_demo(audio_path: str, cfg: dict, on_log, regions: list[dict] | None = None) -> dict:
    """演示识别器：无模型时用来验证「排轴 + 配音 + 摆位」链路。

    它产出的是**占位日文**，不是真实识别结果，因此会显著标注，前端也会提示。
    文本一律带「（示范文本）」前缀，避免被误当成真识别结果。
    """
    on_log("⚠ 使用 Demo 识别器：产出的是占位日文，仅用于验证排轴与配音链路，"
           "并非真实识别结果。要真实识别请安装本地模型或配置云端 ASR。")
    segs = []
    rs = regions or []
    if not rs:
        dur = float(cfg.get("_duration") or 30.0)
        step = 4.0
        n = max(1, int(dur // step))
        rs = [{"start": i * step, "end": min(dur, i * step + step * 0.8)} for i in range(n)]
    for i, r in enumerate(rs):
        segs.append(_seg(r["start"], r["end"], DEMO_LINES[i % len(DEMO_LINES)]))
    return {"segments": segs, "language": "ja", "provider": "demo"}


# ---------------------------------------------------------------------------
# 4) 手工导入
# ---------------------------------------------------------------------------
def run_manual(audio_path: str, cfg: dict, on_log,
               manual_text: str = "", manual_segments: list[dict] | None = None) -> dict:
    if manual_segments:
        on_log("使用手工导入的 %d 条带时间戳文本。" % len(manual_segments))
        return {"segments": _dedupe([_seg(s["start"], s["end"], s["text"])
                                     for s in manual_segments]),
                "language": "ja", "provider": "manual"}
    if manual_text.strip():
        on_log("使用手工导入的纯文本，按标点与 VAD 骨架切分。")
        parts = [p for p in re.split(r"(?<=[。！？!?…\n])", manual_text) if p.strip()]
        dur = float(cfg.get("_duration") or 30.0)
        step = dur / max(1, len(parts))
        segs = [_seg(i * step, min(dur, (i + 1) * step * 0.95), p.strip())
                for i, p in enumerate(parts)]
        return {"segments": _dedupe(segs), "language": "ja", "provider": "manual"}
    raise RuntimeError("manual provider 需要提供 manualText 或 manualSegments。")


# ---------------------------------------------------------------------------
# 调度
# ---------------------------------------------------------------------------
def pick_provider(cfg: dict) -> str:
    """auto 模式：优先本地模型，其次有密钥的云，最后 Demo。

    ``cfg.asr.allowDemo`` 为 False 时不会回退到 Demo，而是抛错并给出安装指引，
    避免用户以为拿到的是真实识别结果。
    """
    want = (cfg.get("asr", {}) or {}).get("provider", "auto")
    if want and want != "auto":
        return want
    if _has("faster_whisper") or _has("whisper"):
        return "whisper_local"
    if cfgmod.get_key(cfg, "GROQ_API_KEY"):
        return "groq_asr"
    if cfgmod.get_key(cfg, "DEEPSEEK_API_KEY"):
        return "deepseek_asr"
    if cfgmod.get_key(cfg, "SILICONFLOW_API_KEY"):
        return "siliconflow_asr"
    if cfgmod.get_key(cfg, "OPENAI_API_KEY"):
        return "openai_asr"
    if (cfg.get("asr", {}) or {}).get("allowDemo", True):
        return "demo"
    raise RuntimeError(
        "没有可用的语音识别后端。三选一：\n"
        "  1) 本地模型（离线免费）：在界面点「安装本地识别模型」，"
        "或执行  python app.py --install faster-whisper\n"
        "  2) 云端识别：填入 GROQ_API_KEY / DEEPSEEK_API_KEY / SILICONFLOW_API_KEY\n"
        "  3) 手工导入：把带时间戳的日文文本贴进来，选 manual")


def install_hint() -> str:
    return ("python app.py --install faster-whisper  "
            "（会自动使用国内 PyPI 镜像，装在工程内 vendor 目录）")


def available_providers() -> dict:
    return {
        "whisper_local": _has("faster_whisper") or _has("whisper"),
        "demo": True,
        "manual": True,
        "openai_asr": bool(os.environ.get("OPENAI_API_KEY")),
        "deepseek_asr": bool(os.environ.get("DEEPSEEK_API_KEY")),
        "groq_asr": bool(os.environ.get("GROQ_API_KEY")),
        "siliconflow_asr": bool(os.environ.get("SILICONFLOW_API_KEY")),
    }


def run(audio_path: str, cfg: dict, on_log, regions: list[dict] | None = None,
        manual_text: str = "", manual_segments: list[dict] | None = None) -> dict:
    try:
        provider = pick_provider(cfg)
    except RuntimeError:
        provider = "demo"
        on_log("⚠ 未找到可用的识别后端，已回退到 Demo（占位文本）。")
        on_log("  安装方式：" + install_hint())
    t0 = time.time()
    on_log("ASR provider = %s" % provider)
    if provider == "whisper_local":
        res = run_whisper_local(audio_path, cfg, on_log)
    elif provider in ASR_ENDPOINTS:
        res = run_openai_compatible(audio_path, cfg, on_log, provider)
    elif provider == "manual":
        res = run_manual(audio_path, cfg, on_log, manual_text, manual_segments)
    elif provider == "demo":
        res = run_demo(audio_path, cfg, on_log, regions)
    else:
        raise RuntimeError("未知 ASR provider: %s" % provider)
    res["elapsed"] = round(time.time() - t0, 2)
    on_log("识别完成：%d 段，用时 %.1fs" % (len(res["segments"]), res["elapsed"]))
    return res
