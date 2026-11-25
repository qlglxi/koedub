# -*- coding: utf-8 -*-
"""TTS（AI 配音）provider 层。

统一接口：
    run(segments, cfg, out_dir, on_log)
        -> {"clips": {id: {"path", "duration", "provider"}}, "provider": ...}

provider 清单见 core/providers.py 的 tts 类别。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time

from . import audiocore
from . import config as cfgmod
from . import edge_tts
from . import httpx as hx

# ---------------------------------------------------------------------------
# 文本清洗（SSML/接口友好）
# ---------------------------------------------------------------------------
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_for_tts(text: str) -> str:
    t = _CTRL_RE.sub(" ", text or "")
    t = t.replace("…", "。").replace("～", "，").replace("~", "，")
    t = re.sub(r"[「」『』]", "", t)
    t = re.sub(r"[（）()]", "", t) if "（" not in t else t
    t = re.sub(r"\s+", "", t)
    return t.strip()


def tts_is_speakable(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ffA-Za-z0-9]", text or ""))


# 翻译层没接通时，translate 会退到 glossary / demo，给每句填「（此句需接入
# 翻译 API）」这类占位串。把它送去 TTS 只会制造一堆失败请求和满屏红字，
# 用户还以为配音坏了——直接跳过，日志里说明原因。
_PLACEHOLDER_RE = re.compile(
    r"(需接入|接入翻译|翻译\s*API|待翻译|占位文本|示例文本|示范文本|placeholder)",
    re.I)


def is_placeholder(text: str) -> bool:
    return bool(_PLACEHOLDER_RE.search(text or ""))


# ---------------------------------------------------------------------------
# Provider 实现
# ---------------------------------------------------------------------------
def _synth_edge(text: str, out: str, cfg: dict, on_log) -> dict:
    t = cfg.get("tts", {})
    voice = t.get("voice") or "zh-CN-XiaoxiaoNeural"
    try:
        return edge_tts.synthesize(text, out, voice=voice,
                                   rate=t.get("rate", "+0%"),
                                   pitch=t.get("pitch", "+0Hz"),
                                   volume=t.get("volume", "+0%"),
                                   on_log=on_log)
    except Exception as e:
        # 403 通常是本地时钟偏差，按服务端 Date 修正后重试一次
        if "403" in str(e):
            on_log("  Edge 返回 403，尝试按服务端时间修正签名后重试…")
            import email.utils
            import urllib.request
            try:
                req = urllib.request.Request("https://www.bing.com", method="HEAD",
                                             headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=10) as r:
                    srv = email.utils.parsedate_to_datetime(r.headers["Date"]).timestamp()
                edge_tts.adjust_clock_skew_seconds(srv - time.time())
            except Exception:
                edge_tts.adjust_clock_skew_seconds(300.0)
            return edge_tts.synthesize(text, out, voice=voice,
                                       rate=t.get("rate", "+0%"),
                                       pitch=t.get("pitch", "+0Hz"),
                                       volume=t.get("volume", "+0%"),
                                       on_log=on_log)
        raise


def _ssml_azure(text: str, cfg: dict) -> str:
    t = cfg.get("tts", {})
    voice = t.get("azureVoice") or t.get("voice") or "zh-CN-XiaoxiaoNeural"
    text = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    return ("<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' "
            "xml:lang='zh-CN'><voice name='%s'>"
            "<prosody rate='%s' pitch='%s' volume='%s'>%s</prosody>"
            "</voice></speak>"
            % (voice, t.get("rate", "+0%"), t.get("pitch", "+0Hz"),
               t.get("volume", "+0%"), text))


def _synth_azure(text: str, out: str, cfg: dict, on_log) -> dict:
    key = cfgmod.get_key(cfg, "AZURE_SPEECH_KEY")
    region = cfgmod.get_key(cfg, "AZURE_SPEECH_REGION") or "eastasia"
    if not key:
        raise RuntimeError("缺少 AZURE_SPEECH_KEY")
    t = cfg.get("tts", {})
    url = "https://%s.tts.speech.microsoft.com/cognitiveservices/v1" % region
    body = _ssml_azure(text, cfg).encode("utf-8")
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "audio-24khz-96kbitrate-mono-mp3",
        "User-Agent": "koedub",
    }
    blob = hx.request(url, method="POST", data=body, headers=headers, timeout=120)
    with open(out, "wb") as f:
        f.write(blob)
    return {"path": out, "bytes": len(blob)}


def _synth_openai(text: str, out: str, cfg: dict, on_log) -> dict:
    key = cfgmod.get_key(cfg, "OPENAI_API_KEY")
    if not key:
        raise RuntimeError("缺少 OPENAI_API_KEY")
    t = cfg.get("tts", {})
    base = (cfgmod.get_key(cfg, "OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    payload = {
        "model": t.get("openaiModel", "gpt-4o-mini-tts"),
        "voice": t.get("openaiVoice", "alloy"),
        "input": text,
        "response_format": "mp3",
    }
    blob = hx.post_json_stream(base + "/audio/speech", payload,
                               headers={"Authorization": "Bearer " + key},
                               timeout=180)
    with open(out, "wb") as f:
        f.write(blob)
    return {"path": out, "bytes": len(blob)}


def _synth_elevenlabs(text: str, out: str, cfg: dict, on_log) -> dict:
    key = cfgmod.get_key(cfg, "ELEVENLABS_API_KEY")
    if not key:
        raise RuntimeError("缺少 ELEVENLABS_API_KEY")
    t = cfg.get("tts", {})
    vid = t.get("elevenVoiceId") or "21m00Tcm4TlvDq8ikWAM"
    url = "https://api.elevenlabs.io/v1/text-to-speech/%s?output_format=mp3_44100_128" % vid
    payload = {"text": text, "model_id": t.get("elevenModel", "eleven_multilingual_v2")}
    blob = hx.post_json_stream(url, payload, headers={"xi-api-key": key}, timeout=180)
    with open(out, "wb") as f:
        f.write(blob)
    return {"path": out, "bytes": len(blob)}


def _synth_custom(text: str, out: str, cfg: dict, on_log) -> dict:
    """自定义 HTTP 端点：POST {"text":..., "voice":...}，期望返回音频字节。"""
    t = cfg.get("tts", {})
    url = t.get("customEndpoint") or cfgmod.get_key(cfg, "CUSTOM_TTS_ENDPOINT")
    if not url:
        raise RuntimeError("未配置自定义语音端点 customEndpoint")
    payload = {"text": text, "voice": t.get("voice"), "language": "zh-CN",
               "format": "mp3"}
    blob = hx.post_json_stream(url, payload, timeout=180)
    if blob[:1] == b"{" or blob[:1] == b"[":
        raise RuntimeError("自定义端点返回的不是音频：%s" % blob[:200])
    with open(out, "wb") as f:
        f.write(blob)
    return {"path": out, "bytes": len(blob)}


def _synth_gpt_sovits(text: str, out: str, cfg: dict, on_log) -> dict:
    """GPT-SoVITS 本地服务（api_v2 的 /tts）—— 用自己调教过的音色合成。

    音色不是靠 voice 名字，而是靠 ref_audio_path + prompt_text 这对"参考锚点"：
    换一段参考音频，就是换一个人说话。所以配置文件里必须同时给这两项。

    前置：在 GPT-SoVITS 目录另开一个服务
        .venv\\Scripts\\python.exe api_v2_sf.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS\\configs\\tts_infer.yaml
    （api_v2_sf.py 是带 torchcodec 补丁和 cuDNN 关闭的启动包装，直接跑官方
      api_v2.py 会因 torchaudio 2.11 缺 torchcodec 而 400，并且慢 40 倍。）
    """
    t = cfg.get("tts", {})
    url = (t.get("sovitsEndpoint")
           or cfgmod.get_key(cfg, "SOVITS_ENDPOINT")
           or "http://127.0.0.1:9880/tts")
    if not url.rstrip("/").endswith("/tts"):
        url = url.rstrip("/") + "/tts"
    ref = (t.get("sovitsRef") or "").strip()
    prompt_text = (t.get("sovitsPromptText") or "").strip()
    if not (ref and os.path.isfile(ref)) or not prompt_text:
        # 界面上「试听这句话」这类单句调用不经过 pipeline，拿不到自动挑的结果；
        # 这里自己补一次，免得用户刚跑完一整套却在试听按钮上撞「参考音频为空」。
        if (t.get("sovitsRefMode") or "auto").lower() == "auto":
            try:
                from . import sovits_proc as _sp
                p2, t2, src = _sp._pick_ref(cfg)
                if p2 and os.path.isfile(p2):
                    ref, prompt_text = p2, t2
                    on_log("  参考音频（%s）：%s" % (src, os.path.basename(p2)))
            except Exception as e:
                on_log("  自动补参考音频失败：%s: %s" % (type(e).__name__, e))
    if not ref or not os.path.isfile(ref):
        raise RuntimeError("tts.sovitsRef 未指向有效音频：%s（可在界面上先跑一次，"
                           "或把「调教音色参考」设为自动）" % (ref or "(空)"))
    if not prompt_text:
        raise RuntimeError("tts.sovitsPromptText 为空，参考音频必须配它对应的台词")

    payload = {
        "text": text,
        "text_lang": t.get("sovitsTextLang") or "zh",
        "ref_audio_path": ref,
        "prompt_text": prompt_text,
        "prompt_lang": t.get("sovitsPromptLang") or "ja",
        "text_split_method": t.get("sovitsSplitMethod") or "cut5",
        "batch_size": int(t.get("sovitsBatchSize") or 1),
        "speed_factor": float(t.get("sovitsSpeed") or 1.0),
        "top_k": int(t.get("sovitsTopK") or 15),
        "top_p": float(t.get("sovitsTopP") or 1.0),
        "temperature": float(t.get("sovitsTemperature") or 1.0),
        "repetition_penalty": float(t.get("sovitsRepetitionPenalty") or 1.35),
        "media_type": "wav",
        "streaming_mode": False,
        "parallel_infer": True,
    }
    blob = hx.post_json_stream(url, payload, timeout=int(t.get("sovitsTimeout") or 300))
    if blob[:1] in (b"{", b"["):
        raise RuntimeError("GPT-SoVITS 返回错误：%s"
                           % blob[:300].decode("utf-8", "replace"))
    if len(blob) < 1024:
        raise RuntimeError("GPT-SoVITS 返回音频过短（%d 字节）" % len(blob))

    wav = os.path.splitext(out)[0] + ".sovits.wav"
    with open(wav, "wb") as f:
        f.write(blob)
    try:
        audiocore.convert(wav, out,
                          sample_rate=int(t.get("sampleRate", 48000)),
                          channels=1,
                          extra=["-c:a", "libmp3lame", "-b:a", "192k"])
    finally:
        try:
            os.remove(wav)
        except OSError:
            pass
    return {"path": out, "bytes": os.path.getsize(out)}


def _sovits_ref_signature(cfg: dict) -> str:
    """参考音频变了，缓存就必须失效。"""
    ref = ((cfg.get("tts", {}) or {}).get("sovitsRef") or "").strip()
    if not ref:
        return ""
    try:
        return "%s:%d" % (os.path.basename(ref), os.path.getsize(ref))
    except OSError:
        return os.path.basename(ref)


_SAPI_PS = r"""
param([string]$Text, [string]$OutPath, [string]$VoiceName, [int]$Rate)
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
try { if ($VoiceName) { $s.SelectVoice($VoiceName) } } catch {}
$s.Rate = $Rate
$s.SetOutputToWaveFile($OutPath)
$s.Speak($Text)
$s.Dispose()
Write-Output "OK"
"""


def _synth_sapi(text: str, out: str, cfg: dict, on_log) -> dict:
    """Windows 内置语音（离线兜底）。"""
    t = cfg.get("tts", {})
    wav = os.path.splitext(out)[0] + ".sapi.wav"
    script = os.path.join(os.path.dirname(out), "_sapi_helper.ps1")
    with open(script, "w", encoding="utf-8-sig") as f:
        f.write(_SAPI_PS)
    exe = shutil.which("pwsh") or shutil.which("powershell")
    if not exe:
        raise RuntimeError("找不到 PowerShell，无法使用 SAPI 离线语音")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    cp = subprocess.run([exe, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script,
                         "-Text", text, "-OutPath", wav,
                         "-VoiceName", t.get("sapiVoice", "Microsoft Huihui Desktop"),
                         "-Rate", str(int(t.get("sapiRate", 0)))],
                        capture_output=True, creationflags=creationflags)
    if cp.returncode != 0 or not os.path.isfile(wav):
        raise RuntimeError("SAPI 合成失败：%s %s"
                           % (cp.stdout.decode("utf-8", "replace")[:200],
                              cp.stderr.decode("utf-8", "replace")[:200]))
    audiocore.convert(wav, out, sample_rate=int(t.get("sampleRate", 48000)),
                      channels=1, extra=["-c:a", "libmp3lame", "-b:a", "128k"])
    os.remove(wav)
    return {"path": out, "bytes": os.path.getsize(out)}


SYNTHERS = {
    "edge": _synth_edge,
    "azure": _synth_azure,
    "openai_tts": _synth_openai,
    "elevenlabs": _synth_elevenlabs,
    "custom_http": _synth_custom,
    "gpt_sovits": _synth_gpt_sovits,
    "pyttsx3_sapi": _synth_sapi,
}
# 别名，容错
SYNTHERS["sapi"] = _synth_sapi
SYNTHERS["pyttsx3"] = _synth_sapi
SYNTHERS["sovits"] = _synth_gpt_sovits
SYNTHERS["gpt-sovits"] = _synth_gpt_sovits
SYNTHERS["gptso"] = _synth_gpt_sovits


# ---------------------------------------------------------------------------
# 调度
# ---------------------------------------------------------------------------
# 别名 -> 规范名，让日志、缓存键、分支判断都只看规范名
_ALIAS = {
    "sovits": "gpt_sovits",
    "gpt-sovits": "gpt_sovits",
    "gptso": "gpt_sovits",
    "sapi": "pyttsx3_sapi",
    "pyttsx3": "pyttsx3_sapi",
}


def canon_provider(name: str) -> str:
    return _ALIAS.get((name or "").strip(), (name or "").strip())


def _sovits_endpoint(cfg: dict) -> str:
    t = cfg.get("tts", {}) or {}
    return (t.get("sovitsEndpoint")
            or cfgmod.get_key(cfg, "SOVITS_ENDPOINT")
            or "http://127.0.0.1:9880")


def sovits_alive(cfg: dict, timeout: float = 1.5) -> bool:
    """GPT-SoVITS 服务在不在——auto 选路时要看这个，免得服务没开就全段失败。"""
    try:
        import socket
        import urllib.parse
        raw = _sovits_endpoint(cfg)
        u = urllib.parse.urlsplit(raw if "//" in raw else "http://" + raw)
        host = u.hostname or "127.0.0.1"
        port = u.port or 9880
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def pick_provider(cfg: dict) -> str:
    want = canon_provider((cfg.get("tts", {}) or {}).get("provider", "edge"))
    if want and want != "auto":
        return want
    # auto：本地服务活着、且参考音频能拿到（固定路径或自动挑选），就用调教音色
    t = cfg.get("tts", {}) or {}
    ref_ok = (t.get("sovitsRef") or "").strip() or \
        (t.get("sovitsRefMode") or "auto").lower() == "auto"
    if ref_ok and sovits_alive(cfg):
        return "gpt_sovits"
    if cfgmod.get_key(cfg, "ELEVENLABS_API_KEY"):
        return "elevenlabs"
    if cfgmod.get_key(cfg, "AZURE_SPEECH_KEY"):
        return "azure"
    if cfgmod.get_key(cfg, "OPENAI_API_KEY"):
        return "openai_tts"
    return "edge"


def voice_label(cfg: dict, provider: str) -> str:
    t = cfg.get("tts", {}) or {}
    if provider == "gpt_sovits":
        ref = (t.get("sovitsRef") or "").strip()
        return (os.path.basename(ref) + " (GPT-SoVITS)") if ref else "GPT-SoVITS"
    return t.get("voice") or t.get("azureVoice") or "-"


def run(segments: list[dict], cfg: dict, out_dir: str, on_log) -> dict:
    provider = pick_provider(cfg)
    if provider not in SYNTHERS:
        raise RuntimeError("未知 TTS provider: %s" % provider)
    synth = SYNTHERS[provider]
    t = cfg.get("tts", {})
    cache_dir = os.path.join(out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    clip_dir = os.path.join(out_dir, "clips")
    os.makedirs(clip_dir, exist_ok=True)

    # 配置要的是调教音色、实际却退回了别的引擎——这种情况必须说清楚。
    # 最常见的原因是 GPT-SoVITS 服务没开（或者刚重启过界面，把它的子进程带走了），
    # 用户不看日志就只会觉得「怎么不像原声了」。
    want = canon_provider((t.get("provider") or "auto"))
    if want in ("", "auto", "gpt_sovits") and provider != "gpt_sovits":
        on_log("⚠ 调教音色没接上：%s 没有响应，这次改用 %s 顶上（音色不随原声）"
               % (_sovits_endpoint(cfg), provider))
        on_log("  想要贴原声，跑之前先在左上「环境体检」里点一下「启动调教音色」。")
    on_log("配音 provider = %s，音色 = %s" % (provider, voice_label(cfg, provider)))
    clips: dict[str, dict] = {}
    skipped = 0
    placeholder = 0
    failed = 0
    t0 = time.time()
    total = len(segments)
    ref_sig = _sovits_ref_signature(cfg) if provider == "gpt_sovits" else ""

    for i, seg in enumerate(segments, 1):
        text = clean_for_tts(seg.get("zh", ""))
        if is_placeholder(text):
            placeholder += 1
            seg["dubStatus"] = "skipped"
            continue
        if not text or not tts_is_speakable(text):
            skipped += 1
            seg["dubStatus"] = "skipped"
            continue
        key = hashlib.sha1(
            ("%s|%s|%s|%s|%s|%s" % (provider, t.get("voice"), t.get("rate"),
                                    t.get("pitch"), ref_sig, text)).encode("utf-8")).hexdigest()[:16]
        cached = os.path.join(cache_dir, key + ".mp3")
        target = os.path.join(clip_dir, "seg_%04d.mp3" % seg["id"])
        try:
            if os.path.isfile(cached) and os.path.getsize(cached) > 512:
                shutil.copyfile(cached, target)
                seg["dubStatus"] = "cached"
            else:
                synth(text, cached, cfg, on_log)
                shutil.copyfile(cached, target)
                seg["dubStatus"] = "ok"
            info = audiocore.probe(target)
            seg["dubPath"] = target
            seg["dubDuration"] = round(info.duration, 3)
            clips[seg["id"]] = {"path": target, "duration": info.duration,
                                "provider": provider}
            if i % 5 == 0 or i == total:
                on_log("  配音进度 %d/%d" % (i, total))
        except Exception as e:
            failed += 1
            seg["dubStatus"] = "failed"
            seg["dubError"] = "%s: %s" % (type(e).__name__, str(e)[:160])
            on_log("  [失败] 第 %d 段：%s" % (seg["id"], seg.get("dubError")))
            if failed >= 5 and not clips:
                raise RuntimeError("配音连续失败（provider=%s）：%s"
                                   % (provider, seg.get("dubError"))) from None

    if placeholder:
        on_log("  跳过 %d 段：译文是翻译层的占位内容，接入翻译 API 后这些句子才会配音"
               % placeholder)
    on_log("配音完成：成功 %d，跳过 %d，失败 %d，用时 %.1fs"
           % (len(clips), skipped + placeholder, failed, time.time() - t0))
    return {"clips": clips, "provider": provider, "skipped": skipped + placeholder,
            "placeholder": placeholder, "failed": failed,
            "elapsed": round(time.time() - t0, 2)}


def list_edge_voices(zh_only: bool = False) -> list[dict]:
    voices = edge_tts.list_voices()
    if zh_only:
        voices = [v for v in voices if str(v.get("Locale", "")).startswith("zh")]
    return [{"name": v.get("ShortName"), "locale": v.get("Locale"),
             "gender": v.get("Gender"), "friendly": v.get("FriendlyName")}
            for v in voices]


def self_test(cfg: dict, on_log=None) -> dict:
    log = on_log or (lambda m: None)
    provider = pick_provider(cfg)
    out = os.path.join(cfgmod.WORK_DIR, "_tts_selftest.mp3")
    os.makedirs(cfgmod.WORK_DIR, exist_ok=True)
    try:
        info = SYNTHERS[provider]("这是一次配音自检。", out, cfg, log)
        dur = audiocore.probe(out).duration
        return {"ok": True, "provider": provider, "bytes": info.get("bytes"),
                "duration": round(dur, 2)}
    except Exception as e:
        return {"ok": False, "provider": provider,
                "error": "%s: %s" % (type(e).__name__, str(e)[:300])}
