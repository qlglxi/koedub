# -*- coding: utf-8 -*-
"""外部运行时探测：那些「不是 pip 包」的依赖。

和 deps.py 的分工：
  deps.py     —— pip 包（faster-whisper / demucs），装到 vendor/，不污染系统
  runtime.py  —— 外部程序（ffmpeg / GPT-SoVITS），只探测、校验、启动，不安装

GPT-SoVITS 不是 pip 包，它是一整个带 torch 的独立环境（官方整合包 6–7GB）。
所以这里做的四件事是：找到它 → 校验文件完整 → 探服务端口 → 给出该怎么修。
"""
from __future__ import annotations

import importlib.util
import os
import socket
import subprocess
import sys

from . import config as cfgmod

# ---------------------------------------------------------------------------
# ffmpeg
# ---------------------------------------------------------------------------

def probe_ffmpeg() -> dict:
    """ffmpeg 是硬依赖：没有它连音频都切不开。"""
    exe = os.environ.get("KOEDUB_FFMPEG") or "ffmpeg"
    try:
        p = subprocess.run([exe, "-version"], capture_output=True, text=True,
                           timeout=8, encoding="utf-8", errors="ignore")
    except Exception as e:
        return {"status": "missing", "version": None,
                "detail": "调不起 ffmpeg（%s）" % type(e).__name__,
                "fix": {"kind": "install_ffmpeg",
                        "label": "一键安装",
                        "hint": "winget install Gyan.FFmpeg"}}
    if p.returncode != 0:
        return {"status": "missing", "version": None,
                "detail": "ffmpeg 返回码 %d" % p.returncode,
                "fix": {"kind": "install_ffmpeg", "label": "一键安装",
                        "hint": "winget install Gyan.FFmpeg"}}
    first = (p.stdout or "").splitlines()[0] if p.stdout else ""
    version = first.split(" version ")[1].split()[0] if " version " in first else "?"
    return {"status": "ok", "version": version, "detail": version, "fix": None}


# ---------------------------------------------------------------------------
# GPT-SoVITS
# ---------------------------------------------------------------------------
# 校验清单来自官方 v2ProPlus 的目录结构；bytes 是实测值，用来抓「下到一半的
# 残文件」—— 只判断存不存在，会把 0 字节的占位文件当成装好了。
SOVITS_FILES = [
    ("api_v2.py", None, "启动入口"),
    ("GPT_SoVITS/pretrained_models/s1v3.ckpt", 155284856, "GPT 预训练"),
    ("GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth", 200125741, "SoVITS 预训练"),
    ("GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt", None,
     "说话人编码器"),
]
SOVITS_DIRS = [
    ("GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large", "中文 BERT", 200 << 20),
    ("GPT_SoVITS/pretrained_models/chinese-hubert-base", "语义 HuBERT", 100 << 20),
]
# 解释器候选（整合包带 runtime/，源码装法带 .venv/）
SOVITS_PYTHONS = [
    os.path.join("runtime", "python.exe"),
    os.path.join(".venv", "Scripts", "python.exe"),
    os.path.join("venv", "Scripts", "python.exe"),
    "python.exe",
]


def _size(p: str) -> int:
    try:
        return os.path.getsize(p)
    except OSError:
        return -1


def _dir_bytes(p: str) -> int:
    total = 0
    if not os.path.isdir(p):
        return 0
    for dp, _dn, fs in os.walk(p):
        for f in fs:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total


def find_sovits(cfg: dict) -> str | None:
    """按「配置 > 环境变量 > 常见位置」的顺序找一个能用的安装目录。"""
    t = cfg.get("tts", {}) or {}
    cands: list[str] = []
    if (t.get("sovitsInstallDir") or "").strip():
        cands.append(t["sovitsInstallDir"].strip())
    if os.environ.get("KOEDUB_SOVITS_DIR"):
        cands.append(os.environ["KOEDUB_SOVITS_DIR"])
    base = cfgmod.BASE_DIR
    up = os.path.dirname(base.rstrip("\\/"))
    for c in (os.path.join(base, "GPT-SoVITS"),
              os.path.join(up, "GPT-SoVITS"),
              os.path.join(up, "gpt-sovits"),
              r"D:\GPT-SoVITS", r"C:\GPT-SoVITS",
              os.path.join(os.path.expanduser("~"), "GPT-SoVITS")):
        cands.append(c)
    for c in cands:
        try:
            if c and os.path.isfile(os.path.join(c, "api_v2.py")):
                return os.path.abspath(c)
        except (OSError, ValueError):
            continue
    return None


def _find_python(root: str, cfg: dict) -> str | None:
    t = cfg.get("tts", {}) or {}
    if (t.get("sovitsPython") or "").strip():
        p = t["sovitsPython"].strip()
        if os.path.isfile(p):
            return p
    for rel in SOVITS_PYTHONS:
        p = os.path.join(root, rel)
        if os.path.isfile(p):
            return p
    return None


def _port_alive(host: str, port: int, timeout: float = 1.2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _endpoint_parts(cfg: dict) -> tuple[str, int]:
    t = cfg.get("tts", {}) or {}
    url = (t.get("sovitsEndpoint") or "http://127.0.0.1:9880").strip()
    url = url.split("//")[-1].split("/")[0]
    if ":" in url:
        h, p = url.rsplit(":", 1)
        try:
            return h or "127.0.0.1", int(p)
        except ValueError:
            return h or "127.0.0.1", 9880
    return url or "127.0.0.1", 9880


def probe_sovits(cfg: dict) -> dict:
    """GPT-SoVITS 的完整状态：装没装 → 文件全不全 → 服务起没起。"""
    host, port = _endpoint_parts(cfg)
    alive = _port_alive(host, port)
    root = find_sovits(cfg)
    out = {
        "status": "missing", "dir": root, "alive": alive,
        "endpoint": "http://%s:%d" % (host, port),
        "python": None, "checks": [], "missing": [],
        "fix": None,
    }
    if not root:
        out["detail"] = "没找到 GPT-SoVITS 安装目录"
        out["fix"] = {"kind": "download_sovits", "label": "一键下载整合包",
                      "hint": "约 8GB，下完自动解压并填好路径",
                      "alts": [{"kind": "locate_sovits", "label": "我自己有，填路径"}]}
        return out

    checks: list[dict] = []
    missing: list[str] = []
    for rel, want, label in SOVITS_FILES:
        p = os.path.join(root, rel)
        actual = _size(p)
        if actual < 0:
            checks.append({"label": label, "path": rel, "ok": False,
                           "detail": "缺失", "want": want, "actual": None})
            missing.append(rel)
            continue
        if want is not None and actual != want:
            checks.append({"label": label, "path": rel, "ok": False,
                           "detail": "大小不符（%s，应为 %s）"
                                     % (_human(actual), _human(want)),
                           "want": want, "actual": actual})
            missing.append(rel)
            continue
        checks.append({"label": label, "path": rel, "ok": True,
                       "detail": _human(actual), "want": want, "actual": actual})

    for rel, label, min_bytes in SOVITS_DIRS:
        p = os.path.join(root, rel)
        total = _dir_bytes(p)
        ok = total >= min_bytes
        checks.append({"label": label, "path": rel, "ok": ok,
                       "detail": (_human(total) if total else "目录空/缺失")
                                 + ("" if ok else "（应 > %s）" % _human(min_bytes)),
                       "want": min_bytes, "actual": total})
        if not ok:
            missing.append(rel)

    py = _find_python(root, cfg)
    checks.append({"label": "Python 解释器", "path": py or "-", "ok": bool(py),
                   "detail": (os.path.relpath(py, root) if py else "没找到"),
                   "want": None, "actual": None})
    if not py:
        missing.append("python")

    out["checks"] = checks
    out["python"] = py
    out["missing"] = missing
    if alive:
        out["status"] = "running" if not missing else "running_partial"
        out["detail"] = "服务在 %s 上响应" % out["endpoint"]
    elif not missing:
        out["status"] = "installed"
        out["detail"] = "文件齐全，但服务没启动"
        out["fix"] = {"kind": "start_sovits", "label": "启动服务",
                      "hint": "python api_v2.py -a 127.0.0.1 -p %d" % port}
    else:
        out["status"] = "partial"
        out["detail"] = "缺 %d 项：%s" % (len(missing), "、".join(missing[:3]))
        out["fix"] = {"kind": "repair_sovits", "label": "补全文件",
                      "hint": "重新解压官方整合包，或单独下缺的权重",
                      "alts": [{"kind": "download_sovits", "label": "重下整合包"}]}
    return out


def _human(n: int) -> str:
    if n < 0:
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return ("%d %s" % (n, unit)) if unit == "B" else ("%.1f %s" % (n, unit))
        n /= 1024.0
    return "%d B" % n


# ---------------------------------------------------------------------------
# 语音识别 / 翻译 / 配音
# ---------------------------------------------------------------------------

def _can_import(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def probe_asr(cfg: dict) -> dict:
    from . import asr as asr_mod
    has_fw = _can_import("faster_whisper")
    has_ow = _can_import("whisper")
    models = asr_mod.models_on_disk()
    want = (cfg.get("asr", {}) or {}).get("model") or "small"
    got = [m.get("model") or m.get("name") for m in models]
    out = {"engine": ("faster-whisper" if has_fw else
                      ("openai-whisper" if has_ow else None)),
           "models": got, "wantModel": want}
    if not has_fw and not has_ow:
        # 本地引擎没装，但云端密钥也算能用——不看这一层的话会把
        # 「只填了 DeepSeek 密钥」的用户误判成缺件，然后拦住他不让跑。
        cloud = []
        for k, alias in (("GROQ_API_KEY", "Groq"),
                         ("DEEPSEEK_API_KEY", "DeepSeek"),
                         ("SILICONFLOW_API_KEY", "SiliconFlow"),
                         ("OPENAI_API_KEY", "OpenAI")):
            if cfgmod.get_key(cfg, k):
                cloud.append(alias)
        if cloud:
            out.update(status="ok", detail="云端识别 · %s" % cloud[0], fix=None)
            return out
        out.update(status="missing", detail="本地识别引擎没装，也没有云端识别密钥",
                   fix={"kind": "install_dep", "target": "faster_whisper",
                        "label": "安装 faster-whisper（约 120MB）"})
        return out
    # 模型在盘上才算真能用；第一次跑会触发下载，国内常因权限/网络失败
    if want and want not in got:
        out.update(status="partial",
                   detail="引擎有，但 %s 模型还没下（首次识别会自动下）" % want,
                   fix={"kind": "note", "label": "首次识别会下载约 460MB",
                        "hint": "HF 权限失败时可设 HF_HOME 到工程目录"})
        return out
    out.update(status="ok", detail="%s · 模型 %s" % (out["engine"], want), fix=None)
    return out


def probe_translate(cfg: dict) -> dict:
    key = cfgmod.get_key(cfg, "DEEPSEEK_API_KEY")
    if key:
        return {"status": "ok", "provider": "deepseek", "fix": None,
                "detail": "DeepSeek 密钥已配置（%s…）" % key[:8]}
    return {"status": "missing", "provider": None,
            "detail": "没有可用的翻译密钥",
            "fix": {"kind": "set_key", "target": "DEEPSEEK_API_KEY",
                    "label": "填密钥"}}


def probe_tts(cfg: dict) -> dict:
    from . import tts as tts_mod
    provider = tts_mod.pick_provider(cfg)
    sov = probe_sovits(cfg)
    if provider == "gpt_sovits":
        return {"status": "ok", "provider": "gpt_sovits", "fix": None,
                "detail": "本地调教音色 · %s" % tts_mod.voice_label(cfg, provider)}
    if provider == "edge":
        # 落到系统音色就是降级：中文是好听的普通话，但跟原声不是一个人。
        # 这里必须说清楚并直接给下一步，否则新用户跑完一听是机器人音，
        # 只会觉得这项目没用，不会想到是自己少装了一个东西。
        st = (sov or {}).get("status")
        if st == "missing":
            return {"status": "partial", "provider": "edge",
                    "detail": "系统音色（过渡方案）：没装 GPT-SoVITS，中文不会像原声",
                    "fix": {"kind": "download_sovits",
                            "label": "装调教音色（约 8GB）",
                            "hint": "不装也能出双语成品，只是音色不贴原声",
                            "alts": [{"kind": "note", "label": "先这样用",
                                      "hint": "想贴原声随时回来装"}]}}
        if st in ("installed", "partial"):
            return {"status": "partial", "provider": "edge",
                    "detail": "系统音色（过渡方案）：GPT-SoVITS 装了但没启动",
                    "fix": {"kind": "start_sovits", "label": "启动调教音色",
                            "hint": "启动后会自动换成原声音色"}}
        return {"status": "partial", "provider": "edge",
                "detail": "系统音色（过渡方案）：调教服务没响应",
                "fix": {"kind": "start_sovits", "label": "启动调教音色"}}
    return {"status": "ok", "provider": provider, "fix": None,
            "detail": tts_mod.voice_label(cfg, provider)}


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------

def diagnose(cfg: dict | None = None) -> dict:
    """把「跑一段音声需要什么」算成一张表。

    分三档：
      ok       已经有，能用
      partial  能用但会掉质量/速度，或者还没下完
      missing  缺了就出不了成品
    """
    cfg = cfg or cfgmod.load_config()
    items = []

    ff = probe_ffmpeg()
    items.append({"key": "ffmpeg", "title": "媒体处理 ffmpeg",
                  "required": True, **ff})

    asr = probe_asr(cfg)
    items.append({"key": "asr", "title": "语音识别（日语转写）",
                  "required": True, **asr})

    tr = probe_translate(cfg)
    items.append({"key": "translate", "title": "自适应翻译",
                  "required": True, **tr})

    tts_p = probe_tts(cfg)
    items.append({"key": "tts", "title": "中文配音",
                  "required": True, **tts_p})

    sov = probe_sovits(cfg)
    # 已经把 GPT-SoVITS 当配音在用时，它缺东西就是硬缺口
    sov_required = tts_p.get("provider") == "gpt_sovits"
    items.append({"key": "sovits", "title": "GPT-SoVITS 调教音色",
                  "required": sov_required, **sov})

    from . import deps as deps_mod
    try:
        dep = deps_mod.status()
    except Exception:
        dep = {}
    for name in ("faster_whisper", "demucs"):
        meta = dep.get(name) or {}
        if not meta:
            continue
        items.append({
            "key": "dep:" + name, "title": meta.get("title", name),
            "required": False,
            "status": "ok" if meta.get("installed") else "optional",
            "detail": ("已安装" if meta.get("installed") else "未安装（可选增强）"),
            "fix": None if meta.get("installed") else
                   {"kind": "install_dep", "target": name,
                    "label": "安装", "hint": meta.get("size", "")},
        })

    blockers = [i["title"] for i in items
                if i["required"] and i["status"] == "missing"]
    warns = [i["title"] for i in items
             if i["status"] in ("partial", "optional")]
    ready = not blockers

    return {
        "ready": ready,
        "blockers": blockers,
        "warnings": warns,
        "items": items,
        "summary": ("可以出成品" if ready
                    else "还缺 %d 项才能出成品" % len(blockers)),
        "env": {
            "baseDir": cfgmod.BASE_DIR,
            "python": sys.executable,
            "sovitsDir": sov.get("dir"),
            "ffmpeg": ff.get("version"),
        },
    }
