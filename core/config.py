# -*- coding: utf-8 -*-
"""配置与密钥管理。

优先级：显式传入 > 环境变量 > config.json > 默认值。
绝不在日志里回显完整密钥。
"""
from __future__ import annotations

import json
import os
from copy import deepcopy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK_DIR = os.path.join(BASE_DIR, "work")
OUT_DIR = os.path.join(BASE_DIR, "output")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

DEFAULTS: dict = {
    "server": {"host": "127.0.0.1", "port": 8722, "autoOpenBrowser": True,
               # 独立应用窗口（Edge/Chrome 应用模式）。默认关闭：部分环境下浏览器
               # 会立刻退出；开启后若失败会自动回退默认浏览器。想要独立窗口设 true。
               "appWindow": False},
    "asr": {
        "provider": "auto",            # auto / whisper_local / openai_asr / demo / manual
        "model": "small",              # tiny / base / small / medium / large-v3
        "language": "ja",
        "device": "auto",              # auto / cpu / cuda
        "computeType": "int8",
        "beamSize": 5,
        "initialPrompt": "",
        "vad": True,
        "maxSegmentSeconds": 7.0,
        # 实测调参（用已知句界的真实朗读样本）：0.40s 能把 5 句精确切成 5 段；
        # 调到 0.3 会把长句内部的换气也切开（切成 6~7 段），所以取 0.40。
        "mergeGapSeconds": 0.40,
        "minSegmentSeconds": 0.5,
        "allowDemo": True,             # 无识别后端时是否允许回退到占位 Demo
    },
    "translate": {
        "provider": "auto",            # auto / deepseek / openai_compat / ollama / glossary
        "style": "adaptive",           # adaptive / literal / natural / soft
        "targetLanguage": "zh-CN",
        "persona": "音声作品听众",
        "keepHonorifics": True,        # 保留「酱/桑/前辈」等称谓
        "glossary": {},
        "batchSegments": 12,
        "temperature": 0.3,
    },
    "tts": {
        # auto：有能用的调教音色（GPT-SoVITS 在跑，或参考音频自动挑）就用它，
        # 否则退回 Edge 在线音色。默认必须是 auto——写成 edge 的话，新用户
        # 装好了 GPT-SoVITS 也不会有任何提示，跑出来是机器人音还以为成了。
        "provider": "auto",            # auto / gpt_sovits / edge / azure / openai_tts / elevenlabs / custom_http / pyttsx3_sapi
        "voice": "zh-CN-XiaoxiaoNeural",
        "rate": "+0%",
        "pitch": "+0Hz",
        "volume": "+0%",
        "sampleRate": 48000,
        "azureVoice": "zh-CN-XiaoxiaoNeural",
        "openaiVoice": "alloy",
        "openaiModel": "gpt-4o-mini-tts",
        "elevenVoiceId": "",
        "customEndpoint": "",
        "sapiVoice": "Microsoft Huihui Desktop",
        # --- GPT-SoVITS 本地服务（自己调教过的音色）---
        # 音色由"参考音频 + 它对应的台词"这对锚点决定，不是靠 voice 名字。
        # 先在 GPT-SoVITS 目录跑：api_v2_sf.py -a 127.0.0.1 -p 9880 ...
        "sovitsEndpoint": "http://127.0.0.1:9880",
        "sovitsInstallDir": "",        # 留空则自动在常见位置找 GPT-SoVITS
        "sovitsPython": "",            # 留空则依次试 runtime/ .venv/ python.exe
        "sovitsRefMode": "auto",       # auto=每次从输入音频里自动挑一句 / fixed=用下面的 sovitsRef
        "sovitsRef": "",
        "sovitsRefSampleRate": 32000,
        "sovitsRefCache": "",          # 留空则用 work/_refcache
        "sovitsPromptText": "",
        "sovitsPromptLang": "ja",      # 参考音频说的语言
        "sovitsTextLang": "zh",        # 要合成的文本语言（中文配音用 zh）
        "sovitsSplitMethod": "cut5",
        "sovitsBatchSize": 1,
        "sovitsSpeed": 1.0,
        "sovitsTopK": 15,
        "sovitsTopP": 1.0,
        "sovitsTemperature": 1.0,
        "sovitsRepetitionPenalty": 1.35,
        "sovitsTimeout": 300,
    },
    "align": {
        "provider": "builtin",
        "maxTempo": 1.45,              # 最大压缩倍率（约等于语速上限）
        "minTempo": 0.85,
        "minGapMs": 60,                # 段间最小间隔，避免糊在一起
        "leadInMs": 0,                 # 配音相对原文可提前量
        "tailPaddingMs": 80,
        "globalShiftMs": 0,
        "maxCharsPerSecond": 7.5,      # 中文语速预算，用于自动控字
    },
    "mix": {
        # media_swap 中/侧分解：原声台词留原耳，对侧耳清出人声位给中文（推荐）
        # swap 整段左右对调 / keep 原样保留 / duck 对侧下压 / mono_split 折叠双耳
        "originalMode": "media_swap",
        "originalGainDb": 0.0,
        "dubGainDb": 0.0,
        "duckAmountDb": -7.0,          # originalMode=duck 时对侧耳的压低量
        "keepSfxOnBothSides": True,
        "loudnorm": True,
        "targetLufs": -16.0,
    },
    "separate": {"provider": "none", "model": "htdemucs"},
    "subtitle": {"enabled": True, "bilingual": True, "burnVideo": False},
    "export": {"format": "mp4", "audioCodec": "aac", "audioBitrate": "192k"},
    "keys": {
        "DEEPSEEK_API_KEY": "",
        "OPENAI_API_KEY": "",
        "OPENAI_BASE_URL": "https://api.openai.com/v1",
        "GROQ_API_KEY": "",
        "SILICONFLOW_API_KEY": "",
        "DASHSCOPE_API_KEY": "",
        "AZURE_SPEECH_KEY": "",
        "AZURE_SPEECH_REGION": "eastasia",
        "ELEVENLABS_API_KEY": "",
        "CUSTOM_TTS_ENDPOINT": "",
    },
}


def _deep_merge(base: dict, over: dict) -> dict:
    out = deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config() -> dict:
    cfg = deepcopy(DEFAULTS)
    if os.path.isfile(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = _deep_merge(cfg, json.load(f))
        except Exception:
            pass
    # 环境变量覆盖密钥
    for k in list(cfg.get("keys", {}).keys()):
        v = os.environ.get(k)
        if v:
            cfg["keys"][k] = v
    return cfg


def save_config(cfg: dict) -> None:
    safe = deepcopy(cfg)
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(safe, f, ensure_ascii=False, indent=2)


def mask_key(v: str) -> str:
    if not v:
        return ""
    if len(v) <= 8:
        return "***"
    return v[:4] + "***" + v[-4:]


def public_config(cfg: dict) -> dict:
    """给前端的配置：密钥只回显掩码与是否已配置。"""
    pub = deepcopy(cfg)
    keys = pub.get("keys", {})
    for k, v in list(keys.items()):
        if k.endswith("BASE_URL") or k.endswith("ENDPOINT") or k.endswith("REGION"):
            continue
        keys[k] = mask_key(v)
    pub["_keyStatus"] = {k: bool((cfg.get("keys", {}) or {}).get(k))
                         for k in (cfg.get("keys", {}) or {})}
    return pub


def get_key(cfg: dict, name: str) -> str:
    return (cfg.get("keys", {}) or {}).get(name, "") or os.environ.get(name, "")


def ensure_dirs() -> None:
    for d in (WORK_DIR, OUT_DIR):
        os.makedirs(d, exist_ok=True)
