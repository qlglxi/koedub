# -*- coding: utf-8 -*-
"""API 能力注册表 —— 本软件的「需要调用多少种 API」单一事实来源。

把整条流水线拆成 8 个能力类别（capability）。每个类别下可以有多个 provider
（厂商实现），但类别本身才是「一种 API」。UI 与文档都从这里生成，不做二次硬编码。

必选 6 类：
  probe        媒体探测        ffprobe/ffmpeg（本地）
  asr          语音识别        faster-whisper / OpenAI / DeepSeek / Groq / 硅基流动 / Demo / 手工
  translate    自适应翻译      DeepSeek / OpenAI 兼容 / Ollama / 术语表引擎 / 直通
  tts          AI 配音         Edge 神经语音 / Azure / OpenAI / ElevenLabs / 自定义端点 / GPT-SoVITS 调教音色 / SAPI 离线
  align        排轴对齐        WSOLA/atempo + 时间轴求解（纯本地算法）
  mix          声道摆位合成    pan/amix/join（本地 ffmpeg 滤镜链）

可选 2 类：
  separate     人声/伴奏分离   Demucs / UVR / 频谱中置提取 / 跳过
  subtitle     字幕导出        SRT / ASS / 双语（本地生成，无需云 API）
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

REQUIRED = "required"
OPTIONAL = "optional"


@dataclass
class Provider:
    id: str
    name: str
    engine: str              # 实际调用的东西：ffmpeg / wss / https / 本地模型 / 内置算法
    endpoint: str = ""
    needs_key: bool = False
    offline: bool = False
    note: str = ""
    recommended: bool = False
    available: bool = True    # 运行时会按网络/依赖重新评估

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Capability:
    id: str
    name: str
    name_en: str
    kind: str
    purpose: str
    default: str
    providers: list[Provider] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["providerCount"] = len(self.providers)
        return d


def build_registry() -> list[Capability]:
    return [
        Capability(
            id="probe", name="媒体探测", name_en="Media Probe", kind=REQUIRED,
            purpose="读取 MP3 的时长/采样率/声道/码率，供后续排轴使用。",
            default="ffprobe",
            providers=[
                Provider("ffprobe", "FFprobe (FFmpeg)", "本地可执行文件",
                         "ffprobe -show_format -show_streams", offline=True,
                         note="随 FFmpeg 一起安装，无需网络", recommended=True),
            ],
        ),
        Capability(
            id="asr", name="语音识别 (ASR)", name_en="Speech-to-Text", kind=REQUIRED,
            purpose="把日语原声转成带时间戳的文本（时间戳是排轴的骨架）。",
            default="whisper_local",
            providers=[
                Provider("whisper_local", "faster-whisper 本地模型", "本地模型",
                         "pip install faster-whisper", offline=True,
                         note="离线、免费、不传音频出门；首次需下载模型", recommended=True),
                Provider("openai_asr", "OpenAI Whisper API", "https",
                         "POST https://api.openai.com/v1/audio/transcriptions",
                         needs_key=True, note="需 OPENAI_API_KEY；本机当前不可达"),
                Provider("deepseek_asr", "DeepSeek 兼容 ASR", "https",
                         "POST https://api.deepseek.com/v1/audio/transcriptions",
                         needs_key=True, note="需 DEEPSEEK_API_KEY；域名可达"),
                Provider("groq_asr", "Groq Whisper (极快)", "https",
                         "POST https://api.groq.com/openai/v1/audio/transcriptions",
                         needs_key=True, note="需 GROQ_API_KEY；本机可达(403)"),
                Provider("siliconflow_asr", "硅基流动 SenseVoice", "https",
                         "POST https://api.siliconflow.cn/v1/audio/transcriptions",
                         needs_key=True, note="国内节点，需 SILICONFLOW_API_KEY；可达"),
                Provider("demo", "Demo 内置识别器", "内置算法",
                         "", offline=True,
                         note="无模型时的演示/自测用，产出占位时间轴"),
                Provider("manual", "手工导入文稿", "本地文件",
                         "", offline=True,
                         note="你自己提供带时间戳的日文文本/字幕"),
            ],
        ),
        Capability(
            id="translate", name="自适应翻译", name_en="Adaptive Translation", kind=REQUIRED,
            purpose="把日语识别文本译成中文；「自适应」= 按上下文、人称、音声语气自动调整，"
                    "不是逐句直译。",
            default="glossary",
            providers=[
                Provider("deepseek", "DeepSeek Chat", "https",
                         "POST https://api.deepseek.com/chat/completions",
                         needs_key=True, note="中文语气把握最好，需 DEEPSEEK_API_KEY", recommended=True),
                Provider("openai_compat", "任意 OpenAI 兼容端点", "https",
                         "POST {base_url}/chat/completions", needs_key=True,
                         note="OpenAI / 硅基流动 / 通义 / 自建 vLLM 都能接"),
                Provider("ollama", "Ollama 本地模型", "https",
                         "POST http://127.0.0.1:11434/api/chat", offline=True,
                         note="完全离线，本机需先跑起 ollama"),
                Provider("glossary", "内置自适应翻译引擎", "内置算法+术语表",
                         "", offline=True,
                         note="离线兜底：术语表替换 + 长短句重排 + 时间预算裁剪",
                         recommended=True),
                Provider("passthrough", "直通（不翻译）", "本地",
                         "", offline=True, note="只做排轴与配音，文本保持原文"),
            ],
        ),
        Capability(
            id="tts", name="AI 配音", name_en="Text-to-Speech", kind=REQUIRED,
            purpose="把中文台词合成语音，按每段的时间预算生成音频。",
            default="edge",
            providers=[
                Provider("edge", "Edge 神经语音", "wss",
                         "wss://speech.platform.bing.com/consumer/speech/synthesize/readaloud/edge/v1",
                         offline=False, note="免费、中文音色自然、无需密钥；已内置零依赖 WebSocket 客户端",
                         recommended=True),
                Provider("azure", "Azure 认知服务语音", "https",
                         "POST https://{region}.tts.speech.microsoft.com/cognitiveservices/v1",
                         needs_key=True, note="音色最多、支持情绪/风格，需 AZURE_SPEECH_KEY"),
                Provider("openai_tts", "OpenAI TTS", "https",
                         "POST https://api.openai.com/v1/audio/speech",
                         needs_key=True, note="需 OPENAI_API_KEY"),
                Provider("elevenlabs", "ElevenLabs", "https",
                         "POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                         needs_key=True, note="音质顶级、中文可用、需 ELEVENLABS_API_KEY"),
                Provider("custom_http", "自定义 HTTP 语音端点", "https",
                         "POST {your_endpoint}", note="任何返回音频字节的接口都能接"),
                Provider("gpt_sovits", "GPT-SoVITS 调教音色", "本地 HTTP",
                         "POST http://127.0.0.1:9880/tts", offline=True,
                         note="用自己训练的权重和参考音频零样本合成，音色与原始音源同源；"
                              "需先启动 GPT-SoVITS 的 api_v2_sf.py，并配好 tts.sovitsRef / sovitsPromptText"),
                Provider("pyttsx3_sapi", "Windows SAPI 离线语音", "本地 COM",
                         "", offline=True,
                         note="完全离线兜底；本机有 Microsoft Huihui(zh-CN) 音色"),
            ],
        ),
        Capability(
            id="align", name="排轴对齐", name_en="Timeline Alignment", kind=REQUIRED,
            purpose="让每句中文严格落在原文对应时间段内：超长按语速阈值压缩，"
                    "过短则留白，绝不提前/延后。",
            default="builtin",
            providers=[
                Provider("builtin", "内置排轴求解器 + WSOLA/atempo", "内置算法",
                         "ffmpeg atempo 变频不变调", offline=True,
                         note="毫秒级吸附，本地完成", recommended=True),
            ],
        ),
        Capability(
            id="mix", name="声道摆位与合成", name_en="Channel Routing & Mixdown",
            kind=REQUIRED,
            purpose="原声保留在原耳，中文配音放到完全对侧耳，声画同步后封装成片。",
            default="builtin",
            providers=[
                Provider("builtin", "内置声道路由 + ffmpeg 滤镜链", "内置算法",
                         " filter_complex: pan/adelay/amix/join", offline=True,
                         note="支持每条语音单独判侧、单耳替换、可选原侧闪避", recommended=True),
            ],
        ),
        Capability(
            id="separate", name="人声分离", name_en="Vocal Separation", kind=OPTIONAL,
            purpose="把原声里的人声与音效/环境音分开，实现真正的「左耳日语、右耳中文」"
                    "且两边都干净。",
            default="none",
            providers=[
                Provider("none", "不分离（推荐先用）", "跳过", "", offline=True,
                         note="立体声本身已能靠声道区分，多数音声够用", recommended=True),
                Provider("demucs", "Demucs (htdemucs)", "本地模型",
                         "pip install demucs", offline=True,
                         note="质量最好，需下载模型 + CPU/GPU 推理"),
                Provider("uvr", "UVR / MDX-Net", "本地模型", "", offline=True,
                         note="需单独安装 Ultimate Vocal Remover"),
                Provider("center_extract", "中置提取", "内置算法",
                         "Mid/Side 分解", offline=True,
                         note="零依赖近似方案，仅对真·居中人声有效"),
            ],
        ),
        Capability(
            id="subtitle", name="字幕导出", name_en="Subtitle Export", kind=OPTIONAL,
            purpose="输出中日双语字幕，可烧录成视频便于核对与分享。",
            default="builtin",
            providers=[
                Provider("builtin", "内置 SRT/ASS 生成", "内置算法", "", offline=True,
                         note="双语 ASS 样式可直接烧录进 MP4", recommended=True),
            ],
        ),
    ]


def summary() -> dict:
    """给 UI / 文档用的统计：总共需要调用多少种不同的 API。"""
    reg = build_registry()
    req = [c for c in reg if c.kind == REQUIRED]
    opt = [c for c in reg if c.kind == OPTIONAL]
    endpoints = set()
    for c in reg:
        for p in c.providers:
            if p.endpoint:
                endpoints.add(p.endpoint.split(" ")[-1].split("{")[0].rstrip("/"))
    engines: dict[str, int] = {}
    for c in reg:
        for p in c.providers:
            engines[p.engine] = engines.get(p.engine, 0) + 1
    return {
        "capabilityTotal": len(reg),
        "capabilityRequired": len(req),
        "capabilityOptional": len(opt),
        "providerTotal": sum(len(c.providers) for c in reg),
        "cloudEndpoints": len(endpoints),
        "engines": engines,
        "capabilities": [c.to_dict() for c in reg],
    }


if __name__ == "__main__":
    import json
    print(json.dumps(summary(), ensure_ascii=False, indent=2))
