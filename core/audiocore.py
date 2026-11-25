# -*- coding: utf-8 -*-
"""音频核心：ffmpeg/ffprobe 封装 + PCM(numpy) 读写与变换。

设计原则：除 ffmpeg 外不依赖任何外部程序；numpy 用于 PCM 数值计算。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

# ----------------------------------------------------------------------------
# 二进制定位
# ----------------------------------------------------------------------------
_FFMPEG_CACHE: dict[str, str] = {}


def _which(name: str) -> str | None:
    p = shutil.which(name)
    if p:
        return p
    # 常见 WinGet / scoop / choco 安装位置兜底
    roots = [
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links"),
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages"),
        r"C:\ffmpeg\bin",
        os.path.expandvars(r"%ProgramFiles%\ffmpeg\bin"),
    ]
    exe = name + (".exe" if os.name == "nt" else "")
    for root in roots:
        if not os.path.isdir(root):
            continue
        direct = os.path.join(root, exe)
        if os.path.isfile(direct):
            return direct
        for dirpath, _dirs, files in os.walk(root):
            if exe in files:
                return os.path.join(dirpath, exe)
            if dirpath.count(os.sep) - root.count(os.sep) > 4:
                _dirs[:] = []
    return None


def ffmpeg_path() -> str:
    if "ffmpeg" not in _FFMPEG_CACHE:
        p = _which("ffmpeg")
        if not p:
            raise RuntimeError("找不到 ffmpeg，请先安装（winget install Gyan.FFmpeg）")
        _FFMPEG_CACHE["ffmpeg"] = p
    return _FFMPEG_CACHE["ffmpeg"]


def ffprobe_path() -> str:
    if "ffprobe" not in _FFMPEG_CACHE:
        p = _which("ffprobe")
        if not p:
            raise RuntimeError("找不到 ffprobe，请先安装 FFmpeg")
        _FFMPEG_CACHE["ffprobe"] = p
    return _FFMPEG_CACHE["ffprobe"]


def have_ffmpeg() -> bool:
    try:
        ffmpeg_path()
        ffprobe_path()
        return True
    except RuntimeError:
        return False


def _run(args: list[str], **kw) -> subprocess.CompletedProcess:
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.run(args, capture_output=True, creationflags=creationflags, **kw)


# ----------------------------------------------------------------------------
# ffprobe
# ----------------------------------------------------------------------------
@dataclass
class MediaInfo:
    path: str
    duration: float = 0.0
    sample_rate: int = 44100
    channels: int = 2
    codec: str = ""
    bit_rate: int = 0
    format_name: str = ""
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "duration": round(self.duration, 3),
            "durationText": format_time(self.duration),
            "sampleRate": self.sample_rate,
            "channels": self.channels,
            "codec": self.codec,
            "bitRate": self.bit_rate,
            "format": self.format_name,
        }


def probe(path: str) -> MediaInfo:
    """读取媒体基本信息。"""
    args = [ffprobe_path(), "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", path]
    cp = _run(args)
    if cp.returncode != 0:
        raise RuntimeError("ffprobe 失败: %s" % cp.stderr.decode("utf-8", "replace")[:400])
    data = json.loads(cp.stdout.decode("utf-8", "replace"))
    info = MediaInfo(path=path, raw=data)
    fmt = data.get("format", {}) or {}
    info.format_name = fmt.get("format_name", "")
    info.duration = float(fmt.get("duration") or 0.0)
    info.bit_rate = int(float(fmt.get("bit_rate") or 0))
    for st in data.get("streams", []):
        if st.get("codec_type") == "audio":
            info.codec = st.get("codec_name", "")
            info.sample_rate = int(st.get("sample_rate") or 44100)
            info.channels = int(st.get("channels") or 2)
            if not info.duration:
                info.duration = float(st.get("duration") or 0.0)
            break
    return info


# ----------------------------------------------------------------------------
# PCM 读写
# ----------------------------------------------------------------------------
def decode_pcm(path: str, sample_rate: int = 16000,
               channels: int = 1) -> np.ndarray:
    """解码为 float32，形状 (n_samples, channels)，取值范围 [-1, 1]。"""
    args = [ffmpeg_path(), "-v", "error", "-i", path,
            "-f", "s16le", "-acodec", "pcm_s16le",
            "-ar", str(sample_rate), "-ac", str(channels), "-"]
    cp = _run(args)
    if cp.returncode != 0 and not cp.stdout:
        raise RuntimeError("ffmpeg 解码失败: %s" % cp.stderr.decode("utf-8", "replace")[:400])
    raw = np.frombuffer(cp.stdout, dtype="<i2")
    if channels > 1:
        usable = (raw.size // channels) * channels
        raw = raw[:usable].reshape(-1, channels)
    return (raw.astype(np.float32) / 32768.0)


def encode_wav(path: str, pcm: np.ndarray, sample_rate: int) -> str:
    """把 float32 PCM 写成 16bit WAV（不依赖 soundfile）。"""
    import wave
    if pcm.ndim == 1:
        pcm = pcm[:, None]
    data = np.clip(pcm, -1.0, 1.0)
    ints = (data * 32767.0).astype("<i2")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(ints.tobytes())
    return path


def write_pcm_to_file(pcm: np.ndarray, sample_rate: int, out_path: str,
                      extra_args: Iterable[str] = ()) -> str:
    """通过 ffmpeg stdin 管道把 PCM 编码成任意格式。"""
    if pcm.ndim == 1:
        pcm = pcm[:, None]
    ch = pcm.shape[1]
    ints = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype("<i2")
    args = [ffmpeg_path(), "-v", "error", "-y",
            "-f", "s16le", "-ar", str(sample_rate), "-ac", str(ch),
            "-i", "pipe:0", *extra_args, out_path]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    cp = subprocess.run(args, input=ints.tobytes(), capture_output=True,
                        creationflags=creationflags)
    if cp.returncode != 0:
        raise RuntimeError("ffmpeg 编码失败: %s" % cp.stderr.decode("utf-8", "replace")[:400])
    return out_path


# ----------------------------------------------------------------------------
# 变换
# ----------------------------------------------------------------------------
def atempo_filter(tempo: float) -> str:
    """构造保持音高的变速滤镜链（atempo 单级限 0.5~2.0，需级联）。"""
    tempo = max(0.25, min(4.0, float(tempo)))
    parts = []
    t = tempo
    while t > 2.0:
        parts.append("atempo=2.0")
        t /= 2.0
    while t < 0.5:
        parts.append("atempo=0.5")
        t /= 0.5
    parts.append("atempo=%.6f" % t)
    return ",".join(parts)


def convert(src: str, dst: str, sample_rate: int | None = None,
            channels: int | None = None, filters: str | None = None,
            extra: Iterable[str] = ()) -> str:
    """通用转码。"""
    args = [ffmpeg_path(), "-v", "error", "-y", "-i", src]
    if filters:
        args += ["-af", filters]
    if sample_rate:
        args += ["-ar", str(sample_rate)]
    if channels:
        args += ["-ac", str(channels)]
    args += list(extra)
    args.append(dst)
    cp = _run(args)
    if cp.returncode != 0:
        raise RuntimeError("ffmpeg 转换失败: %s" % cp.stderr.decode("utf-8", "replace")[:400])
    return dst


def synth_to_fit(src: str, dst: str, target_seconds: float,
                 max_tempo: float = 1.45) -> dict:
    """把配音音频变速到目标时长（保持不变调）。

    排轴模块 align.py 会算好变速倍率并自己渲染；这里保留一个通用封装，
    供脚本或二次开发直接调用。

    只压缩，不把语音拉长超过 max_tempo（拉长会让语音发飘，宁可留白）。
    """
    info = probe(src)
    src_dur = info.duration or 0.001
    ratio = src_dur / max(target_seconds, 0.05)
    tempo = 1.0
    if ratio > 1.0:
        tempo = min(ratio, max_tempo)
    filters = atempo_filter(tempo) if abs(tempo - 1.0) > 0.01 else None
    convert(src, dst, sample_rate=48000, channels=1, filters=filters,
            extra=["-c:a", "pcm_s16le"] if dst.lower().endswith(".wav") else [])
    new_dur = probe(dst).duration
    return {
        "sourceDuration": round(src_dur, 3),
        "targetDuration": round(target_seconds, 3),
        "tempo": round(tempo, 4),
        "outputDuration": round(new_dur, 3),
        "clipped": ratio > max_tempo,
    }


# ----------------------------------------------------------------------------
# 字幕烧录
# ----------------------------------------------------------------------------
def render_subtitles(audio_path: str, ass_or_srt: str, out_path: str,
                     video_size: str = "1280x720") -> str:
    """把字幕烧录进一段静态画面的 mp4，便于直接观看/分享。"""
    ext = os.path.splitext(ass_or_srt)[1].lower()
    if ext == ".ass":
        vf = "ass='%s'" % ass_or_srt.replace("\\", "/").replace(":", "\\:")
    else:
        vf = "subtitles='%s'" % ass_or_srt.replace("\\", "/").replace(":", "\\:")
    args = [ffmpeg_path(), "-v", "error", "-y",
            "-f", "lavfi", "-i", "color=c=0x101418:s=%s:r=8" % video_size,
            "-i", audio_path,
            "-vf", vf, "-shortest",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
            "-c:a", "aac", "-b:a", "192k", out_path]
    cp = _run(args)
    if cp.returncode != 0:
        raise RuntimeError("字幕视频渲染失败: %s" % cp.stderr.decode("utf-8", "replace")[:400])
    return out_path


# ----------------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------------
def format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h:
        return "%d:%02d:%05.2f" % (h, m, s)
    return "%02d:%05.2f" % (m, s)


def srt_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return "%02d:%02d:%02d,%03d" % (h, m, s, ms)


def ass_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    cs = int(round(seconds * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return "%d:%02d:%02d.%02d" % (h, m, s, cs)
