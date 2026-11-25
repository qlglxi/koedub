# -*- coding: utf-8 -*-
"""生成用于自测的立体声样本：日语人声偏一侧 + 环境音效在另一侧。

    python tools/make_sample.py tests/sample_left.mp3 --voice-side left

语音来源优先级：
  1. Windows SAPI 日语语音（Microsoft Haruka）——最真实，含真实停顿；
  2. 纯合成音（带包络的脉冲串）——SAPI 不可用时的兜底，仍能客观验证左右耳判定。

立体声用 numpy 直接搭建，避免 ffmpeg pan 滤镜的语法坑，同时可精确控制
「人声段偏一侧、音效段偏另一侧」的时序结构。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import audiocore  # noqa: E402

SR = 48000

JA_LINES = [
    "ねえ、聞こえてる？",
    "今日はゆっくり休んでね。",
    "耳元でささやくよ。",
    "こちらは日本語の声、中国語は反対側の耳に流れるよ。",
    "ゆっくり息をして。",
]

SAPI_PS = r"""
param([string]$Text,[string]$Out,[string]$Voice)
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
try { if ($Voice) { $s.SelectVoice($Voice) } } catch {}
$s.Rate = -1
$s.SetOutputToWaveFile($Out)
$s.Speak($Text)
$s.Dispose()
"""


def read_wav_mono(path: str) -> np.ndarray:
    """读 WAV 成 48k 单声道 float32。空文件/读取失败返回空数组（调用方需过滤）。"""
    try:
        with wave.open(path, "rb") as w:
            ch = w.getnchannels()
            sr = w.getframerate()
            nframes = w.getnframes()
            if nframes <= 0 or ch <= 0 or sr <= 0:
                return np.zeros(0, dtype=np.float32)
            raw = w.readframes(nframes)
    except Exception:
        return np.zeros(0, dtype=np.float32)
    if not raw:
        return np.zeros(0, dtype=np.float32)
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        usable = (a.size // ch) * ch
        if usable == 0:
            return np.zeros(0, dtype=np.float32)
        a = a[:usable].reshape(-1, ch).mean(axis=1)
    if a.size == 0:
        return np.zeros(0, dtype=np.float32)
    if sr != SR:
        # 线性重采样（样本用途足够）
        n = max(1, int(a.size * SR / sr))
        a = np.interp(np.linspace(0, a.size - 1, n), np.arange(a.size), a).astype(np.float32)
    return a


def sapi_say(text: str, out_wav: str, voice: str = "Microsoft Haruka Desktop") -> bool:
    exe = shutil.which("pwsh") or shutil.which("powershell")
    if not exe:
        return False
    script = os.path.join(os.path.dirname(out_wav) or ".", "_sapi.ps1")
    with open(script, "w", encoding="utf-8-sig") as f:
        f.write(SAPI_PS)
    cf = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    cp = subprocess.run([exe, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script,
                         "-Text", text, "-Out", out_wav, "-Voice", voice],
                        capture_output=True, creationflags=cf)
    ok = cp.returncode == 0 and os.path.isfile(out_wav) and os.path.getsize(out_wav) > 2000
    try:
        os.remove(script)
    except OSError:
        pass
    return ok


def synth_voice(line: str, secs: float) -> np.ndarray:
    """兜底语音：带共振峰包络的谐波脉冲串，频谱接近人声。

    谐振器用 biquad 形式（增益已归一化到 1），避免直接 IIR 递推发散。
    """
    n = int(secs * SR)
    t = np.arange(n) / SR
    f0 = 205 + 25 * np.sin(2 * np.pi * 1.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    src = (1.0 * np.sin(phase) + 0.45 * np.sin(2 * phase) + 0.3 * np.sin(3 * phase)
           + 0.15 * np.sin(5 * phase)).astype(np.float64)
    out = src.copy()
    for fc, bw, amp in ((700, 90, 1.0), (1220, 110, 0.6), (2600, 160, 0.35)):
        r = float(np.exp(-np.pi * bw / SR))
        theta = 2 * np.pi * fc / SR
        a1, a2 = -2.0 * r * np.cos(theta), r * r
        b0 = (1 - r) * np.sqrt(1 - 2 * r * np.cos(2 * theta) + r * r + 1e-12)
        y = np.zeros(n, dtype=np.float64)
        for i in range(2, n):
            y[i] = b0 * src[i] - a1 * y[i - 1] - a2 * y[i - 2]
        m = float(np.max(np.abs(y)) or 1.0)
        out = out + amp * (y / m)
    m = float(np.max(np.abs(out)) or 1.0)
    out = out / m
    # 音节包络（约 5.5 音节/秒）
    syl = 0.5 + 0.5 * np.sin(2 * np.pi * 5.5 * t)
    env = np.clip(syl, 0, 1) ** 1.4
    fade = max(1, int(0.05 * SR))
    env[:fade] *= np.linspace(0, 1, fade)
    env[-fade:] *= np.linspace(1, 0, fade)
    sig = out * env * 0.35
    return np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--voice-side", default="left", choices=["left", "right", "center"])
    ap.add_argument("--batch", type=int, default=5, help="语音句数/重复次数")
    ap.add_argument("--no-sapi", action="store_true")
    ap.add_argument("--leak", type=float, default=0.0,
                    help="对侧人声泄漏比例（0~1），用于模拟双耳混音")
    args = ap.parse_args()

    work = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(work, exist_ok=True)

    # ---------- 1) 取得「日语语音」素材 ----------
    clips: list[np.ndarray] = []
    used_sapi = False
    if not args.no_sapi:
        for i, line in enumerate(JA_LINES):
            wav = os.path.join(work, "_ja_%d.wav" % i)
            if sapi_say(line, wav):
                got = read_wav_mono(wav)
                if got.size > 0:
                    clips.append(got)
                    used_sapi = True
            try:
                os.remove(wav)
            except OSError:
                pass
            if len(clips) >= args.batch:
                break
    if not clips:
        print("SAPI 日语语音不可用，改用合成音（仍可验证左右耳判定）")
        clips = [synth_voice(line, 2.2 + 0.3 * i) for i, line in enumerate(JA_LINES)]
    else:
        print("SAPI 日语语音：OK（%d 句）" % len(clips))

    # ---------- 2) 拼成时间轴：语音段之间留停顿 ----------
    gap = int(0.75 * SR)
    pieces: list[np.ndarray] = []
    for c in clips:
        pieces.append(c)
        pieces.append(np.zeros(gap, dtype=np.float32))
    voice = np.concatenate(pieces)
    peak = float(np.max(np.abs(voice)) or 1.0)
    voice = voice / peak * 0.25
    n = voice.size

    # 语音活动掩码：用于让音效只出现在停顿处（真实音声里环境音本就在台词间隙）
    win = int(0.02 * SR)
    frames = n // win
    rms = np.sqrt(np.mean(voice[:frames * win].reshape(frames, win) ** 2, axis=1))
    active = rms > (float(np.max(rms)) * 0.06)
    voice_mask = np.repeat(active, win)
    voice_mask = np.concatenate([voice_mask, np.zeros(n - voice_mask.size, dtype=bool)])
    # 让掩码边缘平滑
    kernel = np.ones(int(0.12 * SR), dtype=np.float32)
    kernel /= kernel.sum()
    smooth = np.convolve(voice_mask.astype(np.float32), kernel, mode="same")
    sfx_env = np.clip(1.0 - smooth * 3.0, 0.0, 1.0).astype(np.float32)

    # ---------- 3) 环境音效（宽带噪声 + 低频），只在语音停顿处 ----------
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 1, n).astype(np.float32)
    k = 9
    noise = np.convolve(noise, np.ones(k, dtype=np.float32) / k, mode="same")
    t = np.arange(n) / SR
    rumble = (0.5 * np.sin(2 * np.pi * 62 * t) + 0.3 * np.sin(2 * np.pi * 97 * t)).astype(np.float32)
    sfx = ((noise * 0.09 + rumble * 0.07) * sfx_env).astype(np.float32)

    # ---------- 4) 组装立体声 ----------
    # 真实音声的贴耳人声几乎是单侧的，因此人声只放在目标侧、对侧零泄漏；
    # 对侧只承担环境音效（在台词间隙出现）。
    vs = args.voice_side
    leak = max(0.0, min(1.0, args.leak))
    L = np.zeros(n, dtype=np.float32)
    R = np.zeros(n, dtype=np.float32)
    if vs == "left":
        L += voice
        R += sfx + voice * leak
    elif vs == "right":
        R += voice
        L += sfx + voice * leak
    else:  # center：人声两耳等量（双耳混音），音效在右侧
        L += voice * 0.9
        R += voice * 0.9 + sfx

    stereo = np.stack([L, R], axis=1)
    peak = float(np.max(np.abs(stereo)) or 1.0)
    if peak > 0.95:
        stereo = stereo / peak * 0.95

    audiocore.write_pcm_to_file(stereo, SR, args.out,
                                extra_args=["-c:a", "libmp3lame", "-b:a", "192k"])
    info = audiocore.probe(args.out)
    print("已生成：%s（%.2f 秒，%d 声道，人声在原声%s）"
          % (args.out, info.duration, info.channels,
             {"left": "左耳", "right": "右耳", "center": "居中"}[vs]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
