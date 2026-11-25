# -*- coding: utf-8 -*-
"""生成「像真音声」的日语样本：用 Edge 日语神经音色合成台词，再摆成左右耳。

    python tools/make_speech_sample.py out.mp3 --voice-side left

与 make_sample.py 的区别：
  make_sample.py   用合成音（音调脉冲），零依赖、随时可跑，用于验证左右耳判定；
  本脚本           用 Edge 日语神经语音合成真实日语朗读，用于验证识别→翻译→配音全链路。
                  需要能连上 Edge 语音服务（免费、无需密钥）。

立体声构造（贴近真实音声做法）：
  人声合成后按目标侧摆放，对侧按 leak 比例泄漏；环境音效只放在语音停顿处。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import audiocore, edge_tts, tts  # noqa: E402

SR = 48000

# 一段自然的日语音声台词（含停顿与语气）
LINES = [
    "ねえ、聞こえてる？",
    "今日はちょっと疲れてるみたいだね。",
    "無理しないで、ゆっくり休んでいいんだよ。",
    "耳元で、小さな声で話すね。",
    "日本語の声は左側、中国語は反対側に流れるよ。",
]


def say(text: str, out_path: str, voice: str = "ja-JP-NanamiNeural",
        rate: str = "+0%") -> bool:
    try:
        info = tts.clean_for_tts(text)
        edge_tts.synthesize(info or text, out_path, voice=voice, rate=rate,
                            on_log=None)
        return os.path.isfile(out_path) and os.path.getsize(out_path) > 2000
    except Exception as e:
        print("  语音合成失败：%s: %s" % (type(e).__name__, str(e)[:120]))
        return False


def synth_lines(voice: str, pace: float, work: str) -> list[np.ndarray]:
    """合成全部台词，返回 48k 单声道片段列表。一次合成可复用于多种样本。"""
    clips: list[np.ndarray] = []
    rate = "%+d%%" % int(round((pace - 1.0) * 100))
    for i, line in enumerate(LINES):
        wav = os.path.join(work, "_sp_%d.mp3" % i)
        if say(line, wav, voice=voice, rate=rate):
            pcm = audiocore.decode_pcm(wav, SR, 1)
            pcm = pcm[:, 0] if pcm.ndim > 1 else pcm
            clips.append(pcm.astype(np.float32))
            print("  合成第 %d 句：%r（%.2f 秒）" % (i + 1, line, pcm.size / SR))
        try:
            os.remove(wav)
        except OSError:
            pass
    return clips


def build_stereo(clips: list[np.ndarray], voice_side: str, leak: float,
                 gap_seconds: float = 0.7) -> np.ndarray:
    """把语音片段摆成目标立体声。供各处复用（含 selftest）。"""
    gap = np.zeros(int(gap_seconds * SR), dtype=np.float32)
    voice = np.concatenate([np.concatenate([c, gap]) for c in clips])
    peak = float(np.max(np.abs(voice)) or 1.0)
    voice = voice / peak * 0.32

    # 语音活动掩码：音效只铺在停顿处，贴近真实音声
    win = int(0.02 * SR)
    frames = voice.size // win
    rms = np.sqrt(np.mean(voice[:frames * win].reshape(frames, win) ** 2, axis=1))
    active = rms > (float(np.max(rms)) * 0.05)
    vmask = np.concatenate([np.repeat(active, win),
                            np.zeros(voice.size - frames * win, dtype=bool)])
    kernel = np.ones(int(0.12 * SR), dtype=np.float32)
    kernel /= kernel.sum()
    sfx_env = np.clip(1.0 - np.convolve(vmask.astype(np.float32), kernel,
                                        mode="same") * 3.0, 0.0, 1.0)

    n = voice.size
    rng = np.random.default_rng(11)
    noise = np.convolve(rng.normal(0, 1, n).astype(np.float32),
                        np.ones(9, dtype=np.float32) / 9, mode="same")
    t = np.arange(n) / SR
    rumble = (0.5 * np.sin(2 * np.pi * 58 * t) + 0.3 * np.sin(2 * np.pi * 91 * t)).astype(np.float32)
    sfx = ((noise * 0.07 + rumble * 0.05) * sfx_env).astype(np.float32)

    leak = max(0.0, min(1.0, leak))
    L = np.zeros(n, dtype=np.float32)
    R = np.zeros(n, dtype=np.float32)
    if voice_side == "left":
        L += voice
        R += sfx + voice * leak
    elif voice_side == "right":
        R += voice
        L += sfx + voice * leak
    else:                       # center：人声两耳等量，音效偏一侧
        L += voice * 0.9
        R += voice * 0.9 + sfx

    stereo = np.stack([L, R], axis=1)
    pk = float(np.max(np.abs(stereo)) or 1.0)
    if pk > 0.95:
        stereo = stereo / pk * 0.95
    return stereo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default=None,
                    help="输出 mp3；用 --clips-cache 时可省略")
    ap.add_argument("--voice-side", default="left", choices=["left", "right", "center"])
    ap.add_argument("--leak", type=float, default=0.0)
    ap.add_argument("--voice", default="ja-JP-NanamiNeural")
    ap.add_argument("--pace", type=float, default=1.0, help="念稿速度倍率（1.0=原速）")
    ap.add_argument("--clips-cache", default=None,
                    help="把合成好的语音片段存成 npz，供多种摆位复用（只合成一次）")
    args = ap.parse_args()

    if args.clips_cache:
        work = os.path.dirname(os.path.abspath(args.clips_cache)) or "."
        os.makedirs(work, exist_ok=True)
        clips = synth_lines(args.voice, args.pace, work)
        if not clips:
            print("没有合成出任何语音；请检查网络")
            return 1
        np.savez_compressed(args.clips_cache,
                            **{"c%d" % i: c for i, c in enumerate(clips)})
        print("已缓存 %d 段语音 → %s" % (len(clips), args.clips_cache))
        return 0

    if not args.out:
        print("需要提供输出路径")
        return 2
    work = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(work, exist_ok=True)
    clips = synth_lines(args.voice, args.pace, work)
    if not clips:
        print("没有合成出任何语音；请检查网络或改用 make_sample.py（离线合成音）")
        return 1

    stereo = build_stereo(clips, args.voice_side, args.leak)
    audiocore.write_pcm_to_file(stereo, SR, args.out,
                                extra_args=["-c:a", "libmp3lame", "-b:a", "192k"])
    info = audiocore.probe(args.out)
    print("已生成：%s（%.2f 秒，人声在%s%s）"
          % (args.out, info.duration,
             {"left": "左耳", "right": "右耳", "center": "居中"}[args.voice_side],
             "，泄漏 %d%%" % int(args.leak * 100) if args.leak else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
