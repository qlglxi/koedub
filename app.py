#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""KoeDub 启动入口。

    python app.py                 # 启动 Web 界面（默认自动开浏览器）
    python app.py --port 9000     # 指定端口
    python app.py --no-browser    # 不开浏览器
    python app.py --cli 输入.mp3  # 命令行跑完整流水线
    python app.py --selftest      # 环境自检（ffmpeg / ASR / TTS）
    python app.py --apis          # 打印所需 API 种类清单
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _fix_console_encoding() -> None:
    """让控制台中文正常显示。

    Windows 控制台可能是 GBK(936) 也可能是 UTF-8(65001)，写错就是一片乱码。
    这里按控制台**实际**编码去配置 stdout/stderr，无法编码的字符用替代符，
    绝不因为编码问题让程序崩或输出乱码。
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            continue
        try:
            enc = getattr(stream, "encoding", None) or "utf-8"
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding=enc, errors="replace")
            else:
                buf = getattr(stream, "buffer", None)
                if buf is not None:
                    setattr(sys, name, io.TextIOWrapper(
                        buf, encoding=enc, errors="replace", line_buffering=True))
        except Exception:
            pass


_fix_console_encoding()

from core import deps  # noqa: E402

# 可选依赖装在工程内的 vendor/ 目录，导入其它模块前先挂到搜索路径上
deps.ensure_on_path()

from core import asr, audiocore, config as cfgmod, providers, server, tts  # noqa: E402


def cmd_pick_python() -> int:
    """判断当前解释器能否运行本软件（供启动器逐个候选调用）。

    退出码 0 = 可用，非 0 = 不可用，原因打到 stderr。判据：
      1. 能 import numpy（核心音频计算需要）；
      2. 若 vendor/ 里装过可选依赖，当前 Python 版本必须与之一致 ——
         含 C 扩展的包（ctranslate2/av/tokenizers）跨版本会报奇怪的属性错误。
    """
    if sys.version_info < (3, 9):
        sys.stderr.write("需要 Python 3.9+\n")
        return 2
    try:
        import numpy  # noqa: F401
    except Exception as e:
        sys.stderr.write("缺少 numpy: %s\n" % e)
        return 3
    mm = deps.version_mismatch()
    if mm:
        sys.stderr.write("可选依赖为 %s 安装，当前为 %s，版本不一致\n"
                         % (mm["installedWith"], mm["running"]))
        return 4
    print(deps._python_tag())
    return 0


def cmd_apis() -> None:
    s = providers.summary()
    print("=" * 78)
    print("KoeDub 需要调用的 API 种类清单")
    print("=" * 78)
    print("能力类别共 %d 种（必选 %d + 可选 %d），厂商实现 %d 个，云端点 %d 个"
          % (s["capabilityTotal"], s["capabilityRequired"], s["capabilityOptional"],
             s["providerTotal"], s["cloudEndpoints"]))
    print()
    for i, c in enumerate(s["capabilities"], 1):
        tag = "必选" if c["kind"] == "required" else "可选"
        print("%d. [%s] %s / %s" % (i, tag, c["name"], c["name_en"]))
        print("   用途：%s" % c["purpose"])
        print("   默认：%s" % c["default"])
        for p in c["providers"]:
            flags = []
            if p["needs_key"]:
                flags.append("需密钥")
            if p["offline"]:
                flags.append("离线")
            if p["recommended"]:
                flags.append("推荐")
            mark = (" （%s）" % "、".join(flags)) if flags else ""
            ep = ("  %s" % p["endpoint"]) if p["endpoint"] else ""
            print("     - %-28s %s%s%s" % (p["name"], p["engine"], ep, mark))
        print()


def cmd_selftest() -> None:
    cfg = cfgmod.load_config()
    asr.setup_hf_mirror()          # 自检时就把模型下载端点定好，界面能读到
    print("=" * 78)
    print("环境自检")
    print("=" * 78)
    print("Python      :", sys.version.split()[0])
    print("工作目录    :", cfgmod.BASE_DIR)
    ok = audiocore.have_ffmpeg()
    print("ffmpeg      :", audiocore.ffmpeg_path() if ok else "未找到（请安装 FFmpeg）")
    print("ffprobe     :", audiocore.ffprobe_path() if ok else "未找到")
    avail = asr.available_providers()
    print("ASR 可用    :", ", ".join(k for k, v in avail.items() if v) or "无")
    print("ASR 自动选择:", asr.pick_provider(cfg))
    print("识别模型源  :", asr.setup_hf_mirror())
    models = asr.models_on_disk()
    print("已缓存模型  :", "、".join("%s(%.0fMB)" % (m["model"], m["bytes"] / 1048576)
                                    for m in models) or "无（首次识别时会下载）")
    print("可选依赖    :", end=" ")
    deps.self_test(lambda m: print(m.strip(), end="  "))
    print()
    print("翻译自动选择:", __import__("core.translate", fromlist=["x"]).pick_provider(cfg))
    print("配音自动选择:", tts.pick_provider(cfg))
    print("配音自检    :", json.dumps(tts.self_test(cfg, print), ensure_ascii=False))
    print()
    print("安装本地识别: python app.py --install faster-whisper")
    print()


def cmd_cli(path: str, cfg_over: dict | None = None) -> int:
    cfg = cfgmod.load_config()
    if cfg_over:
        for k, v in cfg_over.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v

    def on_event(stage, pct, msg, data):
        prefix = {"log": "  ", "error": "!!"}.get(stage, ">>")
        print("%s %s" % (prefix, msg))

    try:
        res = __import__("core.pipeline", fromlist=["x"]).run(path, cfg, on_event)
    except Exception as e:
        print("\n失败：%s: %s" % (type(e).__name__, e))
        return 1
    print("\n完成。输出目录：%s" % res["outputDir"])
    for k, v in (res.get("outputs") or {}).items():
        print("  %-4s %s" % (k, v))
    for k, v in (res.get("previews") or {}).items():
        if isinstance(v, str) and os.path.isfile(v):
            print("  试听 %s" % v)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="KoeDub - 音声双语化工作台")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--cli", metavar="MP3")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--apis", action="store_true")
    ap.add_argument("--asr", default=None, help="覆盖 ASR provider")
    ap.add_argument("--translate", default=None, help="覆盖翻译 provider")
    ap.add_argument("--tts", default=None, help="覆盖配音 provider")
    ap.add_argument("--voice", default=None, help="覆盖配音音色")
    ap.add_argument("--force-side", default=None, choices=["left", "right", "center"])
    ap.add_argument("--install", default=None, metavar="PKG",
                    help="安装可选依赖（faster-whisper / demucs），自动走国内镜像")
    ap.add_argument("--force", action="store_true",
                    help="与 --install 同用：强制重装（换过 Python 版本时需要）")
    ap.add_argument("--deps", action="store_true", help="查看可选依赖状态")
    ap.add_argument("--app-window", action="store_true",
                    help="用独立应用窗口打开（Edge/Chrome 应用模式，像桌面软件）")
    ap.add_argument("--shortcut", action="store_true", help="在桌面创建快捷方式")
    ap.add_argument("--pick-python", action="store_true",
                    help="内部用：判断当前解释器是否可用（供启动器挑选）")
    ap.add_argument("--tools", default=None, metavar="SCRIPT",
                    help="用选定的 Python 运行 tools/ 下的脚本，"
                         "例如 --tools selftest.py --with-speech")
    args, rest = ap.parse_known_args()

    if args.tools:
        script = args.tools
        if not script.endswith(".py"):
            script += ".py"
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", script)
        if not os.path.isfile(path):
            print("找不到脚本：%s" % path)
            return 2
        import subprocess as _sp
        return _sp.call([sys.executable, path] + list(rest))

    if args.pick_python:
        return cmd_pick_python()
    if args.shortcut:
        from core.launcher import make_shortcut
        print("已创建桌面快捷方式：%s" % make_shortcut())
        return 0
    if args.deps:
        print("可选依赖状态（核心链路不需要它们，装不上也能跑）")
        deps.self_test(print)
        st = deps.status()
        print("  pip 镜像：%s" % st["_env"]["pipIndex"])
        print("  安装位置：%s" % st["_env"]["vendor"])
        models = asr.models_on_disk()
        print("  已下载的本地模型：%s"
              % ("、".join("%s(%.0fMB)" % (m["model"], m["bytes"] / 1048576)
                           for m in models) or "无"))
        return 0
    if args.install:
        key = args.install.replace("-", "_")
        if key not in deps.OPTIONAL:
            key = {"fasterwhisper": "faster_whisper"}.get(key, key)
        print("开始安装 %s …（当前解释器 %s）" % (key, deps._python_tag()))
        res = deps.install(key, on_log=lambda m: print("  " + m, flush=True),
                           force=args.force)
        if res.get("ok"):
            print("安装完成。重启软件即可使用。")
            return 0
        print("安装失败：%s" % res.get("error"))
        return 1
    if args.apis:
        cmd_apis()
        return 0
    if args.selftest:
        cmd_selftest()
        return 0
    if args.cli:
        over: dict = {}
        if args.asr:
            over.setdefault("asr", {})["provider"] = args.asr
        if args.translate:
            over.setdefault("translate", {})["provider"] = args.translate
        if args.tts:
            over.setdefault("tts", {})["provider"] = args.tts
        if args.voice:
            over.setdefault("tts", {})["voice"] = args.voice
        if args.force_side:
            over.setdefault("channel", {})["forceSide"] = args.force_side
        return cmd_cli(args.cli, over)

    cfg = cfgmod.load_config()
    host = args.host or cfg.get("server", {}).get("host", "127.0.0.1")
    port = args.port or int(cfg.get("server", {}).get("port", 8722))
    open_browser = not args.no_browser and bool(
        cfg.get("server", {}).get("autoOpenBrowser", True))
    app_window = args.app_window or bool(cfg.get("server", {}).get("appWindow", False))
    server.serve(host=host, port=port, open_browser=open_browser,
                 app_window=app_window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
