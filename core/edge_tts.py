# -*- coding: utf-8 -*-
"""Edge 神经语音（wss）—— 零依赖实现。

协议要点（与 edge-tts 项目一致）：
  1. 与 wss://speech.platform.bing.com/consumer/speech/synthesize/readaloud/edge/v1
     建连，携带 TrustedClientToken 与 Sec-MS-GEC 签名。
  2. 先发 speech.config 文本帧声明输出格式（mp3）。
  3. 再发 ssml 文本帧（X-RequestId 做消息分组）。
  4. 服务端回文本帧（turn.start / response / audio.metadata / turn.end）
     与二进制帧（音频数据）；收到 turn.end 即结束。
"""
from __future__ import annotations

import hashlib
import re
import time
import uuid

from . import wsclient

TRUSTED_TOKEN = "6A5AA1D4EAFF4E9FB37E23D68491D6F4"
# 与 edge-tts 官方 constants.py 保持同步：微软会校验 Sec-MS-GEC-Version 里的版本号
CHROMIUM_FULL_VERSION = "143.0.3650.75"
CHROMIUM_MAJOR_VERSION = CHROMIUM_FULL_VERSION.split(".", 1)[0]
SEC_MS_GEC_VERSION = "1-%s" % CHROMIUM_FULL_VERSION
EDGE_URL = ("wss://speech.platform.bing.com/consumer/speech/synthesize/"
            "readaloud/edge/v1?TrustedClientToken=" + TRUSTED_TOKEN)
WIN_EPOCH = 11644473600
S_TO_NS = 1e9

# 本地时钟相对服务端的偏移（秒）：签名 403 时按响应 Date 头自动修正后重试
clock_skew_seconds: float = 0.0


def _sec_ms_gec() -> str:
    """生成 Sec-MS-GEC 签名。

    与 edge-tts 官方实现 (drm.py) 完全一致：
      ticks = 秒级 Unix 时间（含时钟偏移修正）+ WIN_EPOCH，向下取整到 300 秒，
      再换算成 100ns 计数；SHA256(ticks + TRUSTED_CLIENT_TOKEN) 取大写十六进制。
    见 https://github.com/rany2/edge-tts/blob/master/src/edge_tts/drm.py
    """
    ticks = time.time() + clock_skew_seconds + WIN_EPOCH
    ticks -= ticks % 300                        # 5 分钟窗口向下取整
    ticks *= S_TO_NS / 100.0                    # 转 100ns 计数
    payload = "%.0f%s" % (ticks, TRUSTED_TOKEN)
    return hashlib.sha256(payload.encode("ascii")).hexdigest().upper()


def adjust_clock_skew_seconds(seconds: float) -> None:
    """按服务端 Date 头修正本地时钟偏差（用于签名过期返回 403 的场景）。"""
    global clock_skew_seconds
    clock_skew_seconds += seconds


def _headers() -> dict[str, str]:
    """握手请求头。

    实测要点：
      - User-Agent 必须带 Edge 标识（Edg/xxx），纯 Chrome UA 会被 403；
      - 不加 Accept-Encoding（禁止压缩），否则二进制音频帧可能被压缩而解不开；
      - 其余保持与官方客户端一致。
    """
    ua = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/%s.0.0.0 Safari/537.36 Edg/%s.0.0.0"
          % (CHROMIUM_MAJOR_VERSION, CHROMIUM_MAJOR_VERSION))
    return {
        "User-Agent": ua,
        "Accept-Language": "en-US,en;q=0.9",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
        "Origin": "chrome-extension://jdiccldimpdaibmpdkjnbmckianbfold",
        "Cookie": "muid=%s;" % __import__("os").urandom(16).hex().upper(),
    }


def _url() -> str:
    """构造 wss 地址。

    注意：实测本机/当前服务端点下带上 &ConnectionId= 会返回 400 Bad Request，
    不带时握手正常（GEC 签名 + TrustedClientToken 齐备即可），因此这里只保留
    必需参数。
    """
    return "%s&Sec-MS-GEC=%s&Sec-MS-GEC-Version=%s" % (
        EDGE_URL, _sec_ms_gec(), SEC_MS_GEC_VERSION)


def _date() -> str:
    return time.strftime("%a %b %d %Y %H:%M:%S GMT+0000 (Coordinated Universal Time)",
                         time.gmtime())


def _xml_escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&apos;"))


def _config_message() -> str:
    return ("X-Timestamp:%s\r\n"
            "Content-Type:application/json; charset=utf-8\r\n"
            "Path:speech.config\r\n\r\n"
            '{"context":{"synthesis":{"audio":{"metadataoptions":'
            '{"sentenceBoundaryEnabled":"false","wordBoundaryEnabled":"false"},'
            '"outputFormat":"audio-24khz-48kbitrate-mono-mp3"}}}}\r\n'
            % _date())


def _ssml_message(text: str, voice: str, rate: str, pitch: str, volume: str,
                  request_id: str) -> str:
    ssml = (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        'xml:lang="en-US"><voice name="%s"><prosody pitch="%s" rate="%s" volume="%s">'
        '%s</prosody></voice></speak>' % (voice, pitch, rate, volume,
                                          _xml_escape(text))
    )
    return ("X-RequestId:%s\r\n"
            "Content-Type:application/ssml+xml\r\n"
            "X-Timestamp:%sZ\r\n"
            "Path:ssml\r\n\r\n%s" % (request_id, _date(), ssml))


def synthesize(text: str, out_path: str, voice: str = "zh-CN-XiaoxiaoNeural",
               rate: str = "+0%", pitch: str = "+0Hz", volume: str = "+0%",
               on_log=None, timeout: float = 60.0) -> dict:
    """合成一段文本为 MP3，返回 {bytes, path, requestId}。"""
    if not text.strip():
        raise ValueError("文本为空")
    log = on_log or (lambda m: None)
    rid = uuid.uuid4().hex
    audio = bytearray()
    events = 0
    ws = wsclient.connect(_url(), headers=_headers(), timeout=timeout)
    try:
        ws.send_text(_config_message())
        ws.send_text(_ssml_message(text, voice, rate, pitch, volume, rid))
        while True:
            opcode, payload = ws.recv()
            if opcode == 0x1:  # 文本帧
                head = payload[:400].decode("utf-8", "replace")
                if "Path:turn.end" in head:
                    break
                if "Path:audio.metadata" in head:
                    events += 1
                if "Path:response" in head and '"status":"' in head:
                    m = re.search(r'"status"\s*:\s*"([^"]+)"', head)
                    if m and m.group(1) not in ("Success", "success"):
                        raise RuntimeError("Edge 语音返回状态异常: %s" % head[:200])
            else:               # 二进制帧：前 2 字节是头长度
                if len(payload) < 2:
                    continue
                hlen = int.from_bytes(payload[:2], "big")
                audio += payload[2 + hlen:]
    finally:
        ws.close()

    if not audio:
        raise RuntimeError("Edge 语音未返回音频数据（可能是签名或网络问题）")
    with open(out_path, "wb") as f:
        f.write(bytes(audio))
    log("  Edge 语音：%s → %d 字节（%s）" % (voice, len(audio), text[:18]))
    return {"path": out_path, "bytes": len(audio), "requestId": rid, "events": events}


def list_voices(timeout: float = 20.0) -> list[dict]:
    """拉取可用音色列表（HTTPS GET，无需登录）。"""
    from . import httpx as hx
    url = ("https://speech.platform.bing.com/consumer/speech/synthesize/"
           "readaloud/voices/list?trustedclienttoken=" + TRUSTED_TOKEN)
    raw = hx.request(url, headers={"User-Agent": _headers()["User-Agent"],
                                   "Accept": "*/*",
                                   "Sec-Fetch-Site": "none",
                                   "Sec-Fetch-Mode": "cors",
                                   "Sec-Fetch-Dest": "empty"},
                     timeout=int(timeout))
    import json as _json
    return _json.loads(raw.decode("utf-8", "replace"))


def self_test(on_log=None) -> dict:
    import os
    import tempfile
    log = on_log or (lambda m: None)
    out = os.path.join(tempfile.gettempdir(), "koedub-edge-test.mp3")
    try:
        info = synthesize("这是一次语音通道自检。", out, on_log=log)
        return {"ok": True, "bytes": info["bytes"], "path": out}
    except Exception as e:
        return {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
