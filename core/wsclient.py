# -*- coding: utf-8 -*-
"""零依赖 WebSocket 客户端（RFC 6455 最小实现）。

只实现 Edgware/Edge-TTS 场景需要的部分：
  - wss:// 握手（含自定义头）
  - 客户端发送 text/binary 帧（含掩码）
  - 接收 text/binary 帧、ping/pong、close
不依赖任何第三方包，仅用标准库 socket + ssl + struct + os.urandom。
"""
from __future__ import annotations

import base64
import os
import socket
import ssl
import struct
from urllib.parse import urlsplit

__all__ = ["WebSocketError", "WebSocketClient", "connect"]

_OP_CONT = 0x0
_OP_TEXT = 0x1
_OP_BINARY = 0x2
_OP_CLOSE = 0x8
_OP_PING = 0x9
_OP_PONG = 0xA


class WebSocketError(RuntimeError):
    pass


class WebSocketClient:
    """极简 WebSocket 客户端。

    recv() 返回 (opcode, payload)；连接关闭时抛 WebSocketError。
    """

    def __init__(self, url: str, headers: dict[str, str] | None = None,
                 timeout: float = 20.0):
        self.url = url
        self._timeout = timeout
        self._buf = b""
        self._closed = False
        parts = urlsplit(url)
        if parts.scheme not in ("ws", "wss"):
            raise WebSocketError("不支持的 scheme: %s" % parts.scheme)
        secure = parts.scheme == "wss"
        host = parts.hostname or ""
        port = parts.port or (443 if secure else 80)
        path = parts.path or "/"
        if parts.query:
            path += "?" + parts.query

        raw = socket.create_connection((host, port), timeout=timeout)
        if secure:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            raw = ctx.wrap_socket(raw, server_hostname=host)
        raw.settimeout(timeout)
        self._sock = raw

        key = base64.b64encode(os.urandom(16)).decode("ascii")
        host_header = host if port in (80, 443) else "%s:%d" % (host, port)
        lines = [
            "GET %s HTTP/1.1" % path,
            "Host: %s" % host_header,
            "Upgrade: websocket",
            "Connection: Upgrade",
            "Sec-WebSocket-Key: %s" % key,
            "Sec-WebSocket-Version: 13",
        ]
        for k, v in (headers or {}).items():
            lines.append("%s: %s" % (k, v))
        raw.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("utf-8"))

        # 读握手响应
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = raw.recv(4096)
            if not chunk:
                raise WebSocketError("握手期间连接被关闭")
            head += chunk
        header_blob, _, rest = head.partition(b"\r\n\r\n")
        self._buf = rest
        status_line = header_blob.split(b"\r\n", 1)[0].decode("latin-1")
        if "101" not in status_line:
            body = header_blob.decode("latin-1", "replace")
            raise WebSocketError("握手失败: %s | %s" % (status_line, body[-400:]))

    # ---------------- 发送 ----------------
    def _send_frame(self, opcode: int, payload: bytes = b"") -> None:
        if self._closed:
            raise WebSocketError("连接已关闭")
        header = bytearray()
        header.append(0x80 | opcode)
        mask = os.urandom(4)
        n = len(payload)
        if n < 126:
            header.append(0x80 | n)
        elif n < (1 << 16):
            header.append(0x80 | 126)
            header += struct.pack(">H", n)
        else:
            header.append(0x80 | 127)
            header += struct.pack(">Q", n)
        header += mask
        masked = bytes(b ^ mask[i & 3] for i, b in enumerate(payload))
        self._sock.sendall(bytes(header) + masked)

    def send_text(self, text: str) -> None:
        self._send_frame(_OP_TEXT, text.encode("utf-8"))

    def ping(self, data: bytes = b"") -> None:
        self._send_frame(_OP_PING, data)

    # ---------------- 接收 ----------------
    def _read_exact(self, n: int) -> bytes:
        while len(self._buf) < n:
            try:
                chunk = self._sock.recv(65536)
            except socket.timeout:
                raise WebSocketError("读取超时")
            if not chunk:
                raise WebSocketError("连接被对端关闭")
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def recv(self) -> tuple[int, bytes]:
        """返回 (opcode, payload)。ping 自动回 pong 并继续读。"""
        while True:
            b0, b1 = self._read_exact(2)
            fin = bool(b0 & 0x80)
            opcode = b0 & 0x0F
            masked = bool(b1 & 0x80)
            length = b1 & 0x7F
            if length == 126:
                length = struct.unpack(">H", self._read_exact(2))[0]
            elif length == 127:
                length = struct.unpack(">Q", self._read_exact(8))[0]
            mask = self._read_exact(4) if masked else None
            payload = self._read_exact(length) if length else b""
            if mask:
                payload = bytes(b ^ mask[i & 3] for i, b in enumerate(payload))

            if opcode == _OP_PING:
                self._send_frame(_OP_PONG, payload)
                continue
            if opcode == _OP_PONG:
                continue
            if opcode == _OP_CLOSE:
                self.close()
                raise WebSocketError("对端发送了关闭帧")
            if not fin:
                # 拼接收续帧（实际 Edge 每次都是单帧，这里保守处理）
                frames = [payload]
                while True:
                    c0, c1 = self._read_exact(2)
                    clen = c1 & 0x7F
                    if clen == 126:
                        clen = struct.unpack(">H", self._read_exact(2))[0]
                    elif clen == 127:
                        clen = struct.unpack(">Q", self._read_exact(8))[0]
                    cm = self._read_exact(4) if (c1 & 0x80) else None
                    cp = self._read_exact(clen) if clen else b""
                    if cm:
                        cp = bytes(b ^ cm[i & 3] for i, b in enumerate(cp))
                    frames.append(cp)
                    if c0 & 0x80:
                        break
                payload = b"".join(frames)
            return opcode, payload

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._send_frame(_OP_CLOSE, struct.pack(">H", 1000))
        except Exception:
            pass
        try:
            self._sock.close()
        except Exception:
            pass

    def __enter__(self) -> "WebSocketClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def connect(url: str, headers: dict[str, str] | None = None,
            timeout: float = 20.0) -> WebSocketClient:
    return WebSocketClient(url, headers=headers, timeout=timeout)
