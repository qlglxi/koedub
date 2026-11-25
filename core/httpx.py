# -*- coding: utf-8 -*-
"""HTTP / multipart 工具，纯标准库。"""
from __future__ import annotations

import json
import mimetypes
import os
import ssl
import urllib.error
import urllib.request
import uuid

DEFAULT_TIMEOUT = 180
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


class HttpError(RuntimeError):
    def __init__(self, status: int, body: str, url: str = ""):
        super().__init__("HTTP %s %s: %s" % (status, url, body[:300]))
        self.status = status
        self.body = body
        self.url = url


def request(url: str, method: str = "GET", data: bytes | None = None,
            headers: dict[str, str] | None = None, timeout: int = DEFAULT_TIMEOUT,
            json_body=None) -> bytes:
    h = dict(headers or {})
    if json_body is not None:
        data = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
        h.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise HttpError(e.code, e.read().decode("utf-8", "replace"), url) from None
    except Exception as e:
        raise RuntimeError("请求失败 %s: %s: %s" % (url, type(e).__name__, e)) from None


def request_json(url: str, method: str = "GET", data: bytes | None = None,
                 headers: dict[str, str] | None = None, timeout: int = DEFAULT_TIMEOUT,
                 json_body=None) -> dict:
    raw = request(url, method=method, data=data, headers=headers, timeout=timeout,
                  json_body=json_body)
    if not raw:
        return {}
    try:
        return json.loads(raw.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return {"_raw": raw.decode("utf-8", "replace")[:2000]}


def post_json_stream(url: str, json_body: dict, headers: dict[str, str] | None = None,
                     timeout: int = DEFAULT_TIMEOUT) -> bytes:
    h = dict(headers or {})
    h.setdefault("Content-Type", "application/json")
    body = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise HttpError(e.code, e.read().decode("utf-8", "replace"), url) from None
    except Exception as e:
        raise RuntimeError("请求失败 %s: %s: %s" % (url, type(e).__name__, e)) from None


def multipart(fields: dict[str, str], files: dict[str, tuple[str, bytes, str]]) -> tuple[bytes, str]:
    """构造 multipart/form-data 体。files: name -> (filename, bytes, mime)"""
    boundary = "----koedub" + uuid.uuid4().hex
    out = bytearray()
    for k, v in (fields or {}).items():
        out += ("--%s\r\n" % boundary).encode()
        out += ('Content-Disposition: form-data; name="%s"\r\n\r\n' % k).encode()
        out += (str(v) + "\r\n").encode("utf-8")
    for name, (fname, blob, mime) in (files or {}).items():
        if mime is None:
            mime = mimetypes.guess_type(fname)[0] or "application/octet-stream"
        out += ("--%s\r\n" % boundary).encode()
        out += ('Content-Disposition: form-data; name="%s"; filename="%s"\r\n'
                % (name, os.path.basename(fname))).encode()
        out += ("Content-Type: %s\r\n\r\n" % mime).encode()
        out += blob + b"\r\n"
    out += ("--%s--\r\n" % boundary).encode()
    return bytes(out), "multipart/form-data; boundary=%s" % boundary
