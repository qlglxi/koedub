# -*- coding: utf-8 -*-
"""零依赖 HTTP 服务端（标准库 ThreadingHTTPServer）。

实现要点：
  - 静态前端 + REST API 同一端口，无需 npm 构建；
  - 上传走流式 multipart 解析（大 MP3 不吃内存爆掉）；
  - 媒体文件支持 HTTP Range，浏览器 <audio> 才能拖动进度条；
  - 任务在后台线程跑，前端用 /api/job/events?since=N 拉增量进度。
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import socket
import sys
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import asr as asr_mod
from . import audiocore
from . import config as cfgmod
from . import deps
from . import edge_tts
from . import pipeline, providers, tts as tts_mod

WEB_DIR = os.path.join(cfgmod.BASE_DIR, "web")
UPLOAD_DIR = os.path.join(cfgmod.WORK_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 任务注册表
# ---------------------------------------------------------------------------
class Job:
    def __init__(self, job_id: str, kind: str, params: dict):
        self.id = job_id
        self.kind = kind
        self.params = params
        self.events: list[dict] = []
        self.status = "running"
        self.result: dict | None = None
        self.error: str | None = None
        self.created = time.time()
        self.finished: float | None = None
        self._lock = threading.Lock()
        self._cancel = False
        self.progress = 0.0

    def emit(self, stage: str, pct: float, message: str, data=None) -> None:
        with self._lock:
            if pct is not None and pct >= 0:
                self.progress = max(self.progress, float(pct))
            self.events.append({
                "seq": len(self.events),
                "t": round(time.time() - self.created, 2),
                "stage": stage,
                "progress": round(self.progress, 2),
                "message": message,
                "data": data,
            })
            if len(self.events) > 4000:
                self.events = self.events[-2000:]

    def events_since(self, since: int) -> dict:
        with self._lock:
            ev = [e for e in self.events if e["seq"] >= since]
            return {
                "id": self.id, "kind": self.kind, "status": self.status,
                "progress": round(self.progress, 2),
                "error": self.error, "events": ev,
                "nextSince": (ev[-1]["seq"] + 1) if ev else since,
                "finished": self.finished is not None,
            }

    def to_dict(self) -> dict:
        with self._lock:
            return {"id": self.id, "kind": self.kind, "status": self.status,
                    "progress": round(self.progress, 2), "error": self.error,
                    "created": self.created, "finished": self.finished,
                    "result": self.result,
                    "eventCount": len(self.events), "params": self.params}


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()
ALIAS: dict[str, str] = {}      # 文件名别名 -> 真实路径（供前端 <audio> 播放入参）


def _job_id() -> str:
    return time.strftime("%H%M%S") + "-" + os.urandom(3).hex()


def _spawn(job: Job, fn) -> None:
    with JOBS_LOCK:
        JOBS[job.id] = job

    def wrapper():
        try:
            job.result = fn(job)
            job.status = "done"
        except Exception as e:
            job.status = "error"
            job.error = "%s: %s" % (type(e).__name__, e)
            tb = traceback.format_exc()
            # 后台线程的异常看不到终端，落盘一份便于排查
            try:
                path = os.path.join(cfgmod.WORK_DIR, "job-error.log")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "a", encoding="utf-8") as f:
                    f.write("\n=== %s job %s %s ===\n%s\n"
                            % (time.strftime("%Y-%m-%d %H:%M:%S"), job.id,
                               job.kind, tb))
            except Exception:
                pass
            job.emit("error", -1, job.error, {"traceback": tb[-2000:]})
        finally:
            job.finished = time.time()

    threading.Thread(target=wrapper, daemon=True, name="job-" + job.id).start()


def _cleanup_jobs(keep_seconds: float = 3600 * 6) -> None:
    now = time.time()
    with JOBS_LOCK:
        for jid in [k for k, v in JOBS.items()
                    if v.finished and now - v.finished > keep_seconds]:
            JOBS.pop(jid, None)


# ---------------------------------------------------------------------------
# 上传
# ---------------------------------------------------------------------------
def _parse_content_disposition(value: str) -> dict:
    out = {}
    for m in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', value or ""):
        out[m.group(1).lower()] = m.group(2)
    return out


class _BoundedReader:
    """只允许读取固定字节数的包装器。

    HTTP/1.1 是长连接：请求体结束后连接并不会关闭，如果解析器读文件读到 EOF
    就会和客户端相互等待（浏览器表现为上传卡死）。所以这里严格按
    Content-Length 截断，读完立刻让上层结束。
    """

    def __init__(self, rf, limit: int):
        self._rf = rf
        self._left = max(0, int(limit))
        self._eof = False

    def read(self, n: int = -1) -> bytes:
        if self._eof or self._left <= 0:
            self._eof = True
            return b""
        want = self._left if n is None or n < 0 else min(n, self._left)
        try:
            data = self._rf.read(want)
        except Exception:
            self._eof = True
            return b""
        if not data:
            self._eof = True
            return b""
        self._left -= len(data)
        if self._left <= 0:
            self._eof = True
        return data


def _stream_multipart(rfile, boundary: bytes, total: int | None,
                      on_progress) -> dict:
    """流式解析 multipart/form-data（本项目用到的结构）。

    思路：把请求体切成「按 boundary 分隔的段」。文件字段的内容一边读一边落盘，
    只保留可能跨越块边界的尾部字节，因此大文件也不会吃爆内存。

    返回 {"fields": {...}, "files": {name: {"path","filename","size"}}}
    """
    delim = b"--" + boundary
    tail_keep = len(delim) + 8
    fields: dict[str, str] = {}
    files: dict[str, dict] = {}
    # 关键：按 Content-Length 界定请求体，绝不读到 EOF（长连接下会死锁）
    reader = _BoundedReader(rfile, total) if total else rfile

    state = "preamble"        # preamble -> headers -> body
    buf = b""
    delivered = 0
    cur: dict = {"name": "", "filename": "", "file": None, "path": "",
                 "size": 0, "is_file": False}

    def read_chunk() -> bytes:
        nonlocal delivered
        try:
            chunk = reader.read(1 << 16)
        except Exception:
            return b""
        if chunk:
            delivered += len(chunk)
            if on_progress:
                on_progress(delivered, total)
        return chunk

    finished = False
    while not finished:
        chunk = read_chunk()
        if not chunk:
            break
        buf += chunk

        while True:
            idx = buf.find(delim)
            if idx < 0:
                # 没找到 boundary：安全地把除尾部外的内容消化掉
                if state == "body":
                    if cur["is_file"] and cur["file"] is not None:
                        n = len(buf) - tail_keep
                        if n > 0:
                            cur["file"].write(buf[:n])
                            cur["size"] += n
                            buf = buf[n:]
                    # 普通字段：保留尾部即可
                    elif len(buf) > tail_keep:
                        fields[cur["name"]] = fields.get(cur["name"], "") + \
                            buf[:-tail_keep].decode("utf-8", "replace")
                        buf = buf[-tail_keep:]
                elif state == "headers":
                    # 头部还没读全，继续等
                    pass
                else:  # preamble
                    if len(buf) > tail_keep:
                        buf = buf[-tail_keep:]
                break

            # 找到 boundary：boundary 之前的内容属于当前部分
            body = buf[:idx]
            rest = buf[idx + len(delim):]

            if state == "body":
                if cur["is_file"] and cur["file"] is not None:
                    # 去掉紧邻 boundary 的 \r\n
                    data = body[:-2] if body.endswith(b"\r\n") else body
                    cur["file"].write(data)
                    cur["size"] += len(data)
                elif not cur["is_file"] and cur["name"]:
                    txt = body
                    if txt.endswith(b"\r\n"):
                        txt = txt[:-2]
                    fields[cur["name"]] = fields.get(cur["name"], "") + \
                        txt.decode("utf-8", "replace")

            # 关闭当前文件
            if cur["is_file"] and cur["file"] is not None:
                cur["file"].close()
                files[cur["name"]] = {"path": cur["path"],
                                      "filename": cur["filename"],
                                      "size": cur["size"]}
                cur = {"name": "", "filename": "", "file": None, "path": "",
                       "size": 0, "is_file": False}

            if rest.startswith(b"--"):
                finished = True
                buf = b""
                break

            if rest.startswith(b"\r\n"):
                rest = rest[2:]

            # 解析新部分的头部
            h_end = rest.find(b"\r\n\r\n")
            if h_end < 0:
                buf = rest
                state = "headers"
                break

            header_blob = rest[:h_end].decode("utf-8", "replace")
            buf = rest[h_end + 4:]
            state = "body"

            headers: dict[str, str] = {}
            for line in header_blob.split("\r\n"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    headers[k.strip().lower()] = v.strip()
            cd = _parse_content_disposition(headers.get("content-disposition", ""))
            name = cd.get("name", "")
            filename = cd.get("filename", "")
            if filename:
                ext = os.path.splitext(filename)[1] or ".bin"
                safe = re.sub(r"[^\w\-.]", "_", os.path.basename(filename))[:60]
                path = os.path.join(UPLOAD_DIR, "%s_%s%s"
                                    % (time.strftime("%H%M%S"),
                                       os.urandom(3).hex(), ext))
                cur = {"name": name, "filename": safe, "file": open(path, "wb"),
                       "path": path, "size": 0, "is_file": True}
            else:
                fields.setdefault(name, "")
                cur = {"name": name, "filename": "", "file": None, "path": "",
                       "size": 0, "is_file": False}
        if state == "headers":
            # 继续读直到头部完整
            continue

    if cur["is_file"] and cur["file"] is not None:
        cur["file"].close()
        files[cur["name"]] = {"path": cur["path"], "filename": cur["filename"],
                              "size": cur["size"]}
    return {"fields": fields, "files": files}


# ---------------------------------------------------------------------------
# 请求处理
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "koedub/1.0"
    protocol_version = "HTTP/1.1"

    # ---- 工具 ----
    def log_message(self, fmt, *args):
        if os.environ.get("KOEDUB_DEBUG_HTTP"):
            sys.stderr.write("[http] " + (fmt % args) + "\n")
            sys.stderr.flush()

    def log_error(self, fmt, *args):
        self.log_message(fmt, *args)

    def log_request(self, code="-", size="-"):
        if os.environ.get("KOEDUB_DEBUG_HTTP"):
            self.log_message('"%s" %s %s', self.requestline, str(code), str(size))

    def handle_expect_100(self) -> bool:
        """应答 Expect: 100-continue。

        urllib / curl / 部分浏览器在上传大文件前会先发这个头并等待应答；
        不应答的话客户端会一直干等，表现为上传「卡住」。
        """
        try:
            self.send_response_only(100)
            self.end_headers()
        except Exception:
            pass
        return True

    def _send(self, code: int, body: bytes, ctype: str = "application/json",
              extra: dict | None = None) -> None:
        if os.environ.get("KOEDUB_DEBUG_HTTP"):
            self._dbg_dump("_send %d %s bytes=%d" % (code, ctype, len(body)))
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"))

    def _err(self, msg: str, code: int = 400) -> None:
        self._json({"ok": False, "error": msg}, code)

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            return {}

    def _dbg_dump(self, text: str) -> None:
        """把诊断信息写到文件（后台运行时 stderr 常看不到）。"""
        try:
            path = os.path.join(cfgmod.WORK_DIR, "server-debug.log")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), text))
        except Exception:
            pass

    def _query(self) -> dict:
        q = urllib.parse.urlparse(self.path).query
        return {k: v[0] for k, v in urllib.parse.parse_qs(q).items()}

    # ---- 路由 ----
    def do_OPTIONS(self):
        self._send(204, b"", "text/plain",
                   {"Access-Control-Allow-Methods": "GET,POST,OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type"})

    def do_GET(self):
        try:
            self._route_get()
        except BrokenPipeError:
            pass
        except Exception as e:
            try:
                self._err("%s: %s" % (type(e).__name__, e), 500)
            except Exception:
                pass

    def do_POST(self):
        try:
            self._route_post()
        except BrokenPipeError:
            pass
        except Exception as e:
            traceback.print_exc()
            self._dbg_dump("do_POST 异常：%s: %s\n%s"
                           % (type(e).__name__, e, traceback.format_exc()[-1500:]))
            try:
                self._err("%s: %s" % (type(e).__name__, e), 500)
            except Exception:
                pass

    # ---- GET ----
    def _route_get(self):
        path = urllib.parse.urlparse(self.path).path
        q = self._query()

        if path in ("/", "/index.html"):
            return self._serve_static("index.html")
        if path.startswith("/static/"):
            return self._serve_static(path[len("/static/"):])
        if path == "/api/health":
            return self._json({
                "ok": True,
                "ffmpeg": audiocore.have_ffmpeg(),
                "ffmpegPath": (audiocore.ffmpeg_path() if audiocore.have_ffmpeg() else None),
                "python": os.sys.version.split()[0],
                "numpy": __import__("numpy").__version__,
            })
        if path == "/api/providers":
            return self._json({"ok": True, "registry": providers.summary(),
                               "asrAvailable": asr_mod.available_providers()})
        if path == "/api/deps":
            return self._json({"ok": True, "deps": deps.status(),
                               "models": asr_mod.models_on_disk(),
                               "hfEndpoint": os.environ.get("HF_ENDPOINT")})
        if path == "/api/diag/asr":
            return self._json(self._diag_asr())
        if path == "/api/diag":
            from . import runtime as rt
            try:
                return self._json({"ok": True, "diag": rt.diagnose(cfgmod.load_config())})
            except Exception as e:
                traceback.print_exc()
                return self._json({"ok": False,
                                   "error": "%s: %s" % (type(e).__name__, e)})
        if path == "/api/sovits/status":
            from . import sovits_proc as sp
            try:
                return self._json({"ok": True, "status": sp.status(cfgmod.load_config())})
            except Exception as e:
                traceback.print_exc()
                return self._json({"ok": False, "error": "%s: %s" % (type(e).__name__, e)})
        if path == "/api/sovits/sources":
            from . import sovits_dl as sdl
            return self._json({"ok": True, "sources": sdl.SOURCES,
                               "downloadDir": sdl.download_dir(),
                               "installRoot": sdl.install_root(),
                               "sevenZip": sdl.find_7z() or ""})
        if path == "/api/sovits/log":
            from . import sovits_proc as sp
            try:
                n = int((q.get("lines") or ["60"])[0])
            except (TypeError, ValueError):
                n = 60
            return self._json({"ok": True, "text": sp.tail_log(n),
                               "path": sp.log_path(), "pid": sp.status(
                                   cfgmod.load_config()).get("pid")})
        if path == "/api/config":
            cfg = cfgmod.load_config()
            return self._json({"ok": True, "config": cfgmod.public_config(cfg)})
        if path == "/api/jobs":
            with JOBS_LOCK:
                return self._json({"ok": True,
                                   "jobs": [j.to_dict() for j in
                                            sorted(JOBS.values(), key=lambda x: -x.created)][:50]})
        if path == "/api/job":
            return self._job_status(q)
        if path == "/api/job/events":
            return self._job_events(q)
        if path == "/api/tts/voices":
            try:
                zh = q.get("zh", "1") == "1"
                return self._json({"ok": True, "voices": tts_mod.list_edge_voices(zh_only=zh)})
            except Exception as e:
                return self._json({"ok": False, "error": "%s: %s" % (type(e).__name__, e)})
        if path == "/api/media":
            return self._serve_media(q)
        if path == "/api/download":
            return self._serve_download(q)
        if path == "/api/selftest":
            return self._json(self._selftest(q))
        if path == "/api/outputs":
            return self._json({"ok": True, "outputs": self._list_outputs()})
        return self._err("未知路径 %s" % path, 404)

    # ---- POST ----
    def _route_post(self):
        path = urllib.parse.urlparse(self.path).path
        if os.environ.get("KOEDUB_DEBUG_HTTP"):
            self._dbg_dump("POST %s CL=%s CT=%s"
                           % (path, self.headers.get("Content-Length"),
                              self.headers.get("Content-Type")))
        if path == "/api/config":
            body = self._read_json()
            cfg = cfgmod.load_config()
            for k, v in (body.get("config") or {}).items():
                if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v
            cfgmod.save_config(cfg)
            return self._json({"ok": True, "config": cfgmod.public_config(cfg)})
        if path == "/api/upload":
            return self._upload()
        if path == "/api/run":
            return self._start_job("dub")
        if path == "/api/analyze":
            return self._start_job("analyze")
        if path == "/api/deps/install":
            return self._install_dep()
        if path == "/api/sovits/start":
            from . import sovits_proc as sp
            logs: list[str] = []
            try:
                r = sp.start(cfgmod.load_config(), lambda m: logs.append(str(m)))
            except Exception as e:
                traceback.print_exc()
                r = {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
            r["logs"] = logs
            return self._json(r)
        if path == "/api/sovits/stop":
            from . import sovits_proc as sp
            try:
                r = sp.stop(cfgmod.load_config())
            except Exception as e:
                traceback.print_exc()
                r = {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
            return self._json(r)
        if path == "/api/sovits/selftest":
            from . import sovits_proc as sp
            logs: list[str] = []
            try:
                r = sp.selftest(cfgmod.load_config(), lambda m: logs.append(str(m)))
            except Exception as e:
                traceback.print_exc()
                r = {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
            r["logs"] = logs
            return self._json(r)
        if path == "/api/sovits/download":
            from . import sovits_dl as sdl
            body = self._read_json()
            sid = body.get("source") or ""
            try:
                mb = int(body.get("maxMB") or 0)
            except (TypeError, ValueError):
                mb = 0
            spec = sdl.pick_source(sid)
            job = Job(_job_id(), "sovits_dl",
                      {"source": spec["id"], "name": spec["name"], "maxMB": mb})

            def worker(j: Job) -> dict:
                cfg = cfgmod.load_config()
                last = [0.0]

                def prog(done, total, speed):
                    now = time.time()
                    if now - last[0] < 1.0:
                        return
                    last[0] = now
                    pct = (done * 100.0 / total) if total else 0.0
                    j.emit("download", min(90, pct),
                           "下载 %s / %s（%s/s）"
                           % (sdl._human(done), sdl._human(total),
                              sdl._human(int(speed))))

                res = sdl.install(cfg, sid,
                                  on_log=lambda m: j.emit("log", -1, str(m)),
                                  on_progress=prog,
                                  should_cancel=lambda: bool(j._cancel),
                                  max_bytes=(mb << 20) if mb else 0)
                if res.get("ok") and res.get("dir"):
                    c2 = cfgmod.load_config()
                    c2.setdefault("tts", {})["sovitsInstallDir"] = res["dir"]
                    cfgmod.save_config(c2)
                    j.emit("done", 100, "已就位并写回配置：%s" % res["dir"])
                elif res.get("canceled"):
                    j.emit("log", -1, "已取消；已下载的部分保留，下次接着下")
                return res

            _spawn(job, worker)
            _cleanup_jobs()
            return self._json({"ok": True, "jobId": job.id,
                               "source": spec["id"], "size": spec["size"]})
        if path == "/api/job/cancel":
            body = self._read_json()
            job = JOBS.get(body.get("id", ""))
            if not job:
                return self._err("任务不存在", 404)
            job._cancel = True
            job.emit("log", -1, "收到取消请求（当前步骤结束后停止）")
            return self._json({"ok": True})
        if path == "/api/tts/preview":
            body = self._read_json()
            cfg = cfgmod.load_config()
            cfg.update(body.get("config") or {})
            text = body.get("text") or "这是一次中文配音试听。"
            out = os.path.join(cfgmod.WORK_DIR, "preview_%s.mp3" % os.urandom(3).hex())
            try:
                info = tts_mod.SYNTHERS[tts_mod.pick_provider(cfg)](text, out, cfg,
                                                                  lambda m: None)
                alias = "preview_" + os.urandom(4).hex()
                ALIAS[alias] = out
                return self._json({"ok": True, "url": "/api/media?alias=%s" % alias,
                                   "bytes": info.get("bytes")})
            except Exception as e:
                return self._json({"ok": False,
                                   "error": "%s: %s" % (type(e).__name__, str(e)[:300])})
        return self._err("未知路径 %s" % path, 404)

    # ---- 静态与媒体 ----
    def _serve_static(self, rel: str):
        rel = rel.replace("\\", "/").lstrip("/")
        full = os.path.normpath(os.path.join(WEB_DIR, rel))
        if not full.startswith(os.path.normpath(WEB_DIR)) or not os.path.isfile(full):
            return self._err("静态文件不存在: %s" % rel, 404)
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript",):
            ctype += "; charset=utf-8"
        with open(full, "rb") as f:
            self._send(200, f.read(), ctype)

    def _serve_media(self, q: dict):
        p = q.get("alias")
        path = ALIAS.get(p) if p else q.get("path")
        if not path or not os.path.isfile(path):
            return self._err("媒体不存在", 404)
        return self._send_file(path)

    def _serve_download(self, q: dict):
        path = q.get("path")
        if not path or not os.path.isfile(path):
            return self._err("文件不存在", 404)
        name = os.path.basename(path)
        return self._send_file(path, download_name=name)

    def _send_file(self, path: str, download_name: str | None = None):
        size = os.path.getsize(path)
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        code = 200
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng.strip())
            if m:
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = int(m.group(2))
                end = min(end, size - 1)
                if start > end:
                    start = 0
                code = 206
        length = end - start + 1
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if code == 206:
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
        if download_name:
            self.send_header("Content-Disposition",
                             "attachment; filename*=UTF-8''%s"
                             % urllib.parse.quote(download_name))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(1 << 16, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)

    # ---- 可选依赖安装（作为后台任务，前端可看进度） ----
    def _install_dep(self):
        body = self._read_json()
        key = body.get("dep") or ""
        if key not in deps.OPTIONAL:
            return self._err("未知依赖：%s（可选：%s）"
                             % (key, "、".join(deps.OPTIONAL)))
        job = Job(_job_id(), "deps", {"dep": key})

        def worker(j: Job) -> dict:
            j.emit("deps", 3, "开始安装 %s…" % deps.OPTIONAL[key]["title"])
            res = deps.install(key, lambda m: j.emit("log", -1, m))
            j.emit("deps", 90, "安装%s。" % ("完成" if res.get("ok") else "失败"))
            # 安装完刷新模块搜索路径，后续任务即可用上
            deps.ensure_on_path()
            return {"dep": key, **res, "status": deps.status().get(key)}

        _spawn(job, worker)
        return self._json({"ok": True, "jobId": job.id})

    # ---- 诊断：服务端进程内的 ASR 导入状态 ----
    def _diag_asr(self) -> dict:
        import importlib
        import importlib.util
        info: dict = {
            "sysPathHead": sys.path[:6],
            "vendorOnPath": deps.vendor_dir() in sys.path,
            "inModules": "faster_whisper" in sys.modules,
        }
        spec = importlib.util.find_spec("faster_whisper")
        info["specOrigin"] = getattr(spec, "origin", None)
        info["specSearch"] = list(getattr(spec, "submodule_search_locations", []) or [])
        mod = sys.modules.get("faster_whisper")
        if mod is not None:
            info["moduleFile"] = getattr(mod, "__file__", None)
            info["modulePath"] = list(getattr(mod, "__path__", []) or [])
            info["hasWhisperModel"] = hasattr(mod, "WhisperModel")
        try:
            mod2 = importlib.import_module("faster_whisper")
            info["importOk"] = True
            info["importedFile"] = getattr(mod2, "__file__", None)
            info["importedHas"] = hasattr(mod2, "WhisperModel")
        except Exception as e:
            info["importOk"] = False
            info["importError"] = "%s: %s" % (type(e).__name__, e)
            info["importTraceback"] = traceback.format_exc()[-1200:]
        try:
            from faster_whisper import WhisperModel  # noqa: F401
            info["fromImportOk"] = True
        except Exception as e:
            info["fromImportOk"] = False
            info["fromImportError"] = "%s: %s" % (type(e).__name__, e)
        return {"ok": True, "diag": info}

    # ---- 上传 ----
    def _upload(self):
        dbg = os.environ.get("KOEDUB_DEBUG_HTTP")
        ctype = self.headers.get("Content-Type", "")
        m = re.search(r"boundary=([^;]+)", ctype)
        if not m:
            return self._err("缺少 multipart boundary")
        boundary = m.group(1).strip().strip('"').encode("ascii")
        total = int(self.headers.get("Content-Length") or 0)
        if dbg:
            sys.stderr.write("[upload] boundary=%r len=%d\n" % (boundary, total))
            sys.stderr.flush()
        last = [0.0]

        def on_progress(done, tot):
            now = time.time()
            if dbg and now - last[0] > 1.0:
                last[0] = now
                sys.stderr.write("[upload] recv %d / %s\n" % (done, tot))
                sys.stderr.flush()

        try:
            parsed = _stream_multipart(self.rfile, boundary, total, on_progress)
        except Exception as e:
            traceback.print_exc()
            self._dbg_dump("multipart failed: %s: %s\n%s"
                           % (type(e).__name__, e, traceback.format_exc()))
            return self._err("解析上传内容失败：%s: %s" % (type(e).__name__, e))
        if not parsed["files"] and not parsed["fields"]:
            self._dbg_dump("multipart produced nothing (len=%d, boundary=%r)"
                           % (total, boundary))
        if dbg:
            sys.stderr.write("[upload] parsed fields=%s files=%s\n"
                             % (parsed["fields"], {k: v["size"] for k, v in parsed["files"].items()}))
            sys.stderr.flush()
        f = (parsed["files"] or {}).get("file")
        if not f:
            return self._err("没有收到文件字段 file")
        info = None
        try:
            info = audiocore.probe(f["path"])
        except Exception as e:
            return self._json({"ok": False,
                               "error": "已接收 %s，但不是可解析的音频：%s"
                                        % (f["filename"], str(e)[:200])})
        alias = "src_" + os.urandom(4).hex()
        ALIAS[alias] = f["path"]
        return self._json({"ok": True, "file": {
            "alias": alias, "path": f["path"], "filename": f["filename"],
            "size": f["size"], "url": "/api/media?alias=%s" % alias,
            "media": info.to_dict(),
        }})

    # ---- 任务 ----
    def _job_status(self, q: dict):
        job = JOBS.get(q.get("id", ""))
        if not job:
            return self._err("任务不存在", 404)
        return self._json({"ok": True, "job": job.to_dict()})

    def _job_events(self, q: dict):
        job = JOBS.get(q.get("id", ""))
        if not job:
            return self._err("任务不存在", 404)
        since = int(q.get("since", "0") or 0)
        return self._json({"ok": True, **job.events_since(since)})

    def _start_job(self, kind: str):
        body = self._read_json()
        cfg = cfgmod.load_config()
        cfgmod.ensure_dirs()
        user_cfg = body.get("config") or {}
        for k, v in user_cfg.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
        alias = body.get("alias")
        path = ALIAS.get(alias) if alias else None
        if body.get("inlineConfig"):
            cfg.update(body["inlineConfig"])
        if kind == "analyze":
            return self._analyze(path, cfg, body)
        if not path:
            return self._err("缺少 alias（请先上传文件）")
        if not os.path.isfile(path):
            return self._err("源文件已失效，请重新上传")

        job = Job(_job_id(), kind, {"alias": alias,
                                   "file": os.path.basename(path)})

        def worker(j: Job) -> dict:
            j.emit("start", 0, "开始处理：%s" % os.path.basename(path))
            res = pipeline.run_safe(
                path, cfg, lambda st, pct, msg, data: j.emit(st, pct, msg, data),
                manual_text=body.get("manualText") or "",
                manual_segments=body.get("manualSegments") or None)
            slim = {k: v for k, v in res.items() if k not in ("logs", "plans")}
            slim["plans"] = res.get("plans")
            slim["providerSummary"] = providers.summary()["capabilityTotal"]
            return slim

        _spawn(job, worker)
        _cleanup_jobs()
        return self._json({"ok": True, "jobId": job.id})

    def _analyze(self, path: str | None, cfg: dict, body: dict) -> None:
        """只做声道判定 + 分段（很快），让用户先确认左右耳结论。"""
        if not path or not os.path.isfile(path):
            return self._err("缺少 alias（请先上传文件）")

        job = Job(_job_id(), "analyze", {"file": os.path.basename(path)})

        def worker(j: Job) -> dict:
            from . import channels, segmentation
            j.emit("probe", 5, "探测媒体信息")
            info = audiocore.probe(path)
            cfg["_duration"] = info.duration
            j.emit("channel", 20, "解码并判定声道")
            pcm = audiocore.decode_pcm(path, 16000, 2)
            if pcm.ndim == 1:
                pcm = pcm[:, None]
            side = channels.analyze(pcm, 16000)
            j.emit("channel", 60, "人声判定：%s" % side.side, side.to_dict())
            regions = segmentation.detect_regions(pcm, 16000)
            j.emit("segment", 80, "VAD 分段：%d 段" % len(regions))
            alias = "src_" + os.urandom(4).hex()
            ALIAS[alias] = path
            return {"media": info.to_dict(), "channel": side.to_dict(),
                    "regions": regions, "regionCount": len(regions),
                    "url": "/api/media?alias=%s" % alias}

        _spawn(job, worker)
        return self._json({"ok": True, "jobId": job.id})

    # ---- 自检 ----
    def _selftest(self, q: dict) -> dict:
        which = q.get("what", "all")
        out: dict = {"ok": True}
        if which in ("all", "ffmpeg"):
            ok = audiocore.have_ffmpeg()
            out["ffmpeg"] = {"ok": ok,
                            "path": audiocore.ffmpeg_path() if ok else None}
        if which in ("all", "tts", "edge"):
            cfg = cfgmod.load_config()
            out["tts"] = tts_mod.self_test(cfg, lambda m: None)
        if which in ("all", "asr"):
            out["asr"] = {"providers": asr_mod.available_providers(),
                          "picked": asr_mod.pick_provider(cfgmod.load_config())}
        if which in ("all", "providers"):
            out["providers"] = providers.summary()
        if which in ("all", "edge-voices"):
            try:
                out["edgeVoices"] = len(edge_tts.list_voices())
            except Exception as e:
                out["edgeVoices"] = "%s: %s" % (type(e).__name__, str(e)[:160])
        return out

    def _list_outputs(self) -> list[dict]:
        items: list[dict] = []
        if not os.path.isdir(cfgmod.WORK_DIR):
            return items

        def scan(root):
            for dirpath, _dirs, files in os.walk(root):
                for fn in files:
                    if fn == "manifest.json":
                        continue
                    if os.path.splitext(fn)[1].lower() not in (
                            ".mp3", ".wav", ".mp4", ".m4a", ".srt", ".ass", ".json"):
                        continue
                    full = os.path.join(dirpath, fn)
                    st = os.stat(full)
                    items.append({"name": fn, "path": full,
                                  "dir": os.path.relpath(dirpath, cfgmod.WORK_DIR),
                                  "size": st.st_size, "mtime": st.st_mtime,
                                  "url": "/api/download?path=%s"
                                         % urllib.parse.quote(full)})
        scan(cfgmod.WORK_DIR)
        scan(cfgmod.OUT_DIR)
        items.sort(key=lambda x: -x["mtime"])
        return items[:200]


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------
def _free_port(host: str, port: int) -> int:
    for p in range(port, port + 40):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    raise RuntimeError("找不到可用端口")


def serve(host: str = "127.0.0.1", port: int = 8722, open_browser: bool = True,
          app_window: bool = False) -> None:
    cfgmod.ensure_dirs()
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    # 启动即确定模型下载端点，界面和识别流程都用同一个
    asr_mod.setup_hf_mirror()
    port = _free_port(host, port)
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    url = "http://%s:%d/" % (host, port)
    lines = [
        "=" * 64,
        "  KoeDub 音声双语化工作台",
        "  已启动：%s" % url,
        "  ffmpeg：%s" % (audiocore.ffmpeg_path() if audiocore.have_ffmpeg() else "未找到！"),
        "  识别后端：%s" % asr_mod.pick_provider(cfgmod.load_config()),
        "  按 Ctrl+C 停止",
        "=" * 64,
    ]
    for ln in lines:
        print(ln, flush=True)
    if open_browser:
        try:
            from .launcher import open_ui

            def _open():
                try:
                    how = open_ui(url, port, prefer_window=app_window,
                                  log=lambda m: print("  窗口：" + m, flush=True))
                    if how != "manual":
                        print("  界面已打开（方式：%s）" % how, flush=True)
                except Exception as e:
                    print("  自动打开界面失败（%s），请手动访问上面的地址" % e,
                          flush=True)

            threading.Timer(0.8, _open).start()
        except Exception as e:
            print("  启动器不可用（%s），请手动访问上面的地址" % e, flush=True)
    try:
        httpd.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\n正在停止…", flush=True)
    finally:
        httpd.server_close()
