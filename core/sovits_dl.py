# -*- coding: utf-8 -*-
"""GPT-SoVITS 整合包的下载 / 解压 / 落位。

整合包都是单文件 7z（6~10GB），放在 ModelScope 上。这里只做三件事：
能续传的下载、7z/zip 解压、解压完自动找到含 api_v2.py 的那层目录并写回配置。

不碰 pip：GPT-SoVITS 的依赖是整合包自带的 runtime，装它 == 解压它。
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import time
import urllib.request
import zipfile

from . import config as cfgmod

CHUNK = 1 << 20

# 内置可选项。size / sha256 取自 ModelScope 仓库文件列表（2026-11 实查）。
SOURCES = [
    {
        "id": "v2pro",
        "name": "GPT-SoVITS-v2pro-20250531.7z",
        "size": 8147076148,
        "sha256": "4b474a7b2ddccd289e7a673a55d8e76145c1e48ed333bb913115ddabdd8d5ce1",
        "label": "v2pro（推荐：本项目的调教权重就是这一代）",
    },
    {
        "id": "v4",
        "name": "GPT-SoVITS-v4-20250529.7z",
        "size": 7493794100,
        "sha256": "6461419c7c81f7c23a08e467c040eb25197bf8291ec1107e0104c46354b77146",
        "label": "v4（更新，权重兼容层更全）",
    },
    {
        "id": "v5",
        "name": "GPT-SoVITS-v5-20261005.7z",
        "size": 10869111823,
        "sha256": "2c1cd034af0e76854a8023d0f1fd6156ef2414bcfd5c88aae3309e1263e0fffe",
        "label": "v5（最新，10.1GB）",
    },
]
_MIRROR = "https://www.modelscope.cn/models/FlowerCry/gpt-sovits-7z-pacakges/resolve/master/"


def pick_source(sid: str = "") -> dict:
    sid = (sid or "").strip().lower()
    for s in SOURCES:
        if s["id"] == sid or s["name"].lower() == sid:
            return dict(s)
    return dict(SOURCES[0])


def download_dir() -> str:
    d = os.path.join(cfgmod.BASE_DIR, "runtime", "downloads")
    os.makedirs(d, exist_ok=True)
    return d


def install_root() -> str:
    d = os.path.join(cfgmod.BASE_DIR, "runtime")
    os.makedirs(d, exist_ok=True)
    return d


def _open(url: str, start: int = 0, timeout: float = 60):
    req = urllib.request.Request(url, headers={
        "User-Agent": "KoeDub/1.0 (+local)",
        "Accept": "*/*",
    })
    if start:
        req.add_header("Range", "bytes=%d-" % start)
    return urllib.request.urlopen(req, timeout=timeout)


def probe_remote(url: str) -> dict:
    """不下内容，只问一句「这地址还有效吗、多大」。"""
    try:
        r = _open(url, 0, timeout=25)
        cl = r.headers.get("Content-Length")
        r.close()
        return {"ok": True, "size": int(cl) if cl else 0, "url": url}
    except Exception as e:
        return {"ok": False, "error": "%s: %s" % (type(e).__name__, e), "url": url}


def download(url: str, dst: str, on_progress=None, should_cancel=None,
             expected_size: int = 0, max_bytes: int = 0) -> dict:
    """带续传的下载。中断后再点一次会从 .part 的断点继续。"""
    part = dst + ".part"
    start = 0
    if os.path.isfile(part):
        start = os.path.getsize(part)
        if expected_size and start >= expected_size:
            os.replace(part, dst)
            return {"ok": True, "bytes": expected_size, "already": True, "resumed": True}
        if expected_size and start > expected_size:
            os.remove(part)
            start = 0

    try:
        resp = _open(url, start)
    except Exception as e:
        return {"ok": False, "error": "连接失败：%s: %s" % (type(e).__name__, e)}

    total = expected_size
    if not total:
        cl = resp.headers.get("Content-Length")
        if cl:
            total = int(cl) + start

    done = start
    t0 = time.time()
    last = 0.0
    mode = "ab" if start else "wb"
    try:
        with open(part, mode) as f:
            while True:
                if should_cancel and should_cancel():
                    return {"ok": False, "canceled": True, "bytes": done,
                            "error": "已取消（下次会从断点继续）"}
                buf = resp.read(CHUNK)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if max_bytes and done >= max_bytes:
                    return {"ok": False, "partial": True, "bytes": done,
                            "error": "达到测试上限 %s" % _human(max_bytes)}
                now = time.time()
                if on_progress and now - last >= 0.5:
                    last = now
                    speed = (done - start) / max(0.001, now - t0)
                    on_progress(done, total, speed)
    except Exception as e:
        return {"ok": False, "error": "下载中断：%s: %s" % (type(e).__name__, e), "bytes": done}
    finally:
        try:
            resp.close()
        except Exception:
            pass

    if total and done < total:
        return {"ok": False, "error": "只拿到 %s / %s" % (_human(done), _human(total)), "bytes": done}
    os.replace(part, dst)
    return {"ok": True, "bytes": done, "path": dst, "seconds": round(time.time() - t0, 1)}


def verify_sha256(path: str, want: str, on_progress=None, should_cancel=None) -> dict:
    if not want:
        return {"ok": True, "skipped": True}
    h = hashlib.sha256()
    size = os.path.getsize(path)
    done = 0
    last = 0.0
    with open(path, "rb") as f:
        while True:
            if should_cancel and should_cancel():
                return {"ok": False, "error": "校验被取消"}
            b = f.read(8 << 20)
            if not b:
                break
            h.update(b)
            done += len(b)
            now = time.time()
            if on_progress and now - last >= 0.7:
                last = now
                on_progress(done, size)
    got = h.hexdigest()
    return {"ok": got.lower() == want.lower(), "sha256": got, "want": want}


def find_7z() -> str | None:
    cands = [shutil.which("7z"), shutil.which("7za"), shutil.which("7zr"),
             r"C:\Program Files\7-Zip\7z.exe",
             r"C:\Program Files (x86)\7-Zip\7z.exe"]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


def extract(archive: str, dst_dir: str, on_log=None) -> dict:
    """解压。zip 用内置，7z 用系统 7-Zip。"""
    def say(m):
        if on_log:
            try:
                on_log(m)
            except Exception:
                pass

    os.makedirs(dst_dir, exist_ok=True)
    low = archive.lower()
    if low.endswith(".zip"):
        say("解压 zip → %s" % dst_dir)
        try:
            with zipfile.ZipFile(archive) as z:
                z.extractall(dst_dir)
            return {"ok": True, "how": "zipfile", "dir": dst_dir}
        except Exception as e:
            return {"ok": False, "error": "zip 解压失败：%s: %s" % (type(e).__name__, e)}

    if low.endswith(".7z"):
        exe = find_7z()
        if not exe:
            return {"ok": False, "need": "7z",
                    "error": "解压 7z 需要 7-Zip。装一下：winget install 7zip.7zip "
                             "（装完点「重新检查」，不用重下）"}
        say("解压 7z（%s，大包要几分钟）…" % os.path.basename(archive))
        try:
            cp = subprocess.run([exe, "x", "-y", "-bsp1", "-o" + dst_dir, archive],
                                capture_output=True, timeout=7200)
        except (OSError, subprocess.SubprocessError) as e:
            return {"ok": False, "error": "调 7-Zip 失败：%s" % e}
        if cp.returncode != 0:
            tail = (cp.stderr or cp.stdout or b"").decode("utf-8", "replace")[-400:]
            return {"ok": False, "error": "7-Zip 返回 %d：%s" % (cp.returncode, tail)}
        return {"ok": True, "how": exe, "dir": dst_dir}

    return {"ok": False, "error": "不认识的压缩格式：%s" % os.path.basename(archive)}


def locate_api_dir(root: str, max_depth: int = 4) -> str | None:
    """在解压结果里找含 api_v2.py 的那一层。整合包通常多套一层目录。"""
    root = os.path.abspath(root)
    if os.path.isfile(os.path.join(root, "api_v2.py")):
        return root
    best = None
    base_depth = root.rstrip("\\/").count(os.sep)
    for dp, dn, fn in os.walk(root):
        if dp.rstrip("\\/").count(os.sep) - base_depth > max_depth:
            dn[:] = []
            continue
        if "api_v2.py" in fn:
            if best is None or len(dp) < len(best):
                best = dp
            dn[:] = []
    return best


def install(cfg: dict, sid: str = "", on_log=None, on_progress=None,
            should_cancel=None, max_bytes: int = 0, url: str = "") -> dict:
    """完整装一遍：下载 → 校验大小 → 解压 → 找到目录 → 写回配置。"""
    def say(m):
        if on_log:
            try:
                on_log(m)
            except Exception:
                pass

    spec = pick_source(sid)
    src_url = url or (_MIRROR + spec["name"])
    dst = os.path.join(download_dir(), spec["name"])

    if os.path.isfile(dst):
        ok_size = (not spec.get("size")) or os.path.getsize(dst) == spec["size"]
        if ok_size:
            say("已经下载过：%s（%s）" % (spec["name"], _human(os.path.getsize(dst))))
        else:
            say("已有文件大小不符，重新下载")
            try:
                os.remove(dst)
            except OSError:
                pass

    if not os.path.isfile(dst):
        say("开始下载 %s（%s）→ %s" % (spec["name"], _human(spec.get("size") or 0), dst))
        r = download(src_url, dst, on_progress=on_progress,
                     should_cancel=should_cancel,
                     expected_size=int(spec.get("size") or 0), max_bytes=max_bytes)
        if not r.get("ok"):
            return {"ok": False, "stage": "download", **{k: v for k, v in r.items() if k != "ok"}}
        if r.get("partial"):
            return {"ok": False, "stage": "download", "partial": True,
                    "error": r.get("error"), "path": dst}
        say("下载完成：%s，用时 %ss" % (_human(r["bytes"]), r.get("seconds")))

    tmp = os.path.join(install_root(), "_extract")
    if os.path.isdir(tmp):
        shutil.rmtree(tmp, ignore_errors=True)
    r = extract(dst, tmp, on_log=say)
    if not r.get("ok"):
        return {"ok": False, "stage": "extract", **{k: v for k, v in r.items() if k != "ok"}}

    found = locate_api_dir(tmp)
    if not found:
        return {"ok": False, "stage": "locate",
                "error": "解压完了但没找到 api_v2.py，包里可能只有运行库（需要配 update 包）"}

    final = os.path.join(install_root(), os.path.basename(found) or "GPT-SoVITS")
    if os.path.isdir(final):
        final = final + "-" + time.strftime("%m%d%H%M")
    try:
        shutil.move(found, final)
    except (OSError, shutil.Error) as e:
        return {"ok": False, "stage": "move", "error": "落位失败：%s" % e, "found": found}

    shutil.rmtree(tmp, ignore_errors=True)
    say("已就位：%s" % final)
    return {"ok": True, "stage": "done", "dir": final, "archive": dst}


def _human(n: int) -> str:
    if not n or n < 0:
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return ("%.1f %s" % (n, unit)) if unit != "B" else ("%d B" % n)
        n /= 1024.0
    return "-"
