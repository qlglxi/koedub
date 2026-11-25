# -*- coding: utf-8 -*-
"""清理运行产物：work/ 下的中间目录、上传缓存、过期日志。

    python tools/cleanup.py            # 列出将删除的内容
    python tools/cleanup.py --yes      # 真删
    python tools/cleanup.py --yes --keep-last 3   # 每个输入保留最近 3 次结果
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from core import config as cfgmod  # noqa: E402


def human(n: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return "%.1f %s" % (n, u)
        n /= 1024.0
    return "%.1f GB" % n


def dir_size(path: str) -> int:
    total = 0
    for dp, _d, fs in os.walk(path):
        for f in fs:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="确认删除")
    ap.add_argument("--keep-last", type=int, default=0,
                    help="每个输入文件名保留最近 N 个结果目录")
    args = ap.parse_args()

    cfgmod.ensure_dirs()
    targets: list[tuple[str, int]] = []

    # 1) work 下的任务目录（按输入名分组，可选保留最近 N 个）
    groups: dict[str, list[str]] = {}
    for name in os.listdir(cfgmod.WORK_DIR):
        p = os.path.join(cfgmod.WORK_DIR, name)
        if not os.path.isdir(p) or name in ("uploads", "cache"):
            continue
        stem = name.rsplit("-", 1)[0]
        groups.setdefault(stem, []).append(p)
    for stem, paths in groups.items():
        paths.sort(key=os.path.getmtime, reverse=True)
        for p in paths[args.keep_last:]:
            targets.append((p, dir_size(p)))

    # 2) 上传缓存
    up = os.path.join(cfgmod.WORK_DIR, "uploads")
    if os.path.isdir(up):
        for name in os.listdir(up):
            p = os.path.join(up, name)
            if os.path.isfile(p):
                targets.append((p, os.path.getsize(p)))

    # 3) 调试日志
    for name in ("server-debug.log",):
        p = os.path.join(cfgmod.WORK_DIR, name)
        if os.path.isfile(p):
            targets.append((p, os.path.getsize(p)))

    if not targets:
        print("没有需要清理的内容。")
        return 0

    total = sum(s for _p, s in targets)
    print("将删除 %d 项，共 %s：" % (len(targets), human(total)))
    for p, s in targets[:40]:
        print("  %-64s %10s" % (os.path.relpath(p, ROOT), human(s)))
    if len(targets) > 40:
        print("  ... 其余 %d 项" % (len(targets) - 40))

    if not args.yes:
        print("\n这是预演。加 --yes 才会真正删除。")
        return 0

    for p, _s in targets:
        try:
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                os.remove(p)
        except OSError as e:
            print("  删除失败 %s: %s" % (p, e))
    print("\n已清理，释放约 %s。" % human(total))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
