# -*- coding: utf-8 -*-
"""可选依赖的检测与安装。

本软件的核心链路零第三方依赖（只用标准库 + ffmpeg）。下面这些是「可选增强」，
装不上也能跑，只是质量或速度不同：

  faster-whisper  本地语音识别（离线、免费、不传音频出门）
  demucs          人声分离（把台词与音效彻底分开）

安装时统一做三件事，避免踩国内网络与权限的坑：
  1. 自动附加 PyPI 镜像（默认清华源，可用 KOEDUB_PIP_INDEX 覆盖）；
  2. 把 pip 的临时目录与缓存放到本工程目录下（系统临时目录常因权限失败）；
  3. 默认装到工程内的 vendor/ 目录，不动系统环境，随时可删。
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import time

from . import config as cfgmod

# 国内镜像优先（快），但镜像可能缺包 —— 实测清华源就没有 faster-whisper。
# 因此安装失败会自动回退到官方 PyPI。两者都可用 KOEDUB_PIP_INDEX 覆盖。
DEFAULT_INDEX = os.environ.get("KOEDUB_PIP_INDEX",
                               "https://pypi.tuna.tsinghua.edu.cn/simple")
FALLBACK_INDEX = os.environ.get("KOEDUB_PIP_FALLBACK",
                                "https://pypi.org/simple")

# 可选依赖清单：import 名 -> 说明 / pip 包名 / 体积提示
OPTIONAL = {
    "faster_whisper": {
        "pip": "faster-whisper",
        "title": "本地语音识别 faster-whisper",
        "desc": "离线识别日语，不把音频传到外部；首次使用会下载模型（small 约 460MB）。",
        "size": "约 120MB（含 ctranslate2）+ 模型",
        "provides": ["asr:whisper_local"],
        "verifyImport": "faster_whisper",
    },
    "demucs": {
        "pip": "demucs",
        "title": "人声分离 Demucs",
        "desc": "把原声拆成「人声」与「伴奏/音效」，实现更干净的左右耳分工。",
        "size": "约 300MB（含 torch 依赖可能更大）",
        "provides": ["separate:demucs"],
        "verifyImport": "demucs",
    },
}


def vendor_dir() -> str:
    return os.path.join(cfgmod.BASE_DIR, "vendor")


# ---------------------------------------------------------------------------
# 遮蔽保护
# ---------------------------------------------------------------------------
# 用 pip --target 装到 vendor/ 时，传递依赖会一并落进来（比如 faster-whisper 会
# 带上 numpy）。vendor/ 又被插到 sys.path 最前面，于是这些同名包会**遮蔽宿主
# 环境自带的版本**。实测这会让音频数值计算出现诡异越界（IndexError），非常难查。
# 因此：凡是「运行时环境里已经有」的包，在 vendor 里一律改名停用。
_SHADOW_SUFFIX = ".shadowed-by-koedub"
_shadow_checked = False


def _runtime_packages() -> set[str]:
    """运行时 python 自身 site-packages 里已有的顶层包名。"""
    names: set[str] = set()
    for p in sys.path:
        if not p or "site-packages" not in p.replace("\\", "/"):
            continue
        if os.path.normcase(os.path.abspath(p)) == \
                os.path.normcase(os.path.abspath(vendor_dir())):
            continue
        try:
            for n in os.listdir(p):
                if n.endswith((".dist-info", ".libs", "__pycache__")):
                    continue
                if os.path.isdir(os.path.join(p, n)) or n.endswith(".py"):
                    names.add(n[:-3] if n.endswith(".py") else n)
        except OSError:
            continue
    return names


def enforce_no_shadowing(on_log=None) -> list[str]:
    """把 vendor 里与运行时重名的包改名，返回被停用的包名列表。

    幂等：每个进程只真正扫描一次。这一点很重要 —— 目录改名与其它进程正在进行的
    导入会互相干扰（表现为子进程无任何输出直接退出），所以绝不能在每次导入时重扫。
    """
    global _shadow_checked
    if _shadow_checked:
        return []
    _shadow_checked = True
    # 只处理会真正被加载的那一层：老式散装目录 + 当前解释器的版本目录。
    # 其它版本的目录（vendor/cp312 之类）不参与导入，不用动。
    roots = [vendor_dir(), install_dir()]
    runtime = _runtime_packages()
    disabled: list[str] = []
    for v in roots:
        if not os.path.isdir(v):
            continue
        for name in os.listdir(v):
            if name.endswith(_SHADOW_SUFFIX) or name.endswith((".dist-info", ".libs",
                                                               "__pycache__")):
                continue
            if name.startswith("cp") and os.path.isdir(os.path.join(v, name)):
                continue          # 版本子目录，跳过
            full = os.path.join(v, name)
            if not (os.path.isdir(full) or name.endswith(".py")):
                continue
            import_name = name[:-3] if name.endswith(".py") else name
            if import_name not in runtime:
                continue
            try:
                os.rename(full, full + _SHADOW_SUFFIX)
                disabled.append(import_name)
            except OSError:
                continue
    if disabled and on_log:
        on_log("已在 vendor 中停用会遮蔽运行时的包：%s"
               % "、".join(sorted(set(disabled))))
    return sorted(set(disabled))


def shadow_report() -> dict:
    """当前依赖目录里被停用的包 / 仍在生效的可选包。"""
    v = install_dir()
    disabled, active = [], []
    if os.path.isdir(v):
        for name in os.listdir(v):
            if name.endswith(_SHADOW_SUFFIX):
                base = name[:-len(_SHADOW_SUFFIX)]
                disabled.append(base[:-3] if base.endswith(".py") else base)
            elif name in OPTIONAL_IMPORTS:
                active.append(name)
    return {"dir": v, "disabled": sorted(disabled), "active": sorted(active)}


OPTIONAL_IMPORTS = {m["verifyImport"] for m in OPTIONAL.values()}


def _has(module: str) -> bool:
    if importlib.util.find_spec(module) is not None:
        return True
    # 也检查当前解释器在 vendor 下对应的目录
    v = install_dir()
    if os.path.isdir(v) and os.path.isdir(os.path.join(v, module)):
        return True
    return False


def status() -> dict:
    out = {}
    for key, meta in OPTIONAL.items():
        out[key] = {
            "installed": _has(meta["verifyImport"]),
            "title": meta["title"],
            "desc": meta["desc"],
            "size": meta["size"],
            "pip": meta["pip"],
            "provides": meta["provides"],
        }
    out["_env"] = {
        "python": sys.executable,
        "pythonTag": _python_tag(),
        "pipIndex": DEFAULT_INDEX,
        "vendor": vendor_dir(),
        "installDir": install_dir(),
        "installDirExists": os.path.isdir(install_dir()),
        "installedTag": installed_tag(),
        "versionMismatch": version_mismatch(),
    }
    return out


def _pip_env() -> dict:
    """构造 pip 运行环境：临时目录与缓存都放进工程，避开权限问题。"""
    env = dict(os.environ)
    tmp = os.path.join(cfgmod.BASE_DIR, ".tmp")
    os.makedirs(tmp, exist_ok=True)
    env["TEMP"] = tmp
    env["TMP"] = tmp
    env["PIP_CACHE_DIR"] = os.path.join(tmp, "pipcache")
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _pip_cmd(pkg: str, target: str, index: str) -> list[str]:
    host = index.split("//")[-1].split("/")[0]
    return [sys.executable, "-m", "pip", "install", "--no-input",
            "--no-cache-dir", "--target", target,
            "--index-url", index, "--trusted-host", host, pkg]


def _python_tag() -> str:
    return "cp%d%d" % (sys.version_info.major, sys.version_info.minor)


def install_dir() -> str:
    """当前解释器对应的依赖目录：``vendor/<cpXXX>``。

    为什么按版本分目录：含 C 扩展的包（ctranslate2 / av / tokenizers）与 Python
    版本强绑定。把两个版本的依赖塞进同一个目录会互相污染 —— 实测会出现旧版本的
    .pyd 被优先加载，报 ``module 'ctranslate2' has no attribute 'StorageView'``。
    分目录后每个解释器各有一份干净依赖，启动器选谁都能跑。
    """
    return os.path.join(vendor_dir(), _python_tag())


_TAG_FILE = "installed-by-python.txt"


def write_install_tag(target: str | None = None) -> None:
    """记录可选依赖是哪个 Python 版本装的。

    含 C 扩展的包（ctranslate2、av、tokenizers…）与 Python 版本强绑定，
    用别的版本解释器去加载会报奇怪的属性错误。记下来才能在启动时给出明确提示。
    """
    target = target or install_dir()
    try:
        os.makedirs(target, exist_ok=True)
        with open(os.path.join(target, _TAG_FILE), "w", encoding="utf-8") as f:
            f.write("%s\n%s\n" % (_python_tag(), sys.version.split()[0]))
    except OSError:
        pass


def installed_tag(target: str | None = None) -> str | None:
    target = target or install_dir()
    path = os.path.join(target, _TAG_FILE)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return f.readline().strip() or None
    except OSError:
        return None


def version_mismatch() -> dict | None:
    """可选依赖的安装版本与当前解释器不一致时返回说明，否则 None。"""
    want = installed_tag()
    if not want:
        return None
    cur = _python_tag()
    if want == cur:
        return None
    return {
        "installedWith": want,
        "running": cur,
        "hint": ("可选依赖是为 %s 安装的，当前解释器是 %s，含 C 扩展的包无法通用。"
                 "请用同一个 Python 重新运行：python app.py --install faster-whisper "
                 "（或启动时选用 %s 的解释器）" % (want, cur, want)),
    }


def check_import_health(module: str) -> tuple[bool, str]:
    """真正导入一次，确认包可用（不只是目录存在）。"""
    try:
        mod = __import__(module)
    except Exception as e:
        return False, "%s: %s" % (type(e).__name__, e)
    return True, getattr(mod, "__file__", "") or ""


def _fix_permissions(target: str, on_log=None) -> bool:
    """让安装目录对当前用户完全可读写（Windows）。

    为什么需要：某些环境下 pip 写入的目录会出现「顶层能列、子目录读不了」的
    权限状态，Python 就把包认成空的命名空间包，导入时报
    ``cannot import name 'X' from 'pkg' (unknown location)``。
    用 icacls 递归授予当前用户完全控制，一次解决。
    """
    if os.name != "nt":
        return True
    import subprocess as _sp
    user = "%s\\%s" % (os.environ.get("USERDOMAIN", ""), os.environ.get("USERNAME", ""))
    if user == "\\":
        return False
    creationflags = getattr(_sp, "CREATE_NO_WINDOW", 0)
    try:
        cp = _sp.run(["icacls", target, "/grant", "%s:(OI)(CI)F" % user,
                      "/T", "/C", "/Q"],
                     capture_output=True, creationflags=creationflags, timeout=900)
    except Exception as e:
        if on_log:
            on_log("权限修复失败：%s" % e)
        return False
    ok = cp.returncode == 0
    if on_log:
        on_log("已确认安装目录可读写" if ok else "权限修复未完成（可能影响导入）")
    return ok


def _try_install(pkg: str, target: str, index: str, timeout: int) -> tuple[bool, str]:
    args = _pip_cmd(pkg, target, index)
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        cp = subprocess.run(args, capture_output=True, env=_pip_env(),
                            timeout=timeout, creationflags=creationflags)
    except subprocess.TimeoutExpired:
        return False, "安装超时（%ds）" % timeout
    text = (cp.stdout or b"").decode("utf-8", "replace") + \
           (cp.stderr or b"").decode("utf-8", "replace")
    return cp.returncode == 0, text


def install(pkg_key: str, on_log=None, target: str | None = None,
            index: str | None = None, timeout: int = 3600,
            force: bool = False) -> dict:
    """安装一个可选依赖。返回 {ok, log, elapsed}。

    先试国内镜像，失败再回退官方源（镜像缺包时只能这样）。
    ``force=True`` 时忽略「已安装」判断强制重装 —— 换过 Python 版本后需要这样
    重建，因为含 C 扩展的包跟 Python 版本强绑定。
    """
    log = on_log or (lambda m: None)
    meta = OPTIONAL.get(pkg_key)
    if not meta:
        return {"ok": False, "error": "未知依赖 %s" % pkg_key, "log": ""}
    if not force and _has(meta["verifyImport"]) and not version_mismatch():
        return {"ok": True, "alreadyInstalled": True, "log": ""}
    if force:
        log("强制重装（当前解释器 %s）" % _python_tag())

    target = target or install_dir()
    os.makedirs(target, exist_ok=True)
    # 换 Python 版本重装时，必须清空目标目录：pip --target 是叠加写入，
    # 旧版本编译的 .pyd（如 ctranslate2 的 cp312 扩展）会留在原地并被优先加载，
    # 表现为“装成功了但导入报奇怪的属性错误”。
    if force and os.path.isdir(target):
        log("清空旧的安装目录（换版本必须重建，否则旧扩展会被误加载）")
        for name in os.listdir(target):
            full = os.path.join(target, name)
            try:
                if os.path.isdir(full):
                    shutil.rmtree(full, ignore_errors=True)
                else:
                    os.remove(full)
            except OSError:
                pass
    indices = [index] if index else [DEFAULT_INDEX]
    if not index and FALLBACK_INDEX not in indices:
        indices.append(FALLBACK_INDEX)

    t0 = time.time()
    text = ""
    ok = False
    used = ""
    for idx in indices:
        log("执行：pip install --target %s %s" % (target, meta["pip"]))
        log("镜像：%s" % idx)
        ok, text = _try_install(meta["pip"], target, idx, timeout)
        used = idx
        if ok:
            break
        if idx != indices[-1]:
            log("该镜像不可用，回退到 %s 重试…" % indices[-1])

    tail = "\n".join(text.strip().splitlines()[-25:])
    # 校验真的装上了（pip 返回 0 不代表包目录存在）
    ok = ok and (os.path.isdir(os.path.join(target, meta["verifyImport"]))
                 or os.path.isdir(os.path.join(target, meta["pip"].replace("-", "_"))))
    if ok:
        # 关键：修正安装目录权限。某些环境下 pip 写入的目录会出现「顶层能列、
        # 子目录读不了」的状态，Python 会把包当成空命名空间包，导入时报
        # cannot import name 'X' from 'pkg' (unknown location)。这里一次修好。
        _fix_permissions(target, log)
        # 记下是哪个 Python 版本装的，供启动时做版本一致性检查
        write_install_tag(target)
    # 装完立刻做遮蔽保护，防止传递依赖（如 numpy）破坏宿主环境
    disabled = enforce_no_shadowing(log)
    log("安装%s（用时 %.1fs，源：%s）"
        % ("成功" if ok else "失败", time.time() - t0, used))
    if not ok:
        for line in tail.splitlines()[-6:]:
            log("  " + line)
    return {"ok": ok, "elapsed": round(time.time() - t0, 1), "log": tail,
            "target": target, "index": used, "shadowed": disabled,
            "error": None if ok else "安装失败（镜像与官方源都不可用或缺包）"}


def ensure_on_path() -> None:
    """把当前解释器对应的依赖目录加入模块搜索路径。

    用的是 ``vendor/<cpXXX>``，不同 Python 版本互不干扰；老版本装在同一层的
    散装包（``vendor/xxx``）也继续兼容，只是要过遮蔽保护。
    """
    enforce_no_shadowing()
    for v in (install_dir(), vendor_dir()):
        if os.path.isdir(v) and v not in sys.path:
            sys.path.insert(0, v)


def self_test(on_log=None) -> dict:
    log = on_log or (lambda m: None)
    ensure_on_path()
    res = {}
    for key, meta in OPTIONAL.items():
        ok = _has(meta["verifyImport"])
        res[key] = ok
        log("  %-16s %s" % (meta["pip"], "已安装" if ok else "未安装（可选）"))
    return res
