# -*- coding: utf-8 -*-
"""桌面化启动：打开界面窗口，并可创建桌面快捷方式。

不引入任何 GUI 依赖（PyQt / Electron 都不需要）。两种打开方式：
  * 默认用系统默认浏览器打开 —— 最稳，一定成功；
  * ``--app-window`` 用 Edge/Chrome 的「应用模式」打开独立窗口（独立任务栏图标，
    看起来就是个桌面软件）。

应用模式不是所有环境都能用（例如某些沙箱/受控桌面里浏览器拉起后立刻退出），
所以这里会**检测它是否真的起来了**，失败自动回退默认浏览器，并把原因写进日志，
避免出现「双击了但什么都没发生」。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

_CANDIDATES = [
    ("Edge", [
        r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe",
        r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe",
        r"%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe",
    ]),
    ("Chrome", [
        r"%ProgramFiles%\Google\Chrome\Application\chrome.exe",
        r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe",
        r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe",
    ]),
]


def find_app_browser() -> tuple[str, str] | None:
    """找一个支持 --app= 的 Chromium 浏览器。返回 (名字, 路径)。"""
    for name, paths in _CANDIDATES:
        for p in paths:
            full = os.path.expandvars(p)
            if os.path.isfile(full):
                return name, full
    for exe, name in (("msedge", "Edge"), ("chrome", "Chrome")):
        p = shutil.which(exe)
        if p:
            return name, p
    return None


def app_browser_name() -> str:
    f = find_app_browser()
    return f[0] if f else "无"


def profile_dir(port: int) -> str:
    """给应用窗口一个独立配置目录：窗口独立、不与日常浏览器互相干扰。"""
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        ".tmp", "appwindow-%d" % port)
    os.makedirs(base, exist_ok=True)
    return base


def open_in_default_browser(url: str) -> bool:
    try:
        import webbrowser
        return bool(webbrowser.open(url))
    except Exception:
        return False


def open_app_window(url: str, port: int, size: str = "1500,950",
                    settle_seconds: float = 2.5, log=None) -> bool:
    """尝试以应用模式打开独立窗口。

    会等一小会儿确认进程真的活着 —— 有些环境（沙箱、受控桌面）里浏览器
    拉起后立刻退出（退出码形如 0x80000003）。这种情况返回 False，
    由调用方回退默认浏览器。
    """
    say = log or (lambda m: None)
    found = find_app_browser()
    if not found:
        say("未找到 Edge/Chrome，改用默认浏览器")
        return False
    name, exe = found
    args = [
        exe,
        "--app=%s" % url,
        "--user-data-dir=%s" % profile_dir(port),
        "--window-size=%s" % size,
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-features=Translate",
    ]
    creationflags = getattr(subprocess, "DETACHED_PROCESS", 0) if os.name == "nt" else 0
    try:
        proc = subprocess.Popen(args, creationflags=creationflags,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        say("应用窗口启动失败（%s），改用默认浏览器" % type(e).__name__)
        return False

    deadline = time.time() + max(0.5, settle_seconds)
    while time.time() < deadline:
        code = proc.poll()
        if code is not None:
            say("应用窗口拉起后立刻退出（%s，退出码 %s），改用默认浏览器"
                % (name, code))
            return False
        time.sleep(0.15)
    say("已用 %s 应用模式打开独立窗口" % name)
    return True


def open_ui(url: str, port: int, prefer_window: bool = False, log=None) -> str:
    """打开界面：要求应用窗口时先试它，失败或未要求则用默认浏览器。

    返回实际使用的方式：app-window / browser / manual
    """
    say = log or (lambda m: None)
    if prefer_window and open_app_window(url, port, log=say):
        return "app-window"
    if open_in_default_browser(url):
        return "browser"
    say("自动打开浏览器失败，请手动访问：%s" % url)
    return "manual"


# ---------------------------------------------------------------------------
# 桌面快捷方式
# ---------------------------------------------------------------------------
def make_shortcut(name: str = "KoeDub 音声双语化") -> str:
    """在桌面创建快捷方式（用 PowerShell 的 WScript.Shell，不依赖 pywin32）。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bat = os.path.join(root, "启动.cmd")
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    if not os.path.isdir(desktop):
        desktop = os.path.expanduser("~")
    lnk = os.path.join(desktop, "%s.lnk" % name)

    ps = r"""
$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut($env:KOEDUB_LNK)
$sc.TargetPath = $env:KOEDUB_BAT
$sc.WorkingDirectory = $env:KOEDUB_ROOT
$sc.IconLocation = "%SystemRoot%\System32\imageres.dll,167"
$sc.Description = "KoeDub 音声双语化工作台"
$sc.Save()
Write-Output "OK"
"""
    env = dict(os.environ)
    env["KOEDUB_LNK"] = lnk
    env["KOEDUB_BAT"] = bat
    env["KOEDUB_ROOT"] = root
    exe = shutil.which("pwsh") or shutil.which("powershell")
    if not exe:
        raise RuntimeError("找不到 PowerShell，无法创建快捷方式")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    cp = subprocess.run([exe, "-NoProfile", "-Command", ps], capture_output=True, env=env,
                        creationflags=creationflags)
    if cp.returncode != 0 or not os.path.isfile(lnk):
        raise RuntimeError("创建快捷方式失败：%s"
                           % (cp.stderr or cp.stdout).decode("utf-8", "replace")[:300])
    return lnk


if __name__ == "__main__":
    print("可用于应用窗口的浏览器：", app_browser_name())
    if len(sys.argv) > 1 and sys.argv[1] == "--shortcut":
        print("已创建：", make_shortcut())
