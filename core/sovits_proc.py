# -*- coding: utf-8 -*-
"""GPT-SoVITS 服务的启动 / 停止 / 日志。

runtime.py 只做「体检」（我在哪、全不全、活了没），本模块负责动手：
把 api_v2.py 拉起来、等它监听端口、把输出写进日志、需要时再停掉它。

两个必须的坑都收在 shim 里（见 _SHIM_TEMPLATE）：
  1. torchaudio 2.11 起解码器搬到 torchcodec，未装时 torchaudio.load 直接抛
     "TorchCodec is required for load_with_torchcodec" —— 用 soundfile 顶替。
  2. cuDNN v9 对 HiFiGAN 的 fp16 conv1d 每次调用要 ~57ms 做引擎查询（CPU 侧），
     一句话合成从 0.6 秒变 27~44 秒 —— 关掉 cudnn.enabled。
"""
from __future__ import annotations

import copy
import os
import subprocess
import sys
import time

from . import config as cfgmod
from . import runtime

SHIM_NAME = "api_v2_sf.py"
PID_NAME = "sovits.pid"
LOG_NAME = "sovits.log"
BOOT_TIMEOUT = 180.0          # 冷启动要加载 4 个模型，180 秒兜底

# 生成到 GPT-SoVITS 根目录的启动包装。内容与手工版一致，仅把 Accel 的导入
# 包在 try 里 —— v2 早期版本没有 Accel 目录，硬 import 会直接崩。
_SHIM_TEMPLATE = '''# -*- coding: utf-8 -*-
"""由 KoeDub 生成的启动包装，可安全删除（删了下次启动会重新生成）。

作用：修两个与本机环境相关的坑，不改官方 api_v2.py 任何一行。
  1) soundfile 顶替 torchaudio.load —— torchaudio 2.11 起解码依赖 torchcodec；
  2) 关闭 cuDNN —— cuDNN v9 对 HiFiGAN 的逐层变化 fp16 conv1d 要做引擎查询，
     实测把一句话从 0.6 秒拖到 27~44 秒。
"""
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import soundfile as sf
import torch
import torchaudio


def _sf_load(path, **kwargs):
    waveform, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(waveform.T.copy()), sample_rate


torchaudio.load = _sf_load

try:
    import GPT_SoVITS.Accel.PyTorch  # noqa: F401  必须先执行它的模块级副作用
except Exception:
    pass

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

sys.argv[0] = os.path.join(ROOT, "api_v2.py")
runpy.run_path(sys.argv[0], run_name="__main__")
'''


def _work_dir() -> str:
    d = os.path.join(cfgmod.BASE_DIR, ".tmp")
    os.makedirs(d, exist_ok=True)
    return d


def log_path() -> str:
    return os.path.join(_work_dir(), LOG_NAME)


def pid_path() -> str:
    return os.path.join(_work_dir(), PID_NAME)


def ensure_shim(root: str) -> tuple[str, bool]:
    """确保 root 下有一个可用的启动包装。返回 (路径, 是不是这次新写的)。"""
    p = os.path.join(root, SHIM_NAME)
    if os.path.isfile(p) and os.path.getsize(p) > 200:
        return p, False
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(_SHIM_TEMPLATE)
    return p, True


def tail_log(lines: int = 40) -> str:
    p = log_path()
    if not os.path.isfile(p):
        return ""
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            rows = f.readlines()
        return "".join(rows[-max(1, int(lines)):])
    except OSError as e:
        return "（读取日志失败：%s）" % e


def _read_pid() -> int | None:
    p = pid_path()
    try:
        with open(p, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _write_pid(pid: int) -> None:
    try:
        with open(pid_path(), "w", encoding="utf-8") as f:
            f.write(str(pid))
    except OSError:
        pass


def _run(cmd: list[str], timeout: float = 30) -> str:
    """跑个外部命令拿文本输出。故意按字节收、再手工按本地代码页解 ——
    taskkill/tasklist 在中文 Windows 上吐的是 GBK，直接 text=True 会
    在 reader 线程里抛 UnicodeDecodeError（PYTHONUTF8=1 时尤其明显）。"""
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return ""
    raw = p.stdout or b""
    for enc in ("mbcs", "utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", "replace")


def _pid_alive(pid: int) -> bool:
    if not pid:
        return False
    return str(pid) in _run(["tasklist", "/FI", "PID eq %d" % pid, "/NH"], 15)


def start(cfg: dict, on_log=None) -> dict:
    """拉起 GPT-SoVITS。已经在跑就直接返回，不重复起。"""
    def say(msg: str) -> None:
        if on_log:
            try:
                on_log(msg)
            except Exception:
                pass

    host, port = runtime._endpoint_parts(cfg)
    if runtime._port_alive(host, port):
        return {"ok": True, "already": True, "endpoint": "http://%s:%d" % (host, port),
                "message": "服务已经在运行"}

    root = runtime.find_sovits(cfg)
    if not root:
        return {"ok": False, "error": "没找到 GPT-SoVITS 目录（需要含 api_v2.py）"}
    py = runtime._find_python(root, cfg)
    if not py:
        return {"ok": False, "error": "没找到可用的 Python 解释器（整合包的 runtime/python.exe）",
                "dir": root}

    try:
        shim, fresh = ensure_shim(root)
    except OSError as e:
        return {"ok": False, "error": "无法写入启动包装 %s：%s" % (SHIM_NAME, e), "dir": root}
    if fresh:
        say("已生成启动包装 %s（soundfile 解码 + 关 cuDNN）" % SHIM_NAME)

    args = [py, shim, "-a", host, "-p", str(port)]
    yaml = os.path.join(root, "GPT_SoVITS", "configs", "tts_infer.yaml")
    if os.path.isfile(yaml):
        args += ["-c", "GPT_SoVITS/configs/tts_infer.yaml"]
    else:
        say("没找到 tts_infer.yaml，按官方默认配置启动")

    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("PYTHONHOME", None)

    lp = log_path()
    try:
        f = open(lp, "w", encoding="utf-8", newline="\n")
    except OSError as e:
        return {"ok": False, "error": "无法写日志 %s：%s" % (lp, e)}

    flags = 0
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    say("启动：%s" % " ".join(args))
    try:
        proc = subprocess.Popen(args, cwd=root, stdout=f, stderr=subprocess.STDOUT,
                                env=env, creationflags=flags)
    except OSError as e:
        f.close()
        return {"ok": False, "error": "进程启动失败：%s" % e, "dir": root}

    _write_pid(proc.pid)
    t0 = time.time()
    while time.time() - t0 < BOOT_TIMEOUT:
        if proc.poll() is not None:
            f.close()
            tail = tail_log(30)
            return {"ok": False, "dir": root, "pid": proc.pid,
                    "error": "服务启动后退出了（退出码 %s）" % proc.returncode,
                    "logTail": tail}
        if runtime._port_alive(host, port):
            f.close()
            cost = time.time() - t0
            say("服务已就绪：%.1f 秒" % cost)
            return {"ok": True, "dir": root, "pid": proc.pid, "seconds": round(cost, 1),
                    "endpoint": "http://%s:%d" % (host, port),
                    "logTail": tail_log(12)}
        time.sleep(1.0)

    f.close()
    return {"ok": False, "dir": root, "pid": proc.pid,
            "error": "等了 %d 秒端口还没起来" % int(BOOT_TIMEOUT),
            "logTail": tail_log(30)}


def stop(cfg: dict) -> dict:
    """停掉自己起的那个进程。不是我们起的就不动，只回报。"""
    host, port = runtime._endpoint_parts(cfg)
    pid = _read_pid()
    if not pid:
        if runtime._port_alive(host, port):
            return {"ok": False, "error": "端口 %d 上有服务，但不是本程序启动的（没有 pid 记录），"
                                          "请在你自己的窗口里关掉" % port}
        return {"ok": True, "message": "没有在运行"}
    if not _pid_alive(pid):
        try:
            os.remove(pid_path())
        except OSError:
            pass
        return {"ok": True, "message": "进程早已退出"}

    _run(["taskkill", "/PID", str(pid), "/T", "/F"], 30)

    for _ in range(20):
        if not runtime._port_alive(host, port):
            break
        time.sleep(0.5)
    try:
        os.remove(pid_path())
    except OSError:
        pass
    alive = runtime._port_alive(host, port)
    return {"ok": not alive, "pid": pid,
            "message": "已停止" if not alive else "进程已杀但端口仍被占用"}


def status(cfg: dict) -> dict:
    host, port = runtime._endpoint_parts(cfg)
    pid = _read_pid()
    return {
        "alive": runtime._port_alive(host, port),
        "endpoint": "http://%s:%d" % (host, port),
        "pid": pid if (pid and _pid_alive(pid)) else None,
        "ours": bool(pid and _pid_alive(pid)),
        "logPath": log_path(),
        "logTail": tail_log(25),
    }


# ---------------------------------------------------------------- 自检
SELFTEST_TEXT = "你好，这是环境自检，听到这句话说明调教音色已经接通。"


def _ref_cache_dir(cfg: dict) -> str:
    return os.path.join(cfgmod.BASE_DIR, "work", "_refcache")


def _side_prompt(ref_path: str) -> str:
    for cand in (os.path.splitext(ref_path)[0] + ".txt", ref_path + ".txt"):
        try:
            if os.path.isfile(cand):
                with open(cand, "r", encoding="utf-8", errors="replace") as f:
                    return f.read().strip()
        except OSError:
            pass
    return ""


def _pick_ref(cfg: dict) -> tuple[str | None, str, str]:
    """找一句能当锚点的参考音频：配置优先，其次自动挑过的缓存。"""
    t = cfg.get("tts", {}) or {}
    ref = (t.get("sovitsRef") or "").strip()
    if ref and os.path.isfile(ref):
        return ref, (t.get("sovitsPromptText") or "").strip(), "配置里指定的"

    d = _ref_cache_dir(cfg)
    if os.path.isdir(d):
        rows = []
        for f in os.listdir(d):
            if not f.lower().endswith(".wav"):
                continue
            p = os.path.join(d, f)
            try:
                if os.path.getsize(p) > 4096:
                    rows.append((os.path.getmtime(p), p))
            except OSError:
                pass
        if rows:
            rows.sort(reverse=True)
            p = rows[0][1]
            return p, _side_prompt(p), "上次自动挑的"
    return None, "", ""


def selftest(cfg: dict, on_log=None) -> dict:
    """装完之后真说一句话：证明服务、参考音频、合成、落盘整条路都通。"""
    def say(msg: str) -> None:
        if on_log:
            try:
                on_log(msg)
            except Exception:
                pass

    host, port = runtime._endpoint_parts(cfg)
    if not runtime._port_alive(host, port):
        return {"ok": False, "stage": "port",
                "error": "端口 %d 没响应，先点「启动服务」" % port}

    ref, prompt, src = _pick_ref(cfg)
    if not ref:
        return {"ok": True, "partial": True, "stage": "port",
                "message": "服务通了，但还没有参考音频，做不了合成试听"
                           "（先上传一段音声，它会自动挑一句）"}
    if not prompt:
        return {"ok": True, "partial": True, "stage": "ref",
                "ref": ref, "refSource": src,
                "message": "服务通了，参考音频也有（%s），但它旁边没有同名 .txt 台词，"
                           "GPT-SoVITS 需要这段台词才能贴合语调。"
                           "随便上传一段音声跑一次，会自动挑句并生成台词文件。"
                           % os.path.basename(ref)}

    say("用这句当参考：%s（%s）" % (os.path.basename(ref), src))
    from . import tts as ttsmod

    c2 = copy.deepcopy(cfg)
    c2.setdefault("tts", {})
    c2["tts"]["sovitsRef"] = ref
    c2["tts"]["sovitsPromptText"] = prompt

    out = os.path.join(_work_dir(), "selftest.mp3")
    t0 = time.time()
    try:
        r = ttsmod._synth_gpt_sovits(SELFTEST_TEXT, out, c2, say)
    except Exception as e:
        return {"ok": False, "stage": "tts", "ref": ref,
                "error": "%s: %s" % (type(e).__name__, e)}
    cost = time.time() - t0

    dur = 0.0
    try:
        from . import audiocore
        dur = audiocore.probe(out).duration
    except Exception:
        pass

    say("自检成功：%.2f 秒合成出 %.2f 秒音频" % (cost, dur))
    return {"ok": True, "stage": "done", "seconds": round(cost, 2),
            "duration": round(dur, 2), "bytes": r.get("bytes"),
            "path": out, "ref": ref, "refSource": src,
            "promptText": prompt, "text": SELFTEST_TEXT}
