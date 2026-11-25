# KoeDub · 音声双语化工作台

把音声（同人音声 / ASMR / 广播剧）的 MP3 **注入**进来，自动完成：

```
MP3 注入 → 左右耳声源判定 → 语音识别 → 自适应译中 → AI 配音 → 排轴吸附 → 声道摆位 → 成品
```

最终得到一条音频：**日语原声仍在原来那只耳朵，中文配音出现在完全对侧的那只耳朵**，
时间轴与原文逐句对齐。

---

## 一、先回答：「总共需要调用多少种不同的 API」

**8 种能力类别 = 8 种 API**，其中 **6 种必选 + 2 种可选**。
类别才是「一种 API」；同一类别里换厂商（比如识别从本地模型换成 Groq）不算新增种类。

| # | 能力类别 | 必选/可选 | 作用 | 默认实现 | 可替换为 |
|---|---|---|---|---|---|
| 1 | 媒体探测 | 必选 | 读时长/采样率/声道 | FFprobe（本地） | — |
| 2 | 语音识别 ASR | 必选 | 日语原声 → 带时间戳文本 | 本地 faster-whisper | OpenAI / DeepSeek / Groq / 硅基流动 / Demo / 手工导入 |
| 3 | 自适应翻译 | 必选 | 日语 → 中文（带上下文与时长预算） | 内置引擎 | DeepSeek / 任意 OpenAI 兼容端点 / Ollama / 直通 |
| 4 | AI 配音 TTS | 必选 | 中文台词 → 语音 | Edge 神经语音（免费） | Azure / OpenAI / ElevenLabs / 自定义 HTTP / Windows SAPI |
| 5 | 排轴对齐 | 必选 | 每句中文吸附到原文时段 | 内置求解器 + atempo | — |
| 6 | 声道摆位合成 | 必选 | 原声留原耳、中文放对侧耳 | 内置中/侧分解 + ffmpeg | — |
| 7 | 人声分离 | 可选 | 把台词与音效彻底分开 | 不分离 | Demucs / UVR / 中置提取 |
| 8 | 字幕导出 | 可选 | 中日双语 SRT / ASS | 内置生成 | — |

**实测环境下的可达性**（本机探测结果，供选型参考）：

| 服务 | 状态 | 说明 |
|---|---|---|
| 本地 Whisper（faster-whisper） | ✅ 已实测可用 | 免费离线；模型走 hf-mirror，21 秒音频约 5～40 秒 |
| Edge 神经语音 | ✅ 已实测可用 | 免费、无需密钥，中文音色 14 个 |
| DeepSeek | ✅ 可达（401=没密钥） | 翻译首选 |
| Groq | ✅ 可达（403=没密钥） | Whisper 识别极快 |
| 硅基流动 | ✅ 可达（401） | 国内节点 ASR/LLM |
| 阿里 DashScope | ✅ 可达（401） | ASR/TTS |
| Azure Speech | ✅ 可达（404，路径需密钥） | 音色最多 |
| ElevenLabs | ✅ 可达（200） | 音质顶级 |
| OpenAI | ❌ 不通 | 需要代理 |
| Google TTS | ❌ 超时 | 已不作为默认 |
| HuggingFace 官方 | ❌ 不通 | **hf-mirror.com ✅（代码已默认走它）** |
| PyPI 官方 | ✅ 可达 | 清华镜像更快；清华缺 faster-whisper，代码会自动回退官方源 |

> **完全零密钥也能出成品**：识别用本地模型、翻译用内置引擎、配音用 Edge，
> 三个必选环节都有免费/离线实现。

查看清单：`python app.py --apis`，或点界面右上角的 **8 种 API 能力** 徽章。

---

## 二、快速开始

### 1. 环境要求

- Python 3.9+（3.12 实测通过）
- **FFmpeg**（必须）：`winget install Gyan.FFmpeg`
- 依赖：核心链路只用标准库 + `numpy`，**不装任何 pip 包也能出成品**
- 想要离线真实识别：一条命令装本地模型（见下）

### 2. 启动

双击 **`启动.cmd`**。启动器会先挑一个能真正运行本软件的 Python（检查 `numpy`
是否可用、Python 版本是否与已装的可选依赖匹配），再拉起本地服务并打开界面。
出错时会把原因和排查命令留在窗口里，不会一闪而过。

也可以命令行启动：

```powershell
python app.py                 # 启动界面（默认用系统默认浏览器打开）
python app.py --app-window    # 用独立应用窗口打开（Edge/Chrome 应用模式）
python app.py --no-browser    # 不自动打开界面
python app.py --port 9000     # 指定端口
python app.py --shortcut      # 在桌面创建快捷方式
python app.py --selftest      # 环境自检
python app.py --deps          # 查看可选依赖与已下载模型
python app.py --apis          # 打印 API 清单
python app.py --tools selftest.py --with-speech   # 跑 tools/ 下的脚本
```

浏览器/应用窗口打开后：**拖入 MP3 → 看左右耳判定 → 点「开始注入」**。

> 界面「可选增强」里有一键安装按钮，等价于下面的命令。

### 3. 装本地识别模型（可选，但强烈建议）

```powershell
python app.py --install faster-whisper
```

- 自动走 **PyPI 清华镜像**，装在工程内 `vendor/`，**不动系统环境**，删掉即回退；
- 模型从 **hf-mirror.com** 下载（HuggingFace 官方在国内常超时；代码已自动设好端点，
  并禁用镜像不支持的 Xet 后端）；
- 装好后识别完全离线，音频不出本机。

模型大小与效果（首次识别某档会下载一次）：

| 模型 | 体积 | 日语效果 | 21 秒音频推理耗时（CPU） |
|---|---|---|---|
| tiny | 75MB | 能听懂大致内容 | ~5s（首次加载 34s） |
| small（推荐） | 464MB | 基本准确 | ~20-40s |
| medium / large-v3 | 1.5G / 3G | 明显更准 | 更慢 |

### 4. 接入调教音色（GPT-SoVITS，可选但强烈推荐）

默认配音用 Edge 在线音色（`zh-CN-XiaoxiaoNeural`）——**音色与原声无关**，听感是「
另一个人在念稿」。换成 GPT-SoVITS 后，中文直接克隆原声说话人的嗓子。

**全程在界面里点，不用碰命令行**：

1. 左栏最上面那个「环境体检」面板会自动检查；
2. 没装 GPT-SoVITS 时点 **「一键下载整合包」**（约 8GB，可取消，断点续传）——
   下载完自动解压、自动找到 `api_v2.py`、自动把路径写回配置；
3. 解压 7z 需要 7-Zip：`winget install 7zip.7zip`（已装会跳过）；
4. 点 **「启动 GPT-SoVITS」**（约 13 秒，加载 4 个模型）；
5. 点 **「自检试听」**：用一句中文走完整条合成链路，能听到声音就说明通了。

实测（8GB 显存的中端移动独显，参考音频 6 秒）：

| 环节 | 耗时 |
|---|---|
| 启动服务（冷） | 13.3 秒 |
| 自检合成一句（5.4 秒音频） | 5.4 秒 |
| 90 秒音声全流程（14 段） | 44.6 秒 |

> **音色从哪来**：不用预先训练。程序会从你自己上传的音声里自动挑一句最干净的
> （时长 5.5~9.5 秒、信噪比高、浊音占比健康、共振峰丰富），拿它当参考音频做零样本克隆。
> 挑句实测 0.75 秒，选中的那句台词会写进成品的 `manifest.json` 里，可复查。

手动指定也行——在 `config.json` 里填：

```json
"tts": {
  "provider": "gpt_sovits",
  "sovitsInstallDir": "D:\\GPT-SoVITS",
  "sovitsRefMode": "fixed",
  "sovitsRef": "D:\\你的参考音频.wav",
  "sovitsPromptText": "参考音频里说的那句话"
}
```

`tts.sovitsRefMode` 是 `auto`（默认，自动挑句）或 `fixed`（用你指定的）。

### 5. 命令行直接跑

```powershell
python app.py --cli 你的音声.mp3                          # 用默认配置跑完整流水线
python app.py --cli a.mp3 --asr whisper_local --tts edge  # 指定各环节 provider
python app.py --cli a.mp3 --force-side left               # 手动指定原声在左耳
python app.py --cli a.mp3 --voice zh-CN-YunxiNeural       # 换音色
```

实测：21.88 秒真实日语音声，从识别到出成品 **15.1 秒**（tiny 模型，模型已缓存）。

---

## 三、界面怎么用

| 区域 | 说明 |
|---|---|
| 左栏 1 | 拖入 MP3，立即显示时长/采样率/声道，并可直接试听 |
| 左栏 2 | **左右耳判定结果**：哪只耳是原声、哪只耳会放中文，附置信度/电平差/ITD |
| 左栏 3 | 识别 / 翻译 / 配音三项 provider 与音色选择（带「试听这个音色」） |
| 左栏 4 | 开始；可勾选额外烧录字幕视频 |
| 中栏 | 十步流水线进度 + 实时日志 |
| 右栏 | **对耳试听**（原声耳 / 中文耳分别一个文件，戴耳机点播即可验收）、下载、逐句台词表 |

---

## 四、核心机制说明

### 4.1 左右耳声源怎么判定的

三重证据互相校验（`core/channels.py`）：

1. **能量差 ILD**：逐帧算 L/R 电平，用能量加权的线性差判定方向。
   （不用 dB 均值——对侧近乎静音时 dB 会飙到 ±100 dB 把结论带偏。）
2. **方向一致性**：强偏帧里方向一致的比例。
3. **互相关 ITD**：通道间时差，用来识别「真·侧向声源」还是「混响造成的假侧向」。

判定结果由界面可视化：`人声在左耳（置信度 80%，电平差 +22.8 dB）→ 中文配音放右耳`。
还会**逐段再判一次**，因为音声常常在句子之间换耳——但逐段结果要过三道闸门
（时长 ≥0.35s、置信度 ≥0.6、与整体判定冲突时需 ≥0.78）才允许覆盖整体结论。
这一步是必需的：句间停顿/纯音效片段一侧近乎静音，逐段电平差会飙到 ±60 dB
看着"极其确定"，其实是噪声；照单全收会把中文放到原声那一耳上。

### 4.1.1 判定精度（自检实测）

| 样本（Edge 日语真实朗读构造） | 期望 | 实际 | 结果 |
|---|---|---|---|
| 人声在左耳 | left | left（置信度 1.00，+25.1 dB） | ✓ |
| 人声在右耳 | right | right（置信度 0.97，−25.1 dB） | ✓ |
| 人声居中（双耳混音） | center | center（置信度 0.18，−0.2 dB） | ✓ |
| 左耳人声 + 45% 对侧泄漏 | left | left（置信度 0.82，+6.3 dB） | ✓ |

### 4.2 中文怎么放到「完全对侧」

`cfg.mix.originalMode` 四种策略：

| 值 | 行为 | 何时用 |
|---|---|---|
| `media_swap`（默认） | 中/侧分解：原声台词（Mid+Side）留原耳，对侧耳只留音效（Side），人声位腾给中文 | 大多数情况，听感最自然 |
| `swap` | 配音时段整段左右对调 | 想要「这段时间这只耳完全是中文」 |
| `duck` | 对侧耳在配音时段下压若干 dB | 原声与配音都想留着 |
| `keep` | 原声立体声原样不动 | 调试/对比 |
| `mono_split` | 音效折叠到双耳 | 音效明显偏一侧时 |

### 4.3 排轴怎么保证不跑偏

1. 每段可用空间 = `min(原文时长, 到下一段之前的空隙)`，绝不抢下一句的时间。
2. 配音太长 → `atempo` 变速压缩（不变调），上限 `maxTempo`（默认 1.45x）。
3. 配音太短 → 补静音填满，保持与原文同节奏。
4. 压到上限仍放不下 → 标记 `overrun` 并在台词表里显示 ⚠，提示精简文案。

**实测时间对齐误差：-0.1 ms。**

---

## 五、配置

设置存在 `config.json`（界面里改会写回这里）。关键项：

```jsonc
{
  "asr":       { "provider": "auto", "model": "small", "language": "ja", "maxSegmentSeconds": 7.0 },
  "translate": { "provider": "auto", "style": "adaptive", "maxCharsPerSecond": 7.5,
                 "glossary": { "先輩": "前辈" } },
  "tts":       { "provider": "edge", "voice": "zh-CN-XiaoxiaoNeural", "rate": "+8%" },
  "align":     { "maxTempo": 1.45, "minGapMs": 60 },
  "mix":       { "originalMode": "media_swap", "targetLufs": -16.0 },
  "keys":      { "DEEPSEEK_API_KEY": "", "GROQ_API_KEY": "", "AZURE_SPEECH_KEY": "" }
}
```

密钥也可以用环境变量给（优先级更高）：`DEEPSEEK_API_KEY`、`OPENAI_API_KEY`、
`GROQ_API_KEY`、`SILICONFLOW_API_KEY`、`AZURE_SPEECH_KEY`+`AZURE_SPEECH_REGION`、
`ELEVENLABS_API_KEY`。

### 提升质量的三条建议

1. **识别**：`pip install faster-whisper`，模型选 `medium` 或 `large-v3`（日语明显更准）。
2. **翻译**：填 `DEEPSEEK_API_KEY`，翻译从「术语表替换」升级为「带上下文的整段意译」。
3. **配音**：Edge 已够用；追求音质可上 Azure 或 ElevenLabs。

---

## 六、自检与验证

```powershell
python tools/selftest.py --with-speech   # 完整套件：4 种摆位 + 真实识别（推荐）
python tools/selftest.py                 # 只跑 4 种摆位（不调识别）
python tools/selftest.py --offline-samples   # 不联网，用离线合成音当样本
```

样本统一用 **Edge 日语真实朗读**构造（合成音的峰均比与真人差很多，拿它验证
"居中人声是否被判成居中"会得出误导性结论），语音只合成一次、四种摆位复用。

每项检查都基于信号本身（不看日志）：

| 检查 | 判据 |
|---|---|
| A 声道判定 | left / right / center 是否判对 |
| A2 真实识别 | 用的是真识别后端（非 Demo）、段落数合理、文本不含占位符 |
| B 中文落在对侧耳 | 中文音频与成品「对侧耳」的相关，须高于与「原声耳」的相关 |
| C 原声留在原耳 | 配音时段内，源的原声侧声道与「原声耳」的相关须高于与「对侧耳」的相关 |
| D 时间对齐 | 配音实际落点相对排轴的偏移 ≤ ±60ms |
| E 成品完整 | mp3 / wav 存在且时长与原声一致 |

**实测结果（5/5 全部通过）**：

```
A✓ A2✓ B✓ C✓ D✓ E✓   left    中文→右耳  相关 0.256 vs 0.397，原声耳 1.000 vs 0.397
A✓ A2✓ B✓ C✓ D✓ E✓   right   中文→左耳  相关 0.358 vs 0.382，原声耳 0.995
A✓ A2✓ B✓ C✓ D✓ E✓   center  中文→左耳  原声耳 0.992 vs 0.039
A✓ A2✓ B✓ C✓ D✓ E✓   leak    中文→右耳（45% 泄漏样本）
A✓ A2✓ B✓ C✓ D✓ E✓   speech  whisper_local 真实识别 4 段，时间对齐 −0.1 ms
```

**Web 端全链路验收**（上传 → 判定 → 真实识别 → 配音 → 摆位 → 出成品）：

```
上传真实日语音声        case_left.mp3 | 21.88s | 2ch | 0.5 MB
左右耳判定             人声在 left（置信度 100%，+25.1 dB）→ 中文放 right
识别（whisper_local）   5 段，日文逐字正确，时间轴与原文对齐
  0.00- 3.00 ねぇ、聞こえてる?
  3.00- 7.00 今日はちょっと疲れてるみたいだね。
  7.00-11.40 無理しないで、ゆっくりやすんでいいんだよ。
 11.40-15.80 耳元で、小さな声で話すね。
 15.80-21.80 日本語の声は左側、中国語は反対側に流れるよ。
排轴                   覆盖 98.5%，平均压缩 1.00x
成品                   wav / mp3 / m4a / mp4    总耗时 15.0 秒
```

辅助工具：

```powershell
python tools/make_speech_sample.py out.mp3 --voice-side right   # 造真实语音样本
python tools/make_sample.py out.mp3 --voice-side left --leak .4 # 造离线合成音样本
python tools/verify_output.py 成品.mp3                          # 量化成品左右耳分布
python tools/cleanup.py --yes                                   # 清理中间产物
```

---

## 七、目录结构

```
koedub/
├─ app.py                 启动入口（应用窗口 / CLI / 自检 / 装依赖 / API 清单）
├─ 启动.cmd                双击启动（独立应用窗口）
├─ config.json            配置与密钥（首次运行后生成）
├─ vendor/                可选依赖安装位置（与系统环境隔离，删掉即回退）
├─ core/
│  ├─ providers.py        ★ API 能力注册表（8 类 / 26 个实现）
│  ├─ pipeline.py         主流水线（十步）
│  ├─ audiocore.py        ffmpeg/ffprobe 封装 + PCM 读写
│  ├─ channels.py         ★ 左右耳声源判定（三重证据 + 逐段闸门）
│  ├─ segmentation.py     VAD 分段 + 能量谷切分
│  ├─ asr.py              语音识别 provider（含 hf-mirror、PyAV 兼容、ffmpeg 预解码）
│  ├─ translate.py        自适应翻译（LLM 批译 + 离线引擎）
│  ├─ tts.py              AI 配音 provider
│  ├─ edge_tts.py         ★ 零依赖 Edge 神经语音（含签名）
│  ├─ wsclient.py         零依赖 WebSocket 客户端
│  ├─ align.py            排轴求解 + 变速渲染
│  ├─ mix.py              ★ 声道摆位与合成
│  ├─ subtitles.py        SRT / ASS 双语字幕
│  ├─ deps.py             可选依赖安装 + 遮蔽保护
│  ├─ runtime.py          ★ 外部程序体检（ffmpeg / GPT-SoVITS 挂载校验）
│  ├─ sovits_proc.py      ★ GPT-SoVITS 启停 / 自检（含 cuDNN 补丁 shim）
│  ├─ sovits_dl.py        ★ 整合包下载 / 解压 / 落位
│  ├─ refsel.py           ★ 参考音频自动挑句（四维打分）
│  ├─ launcher.py         应用窗口 / 桌面快捷方式
│  ├─ httpx.py            HTTP / multipart 工具
│  └─ server.py           零依赖 HTTP 服务端
├─ web/                   前端（原生 HTML/CSS/JS，无构建）
├─ tools/                 样本生成 / 成品验证 / 自检 / 清理
├─ runtime/               一键下载的整合包与解压结果（8GB+，已被 .gitignore 排除）
└─ work/  output/         中间产物与成品
```

★ = 需求核心所在的模块。

---

## 八、常见问题

### 启动类

**Q: 双击 `启动.cmd` 没反应 / 一闪而过？**
启动器现在会：① 自己挑一个**真能跑本软件**的 Python（会逐个试 `import numpy`，
并检查 Python 版本与已装的可选依赖是否匹配）；② 把失败原因和排查命令打在窗口里
并 `pause`，不会再一闪而过。如果它报"没有找到可用的 Python"，按提示装一个
Python 3.9+ 并勾选 *Add Python to PATH*。

**Q: 启动器选的 Python 和我装依赖时用的不是同一个？**
可选依赖按 Python 版本隔离存放：`vendor/cp312`、`vendor/cp313` …。
含 C 扩展的包（ctranslate2 / av / tokenizers）跨版本会报奇怪错误，
所以启动器会优先挑版本匹配的那个解释器。想给另一个 Python 也装一份：

```powershell
# 用目标 Python 直接执行（把 python 换成它的路径）
python app.py --install faster-whisper --force
```

`--force` 会清空该版本的目录重建，避免旧版本的 `.pyd` 残留被误加载。

**Q: 想用独立应用窗口（像桌面软件那样）？**
`启动.cmd --app-window`，或把 `config.json` 里的 `server.appWindow` 设为 `true`。
注意：部分环境（沙箱、受控桌面）里浏览器应用模式会立刻退出，此时会自动回退到
默认浏览器并在控制台说明原因 —— 不会出现"什么都没发生"。

**Q: 想用启动器选定的 Python 跑脚本？**
`启动.cmd --tools selftest.py --with-speech`

### 功能类

**Q: 没装 ffmpeg？**
`winget install Gyan.FFmpeg`，装完重开终端。`python app.py --selftest` 会告诉你路径对不对。

**Q: 识别出来的是「（示范文本）…」？**
那是 Demo 识别器的占位内容，不是识别结果——界面会黄条提示。装本地模型
（`python app.py --install faster-whisper`）或填云端密钥即可。

**Q: 本地模型下载不动？**
代码已默认指向 `hf-mirror.com` 并禁用 Xet 后端。如果仍失败，检查能否访问该站，
或用 `HF_ENDPOINT` 指定别的镜像。

**Q: 模型明明已经下好了，加载时却报代理/网络相关的错？**
已修：模型缓存存在时直接读本地快照目录，完全不经过 huggingface_hub。之前只要
系统里 `HTTP_PROXY` 那类变量写得不规范（例如端口部分带方括号），
`httpx` 会在建 client 时就抛 `InvalidURL: Invalid port`，看起来像模型坏了。
现在只要模型在盘上，加载过程零联网。

**Q: 装了 faster-whisper 却报 `metadata_errors` 错误？**
已内置兼容处理：改用 **ffmpeg 预解码 + numpy 输入**，绕开 PyAV 的版本差异
（faster-whisper 1.2.x 会传 `metadata_errors`，而已发布的 PyAV 还没有这个参数）。

**Q: 报 `cannot import name 'WhisperModel' ... (unknown location)`？**
这是 Windows 权限问题：pip 写入的目录出现「顶层能列、子目录读不了」的状态，
Python 就把包当成空的命名空间包。新版 `--install` 会自动用 `icacls` 递归修正；
旧版本装的可以手动补一次：

```powershell
icacls "vendor" /grant "%USERDOMAIN%\%USERNAME%:(OI)(CI)F" /T /C /Q
```

服务端进程内的导入状态可用 `GET /api/diag/asr` 查看；任务异常堆栈会落到
`work/job-error.log`。

**Q: 上传卡住不动？**
已在 1.0 修复：旧版按「读到连接结束」解析请求体，与 HTTP/1.1 长连接互等。
现在严格按 `Content-Length` 读取。

**Q: 判定成了「居中」，但我听得出人声偏一侧？**
该音频可能是双耳等量混音（没有侧向信息）。可以手动在「原声侧」里强制指定左/右。

**Q: 某句中文被压得很赶（台词表里带 ⚠）？**
那段时间装不下译文。可以：调大「最大压缩」、把「中文语速」调高，或接上 LLM
（翻译时已带 `maxChars` 约束，DeepSeek 会自动压字数）。

**Q: 装的可选依赖会不会污染我的 Python 环境？**
不会。全部装在工程内 `vendor/`。而且代码有**遮蔽保护**：一旦某个包与运行时
自带的重名（例如 faster-whisper 会带上 numpy），会自动在 vendor 里停用该包
（改名加 `.shadowed-by-koedub`），避免版本冲突破坏宿主环境。

**Q: 能处理视频文件吗？**
可以，ffmpeg 能解码的容器都收（mkv/mp4/wav/flac…），输出仍是音频。

**Q: 我的音声是人声+音效混在同一声道怎么办？**
把「人声分离」换成 Demucs（界面里可一键安装），分离后再摆位会更干净。
