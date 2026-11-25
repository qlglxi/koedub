# -*- coding: utf-8 -*-
"""自适应翻译层。

「自适应」在这里是三个具体机制，不是空话：
  1. 上下文自适应：整段音声一次性给出，LLM 能看到前后文、人称、语气，
     而不是逐句丢进去翻译；批处理时携带前文摘要。
  2. 时长自适应：结合每段的秒数计算「中文可念字数预算」，
     要求译文长度落在预算内，给后面的排轴留出空间。
  3. 人称/称谓自适应：保留「酱/桑/前辈/主人」等二次元称谓，
     并统一敬语与亲密度，术语表可强制覆盖。

provider 清单见 core/providers.py 的 translate 类别。
"""
from __future__ import annotations

import json
import re
import time

from . import config as cfgmod
from . import httpx as hx

# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------
JA_PUNCT = "。、！？…「」『』（）　 "
CJK_RE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")


def _clean_ja(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^[\s\-–—]+", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def budget_chars(seconds: float, max_cps: float = 7.5, floor: int = 2) -> int:
    """中文语速预算：这段时间内能自然念出的字数。"""
    return max(floor, int(round(max(0.0, seconds) * max_cps)))


def fit_text_to_budget(text: str, seconds: float, max_cps: float = 7.5,
                       keep_head: bool = True) -> str:
    """把译文裁剪/压缩到时长预算内。

    优先按标点断句删减；没有标点时按字数截断并补省略号。
    """
    text = (text or "").strip()
    limit = budget_chars(seconds, max_cps)
    if len(text) <= limit:
        return text
    if keep_head:
        head = text[:limit]
        # 尽量在标点处收尾
        m = re.search(r"[。！？!?，、；;：:…]+(?=[^。！？!?，、；;：:…]*$)", head)
        if m and m.start() >= limit * 0.5:
            head = head[:m.end()]
        if len(head) < len(text):
            head = head.rstrip("，、；;：: ")
            if not head.endswith(("。", "！", "？", "…", "!", "?")):
                head += "…"
        return head
    return text[-limit:]


# ---------------------------------------------------------------------------
# LLM 提示词
# ---------------------------------------------------------------------------
STYLE_HINT = {
    "adaptive": "按音声作品的语气自然翻译，允许意译，追求「听起来就是中文音声」的效果",
    "literal": "尽量贴近字面，保留原句结构",
    "natural": "完全按中文口语习惯重写，读起来最顺",
    "soft": "轻柔低语的风格，适合耳语/助眠类音声，用词柔和",
}


def build_prompt(batch: list[dict], cfg: dict, context_note: str = "") -> str:
    t = cfg.get("translate", {})
    per = cfg.get("_perSegmentBudget", {})
    glossary = t.get("glossary") or {}
    gl_lines = "\n".join("  %s → %s" % (k, v) for k, v in glossary.items()) or "  （无）"

    rows = []
    for s in batch:
        lim = per.get(str(s["id"]), budget_chars(s.get("duration", 2.0),
                                                  t.get("maxCharsPerSecond", 7.5)))
        rows.append({
            "id": s["id"],
            "seconds": round(float(s.get("duration", 2.0)), 2),
            "maxChars": lim,
            "ja": s.get("text", ""),
        })

    if t.get("keepHonorifics", True):
        keep_line = "保留「酱/桑/前辈/主人/老师」等称谓，不要替换成「小X」或省略。"
    else:
        keep_line = "称谓可按中文习惯本地化。"

    return f"""你是专业的日语音声（同人音声/ASMR）中文本地化译者，同时负责配音友好度。

任务：把下面这批带时间戳的日文台词翻译成简体中文，用于 AI 配音。

风格要求：{STYLE_HINT.get(t.get("style", "adaptive"), STYLE_HINT["adaptive"])}。
听众设定：{t.get("persona", "音声作品听众")}。
{keep_line}

硬性约束：
1. 每条译文长度必须 ≤ maxChars（这是它对应的秒数能念完的上限）。超了就精简，不要硬塞。
2. 口播友好：避免生僻词和长定语；数字、单位用中文口语读法。
3. 不要输出解释、不要加引号外的修饰、不要合并或拆分行。
4. 译文要单独成句，听感连得上前后文。
5. 遇到拟声词/喘息/水声等非语言内容，保留为（）括注，如（耳语）、（水声）。

术语表（必须遵守，优先级最高）：
{gl_lines}

{("上文脉络：" + context_note) if context_note else ""}

待翻译内容（JSON）：
{json.dumps(rows, ensure_ascii=False, indent=1)}

只输出 JSON，结构为：
{{"items":[{{"id":1,"zh":"译文"}}]}}
"""


def parse_items(raw: str, expect_ids: list[int]) -> dict[int, str]:
    """从 LLM 回复里抠出 JSON（容忍 ```json 包裹和前后废话）。"""
    if not raw:
        return {}
    text = raw.strip()
    m = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 退一步：逐条抓 "id":n "zh":"..."
        out = {}
        for mm in re.finditer(r'"id"\s*:\s*(\d+)\s*,\s*"zh"\s*:\s*"((?:[^"\\]|\\.)*)"', raw):
            try:
                out[int(mm.group(1))] = json.loads('"%s"' % mm.group(2))
            except Exception:
                pass
        return out
    items = data.get("items") if isinstance(data, dict) else data
    out: dict[int, str] = {}
    if isinstance(items, list):
        for it in items:
            if not isinstance(it, dict):
                continue
            i = it.get("id")
            zh = it.get("zh") or it.get("text") or it.get("translation")
            if i is None or not zh:
                continue
            try:
                out[int(i)] = str(zh).strip()
            except (TypeError, ValueError):
                continue
    return out


# ---------------------------------------------------------------------------
# LLM provider
# ---------------------------------------------------------------------------
LLM_ENDPOINTS = {
    "deepseek": ("https://api.deepseek.com/chat/completions", "deepseek-chat",
                 "DEEPSEEK_API_KEY"),
    "ollama": ("http://127.0.0.1:11434/api/chat", "qwen2.5:7b", None),
}


def _call_openai_style(base_url: str, key: str, model: str, prompt: str,
                       temperature: float, on_log, provider_label: str) -> str:
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是严谨的字幕翻译引擎，只输出要求的 JSON。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "stream": False,
    }
    headers = {"Authorization": "Bearer " + key} if key else {}
    on_log("[%s] 请求 %s (model=%s)" % (provider_label, url.split("/")[2], model))
    data = hx.request_json(url, method="POST", headers=headers, json_body=payload, timeout=300)
    if "choices" in data:
        return (data["choices"][0].get("message", {}) or {}).get("content", "") or ""
    return data.get("_raw", "") or json.dumps(data, ensure_ascii=False)[:400]


def _call_ollama(base_url: str, model: str, prompt: str, temperature: float, on_log) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature},
    }
    on_log("[ollama] 请求 %s (model=%s)" % (base_url, model))
    data = hx.request_json(base_url, method="POST", json_body=payload, timeout=600)
    msg = data.get("message") or {}
    return msg.get("content", "") or data.get("response", "")


def _llm_translate(segments: list[dict], cfg: dict, on_log, provider: str) -> dict[int, str]:
    t = cfg.get("translate", {})
    batch_size = max(1, int(t.get("batchSegments", 12)))
    temp = float(t.get("temperature", 0.3))
    out: dict[int, str] = {}
    context_note = ""

    for i in range(0, len(segments), batch_size):
        batch = segments[i:i + batch_size]
        prompt = build_prompt(batch, cfg, context_note)
        raw = ""
        if provider == "deepseek":
            url, model, keyname = LLM_ENDPOINTS["deepseek"]
            key = cfgmod.get_key(cfg, keyname)
            if not key:
                raise RuntimeError("缺少 DEEPSEEK_API_KEY，无法使用 DeepSeek 翻译。")
            raw = _call_openai_style(url.rsplit("/chat/completions", 1)[0], key, model,
                                     prompt, temp, on_log, "deepseek")
        elif provider == "openai_compat":
            base = cfgmod.get_key(cfg, "OPENAI_BASE_URL") or "https://api.openai.com/v1"
            key = cfgmod.get_key(cfg, "OPENAI_API_KEY")
            if not key:
                raise RuntimeError("缺少 OPENAI_API_KEY（或把 OPENAI_BASE_URL 指向自建端点）。")
            raw = _call_openai_style(base, key, cfg.get("translate", {}).get("model", "gpt-4o-mini"),
                                     prompt, temp, on_log, "openai_compat")
        elif provider == "ollama":
            url, model, _k = LLM_ENDPOINTS["ollama"]
            model = t.get("ollamaModel") or model
            raw = _call_ollama(url, model, prompt, temp, on_log)
        items = parse_items(raw, [s["id"] for s in batch])
        got = 0
        for s in batch:
            zh = items.get(s["id"])
            if zh:
                out[s["id"]] = _postprocess(zh, s, cfg)
                got += 1
        on_log("  批 %d/%d：译文 %d/%d 条" % (i // batch_size + 1,
                                          (len(segments) + batch_size - 1) // batch_size,
                                          got, len(batch)))
        # 生成简短上下文摘要给下一批
        tail = "；".join((s.get("text") or "")[:24] for s in batch[-3:] if s.get("text"))
        context_note = ("前面已翻译到：「%s」" % tail)[:200]
    return out


def _postprocess(zh: str, seg: dict, cfg: dict) -> str:
    t = cfg.get("translate", {})
    zh = (zh or "").strip()
    zh = re.sub(r"^[\"'「『]|[\"'」』]$", "", zh)
    zh = re.sub(r"\s+", "", zh) if CJK_RE.search(zh) else zh
    return fit_text_to_budget(zh, seg.get("duration", 2.0),
                              t.get("maxCharsPerSecond", 7.5))


# ---------------------------------------------------------------------------
# 内置自适应翻译引擎（离线兜底）
# ---------------------------------------------------------------------------
# 常见音声用语：先做长词优先替换，让离线结果也「听得懂」
DEFAULT_GLOSSARY = {
    "お兄ちゃん": "哥哥", "お姉ちゃん": "姐姐", "お兄さん": "哥哥",
    "先輩": "前辈", "先生": "老师", "ご主人様": "主人",
    "ちゃん": "酱", "さん": "桑", "くん": "君", "様": "大人",
    "ありがとう": "谢谢", "ごめん": "对不起", "ごめんなさい": "对不起",
    "大好き": "最喜欢你", "好き": "喜欢", "愛してる": "我爱你",
    "かわいい": "好可爱", "すごい": "好厉害", "やばい": "糟了",
    "大丈夫": "没事吧", "頑張って": "加油", "おやすみ": "晚安",
    "おはよう": "早上好", "こんにちは": "你好", "またね": "回头见",
    "いただきます": "我开动了", "ごちそうさま": "多谢款待",
    "ねえ": "呐", "えっと": "那个…", "うん": "嗯",
    "ちょっと": "稍微", "一緒に": "一起", "本当に": "真的",
    "気持ちいい": "好舒服", "くすぐったい": "好痒",
    "耳元": "耳边", "囁く": "低语", "ささやき": "耳语",
    "リラックス": "放松", "眠れ": "睡着",
    "了解": "了解", "はじめまして": "初次见面",
    "お願い": "拜托了", "無理": "勉强", "安心": "安心",
    "優しい": "温柔", "恥ずかしい": "好害羞",
}
KANA_RE = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")


def run_glossary(segments: list[dict], cfg: dict, on_log) -> dict[int, str]:
    """离线引擎：术语表（用户表 > 内置表）+ 去残留假名标记 + 时长裁剪。

    它不是真翻译，但在没有 LLM 时可保证流水线完整跑通，
    并把「未被术语表覆盖」的句子显式标注出来，提示用户接 LLM。
    """
    t = cfg.get("translate", {})
    table = dict(DEFAULT_GLOSSARY)
    table.update(t.get("glossary") or {})
    keys = sorted(table.keys(), key=len, reverse=True)
    out: dict[int, str] = {}
    unresolved = 0
    for s in segments:
        text = _clean_ja(s.get("text", ""))
        if not text:
            out[s["id"]] = ""
            continue
        zh = text
        for k in keys:
            if k in zh:
                zh = zh.replace(k, table[k])
        zh = zh.replace("　", "")
        # 残留假名统一标成 ❓，提示需要 LLM 才能译准
        if KANA_RE.search(zh):
            unresolved += 1
            zh = KANA_RE.sub("", zh) or text
            if KANA_RE.search(zh):
                zh = re.sub(r"[\u3040-\u309f\u30a0-\u30ff]+", "", zh)
            if not zh.strip():
                zh = "（此句需接入翻译 API）"
        out[s["id"]] = fit_text_to_budget(zh.strip(), s.get("duration", 2.0),
                                          t.get("maxCharsPerSecond", 7.5))
    if unresolved:
        on_log("离线引擎有 %d 句未能真正翻译（已标注）。"
               "接入 DeepSeek/OpenAI 兼容端点可获得完整自适应翻译。" % unresolved)
    return out


def run_passthrough(segments: list[dict], cfg: dict, on_log) -> dict[int, str]:
    on_log("直通模式：不做翻译，仅按时间预算裁剪原文。")
    return {s["id"]: fit_text_to_budget(_clean_ja(s.get("text", "")),
                                        s.get("duration", 2.0),
                                        cfg.get("translate", {}).get("maxCharsPerSecond", 7.5))
            for s in segments}


# ---------------------------------------------------------------------------
# 调度
# ---------------------------------------------------------------------------
def pick_provider(cfg: dict) -> str:
    want = (cfg.get("translate", {}) or {}).get("provider", "auto")
    if want and want != "auto":
        return want
    if cfgmod.get_key(cfg, "DEEPSEEK_API_KEY"):
        return "deepseek"
    if cfgmod.get_key(cfg, "OPENAI_API_KEY"):
        return "openai_compat"
    return "glossary"


def run(segments: list[dict], cfg: dict, on_log) -> dict:
    provider = pick_provider(cfg)
    t0 = time.time()
    on_log("翻译 provider = %s（风格：%s）"
           % (provider, cfg.get("translate", {}).get("style", "adaptive")))
    if provider in ("deepseek", "openai_compat", "ollama"):
        mapping = _llm_translate(segments, cfg, on_log, provider)
        # LLM 漏掉的句子用离线引擎补
        missing = [s for s in segments if not mapping.get(s["id"])]
        if missing:
            on_log("LLM 未覆盖 %d 句，用离线引擎补齐。" % len(missing))
            mapping.update(run_glossary(missing, cfg, on_log))
    elif provider == "glossary":
        mapping = run_glossary(segments, cfg, on_log)
    elif provider == "passthrough":
        mapping = run_passthrough(segments, cfg, on_log)
    else:
        raise RuntimeError("未知翻译 provider: %s" % provider)

    out_segments = []
    empty = 0
    for s in segments:
        zh = mapping.get(s["id"], "")
        if not zh:
            empty += 1
        item = dict(s)
        item["zh"] = zh
        item["zhChars"] = len(zh)
        item["budgetChars"] = budget_chars(s.get("duration", 2.0),
                                           cfg.get("translate", {}).get("maxCharsPerSecond", 7.5))
        out_segments.append(item)
    on_log("翻译完成：%d 段，用时 %.1fs%s"
           % (len(out_segments), time.time() - t0,
              ("，%d 段空译文" % empty) if empty else ""))
    return {"segments": out_segments, "provider": provider,
            "elapsed": round(time.time() - t0, 2)}
