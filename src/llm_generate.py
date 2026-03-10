"""
src/llm_generate.py  [Step D]
职责：构建 Prompt → 调用 SiliconFlow (Qwen) API → 解析 JSON 返回

LangSmith 集成（可选）：
  pip install langsmith
  export LANGCHAIN_API_KEY=ls__xxxxxxxxxx
  export LANGCHAIN_TRACING_V2=true
  export LANGCHAIN_PROJECT=GlyphDrift

  启用后，每次 LLM 调用自动记录：
    · 完整 prompt（system + user）
    · 模型返回
    · 延迟 / token 用量
    · run_id（可在 LangSmith UI 里追踪）
"""

import os
import sys
import json
import re
import time
from typing import Optional
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── LangSmith 可选初始化 ──────────────────────────────────────

def _init_langsmith():
    """
    尝试初始化 LangSmith tracer。
    若未安装或未配置 API Key，静默跳过，不影响主流程。
    """
    if not config.LANGSMITH_ENABLED:
        return None
    try:
        from langsmith import Client
        from langsmith.wrappers import wrap_openai
        client = Client()
        print(" LangSmith tracing 已启用 "
              f"(project: {os.environ.get('LANGCHAIN_PROJECT', 'default')})")
        return client
    except ImportError:
        print("ℹ️  langsmith 未安装，跳过 tracing（pip install langsmith 可启用）")
        return None
    except Exception as e:
        print(f"ℹ️  LangSmith 初始化失败：{e}，跳过 tracing")
        return None


_ls_client = None   # 模块级单例，只初始化一次


# ── 客户端 ────────────────────────────────────────────────────

def build_client() -> OpenAI:
    """构建 SiliconFlow 客户端（OpenAI 兼容）。"""
    global _ls_client
    _ls_client = _init_langsmith()

    client = OpenAI(
        api_key  = config.LLM_API_KEY,
        base_url = config.LLM_API_BASE_URL,
    )

    # 如果 LangSmith 可用，用 wrap_openai 包装（自动 tracing）
    if _ls_client is not None:
        try:
            from langsmith.wrappers import wrap_openai
            client = wrap_openai(client)
            print("✅ OpenAI 客户端已被 LangSmith 包装")
        except Exception:
            pass   # wrap 失败时继续用原始客户端

    return client


# ── System Prompt ─────────────────────────────────────────────

SYSTEM_PROMPT = """你是一位古汉语训诂专家，精通《说文解字》与汉字形声理论。

你的任务是：根据提供的训诂证据，推测一个形声字的核心字义。

**输出格式**（严格遵守，仅返回 JSON，不包含任何其他文字）：
{
  "prediction": "推测的字义（20-60字，仿照训诂风格，体现推测语气如「当有…之义」）",
  "reasoning":  "推理依据（50-100字，说明声旁语义基础、部首语义类型与近邻参考字的综合判断）"
}

**推测规则**：
1. 说明声旁提供的语音与语义基础
2. 说明目标部首通常赋予字的语义类型
3. 结合近邻参考字的字义，推断可能的义域
4. 语气体现推测性（「当有…之义」「疑指…」「或谓…」）
5. 不编造字形笔画或具体出处
"""


# ── Prompt 构建 ───────────────────────────────────────────────

def _build_evidence(result) -> str:
    """
    根据 PredictionResult 构建证据段落。
    这是 prompt engineering 的核心，证据层次：
      锚点字义 → 偏移路径 → 近邻字义 → 目标部首背景
    """
    lines = [
        f"**查询**：声旁「{result.phonetic}」+ 部首「{result.radical}」",
        f"**预测模式**：{result.mode}",
        "",
    ]

    if result.anchor_chars:
        lines.append("**已知同声旁字（锚点）**：")
        for a in result.anchor_chars:
            lines.append(
                f"  · {a['character']}（{a['radical']}）\n"
                f"    說文：{a.get('shuowen', '无')}\n"
                f"    大字典：{a.get('dazidian', '无')}"
            )
        lines.append("")

    if result.shift_direction:
        lines += [
            f"**部首偏移路径**：{result.shift_direction}",
            "（依据数据集同声旁字的 768 维语义偏移规律推算）",
            "",
        ]

    if result.neighbors:
        lines.append("**语义近邻参考字**（BGE 向量空间中距预测位置最近）：")
        for n in result.neighbors[:5]:
            sim_str = f"{n['similarity']*100:.1f}%"
            lines.append(
                f"  · {n['character']}（{n['radical']}）相似度 {sim_str}\n"
                f"    說文：{n.get('shuowen', '无')}\n"
                f"    大字典：{n.get('dazidian', '无')}"
            )
        lines.append("")

    radical_hint = config.RADICAL_SEMANTICS.get(
        result.radical, f"与「{result.radical[0]}」部字义域相关"
    )
    lines += [
        f"**目标部首语义背景**：{result.radical} — {radical_hint}",
        "",
        "请根据以上证据推测该组合可能形成的字义。",
    ]
    return "\n".join(lines)


# ── LangSmith 手动追踪封装 ────────────────────────────────────

def _traced_call(client: OpenAI, messages: list, phonetic: str, radical: str) -> dict:
    """
    实际 API 调用。
    若 LangSmith 可用且 wrap_openai 失败，手动用 @traceable 包装。
    """
    t0 = time.time()
    response = client.chat.completions.create(
        model       = config.LLM_MODEL,
        temperature = config.LLM_TEMPERATURE,
        max_tokens  = config.LLM_MAX_TOKENS,
        messages    = messages,
        # LangSmith 额外元数据（wrap_openai 会自动附加，手动也可以传）
        extra_body  = {
            "metadata": {
                "phonetic": phonetic,
                "radical":  radical,
            }
        } if config.LANGSMITH_ENABLED else {},
    )
    latency = time.time() - t0
    raw     = response.choices[0].message.content.strip()

    # 手动记录到 LangSmith（当 wrap_openai 未生效时的备用方案）
    if _ls_client is not None and config.LANGSMITH_ENABLED:
        try:
            from langsmith import traceable
            # 简单记录：创建一个 run
            _ls_client.create_run(
                name        = "glyphdrift_predict",
                run_type    = "llm",
                inputs      = {"messages": messages},
                outputs     = {"content": raw},
                extra       = {
                    "latency_s":     latency,
                    "model":         config.LLM_MODEL,
                    "phonetic":      phonetic,
                    "radical":       radical,
                    "tokens_used":   getattr(response.usage, "total_tokens", None),
                },
            )
        except Exception:
            pass   # LangSmith 记录失败不影响主流程

    return {"raw": raw, "latency": latency}


# ── 主生成函数 ────────────────────────────────────────────────

def generate_prediction(result, client: OpenAI) -> dict:
    """
    调用 Qwen，返回 { "prediction": str, "reasoning": str }。
    失败时返回 { "prediction": "", "reasoning": "", "error": str }。
    """
    evidence = _build_evidence(result)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": evidence},
    ]

    try:
        out = _traced_call(client, messages, result.phonetic, result.radical)
        parsed = _parse_json(out["raw"])
        parsed["latency"] = out["latency"]
        return parsed
    except Exception as e:
        return {"prediction": "", "reasoning": "", "error": str(e)}


def _parse_json(raw: str) -> dict:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
    try:
        data = json.loads(cleaned)
        return {
            "prediction": str(data.get("prediction", "")),
            "reasoning":  str(data.get("reasoning",  "")),
        }
    except json.JSONDecodeError:
        return {"prediction": cleaned, "reasoning": "（JSON 解析失败）"}


# ── 测试入口 ──────────────────────────────────────────────────

def run_test():
    """构造 mock result，测试 prompt 和 API 调用。"""
    from src.predict import PredictionResult

    mock = PredictionResult(
        phonetic="童", radical="言部",
        mode="PREDICTED", confidence="medium",
        anchor_chars=[
            {"character": "僮", "radical": "人部",
             "shuowen": "未冠也", "dazidian": "未成年的男子"},
            {"character": "憧", "radical": "心部",
             "shuowen": "意不定也", "dazidian": "意不定"},
        ],
        shift_direction="人部→言部 & 心部→言部",
        neighbors=[
            {"character": "諷", "radical": "言部", "similarity": 0.80,
             "shuowen": "诵也", "dazidian": "讽诵；诵读"},
            {"character": "謠", "radical": "言部", "similarity": 0.76,
             "shuowen": "徒歌也", "dazidian": "歌谣"},
        ],
    )

    evidence = _build_evidence(mock)
    print("=" * 60)
    print("【构建的 Prompt（User 部分）】")
    print("=" * 60)
    print(evidence)
    print("=" * 60)

    print(f"\n🚀 调用 {config.LLM_MODEL} API...")
    client = build_client()
    result = generate_prediction(mock, client)
    print(f"\n预测字义：{result.get('prediction', '')}")
    print(f"推理依据：{result.get('reasoning',  '')}")
    if "error" in result:
        print(f"⚠️  错误：{result['error']}")
    if "latency" in result:
        print(f"⏱  延迟：{result['latency']:.2f}s")


if __name__ == "__main__":
    run_test()
