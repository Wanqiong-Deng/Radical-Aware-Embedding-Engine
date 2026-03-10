"""
config.py — 全局配置
所有脚本从这里读取路径、模型参数、API 配置
"""
import os

# ── 路径 ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_OUT = os.path.join(BASE_DIR, "data", "processed")

# 原始 Excel 文件
# 模式 A（推荐）：单一合并文件，包含所有部首
RAW_EXCEL_SINGLE   = os.path.join(DATA_RAW, "characters.xlsx")

# 模式 B：三个两两对比文件（如果单文件不存在则尝试这三个）
RAW_EXCEL_REN_XIN  = os.path.join(DATA_RAW, "ren_xin.xlsx")
RAW_EXCEL_REN_YAN  = os.path.join(DATA_RAW, "ren_yan.xlsx")
RAW_EXCEL_XIN_YAN  = os.path.join(DATA_RAW, "xin_yan.xlsx")

# 处理后的统一文件
CLEAN_CSV   = os.path.join(DATA_OUT, "cleaned.csv")
EMBED_NPY   = os.path.join(DATA_OUT, "embeddings.npy")
COORDS_CSV  = os.path.join(DATA_OUT, "coords.csv")
METRICS_CSV = os.path.join(DATA_OUT, "metrics.csv")

# ── BGE 模型 ──────────────────────────────────────────────────
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
MODEL_NAME      = "BAAI/bge-base-zh-v1.5"
BGE_INSTRUCTION = "为古汉语词条提取语义特征："

# ── LLM API（SiliconFlow，OpenAI 兼容）────────────────────────
LLM_API_KEY      = ""
LLM_API_BASE_URL = "https://api.siliconflow.cn/v1"
LLM_MODEL        = "Qwen/Qwen2.5-72B-Instruct"   # SiliconFlow 上的 Qwen Plus 对应模型
LLM_TEMPERATURE  = 0.3
LLM_MAX_TOKENS   = 500

# ── LangSmith（可选，装了就自动追踪）─────────────────────────
# 使用前：pip install langsmith
# 然后设置环境变量：
#   export LANGCHAIN_API_KEY=ls__xxxxxxxx
#   export LANGCHAIN_TRACING_V2=true
#   export LANGCHAIN_PROJECT=GlyphDrift
LANGSMITH_ENABLED = bool(os.environ.get("LANGCHAIN_TRACING_V2", ""))

# ── 绘图 ──────────────────────────────────────────────────────
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "PingFang SC"]
matplotlib.rcParams["axes.unicode_minus"] = False

RADICAL_PALETTE = {
    "人部": "#1f77b4",   # 蓝
    "心部": "#d62728",   # 红
    "言部": "#2ca02c",   # 绿
}

# 所有合法部首（用于验证和下拉菜单）
ALL_RADICALS = list(RADICAL_PALETTE.keys())

# ── 分析阈值 ──────────────────────────────────────────────────
SHIFT_LEVELS = [
    (0,          2,           "微弱变动"),
    (2,          5,           "显著变动"),
    (5,          float("inf"),"剧烈变动"),
]

# ── 部首语义背景（供 LLM prompt 使用）───────────────────────
RADICAL_SEMANTICS = {
    "人部": "与人的行为、社会关系、身份地位、人的状态相关",
    "心部": "与心理活动、情感、意志、思想、性格相关",
    "言部": "与语言表达、言辞、约定、命名、交流相关",
    "水部": "与水流、液体、湿润、泽被相关",
    "木部": "与树木、植物、木质器物相关",
    "口部": "与言语、饮食、开口动作相关",
}

EXCEL_COL_NAMES = {
    "group":     "GroupID",
    "character": "Character",
    "radical":   "Radical",
    "phonetic":  "Phonetic",
    "dazidian":  "Dazidian",
    "duanzhu":   "Duanzhu",
    "shuowen":   "Shuowen",
}