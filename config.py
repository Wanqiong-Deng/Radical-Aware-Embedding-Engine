"""
config.py — 全局配置，所有脚本从这里读取路径和参数
"""
import os

# ── 路径 ──────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_RAW    = os.path.join(BASE_DIR, "data", "raw")
DATA_OUT    = os.path.join(BASE_DIR, "data", "processed")

RAW_EXCEL   = os.path.join(DATA_RAW,  "characters.xlsx")
CLEAN_CSV   = os.path.join(DATA_OUT,  "cleaned.csv")
EMBED_NPY   = os.path.join(DATA_OUT,  "embeddings.npy")
COORDS_CSV  = os.path.join(DATA_OUT,  "coords.csv")
METRICS_CSV = os.path.join(DATA_OUT,  "metrics.csv")

# ── 模型 ──────────────────────────────────────────────
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
MODEL_NAME  = "BAAI/bge-base-zh-v1.5"
BGE_INSTRUCTION = "为古汉语词条提取语义特征："

# ── 绘图 ──────────────────────────────────────────────
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "PingFang SC"]
matplotlib.rcParams["axes.unicode_minus"] = False

RADICAL_PALETTE = {
    "人部": "#1f77b4",
    "心部": "#d62728",
}

# ── 分析 ──────────────────────────────────────────────
SHIFT_LEVELS = [(0, 2, "微弱变动"), (2, 5, "显著变动"), (5, float("inf"), "剧烈变动")]