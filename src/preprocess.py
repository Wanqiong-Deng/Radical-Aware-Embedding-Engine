"""
src/preprocess.py
职责：读取原始 Excel (D列大字典, E列段注) -> 解析配对 -> 清洗合并 -> 输出 cleaned.csv
"""

import re
import os
import sys
import pandas as pd


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import config
except ImportError:

    class ConfigMock:
        RAW_EXCEL = "data/characters.xlsx"
        DATA_OUT = "data/output"
        CLEAN_CSV = "data/output/cleaned.csv"
    config = ConfigMock()

# ── 基础清洗工具 ──────────────────────────────────────────

def remove_punctuation(text: str) -> str:
    """去除所有中英文标点符号"""
    if not isinstance(text, str) or not text.strip():
        return ""

    return re.sub(r"[^\w\s\u4e00-\u9fff]+", "", text).strip()

def clean_shuowen_core(text: str) -> str:
    """
    说文基础清洗：取第一句并剔除冗余信息
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    core = re.split(r"[。；]", text)[0]
    core = re.sub(r"[\u4e00-\u9fff]{2,4}[切反][。，]?", "", core)
    core = re.sub(r"从[\u4e00-\u9fff]{1,4}(?:[聲声])?[。，]?", "", core)
    core = re.sub(r"[\u4e00-\u9fff]{1,2}部[曰云]", "", core)
    
    return core.strip()

# ── 核心逻辑：合并说文与段注 ──────────────────────────────────

def process_merged_definition(sw_text: str, dz_text: str) -> str:
    """
    处理说文与段注的合并:
    1. 段注(dz_text)只取第一句话（句号分割）。
    2. 若段注首句与说文(sw_text)清洗后的首句完全相同，则舍弃段注。
    3. 将段注接在说文后面。
    4. 移除所有标点。
    """
    # 得到说文清洗后的核心义
    sw_core = clean_shuowen_core(sw_text)
    
    # 段注处理：取第一句
    dz_core = ""
    if isinstance(dz_text, str) and dz_text.strip():
        dz_core = re.split(r"[。]", dz_text)[0].strip()
    
    # 比较去重：如果段注第一句和说文第一句一样，就不要段注了
    if dz_core == sw_core:
        dz_core = ""
    
    # 合并
    combined = sw_core + dz_core
    
    # 全文本去标点
    return remove_punctuation(combined)

# ── 解析 Excel 结构 ────────────────────────────────────────

def _infer_radical_from_structure(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    严格对应列顺序:
    A: 对号, B: 字形, C: 說文, D: 大字典, E: 段注, F: 義項
    """
    col_pair = df_raw.columns[0]   # A: 对号
    col_char = df_raw.columns[1]   # B: 字形
    col_sw   = df_raw.columns[2]   # C: 說文
    col_dzd  = df_raw.columns[3]   # D: 大字典 (修正位置)
    col_dz   = df_raw.columns[4]   # E: 段注 (修正位置)
    
    # 填充合并单元格的对号
    df_raw[col_pair] = df_raw[col_pair].ffill()
    
    records = []
    RADICAL_ORDER = ["人部", "心部"] 
    
    for pair_id, group in df_raw.groupby(col_pair, sort=False):
        group = group.reset_index(drop=True)
        for idx, row in group.iterrows():
            if idx >= len(RADICAL_ORDER):
                break
            records.append({
                "PairID":   pair_id,
                "Character": row[col_char],
                "Radical":  RADICAL_ORDER[idx],
                "Shuowen_Raw":  row[col_sw],
                "Dazidian_Raw": row[col_dzd],
                "Duanzhu_Raw":  row[col_dz],
            })
    
    return pd.DataFrame(records)

def _add_phonetic(df: pd.DataFrame) -> pd.DataFrame:
    """保持声旁识别功能"""
    if "Phonetic" in df.columns:
        return df
    df["Phonetic"] = df["PairID"].astype(str)
    return df

# ── 主入口 ────────────────────────────────────────────

def run():
    print(" 正在读取数据 (确认 D:大字典, E:段注)...")
    # 读取原始 Excel
    df_raw = pd.read_excel(config.RAW_EXCEL, header=0)
    df_raw = df_raw.dropna(how="all").reset_index(drop=True)
    
    print(" 解析配对结构...")
    df = _infer_radical_from_structure(df_raw)
    df = _add_phonetic(df)
    
    print("✂️  执行清洗逻辑：合并说文与段注并去除标点...")
    # 1. 处理说文+段注的合并列
    df["Shuowen_clean"] = df.apply(
        lambda r: process_merged_definition(r["Shuowen_Raw"], r["Duanzhu_Raw"]), axis=1
    )
    
    # 2. 处理大字典清洗（取首项并去标点）
    def clean_dzd(text):
        if not isinstance(text, str): return ""
        first = re.split(r"[；;]", text)[0]
        return remove_punctuation(first)
    
    df["Dazidian_clean"] = df["Dazidian_Raw"].apply(clean_dzd)
    
    # 3. 最终 Embedding 用的 Combined 列
    df["Combined"] = df.apply(
        lambda r: " ".join([p for p in [r["Shuowen_clean"], r["Dazidian_clean"]] if p]), axis=1
    )
    
    # 检查是否有空数据
    empty = df[df["Combined"].str.strip() == ""]
    if not empty.empty:
        print(f"⚠️  警告：{len(empty)} 条记录清洗后文本为空")
    
    # 保存结果
    os.makedirs(config.DATA_OUT, exist_ok=True)
    df.to_csv(config.CLEAN_CSV, index=False, encoding="utf_8_sig")
    print(f" 处理完成！共 {len(df)} 条记录已存入 {config.CLEAN_CSV}")
    return df

if __name__ == "__main__":
    run()