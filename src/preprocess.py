"""
src/preprocess.py
职责：读取原始 Excel → 处理段注列 → 清洗文本 → 输出 cleaned.csv

数据列结构（当前）：
  GroupID | Character | Radical | Phonetic | Dazidian | Duanzhu | Shuowen

段注（Duanzhu）处理逻辑：
  1. 提取 Duanzhu 第一句
  2. 与 Shuowen 第一句对比（去除标点后比较）
     - 相同   → 清空 Duanzhu（说文原文已够用）
     - 不同   → 将 Duanzhu 第一句附加到 Shuowen 末尾
     - Shuowen 为空 → 直接用 Duanzhu 第一句填入 Shuowen
  3. 删除 Duanzhu 列

文件模式：
  - 优先读取 data/raw/characters.xlsx（单文件模式）
  - 若不存在，尝试读取 ren_xin.xlsx / ren_yan.xlsx / xin_yan.xlsx（三文件模式）
"""

import re
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── 工具函数 ──────────────────────────────────────────────────

def _first_sentence(text: str) -> str:
    """提取文本第一句（以句号、分号为分隔符）。"""
    if not isinstance(text, str) or not text.strip():
        return ""
    return re.split(r"[。；]", text.strip())[0].strip()


def _strip_punct(text: str) -> str:
    """去除所有标点符号，用于两句话的比较。"""
    return re.sub(r"[^\u4e00-\u9fff\w]", "", text)


def _merge_duanzhu(shuowen: str, duanzhu: str) -> str:
    """
    段注合并逻辑（核心函数）：
      - 两者均空  → 返回空
      - Shuowen 空 → 返回 Duanzhu 第一句
      - 第一句相同 → 返回原 Shuowen（不附加）
      - 第一句不同 → Shuowen + "；" + Duanzhu 第一句
    """
    sw  = shuowen  if isinstance(shuowen,  str) else ""
    dzh = duanzhu  if isinstance(duanzhu,  str) else ""

    dzh_first = _first_sentence(dzh)
    if not dzh_first:
        return sw                               # Duanzhu 为空，原样返回

    if not sw.strip():
        return dzh_first                        # Shuowen 为空，填入 Duanzhu 第一句

    sw_first = _first_sentence(sw)
    if _strip_punct(sw_first) == _strip_punct(dzh_first):
        return sw                               # 内容相同，不附加
    else:
        sep = "；" if not sw.rstrip().endswith("。") else ""
        return sw.rstrip() + sep + dzh_first   # 内容不同，附加


# ── 文本清洗 ──────────────────────────────────────────────────

def clean_shuowen(text: str) -> str:
    """
    从说文条目提取核心训诂义（在 Duanzhu 合并之后调用）：
      - 只取第一句
      - 去反切注音（X切 / X反）
      - 去形声说明（从X聲 / 从X，X聲）
      - 去部首归属（X部曰）
      - 去段玉裁按语开头（按/玉裁謂）
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    core = re.split(r"[。；]", text)[0]
    core = re.sub(r"[\u4e00-\u9fff]{2,4}[切反][。，]?", "", core)
    core = re.sub(r"从[\u4e00-\u9fff]{1,4}(?:[聲声])?[。，]?", "", core)
    core = re.sub(r"[\u4e00-\u9fff]{1,2}部[曰云]", "", core)
    core = re.sub(r"(?:^|[，。])(?:按|今按|玉裁謂|玉裁谓)[^。；]*", "", core)
    core = re.sub(r"[，、\s]+$", "", core.strip())
    return core if core else ""


def clean_dazidian(text: str) -> str:
    """大字典：取第一义项，去义项编号。"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[義义]項\d+|义项\d+", "", text)
    return re.split(r"[；;]", text)[0].strip()


def build_combined(sw: str, dzd: str) -> str:
    return " ".join(p for p in [sw, dzd] if p)


# ── 读取单个 Excel ────────────────────────────────────────────

def _parse_excel(path: str) -> pd.DataFrame:
    """
    读取一个 Excel 文件，返回规范化 DataFrame。
    自动识别列名（支持中英文别名），兜底按位置读取。
    GroupID 合并单元格导致的 NaN 向下填充。
    """
    df_raw = pd.read_excel(path, header=0).dropna(how="all").reset_index(drop=True)
    if df_raw.empty:
        return pd.DataFrame()

    cols = df_raw.columns.tolist()

    # 列名别名映射（小写匹配）
    ALIASES = {
        "group":     ["groupid", "group_id", "group", "对号", "组号", "编号"],
        "character": ["character", "char", "字形", "字"],
        "radical":   ["radical", "部首"],
        "phonetic":  ["phonetic", "声旁", "聲旁"],
        "dazidian":  ["dazidian", "大字典", "《大字典》(第二版)", "大字典(第二版)"],
        "duanzhu":   ["duanzhu", "段注", "duanzhu"],
        "shuowen":   ["shuowen", "说文", "說文"],
    }
    FALLBACK_IDX = {
        "group": 0, "character": 1, "radical": 2,
        "phonetic": 3, "dazidian": 4, "duanzhu": 5, "shuowen": 6,
    }

    def find_col(key):
        for alias in ALIASES[key]:
            for c in cols:
                if str(c).strip().lower() == alias.lower():
                    return str(c)
        idx = FALLBACK_IDX[key]
        return cols[idx] if idx < len(cols) else None

    col_group    = find_col("group")
    col_char     = find_col("character")
    col_radical  = find_col("radical")
    col_phonetic = find_col("phonetic")
    col_dzd      = find_col("dazidian")
    col_dzh      = find_col("duanzhu")
    col_sw       = find_col("shuowen")

    # GroupID 向下填充（合并单元格）
    if col_group:
        df_raw[col_group] = df_raw[col_group].ffill()

    def get(row, col):
        if col is None:
            return ""
        val = row.get(col, "")
        return str(val).strip() if pd.notna(val) else ""

    records = []
    for _, row in df_raw.iterrows():
        char = get(row, col_char)
        if not char or char.lower() == "nan":
            continue
        records.append({
            "GroupID":   get(row, col_group),
            "Character": char,
            "Radical":   get(row, col_radical),
            "Phonetic":  get(row, col_phonetic),
            "Dazidian":  get(row, col_dzd),
            "Duanzhu":   get(row, col_dzh),
            "Shuowen":   get(row, col_sw),
        })

    return pd.DataFrame(records)


# ── 合并多文件（三文件模式）──────────────────────────────────

def _merge_excels() -> pd.DataFrame:
    frames = []
    file_map = [
        (config.RAW_EXCEL_REN_XIN, "ren_xin"),
        (config.RAW_EXCEL_REN_YAN, "ren_yan"),
        (config.RAW_EXCEL_XIN_YAN, "xin_yan"),
    ]
    for path, label in file_map:
        if not os.path.exists(path):
            print(f"   ⚠️  文件不存在，跳过：{os.path.basename(path)}")
            continue
        df = _parse_excel(path)
        if not df.empty:
            frames.append(df)
            print(f"   ✓ {os.path.basename(path)}: {len(df)} 条")

    if not frames:
        raise FileNotFoundError(
            f"data/raw/ 下未找到任何 Excel 文件。\n"
            f"期望文件（单文件）：characters.xlsx\n"
            f"或（三文件）：{[os.path.basename(p) for p, _ in file_map]}"
        )

    df_all = pd.concat(frames, ignore_index=True)
    before = len(df_all)

    # 去重：(Character, Radical) 唯一，保留 Shuowen 最长的
    df_all["_sw_len"] = df_all["Shuowen"].str.len().fillna(0)
    df_all = (df_all
              .sort_values("_sw_len", ascending=False)
              .drop_duplicates(subset=["Character", "Radical"], keep="first")
              .drop(columns=["_sw_len"])
              .reset_index(drop=True))
    print(f"   去重：{before} → {len(df_all)} 条")

    # 重新按 Phonetic 分配 GroupID
    df_all["_gkey"] = df_all["Phonetic"].where(
        df_all["Phonetic"].notna() & (df_all["Phonetic"] != ""),
        df_all["GroupID"]
    )
    ph_map = {k: str(i+1) for i, k in enumerate(df_all["_gkey"].unique())}
    df_all["GroupID"] = df_all["_gkey"].map(ph_map)
    df_all = df_all.drop(columns=["_gkey"])

    return df_all


# ── 段注处理（Duanzhu）────────────────────────────────────────

def apply_duanzhu(df: pd.DataFrame) -> pd.DataFrame:
    """
    处理 Duanzhu 列：
      1. 合并逻辑写入 Shuowen
      2. 删除 Duanzhu 列
    打印处理统计。
    """
    if "Duanzhu" not in df.columns:
        return df

    # 计数器
    cnt_same, cnt_diff, cnt_filled, cnt_empty = 0, 0, 0, 0

    new_sw = []
    for _, row in df.iterrows():
        sw  = row["Shuowen"]  if isinstance(row["Shuowen"],  str) else ""
        dzh = row["Duanzhu"]  if isinstance(row["Duanzhu"],  str) else ""

        dzh_first = _first_sentence(dzh)

        if not dzh_first:
            new_sw.append(sw)
            cnt_empty += 1
        elif not sw.strip():
            new_sw.append(dzh_first)
            cnt_filled += 1
        elif _strip_punct(_first_sentence(sw)) == _strip_punct(dzh_first):
            new_sw.append(sw)
            cnt_same += 1
        else:
            sep = "；" if not sw.rstrip().endswith("。") else ""
            new_sw.append(sw.rstrip() + sep + dzh_first)
            cnt_diff += 1

    df = df.copy()
    df["Shuowen"] = new_sw
    df = df.drop(columns=["Duanzhu"])

    print(f"   段注处理完成：")
    print(f"     与说文相同（已忽略）：{cnt_same}")
    print(f"     与说文不同（已附加）：{cnt_diff}")
    print(f"     说文为空（已填入）：  {cnt_filled}")
    print(f"     段注为空（已跳过）：  {cnt_empty}")
    return df


# ── 主入口 ────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    print("📂 读取原始 Excel（段注处理同步进行）...")

    # ── 自动选择文件模式 ─────────────────────────────────────
    if os.path.exists(config.RAW_EXCEL_SINGLE):
        print(f"   ✓ 单文件模式：{os.path.basename(config.RAW_EXCEL_SINGLE)}")
        df = _parse_excel(config.RAW_EXCEL_SINGLE)
        if df.empty:
            raise ValueError("Excel 文件解析后为空，请检查格式")
        print(f"   共读取 {len(df)} 条记录")
    else:
        print("   单文件不存在，尝试三文件模式...")
        df = _merge_excels()

    # ── 段注处理 ─────────────────────────────────────────────
    print("\n📝 处理段注（Duanzhu）列...")
    df = apply_duanzhu(df)

    # ── 文本清洗 ─────────────────────────────────────────────
    print("\n✂️  清洗文本...")
    df["Shuowen_clean"]  = df["Shuowen"].apply(clean_shuowen)
    df["Dazidian_clean"] = df["Dazidian"].apply(clean_dazidian)
    df["Combined"]       = df.apply(
        lambda r: build_combined(r["Shuowen_clean"], r["Dazidian_clean"]), axis=1
    )

    # ── 数据质量警告 ─────────────────────────────────────────
    empty_combined = df[df["Combined"].str.strip() == ""]
    if not empty_combined.empty:
        print(f"⚠️  {len(empty_combined)} 条记录清洗后文本为空：")
        print(empty_combined[["GroupID", "Character", "Radical"]].to_string())

    no_phonetic = df[df["Phonetic"].isin(["", "nan"])]
    if not no_phonetic.empty:
        print(f"⚠️  {len(no_phonetic)} 条记录缺少 Phonetic（声旁）列")

    no_radical = df[df["Radical"].isin(["", "nan"])]
    if not no_radical.empty:
        print(f"⚠️  {len(no_radical)} 条记录缺少 Radical（部首）列")

    # ── 保存 ─────────────────────────────────────────────────
    os.makedirs(config.DATA_OUT, exist_ok=True)
    df.to_csv(config.CLEAN_CSV, index=False, encoding="utf_8_sig")

    # ── 摘要 ─────────────────────────────────────────────────
    print(f"\n✅ 清洗完成 → {config.CLEAN_CSV}")
    print(f"   总字数：{len(df)}")
    if not df.empty and "Radical" in df.columns:
        radical_counts = df.groupby("Radical")["Character"].count()
        print(f"   各部首：\n" + radical_counts.to_string())
        group_sizes = df.groupby("GroupID").size().value_counts().sort_index()
        print(f"   各组大小（含n个部首的组数）：\n" + group_sizes.to_string())

    return df


if __name__ == "__main__":
    run()