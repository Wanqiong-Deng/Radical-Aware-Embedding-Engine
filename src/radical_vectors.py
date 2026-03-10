"""
src/radical_vectors.py  [Step A]
职责：在 768 维空间计算所有 (部首A → 部首B) 的平均语义偏移向量

原理：
  king - man + woman ≈ queen
  vec(心部字) - vec(人部字) = 「心部相对人部的语义拉力方向」
  对同一声旁组所有配对取平均 → 稳健的部首偏移向量

三部首产生最多 6 个方向：
  人→心  人→言  心→人  心→言  言→人  言→心

输出：
  data/processed/radical_shift_vectors.npz
  data/processed/radical_shift_report.txt
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── 加载向量索引 ──────────────────────────────────────────────

def load_embed_index() -> dict:
    """
    加载 embed_index.npz，返回 { character: { radical, phonetic, group_id, vector } }
    """
    path = os.path.join(config.DATA_OUT, "embed_index.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 {path}，请先运行 --step embed")
    data = np.load(path, allow_pickle=True)
    index = {}
    for i, char in enumerate(data["characters"]):
        index[str(char)] = {
            "radical":  str(data["radicals"][i]),
            "phonetic": str(data["phonetics"][i]),
            "group_id": str(data["group_ids"][i]) if "group_ids" in data else "",
            "vector":   data["vectors"][i],
            "idx":      i,
        }
    return index


# ── 计算偏移向量 ──────────────────────────────────────────────

def compute_shift_vectors(embed_index: dict, df_coords: pd.DataFrame) -> dict:
    """
    枚举 embed_index 中所有同 GroupID 的字对，
    按 (from_radical, to_radical) 收集偏移向量并取平均。

    返回：{ ("人部", "心部"): ndarray(768), ... }
    """
    # 按 GroupID 分组
    group_map = defaultdict(list)
    for char, info in embed_index.items():
        gid_rows = df_coords[df_coords["Character"] == char]
        if gid_rows.empty:
            continue
        gid = str(gid_rows.iloc[0]["GroupID"])
        group_map[gid].append((char, info["radical"], info["vector"]))

    raw: dict[tuple, list] = defaultdict(list)
    for gid, members in group_map.items():
        for i in range(len(members)):
            for j in range(len(members)):
                if i == j:
                    continue
                ci, ri, vi = members[i]
                cj, rj, vj = members[j]
                if ri == rj:
                    continue
                raw[(ri, rj)].append(vj - vi)

    mean_shifts = {}
    for key, vecs in raw.items():
        mean_shifts[key] = np.mean(vecs, axis=0)

    return mean_shifts


# ── 方向一致性评估 ────────────────────────────────────────────

def shift_consistency(raw_shifts_by_direction: dict) -> dict:
    """
    对每个方向计算各配对偏移向量之间的两两余弦相似度（一致性）。
    一致性高 → 该部首功能稳定；一致性低 → 部首功能受声旁影响大。
    """
    result = {}
    for key, vecs in raw_shifts_by_direction.items():
        if len(vecs) < 2:
            result[key] = {"n": len(vecs), "mean_cos": None, "std_cos": None}
            continue
        cos_sims = []
        for i in range(len(vecs)):
            for j in range(i+1, len(vecs)):
                a, b = vecs[i], vecs[j]
                cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                cos_sims.append(cos)
        result[key] = {
            "n":        len(vecs),
            "mean_cos": float(np.mean(cos_sims)),
            "std_cos":  float(np.std(cos_sims)),
        }
    return result


# ── 保存 ──────────────────────────────────────────────────────

def save_shift_vectors(mean_shifts: dict, consistency: dict):
    out_path = os.path.join(config.DATA_OUT, "radical_shift_vectors.npz")
    keys = list(mean_shifts.keys())
    np.savez(
        out_path,
        shift_vectors = np.array([mean_shifts[k] for k in keys]),
        from_radicals = np.array([k[0] for k in keys]),
        to_radicals   = np.array([k[1] for k in keys]),
    )

    report_lines = ["=" * 60, "  部首语义偏移向量报告（768维空间）", "=" * 60, ""]
    for key, vec in mean_shifts.items():
        r_from, r_to = key
        norm = float(np.linalg.norm(vec))
        cons = consistency.get(key, {})
        mc   = f"{cons['mean_cos']:.4f}" if cons.get("mean_cos") is not None else "样本不足"
        report_lines += [
            f"  {r_from} → {r_to}",
            f"    样本配对数：  {cons.get('n', '?')}",
            f"    偏移向量模长：{norm:.4f}  （越大语义拉力越强）",
            f"    方向一致性：  {mc}  （越接近1部首功能越稳定）",
            "",
        ]

    text = "\n".join(report_lines)
    rpt_path = os.path.join(config.DATA_OUT, "radical_shift_report.txt")
    with open(rpt_path, "w", encoding="utf_8_sig") as f:
        f.write(text)
    print(text)
    print(f"✅ 偏移向量 → {out_path}")
    print(f"✅ 报告     → {rpt_path}")


# ── 对外接口 ──────────────────────────────────────────────────

def load_shift_vectors() -> dict:
    """供 predict.py 调用：返回 { (from, to): ndarray(768) }"""
    path = os.path.join(config.DATA_OUT, "radical_shift_vectors.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 {path}，请先运行 --step radical_vectors")
    data = np.load(path, allow_pickle=True)
    result = {}
    for i in range(len(data["from_radicals"])):
        key = (str(data["from_radicals"][i]), str(data["to_radicals"][i]))
        result[key] = data["shift_vectors"][i]
    return result


# ── 主入口 ────────────────────────────────────────────────────

def run():
    print("📂 加载向量索引...")
    embed_index = load_embed_index()
    df_coords   = pd.read_csv(config.COORDS_CSV)

    radicals = set(v["radical"] for v in embed_index.values())
    print(f"   字符总数：{len(embed_index)}，已有部首：{radicals}")

    print("\n📐 计算部首偏移向量...")
    # 同时收集原始向量列表（用于一致性评估）
    group_map = defaultdict(list)
    for char, info in embed_index.items():
        gid_rows = df_coords[df_coords["Character"] == char]
        if gid_rows.empty:
            continue
        gid = str(gid_rows.iloc[0]["GroupID"])
        group_map[gid].append((char, info["radical"], info["vector"]))

    raw: dict[tuple, list] = defaultdict(list)
    for gid, members in group_map.items():
        for i in range(len(members)):
            for j in range(len(members)):
                if i == j:
                    continue
                _, ri, vi = members[i]
                _, rj, vj = members[j]
                if ri != rj:
                    raw[(ri, rj)].append(vj - vi)

    mean_shifts  = {k: np.mean(v, axis=0) for k, v in raw.items()}
    consistency  = shift_consistency(raw)

    if not mean_shifts:
        print("⚠️  未找到任何配对，请检查数据 GroupID 和 Radical 列")
        return

    print(f"   发现 {len(mean_shifts)} 个偏移方向：" +
          ", ".join(f"{a}→{b}" for a, b in mean_shifts))
    save_shift_vectors(mean_shifts, consistency)


if __name__ == "__main__":
    run()
