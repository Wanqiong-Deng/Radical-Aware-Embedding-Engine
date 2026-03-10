"""
src/analyze.py
职责：
  1. 计算所有部首对 (A→B) 的语义偏移向量（距离、角度）
  2. Rayleigh 检验（每个方向是否具有统计显著的方向一致性）
  3. 各部首质心距离与组内方差
  4. 生成 metrics.csv + statistical_report.txt
  
  支持人/心/言 三个部首，自动枚举所有配对方向。
"""

import os
import sys
import math
import numpy as np
import pandas as pd
from itertools import permutations
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── 偏移向量计算 ──────────────────────────────────────────────

def compute_pair_offsets(df: pd.DataFrame) -> pd.DataFrame:
    """
    对每个 GroupID，计算所有部首两两之间的语义偏移。
    例如一个组有人/心/言三字，产生 人→心, 人→言, 心→言, 心→人, 言→人, 言→心 六条记录。
    """
    records = []
    for gid, group in df.groupby("GroupID", sort=False):
        members = group.reset_index(drop=True)
        for i, row_from in members.iterrows():
            for j, row_to in members.iterrows():
                if i == j or row_from["Radical"] == row_to["Radical"]:
                    continue
                dx = row_to["x"] - row_from["x"]
                dy = row_to["y"] - row_from["y"]
                dist  = math.sqrt(dx**2 + dy**2)
                angle = math.degrees(math.atan2(dy, dx))
                records.append({
                    "GroupID":     gid,
                    "Phonetic":    str(row_from.get("Phonetic", gid)),
                    "Pair":        f"{row_from['Character']}({row_from['Radical']}) → {row_to['Character']}({row_to['Radical']})",
                    "Char_From":   row_from["Character"],
                    "Char_To":     row_to["Character"],
                    "Radical_From":row_from["Radical"],
                    "Radical_To":  row_to["Radical"],
                    "Direction":   f"{row_from['Radical']} → {row_to['Radical']}",
                    "Distance":    dist,
                    "Angle":       angle,
                    "dx":          dx,
                    "dy":          dy,
                })
    return pd.DataFrame(records)


# ── Rayleigh 检验（每个方向独立检验）────────────────────────

def rayleigh_test(angles_deg: pd.Series) -> dict:
    """
    Rayleigh 检验：角度是否具有显著方向一致性。
    R: mean resultant length (0=随机, 1=完全一致)
    """
    if len(angles_deg) < 3:
        return {"R": None, "Z": None, "p_value": None, "mean_angle": None, "n": len(angles_deg)}
    rad = np.radians(angles_deg)
    n   = len(rad)
    C, S = np.cos(rad).mean(), np.sin(rad).mean()
    R    = math.sqrt(C**2 + S**2)
    Z    = n * R**2
    p    = math.exp(-Z) * (1 + (2*Z - Z**2) / (4*n) - (24*Z - 132*Z**2 + 76*Z**3 - 9*Z**4) / (288*n**2))
    p    = float(np.clip(p, 0.0, 1.0))
    return {
        "R":          R,
        "Z":          Z,
        "p_value":    p,
        "mean_angle": math.degrees(math.atan2(S, C)),
        "n":          n,
    }


def rayleigh_by_direction(metrics_df: pd.DataFrame) -> dict:
    """对每个偏移方向独立做 Rayleigh 检验，返回 { "人部→心部": {...}, ... }"""
    result = {}
    for direction, sub in metrics_df.groupby("Direction"):
        result[direction] = rayleigh_test(sub["Angle"])
    return result


# ── 质心统计 ──────────────────────────────────────────────────

def centroid_stats(df_coords: pd.DataFrame) -> dict:
    result = {}
    radical_groups = df_coords.groupby("Radical")
    
    for radical, group in radical_groups:
        cx, cy = group["x"].mean(), group["y"].mean()
        dists  = np.sqrt((group["x"] - cx)**2 + (group["y"] - cy)**2)
        result[radical] = {
            "centroid":         (cx, cy),
            "intra_dist_mean":  float(dists.mean()),
            "intra_dist_std":   float(dists.std()),
            "n":                len(group),
        }
    
    # 所有部首两两质心距
    radicals = list(result.keys())
    for i in range(len(radicals)):
        for j in range(i+1, len(radicals)):
            r1, r2 = radicals[i], radicals[j]
            c1 = np.array(result[r1]["centroid"])
            c2 = np.array(result[r2]["centroid"])
            key = f"{r1} ↔ {r2}"
            result[key] = float(np.linalg.norm(c1 - c2))
    
    return result


def get_shift_level(dist: float) -> str:
    for lo, hi, label in config.SHIFT_LEVELS:
        if lo <= dist < hi:
            return label
    return "未知"


# ── 报告 ──────────────────────────────────────────────────────

def write_report(metrics_df, rayleigh_dict, centroid, path):
    lines = [
        "=" * 60,
        "  部首语义偏移量化统计报告（人 / 心 / 言 三部首）",
        "=" * 60, "",
        "── 1. 总体偏移统计 ─────────────────────────────",
        f"  总配对数:       {len(metrics_df)}",
        f"  全局平均偏移距: {metrics_df['Distance'].mean():.4f}",
        f"  全局偏移距离SD: {metrics_df['Distance'].std():.4f}",
        "",
        "── 2. 各方向 Rayleigh 检验 ──────────────────────",
    ]

    for direction, r in rayleigh_dict.items():
        if r["R"] is None:
            lines.append(f"  {direction:20s}  样本不足（n={r['n']}）")
            continue
        sig = "★ 显著" if (r["p_value"] or 1) < 0.05 else "  不显著"
        lines.append(
            f"  {direction:20s}  R={r['R']:.4f}  p={r['p_value']:.4f}  "
            f"均值方向={r['mean_angle']:.1f}°  {sig}"
        )

    lines += ["", "── 3. 部首质心分析 ──────────────────────────────"]
    for radical in config.ALL_RADICALS:
        if radical not in centroid:
            continue
        s = centroid[radical]
        lines.append(f"  {radical}  质心=({s['centroid'][0]:.4f}, {s['centroid'][1]:.4f})"
                     f"  组内均距={s['intra_dist_mean']:.4f}")

    lines += [""]
    for key, val in centroid.items():
        if "↔" in str(key):
            lines.append(f"  质心间距 {key}: {val:.4f}")

    lines += ["", "── 4. 偏移极端个案 ──────────────────────────────"]
    for direction, sub in metrics_df.groupby("Direction"):
        lines.append(f"\n  {direction}:")
        lines.append("    最相似（部首影响微弱）:")
        for _, row in sub.nsmallest(2, "Distance").iterrows():
            lines.append(f"      {row['Pair']:<40} Δ={row['Distance']:.4f}")
        lines.append("    差异最大（部首影响显著）:")
        for _, row in sub.nlargest(2, "Distance").iterrows():
            lines.append(f"      {row['Pair']:<40} Δ={row['Distance']:.4f}")

    text = "\n".join(lines)
    with open(path, "w", encoding="utf_8_sig") as f:
        f.write(text)
    print(text)


# ── 主入口 ────────────────────────────────────────────────────

def run():
    df_coords = pd.read_csv(config.COORDS_CSV)

    print("📐 计算语义偏移向量（所有部首对）...")
    metrics_df = compute_pair_offsets(df_coords)
    metrics_df["Shift_Level"] = metrics_df["Distance"].apply(get_shift_level)
    metrics_df_sorted = metrics_df.sort_values(["Direction", "Distance"]).reset_index(drop=True)
    metrics_df_sorted.to_csv(config.METRICS_CSV, index=False, encoding="utf_8_sig")

    print("\n📊 Rayleigh 检验（各方向独立）...")
    rayleigh_dict = rayleigh_by_direction(metrics_df)

    print("🎯 质心统计...")
    centroid = centroid_stats(df_coords)

    report_path = os.path.join(config.DATA_OUT, "statistical_report.txt")
    write_report(metrics_df_sorted, rayleigh_dict, centroid, report_path)

    print(f"\n✅ metrics → {config.METRICS_CSV}")
    print(f"✅ report  → {report_path}")
    return metrics_df_sorted, rayleigh_dict, centroid


if __name__ == "__main__":
    run()
