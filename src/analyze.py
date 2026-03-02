"""
src/analyze.py
职责：
  1. 计算每个 pair 的语义偏移向量（距离、角度）
  2. Rayleigh 检验（方向是否一致性显著）
  3. 部首间语义重叠度（质心距离、组内方差）
  4. 生成统计报告 + metrics.csv
"""

import os
import sys
import math
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── 向量统计 ──────────────────────────────────────────

def compute_pair_offsets(df: pd.DataFrame) -> pd.DataFrame:
    """
    对每个 PairID，计算从人部字 → 心部字的语义偏移向量。
    返回包含 Pair, Distance, Angle, dx, dy 的 DataFrame。
    """
    records = []
    for pair_id, group in df.groupby("PairID", sort=False):
        ren  = group[group["Radical"] == "人部"]
        xin  = group[group["Radical"] == "心部"]
        
        if ren.empty or xin.empty:
            continue
        

        r = ren.iloc[0]
        x = xin.iloc[0]
        
        dx = x["x"] - r["x"]
        dy = x["y"] - r["y"]
        dist = math.sqrt(dx**2 + dy**2)
        angle = math.degrees(math.atan2(dy, dx))
        
        records.append({
            "PairID":    pair_id,
            "Pair":      f"{r['Character']}({r['Radical']}) → {x['Character']}({x['Radical']})",
            "Char_Ren":  r["Character"],
            "Char_Xin":  x["Character"],
            "Phonetic":  r.get("Phonetic", str(pair_id)),
            "Distance":  dist,
            "Angle":     angle,
            "dx":        dx,
            "dy":        dy,
        })
    
    return pd.DataFrame(records)


def rayleigh_test(angles_deg: pd.Series) -> dict:
    """
    Rayleigh 检验：零假设 = 角度均匀分布（无定向）。
    p < 0.05 意味着部首引发的语义偏移有一致方向性。
    
    返回：
      R          : 平均结果向量长度（0-1），越接近 1 越一致
      p_value    : 显著性
      mean_angle : 平均偏移方向（度）
    """
    rad = np.radians(angles_deg)
    n = len(rad)
    C = np.cos(rad).mean()
    S = np.sin(rad).mean()
    R = math.sqrt(C**2 + S**2)          

    Z = n * R**2

    p = math.exp(-Z) * (1 + (2*Z - Z**2) / (4*n) - (24*Z - 132*Z**2 + 76*Z**3 - 9*Z**4) / (288*n**2))
    p = max(0.0, min(1.0, p))
    mean_angle = math.degrees(math.atan2(S, C))
    return {"R": R, "Z": Z, "p_value": p, "mean_angle": mean_angle}


def centroid_stats(df_coords: pd.DataFrame) -> dict:
    """
    计算各部首质心及组内方差，衡量语义疆域的内聚性。
    """
    result = {}
    for radical, group in df_coords.groupby("Radical"):
        cx, cy = group["x"].mean(), group["y"].mean()

        dists = np.sqrt((group["x"] - cx)**2 + (group["y"] - cy)**2)
        result[radical] = {
            "centroid": (cx, cy),
            "intra_dist_mean": dists.mean(),
            "intra_dist_std":  dists.std(),
            "n": len(group),
        }

    radicals = list(result.keys())
    if len(radicals) >= 2:
        c1 = np.array(result[radicals[0]]["centroid"])
        c2 = np.array(result[radicals[1]]["centroid"])
        result["centroid_distance"] = float(np.linalg.norm(c1 - c2))
    return result


def get_shift_level(dist: float) -> str:
    for lo, hi, label in config.SHIFT_LEVELS:
        if lo <= dist < hi:
            return label
    return "未知"


# ── 报告 ──────────────────────────────────────────────

def write_report(metrics_df: pd.DataFrame, rayleigh: dict, centroid: dict, path: str):
    lines = [
        "=" * 50,
        "  部首语义偏移量化统计报告",
        "=" * 50,
        "",
        "── 1. 偏移向量统计 ──────────────────",
        f"  总配对数:          {len(metrics_df)}",
        f"  平均偏移距离:      {metrics_df['Distance'].mean():.4f}",
        f"  偏移距离标准差:    {metrics_df['Distance'].std():.4f}",
        f"  平均偏移角度:      {metrics_df['Angle'].mean():.2f}°",
        f"  角度标准差:        {metrics_df['Angle'].std():.2f}°",
        "",
        "── 2. Rayleigh 方向一致性检验 ────────",
        f"  平均结果向量 R:    {rayleigh['R']:.4f}  (0→无规律, 1→完全一致)",
        f"  平均偏移方向:      {rayleigh['mean_angle']:.2f}°",
        f"  Rayleigh Z:        {rayleigh['Z']:.4f}",
        f"  p 值:              {rayleigh['p_value']:.4f}  "
        + ("← 方向一致性显著！" if rayleigh["p_value"] < 0.05 else "← 方向不显著，接近随机"),
        "",
        "── 3. 部首语义疆域 ─────────────────",
    ]
    
    for radical, stats_d in centroid.items():
        if radical == "centroid_distance":
            continue
        lines.append(f"  {radical}:")
        lines.append(f"    质心坐标:  ({stats_d['centroid'][0]:.4f}, {stats_d['centroid'][1]:.4f})")
        lines.append(f"    组内平均距质心距离: {stats_d['intra_dist_mean']:.4f}")
    
    if "centroid_distance" in centroid:
        lines.append(f"  人部-心部质心间距:  {centroid['centroid_distance']:.4f}")
    
    lines += [
        "",
        "── 4. 极端个案 ─────────────────────",
        "  语义最接近（人心部用法最相似）:",
    ]
    for _, row in metrics_df.nsmallest(3, "Distance").iterrows():
        lines.append(f"    {row['Pair']:<30}  Δ={row['Distance']:.4f}")
    
    lines.append("  语义差异最大（部首功能最显著）:")
    for _, row in metrics_df.nlargest(3, "Distance").iterrows():
        lines.append(f"    {row['Pair']:<30}  Δ={row['Distance']:.4f}")
    
    text = "\n".join(lines)
    with open(path, "w", encoding="utf_8_sig") as f:
        f.write(text)
    print(text)


# ── 主入口 ────────────────────────────────────────────

def run():
    df_coords = pd.read_csv(config.COORDS_CSV)
    
    print("📐 计算语义偏移向量...")
    metrics_df = compute_pair_offsets(df_coords)
    metrics_df["Shift_Level"] = metrics_df["Distance"].apply(get_shift_level)
    metrics_df_sorted = metrics_df.sort_values("Distance").reset_index(drop=True)
    metrics_df_sorted.to_csv(config.METRICS_CSV, index=False, encoding="utf_8_sig")
    
    print(" Rayleigh 检验...")
    rayleigh = rayleigh_test(metrics_df["Angle"])
    
    print(" 质心统计...")
    centroid = centroid_stats(df_coords)
    
    report_path = os.path.join(config.DATA_OUT, "statistical_report.txt")
    write_report(metrics_df_sorted, rayleigh, centroid, report_path)
    
    print(f"\n metrics → {config.METRICS_CSV}")
    print(f" report  → {report_path}")
    return metrics_df_sorted, rayleigh, centroid


if __name__ == "__main__":
    run()