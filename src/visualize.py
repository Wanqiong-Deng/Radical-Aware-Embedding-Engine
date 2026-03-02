"""
src/visualize.py
职责：生成所有图表
  - scatter_semantic_space()   : 语义空间散点图 + 偏移箭头
  - polar_rose()               : 偏移角度玫瑰图
  - density_comparison()       : KDE 密度对比图
  - offset_ranking_bar()       : 偏移强度排名条形图（新增）
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from adjustText import adjust_text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


OUT = config.DATA_OUT


def scatter_semantic_space(df_coords: pd.DataFrame, df_metrics: pd.DataFrame):
    """
    主图：PCA 语义空间散点图。
    人部字和心部字用颜色区分，同声旁的对用箭头连接，显示偏移方向。
    """
    fig, ax = plt.subplots(figsize=(13, 10))
    
    palette = config.RADICAL_PALETTE
    
    # 底层：KDE 密度晕
    for radical, color in palette.items():
        sub = df_coords[df_coords["Radical"] == radical]
        if len(sub) > 3:
            sns.kdeplot(data=sub, x="x", y="y", ax=ax,
                        fill=True, alpha=0.08, color=color, bw_adjust=0.8)
    
    # 散点
    for radical, color in palette.items():
        sub = df_coords[df_coords["Radical"] == radical]
        ax.scatter(sub["x"], sub["y"], c=color, s=90, zorder=3,
                   alpha=0.85, edgecolors="white", linewidths=0.5,
                   label=radical)
    
    # 偏移箭头（按配对连线）
    for _, row in df_metrics.iterrows():
        ren = df_coords[(df_coords["Character"] == row["Char_Ren"]) &
                        (df_coords["Radical"] == "人部")]
        xin = df_coords[(df_coords["Character"] == row["Char_Xin"]) &
                        (df_coords["Radical"] == "心部")]
        if ren.empty or xin.empty:
            continue
        x0, y0 = ren.iloc[0]["x"], ren.iloc[0]["y"]
        x1, y1 = xin.iloc[0]["x"], xin.iloc[0]["y"]
        
        # 箭头颜色按偏移强度
        dist = row["Distance"]
        alpha = min(0.9, 0.3 + dist / df_metrics["Distance"].max() * 0.6)
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="#888888",
                                   lw=0.8, alpha=alpha))
    
    # 标签（adjustText 防重叠）
    labels = []
    for _, row in df_coords.iterrows():
        t = ax.text(row["x"], row["y"], row["Character"],
                    fontsize=10, ha="center", va="bottom", zorder=4)
        labels.append(t)
    adjust_text(labels, ax=ax, arrowprops=dict(arrowstyle="-", color="gray",
                                               lw=0.4, alpha=0.5))
    
    ax.set_title("人部 vs 心部：BGE 语义空间与部首偏移向量", fontsize=14, pad=16)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)
    
    path = os.path.join(OUT, "scatter_semantic_space.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  散点图 → {path}")
    return path


def polar_rose(df_metrics: pd.DataFrame):
    """偏移角度玫瑰图：展示部首语义"拉力"的方向分布。"""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    
    angles_rad = np.radians(df_metrics["Angle"])
    bins = 24
    n, edges = np.histogram(angles_rad, bins=bins, range=(-np.pi, np.pi))
    width = 2 * np.pi / bins
    
    # 按强度着色
    norm_n = n / n.max() if n.max() > 0 else n
    colors = plt.cm.Reds(0.3 + 0.7 * norm_n)
    
    ax.bar(edges[:-1], n, width=width, color=colors,
           edgecolor="white", linewidth=0.5, alpha=0.85)
    
    # 标注平均方向
    vec_x = np.cos(angles_rad).mean()
    vec_y = np.sin(angles_rad).mean()
    mean_angle = math.atan2(vec_y, vec_x)
    R = math.sqrt(vec_x**2 + vec_y**2)
    ax.annotate("", xy=(mean_angle, R * n.max()),
                xytext=(0, 0), xycoords="data",
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=2))
    
    ax.set_title("心部相对人部的语义偏移方向分布\n（箭头=平均方向，R={:.3f}）".format(R),
                 va="bottom", fontsize=12)
    ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
    ax.set_xticklabels(["0°", "45°", "90°", "135°", "±180°", "-135°", "-90°", "-45°"])
    
    path = os.path.join(OUT, "polar_rose.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" 玫瑰图 → {path}")
    return path


def density_comparison(df_coords: pd.DataFrame):
    """KDE 密度对比图：两个部首的语义覆盖区域。"""
    fig, ax = plt.subplots(figsize=(10, 7))
    palette = config.RADICAL_PALETTE
    
    for radical, color in palette.items():
        sub = df_coords[df_coords["Radical"] == radical]
        if len(sub) > 3:
            sns.kdeplot(data=sub, x="x", y="y", ax=ax, fill=True,
                        alpha=0.3, color=color, label=f"{radical} 密度")
        ax.scatter(sub["x"], sub["y"], c=color, s=30, alpha=0.6)
    
    ax.set_title("人部 vs 心部 语义密度与覆盖场对比", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.2)
    
    path = os.path.join(OUT, "density_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"🗺  密度图 → {path}")
    return path


def offset_ranking_bar(df_metrics: pd.DataFrame):
    """
    新增：横向条形图，直观展示每个声旁组的偏移强度排名。
    这是论文插图的好选择，也是 GitHub README 的好展示图。
    """
    df_sorted = df_metrics.sort_values("Distance", ascending=True)
    
    # 颜色按偏移等级
    level_colors = {"微弱变动": "#74c476", "显著变动": "#fd8d3c", "剧烈变动": "#d62728"}
    colors = df_sorted["Shift_Level"].map(level_colors).tolist()
    
    fig, ax = plt.subplots(figsize=(10, max(5, len(df_sorted) * 0.45)))
    bars = ax.barh(df_sorted["Pair"], df_sorted["Distance"],
                   color=colors, edgecolor="white", linewidth=0.5)
    
    # 数值标签
    for bar, val in zip(bars, df_sorted["Distance"]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8.5)
    
    # 图例
    legend_patches = [mpatches.Patch(color=c, label=l)
                      for l, c in level_colors.items()]
    ax.legend(handles=legend_patches, fontsize=9, loc="lower right")
    
    ax.set_xlabel("语义偏移距离（BGE 向量空间）")
    ax.set_title("各声旁组：心部对人部的语义偏移强度排名", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    
    path = os.path.join(OUT, "offset_ranking.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" 排名图 → {path}")
    return path


def run():
    df_coords  = pd.read_csv(config.COORDS_CSV)
    df_metrics = pd.read_csv(config.METRICS_CSV)
    
    scatter_semantic_space(df_coords, df_metrics)
    polar_rose(df_metrics)
    density_comparison(df_coords)
    offset_ranking_bar(df_metrics)
    
    print("\n 所有图表已生成。")


if __name__ == "__main__":
    run()