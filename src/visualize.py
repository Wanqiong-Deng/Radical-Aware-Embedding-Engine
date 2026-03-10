"""
src/visualize.py
职责：生成所有静态图表（全部支持人/心/言三部首）
  - scatter_semantic_space()  语义空间散点图 + 偏移箭头（按方向分色）
  - polar_rose_by_direction() 每个偏移方向独立玫瑰图（2x3 子图）
  - density_comparison()      三部首 KDE 密度对比图
  - offset_ranking_bar()      按方向分组的偏移强度排名
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from adjustText import adjust_text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

OUT = config.DATA_OUT


# ── 散点图 ────────────────────────────────────────────────────

def scatter_semantic_space(df_coords: pd.DataFrame, df_metrics: pd.DataFrame):
    """
    PCA 语义空间散点图。
    三个部首各一色，箭头颜色按偏移方向（如人→心用一种颜色）。
    """
    palette = config.RADICAL_PALETTE
    # 每个偏移方向的箭头颜色（淡化版本）
    arrow_palette = {
        "人部 → 心部": "#aec7e8",
        "人部 → 言部": "#98df8a",
        "心部 → 人部": "#ffbb78",
        "心部 → 言部": "#ff9896",
        "言部 → 人部": "#c5b0d5",
        "言部 → 心部": "#c49c94",
    }

    fig, ax = plt.subplots(figsize=(14, 10))

    # KDE 底层
    for radical, color in palette.items():
        sub = df_coords[df_coords["Radical"] == radical]
        if len(sub) > 3:
            sns.kdeplot(data=sub, x="x", y="y", ax=ax,
                        fill=True, alpha=0.07, color=color, bw_adjust=0.8)

    # 散点
    for radical, color in palette.items():
        sub = df_coords[df_coords["Radical"] == radical]
        ax.scatter(sub["x"], sub["y"], c=color, s=90, zorder=3,
                   alpha=0.85, edgecolors="white", linewidths=0.5, label=radical)

    # 偏移箭头
    for _, row in df_metrics.iterrows():
        from_row = df_coords[
            (df_coords["Character"] == row["Char_From"]) &
            (df_coords["Radical"]   == row["Radical_From"])
        ]
        to_row = df_coords[
            (df_coords["Character"] == row["Char_To"]) &
            (df_coords["Radical"]   == row["Radical_To"])
        ]
        if from_row.empty or to_row.empty:
            continue
        x0, y0 = float(from_row.iloc[0]["x"]), float(from_row.iloc[0]["y"])
        x1, y1 = float(to_row.iloc[0]["x"]),   float(to_row.iloc[0]["y"])
        color = arrow_palette.get(row["Direction"], "#cccccc")
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.7, alpha=0.55))

    # 标签
    labels = []
    for _, row in df_coords.iterrows():
        t = ax.text(row["x"], row["y"], row["Character"],
                    fontsize=10, ha="center", va="bottom", zorder=4)
        labels.append(t)
    adjust_text(labels, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.4, alpha=0.4))

    ax.set_title("人部 / 心部 / 言部：BGE 语义空间与部首偏移向量", fontsize=14, pad=14)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)

    path = os.path.join(OUT, "scatter_semantic_space.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"🖼  散点图 → {path}")
    return path


# ── 玫瑰图（每方向一个子图）─────────────────────────────────

def polar_rose_by_direction(df_metrics: pd.DataFrame):
    """
    6 个偏移方向各一个玫瑰子图，2 行 × 3 列布局。
    """
    directions = sorted(df_metrics["Direction"].unique())
    ncols = 3
    nrows = math.ceil(len(directions) / ncols)

    fig = plt.figure(figsize=(15, 5 * nrows))
    fig.suptitle("各偏移方向的角度分布玫瑰图", fontsize=14, y=1.01)

    for idx, direction in enumerate(directions):
        sub = df_metrics[df_metrics["Direction"] == direction]
        ax  = fig.add_subplot(nrows, ncols, idx + 1, projection="polar")

        angles_rad = np.radians(sub["Angle"])
        n_bins = min(24, max(8, len(sub) * 2))
        n, edges = np.histogram(angles_rad, bins=n_bins, range=(-np.pi, np.pi))
        width = 2 * np.pi / n_bins
        norm_n = n / max(n.max(), 1)
        colors = plt.cm.Reds(0.3 + 0.7 * norm_n)
        ax.bar(edges[:-1], n, width=width, color=colors, edgecolor="white", linewidth=0.4, alpha=0.85)

        # 平均方向箭头
        if len(angles_rad) >= 2:
            C, S = np.cos(angles_rad).mean(), np.sin(angles_rad).mean()
            R    = math.sqrt(C**2 + S**2)
            mean_angle = math.atan2(S, C)
            ax.annotate("", xy=(mean_angle, R * n.max()),
                        xytext=(0, 0), xycoords="data",
                        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=2))

        ax.set_title(f"{direction}\n(n={len(sub)})", va="bottom", fontsize=10)
        ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
        ax.set_xticklabels(["0°","45°","90°","135°","±180°","-135°","-90°","-45°"], fontsize=7)

    plt.tight_layout()
    path = os.path.join(OUT, "polar_rose.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"🌹 玫瑰图 → {path}")
    return path


# ── 密度对比图 ────────────────────────────────────────────────

def density_comparison(df_coords: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 8))
    palette = config.RADICAL_PALETTE

    for radical, color in palette.items():
        sub = df_coords[df_coords["Radical"] == radical]
        if len(sub) > 3:
            sns.kdeplot(data=sub, x="x", y="y", ax=ax, fill=True,
                        alpha=0.25, color=color, label=f"{radical} 密度")
        ax.scatter(sub["x"], sub["y"], c=color, s=35, alpha=0.65)

    ax.set_title("人部 / 心部 / 言部 语义密度与覆盖场对比", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.2)

    path = os.path.join(OUT, "density_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"🗺  密度图 → {path}")
    return path


# ── 偏移强度排名（按方向分面）───────────────────────────────

def offset_ranking_bar(df_metrics: pd.DataFrame):
    """
    按偏移方向分组，每个方向一个子图，横向条形图排名。
    """
    directions = sorted(df_metrics["Direction"].unique())
    ncols = min(3, len(directions))
    nrows = math.ceil(len(directions) / ncols)

    level_colors = {
        "微弱变动": "#74c476",
        "显著变动": "#fd8d3c",
        "剧烈变动": "#d62728",
    }

    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, max(4, len(df_metrics) // ncols * 0.45)))
    if len(directions) == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]

    for idx, direction in enumerate(directions):
        r, c = divmod(idx, ncols)
        ax   = axes[r][c]
        sub  = df_metrics[df_metrics["Direction"] == direction].sort_values("Distance")
        colors = sub["Shift_Level"].map(level_colors).tolist()
        bars = ax.barh(sub["Phonetic"], sub["Distance"],
                       color=colors, edgecolor="white", linewidth=0.4)
        for bar, val in zip(bars, sub["Distance"]):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=7.5)
        ax.set_title(direction, fontsize=11)
        ax.set_xlabel("偏移距离")
        ax.grid(axis="x", alpha=0.3)

    # 隐藏多余子图
    for idx in range(len(directions), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    # 图例
    patches = [mpatches.Patch(color=v, label=k) for k, v in level_colors.items()]
    fig.legend(handles=patches, fontsize=9, loc="lower right")
    fig.suptitle("各声旁组偏移强度排名（按偏移方向分组）", fontsize=13)
    plt.tight_layout()

    path = os.path.join(OUT, "offset_ranking.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 排名图 → {path}")
    return path


# ── 主入口 ────────────────────────────────────────────────────

def run():
    df_coords  = pd.read_csv(config.COORDS_CSV)
    df_metrics = pd.read_csv(config.METRICS_CSV)

    scatter_semantic_space(df_coords, df_metrics)
    polar_rose_by_direction(df_metrics)
    density_comparison(df_coords)
    offset_ranking_bar(df_metrics)
    print("\n✅ 所有图表已生成。")


if __name__ == "__main__":
    run()
