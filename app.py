"""
app.py — Streamlit 交互 Demo
这是挂在 GitHub 上展示用的核心文件

运行：streamlit run app.py
"""

import os
import sys
import pickle
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ── 页面配置 ──────────────────────────────────────────
st.set_page_config(
    page_title="部首语义指南针",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧭 部首语义指南针")
st.markdown(
    """
    **Research Question**: 相同声旁、不同部首的汉字，部首如何在语义空间中"拉动"字义？
    
    本工具基于 [BGE-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5) 
    将《说文解字》与《汉语大字典》的训诂文本嵌入高维语义空间，
    通过 PCA 降维可视化**人部 vs 心部**的语义版图与偏移向量。
    """
)

# ── 数据加载（缓存） ──────────────────────────────────

@st.cache_data
def load_data():
    coords   = pd.read_csv(config.COORDS_CSV)
    metrics  = pd.read_csv(config.METRICS_CSV)
    return coords, metrics

@st.cache_resource
def load_pca():
    path = os.path.join(config.DATA_OUT, "pca_model.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


# 检查数据是否存在
if not os.path.exists(config.COORDS_CSV):
    st.error("⚠️ 数据文件不存在，请先运行 `python run_pipeline.py`")
    st.stop()

df_coords, df_metrics = load_data()
pca = load_pca()

# ── 侧边栏控件 ────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 显示设置")
    
    show_kde = st.checkbox("显示密度晕", value=True)
    show_arrows = st.checkbox("显示偏移箭头", value=True)
    show_labels = st.checkbox("显示字形标签", value=True)
    
    st.divider()
    st.header("🔍 筛选")
    
    all_radicals = sorted(df_coords["Radical"].unique())
    selected_radicals = st.multiselect(
        "显示部首", all_radicals, default=all_radicals
    )
    
    min_dist, max_dist = float(df_metrics["Distance"].min()), float(df_metrics["Distance"].max())
    dist_range = st.slider(
        "偏移距离范围", min_dist, max_dist,
        (min_dist, max_dist), step=0.01, format="%.3f"
    )
    
    st.divider()
    st.caption("👨‍🎓 NLP + 古文字学 毕业设计")
    st.caption("模型: BAAI/bge-base-zh-v1.5")


# ── 主区域：Tab 布局 ──────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📍 语义空间", "🌹 方向分析", "📊 偏移排名", "📋 数据表"])


# ── Tab 1: 语义散点图 ─────────────────────────────────
with tab1:
    st.subheader("语义空间散点图")
    st.caption('同声旁的人部字与心部字用箭头相连，箭头方向和长度反映部首的语义"拉力"')
    
    # 过滤
    df_show = df_coords[df_coords["Radical"].isin(selected_radicals)]
    df_met_show = df_metrics[
        (df_metrics["Distance"] >= dist_range[0]) &
        (df_metrics["Distance"] <= dist_range[1])
    ]
    
    palette = config.RADICAL_PALETTE
    
    fig = go.Figure()
    
    # 散点
    for radical in selected_radicals:
        sub = df_show[df_show["Radical"] == radical]
        color = palette.get(radical, "#888888")
        hover = (sub["Character"] + "<br>" +
                 "說文: " + sub.get("Shuowen_clean", sub.get("Shuowen", "")).fillna("") +
                 "<br>大字典: " + sub.get("Dazidian_clean", sub.get("Dazidian", "")).fillna(""))
        fig.add_trace(go.Scatter(
            x=sub["x"], y=sub["y"],
            mode="markers+text" if show_labels else "markers",
            name=radical,
            marker=dict(color=color, size=12, opacity=0.85,
                        line=dict(width=1, color="white")),
            text=sub["Character"] if show_labels else None,
            textposition="top center",
            textfont=dict(size=13),
            hovertext=hover,
            hoverinfo="text",
        ))
    
    # 偏移箭头
    if show_arrows:
        for _, row in df_met_show.iterrows():
            ren = df_coords[(df_coords["Character"] == row["Char_Ren"]) &
                            (df_coords["Radical"] == "人部")]
            xin = df_coords[(df_coords["Character"] == row["Char_Xin"]) &
                            (df_coords["Radical"] == "心部")]
            if ren.empty or xin.empty:
                continue
            x0, y0 = float(ren.iloc[0]["x"]), float(ren.iloc[0]["y"])
            x1, y1 = float(xin.iloc[0]["x"]), float(xin.iloc[0]["y"])
            
            # 按偏移强度决定箭头颜色
            ratio = (row["Distance"] - dist_range[0]) / max(dist_range[1] - dist_range[0], 1e-9)
            r_val = int(200 * ratio)
            arrow_color = f"rgba({r_val}, 100, {200 - r_val}, 0.6)"
            
            fig.add_annotation(
                x=x1, y=y1, ax=x0, ay=y0,
                axref="x", ayref="y",
                arrowhead=2, arrowsize=1, arrowwidth=1.5,
                arrowcolor=arrow_color,
                showarrow=True,
                hovertext=f"{row['Pair']}<br>Δ={row['Distance']:.4f}",
            )
    
    fig.update_layout(
        height=620,
        plot_bgcolor="#f9f9f9",
        legend=dict(orientation="h", y=-0.1),
        xaxis_title="PC1",
        yaxis_title="PC2",
        hoverlabel=dict(bgcolor="white", font_size=12),
        margin=dict(l=0, r=0, t=20, b=0),
    )
    
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 2: 方向玫瑰图 + Rayleigh ──────────────────────
with tab2:
    st.subheader("语义偏移方向分析")
    
    col_l, col_r = st.columns([1, 1])
    
    with col_l:
        # Rayleigh 统计卡片
        angles_rad = np.radians(df_met_show["Angle"])
        n = len(angles_rad)
        if n > 0:
            C = np.cos(angles_rad).mean()
            S = np.sin(angles_rad).mean()
            R = math.sqrt(C**2 + S**2)
            Z = n * R**2
            p = math.exp(-Z) * (1 + (2*Z - Z**2) / (4*max(n,1)))
            p = max(0.0, min(1.0, p))
            mean_angle = math.degrees(math.atan2(S, C))
            
            st.metric("平均结果向量 R", f"{R:.4f}",
                      help="0=完全随机方向, 1=所有偏移方向完全一致")
            st.metric("平均偏移方向", f"{mean_angle:.1f}°")
            st.metric("Rayleigh p 值", f"{p:.4f}",
                      delta="显著 < 0.05" if p < 0.05 else "不显著",
                      delta_color="normal" if p < 0.05 else "off")
            
            st.info(
                "**Rayleigh 检验**：检验偏移角度是否具有显著方向性（非随机）。\n\n"
                f"- R = {R:.4f}，{'方向较一致' if R > 0.5 else '方向较分散'}\n"
                f"- p {'< 0.05，部首引发的语义偏移有统计上显著的定向性' if p < 0.05 else '> 0.05，尚未达到统计显著性（可能样本量不足）'}"
            )
    
    with col_r:
        # Plotly 极坐标玫瑰图
        angles_deg = df_met_show["Angle"].tolist()
        with col_r:
                # 修正：使用 bar_polar 替代已移除的 histogram_polar
                # 将角度按 15 度一个区间进行计数 (24 bins)
                df_met_show['angle_bin'] = (df_met_show['Angle'] // 15) * 15
                df_bins = df_met_show.groupby('angle_bin').size().reset_index(name='count')
                
                fig_polar = px.bar_polar(
                    df_bins, 
                    r="count", 
                    theta="angle_bin",
                    template="plotly_white",
                    color_discrete_sequence=["#d62728"],
                    title="偏移角度分布 (玫瑰图)",
                    start_angle=0,
                    direction="clockwise"
                )
                
                fig_polar.update_layout(height=400, margin=dict(t=50, b=20, l=20, r=20))
        st.plotly_chart(fig_polar, use_container_width=True)


# ── Tab 3: 偏移排名 ──────────────────────────────────
with tab3:
    st.subheader("声旁组偏移强度排名")
    st.caption("距离越大，说明加上不同部首后，字义的语义差距越大（部首功能越显著）")
    
    level_color = {"微弱变动": "green", "显著变动": "orange", "剧烈变动": "red"}
    
    fig_bar = px.bar(
        df_met_show.sort_values("Distance"),
        x="Distance", y="Pair",
        orientation="h",
        color="Shift_Level",
        color_discrete_map=level_color,
        text=df_met_show.sort_values("Distance")["Distance"].apply(lambda v: f"{v:.4f}"),
        labels={"Distance": "语义偏移距离", "Pair": "声旁组"},
        height=max(400, len(df_met_show) * 38),
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(
        plot_bgcolor="#f9f9f9",
        legend_title="偏移等级",
        margin=dict(l=0, r=60, t=20, b=0),
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ── Tab 4: 数据表 ─────────────────────────────────────
with tab4:
    st.subheader("原始数据与指标")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**坐标数据**")
        cols_show = [c for c in ["Character", "Radical", "Shuowen_clean",
                                  "Dazidian_clean", "x", "y"] if c in df_coords.columns]
        st.dataframe(df_coords[cols_show], use_container_width=True, height=400)
    
    with col_b:
        st.markdown("**偏移指标**")
        st.dataframe(df_metrics, use_container_width=True, height=400)
    
    st.download_button(
        "⬇️ 下载偏移指标 CSV",
        df_metrics.to_csv(index=False, encoding="utf_8_sig").encode("utf_8_sig"),
        "metrics.csv",
        "text/csv",
    )
