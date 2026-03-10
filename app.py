"""
app.py — Streamlit 交互 Demo
四个分析 Tab + 第五个预测 Tab

运行：streamlit run app.py
"""

import os
import sys
import math
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ── 页面配置 ──────────────────────────────────────────────────
st.set_page_config(
    page_title="GlyphDrift · 部首语义指南针",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧭 GlyphDrift · 部首语义指南针")
st.markdown(
    """
    **Research Question**: 相同声旁、不同部首（人 / 心 / 言）的汉字，
    部首如何在语义空间中「拉动」字义方向？

    基于 [BGE-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5) 
    对《说文解字》与《汉语大字典》训诂文本的密集向量表示，
    结合 Rayleigh 检验量化部首偏移的方向一致性。
    """
)

# ── 数据加载 ──────────────────────────────────────────────────

@st.cache_data
def load_data():
    if not os.path.exists(config.COORDS_CSV):
        return None, None
    coords  = pd.read_csv(config.COORDS_CSV)
    metrics = pd.read_csv(config.METRICS_CSV)
    return coords, metrics


@st.cache_resource
def load_predictor():
    try:
        from src.predict import RadicalPredictor
        return RadicalPredictor.load(use_llm=True)
    except Exception as e:
        st.warning(f"预测器加载失败（可能数据尚未生成）：{e}")
        return None


df_coords, df_metrics = load_data()

if df_coords is None:
    st.error("⚠️ 数据文件不存在，请先运行 `python run_pipeline.py`")
    st.stop()

# ── 侧边栏 ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 显示设置")
    show_arrows = st.checkbox("显示偏移箭头", value=True)
    show_labels = st.checkbox("显示字形标签", value=True)
    show_kde    = st.checkbox("显示密度晕", value=False)

    st.divider()
    st.header("🔍 筛选")

    all_radicals = sorted(df_coords["Radical"].unique())
    selected_radicals = st.multiselect("显示部首", all_radicals, default=all_radicals)

    all_directions = sorted(df_metrics["Direction"].unique()) if df_metrics is not None else []
    selected_directions = st.multiselect("偏移方向", all_directions, default=all_directions)

    min_d = float(df_metrics["Distance"].min())
    max_d = float(df_metrics["Distance"].max())
    dist_range = st.slider("偏移距离范围", min_d, max_d, (min_d, max_d),
                           step=0.01, format="%.3f")

    st.divider()
    st.caption("📦 模型: BAAI/bge-base-zh-v1.5")
    st.caption("🤖 LLM: Qwen via SiliconFlow")

# ── 过滤数据 ──────────────────────────────────────────────────
df_show = df_coords[df_coords["Radical"].isin(selected_radicals)]
df_met_show = df_metrics[
    df_metrics["Direction"].isin(selected_directions) &
    df_metrics["Distance"].between(*dist_range)
] if df_metrics is not None else pd.DataFrame()

palette = config.RADICAL_PALETTE

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📍 语义空间", "🌹 方向分析", "📊 偏移排名", "📋 数据表", "🔮 字义预测"
])


# ════════════════════════════════════════════════════════════
# Tab 1：语义空间散点图
# ════════════════════════════════════════════════════════════
with tab1:
    st.subheader("语义空间散点图")
    st.caption("同声旁的字用箭头相连，箭头方向和长度反映部首的语义「拉力」")

    fig = go.Figure()

    # 散点（每个部首一个 trace）
    for radical in selected_radicals:
        sub   = df_show[df_show["Radical"] == radical]
        color = palette.get(radical, "#888888")
        hover = (
            sub["Character"] + "<br>" +
            sub.get("Radical", "").fillna("") + "<br>" +
            "說文: " + sub.get("Shuowen_clean", sub.get("Shuowen", "")).fillna("") + "<br>" +
            "大字典: " + sub.get("Dazidian_clean", sub.get("Dazidian", "")).fillna("")
        )
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
        direction_colors = {
            "人部 → 心部": "rgba(100,120,220,0.5)",
            "人部 → 言部": "rgba(80,180,80,0.5)",
            "心部 → 人部": "rgba(220,100,80,0.5)",
            "心部 → 言部": "rgba(220,140,80,0.5)",
            "言部 → 人部": "rgba(140,80,200,0.5)",
            "言部 → 心部": "rgba(200,80,140,0.5)",
        }
        for _, row in df_met_show.iterrows():
            from_r = df_coords[
                (df_coords["Character"] == row["Char_From"]) &
                (df_coords["Radical"]   == row["Radical_From"])
            ]
            to_r = df_coords[
                (df_coords["Character"] == row["Char_To"]) &
                (df_coords["Radical"]   == row["Radical_To"])
            ]
            if from_r.empty or to_r.empty:
                continue
            x0, y0 = float(from_r.iloc[0]["x"]), float(from_r.iloc[0]["y"])
            x1, y1 = float(to_r.iloc[0]["x"]),   float(to_r.iloc[0]["y"])
            color  = direction_colors.get(row["Direction"], "rgba(150,150,150,0.4)")
            fig.add_annotation(
                x=x1, y=y1, ax=x0, ay=y0,
                axref="x", ayref="y",
                arrowhead=2, arrowsize=1, arrowwidth=1.3,
                arrowcolor=color, showarrow=True,
                hovertext=f"{row['Pair']}<br>Δ={row['Distance']:.4f}",
            )

    fig.update_layout(
        height=650, plot_bgcolor="#f9f9f9",
        legend=dict(orientation="h", y=-0.1),
        xaxis_title="PC1", yaxis_title="PC2",
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════
# Tab 2：方向分析（Rayleigh + 玫瑰图）
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("语义偏移方向分析")

    # 按方向分组计算 Rayleigh
    rayleigh_rows = []
    for direction in all_directions:
        sub = df_metrics[df_metrics["Direction"] == direction]
        if len(sub) < 3:
            continue
        angles_rad = np.radians(sub["Angle"])
        n = len(angles_rad)
        C, S = np.cos(angles_rad).mean(), np.sin(angles_rad).mean()
        R    = math.sqrt(C**2 + S**2)
        Z    = n * R**2
        p    = float(np.clip(
            math.exp(-Z) * (1 + (2*Z - Z**2) / (4*n)), 0.0, 1.0
        ))
        rayleigh_rows.append({
            "方向": direction, "n": n,
            "R（一致性）": round(R, 4),
            "p 值": round(p, 4),
            "均值方向（°）": round(math.degrees(math.atan2(S, C)), 1),
            "显著": "★" if p < 0.05 else "",
        })

    if rayleigh_rows:
        st.dataframe(pd.DataFrame(rayleigh_rows), use_container_width=True)
        st.caption("R 越接近 1 → 该部首偏移方向越一致；p < 0.05 → 方向一致性统计显著")

    st.divider()

# 玫瑰图（每个方向一个子图）
    directions_to_show = [d for d in all_directions if d in selected_directions]
    ncols = min(3, len(directions_to_show))
    
    if ncols > 0:
        cols = st.columns(ncols)
        for idx, direction in enumerate(directions_to_show):
            # 1. 筛选当前方向的数据
            sub = df_metrics[df_metrics["Direction"] == direction].copy()
            
            with cols[idx % ncols]:
                st.markdown(f"**{direction}** (n={len(sub)})")
                
                if not sub.empty:
                    # 2. 手动分桶：将 0-360 度划分为 24 个区间 (每区间 15 度)
                    # 使用 % 360 确保负角度也能正确归位
                    sub['angle_bin'] = (sub['Angle'] % 360 // 15) * 15
                    
                    # 3. 统计每个区间内的数量
                    counts_df = sub.groupby('angle_bin').size().reset_index(name='count')
                    
                    # 4. 使用 px.bar_polar 绘图
                    fig_p = px.bar_polar(
                        counts_df, 
                        r="count",           # 数量作为半径
                        theta="angle_bin",    # 分桶角度
                        color_discrete_sequence=["#d62728"],
                        template="plotly_white"
                    )
                    
                    # 5. 样式美化：消除柱子间的缝隙，让它更像直方图
                    fig_p.update_traces(marker_line_color="white", marker_line_width=0.5)
                    fig_p.update_layout(
                        height=320, 
                        margin=dict(l=10, r=10, t=20, b=10),
                        polar=dict(
                            angularaxis=dict(direction="clockwise", period=360),
                            radialaxis=dict(showticklabels=False, ticks="") # 隐藏半径刻度更美观
                        )
                    )
                    st.plotly_chart(fig_p, use_container_width=True)
                else:
                    st.caption("暂无数据")

# ════════════════════════════════════════════════════════════
# Tab 3：偏移强度排名
# ════════════════════════════════════════════════════════════
with tab3:
    st.subheader("声旁组偏移强度排名")
    st.caption("偏移距离越大 → 部首替换导致字义差异越显著")

    level_color = {"微弱变动": "green", "显著变动": "orange", "剧烈变动": "red"}

    direction_tabs = st.tabs(selected_directions) if selected_directions else []
    for i, direction in enumerate(selected_directions):
        sub = df_met_show[df_met_show["Direction"] == direction].sort_values("Distance")
        with direction_tabs[i]:
            fig_bar = px.bar(
                sub, x="Distance", y="Phonetic", orientation="h",
                color="Shift_Level", color_discrete_map=level_color,
                text=sub["Distance"].apply(lambda v: f"{v:.4f}"),
                labels={"Distance": "语义偏移距离", "Phonetic": "声旁"},
                height=max(350, len(sub) * 38),
            )
            fig_bar.update_traces(textposition="outside")
            fig_bar.update_layout(
                plot_bgcolor="#f9f9f9",
                legend_title="偏移等级",
                margin=dict(l=0, r=60, t=20, b=0),
            )
            st.plotly_chart(fig_bar, use_container_width=True)


# ════════════════════════════════════════════════════════════
# Tab 4：数据表
# ════════════════════════════════════════════════════════════
with tab4:
    st.subheader("原始数据与指标")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**字符坐标**")
        show_cols = [c for c in ["Character","Radical","Phonetic","GroupID",
                                  "Shuowen_clean","Dazidian_clean","x","y"]
                     if c in df_coords.columns]
        st.dataframe(df_coords[show_cols], use_container_width=True, height=400)
    with col_b:
        st.markdown("**偏移指标**")
        st.dataframe(df_met_show, use_container_width=True, height=400)

    st.download_button(
        "⬇️ 下载偏移指标 CSV",
        df_metrics.to_csv(index=False, encoding="utf_8_sig").encode("utf_8_sig"),
        "metrics.csv", "text/csv",
    )


# ════════════════════════════════════════════════════════════
# Tab 5：字义预测  [Step E]
# ════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🔮 字义预测")
    st.markdown("""
    输入**声旁**和**目标部首**，系统通过向量算术预测可能的字义：
    - **EXACT** — 该组合已在语料库中，直接返回
    - **PREDICTED** — 声旁已知，通过偏移向量推算 + LLM 生成
    - **INFERRED** — 声旁未知，降级处理（置信度较低）
    """)

    col_l, col_r = st.columns([1, 2])

    with col_l:
        st.markdown("#### 输入")
        known_phonetics = sorted(df_coords["Phonetic"].dropna().unique()) \
                          if "Phonetic" in df_coords.columns else []
        
        phonetic_input = st.text_input(
            "声旁（如：童、刃、需）",
            placeholder="输入声旁字",
            help="直接输入声旁字，可不在数据集中（触发 INFERRED 模式）"
        )
        # 快捷选择已知声旁
        if known_phonetics:
            selected_phonetic = st.selectbox(
                "或从已知声旁中选择",
                ["（手动输入）"] + known_phonetics,
            )
            if selected_phonetic != "（手动输入）":
                phonetic_input = selected_phonetic

        target_radical = st.selectbox("目标部首", config.ALL_RADICALS)
        top_k          = st.slider("近邻数量", 3, 10, 5)
        use_llm        = st.checkbox("调用 LLM 生成字义", value=True)
        run_btn        = st.button("🚀 开始预测", type="primary",
                                   disabled=not phonetic_input.strip())

    with col_r:
        st.markdown("#### 预测结果")

        if run_btn and phonetic_input.strip():
            phonetic = phonetic_input.strip()

            with st.spinner(f"正在预测「{phonetic}」+「{target_radical}」..."):
                try:
                    predictor = load_predictor()
                    if predictor is None:
                        st.error("预测器未加载，请先运行完整 pipeline")
                    else:
                        result = predictor.predict(
                            phonetic, target_radical,
                            top_k=top_k, use_llm=use_llm
                        )

                        # ── 模式标签 ──────────────────────────────────
                        mode_color = {"EXACT": "green", "PREDICTED": "orange", "INFERRED": "red"}
                        mode_label = {"EXACT": "精确命中", "PREDICTED": "向量预测", "INFERRED": "降级推断"}
                        st.markdown(
                            f"**预测模式**：:{mode_color.get(result.mode,'gray')}[{mode_label.get(result.mode, result.mode)}]  "
                            f"&nbsp;&nbsp;**置信度**：`{result.confidence}`"
                        )

                        # ── EXACT ──────────────────────────────────────
                        if result.mode == "EXACT":
                            st.success(f"在语料库中找到：**{result.exact_char}**（{target_radical}）")
                            st.markdown(f"**說文**：{result.exact_shuowen}")
                            st.markdown(f"**大字典**：{result.exact_dazidian}")

                        else:
                            # ── 锚点字 ───────────────────────────────
                            if result.anchor_chars:
                                st.markdown("**锚点字**：")
                                for a in result.anchor_chars:
                                    st.markdown(
                                        f"> **{a['character']}**（{a['radical']}）— "
                                        f"說文：{a.get('shuowen','无')}  ｜  "
                                        f"大字典：{a.get('dazidian','无')}"
                                    )
                                st.markdown(f"**偏移路径**：`{result.shift_direction}`")

                            # ── 近邻参考 ─────────────────────────────
                            if result.neighbors:
                                with st.expander("📍 语义近邻参考字", expanded=True):
                                    nb_data = [{
                                        "字": n["character"],
                                        "部首": n["radical"],
                                        "相似度": f"{n['similarity']*100:.1f}%",
                                        "說文": n.get("shuowen",""),
                                        "大字典": n.get("dazidian",""),
                                    } for n in result.neighbors]
                                    st.dataframe(pd.DataFrame(nb_data),
                                                 use_container_width=True, hide_index=True)

                            # ── LLM 结果 ─────────────────────────────
                            if result.llm_prediction:
                                st.markdown("---")
                                st.markdown("**🔮 预测字义**")
                                st.info(result.llm_prediction)
                                if result.reasoning:
                                    with st.expander("💡 推理依据"):
                                        st.write(result.reasoning)
                            elif use_llm:
                                st.warning("LLM 未返回结果，请检查 API 配置")

                        # ── 预测向量可视化（投影到 PCA 空间）──────────
                        if result.predicted_vec is not None:
                            pca_path = os.path.join(config.DATA_OUT, "pca_model.pkl")
                            if os.path.exists(pca_path):
                                with open(pca_path, "rb") as f:
                                    pca = pickle.load(f)
                                proj = pca.transform(result.predicted_vec.reshape(1, -1))[0]

                                with st.expander("🗺 在语义空间中的预测位置"):
                                    fig_pred = go.Figure()
                                    for radical in config.ALL_RADICALS:
                                        sub = df_coords[df_coords["Radical"] == radical]
                                        fig_pred.add_trace(go.Scatter(
                                            x=sub["x"], y=sub["y"],
                                            mode="markers+text",
                                            name=radical,
                                            marker=dict(color=palette[radical], size=9, opacity=0.6),
                                            text=sub["Character"],
                                            textposition="top center",
                                            textfont=dict(size=10),
                                        ))
                                    fig_pred.add_trace(go.Scatter(
                                        x=[proj[0]], y=[proj[1]],
                                        mode="markers+text",
                                        name=f"预测位置",
                                        marker=dict(color="#FFD700", size=18,
                                                    symbol="star", line=dict(width=1.5, color="black")),
                                        text=["⭐"],
                                        textposition="top center",
                                    ))
                                    if result.anchor_chars:
                                        for a in result.anchor_chars:
                                            ar = df_coords[df_coords["Character"] == a["character"]]
                                            if not ar.empty:
                                                fig_pred.add_annotation(
                                                    x=proj[0], y=proj[1],
                                                    ax=float(ar.iloc[0]["x"]),
                                                    ay=float(ar.iloc[0]["y"]),
                                                    axref="x", ayref="y",
                                                    arrowhead=2, arrowcolor="#FFD700",
                                                    arrowwidth=2, showarrow=True,
                                                )
                                    fig_pred.update_layout(
                                        height=450, plot_bgcolor="#f9f9f9",
                                        title=f"「{phonetic}」+「{target_radical}」预测位置（⭐）",
                                        margin=dict(l=0, r=0, t=40, b=0),
                                    )
                                    st.plotly_chart(fig_pred, use_container_width=True)

                except Exception as e:
                    st.error(f"预测失败：{e}")
                    st.exception(e)

        elif not phonetic_input.strip():
            st.info("请在左侧输入声旁，然后点击「开始预测」")

        # ── LangSmith 状态提示 ───────────────────────────────
        if config.LANGSMITH_ENABLED:
            st.sidebar.success("🔭 LangSmith tracing 已启用")
        else:
            st.sidebar.info(
                "💡 启用 LangSmith 追踪：\n"
                "```\npip install langsmith\n"
                "export LANGCHAIN_API_KEY=ls__xxx\n"
                "export LANGCHAIN_TRACING_V2=true\n"
                "export LANGCHAIN_PROJECT=GlyphDrift\n```"
            )
