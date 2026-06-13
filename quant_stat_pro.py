"""
================================================================================
 学术统计与建模工具 (Academic Statistics & Modeling Toolkit)
 基于 Streamlit 构建 — SPSS / AMOS 的开源替代方案
 适用场景：问卷调查研究、实验数据分析、经济学截面数据建模
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import io, base64, os, sys, re

# ──────────────────────────────────────────────────────────────
#  可视化库
# ──────────────────────────────────────────────────────────────
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

# ──────────────────────────────────────────────────────────────
#  统计库
# ──────────────────────────────────────────────────────────────
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import skew, kurtosis

# ================================================================
#  Streamlit 页面配置
# ================================================================
st.set_page_config(
    page_title="学术统计与建模工具",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
#  全局 CSS — 极简学术风格
# ================================================================
st.markdown("""
<style>
    /* ── 学术三线表 ── */
    .three-line-table {
        border-collapse: collapse;
        width: 100%;
        font-size: 13px;
        font-family: 'Times New Roman', 'SimSun', 'STSong', serif;
    }
    .three-line-table thead tr {
        border-top: 2px solid #222;
        border-bottom: 1px solid #222;
    }
    .three-line-table thead th {
        padding: 8px 14px;
        text-align: center;
        font-weight: 700;
        background-color: transparent;
        color: #111;
    }
    .three-line-table tbody td {
        padding: 6px 14px;
        text-align: center;
        border: none;
        color: #333;
    }
    .three-line-table tbody tr:last-child {
        border-bottom: 2px solid #222;
    }

    /* ── 诊断通过 ── */
    .diag-pass {
        color: #2E7D32;
        font-weight: 600;
        font-size: 15px;
    }
    /* ── 诊断警告 ── */
    .diag-warn {
        color: #C62828;
        font-weight: 600;
        font-size: 15px;
    }

    /* ── 模块标题 ── */
    .module-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #263238;
        padding: 8px 0;
        border-bottom: 2px solid #37474F;
        margin-bottom: 16px;
    }

    /* ── 主标题 ── */
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #263238;
        text-align: center;
        letter-spacing: 2px;
    }
    .sub-title {
        text-align: center;
        color: #78909C;
        font-size: 14px;
        margin-bottom: 24px;
    }

    /* ── 脚注 ── */
    .footnote {
        color: #999;
        font-size: 11px;
        text-align: center;
        padding-top: 24px;
    }

    /* ── 提示框 ── */
    .info-box {
        background: #ECEFF1;
        border-left: 4px solid #37474F;
        padding: 12px 18px;
        border-radius: 4px;
        font-size: 13px;
        color: #455A64;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# ================================================================
#  工具函数
# ================================================================

def render_three_line_table(df: pd.DataFrame, caption: str = None, precision: int = 4) -> str:
    """
    将 pandas DataFrame 渲染为符合学术出版规范的三线表 HTML。
    参数:
        df: 数据框
        caption: 表题（可选，如"表 1：描述性统计"）
        precision: 数值列保留小数位数
    返回:
        HTML 字符串，通过 st.markdown(..., unsafe_allow_html=True) 渲染
    """
    df_out = df.copy()
    for col in df_out.select_dtypes(include=[np.number]).columns:
        df_out[col] = df_out[col].round(precision)
    html = '<div style="overflow-x:auto;">'
    if caption:
        html += (
            f'<p style="text-align:center;font-weight:700;'
            f'font-size:14px;margin:0 0 4px 0;">{caption}</p>'
        )
    html += df_out.to_html(classes='three-line-table', index=False, escape=False)
    html += '</div>'
    return html


def csv_download_link(df: pd.DataFrame, filename: str, label: str) -> str:
    """
    生成 CSV 下载按钮的 HTML。
    CSV 使用 UTF-8 BOM 编码，确保中文在 Excel 中正常显示。
    """
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding='utf-8-sig')
    b64 = base64.b64encode(buf.getvalue()).decode()
    return (
        f'<a href="data:file/csv;base64,{b64}" download="{filename}" '
        f'style="display:inline-block;padding:6px 16px;background:#37474F;'
        f'color:white;text-decoration:none;border-radius:4px;font-size:13px;">'
        f'📥 {label}</a>'
    )


def xlsx_download_link(df: pd.DataFrame, filename: str, label: str) -> str:
    """生成 Excel 下载按钮的 HTML（使用 openpyxl）。"""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    b64 = base64.b64encode(buf.getvalue()).decode()
    return (
        f'<a href="data:application/vnd.openxmlformats-officedocument.'
        f'spreadsheetml.sheet;base64,{b64}" download="{filename}" '
        f'style="display:inline-block;padding:6px 16px;background:#00695C;'
        f'color:white;text-decoration:none;border-radius:4px;font-size:13px;">'
        f'📥 {label}</a>'
    )


def sig_stars(p: float) -> str:
    """显著性星标：*** p<.001, ** p<.01, * p<.05, † p<.10"""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    elif p < 0.10:
        return '†'
    else:
        return ''


# ================================================================
#  标题
# ================================================================
st.markdown('<p class="main-title">学术统计与建模工具</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Academic Statistics & Modeling Toolkit '
    '— SPSS / AMOS 开源替代方案，专为问卷、实验与截面数据设计</p>',
    unsafe_allow_html=True
)

# ================================================================
#  === 模块 1：数据管理与清洗（侧边栏）===
# ================================================================
st.sidebar.markdown("## 📁 模块 1：数据管理")

uploaded_file = st.sidebar.file_uploader(
    "上传数据文件",
    type=['csv', 'xlsx', 'xls'],
    help="支持 CSV (.csv) 或 Excel (.xlsx / .xls) 格式"
)

# ── 内置演示数据 ──
USE_DEMO = False
if uploaded_file is None:
    USE_DEMO = True
    st.sidebar.info("👆 未检测到上传文件，将自动加载学术调查演示数据集。")

    # ----------------------------------------------------------------
    # 演示数据：心理学 / 教育学问卷调查模拟数据
    # 场景：大学生学习行为与学业成就研究
    # 变量说明：
    #   SelfEfficacy  — 学业自我效能感（Likert 5 点，5 题均值）
    #   Motivation    — 学习动机（Likert 5 点，5 题均值）
    #   Anxiety       — 考试焦虑（Likert 5 点，5 题均值）
    #   Engagement    — 学习投入度（Likert 5 点，5 题均值）
    #   Support       — 感知社会支持（Likert 5 点，5 题均值）
    #   Achievement   — 学业成绩（GPA，4.0 量表）
    #   StudyHours    — 每周自习时长（小时）
    #   SleepQuality  — 睡眠质量指数（PSQI，0-21分，越低越好）
    # ----------------------------------------------------------------
    # 统计学意义：构建具有真实数据特征的模拟截面数据集——
    #   - 变量间存在合理的线性相关（如自我效能 → 学业成绩）
    #   - 存在一定的共线性（如动机 + 投入度高度相关）
    #   - 允许用户练习 VIF 诊断和模型简化
    # ----------------------------------------------------------------
    np.random.seed(210)  # 固定种子，保证可复现
    n = 300  # 模拟 300 名被试

    # 潜在因子（Latent Factors）
    latent_ability = np.random.normal(0, 1, n)          # 共同能力因子
    latent_wellbeing = np.random.normal(0, 1, n)         # 共同幸福感因子

    demo = pd.DataFrame({
        'SelfEfficacy':  3.5 + 0.6 * latent_ability + np.random.normal(0, 0.5, n),
        'Motivation':    3.3 + 0.5 * latent_ability + np.random.normal(0, 0.6, n),
        'Anxiety':       3.0 - 0.4 * latent_wellbeing + np.random.normal(0, 0.7, n),
        'Engagement':    3.4 + 0.5 * latent_ability + np.random.normal(0, 0.5, n),
        'Support':       3.6 + 0.3 * latent_wellbeing + np.random.normal(0, 0.6, n),
        'Achievement':   3.0 + 0.25 * latent_ability + np.random.normal(0, 0.35, n),
        'StudyHours':   15 + 4 * latent_ability + np.random.normal(0, 5, n),
        'SleepQuality':  8 - 1.5 * latent_wellbeing + np.random.normal(0, 2.5, n),
    })

    # 截断到合理范围
    demo['SelfEfficacy'] = demo['SelfEfficacy'].clip(1, 5)
    demo['Motivation'] = demo['Motivation'].clip(1, 5)
    demo['Anxiety'] = demo['Anxiety'].clip(1, 5)
    demo['Engagement'] = demo['Engagement'].clip(1, 5)
    demo['Support'] = demo['Support'].clip(1, 5)
    demo['Achievement'] = demo['Achievement'].clip(1.0, 4.0)
    demo['StudyHours'] = demo['StudyHours'].clip(0, 60)
    demo['SleepQuality'] = demo['SleepQuality'].clip(0, 21)

    # 人为加入少量缺失值（模拟问卷中常见的漏填）
    for col in ['Motivation', 'StudyHours', 'SleepQuality']:
        missing_idx = np.random.choice(n, size=8, replace=False)
        demo.loc[missing_idx, col] = np.nan

    df_raw = demo.copy()
    df = demo.copy()
else:
    # ── 读取用户上传文件 ──
    try:
        ext = uploaded_file.name.rsplit('.', 1)[-1].lower()
        if ext == 'csv':
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file, engine='openpyxl')
        df = df_raw.copy()
        st.sidebar.success(f"✅ 已加载：{uploaded_file.name}（{len(df_raw)} 行 × {len(df_raw.columns)} 列）")
    except Exception as e:
        st.sidebar.error(f"❌ 文件读取失败：{e}")
        st.stop()

# ── 缺失值处理 ──
st.sidebar.markdown("### ⚙️ 缺失值处理")
missing_option = st.sidebar.radio(
    "处理策略",
    options=['不处理（保留缺失）', '均值填充（仅数值列）', '直接删除含缺失值的行'],
    index=0,
    help="均值填充会保留样本量，但会轻微压缩方差；删除行会损失信息。"
)

if missing_option == '均值填充（仅数值列）':
    num_cols = df.select_dtypes(include=[np.number]).columns
    filled = 0
    for c in num_cols:
        if df[c].isna().any():
            df[c].fillna(df[c].mean(), inplace=True)
            filled += 1
    st.sidebar.success(f"已对 {filled} 个数值列填充均值。")
elif missing_option == '直接删除含缺失值的行':
    before = len(df)
    df.dropna(inplace=True)
    st.sidebar.success(f"已删除 {before - len(df)} 行，保留 {len(df)} 行。")

# ── 数据概览 ──
st.sidebar.markdown("### 📋 数据概览")
st.sidebar.write(f"有效样本量：**N = {len(df)}**")
num_count = len(df.select_dtypes(include=[np.number]).columns)
st.sidebar.write(f"数值变量数：**{num_count}**")

if USE_DEMO:
    st.sidebar.markdown("---")
    st.sidebar.caption("当前使用**演示数据集**：大学生学习行为与学业成就研究（N=300）")

# ================================================================
#  === 模块 2：学术描述性统计 ===
# ================================================================
st.markdown('<p class="module-title">📊 模块 2：学术描述性统计</p>', unsafe_allow_html=True)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if not num_cols:
    st.warning("数据集中无可分析的数值列，请检查数据格式。")
    st.stop()

tab_desc, tab_corr = st.tabs(["📋 描述统计表", "🔥 相关系数矩阵"])

with tab_desc:
    st.markdown("""
    <div class="info-box">
    <b>说明：</b>下表为各数值变量的集中趋势与离散程度汇总。<br>
    <b>偏度 (Skewness)：</b>正值 = 右偏（多数被试得分偏低）；负值 = 左偏（多数得分偏高）。绝对值 > 1 提示明显偏离正态。<br>
    <b>峰度 (Kurtosis)：</b>采用 Pearson 定义（正态分布 = 3）。> 3 表示"尖峰厚尾"，< 3 表示扁平分布。
    </div>
    """, unsafe_allow_html=True)

    # 构建描述统计表
    desc_rows = []
    for c in num_cols:
        col_data = df[c].dropna()
        desc_rows.append({
            '变量': c,
            'N': len(col_data),
            '最小值': col_data.min(),
            '最大值': col_data.max(),
            '均值': col_data.mean(),
            '标准差': col_data.std(ddof=1),
            '偏度': skew(col_data),
            '峰度': kurtosis(col_data, fisher=False),
        })
    desc_df = pd.DataFrame(desc_rows)
    desc_df = desc_df.round(4)

    st.markdown(
        render_three_line_table(desc_df, caption=f"表 1：变量描述性统计（N = {len(df)}）"),
        unsafe_allow_html=True
    )

    col_dl1, col_dl2, _ = st.columns([2, 2, 6])
    with col_dl1:
        st.markdown(csv_download_link(desc_df, "descriptive_stats.csv", "CSV"), unsafe_allow_html=True)
    with col_dl2:
        st.markdown(xlsx_download_link(desc_df, "descriptive_stats.xlsx", "Excel"), unsafe_allow_html=True)

with tab_corr:
    st.markdown("""
    <div class="info-box">
    <b>Pearson 积差相关系数：</b>度量两连续变量间的线性关联强度，r ∈ [-1, 1]。<br>
    一般经验标准：|r| < 0.3 弱相关；0.3 ≤ |r| < 0.7 中等相关；|r| ≥ 0.7 强相关。<br>
    相关系数矩阵是后续回归分析中 <b>多重共线性初筛</b> 的首要步骤。
    </div>
    """, unsafe_allow_html=True)

    if len(num_cols) >= 2:
        corr = df[num_cols].corr(method='pearson')

        # 高质量热力图
        fig_heat = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                colorscale='RdBu_r',       # 蓝-白-红，学术常用
                zmin=-1, zmax=1,
                text=np.round(corr.values, 3),
                texttemplate='%{text}',
                textfont=dict(size=11, color='#222'),
                colorbar=dict(title='Pearson r', title_font_size=12),
                xgap=1, ygap=1,
            )
        )
        fig_heat.update_layout(
            template='plotly_white',
            font=dict(family='Times New Roman, SimSun', size=11),
            height=500,
            width=680,
            margin=dict(l=10, r=10, t=10, b=60),
            title=None,
        )
        fig_heat.update_yaxes(autorange='reversed')
        st.plotly_chart(fig_heat, use_container_width=True)

        col_dl3, col_dl4, _ = st.columns([2, 2, 6])
        with col_dl3:
            st.markdown(csv_download_link(corr.round(4), "correlation_matrix.csv", "CSV"), unsafe_allow_html=True)
        with col_dl4:
            st.markdown(xlsx_download_link(corr.round(4), "correlation_matrix.xlsx", "Excel"), unsafe_allow_html=True)
    else:
        st.info("需要至少 2 个数值列方可计算相关系数。")

# ================================================================
#  === 模块 3：多元线性回归 (OLS) ===
# ================================================================
st.markdown('<p class="module-title">📐 模块 3：多元线性回归 (OLS)</p>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
<b>模型设定：</b>Y = β₀ + β₁X₁ + β₂X₂ + ... + βₖXₖ + ε，ε ~ N(0, σ²)<br>
使用 <b>OLS（普通最小二乘法）</b> 估计系数向量 β̂ = (X'X)⁻¹X'Y。<br>
<b>强制诊断项目：</b>多重共线性 (VIF)、异方差 (Breusch-Pagan)、残差自相关 (Durbin-Watson)。
</div>
""", unsafe_allow_html=True)

if len(num_cols) < 2:
    st.warning("回归分析需要至少 2 个数值列。")
else:
    col_y, col_x = st.columns([3, 5])

    with col_y:
        y_var = st.selectbox(
            "因变量 Y（被解释变量）",
            options=num_cols,
            index=num_cols.index('Achievement') if 'Achievement' in num_cols else len(num_cols)-1,
            key="y_select"
        )

    with col_x:
        x_candidates = [c for c in num_cols if c != y_var]
        x_vars = st.multiselect(
            "自变量 X（解释变量，可多选）",
            options=x_candidates,
            default=[c for c in x_candidates if c != y_var][:4],
            key="x_select",
            help="按住 Ctrl/Cmd 可选中或取消单个变量；点击选项上方的 ✕ 可移除。"
        )

    run_ols = st.button("▶ 执行回归分析", type="primary", key="run_ols")

    if run_ols and x_vars:
        try:
            # ── 数据准备 ──
            Y = df[y_var].dropna()
            X = df[x_vars].loc[Y.index].dropna()
            common_idx = X.index
            Y = Y.loc[common_idx]

            if len(Y) < 30:
                st.error(f"有效样本量过低（n = {len(Y)}），回归分析不可靠。")
                st.stop()

            X_with_const = sm.add_constant(X)

            # ── OLS 估计 ──
            model = sm.OLS(Y, X_with_const).fit()

            # ── 回归系数表 ──
            st.markdown("#### 回归系数估计")
            coef_rows = []
            for var_name in model.params.index:
                coef_rows.append({
                    '变量': var_name,
                    'B (非标准化系数)': model.params[var_name],
                    'SE (标准误)': model.bse[var_name],
                    'β (标准化系数)': np.nan,  # 后面填充
                    't 值': model.tvalues[var_name],
                    'p 值': model.pvalues[var_name],
                    '显著性': sig_stars(model.pvalues[var_name]),
                })
            coef_df = pd.DataFrame(coef_rows)

            # 标准化回归系数 β
            # 统计学意义：对 (X, Y) 进行 Z 变换后回归得到的系数，称为标准化回归系数，
            # 其大小可直接比较各预测变量对因变量的相对重要性（不受原始测量单位影响）。
            Y_std = (Y - Y.mean()) / Y.std(ddof=1)
            X_std = (X - X.mean()) / X.std(ddof=1)
            X_std_c = sm.add_constant(X_std)
            beta_model = sm.OLS(Y_std, X_std_c).fit()
            for i, var_name in enumerate(model.params.index):
                if var_name != 'const':
                    coef_df.loc[coef_df['变量'] == var_name, 'β (标准化系数)'] = round(
                        beta_model.params[var_name], 4
                    )

            st.markdown(
                render_three_line_table(coef_df, caption="表 2：多元线性回归系数估计"),
                unsafe_allow_html=True
            )
            st.caption("*显著性编码：*** p < .001  ** p < .01  * p < .05  † p < .10")

            # 模型拟合摘要
            st.markdown(
                f"**R²** = {model.rsquared:.5f}　|　"
                f"**调整 R²** = {model.rsquared_adj:.5f}　|　"
                f"**F({int(model.fvalue):.0f}, {int(model.df_resid)})** = {model.fvalue:.4f}　|　"
                f"**p(F)** = {model.f_pvalue:.5f}　|　"
                f"**AIC** = {model.aic:.2f}　|　**BIC** = {model.bic:.2f}"
            )

            # ── 诊断 1：VIF 共线性检验 ──
            st.markdown("---")
            st.markdown("#### 🔍 诊断 1：多重共线性检验 (VIF)")
            st.markdown("""
            <div class="info-box">
            <b>VIF（方差膨胀因子）：</b>VIFⱼ = 1 / (1 - Rⱼ²)，其中 Rⱼ² 是把 Xⱼ 当作因变量、
            对其余自变量回归的决定系数。<br>
            <b>判定标准：</b>VIF < 5：无共线性问题；5 ≤ VIF ≤ 10：中度共线性，建议关注；
            VIF > 10：严重共线性，必须处理（删除变量或使用岭回归 / LASSO）。
            </div>
            """, unsafe_allow_html=True)

            try:
                X_vif = X.copy()
                vif_vals = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
                vif_df = pd.DataFrame({
                    '变量': X_vif.columns.tolist(),
                    'VIF': vif_vals,
                })
                vif_df['判定'] = vif_df['VIF'].apply(
                    lambda v: '⚠ 严重共线性 (VIF > 10)' if v > 10
                    else ('⚡ 中度共线性 (5 < VIF ≤ 10)' if v > 5 else '✓ 正常 (VIF ≤ 5)')
                )
                st.markdown(
                    render_three_line_table(vif_df, caption="表 3：方差膨胀因子 (VIF) 诊断"),
                    unsafe_allow_html=True
                )
                if (vif_df['VIF'] > 10).any():
                    st.markdown(
                        '<p class="diag-warn">⚠ 警告：存在严重多重共线性（VIF > 10）。建议删除高度相关的自变量，或改用岭回归 / LASSO。</p>',
                        unsafe_allow_html=True
                    )
                elif (vif_df['VIF'] > 5).any():
                    st.markdown(
                        '<p class="diag-warn">⚡ 注意：部分变量存在中度共线性（5 < VIF ≤ 10），建议关注但可保留。</p>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown('<p class="diag-pass">✓ VIF 检验通过：各自变量间无严重共线性。</p>',
                                unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"VIF 计算异常：{e}（可能原因：样本量不足或 X 矩阵不满秩）")

            # ── 诊断 2：异方差检验 ──
            st.markdown("---")
            st.markdown("#### 🔍 诊断 2：异方差检验 (Breusch-Pagan)")
            st.markdown("""
            <div class="info-box">
            <b>Breusch-Pagan 检验：</b>H₀ 假设为残差同方差。若 p < .05，则拒绝 H₀，
            表明存在异方差性，此时 OLS 的系数估计仍无偏但标准误有偏，
            导致 t 检验和 F 检验不可靠。
            </div>
            """, unsafe_allow_html=True)
            try:
                bp_stat, bp_p, _, _ = het_breuschpagan(model.resid, model.model.exog)
                hetero_df = pd.DataFrame({
                    '检验': ['Breusch-Pagan'],
                    '统计量': [bp_stat],
                    'p 值': [bp_p],
                    '结论': ['⚠ 存在异方差 (p < .05)' if bp_p < 0.05 else '✓ 未检出异方差 (p ≥ .05)']
                })
                st.markdown(
                    render_three_line_table(hetero_df, caption="表 4：Breusch-Pagan 异方差检验"),
                    unsafe_allow_html=True
                )
                if bp_p < 0.05:
                    st.markdown(
                        '<p class="diag-warn">⚠ 异方差检验未通过。建议报告 heteroskedasticity-robust 标准误（HC1/HC3），或使用 WLS 估计。</p>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown('<p class="diag-pass">✓ 异方差检验通过（残差同方差）。</p>',
                                unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"异方差检验失败：{e}")

            # ── 诊断 3：Durbin-Watson 自相关 ──
            st.markdown("---")
            st.markdown("#### 🔍 诊断 3：残差自相关 (Durbin-Watson)")
            st.markdown("""
            <div class="info-box">
            <b>Durbin-Watson 统计量：</b>检验残差的一阶自相关。DW ≈ 2 表示无自相关；
            DW → 0 为正自相关；DW → 4 为负自相关。<br>
            对于截面数据，此项检验通常不会出现问题；若数据有时间结构，需格外关注。
            </div>
            """, unsafe_allow_html=True)
            dw = durbin_watson(model.resid)
            dw_df = pd.DataFrame({
                '指标': ['Durbin-Watson'],
                '值': [dw],
                '诊断': ['✓ 接近 2，无自相关' if abs(dw - 2) < 0.5
                        else '⚠ 存在正自相关 (DW < 1.5)' if dw < 1.5
                        else '⚠ 存在负自相关 (DW > 2.5)']
            })
            st.markdown(
                render_three_line_table(dw_df, caption="表 5：Durbin-Watson 残差自相关检验"),
                unsafe_allow_html=True
            )
            if abs(dw - 2) >= 0.5:
                st.markdown(
                    '<p class="diag-warn">⚠ DW 统计量偏离 2。对于截面数据，建议检查数据录入顺序是否可能引入虚假自相关。</p>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown('<p class="diag-pass">✓ Durbin-Watson 检验通过。</p>',
                            unsafe_allow_html=True)

            # ── 残差诊断图 ──
            st.markdown("---")
            st.markdown("#### 📈 残差诊断图")
            resid = model.resid
            fitted = model.fittedvalues
            std_resid = resid / np.std(resid, ddof=1)

            fig_diag = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    '(a) 残差 vs 拟合值',
                    '(b) Q-Q 正态概率图',
                    '(c) 标准化残差直方图',
                    '(d) 标准化残差 vs 杠杆值',
                )
            )

            # (a) Residual vs Fitted
            fig_diag.add_trace(
                go.Scatter(x=fitted, y=resid, mode='markers',
                           marker=dict(color='#455A64', size=6, opacity=0.5),
                           showlegend=False),
                row=1, col=1
            )
            fig_diag.add_hline(y=0, line_dash='dash', line_color='#C62828', row=1, col=1)

            # (b) Q-Q Plot
            qq = stats.probplot(resid, dist="norm")
            fig_diag.add_trace(
                go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers',
                           marker=dict(color='#455A64', size=6, opacity=0.5),
                           showlegend=False),
                row=1, col=2
            )
            slope, intercept = qq[1]
            x_ends = np.array([qq[0][0].min(), qq[0][0].max()])
            fig_diag.add_trace(
                go.Scatter(x=x_ends, y=intercept + slope * x_ends,
                           mode='lines', line=dict(color='#C62828', dash='dash', width=1.5),
                           showlegend=False),
                row=1, col=2
            )

            # (c) 标准化残差直方图
            fig_diag.add_trace(
                go.Histogram(x=std_resid.values, nbinsx=30,
                             marker_color='#607D8B', opacity=0.7,
                             showlegend=False),
                row=2, col=1
            )

            # (d) 标准化残差 vs Leverage
            influence = model.get_influence()
            leverage = influence.hat_matrix_diag
            fig_diag.add_trace(
                go.Scatter(x=leverage, y=std_resid, mode='markers',
                           marker=dict(color='#455A64', size=6, opacity=0.5),
                           showlegend=False),
                row=2, col=2
            )
            fig_diag.add_hline(y=0, line_dash='dash', line_color='#C62828', row=2, col=2)
            fig_diag.add_hline(y=2, line_dash='dot', line_color='#FF8F00', row=2, col=2)
            fig_diag.add_hline(y=-2, line_dash='dot', line_color='#FF8F00', row=2, col=2)

            fig_diag.update_layout(
                template='plotly_white',
                font=dict(family='Times New Roman, SimSun', size=11),
                height=600,
                margin=dict(l=30, r=20, t=50, b=30),
            )
            st.plotly_chart(fig_diag, use_container_width=True)

            # ── 导出 ──
            st.markdown("---")
            col_export1, col_export2, _ = st.columns([2, 2, 6])
            with col_export1:
                st.markdown(csv_download_link(coef_df, "regression_coefficients.csv", "系数表 CSV"),
                            unsafe_allow_html=True)
            with col_export2:
                st.markdown(xlsx_download_link(coef_df, "regression_coefficients.xlsx", "系数表 Excel"),
                            unsafe_allow_html=True)

        except Exception as e:
            st.error(f"回归分析失败：{e}")
            st.code(str(e))

# ================================================================
#  === 模块 4：结构方程模型 (SEM) ===
# ================================================================
st.markdown('<p class="module-title">🔗 模块 4：结构方程模型 (SEM)</p>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
<b>结构方程模型（Structural Equation Modeling）</b> 同时估计测量模型与结构模型，
是验证性因子分析（CFA）和路径分析（Path Analysis）的一般框架。<br>
本模块基于 <code>semopy</code> 库实现，语法与 <code>lavaan</code>（R 语言）类似。<br>
<b>语法参考：</b><br>
&nbsp;&nbsp; <code>=~</code> 定义测量关系（潜变量 ← 观测指标）<br>
&nbsp;&nbsp; <code>~</code> 定义回归关系（因变量 ← 自变量）<br>
&nbsp;&nbsp; <code>~~</code> 定义方差 / 协方差
</div>
""", unsafe_allow_html=True)

col_sem_input, col_sem_output = st.columns([5, 5])

with col_sem_input:
    sem_code = st.text_area(
        "SEM 模型语法",
        value="""# === 测量模型（Measurement Model）===
# 潜变量 F1（学业自我效能）由三个观测指标定义
F1 =~ SelfEfficacy + Motivation + Engagement

# 潜变量 F2（心理健康）由三个观测指标定义
F2 =~ Anxiety + SleepQuality + Support

# === 结构模型（Structural Model）===
# 学业成绩受两个潜变量共同预测
Achievement ~ F1 + F2

# 学业时长也受潜变量影响
StudyHours ~ F1""",
        height=340,
        key="sem_syntax_input",
        help="每行一个关系定义。可自由修改以匹配你的变量名。"
    )

    run_sem = st.button("▶ 估计 SEM 模型", type="primary", key="run_sem")

with col_sem_output:
    if run_sem:
        try:
            import semopy

            # ── 解析语法中出现的观测变量名 ──
            tokens = set(re.findall(r'[A-Za-z_]\w*', sem_code))
            reserved = {'F1', 'F2', 'F3', 'F4', 'F5', 'F6'}
            observed_vars = sorted(tokens - reserved)

            # ── 准备数据 ──
            available = [v for v in observed_vars if v in df.columns]
            if len(available) < 3:
                st.error(
                    f"语法中引用的变量 ({', '.join(sorted(observed_vars))}) 在数据中匹配不足。\n"
                    f"当前可用数值列：{', '.join(num_cols)}\n\n"
                    f"请修改语法以匹配实际列名。"
                )
                st.stop()

            sem_data = df[available].dropna()

            if len(sem_data) < 100:
                st.warning(
                    f"有效样本量 N = {len(sem_data)}，较小。"
                    f"SEM 通常要求 N ≥ 200 才能保证参数估计和拟合指标稳定。"
                )

            # ── 模型拟合 ──
            sem_model = semopy.Model(sem_code)
            sem_model.fit(sem_data, obj='MLW')

            # ── 拟合优度指标 ──
            stats_sem = semopy.calc_stats(sem_model)

            def safe_stat(d, key, fmt='.5f'):
                """安全提取拟合指标，若不存在则返回 '—'"""
                if key in d.columns and len(d[key]) > 0:
                    val = d[key].values[0]
                    if pd.isna(val):
                        return '—'
                    return f'{val:{fmt}}'
                return '—'

            fit_records = [
                {
                    '指标': 'χ² (Chi-square)',
                    '值': safe_stat(stats_sem, 'chi2'),
                    '参考标准': '越小越好',
                },
                {
                    '指标': 'df (自由度)',
                    '值': safe_stat(stats_sem, 'df', '.0f'),
                    '参考标准': '—',
                },
                {
                    '指标': 'p (χ²)',
                    '值': safe_stat(stats_sem, 'p-value'),
                    '参考标准': '> .05 表示模型拟合可接受',
                },
                {
                    '指标': 'RMSEA',
                    '值': safe_stat(stats_sem, 'RMSEA'),
                    '参考标准': '< .05 优良；< .08 可接受',
                },
                {
                    '指标': 'CFI',
                    '值': safe_stat(stats_sem, 'CFI'),
                    '参考标准': '> .95 优良；> .90 可接受',
                },
                {
                    '指标': 'TLI (NNFI)',
                    '值': safe_stat(stats_sem, 'TLI'),
                    '参考标准': '> .95 优良；> .90 可接受',
                },
                {
                    '指标': 'GFI',
                    '值': safe_stat(stats_sem, 'GFI'),
                    '参考标准': '> .95 优良；> .90 可接受',
                },
                {
                    '指标': 'AGFI',
                    '值': safe_stat(stats_sem, 'AGFI'),
                    '参考标准': '> .90 可接受',
                },
                {
                    '指标': 'NFI',
                    '值': safe_stat(stats_sem, 'NFI'),
                    '参考标准': '> .90 可接受',
                },
                {
                    '指标': 'SRMR',
                    '值': safe_stat(stats_sem, 'SRMR'),
                    '参考标准': '< .05 优良；< .08 可接受',
                },
                {
                    '指标': 'AIC',
                    '值': safe_stat(stats_sem, 'AIC'),
                    '参考标准': '模型比较用，越小越好',
                },
                {
                    '指标': 'BIC',
                    '值': safe_stat(stats_sem, 'BIC'),
                    '参考标准': '模型比较用，越小越好',
                },
            ]
            fit_df = pd.DataFrame(fit_records)
            st.markdown(
                render_three_line_table(fit_df, caption="表 6：SEM 模型拟合优度指标"),
                unsafe_allow_html=True
            )

            # 核心指标判定
            def judge_fit(fit_data):
                rmsea_val = None
                cfi_val = None
                for row in fit_data:
                    if row['指标'] == 'RMSEA':
                        try:
                            rmsea_val = float(row['值'])
                        except (ValueError, TypeError):
                            pass
                    if row['指标'] == 'CFI':
                        try:
                            cfi_val = float(row['值'])
                        except (ValueError, TypeError):
                            pass
                if rmsea_val is not None and cfi_val is not None:
                    if rmsea_val < 0.05 and cfi_val > 0.95:
                        return '<p class="diag-pass">✓ 模型拟合优良：RMSEA < .05 且 CFI > .95</p>'
                    elif rmsea_val < 0.08 and cfi_val > 0.90:
                        return '<p class="diag-warn">⚡ 模型拟合可接受但未达优良水平（.05 ≤ RMSEA < .08 或 .90 < CFI ≤ .95）</p>'
                    else:
                        return '<p class="diag-warn">⚠ 模型拟合欠佳（RMSEA ≥ .08 或 CFI ≤ .90），建议修正模型设定。</p>'
                return ''
            st.markdown(judge_fit(fit_records), unsafe_allow_html=True)

            # ── 路径系数 ──
            st.markdown("#### 路径系数估计")
            param_df = sem_model.inspect()
            if param_df is not None and not param_df.empty:
                param_disp = param_df[['lval', 'op', 'rval', 'Estimate', 'Std. Err', 'z-value', 'p-value']].copy()
                param_disp.columns = ['左变量', '关系', '右变量', '估计值', '标准误', 'z 值', 'p 值']
                param_disp['显著性'] = [sig_stars(p) for p in param_disp['p 值']]
                st.markdown(
                    render_three_line_table(param_disp, caption="表 7：SEM 路径系数估计"),
                    unsafe_allow_html=True
                )
                st.markdown(
                    csv_download_link(param_disp, "sem_path_coefficients.csv", "路径系数 CSV"),
                    unsafe_allow_html=True
                )
            else:
                st.info("无法提取路径系数。请检查语法是否正确，以及模型是否成功收敛。")

        except ImportError:
            st.error("""
            `semopy` 库未正确安装。请运行：
            ```
            pip install semopy
            ```
            """)
        except Exception as e:
            st.error(f"SEM 估计失败：{e}")
            import traceback
            with st.expander("查看详细错误"):
                st.code(traceback.format_exc())

# ================================================================
#  页脚
# ================================================================
st.markdown("---")
st.markdown(
    '<p class="footnote">'
    '📊 Academic Statistics & Modeling Toolkit — Built with Streamlit, statsmodels & semopy<br>'
    '适用于学术论文数据分析场景（问卷研究 · 实验设计 · 截面数据建模）</p>',
    unsafe_allow_html=True
)
