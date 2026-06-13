"""
================================================================================
 高级统计与计量经济学可视化看板 (Advanced Statistics & Econometrics Dashboard)
 基于 Streamlit 构建，适用于学术论文数据分析
 作者：量化金融数据科学工具集
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
from io import BytesIO

# ----------------------------------------------------------------
# 可视化库
# ----------------------------------------------------------------
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

# ----------------------------------------------------------------
# 统计与计量库
# ----------------------------------------------------------------
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# ----------------------------------------------------------------
# 数学库
# ----------------------------------------------------------------
from scipy.stats import skew, kurtosis

# ----------------------------------------------------------------
# 页面配置
# ----------------------------------------------------------------
st.set_page_config(
    page_title="高级统计与计量经济学看板",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
#  全局 CSS：学术三线表样式 + 页面美化
# ================================================================
st.markdown("""
<style>
    /* 学术三线表样式 */
    .three-line-table {
        border-collapse: collapse;
        width: 100%;
        font-size: 13px;
        font-family: 'Times New Roman', 'SimSun', serif;
    }
    .three-line-table thead tr {
        border-top: 2px solid black;
        border-bottom: 1px solid black;
    }
    .three-line-table thead th {
        padding: 6px 12px;
        text-align: center;
        font-weight: bold;
        background-color: transparent;
    }
    .three-line-table tbody td {
        padding: 4px 12px;
        text-align: center;
        border: none;
    }
    .three-line-table tbody tr:last-child {
        border-bottom: 2px solid black;
    }

    /* 警告文本 */
    .warning-text {
        color: #CC0000;
        font-weight: bold;
        font-size: 15px;
    }

    /* 通过文本 */
    .pass-text {
        color: #008800;
        font-weight: bold;
    }

    /* 主标题 */
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1a3a5c;
        text-align: center;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #1a3a5c;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================
#  工具函数：学术三线表渲染
# ================================================================
def render_three_line_table(df, caption=None, precision=4):
    """
    将 pandas DataFrame 渲染为学术三线表格式的 HTML。
    参数:
        df: pandas DataFrame
        caption: 表格标题（可选）
        precision: 数值精度
    返回:
        HTML 字符串
    """
    # 数值列四舍五入
    df_out = df.copy()
    for col in df_out.select_dtypes(include=[np.number]).columns:
        df_out[col] = df_out[col].round(precision)

    html = '<div style="overflow-x:auto;">'
    if caption:
        html += f'<p style="text-align:center;font-weight:bold;margin-bottom:2px;">{caption}</p>'
    html += df_out.to_html(
        classes='three-line-table',
        index=False,
        escape=False
    )
    html += '</div>'
    return html


def convert_df_to_csv(df):
    """将 DataFrame 转为 CSV 下载链接（UTF-8 BOM 兼容 Excel 中文）"""
    output = io.BytesIO()
    df.to_csv(output, index=False, encoding='utf-8-sig')
    return output.getvalue()


def download_button(df, filename, button_text):
    """生成下载按钮"""
    csv = convert_df_to_csv(df)
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" ' \
           f'style="display:inline-block;padding:8px 16px;background-color:#1a3a5c;' \
           f'color:white;text-decoration:none;border-radius:4px;font-size:14px;">' \
           f'📥 {button_text}</a>'
    return href


# ================================================================
#  工具函数：VIF 计算
# ================================================================
def calc_vif(X):
    """
    计算方差膨胀因子（VIF），用于检测多重共线性。
    统计学意义：
        VIF_j = 1 / (1 - R_j^2)，其中 R_j^2 是将 X_j 对其余自变量回归的决定系数。
        VIF > 10 通常被视为存在严重多重共线性的标志（经验阈值）。
        VIF = 1 表示该变量与其他自变量完全正交。
    """
    vif_data = pd.DataFrame({
        '变量': X.columns,
        'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    vif_data['诊断'] = vif_data['VIF'].apply(
        lambda x: '⚠ 严重共线性' if x > 10 else ('⚡ 中度共线性' if x > 5 else '✓ 正常')
    )
    return vif_data


# ================================================================
#  工具函数：偏度与峰度计算
# ================================================================
def compute_descriptive_stats(df):
    """
    计算数据框各数值列的基本描述性统计量。
    统计学意义：
        偏度（Skewness）：度量分布的对称性。
            skew > 0 → 右偏（厚尾在右）；skew < 0 → 左偏（厚尾在左）。
        峰度（Kurtosis）：度量分布的"峰度"，即极端值出现的倾向。
            正态分布的峰度 ≈ 3（Fisher 定义下为 0）。
            峰度 > 3 表示"厚尾"（金融收益率常见特征）。
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return pd.DataFrame()

    stats_df = pd.DataFrame({
        '变量': numeric_cols,
        '均值': [df[c].mean() for c in numeric_cols],
        '标准差': [df[c].std(ddof=1) for c in numeric_cols],
        '最小值': [df[c].min() for c in numeric_cols],
        'Q1 (25%)': [df[c].quantile(0.25) for c in numeric_cols],
        '中位数': [df[c].median() for c in numeric_cols],
        'Q3 (75%)': [df[c].quantile(0.75) for c in numeric_cols],
        '最大值': [df[c].max() for c in numeric_cols],
        '偏度': [skew(df[c].dropna()) for c in numeric_cols],
        '峰度': [kurtosis(df[c].dropna(), fisher=False) for c in numeric_cols],
        '缺失数': [df[c].isna().sum() for c in numeric_cols],
        '缺失率(%)': [round(df[c].isna().sum() / len(df) * 100, 2) for c in numeric_cols]
    })
    return stats_df


# ================================================================
#  主界面标题
# ================================================================
st.markdown('<p class="main-title">📊 高级统计与计量经济学可视化看板</p>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;color:#666;margin-top:-10px;">'
    'Advanced Statistics & Econometrics Dashboard for Academic Research</p>',
    unsafe_allow_html=True
)

# ================================================================
#  =================== 模块 1：数据控制台（侧边栏）=================
# ================================================================
st.sidebar.markdown("## 📁 模块 1：数据控制台")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "上传数据文件",
    type=['csv', 'xlsx', 'xls'],
    help="支持 CSV 或 Excel 格式的数据文件"
)

if uploaded_file is not None:
    try:
        # ------------------------------------------------------------
        # 读取文件：根据扩展名选择读取方式
        # ------------------------------------------------------------
        file_ext = uploaded_file.name.split('.')[-1].lower()
        if file_ext == 'csv':
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file, engine='openpyxl')

        # ------------------------------------------------------------
        # 数据预处理选项
        # ------------------------------------------------------------
        st.sidebar.markdown("### ⚙️ 数据预处理选项")

        # 缺失值处理策略选择
        missing_strategy = st.sidebar.radio(
            "缺失值处理方式",
            options=['不处理', '均值填充（数值列）', '中位数填充（数值列）', '直接删除含缺失值的行'],
            index=0,
            help="选择如何处理数据中的缺失值。均值/中位数填充仅对数值列生效。"
        )

        # Z-score 标准化复选框
        standardize = st.sidebar.checkbox(
            "Z-score 标准化（数值列）",
            value=False,
            help="将数值列转换为均值为 0、标准差为 1 的标准正态分布。"
        )

        # ------------------------------------------------------------
        # 执行数据预处理
        # ------------------------------------------------------------
        df = df_raw.copy()

        if missing_strategy == '均值填充（数值列）':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isna().any():
                    df[col].fillna(df[col].mean(), inplace=True)
            st.sidebar.success(f"已使用均值填充 {len(numeric_cols)} 个数值列的缺失值")

        elif missing_strategy == '中位数填充（数值列）':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isna().any():
                    df[col].fillna(df[col].median(), inplace=True)
            st.sidebar.success(f"已使用中位数填充 {len(numeric_cols)} 个数值列的缺失值")

        elif missing_strategy == '直接删除含缺失值的行':
            before = len(df)
            df.dropna(inplace=True)
            after = len(df)
            st.sidebar.success(f"已删除 {before - after} 行，剩余 {after} 行")

        if standardize:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std(ddof=1)
                st.sidebar.success(f"已对 {len(numeric_cols)} 个数值列进行 Z-score 标准化")

        # ------------------------------------------------------------
        # 数据预览
        # ------------------------------------------------------------
        st.sidebar.markdown("### 📋 数据预览")
        st.sidebar.write(f"原始行数: {len(df_raw)} | 预处理后行数: {len(df)}")
        st.sidebar.write(f"数值列数: {len(df.select_dtypes(include=[np.number]).columns)}")
        st.sidebar.write(f"分类列数: {len(df.select_dtypes(exclude=[np.number]).columns)}")

    except Exception as e:
        st.sidebar.error(f"❌ 读取文件时出错：{str(e)}")
        st.stop()

else:
    # 未上传文件时提供演示数据
    st.info("👈 请在左侧边栏上传数据文件以开始分析，或使用下方的演示数据。")
    # ------------------------------------------------------------
    # 生成演示数据集：模拟金融资产收益率数据
    # ------------------------------------------------------------
    # 统计学意义：演示数据模拟了典型的金融时间序列特征——
    # 资产收益率接近正态分布但有轻微的厚尾（峰度 > 3），
    # 变量间存在一定相关性（如市场因子与个股收益），
    # 适合展示多元线性回归和 ARIMA 建模流程。
    np.random.seed(42)
    n = 200
    dates = pd.date_range(start='2023-01-01', periods=n, freq='B')  # 工作日频率

    demo_data = pd.DataFrame({
        '日期': dates,
        '市场收益率(%)': np.random.normal(0.05, 1.2, n),
        '个股A收益率(%)': np.random.normal(0.08, 2.0, n),
        '个股B收益率(%)': np.random.normal(0.03, 1.8, n),
        'GDP增速(%)': np.random.normal(3.5, 0.5, n),
        'CPI通胀率(%)': np.random.normal(2.0, 0.3, n),
        '交易量(百万)': np.random.lognormal(4.0, 0.6, n),
        '波动率指数VIX': np.random.gamma(2.0, 8.0, n),
    })
    # 引入一些相关性：让个股A与市场收益率部分相关
    demo_data['个股A收益率(%)'] = 0.6 * demo_data['市场收益率(%)'] + \
                                   np.random.normal(0, 1.3, n)
    # 引入少量缺失值
    demo_data.loc[np.random.choice(n, 10, replace=False), 'GDP增速(%)'] = np.nan

    df = demo_data.copy()
    df_raw = demo_data.copy()
    st.markdown("*当前使用内置演示数据集（金融资产收益率模拟数据）*")

# ================================================================
#  =================== 模块 2：描述性统计与 EDA ====================
# ================================================================
st.markdown("---")
st.markdown("## 📈 模块 2：描述性统计与探索性数据分析 (EDA)")

tab_eda1, tab_eda2, tab_eda3 = st.tabs(["📊 基本统计量", "📉 分布直方图", "🔥 相关系数热力图"])

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

with tab_eda1:
    st.markdown("### 基本描述性统计量")
    st.markdown("""
    *偏度 > 0 表示右偏（长尾在右侧），偏度 < 0 表示左偏（长尾在左侧）。
    峰度 > 3（Fisher 定义下 > 0）表示分布比正态分布更"尖峰厚尾"——这是金融收益率数据的典型特征。*
    """)
    stats_result = compute_descriptive_stats(df)
    if not stats_result.empty:
        st.markdown(
            render_three_line_table(stats_result, caption="表 1：数值变量描述性统计"),
            unsafe_allow_html=True
        )
        # 导出按钮
        st.markdown(
            download_button(stats_result, "descriptive_stats.csv", "导出描述性统计表 (CSV)"),
            unsafe_allow_html=True
        )
    else:
        st.warning("数据集中没有数值列可供分析。")

with tab_eda2:
    st.markdown("### 交互式数据分布直方图")
    st.markdown("*使用 Plotly 绘制，支持缩放、平移、悬停查看数据。直方图叠加核密度估计（KDE）曲线。*")

    if numeric_cols:
        selected_hist_col = st.selectbox("选择要可视化的变量", numeric_cols, key="hist_col")
        bins_slider = st.slider("分箱数 (bins)", min_value=10, max_value=100, value=40, step=5, key="bins_slider")

        fig_hist = ff.create_distplot(
            [df[selected_hist_col].dropna().values],
            group_labels=[selected_hist_col],
            bin_size=(df[selected_hist_col].max() - df[selected_hist_col].min()) / bins_slider,
            show_hist=True,
            show_rug=True,
            curve_type='kde',
            colors=['#2c3e50']
        )
        # 学术风格美化
        fig_hist.update_layout(
            title=f'变量分布直方图：{selected_hist_col}',
            xaxis_title=selected_hist_col,
            yaxis_title='概率密度',
            template='plotly_white',
            font=dict(family='Times New Roman', size=13),
            height=480,
            margin=dict(l=40, r=20, t=60, b=40)
        )
        # 添加正态分布参考曲线
        x_range = np.linspace(
            df[selected_hist_col].min(),
            df[selected_hist_col].max(),
            200
        )
        mu, std = df[selected_hist_col].mean(), df[selected_hist_col].std(ddof=1)
        normal_pdf = stats.norm.pdf(x_range, mu, std)
        fig_hist.add_trace(go.Scatter(
            x=x_range, y=normal_pdf,
            mode='lines', name='理论正态分布',
            line=dict(color='crimson', dash='dash', width=2)
        ))
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.warning("数据集中没有数值列可用于绘制直方图。")

with tab_eda3:
    st.markdown("### 相关系数热力图（Pearson 相关系数）")
    st.markdown("*Pearson 相关系数度量变量间的线性相关程度。r ∈ [-1, 1]，接近 ±1 表示强线性相关，接近 0 表示弱相关。*")

    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr(method='pearson')

        # 使用 Plotly 绘制热力图（学术配色：蓝-白-红）
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu_r',
            zmin=-1, zmax=1,
            text=np.round(corr_matrix.values, 3),
            texttemplate='%{text}',
            textfont=dict(size=11, color='black'),
            colorbar=dict(title='Pearson r', title_font=dict(size=12)),
            xgap=2, ygap=2
        ))
        fig_corr.update_layout(
            title='Pearson 相关系数矩阵热力图',
            template='plotly_white',
            font=dict(family='Times New Roman', size=12),
            height=520,
            width=680,
            margin=dict(l=40, r=40, t=60, b=80)
        )
        # 让热力图正方形单元格
        fig_corr.update_yaxes(autorange='reversed')
        st.plotly_chart(fig_corr, use_container_width=True)

        # 导出相关系数矩阵
        st.markdown(
            download_button(corr_matrix.round(4), "correlation_matrix.csv", "导出相关系数矩阵 (CSV)"),
            unsafe_allow_html=True
        )
    else:
        st.warning("需要至少 2 个数值列才能计算相关系数矩阵。")

# ================================================================
#  =================== 模块 3：核心计量与统计模型 ==================
# ================================================================
st.markdown("---")
st.markdown("## 🔬 模块 3：核心计量与统计模型")

model_tab1, model_tab2, model_tab3 = st.tabs([
    "📐 多元线性回归 (OLS)",
    "📅 时间序列分析 (ARIMA)",
    "🔗 结构方程模型 (SEM)"
])

# ================================================================
#  3-1 多元线性回归 (OLS)
# ================================================================
with model_tab1:
    st.markdown("### 多元线性回归 (OLS)")
    st.markdown("""
    **OLS（普通最小二乘法）** 是最基本的参数估计方法。
    模型形式：**Y = Xβ + ε**，其中 ε ~ N(0, σ²)。
    OLS 通过最小化残差平方和来估计 β̂ = (X'X)⁻¹X'Y。
    """)

    if len(numeric_cols) >= 2:
        col_left, col_right = st.columns([1, 2])
        with col_left:
            # 因变量 Y 选择
            y_col = st.selectbox("选择因变量 (Y)", numeric_cols, key="ols_y")
            # 自变量 X 多选
            x_candidates = [c for c in numeric_cols if c != y_col]
            x_cols = st.multiselect(
                "选择自变量 (X)", x_candidates,
                default=x_candidates[:min(3, len(x_candidates))],
                key="ols_x",
                help="选择一个或多个自变量用于回归模型"
            )
            include_const = st.checkbox("包含截距项 (const)", value=True, key="ols_const")
            run_ols = st.button("▶ 运行 OLS 回归", type="primary", key="run_ols_btn")

        with col_right:
            if run_ols and x_cols:
                try:
                    Y = df[y_col].dropna()
                    X = df[x_cols].loc[Y.index]

                    # 再次确保 X 中没有缺失
                    valid_idx = X.dropna().index
                    Y = Y.loc[valid_idx]
                    X = X.loc[valid_idx]

                    if len(Y) < 10:
                        st.error("有效样本量过小（n < 10），无法进行回归分析。")
                    else:
                        if include_const:
                            X = sm.add_constant(X)

                        # ------------------------------------------------------------
                        # OLS 拟合
                        # ------------------------------------------------------------
                        model = sm.OLS(Y, X).fit()

                        # ------------------------------------------------------------
                        # 模型诊断
                        # ------------------------------------------------------------
                        st.markdown("#### 📋 回归结果（学术三线表）")

                        # 构建系数表
                        coef_table = pd.DataFrame({
                            '变量': model.params.index,
                            '系数 (β̂)': model.params.values,
                            '标准误 (SE)': model.bse.values,
                            't 值': model.tvalues.values,
                            'p 值': model.pvalues.values,
                        })
                        # 显著性星标
                        def sig_stars(p):
                            if p < 0.001: return '***'
                            elif p < 0.01: return '**'
                            elif p < 0.05: return '*'
                            elif p < 0.1: return '.'
                            else: return ''
                        coef_table['显著性'] = [sig_stars(p) for p in model.pvalues.values]
                        coef_table['95% CI 下限'] = model.conf_int().iloc[:, 0].values
                        coef_table['95% CI 上限'] = model.conf_int().iloc[:, 1].values

                        st.markdown(
                            render_three_line_table(coef_table, caption="表 2：OLS 回归系数估计"),
                            unsafe_allow_html=True
                        )
                        st.caption("*显著性水平：*** p<0.001, ** p<0.01, * p<0.05, . p<0.1*")

                        st.markdown(f"**R² = {model.rsquared:.5f}**  |  "
                                    f"**调整 R² = {model.rsquared_adj:.5f}**  |  "
                                    f"**F 统计量 = {model.fvalue:.4f}** (p = {model.f_pvalue:.5f})  |  "
                                    f"**AIC = {model.aic:.4f}**  |  **BIC = {model.bic:.4f}**")

                        # ------------------------------------------------------------
                        # 诊断 1：多重共线性检验（VIF）
                        # ------------------------------------------------------------
                        st.markdown("---")
                        st.markdown("#### 🔍 诊断 1：多重共线性检验（VIF）")
                        st.markdown("*方差膨胀因子 VIF > 10 表示存在严重共线性问题，OLS 估计量的方差会被大幅放大。*")
                        try:
                            if include_const:
                                X_vif = X.drop(columns=['const'])
                            else:
                                X_vif = X.copy()
                            vif_result = calc_vif(X_vif)
                            st.markdown(render_three_line_table(vif_result, caption="表 3：方差膨胀因子 (VIF) 诊断"),
                                        unsafe_allow_html=True)
                            if (vif_result['VIF'] > 10).any():
                                st.markdown(
                                    '<p class="warning-text">⚠ 警告：存在 VIF > 10 的变量，建议删除或进行正则化处理（如 Ridge/Lasso 回归）</p>',
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown('<p class="pass-text">✓ VIF 检验通过：未发现严重多重共线性</p>',
                                            unsafe_allow_html=True)
                        except Exception as vif_err:
                            st.warning(f"VIF 计算失败：{vif_err}（X 矩阵可能非满秩）")

                        # ------------------------------------------------------------
                        # 诊断 2：异方差检验（Breusch-Pagan Test）
                        # ------------------------------------------------------------
                        st.markdown("---")
                        st.markdown("#### 🔍 诊断 2：异方差检验（Breusch-Pagan / White Test）")
                        st.markdown("""
                        *异方差性指残差的方差不是常数，违反 OLS 的同方差假设。
                        BP 检验的 H₀：残差同方差；若 p < 0.05 则拒绝 H₀，存在异方差。*
                        """)
                        try:
                            residuals = model.resid
                            # BP Test
                            bp_stat, bp_pval, bp_f, bp_f_pval = het_breuschpagan(residuals, model.model.exog)
                            # White Test
                            white_stat, white_pval, white_f, white_f_pval = het_white(residuals, model.model.exog)

                            diag_hetero = pd.DataFrame({
                                '检验方法': ['Breusch-Pagan Test', 'White Test'],
                                '检验统计量': [bp_stat, white_stat],
                                'p 值': [bp_pval, white_pval],
                                '结论': [
                                    '⚠ 存在异方差' if bp_pval < 0.05 else '✓ 无异方差（同方差）',
                                    '⚠ 存在异方差' if white_pval < 0.05 else '✓ 无异方差（同方差）'
                                ]
                            })
                            st.markdown(render_three_line_table(diag_hetero, caption="表 4：异方差检验"),
                                        unsafe_allow_html=True)

                            if bp_pval < 0.05 or white_pval < 0.05:
                                st.markdown(
                                    '<p class="warning-text">⚠ 警告：异方差检验未通过！建议使用异方差稳健标准误（HC1/HC3）或考虑 WLS 估计</p>',
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown('<p class="pass-text">✓ 异方差检验通过：未发现显著异方差性</p>',
                                            unsafe_allow_html=True)
                        except Exception as het_err:
                            st.warning(f"异方差检验失败：{het_err}")

                        # ------------------------------------------------------------
                        # 诊断 3：自相关检验（Durbin-Watson）
                        # ------------------------------------------------------------
                        st.markdown("---")
                        st.markdown("#### 🔍 诊断 3：自相关检验（Durbin-Watson）")
                        st.markdown("""
                        *DW 统计量检验残差的一阶自相关。
                        DW ≈ 2：无自相关；DW → 0：正自相关；DW → 4：负自相关。
                        若数据不是时间序列，此检验可能意义有限。*
                        """)
                        dw_stat = durbin_watson(model.resid)
                        dw_diag = pd.DataFrame({
                            '指标': ['Durbin-Watson 统计量'],
                            '值': [dw_stat],
                            '诊断': ['✓ 接近 2，无自相关' if abs(dw_stat - 2) < 0.5 else
                                    '⚠ 存在正自相关' if dw_stat < 1.5 else '⚠ 存在负自相关']
                        })
                        st.markdown(render_three_line_table(dw_diag, caption="表 5：Durbin-Watson 自相关检验"),
                                    unsafe_allow_html=True)
                        if abs(dw_stat - 2) >= 0.5:
                            st.markdown(
                                '<p class="warning-text">⚠ 警告：DW 统计量显著偏离 2，残差可能存在自相关。'
                                '若数据为时间序列，建议使用 Newey-West 标准误或考虑自回归模型</p>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown('<p class="pass-text">✓ Durbin-Watson 检验通过：无明显自相关</p>',
                                        unsafe_allow_html=True)

                        # ------------------------------------------------------------
                        # 残差诊断图
                        # ------------------------------------------------------------
                        st.markdown("---")
                        st.markdown("#### 📈 残差诊断图")
                        fig_resid = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=('残差 vs 拟合值', 'Q-Q 图（正态性检验）',
                                            '残差直方图', '标准化残差序列')
                        )
                        # (1) Residual vs Fitted
                        fitted = model.fittedvalues
                        fig_resid.add_trace(
                            go.Scatter(x=fitted, y=residuals, mode='markers',
                                       marker=dict(color='#2c3e50', size=6, opacity=0.6),
                                       name='残差'),
                            row=1, col=1
                        )
                        fig_resid.add_hline(y=0, line_dash='dash', line_color='red', row=1, col=1)
                        # (2) Q-Q plot
                        qq = stats.probplot(residuals, dist="norm")
                        qq_x = qq[0][0]
                        qq_y = qq[0][1]
                        qq_line = qq[1]
                        fig_resid.add_trace(
                            go.Scatter(x=qq_x, y=qq_y, mode='markers',
                                       marker=dict(color='#2c3e50', size=6, opacity=0.6),
                                       name='Q-Q'),
                            row=1, col=2
                        )
                        fig_resid.add_trace(
                            go.Scatter(x=[qq_x.min(), qq_x.max()],
                                       y=[qq_line[1] + qq_line[0]*qq_x.min(),
                                          qq_line[1] + qq_line[0]*qq_x.max()],
                                       mode='lines', line=dict(color='red', dash='dash'),
                                       name='理论线'),
                            row=1, col=2
                        )
                        # (3) Residual histogram
                        fig_resid.add_trace(
                            go.Histogram(x=residuals.values, nbinsx=30,
                                         marker_color='#2c3e50', opacity=0.7,
                                         name='残差分布'),
                            row=2, col=1
                        )
                        # (4) Standardized residuals
                        std_resid = residuals / np.std(residuals, ddof=1)
                        fig_resid.add_trace(
                            go.Scatter(y=std_resid, mode='lines+markers',
                                       marker=dict(size=4), line=dict(color='#2c3e50'),
                                       name='标准化残差'),
                            row=2, col=2
                        )
                        fig_resid.add_hline(y=0, line_dash='dash', line_color='red', row=2, col=2)
                        fig_resid.add_hline(y=2, line_dash='dot', line_color='orange', row=2, col=2)
                        fig_resid.add_hline(y=-2, line_dash='dot', line_color='orange', row=2, col=2)

                        fig_resid.update_layout(
                            template='plotly_white',
                            font=dict(family='Times New Roman', size=11),
                            height=620,
                            showlegend=False,
                            margin=dict(l=40, r=20, t=60, b=40)
                        )
                        st.plotly_chart(fig_resid, use_container_width=True)

                        # 导出回归结果
                        st.markdown(
                            download_button(coef_table, "ols_regression_results.csv", "导出回归系数表 (CSV)"),
                            unsafe_allow_html=True
                        )

                except Exception as ols_err:
                    st.error(f"❌ OLS 回归运行失败：{str(ols_err)}")
                    import traceback
                    st.code(traceback.format_exc())

            elif run_ols and not x_cols:
                st.error("请至少选择一个自变量 (X)。")
    else:
        st.warning("需要至少 2 个数值列才能进行回归分析。")

# ================================================================
#  3-2 时间序列分析 (ARIMA)
# ================================================================
with model_tab2:
    st.markdown("### 📅 ARIMA 时间序列建模与预测")
    st.markdown("""
    **ARIMA(p, d, q)** 是经典的时间序列预测模型：
    - **AR(p)**：自回归项，使用过去的 p 个观测值
    - **I(d)**：差分阶数，使序列平稳化
    - **MA(q)**：移动平均项，使用过去的 q 个预测误差

    适用于金融资产收益率预测、宏观经济指标分析等场景。
    """)

    ts_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 时间戳列选择
    datetime_cols = df.select_dtypes(include=['datetime64', 'object']).columns.tolist()

    col_ts1, col_ts2 = st.columns([1, 2])

    with col_ts1:
        if datetime_cols:
            ts_date_col = st.selectbox("选择时间戳列（可选）", ['（无，使用行索引）'] + datetime_cols, key="ts_date")
        else:
            ts_date_col = '（无，使用行索引）'

        ts_target = st.selectbox("选择时间序列变量 (Y)", ts_cols, key="ts_target")

        st.markdown("**ARIMA 参数设置**")
        p_order = st.slider("p (自回归阶数)", 0, 10, 2, 1, key="p_slider",
                            help="AR(p)：使用前 p 个观测值进行自回归")
        d_order = st.slider("d (差分阶数)", 0, 3, 0, 1, key="d_slider",
                            help="I(d)：使序列平稳所需的差分次数")
        q_order = st.slider("q (移动平均阶数)", 0, 10, 1, 1, key="q_slider",
                            help="MA(q)：使用前 q 个预测误差")

        forecast_steps = st.slider("预测步数", 5, 100, 30, 5, key="forecast_steps",
                                   help="向未来预测的期数")

        run_arima = st.button("▶ 运行 ARIMA 模型", type="primary", key="run_arima_btn")

    with col_ts2:
        if run_arima:
            try:
                ts_data = df[ts_target].dropna().values

                if len(ts_data) < 20:
                    st.error("有效样本量过小（n < 20），无法进行时间序列分析。")
                else:
                    # ------------------------------------------------------------
                    # ADF 平稳性检验
                    # ------------------------------------------------------------
                    st.markdown("#### 📋 ADF 平稳性检验")
                    st.markdown("*ADF（Augmented Dickey-Fuller）检验的 H₀：序列存在单位根（非平稳）。p < 0.05 则拒绝 H₀。*")
                    adf_result = adfuller(ts_data, autolag='AIC')
                    adf_table = pd.DataFrame({
                        '指标': ['ADF 统计量', 'p 值', '滞后阶数', '观测数',
                                '1% 临界值', '5% 临界值', '10% 临界值'],
                        '值': [f'{adf_result[0]:.5f}', f'{adf_result[1]:.5f}',
                               f'{adf_result[2]}', f'{adf_result[3]}',
                               f'{adf_result[4]["1%"]:.5f}', f'{adf_result[4]["5%"]:.5f}',
                               f'{adf_result[4]["10%"]:.5f}']
                    })
                    st.markdown(render_three_line_table(adf_table, caption="表 6：ADF 平稳性检验"),
                                unsafe_allow_html=True)
                    if adf_result[1] < 0.05:
                        st.markdown('<p class="pass-text">✓ 序列平稳（拒绝单位根假设）</p>',
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(
                            '<p class="warning-text">⚠ 序列非平稳（无法拒绝单位根假设），建议增大参数 d 进行差分</p>',
                            unsafe_allow_html=True
                        )

                    # ------------------------------------------------------------
                    # ARIMA 拟合
                    # ------------------------------------------------------------
                    st.markdown("---")
                    st.markdown(f"#### 📈 ARIMA({p_order},{d_order},{q_order}) 模型结果")

                    model_arima = ARIMA(ts_data, order=(p_order, d_order, q_order))
                    fitted_arima = model_arima.fit()

                    # 模型摘要
                    arima_summary = pd.DataFrame({
                        '指标': ['AIC', 'BIC', 'HQIC', '对数似然值 (Log-Likelihood)'],
                        '值': [f'{fitted_arima.aic:.4f}',
                              f'{fitted_arima.bic:.4f}',
                              f'{fitted_arima.hqic:.4f}',
                              f'{fitted_arima.llf:.4f}']
                    })
                    st.markdown(render_three_line_table(arima_summary, caption="表 7：ARIMA 模型拟合优度"),
                                unsafe_allow_html=True)

                    # ------------------------------------------------------------
                    # 预测与置信区间
                    # ------------------------------------------------------------
                    st.markdown("---")
                    st.markdown("#### 🔮 预测结果与置信区间")

                    forecast_result = fitted_arima.get_forecast(steps=forecast_steps)
                    forecast_values = forecast_result.predicted_mean
                    forecast_ci = forecast_result.conf_int(alpha=0.05)

                    # 构建图表
                    fig_arima = go.Figure()

                    # 历史数据
                    hist_indices = list(range(len(ts_data)))
                    fig_arima.add_trace(go.Scatter(
                        x=hist_indices,
                        y=ts_data,
                        mode='lines',
                        name='历史数据',
                        line=dict(color='#2c3e50', width=1.8)
                    ))

                    # 预测值
                    forecast_indices = list(range(len(ts_data), len(ts_data) + forecast_steps))
                    fig_arima.add_trace(go.Scatter(
                        x=forecast_indices,
                        y=forecast_values,
                        mode='lines+markers',
                        name='ARIMA 预测值',
                        line=dict(color='#e74c3c', width=2.5),
                        marker=dict(size=5)
                    ))

                    # 95% 置信区间
                    fig_arima.add_trace(go.Scatter(
                        x=forecast_indices + forecast_indices[::-1],
                        y=np.concatenate([forecast_ci[:, 0], forecast_ci[:, 1][::-1]]),
                        fill='toself',
                        fillcolor='rgba(231, 76, 60, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='95% 置信区间',
                        hoverinfo='skip'
                    ))

                    fig_arima.add_vline(
                        x=len(ts_data) - 1,
                        line_dash='dash',
                        line_color='gray',
                        annotation_text='预测起点',
                        annotation_position='top left'
                    )

                    fig_arima.update_layout(
                        title=f'ARIMA({p_order},{d_order},{q_order}) 时间序列预测',
                        xaxis_title='时间索引 (t)',
                        yaxis_title=ts_target,
                        template='plotly_white',
                        font=dict(family='Times New Roman', size=13),
                        height=480,
                        hovermode='x unified',
                        margin=dict(l=40, r=20, t=60, b=40)
                    )
                    st.plotly_chart(fig_arima, use_container_width=True)

                    # 预测数值表
                    forecast_table = pd.DataFrame({
                        '步数': list(range(1, forecast_steps + 1)),
                        '预测值': forecast_values.round(6),
                        '95% CI 下限': forecast_ci[:, 0].round(6),
                        '95% CI 上限': forecast_ci[:, 1].round(6)
                    })
                    st.markdown(render_three_line_table(forecast_table.head(10),
                                                         caption="表 8：ARIMA 预测结果（前 10 步）"),
                                unsafe_allow_html=True)

                    # 导出预测结果
                    st.markdown(
                        download_button(forecast_table, "arima_forecast.csv", "导出 ARIMA 预测表 (CSV)"),
                        unsafe_allow_html=True
                    )

            except Exception as arima_err:
                st.error(f"❌ ARIMA 模型运行失败：{str(arima_err)}")
                import traceback
                st.code(traceback.format_exc())

# ================================================================
#  3-3 结构方程模型 (SEM)
# ================================================================
with model_tab3:
    st.markdown("### 🔗 结构方程模型 (SEM)")
    st.markdown("""
    **结构方程模型** 融合了因子分析和路径分析，用于检验观测变量与潜变量之间的因果关系假设。
    本模块使用 `semopy` 库，支持标准 SEM 语法输入。

    **示例语法**（可修改）：
    ```
    # 测量模型（Measurement Model）
    f1 =~ x1 + x2 + x3
    f2 =~ x4 + x5 + x6

    # 结构模型（Structural Model）
    f2 ~ f1 + x7
    ```
    """)

    col_sem1, col_sem2 = st.columns([1, 1])

    with col_sem1:
        sem_syntax = st.text_area(
            "输入 semopy 语法",
            value="""# 测量模型
f1 =~ x1 + x2 + x3
f2 =~ x4 + x5 + x6
# 结构模型
f2 ~ f1""",
            height=200,
            key="sem_syntax",
            help="使用 semopy 的标准语法：=~ 表示测量关系，~ 表示回归关系"
        )

        run_sem = st.button("▶ 运行 SEM 模型", type="primary", key="run_sem_btn")

    with col_sem2:
        if run_sem:
            try:
                # 尝试导入 semopy
                import semopy

                numeric_cols_for_sem = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols_for_sem) < 3:
                    st.warning("需要至少 3 个数值列才能运行 SEM。对于演示数据，将自动重命名列以匹配语法。")

                # 为方便用户，自动将数据的前 N 列名字映射为 x1, x2, ...
                # 但前提是用户语法中使用了 x1, x2 这样的命名
                import re
                vars_in_syntax = set(re.findall(r'\b(x\d+|[a-z]\w*)\b', sem_syntax))
                # 过滤掉关键字
                keywords = {'f1', 'f2', 'f3', 'f4', 'f5'}
                vars_in_syntax = vars_in_syntax - keywords

                # 尝试智能匹配：如果语法中使用了 x1, x2...，则将数值列重命名为 x1, x2...
                if all(v.startswith('x') and v[1:].isdigit() for v in vars_in_syntax):
                    # 用户使用了 x1, x2 命名
                    n_needed = max(int(v[1:]) for v in vars_in_syntax)
                    if n_needed <= len(numeric_cols_for_sem):
                        sem_df = df[numeric_cols_for_sem[:n_needed]].dropna().copy()
                        sem_df.columns = [f'x{i+1}' for i in range(n_needed)]
                    else:
                        st.error(f"语法需要 x1 到 x{n_needed}，但只有 {len(numeric_cols_for_sem)} 个数值列可用。")
                        st.stop()
                else:
                    # 直接使用原始列名
                    used_cols = [c for c in numeric_cols_for_sem if c in vars_in_syntax]
                    if len(used_cols) >= 3:
                        sem_df = df[used_cols].dropna()
                    else:
                        # 回退：用前 6 列并重命名
                        n_use = min(6, len(numeric_cols_for_sem))
                        sem_df = df[numeric_cols_for_sem[:n_use]].dropna().copy()
                        sem_df.columns = [f'x{i+1}' for i in range(n_use)]

                if len(sem_df) < 20:
                    st.error(f"有效样本量过小（n = {len(sem_df)}），SEM 需要足够大的样本量（建议 n > 100）。")

                # 构建并拟合模型
                model_sem = semopy.Model(sem_syntax)
                model_sem.fit(sem_df)

                # 获取拟合优度指标
                stats_sem = semopy.calc_stats(model_sem)

                st.markdown("#### 📊 SEM 拟合优度指标")
                fit_indices = pd.DataFrame({
                    '指标': ['χ² (Chi-square)', '自由度 (df)', 'p 值',
                            'RMSEA', 'CFI', 'GFI', 'AGFI', 'NFI', 'TLI (NNFI)', 'AIC', 'BIC'],
                    '值': [
                        f'{stats_sem["chi2"].values[0]:.4f}' if 'chi2' in stats_sem.columns else 'N/A',
                        f'{stats_sem["df"].values[0]}' if 'df' in stats_sem.columns else 'N/A',
                        f'{stats_sem["p-value"].values[0]:.5f}' if 'p-value' in stats_sem.columns else 'N/A',
                        f'{stats_sem["RMSEA"].values[0]:.5f}' if 'RMSEA' in stats_sem.columns else 'N/A',
                        f'{stats_sem["CFI"].values[0]:.5f}' if 'CFI' in stats_sem.columns else 'N/A',
                        f'{stats_sem["GFI"].values[0]:.5f}' if 'GFI' in stats_sem.columns else 'N/A',
                        f'{stats_sem["AGFI"].values[0]:.5f}' if 'AGFI' in stats_sem.columns else 'N/A',
                        f'{stats_sem["NFI"].values[0]:.5f}' if 'NFI' in stats_sem.columns else 'N/A',
                        f'{stats_sem["TLI"].values[0]:.5f}' if 'TLI' in stats_sem.columns else 'N/A',
                        f'{stats_sem["AIC"].values[0]:.5f}' if 'AIC' in stats_sem.columns else 'N/A',
                        f'{stats_sem["BIC"].values[0]:.5f}' if 'BIC' in stats_sem.columns else 'N/A',
                    ]
                })
                st.markdown(render_three_line_table(fit_indices, caption="表 9：SEM 拟合优度指标"),
                            unsafe_allow_html=True)

                # 评价标准提示
                st.markdown("""
                **常用评价标准**：
                - **RMSEA** < 0.05 优秀，< 0.08 可接受
                - **CFI** > 0.95 优秀，> 0.90 可接受
                - **GFI** > 0.95 优秀，> 0.90 可接受
                """)

                # 路径系数
                st.markdown("#### 📐 路径系数估计")
                param_df = model_sem.inspect()
                if param_df is not None and not param_df.empty:
                    # 格式化参数表
                    param_display = param_df[['lval', 'op', 'rval', 'Estimate', 'Std. Err', 'z-value', 'p-value']].copy()
                    param_display.columns = ['左变量', '关系', '右变量', '估计值', '标准误', 'z 值', 'p值']
                    param_display['显著性'] = [sig_stars(p) for p in param_display['p值']]
                    st.markdown(render_three_line_table(param_display, caption="表 10：SEM 路径系数估计"),
                                unsafe_allow_html=True)
                else:
                    st.info("无法提取路径系数表。")

            except ImportError:
                st.error("""
                ❌ `semopy` 库未安装。请运行以下命令安装：
                ```
                pip install semopy
                ```
                注意：semopy 依赖 scipy 和 numpy，可能需要额外的编译工具。
                如安装困难，可考虑使用 `pip install semopy --only-binary :all:`（若有预编译版本）。
                """)
            except Exception as sem_err:
                st.error(f"❌ SEM 模型运行失败：{str(sem_err)}")
                import traceback
                st.code(traceback.format_exc())

# ================================================================
#  =================== 模块 4：学术级输出 =========================
# ================================================================
st.markdown("---")
st.markdown("## 📄 模块 4：学术级输出与导出")
st.markdown("""
以下汇总所有已计算的分析结果。您可以点击对应按钮将表格导出为 CSV 文件，
直接嵌入 Word 论文或 LaTeX 文档中。

> **提示**：CSV 使用 UTF-8 BOM 编码，可直接用 Excel 打开，中文不会乱码。
""")

st.markdown("### 📦 批量导出数据")

# 收集所有已计算的结果并提供导出
export_col1, export_col2, export_col3 = st.columns(3)

with export_col1:
    st.markdown("**描述性统计**")
    if 'stats_result' in locals() and not stats_result.empty:
        st.markdown(
            download_button(stats_result, "01_descriptive_stats.csv", "下载描述性统计表"),
            unsafe_allow_html=True
        )
    else:
        st.caption("（请先在模块 2 中查看基本统计量）")

    st.markdown("**相关系数矩阵**")
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr(method='pearson')
        st.markdown(
            download_button(corr_matrix.round(4), "02_correlation_matrix.csv", "下载相关系数矩阵"),
            unsafe_allow_html=True
        )
    else:
        st.caption("（需要至少 2 个数值列）")

with export_col2:
    st.markdown("**原始数据（预处理后）**")
    st.markdown(
        download_button(df, "00_preprocessed_data.csv", "下载预处理后的数据"),
        unsafe_allow_html=True
    )

    st.markdown("**数据预处理日志**")
    preproc_log = pd.DataFrame({
        '操作': ['缺失值处理策略', 'Z-score 标准化', '原始行数', '预处理后行数'],
        '设置': [missing_strategy if 'missing_strategy' in dir() else 'N/A',
                '是' if 'standardize' in dir() and standardize else '否',
                len(df_raw) if 'df_raw' in dir() else 'N/A',
                len(df)]
    })
    st.markdown(
        download_button(preproc_log, "03_preprocessing_log.csv", "下载预处理日志"),
        unsafe_allow_html=True
    )

with export_col3:
    st.markdown("**使用说明**")
    st.markdown("""
    <div style="background:#f8f9fa;padding:15px;border-radius:8px;font-size:13px;">
    <b>📝 学术三线表使用指南：</b><br><br>
    1. 点击上方按钮下载 CSV 文件<br>
    2. 用 Excel 打开 CSV 文件<br>
    3. 在 Word 中插入表格，导入数据<br>
    4. 设置三线表格式：<br>
       &nbsp;&nbsp;• 顶线：1.5pt 粗线<br>
       &nbsp;&nbsp;• 表头线：0.75pt 细线<br>
       &nbsp;&nbsp;• 底线：1.5pt 粗线<br>
    5. 删除所有竖线（三线表无竖线）<br>
    </div>
    """, unsafe_allow_html=True)

# ================================================================
#  页脚
# ================================================================
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#999;font-size:12px;">'
    '📊 高级统计与计量经济学可视化看板 | Built with Streamlit + statsmodels + Plotly | '
    '适用于学术论文数据分析</p>',
    unsafe_allow_html=True
)

# ================================================================
#  启动提示：在终端运行 `streamlit run quant_stat_pro.py`
# ================================================================
if __name__ == "__main__":
    # 当直接运行此文件时不做额外操作（streamlit run 会自动解析页面内容）
    pass
