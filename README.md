# 📊 高级统计与计量经济学可视化看板

Advanced Statistics & Econometrics Dashboard for Academic Research.

## 功能模块

- **数据控制台**：CSV/Excel 上传、缺失值处理、Z-score 标准化
- **描述性统计与 EDA**：基本统计量、交互式直方图、相关系数热力图
- **核心计量模型**：
  - 多元线性回归 (OLS) + VIF/异方差/自相关诊断
  - ARIMA 时间序列预测 (含置信区间)
  - 结构方程模型 (SEM, semopy 语法)
- **学术级输出**：三线表格式、CSV 导出

## 本地运行

```bash
pip install -r requirements.txt
streamlit run quant_stat_pro.py
```

## 在线访问

部署于 [Streamlit Community Cloud](https://streamlit.io/cloud)
