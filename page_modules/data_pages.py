"""
数据相关页面模块
包含数据上传、数据概览、数据清洗、探索性分析页面
"""
import streamlit as st
import pandas as pd
import numpy as np
from data_loader import (
    load_data, calculate_data_quality_score, 
    get_data_overview, clean_data
)
from visualization import (
    create_histogram, create_box_plot, create_scatter_plot,
    create_bar_chart, create_correlation_heatmap, 
    create_scatter_matrix, recommend_charts,
    create_violin_plot, create_density_contour,
    create_parallel_coordinates
)


def show_data_upload():
    """数据上传页面"""
    st.markdown("### 📤 数据上传")
    
    col1, col2 = st.columns([2, 1], gap="large")
    
    with col1:
        with st.container():
            st.write("##### 上传文件")
            uploaded_file = st.file_uploader(
                "选择数据文件 (CSV, Excel, JSON)",
                type=['csv', 'xlsx', 'xls', 'json'],
                help="支持 CSV、Excel (.xlsx, .xls) 和 JSON 格式"
            )
            
            if uploaded_file is not None:
                file_type = uploaded_file.name.split('.')[-1].lower()
                
                # CSV 分隔符选择
                sep = None
                if file_type == 'csv':
                    st.write("CSV 选项")
                    sep_option = st.radio(
                        "分隔符",
                        ["自动检测", "逗号 (,)", "分号 (;)", "制表符 (\\t)"],
                        horizontal=True,
                        index=0,
                        key='csv_sep'
                    )
                    sep_map = {
                        "自动检测": None,
                        "逗号 (,)": ',',
                        "分号 (;)": ';',
                        "制表符 (\\t)": '\t'
                    }
                    sep = sep_map[sep_option]
                
                if st.button("🚀 加载数据", type="primary", use_container_width=True):
                    with st.spinner("正在解析数据..."):
                        df = load_data(uploaded_file, file_type, sep=sep)
                        
                        if df is not None:
                            if len(df.columns) <= 1:
                                st.warning("⚠️ 警告：数据只有 1 列，可能是分隔符设置不正确。")
                                st.info("💡 提示：请尝试更改 CSV 分隔符选项。")
                                st.dataframe(df.head(5), use_container_width=True)
                            else:
                                st.session_state.df = df
                                st.session_state.df_cleaned = df.copy()
                                st.toast("✅ 数据加载成功！", icon="🎉")
                                
                                st.session_state.data_overview = get_data_overview(df)
                                st.session_state.quality_scores = calculate_data_quality_score(df)
                                st.rerun()
                        else:
                            st.error("❌ 数据加载失败，请检查文件格式。")

    with col2:
        with st.expander("📋 数据预览", expanded=True):
            if st.session_state.df is not None:
                st.dataframe(st.session_state.df.head(10), use_container_width=True)
                st.caption(f"显示前 10 行 / 共 {st.session_state.df.shape[0]} 行")
            else:
                st.info("暂无数据，请先上传。")
        
        with st.expander("ℹ️ 快速指南"):
            st.markdown("""
            **支持格式：**
            - CSV (支持多种分隔符)
            - Excel (.xlsx, .xls)
            - JSON
            
            **提示：**
            - 确保第一行为列名
            - 大文件 (>50MB) 可能加载较慢
            """)


def show_data_overview():
    """数据概览页面"""
    st.markdown("### 📋 数据概览")
    
    if st.session_state.df is None:
        st.warning("⚠️ 请先上传数据文件")
        return
    
    df = st.session_state.df
    
    # 关键指标 Dashboard
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("行数 (Samples)", f"{df.shape[0]:,}")
    with col2:
        st.metric("列数 (Features)", f"{df.shape[1]}")
    with col3:
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("内存占用", f"{memory_mb:.2f} MB")
    with col4:
        if st.session_state.quality_scores:
            score = st.session_state.quality_scores['overall_score']
            st.metric("质量评分", f"{score:.1f}", delta=None)

    st.divider()

    # 详细信息 Tabs
    tab1, tab2, tab3 = st.tabs(["数据预览", "列详细信息", "质量报告"])
    
    with tab1:
        st.dataframe(df.head(50), use_container_width=True)
        
    with tab2:
        if st.session_state.data_overview:
            overview = st.session_state.data_overview
            col_info = pd.DataFrame({
                '列名': overview['columns'],
                '类型': [str(overview['dtypes'].get(col, 'unknown')) for col in overview['columns']],
                '缺失值': [overview['missing_values'].get(col, 0) for col in overview['columns']],
                '缺失率 (%)': [overview['missing_percentage'].get(col, 0) for col in overview['columns']]
            })
            st.dataframe(col_info, use_container_width=True, hide_index=True)
            
            st.markdown("##### 数值型变量统计")
            st.dataframe(df.describe(), use_container_width=True)

    with tab3:
        if st.session_state.quality_scores:
            scores = st.session_state.quality_scores
            cols = st.columns(3)
            cols[0].metric("完整性", f"{scores['completeness']:.1f}%")
            cols[1].metric("唯一性", f"{scores['uniqueness']:.1f}%")
            cols[2].metric("一致性", f"{scores['consistency']:.1f}%")
            
            if scores['missing_count'] > 0:
                st.warning(f"⚠️ 发现 {scores['missing_count']} 个缺失值")
            if scores['duplicate_count'] > 0:
                st.warning(f"⚠️ 发现 {scores['duplicate_count']} 行重复数据")


def show_data_cleaning():
    """数据清洗页面"""
    st.markdown("### 🧹 数据清洗")
    
    if st.session_state.df is None:
        st.warning("⚠️ 请先上传数据文件")
        return
    
    df = st.session_state.df_cleaned if st.session_state.df_cleaned is not None else st.session_state.df
    
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.markdown("#### 清洗配置")
        
        with st.expander("1. 缺失值处理", expanded=True):
            missing_strategy = st.radio(
                "处理策略",
                ["保留", "删除", "填充"],
                key='clean_missing_strat'
            )
            missing_method = "mean"
            if missing_strategy == "填充":
                missing_method = st.selectbox(
                    "填充方法 (数值型)",
                    ["均值", "中位数", "众数", "固定值0"],
                    key='clean_missing_method'
                )
        
        with st.expander("2. 重复值处理"):
            remove_duplicates = st.checkbox("删除重复行", value=False)
        
        with st.expander("3. 异常值处理"):
            outlier_method = st.selectbox(
                "检测方法",
                ["不处理", "Z-score方法", "IQR方法"],
                key='clean_outlier_method'
            )
            outlier_threshold = 3.0
            outlier_action = "删除"
            if outlier_method != "不处理":
                outlier_threshold = st.slider("阈值 (Z-score / IQR倍数)", 1.0, 5.0, 3.0, 0.1)
                outlier_action = st.radio("处理方式", ["删除", "修正为边界值"])
        
        if st.button("🚀 执行清洗", type="primary", use_container_width=True):
            cleaning_options = {
                'missing_value_strategy': 'none' if missing_strategy == "保留" else ('drop' if missing_strategy == "删除" else 'fill'),
                'missing_value_method': {'均值': 'mean', '中位数': 'median', '众数': 'mode', '固定值0': 'zero'}.get(missing_method, 'mean'),
                'remove_duplicates': remove_duplicates,
                'outlier_method': {'不处理': 'none', 'Z-score方法': 'zscore', 'IQR方法': 'iqr'}.get(outlier_method, 'none'),
                'outlier_threshold': outlier_threshold,
                'outlier_action': outlier_action.lower()
            }
            
            with st.spinner("正在清洗数据..."):
                df_cleaned, cleaning_log = clean_data(df, cleaning_options)
                st.session_state.df_cleaned = df_cleaned
                st.session_state.cleaning_log = cleaning_log
                
                st.toast("清洗完成！", icon="✨")
                st.success(f"数据行数变化: {len(df)} → {len(df_cleaned)}")

    with col2:
        st.markdown("#### 结果预览")
        
        tab1, tab2 = st.tabs(["清洗后数据", "清洗日志"])
        
        with tab1:
            if st.session_state.df_cleaned is not None:
                st.dataframe(st.session_state.df_cleaned.head(100), use_container_width=True)
            else:
                st.info("暂无清洗后的数据")
        
        with tab2:
            if st.session_state.cleaning_log:
                for log in st.session_state.cleaning_log:
                    st.info(f"📝 {log}")
            else:
                st.caption("暂无清洗操作记录")


def show_eda():
    """探索性分析页面"""
    st.markdown("### 🔍 探索性数据分析 (EDA)")
    
    df = st.session_state.df_cleaned if st.session_state.df_cleaned is not None else st.session_state.df
    
    if df is None:
        st.warning("⚠️ 请先上传数据文件")
        return
    
    col_params, col_chart = st.columns([1, 3], gap="medium")
    
    chart_type_tab = st.tabs(["单变量", "双变量", "多变量", "相关性"])
    
    with chart_type_tab[0]:
        with col_params:
            st.markdown("##### 配置")
            col_selected = st.selectbox("选择列", df.columns.tolist(), key='eda_1_col')
            if col_selected:
                recommended = recommend_charts(df, col_selected)
                chart_mode = st.selectbox("图表类型", recommended, key='eda_1_mode')
                
                bins = 30
                if chart_mode == 'histogram':
                    bins = st.slider("分组数量 (Bins)", 5, 100, 30, key='eda_1_bins')
                
                group_col = None
                if chart_mode == 'violin':
                    group_col = st.selectbox("分组列 (可选)", [None] + df.columns.tolist(), key='eda_1_group')

        with col_chart:
            if col_selected:
                try:
                    st.markdown(f"#### {col_selected} - {chart_mode} 分析")
                    if chart_mode == 'histogram':
                        fig = create_histogram(df, col_selected, bins=bins)
                    elif chart_mode == 'box_plot':
                        fig = create_box_plot(df, col_selected)
                    elif chart_mode == 'violin':
                        fig = create_violin_plot(df, col_selected, by=group_col)
                    elif chart_mode == 'bar_chart':
                        fig = create_bar_chart(df, col_selected)
                    else:
                        fig = None
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("无法生成图表，请检查数据列类型")
                except Exception as e:
                    st.error(f"生成图表时出错: {str(e)}")

    with chart_type_tab[1]:
        with col_params:
            st.markdown("##### 配置")
            x_col = st.selectbox("X 轴", df.columns.tolist(), key='eda_2_x')
            y_col = st.selectbox("Y 轴", df.columns.tolist(), key='eda_2_y')
            color_col = st.selectbox("颜色分组 (可选)", [None] + df.columns.tolist(), key='eda_2_color')
            plot_type_2 = st.radio("展示方式", ["散点图", "密度等高线"], key='eda_2_type')

        with col_chart:
            if x_col and y_col:
                try:
                    st.markdown(f"#### {x_col} vs {y_col}")
                    if plot_type_2 == "散点图":
                        fig = create_scatter_plot(df, x_col, y_col, color_col=color_col)
                    else:
                        fig = create_density_contour(df, x_col, y_col)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("无法生成图表，请检查数据列类型")
                except Exception as e:
                    st.error(f"生成图表时出错: {str(e)}")

    with chart_type_tab[2]:
        with col_params:
            st.markdown("##### 配置")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                selected_cols = st.multiselect(
                    "选择列 (2-5个)", 
                    numeric_cols, 
                    default=numeric_cols[:min(4, len(numeric_cols))],
                    key='eda_3_cols'
                )
                view_type = st.radio("视图", ["散点矩阵", "平行坐标"], key='eda_3_type')
            else:
                st.warning("数值型列不足 2 个")
                selected_cols = []

        with col_chart:
            if len(selected_cols) >= 2:
                try:
                    st.markdown(f"#### 多变量分析 ({view_type})")
                    if view_type == "散点矩阵":
                        fig = create_scatter_matrix(df, columns=selected_cols)
                    else:
                        fig = create_parallel_coordinates(df, columns=selected_cols)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("无法生成图表，请检查数据")
                except Exception as e:
                    st.error(f"生成图表时出错: {str(e)}")

    with chart_type_tab[3]:
        st.markdown("#### 特征相关性热力图")
        fig = create_correlation_heatmap(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("需要至少 2 个数值型列")

