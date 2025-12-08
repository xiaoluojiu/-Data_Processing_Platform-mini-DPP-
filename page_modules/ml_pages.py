"""
机器学习相关页面模块
包含机器学习分析、监督学习、聚类、降维、关联规则等页面
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from collections import Counter

# 导入机器学习模块
from ml_models import (
    prepare_data_for_ml, train_regression_model, 
    train_classification_model, perform_kmeans_clustering,
    find_optimal_k, perform_dbscan_clustering,
    perform_pca, perform_apriori, perform_fpgrowth
)
from ml_visualization import (
    plot_confusion_matrix_heatmap, plot_roc_curve,
    plot_learning_curve, plot_decision_tree_structure,
    plot_prediction_distribution, plot_silhouette_plot,
    plot_pca_scatter, plot_rules_heatmap, plot_rules_sankey,
    plot_residual_analysis, plot_prediction_vs_actual,
    plot_feature_interaction_heatmap, plot_error_distribution,
    plot_classification_report_heatmap
)
from visualization import (
    create_residual_plot, create_feature_importance_plot
)


def show_ml_analysis():
    """机器学习页面"""
    st.markdown("### 🤖 机器学习与建模")
    
    df = st.session_state.df_cleaned if st.session_state.df_cleaned is not None else st.session_state.df
    
    if df is None:
        st.warning("⚠️ 请先上传数据文件")
        return
    
    # 顶部任务选择条
    task_type = option_menu(
        menu_title=None,
        options=["分类", "回归", "聚类", "降维", "关联规则"],
        icons=["diagram-3", "graph-up-arrow", "boxes", "fullscreen-exit", "link"],
        orientation="horizontal",
        styles={"nav-link": {"font-size": "0.9rem"}}
    )
    
    st.divider()
    
    if task_type in ["分类", "回归"]:
        _show_ml_supervised(df, task_type)
    elif task_type == "聚类":
        _show_ml_clustering(df)
    elif task_type == "降维":
        _show_ml_pca(df)
    elif task_type == "关联规则":
        _show_ml_rules(df)


def _show_ml_supervised(df, task_type):
    """监督学习子页面"""
    col_settings, col_results = st.columns([1, 3], gap="large")
    
    with col_settings:
        st.markdown("#### ⚙️ 模型配置")
        target_col = st.selectbox("目标变量", df.columns.tolist(), key='ml_target')
        
        if not target_col:
            return

        # 自动建议
        target_dtype = df[target_col].dtype
        unique_count = df[target_col].nunique()
        is_numeric = pd.api.types.is_numeric_dtype(target_dtype)
        
        suggested = "分类" if (not is_numeric or unique_count <= 20) else "回归"
        if task_type != suggested:
            st.info(f"💡 检测到目标变量特性，建议使用【{suggested}】任务")

        # 特征选择
        available_features = [c for c in df.columns if c != target_col]
        use_all_features = st.checkbox("使用所有特征", value=True)
        if use_all_features:
            selected_features = available_features
        else:
            selected_features = st.multiselect("选择特征", available_features, default=available_features[:5])
        
        st.divider()
        
        # 模型选择
        auto_optimize = st.checkbox("🤖 自动优化参数", value=False, help="自动搜索最优参数（耗时较长）")
        
        if task_type == "分类":
            model_type = st.selectbox("选择模型", ["逻辑回归", "朴素贝叶斯", "KNN", "决策树"])
            tree_algorithm = None
            if model_type == "决策树":
                tree_algorithm = st.selectbox("算法", ["CART", "ID3 (信息增益)", "C4.5 (增益率)"])
        else:
            model_type = st.selectbox("选择模型", ["线性回归", "KNN回归", "决策树回归"])
            tree_algorithm = None

        # 简单的参数面板 (手动模式下)
        params = {}
        if not auto_optimize:
            if "KNN" in model_type:
                params['n_neighbors'] = st.slider("K值", 1, 20, 5)
            elif "决策树" in model_type:
                params['max_depth'] = st.slider("最大深度", 1, 20, 10)
        
        if tree_algorithm:
            params['tree_algorithm'] = tree_algorithm
        
        st.divider()
        
        # 可视化选项（前置，训练前选择）
        st.markdown("#### 📈 可视化选项（训练前选择）")
        with st.expander("选择要生成的可视化图表", expanded=False):
            if task_type == "分类":
                st.markdown("**基础评估**")
                viz_cm = st.checkbox("混淆矩阵", value=True, key="viz_cm")
                viz_roc = st.checkbox("ROC曲线", value=False, key="viz_roc")
                viz_report = st.checkbox("分类报告热力图", value=False, key="viz_report")
                
                st.markdown("**高级分析**")
                viz_lc = st.checkbox("学习曲线", value=False, key="viz_lc", help="耗时较长")
                viz_fi = st.checkbox("特征重要性", value=True, key="viz_fi")
                viz_inter = st.checkbox("特征交互热力图", value=False, key="viz_inter")
                viz_dist = st.checkbox("预测分布", value=False, key="viz_dist")
                viz_tree = st.checkbox("决策树结构", value=False, key="viz_tree")
                
                # 保存到session_state
                st.session_state.viz_options[task_type] = {
                    "混淆矩阵": viz_cm,
                    "ROC曲线": viz_roc,
                    "分类报告热力图": viz_report,
                    "学习曲线": viz_lc,
                    "特征重要性": viz_fi,
                    "特征交互热力图": viz_inter,
                    "预测分布": viz_dist,
                    "决策树结构": viz_tree
                }
            else:  # 回归
                st.markdown("**基础评估**")
                viz_res = st.checkbox("残差图", value=True, key="viz_res")
                viz_pva = st.checkbox("预测值vs真实值", value=True, key="viz_pva")
                viz_err = st.checkbox("误差分布", value=False, key="viz_err")
                
                st.markdown("**高级分析**")
                viz_ra = st.checkbox("残差分析（详细）", value=False, key="viz_ra")
                viz_lc = st.checkbox("学习曲线", value=False, key="viz_lc", help="耗时较长")
                viz_fi = st.checkbox("特征重要性", value=True, key="viz_fi")
                viz_inter = st.checkbox("特征交互热力图", value=False, key="viz_inter")
                
                # 保存到session_state
                st.session_state.viz_options[task_type] = {
                    "残差图": viz_res,
                    "预测值vs真实值": viz_pva,
                    "误差分布": viz_err,
                    "残差分析": viz_ra,
                    "学习曲线": viz_lc,
                    "特征重要性": viz_fi,
                    "特征交互热力图": viz_inter
                }

        train_btn = st.button("🚀 开始训练", type="primary", use_container_width=True)

    with col_results:
        if train_btn and len(selected_features) > 0:
            with st.spinner(f"正在训练 {model_type}..."):
                try:
                    task_code = 'classification' if task_type == "分类" else 'regression'
                    
                    # 1. 准备数据
                    X, y, feature_names, _ = prepare_data_for_ml(
                        df, target_col, feature_columns=selected_features, task_type=task_code
                    )
                    
                    # 2. 划分
                    stratify = y if task_code == 'classification' and y.dtype != 'float' else None
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42, stratify=stratify
                        )
                    except:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )

                    # 3. 训练
                    if task_type == "分类":
                        model_map = {
                            "逻辑回归": "logistic", "朴素贝叶斯": "naive_bayes", "KNN": "knn",
                            "决策树": "tree"
                        }
                        results = train_classification_model(
                            X_train, y_train, X_test, y_test, 
                            model_type=model_map[model_type], auto_optimize=auto_optimize, **params
                        )
                    else:
                        model_map = {
                            "线性回归": "linear", "KNN回归": "knn", "决策树回归": "tree"
                        }
                        results = train_regression_model(
                            X_train, y_train, X_test, y_test, 
                            model_type=model_map[model_type], auto_optimize=auto_optimize, **params
                        )
                    
                    st.session_state.ml_results = results
                    
                    # 保存训练配置
                    train_config = {
                        'task_type': task_type,
                        'model_type': model_type,
                        'target_col': target_col,
                        'selected_features': selected_features,
                        'auto_optimize': auto_optimize,
                        'params': params,
                        'feature_names': feature_names,
                        'train_size': len(X_train),
                        'test_size': len(X_test)
                    }
                    st.session_state.ml_train_config = train_config
                    
                    st.toast("训练完成！", icon="✅")
                    
                    # 4. 展示结果（传入可视化选项）
                    viz_opts = st.session_state.viz_options.get(task_type, {})
                    _show_supervised_results(results, task_type, feature_names, X_train, y_train, X_test, y_test, 
                                            train_config, viz_opts)
                    
                except Exception as e:
                    st.error(f"训练失败: {str(e)}")
                    st.exception(e)


def _show_supervised_results(results, task_type, feature_names, X_train, y_train, X_test, y_test, train_config, viz_opts):
    """展示监督学习结果"""
    st.markdown("#### 📊 训练结果")
    
    # 指标卡片
    metrics = results['metrics']
    cols = st.columns(len(metrics))
    for idx, (k, v) in enumerate(metrics.items()):
        if isinstance(v, (int, float)):
            cols[idx].metric(k, f"{v:.4f}")
            
    if results.get('best_params'):
        with st.expander("🔎 最优参数配置", expanded=True):
            st.json(results['best_params'])
    
    # 存储生成的图表
    generated_charts = {}
    
    st.markdown("---")
    st.markdown("#### 📈 可视化结果（根据训练前选择的选项生成）")
    
    # 根据训练前选择的可视化选项生成图表
    if task_type == "分类":
        # 基础评估
        if viz_opts.get("混淆矩阵", False):
            st.markdown("##### 混淆矩阵")
            fig_cm = plot_confusion_matrix_heatmap(y_test, results['y_pred'])
            st.plotly_chart(fig_cm, use_container_width=True)
            generated_charts["混淆矩阵"] = fig_cm
        
        if viz_opts.get("ROC曲线", False) and hasattr(results['model'], 'predict_proba'):
            st.markdown("##### ROC曲线")
            fig_roc = plot_roc_curve(y_test, results['model'].predict_proba(X_test))
            st.plotly_chart(fig_roc, use_container_width=True)
            generated_charts["ROC曲线"] = fig_roc
        
        if viz_opts.get("分类报告热力图", False):
            st.markdown("##### 分类报告热力图")
            fig_report = plot_classification_report_heatmap(y_test, results['y_pred'])
            if fig_report:
                st.plotly_chart(fig_report, use_container_width=True)
                generated_charts["分类报告热力图"] = fig_report
        
        # 高级分析
        if viz_opts.get("学习曲线", False):
            st.markdown("##### 学习曲线")
            with st.spinner("生成学习曲线中（可能需要一些时间）..."):
                fig_lc = plot_learning_curve(results['model'], X_train, y_train)
                st.plotly_chart(fig_lc, use_container_width=True)
                generated_charts["学习曲线"] = fig_lc
        
        if viz_opts.get("特征重要性", False) and results.get('feature_importance'):
            st.markdown("##### 特征重要性")
            try:
                if isinstance(results['feature_importance'], dict) and len(results['feature_importance']) > 0:
                    fig = create_feature_importance_plot(results['feature_importance'])
                    st.plotly_chart(fig, use_container_width=True)
                    generated_charts["特征重要性"] = fig
                else:
                    st.warning("特征重要性数据为空或格式不正确")
            except Exception as e:
                st.error(f"生成特征重要性图失败: {str(e)}")
        
        if viz_opts.get("特征交互热力图", False):
            st.markdown("##### 特征交互热力图")
            with st.spinner("生成特征交互热力图中..."):
                try:
                    fig_inter = plot_feature_interaction_heatmap(X_train, feature_names)
                    if fig_inter:
                        st.plotly_chart(fig_inter, use_container_width=True)
                        generated_charts["特征交互热力图"] = fig_inter
                except Exception as e:
                    st.error(f"生成特征交互热力图失败: {str(e)}")
        
        if viz_opts.get("预测分布", False):
            st.markdown("##### 预测分布")
            try:
                fig_dist = plot_prediction_distribution(y_test, results['y_pred'], task_type='classification')
                if fig_dist:
                    st.plotly_chart(fig_dist, use_container_width=True)
                    generated_charts["预测分布"] = fig_dist
            except Exception as e:
                st.error(f"生成预测分布图失败: {str(e)}")
        
        if viz_opts.get("决策树结构", False):
            st.markdown("##### 决策树结构")
            try:
                model_str = str(type(results['model']).__name__).lower()
                if 'tree' in model_str or 'decision' in model_str:
                    img = plot_decision_tree_structure(results['model'], feature_names, max_depth=5)
                    if img:
                        st.image(f"data:image/png;base64,{img}")
                        generated_charts["决策树结构"] = img
                    else:
                        st.warning("无法生成决策树结构图")
                else:
                    st.info("当前模型不是决策树，无法显示树结构")
            except Exception as e:
                st.error(f"生成决策树结构失败: {str(e)}")
    
    else:  # 回归任务
        # 基础评估
        if viz_opts.get("残差图", False):
            st.markdown("##### 残差图")
            fig_res = create_residual_plot(y_test, results['y_pred'])
            st.plotly_chart(fig_res, use_container_width=True)
            generated_charts["残差图"] = fig_res
        
        if viz_opts.get("预测值vs真实值", False):
            st.markdown("##### 预测值 vs 真实值")
            fig_pva = plot_prediction_vs_actual(y_test, results['y_pred'], task_type='regression')
            st.plotly_chart(fig_pva, use_container_width=True)
            generated_charts["预测值vs真实值"] = fig_pva
        
        if viz_opts.get("误差分布", False):
            st.markdown("##### 误差分布")
            fig_err = plot_error_distribution(y_test, results['y_pred'])
            st.plotly_chart(fig_err, use_container_width=True)
            generated_charts["误差分布"] = fig_err
        
        # 高级分析
        if viz_opts.get("残差分析", False):
            st.markdown("##### 残差分析（详细）")
            with st.spinner("生成残差分析中..."):
                fig_ra = plot_residual_analysis(y_test, results['y_pred'])
                st.plotly_chart(fig_ra, use_container_width=True)
                generated_charts["残差分析"] = fig_ra
        
        if viz_opts.get("学习曲线", False):
            st.markdown("##### 学习曲线")
            with st.spinner("生成学习曲线中（可能需要一些时间）..."):
                fig_lc = plot_learning_curve(results['model'], X_train, y_train)
                st.plotly_chart(fig_lc, use_container_width=True)
                generated_charts["学习曲线"] = fig_lc
        
        if viz_opts.get("特征重要性", False) and results.get('feature_importance'):
            st.markdown("##### 特征重要性")
            try:
                if isinstance(results['feature_importance'], dict) and len(results['feature_importance']) > 0:
                    fig = create_feature_importance_plot(results['feature_importance'])
                    st.plotly_chart(fig, use_container_width=True)
                    generated_charts["特征重要性"] = fig
                else:
                    st.warning("特征重要性数据为空或格式不正确")
            except Exception as e:
                st.error(f"生成特征重要性图失败: {str(e)}")
        
        if viz_opts.get("特征交互热力图", False):
            st.markdown("##### 特征交互热力图")
            with st.spinner("生成特征交互热力图中..."):
                try:
                    fig_inter = plot_feature_interaction_heatmap(X_train, feature_names)
                    if fig_inter:
                        st.plotly_chart(fig_inter, use_container_width=True)
                        generated_charts["特征交互热力图"] = fig_inter
                except Exception as e:
                    st.error(f"生成特征交互热力图失败: {str(e)}")


def _show_ml_clustering(df):
    """聚类分析子页面"""
    col_settings, col_results = st.columns([1, 3], gap="large")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("聚类需要至少 2 个数值型特征")
        return

    with col_settings:
        st.markdown("#### ⚙️ 聚类配置")
        algo = st.selectbox("算法", ["K-means", "DBSCAN"])
        cols = st.multiselect("特征选择", numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))])
        
        # 数据标准化选项
        normalize = st.checkbox("标准化数据", value=True, help="建议开启，可提升聚类效果")
        
        # 大数据集采样选项
        max_samples = st.number_input("最大样本数（性能优化）", min_value=100, max_value=10000, value=5000, step=500,
                                      help="超过此数量的数据将进行采样，以提升性能")
        
        params = {}
        if algo == "K-means":
            if cols:
                X_temp = df[cols].dropna()
                max_k = min(10, len(X_temp) // 2) if len(X_temp) > 0 else 10
                params['n_clusters'] = st.slider("聚类数 (K)", 2, max(2, max_k), min(3, max_k))
            else:
                params['n_clusters'] = 3
            
            # 寻找最优K功能
            if st.button("🔍 寻找最优 K", use_container_width=True):
                st.session_state.find_optimal_k = True
        else:
            params['eps'] = st.slider("Eps (半径)", 0.1, 5.0, 0.5, 0.1)
            params['min_samples'] = st.slider("Min Samples", 2, 20, 5)
            
        run_cluster = st.button("🚀 执行聚类", type="primary", use_container_width=True)

    with col_results:
        # 寻找最优K
        if st.session_state.get('find_optimal_k', False) and cols:
            st.session_state.find_optimal_k = False
            with st.spinner("正在寻找最优K值（这可能需要一些时间）..."):
                try:
                    X = df[cols].dropna()
                    if len(X) > max_samples:
                        X = X.sample(n=max_samples, random_state=42)
                        st.info(f"⚠️ 数据量较大，已采样 {max_samples} 个样本进行分析")
                    
                    if normalize:
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
                    
                    optimal_result = find_optimal_k(X, max_k=min(10, len(X) // 2))
                    
                    # 绘制肘部法则和轮廓系数图
                    col_a, col_b = st.columns(2)
                    with col_a:
                        fig_elbow = go.Figure()
                        fig_elbow.add_trace(go.Scatter(
                            x=optimal_result['k_range'],
                            y=optimal_result['inertias'],
                            mode='lines+markers',
                            name='惯性',
                            line=dict(color='blue', width=2)
                        ))
                        fig_elbow.update_layout(
                            title='肘部法则 (Elbow Method)',
                            xaxis_title='K值',
                            yaxis_title='惯性 (Inertia)',
                            template='plotly_white',
                            height=400
                        )
                        st.plotly_chart(fig_elbow, use_container_width=True)
                    
                    with col_b:
                        fig_sil = go.Figure()
                        fig_sil.add_trace(go.Scatter(
                            x=optimal_result['k_range'],
                            y=optimal_result['silhouette_scores'],
                            mode='lines+markers',
                            name='轮廓系数',
                            line=dict(color='green', width=2)
                        ))
                        fig_sil.add_vline(
                            x=optimal_result['optimal_k'],
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"最优K={optimal_result['optimal_k']}"
                        )
                        fig_sil.update_layout(
                            title='轮廓系数分析',
                            xaxis_title='K值',
                            yaxis_title='轮廓系数',
                            template='plotly_white',
                            height=400
                        )
                        st.plotly_chart(fig_sil, use_container_width=True)
                    
                    st.success(f"✅ 推荐的最优K值: **{optimal_result['optimal_k']}** (轮廓系数: {max(optimal_result['silhouette_scores']):.4f})")
                    
                except Exception as e:
                    st.error(f"寻找最优K失败: {str(e)}")
                    st.exception(e)
        
        # 执行聚类
        if run_cluster and cols:
            if len(cols) < 2:
                st.error("⚠️ 聚类需要至少选择 2 个特征")
            else:
                with st.spinner("正在聚类..."):
                    try:
                        X = df[cols].dropna()
                        original_size = len(X)
                        
                        # 数据采样（如果数据量太大）
                        if len(X) > max_samples:
                            X = X.sample(n=max_samples, random_state=42)
                            st.warning(f"⚠️ 数据量较大 ({original_size} 行)，已采样 {max_samples} 行进行分析以提升性能")
                        
                        if len(X) < 2:
                            st.error("⚠️ 有效数据样本不足（删除缺失值后少于2个）")
                        else:
                            # 数据标准化
                            if normalize:
                                from sklearn.preprocessing import StandardScaler
                                scaler = StandardScaler()
                                X_scaled = scaler.fit_transform(X)
                                X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
                            
                            if algo == "K-means":
                                if params['n_clusters'] > len(X):
                                    st.error(f"⚠️ 聚类数 ({params['n_clusters']}) 不能大于样本数 ({len(X)})")
                                else:
                                    res = perform_kmeans_clustering(X, params['n_clusters'])
                            else:
                                res = perform_dbscan_clustering(X, params['eps'], params['min_samples'])
                            
                            st.session_state.ml_results = res
                            
                            # 显示结果
                            if res.get('silhouette_score') is not None:
                                st.success(f"✅ 聚类完成！轮廓系数: {res['silhouette_score']:.4f}")
                            else:
                                st.success("✅ 聚类完成！")
                            
                            if algo == "DBSCAN":
                                st.info(f"📊 发现 {res.get('n_clusters', 0)} 个聚类，{res.get('n_noise', 0)} 个噪声点")
                            
                            # 保存聚类配置和结果
                            cluster_config = {
                                'task_type': '聚类',
                                'algorithm': algo,
                                'selected_features': cols,
                                'normalize': normalize,
                                'params': params,
                                'n_samples': len(X)
                            }
                            st.session_state.ml_train_config = cluster_config
                            
                            # 可视化
                            tab1, tab2, tab3, tab4 = st.tabs(["散点分布", "轮廓系数分析", "聚类分析", "特征分布"])
                            with tab1:
                                if len(cols) >= 2:
                                    # 限制散点图点数，避免浏览器卡顿
                                    plot_data = df.loc[X.index].copy()
                                    plot_data['cluster'] = res['labels'].astype(str)
                                    
                                    # 如果数据点太多，进行采样
                                    max_plot_points = 2000
                                    if len(plot_data) > max_plot_points:
                                        plot_data = plot_data.sample(n=max_plot_points, random_state=42)
                                        st.caption(f"⚠️ 散点图已采样显示 {max_plot_points} 个点（共 {len(X)} 个）")
                                    
                                    fig = px.scatter(
                                        plot_data, 
                                        x=cols[0], 
                                        y=cols[1], 
                                        color='cluster',
                                        title="聚类结果 (前两个特征)",
                                        labels={'cluster': '聚类标签'}
                                    )
                                    fig.update_traces(marker=dict(size=5, opacity=0.6))
                                    
                                    # 如果是K-means，显示聚类中心
                                    if algo == "K-means" and res.get('centers') is not None:
                                        centers = res['centers']
                                        if len(centers) > 0 and len(centers[0]) >= 2:
                                            centers_2d = centers[:, :2] if centers.shape[1] >= 2 else centers
                                            fig.add_trace(go.Scatter(
                                                x=centers_2d[:, 0],
                                                y=centers_2d[:, 1],
                                                mode='markers',
                                                marker=dict(symbol='x', size=15, color='red', line=dict(width=2, color='darkred')),
                                                name='聚类中心',
                                                showlegend=True
                                            ))
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("需要至少2个特征才能显示散点图")
                            
                            with tab2:
                                if res.get('silhouette_score') is not None:
                                    fig_sil = plot_silhouette_plot(X.values, res['labels'], max_samples=1000)
                                    if fig_sil:
                                        st.plotly_chart(fig_sil, use_container_width=True)
                                    else:
                                        st.warning("无法生成轮廓系数图")
                                else:
                                    st.info("DBSCAN 聚类结果中无法计算轮廓系数（可能聚类数过少或噪声点过多）")
                            
                            with tab3:
                                # 聚类大小分布
                                cluster_counts = Counter(res['labels'])
                                cluster_sizes = [cluster_counts.get(i, 0) for i in sorted(set(res['labels'])) if i != -1]
                                cluster_labels = [f"聚类 {i}" for i in sorted(set(res['labels'])) if i != -1]
                                
                                if -1 in cluster_counts:
                                    cluster_labels.append("噪声点")
                                    cluster_sizes.append(cluster_counts[-1])
                                
                                fig_size = px.bar(
                                    x=cluster_labels,
                                    y=cluster_sizes,
                                    title="聚类大小分布",
                                    labels={'x': '聚类', 'y': '样本数'},
                                    color=cluster_sizes,
                                    color_continuous_scale='Viridis'
                                )
                                st.plotly_chart(fig_size, use_container_width=True)
                                
                                # 聚类中心距离热力图（仅K-means）
                                if algo == "K-means" and res.get('centers') is not None:
                                    try:
                                        from sklearn.metrics.pairwise import euclidean_distances
                                        centers = res['centers']
                                        distances = euclidean_distances(centers)
                                        
                                        fig_dist = px.imshow(
                                            distances,
                                            labels=dict(x="聚类", y="聚类", color="距离"),
                                            x=[f"聚类 {i}" for i in range(len(centers))],
                                            y=[f"聚类 {i}" for i in range(len(centers))],
                                            color_continuous_scale='RdYlBu_r',
                                            title="聚类中心距离热力图",
                                            aspect="auto"
                                        )
                                        st.plotly_chart(fig_dist, use_container_width=True)
                                    except Exception as e:
                                        st.warning(f"无法生成聚类中心距离图: {str(e)}")
                            
                            with tab4:
                                # 各特征在聚类中的分布（箱线图）
                                if len(cols) > 0:
                                    selected_feature = st.selectbox("选择特征查看分布", cols, key="cluster_feature_dist")
                                    if selected_feature:
                                        plot_data_dist = df.loc[X.index].copy()
                                        plot_data_dist['cluster'] = res['labels'].astype(str)
                                        
                                        fig_box = px.box(
                                            plot_data_dist,
                                            x='cluster',
                                            y=selected_feature,
                                            title=f"{selected_feature} 在各聚类中的分布",
                                            color='cluster'
                                        )
                                        st.plotly_chart(fig_box, use_container_width=True)
                            
                            
                    except Exception as e:
                        st.error(f"聚类失败: {str(e)}")
                        st.exception(e)


def _show_ml_pca(df):
    """降维分析子页面"""
    col_settings, col_results = st.columns([1, 3], gap="large")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    with col_settings:
        st.markdown("#### ⚙️ PCA 配置")
        cols = st.multiselect("特征选择", numeric_cols, default=numeric_cols)
        if cols:
            max_comps = min(len(cols), len(df))
            n_comps = st.slider("主成分数量", 2, max(2, max_comps), min(2, max_comps))
        else:
            n_comps = 2
            st.info("请先选择特征")
        run_pca = st.button("🚀 执行降维", type="primary", use_container_width=True)

    with col_results:
        if run_pca and cols:
            if len(cols) < 2:
                st.error("⚠️ PCA 需要至少选择 2 个特征")
            else:
                try:
                    X = df[cols].dropna()
                    if len(X) < 2:
                        st.error("⚠️ 有效数据样本不足（删除缺失值后少于2个）")
                    else:
                        # 确保主成分数量不超过特征数量
                        max_comps = min(n_comps, len(cols), len(X))
                        if max_comps < n_comps:
                            st.warning(f"⚠️ 主成分数量已调整为 {max_comps}（受特征数和样本数限制）")
                        
                        res = perform_pca(X, max_comps)
                        
                        st.info(f"前 {max_comps} 个主成分解释了 {res['cumulative_variance'][-1]*100:.2f}% 的方差")
                        
                        # 保存PCA配置
                        pca_config = {
                            'task_type': '降维',
                            'algorithm': 'PCA',
                            'selected_features': cols,
                            'n_components': max_comps,
                            'n_samples': len(X)
                        }
                        st.session_state.ml_train_config = pca_config
                        
                        tab1, tab2 = st.tabs(["方差解释率", "2D 投影"])
                        generated_charts = {}
                        with tab1:
                            fig = px.bar(
                                y=res['explained_variance'], 
                                x=[f"PC{i+1}" for i in range(len(res['explained_variance']))],
                                title="主成分方差解释率"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            generated_charts["方差解释率"] = fig
                        with tab2:
                            if res['X_transformed'].shape[1] >= 2:
                                fig_pca = plot_pca_scatter(res['X_transformed'])
                                if fig_pca:
                                    st.plotly_chart(fig_pca, use_container_width=True)
                                    generated_charts["2D投影"] = fig_pca
                            else:
                                st.warning("需要至少2个主成分才能显示2D投影")
                        
                            
                except Exception as e:
                    st.error(f"PCA 降维失败: {str(e)}")
                    st.exception(e)


def _show_ml_rules(df):
    """关联规则子页面"""
    col_settings, col_results = st.columns([1, 3], gap="large")
    
    with col_settings:
        st.markdown("#### ⚙️ 关联规则配置")
        
        algorithm = st.selectbox("算法选择", ["Apriori", "FP-Growth"], 
                                help="Apriori: 经典算法，逐层生成候选集\nFP-Growth: 高效算法，使用FP-tree结构")
        
        st.markdown(f"**当前算法**: {algorithm}")
        if algorithm == "FP-Growth":
            st.info("💡 FP-Growth 算法比 Apriori 更高效，适合大数据集")
        
        min_sup = st.slider("最小支持度", 0.01, 0.5, 0.05, 0.01)
        min_conf = st.slider("最小置信度", 0.1, 1.0, 0.5, 0.1)
        run_rules = st.button("🚀 挖掘规则", type="primary", use_container_width=True)

    with col_results:
        if run_rules:
            with st.spinner(f"使用 {algorithm} 算法挖掘中..."):
                try:
                    if algorithm == "Apriori":
                        res = perform_apriori(df, min_sup, min_conf)
                    else:
                        res = perform_fpgrowth(df, min_sup, min_conf)
                    
                    rules = res['rules']
                    
                    rules_config = {
                        'task_type': '关联规则',
                        'algorithm': res.get('algorithm', algorithm),
                        'min_support': min_sup,
                        'min_confidence': min_conf,
                        'n_rules': len(rules) if not rules.empty else 0
                    }
                    st.session_state.ml_train_config = rules_config
                    
                    if not rules.empty:
                        st.success(f"✅ 使用 {algorithm} 算法找到 {len(rules)} 条规则")
                        
                        tab1, tab2, tab3 = st.tabs(["规则列表", "热力图", "桑基图"])
                        generated_charts = {}
                        with tab1:
                            st.dataframe(rules, use_container_width=True)
                        with tab2:
                            try:
                                fig_heat = plot_rules_heatmap(rules)
                                if fig_heat: 
                                    st.plotly_chart(fig_heat, use_container_width=True)
                                    generated_charts["规则热力图"] = fig_heat
                                else:
                                    st.info("无法生成热力图")
                            except Exception as e:
                                st.warning(f"生成热力图时出错: {str(e)}")
                        with tab3:
                            try:
                                fig_sankey = plot_rules_sankey(rules)
                                if fig_sankey: 
                                    st.plotly_chart(fig_sankey, use_container_width=True)
                                    generated_charts["规则桑基图"] = fig_sankey
                                else:
                                    st.info("无法生成桑基图")
                            except Exception as e:
                                st.warning(f"生成桑基图时出错: {str(e)}")
                        
                    else:
                        st.warning("未找到满足条件的规则，请尝试降低最小支持度或最小置信度")
                except Exception as e:
                    st.error(f"关联规则挖掘失败: {str(e)}")
                    st.exception(e)

