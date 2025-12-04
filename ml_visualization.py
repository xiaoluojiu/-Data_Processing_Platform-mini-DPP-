"""
机器学习可视化模块
提供科学、专业的模型可视化
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, 
    precision_recall_curve, silhouette_samples
)
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import io
import base64


def plot_confusion_matrix_heatmap(y_true, y_pred, class_names=None):
    """
    绘制混淆矩阵热力图
    
    Parameters:
    -----------
    y_true : array-like
        真实标签
    y_pred : array-like
        预测标签
    class_names : list, optional
        类别名称
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = [f'类别 {i}' for i in range(len(cm))]
    
    fig = px.imshow(
        cm,
        labels=dict(x="预测值", y="真实值", color="数量"),
        x=class_names,
        y=class_names,
        color_continuous_scale='Blues',
        aspect="auto",
        title="混淆矩阵热力图"
    )
    
    # 添加数值标注
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            fig.add_annotation(
                x=j, y=i,
                text=str(cm[i, j]),
                showarrow=False,
                font=dict(color='white' if cm[i, j] > cm.max() / 2 else 'black', size=14)
            )
    
    fig.update_layout(
        template='plotly_white',
        height=500
    )
    
    return fig


def plot_roc_curve(y_true, y_pred_proba, class_names=None):
    """
    绘制ROC曲线（多分类）
    
    Parameters:
    -----------
    y_true : array-like
        真实标签
    y_pred_proba : array-like
        预测概率
    class_names : list, optional
        类别名称
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    from itertools import cycle
    
    # 二分类
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC曲线 (AUC = {roc_auc:.2f})',
            line=dict(width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='随机分类器',
            line=dict(dash='dash', color='gray')
        ))
    else:
        # 多分类
        n_classes = len(np.unique(y_true))
        y_bin = label_binarize(y_true, classes=np.unique(y_true))
        
        fig = go.Figure()
        colors = cycle(['blue', 'red', 'green', 'orange', 'purple'])
        
        for i, color in zip(range(n_classes), colors):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            class_name = class_names[i] if class_names else f'类别 {i}'
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{class_name} (AUC = {roc_auc:.2f})',
                line=dict(width=2, color=color)
            ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='随机分类器',
            line=dict(dash='dash', color='gray')
        ))
    
    fig.update_layout(
        title='ROC曲线',
        xaxis_title='假正率 (FPR)',
        yaxis_title='真正率 (TPR)',
        template='plotly_white',
        height=500
    )
    
    return fig


def plot_precision_recall_curve(y_true, y_pred_proba):
    """
    绘制精确率-召回率曲线
    
    Parameters:
    -----------
    y_true : array-like
        真实标签
    y_pred_proba : array-like
        预测概率
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
    pr_auc = auc(recall, precision)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name=f'PR曲线 (AUC = {pr_auc:.2f})',
        line=dict(width=2)
    ))
    
    fig.update_layout(
        title='精确率-召回率曲线',
        xaxis_title='召回率 (Recall)',
        yaxis_title='精确率 (Precision)',
        template='plotly_white',
        height=500
    )
    
    return fig


def plot_learning_curve(estimator, X, y, cv=5, train_sizes=None):
    """
    绘制学习曲线
    
    Parameters:
    -----------
    estimator : 模型
    X : 特征数据
    y : 目标变量
    cv : int
        交叉验证折数
    train_sizes : array-like, optional
        训练集大小
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes_abs, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes, n_jobs=-1
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    fig = go.Figure()
    
    # 训练集曲线
    fig.add_trace(go.Scatter(
        x=train_sizes_abs,
        y=train_mean,
        mode='lines+markers',
        name='训练集得分',
        line=dict(color='blue', width=2),
        error_y=dict(type='data', array=train_std, visible=True)
    ))
    
    # 验证集曲线
    fig.add_trace(go.Scatter(
        x=train_sizes_abs,
        y=val_mean,
        mode='lines+markers',
        name='验证集得分',
        line=dict(color='red', width=2),
        error_y=dict(type='data', array=val_std, visible=True)
    ))
    
    fig.update_layout(
        title='学习曲线',
        xaxis_title='训练样本数',
        yaxis_title='得分',
        template='plotly_white',
        height=500
    )
    
    return fig


def plot_validation_curve(estimator, X, y, param_name, param_range, cv=5):
    """
    绘制验证曲线
    
    Parameters:
    -----------
    estimator : 模型
    X : 特征数据
    y : 目标变量
    param_name : str
        参数名称
    param_range : array-like
        参数范围
    cv : int
        交叉验证折数
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    train_scores, val_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range, cv=cv, n_jobs=-1
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=param_range,
        y=train_mean,
        mode='lines+markers',
        name='训练集得分',
        line=dict(color='blue', width=2),
        error_y=dict(type='data', array=train_std, visible=True)
    ))
    
    fig.add_trace(go.Scatter(
        x=param_range,
        y=val_mean,
        mode='lines+markers',
        name='验证集得分',
        line=dict(color='red', width=2),
        error_y=dict(type='data', array=val_std, visible=True)
    ))
    
    fig.update_layout(
        title=f'验证曲线 - {param_name}',
        xaxis_title=param_name,
        yaxis_title='得分',
        template='plotly_white',
        height=500
    )
    
    return fig


def plot_decision_tree_structure(tree_model, feature_names, class_names=None, max_depth=3):
    """
    绘制决策树结构（使用matplotlib，然后转换为plotly）
    
    Parameters:
    -----------
    tree_model : 决策树模型
    feature_names : list
        特征名称
    class_names : list, optional
        类别名称
    max_depth : int
        最大显示深度
    
    Returns:
    --------
    str
        base64编码的图片
    """
    try:
        from sklearn.tree import plot_tree
        
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(
            tree_model,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            max_depth=max_depth,
            ax=ax,
            fontsize=10
        )
        
        # 转换为base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        
        return img_base64
    except Exception as e:
        return None


def plot_feature_importance_comparison(importance_dicts, model_names):
    """
    对比多个模型的特征重要性
    
    Parameters:
    -----------
    importance_dicts : list of dict
        多个模型的特征重要性字典
    model_names : list
        模型名称列表
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    # 获取所有特征
    all_features = set()
    for imp_dict in importance_dicts:
        all_features.update(imp_dict.keys())
    all_features = sorted(list(all_features))
    
    # 构建数据
    data = []
    for model_name, imp_dict in zip(model_names, importance_dicts):
        for feature in all_features:
            data.append({
                '特征': feature,
                '重要性': imp_dict.get(feature, 0),
                '模型': model_name
            })
    
    df = pd.DataFrame(data)
    
    fig = px.bar(
        df,
        x='特征',
        y='重要性',
        color='模型',
        barmode='group',
        title='特征重要性对比',
        labels={'重要性': '特征重要性', '特征': '特征名称'}
    )
    
    fig.update_layout(
        template='plotly_white',
        height=500,
        xaxis_tickangle=-45
    )
    
    return fig


def plot_prediction_distribution(y_true, y_pred, task_type='classification'):
    """
    绘制预测值分布
    
    Parameters:
    -----------
    y_true : array-like
        真实值
    y_pred : array-like
        预测值
    task_type : str
        任务类型
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    if task_type == 'classification':
        # 分类任务：显示预测准确率分布
        correct = (y_true == y_pred).astype(int)
        fig = px.histogram(
            x=correct,
            nbins=2,
            labels={'x': '预测正确性', 'count': '样本数'},
            title='预测准确率分布'
        )
    else:
        # 回归任务：显示预测误差分布
        errors = y_true - y_pred
        fig = px.histogram(
            x=errors,
            nbins=30,
            labels={'x': '预测误差', 'count': '样本数'},
            title='预测误差分布'
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="零误差线")
    
    fig.update_layout(template='plotly_white', height=400)
    return fig


def plot_silhouette_plot(X, labels, max_samples=1000):
    """
    绘制聚类的轮廓系数分布图
    
    Parameters:
    -----------
    X : array-like
        特征数据
    labels : array-like
        聚类标签
    max_samples : int
        最大采样数量，超过此数量将进行采样以提升性能
    """
    unique_labels = np.unique(labels)
    # 至少需要2个聚类才有意义
    if len(unique_labels) <= 1:
        return None
    
    # 对于大数据集，进行采样以提升性能
    X = np.array(X)
    labels = np.array(labels)
    
    if len(X) > max_samples:
        # 按聚类比例采样
        from sklearn.utils import resample
        sampled_indices = []
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            if len(label_indices) > 0:
                # 每个聚类至少采样 min(100, 该聚类数量)
                n_samples = min(100, len(label_indices), max_samples // len(unique_labels))
                if n_samples < len(label_indices):
                    sampled = resample(label_indices, n_samples=n_samples, random_state=42, replace=False)
                else:
                    sampled = label_indices
                sampled_indices.extend(sampled)
        
        sampled_indices = np.array(sampled_indices)
        X_sampled = X[sampled_indices]
        labels_sampled = labels[sampled_indices]
    else:
        X_sampled = X
        labels_sampled = labels
    
    # 计算轮廓系数（只对采样后的数据）
    try:
        sil_values = silhouette_samples(X_sampled, labels_sampled)
    except Exception as e:
        # 如果计算失败，返回简单的统计图
        return None
    
    fig = go.Figure()
    
    y_lower = 0
    for i, cluster in enumerate(unique_labels):
        cluster_mask = labels_sampled == cluster
        cluster_sil = sil_values[cluster_mask]
        
        if len(cluster_sil) == 0:
            continue
            
        cluster_sil_sorted = np.sort(cluster_sil)
        y_upper = y_lower + len(cluster_sil_sorted)
        
        if cluster == -1:
            color = 'lightgray'
            name = '噪声点'
        else:
            color = px.colors.qualitative.Set2[i % len(px.colors.qualitative.Set2)]
            name = f'聚类 {cluster}'
        
        fig.add_trace(go.Bar(
            x=cluster_sil_sorted,
            y=list(range(y_lower, y_upper)),
            orientation='h',
            marker_color=color,
            name=name,
            hoverinfo='x+y',
            showlegend=True
        ))
        
        y_lower = y_upper
    
    # 添加平均轮廓系数线
    avg_sil = np.mean(sil_values)
    fig.add_vline(
        x=avg_sil, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"平均: {avg_sil:.3f}",
        annotation_position="top"
    )
    
    fig.update_layout(
        title='聚类轮廓系数分布' + (f' (已采样，显示 {len(X_sampled)}/{len(X)} 个样本)' if len(X) > max_samples else ''),
        xaxis_title='轮廓系数',
        yaxis_title='样本索引（按聚类分组）',
        template='plotly_white',
        height=500,
        bargap=0.01
    )
    return fig


def plot_pca_scatter(X_pca, y=None):
    """绘制前两个主成分的散点图"""
    if X_pca.shape[1] < 2:
        return None
    
    df_pca = pd.DataFrame(X_pca[:, :2], columns=['PC1', 'PC2'])
    if y is not None and len(y) == len(df_pca):
        df_pca['target'] = y
        fig = px.scatter(
            df_pca,
            x='PC1',
            y='PC2',
            color='target',
            title='PCA前两主成分散点图',
        )
    else:
        fig = px.scatter(
            df_pca,
            x='PC1',
            y='PC2',
            title='PCA前两主成分散点图',
        )
    
    fig.update_layout(
        template='plotly_white',
        height=500
    )
    return fig


def plot_rules_heatmap(rules, metric='lift', top_n=30):
    """
    关联规则热力图（前件 × 后件）
    
    Parameters
    ----------
    rules : pd.DataFrame
        perform_apriori 返回的 rules 数据表
    metric : str
        使用的指标（如 'lift'、'confidence'、'support'）
    top_n : int
        选择前 top_n 条规则
    """
    if rules is None or len(rules) == 0 or metric not in rules.columns:
        return None
    
    top_rules = rules.sort_values(by=metric, ascending=False).head(top_n).copy()
    records = []
    for _, r in top_rules.iterrows():
        antecedent = ', '.join(sorted(list(r['antecedents'])))
        consequent = ', '.join(sorted(list(r['consequents'])))
        records.append({
            '前件': antecedent,
            '后件': consequent,
            metric: r[metric]
        })
    
    if not records:
        return None
    
    df_pairs = pd.DataFrame(records)
    pivot = df_pairs.pivot_table(
        index='前件',
        columns='后件',
        values=metric,
        aggfunc='max'
    ).fillna(0)
    
    fig = px.imshow(
        pivot,
        labels=dict(x="后件", y="前件", color=metric),
        x=pivot.columns,
        y=pivot.index,
        color_continuous_scale='Viridis',
        title=f"关联规则热力图（按 {metric}，前 {top_n} 条）"
    )
    fig.update_layout(
        template='plotly_white',
        height=max(500, 40 * len(pivot.index))
    )
    return fig


def plot_rules_sankey(rules, metric='lift', top_n=30):
    """
    使用桑基图展示关联规则网络（项级别）
    
    Parameters
    ----------
    rules : pd.DataFrame
    metric : str
        边权重使用的指标
    top_n : int
        使用的规则数量
    """
    if rules is None or len(rules) == 0 or metric not in rules.columns:
        return None
    
    top_rules = rules.sort_values(by=metric, ascending=False).head(top_n).copy()
    
    # 收集节点（项）
    items = set()
    for _, r in top_rules.iterrows():
        items.update(list(r['antecedents']))
        items.update(list(r['consequents']))
    items = sorted(list(items))
    if not items:
        return None
    
    node_index = {item: i for i, item in enumerate(items)}
    source = []
    target = []
    value = []
    
    for _, r in top_rules.iterrows():
        antecedents = list(r['antecedents'])
        consequents = list(r['consequents'])
        w = float(r[metric])
        for a in antecedents:
            for c in consequents:
                if a in node_index and c in node_index:
                    source.append(node_index[a])
                    target.append(node_index[c])
                    value.append(max(w, 1e-6))  # 避免0
    
    if not source:
        return None
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=15,
            line=dict(color="black", width=0.5),
            label=items
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
        )
    )])
    
    fig.update_layout(
        title_text=f"关联规则网络（桑基图，按 {metric}，前 {top_n} 条）",
        font_size=12,
        template='plotly_white',
        height=600
    )
    return fig


def plot_residual_analysis(y_true, y_pred):
    """
    绘制残差分析图（回归任务）
    包含：残差分布、残差vs预测值、Q-Q图
    
    Parameters:
    -----------
    y_true : array-like
        真实值
    y_pred : array-like
        预测值
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    residuals = y_true - y_pred
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('残差分布直方图', '残差 vs 预测值', 'Q-Q图', '残差时间序列'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. 残差分布直方图
    fig.add_trace(
        go.Histogram(x=residuals, nbinsx=30, name='残差分布', showlegend=False),
        row=1, col=1
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=1)
    
    # 2. 残差 vs 预测值
    fig.add_trace(
        go.Scatter(x=y_pred, y=residuals, mode='markers', name='残差', showlegend=False),
        row=1, col=2
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
    
    # 3. Q-Q图（正态性检验）
    try:
        from scipy import stats
        sorted_residuals = np.sort(residuals)
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_residuals)))
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sorted_residuals, mode='markers', name='Q-Q', showlegend=False),
            row=2, col=1
        )
        # 添加对角线
        min_val = min(theoretical_quantiles.min(), sorted_residuals.min())
        max_val = max(theoretical_quantiles.max(), sorted_residuals.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', 
                      line=dict(dash='dash', color='red'), name='理论线', showlegend=False),
            row=2, col=1
        )
    except:
        pass
    
    # 4. 残差时间序列（按索引）
    fig.add_trace(
        go.Scatter(x=list(range(len(residuals))), y=residuals, mode='lines+markers', 
                  name='残差序列', showlegend=False),
        row=2, col=2
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)
    
    fig.update_layout(
        title='残差分析',
        template='plotly_white',
        height=700,
        showlegend=False
    )
    
    return fig


def plot_prediction_vs_actual(y_true, y_pred, task_type='regression'):
    """
    绘制预测值 vs 真实值散点图
    
    Parameters:
    -----------
    y_true : array-like
        真实值
    y_pred : array-like
        预测值
    task_type : str
        任务类型
    """
    fig = go.Figure()
    
    if task_type == 'regression':
        # 回归：散点图 + 完美预测线
        fig.add_trace(go.Scatter(
            x=y_true,
            y=y_pred,
            mode='markers',
            name='预测值',
            marker=dict(size=8, opacity=0.6)
        ))
        
        # 添加完美预测线 y=x
        min_val = min(float(y_true.min()), float(y_pred.min()))
        max_val = max(float(y_true.max()), float(y_pred.max()))
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='完美预测线',
            line=dict(dash='dash', color='red', width=2)
        ))
        
        fig.update_layout(
            title='预测值 vs 真实值',
            xaxis_title='真实值',
            yaxis_title='预测值',
            template='plotly_white',
            height=500
        )
    
    return fig


def plot_feature_interaction_heatmap(X, feature_names, top_n=15):
    """
    绘制特征交互热力图（相关性矩阵）
    
    Parameters:
    -----------
    X : array-like
        特征数据
    feature_names : list
        特征名称
    top_n : int
        显示前n个特征
    """
    if len(feature_names) > top_n:
        # 选择方差最大的top_n个特征
        variances = np.var(X, axis=0)
        top_indices = np.argsort(variances)[-top_n:]
        X_selected = X[:, top_indices]
        feature_names_selected = [feature_names[i] for i in top_indices]
    else:
        X_selected = X
        feature_names_selected = feature_names
    
    corr_matrix = np.corrcoef(X_selected.T)
    
    fig = px.imshow(
        corr_matrix,
        labels=dict(x="特征", y="特征", color="相关系数"),
        x=feature_names_selected,
        y=feature_names_selected,
        color_continuous_scale='RdBu',
        aspect="auto",
        title=f'特征交互热力图（前 {len(feature_names_selected)} 个特征）'
    )
    
    fig.update_layout(
        template='plotly_white',
        height=600
    )
    
    return fig


def plot_error_distribution(y_true, y_pred, bins=30):
    """
    绘制误差分布图（回归任务）
    
    Parameters:
    -----------
    y_true : array-like
        真实值
    y_pred : array-like
        预测值
    bins : int
        直方图bins数量
    """
    errors = y_true - y_pred
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=errors,
        nbinsx=bins,
        name='误差分布',
        marker_color='lightblue'
    ))
    
    # 添加正态分布曲线
    try:
        from scipy import stats
        mu, sigma = np.mean(errors), np.std(errors)
        x_norm = np.linspace(errors.min(), errors.max(), 100)
        y_norm = stats.norm.pdf(x_norm, mu, sigma) * len(errors) * (errors.max() - errors.min()) / bins
        fig.add_trace(go.Scatter(
            x=x_norm,
            y=y_norm,
            mode='lines',
            name='理论正态分布',
            line=dict(color='red', width=2)
        ))
    except:
        pass
    
    mu = np.mean(errors)
    fig.add_vline(x=0, line_dash="dash", line_color="green", annotation_text="零误差")
    fig.add_vline(x=mu, line_dash="dash", line_color="orange", annotation_text=f"均值: {mu:.3f}")
    
    fig.update_layout(
        title='误差分布图',
        xaxis_title='误差',
        yaxis_title='频数',
        template='plotly_white',
        height=500
    )
    
    return fig


def plot_classification_report_heatmap(y_true, y_pred, class_names=None):
    """
    绘制分类报告热力图（精确率、召回率、F1分数）
    
    Parameters:
    -----------
    y_true : array-like
        真实标签
    y_pred : array-like
        预测标签
    class_names : list, optional
        类别名称
    """
    from sklearn.metrics import classification_report
    
    report = classification_report(y_true, y_pred, output_dict=True, target_names=class_names, zero_division=0)
    
    # 提取每个类别的指标
    metrics_data = []
    for class_name, metrics in report.items():
        if isinstance(metrics, dict) and 'precision' in metrics:
            metrics_data.append({
                '类别': class_name,
                '精确率': metrics['precision'],
                '召回率': metrics['recall'],
                'F1分数': metrics['f1-score'],
                '支持数': int(metrics['support'])
            })
    
    if not metrics_data:
        return None
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # 创建热力图
    metrics_matrix = df_metrics[['精确率', '召回率', 'F1分数']].values.T
    
    fig = px.imshow(
        metrics_matrix,
        labels=dict(x="类别", y="指标", color="分数"),
        x=df_metrics['类别'].tolist(),
        y=['精确率', '召回率', 'F1分数'],
        color_continuous_scale='RdYlGn',
        aspect="auto",
        title='分类报告热力图'
    )
    
    # 添加数值标注
    for i, metric_name in enumerate(['精确率', '召回率', 'F1分数']):
        for j, class_name in enumerate(df_metrics['类别']):
            value = metrics_matrix[i, j]
            fig.add_annotation(
                x=j, y=i,
                text=f'{value:.3f}',
                showarrow=False,
                font=dict(color='white' if value < 0.5 else 'black', size=12)
            )
    
    fig.update_layout(
        template='plotly_white',
        height=400
    )
    
    return fig

