"""
K均值聚类算法
基于距离的划分聚类算法
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# 导入公共工具函数
try:
    from ..utils import validate_clustering_data, safe_silhouette_score
except (ImportError, ValueError):
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from 算法.utils import validate_clustering_data, safe_silhouette_score


def fit(X, n_clusters, random_state=42):
    """
    执行K-means聚类
    
    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        特征数据
    n_clusters : int
        聚类数量
    random_state : int
        随机种子
    
    Returns:
    --------
    dict
        包含聚类结果和评估指标的字典
    """
    # 验证输入数据
    validate_clustering_data(X, min_samples=n_clusters)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # 安全计算轮廓系数
    silhouette_avg = safe_silhouette_score(X, labels)
    
    return {
        'model': kmeans,
        'labels': labels,
        'inertia': kmeans.inertia_,
        'silhouette_score': silhouette_avg,
        'centers': kmeans.cluster_centers_
    }


def find_optimal_k(X, max_k=10):
    """
    使用肘部法则和轮廓系数找到最优K值
    
    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        特征数据
    max_k : int
        最大K值
    
    Returns:
    --------
    dict
        包含不同K值的评估指标
    """
    # 验证输入数据
    validate_clustering_data(X, min_samples=2)
    
    inertias = []
    silhouette_scores = []
    k_range = range(2, min(max_k + 1, len(X) // 2))
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        score = safe_silhouette_score(X, labels)
        silhouette_scores.append(score if score is not None else -1)
    
    # 找到轮廓系数最大的K
    optimal_k = k_range[np.argmax(silhouette_scores)]
    
    return {
        'k_range': list(k_range),
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'optimal_k': optimal_k
    }

