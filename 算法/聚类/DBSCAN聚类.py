"""
DBSCAN聚类算法
基于密度的聚类算法
"""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

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


def fit(X, eps, min_samples):
    """
    执行DBSCAN聚类
    
    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        特征数据
    eps : float
        eps参数（邻域半径）
    min_samples : int
        min_samples参数（最小样本数）
    
    Returns:
    --------
    dict
        包含聚类结果的字典
    """
    # 验证输入数据
    validate_clustering_data(X, min_samples=min_samples)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    # 安全计算轮廓系数
    silhouette_avg = safe_silhouette_score(X, labels)
    
    return {
        'model': dbscan,
        'labels': labels,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette_score': silhouette_avg
    }

