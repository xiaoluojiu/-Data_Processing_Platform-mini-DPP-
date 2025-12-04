"""
主成分分析（PCA）算法
线性降维算法
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import warnings

# 导入公共工具函数（如果有需要）
try:
    from ..utils import validate_clustering_data
except (ImportError, ValueError):
    pass


def fit(X, n_components=None):
    """
    执行PCA降维
    
    Parameters:
    -----------
    X : pd.DataFrame
        特征数据
    n_components : int
        主成分数量，如果为None则保留所有
    
    Returns:
    --------
    dict
        包含降维结果和解释方差的字典
    """
    # 基本验证
    if not isinstance(X, (pd.DataFrame, np.ndarray)):
        raise ValueError("X 必须是 pandas.DataFrame 或 numpy.ndarray")
    
    if len(X) == 0 or X.shape[1] == 0:
        raise ValueError("输入数据为空或没有特征")
    
    if n_components is None:
        n_components = min(X.shape[1], X.shape[0])
    else:
        if n_components > X.shape[1]:
            warnings.warn(f"n_components ({n_components}) 大于特征数 ({X.shape[1]})，将使用最大特征数")
            n_components = X.shape[1]
    
    try:
        pca = PCA(n_components=n_components)
        X_transformed = pca.fit_transform(X)
    except Exception as e:
        raise ValueError(f"PCA拟合失败: {str(e)}")
    
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    return {
        'model': pca,
        'X_transformed': X_transformed,
        'explained_variance': explained_variance,
        'cumulative_variance': cumulative_variance,
        'components': pca.components_,
        'feature_names': X.columns.tolist() if hasattr(X, 'columns') else None
    }

