"""
算法模块
包含所有机器学习算法的实现

提供统一的算法接口，支持：
- 分类算法：逻辑回归、K近邻、决策树、朴素贝叶斯
- 回归算法：线性回归、K近邻回归、决策树回归
- 聚类算法：K均值、DBSCAN
- 降维算法：主成分分析（PCA）
- 关联规则：Apriori算法
- 模型解释：SHAP解释器
"""

# 数据准备
from .数据准备 import prepare_data_for_ml

# 分类算法
from .分类.逻辑回归 import train as train_logistic_regression
from .分类.K近邻分类 import train as train_knn_classification
from .分类.决策树分类 import train as train_tree_classification
from .分类.朴素贝叶斯 import train as train_naive_bayes

# 回归算法
from .回归.线性回归 import train as train_linear_regression
from .回归.K近邻回归 import train as train_knn_regression
from .回归.决策树回归 import train as train_tree_regression

# 聚类算法
from .聚类.K均值聚类 import fit as fit_kmeans, find_optimal_k
from .聚类.DBSCAN聚类 import fit as fit_dbscan

# 降维算法
from .降维.主成分分析 import fit as fit_pca

# 关联规则
from .关联规则.Apriori关联规则 import fit as fit_apriori

# 模型解释
from .模型解释.SHAP解释器 import explain as explain_shap

# 工具函数
from .utils import (
    validate_train_data,
    validate_clustering_data,
    compute_classification_metrics,
    compute_regression_metrics,
    extract_feature_importance,
    safe_silhouette_score
)

__all__ = [
    # 数据准备
    'prepare_data_for_ml',
    # 分类算法
    'train_logistic_regression',
    'train_knn_classification',
    'train_tree_classification',
    'train_naive_bayes',
    # 回归算法
    'train_linear_regression',
    'train_knn_regression',
    'train_tree_regression',
    # 聚类算法
    'fit_kmeans',
    'find_optimal_k',
    'fit_dbscan',
    # 降维算法
    'fit_pca',
    # 关联规则
    'fit_apriori',
    # 模型解释
    'explain_shap',
    # 工具函数
    'validate_train_data',
    'validate_clustering_data',
    'compute_classification_metrics',
    'compute_regression_metrics',
    'extract_feature_importance',
    'safe_silhouette_score',
]
