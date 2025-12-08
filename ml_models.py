"""
机器学习建模与评估模块（统一接口层）
支持多种算法、超参数调优、模型解释
所有算法实现已拆分到 算法/ 目录下的独立文件中
"""
import pandas as pd
import numpy as np
import warnings

# 导入数据准备模块
from 算法.数据准备 import prepare_data_for_ml

# 导入回归算法
from 算法.回归.线性回归 import train as train_linear_regression
from 算法.回归.K近邻回归 import train as train_knn_regression
from 算法.回归.决策树回归 import train as train_tree_regression

# 导入分类算法
from 算法.分类.逻辑回归 import train as train_logistic_regression
from 算法.分类.朴素贝叶斯 import train as train_naive_bayes
from 算法.分类.K近邻分类 import train as train_knn_classification
from 算法.分类.决策树分类 import train as train_tree_classification

# 导入聚类算法
from 算法.聚类.K均值聚类 import fit as fit_kmeans, find_optimal_k
from 算法.聚类.DBSCAN聚类 import fit as fit_dbscan

# 导入降维算法
from 算法.降维.主成分分析 import fit as fit_pca

# 导入关联规则算法
from 算法.关联规则.Apriori关联规则 import fit as fit_apriori
from 算法.关联规则.FPGrowth关联规则 import fit as fit_fpgrowth

# 导入模型解释算法
from 算法.模型解释.SHAP解释器 import explain as explain_with_shap


# ==================== 数据准备 ====================
# prepare_data_for_ml 已在上面导入，直接使用


# ==================== 回归模型 ====================
def train_regression_model(X_train, y_train, X_test, y_test, model_type='linear', auto_optimize=False, **params):
    """
    训练回归模型（统一接口）
    
    Parameters:
    -----------
    X_train, y_train, X_test, y_test : 训练和测试数据
    model_type : str
        模型类型 ('linear', 'knn', 'tree')
    auto_optimize : bool
        是否自动优化参数
    **params : 模型超参数
    
    Returns:
    --------
    dict
        包含模型、预测结果和评估指标的字典
    """
    model_map = {
        'linear': train_linear_regression,
        'knn': train_knn_regression,
        'tree': train_tree_regression
    }
    
    train_func = model_map.get(model_type, train_linear_regression)
    return train_func(X_train, y_train, X_test, y_test, auto_optimize=auto_optimize, **params)


# ==================== 分类模型 ====================
def train_classification_model(X_train, y_train, X_test, y_test, model_type='logistic', auto_optimize=False, **params):
    """
    训练分类模型（统一接口）
    
    Parameters:
    -----------
    X_train, y_train, X_test, y_test : 训练和测试数据
    model_type : str
        模型类型 ('logistic', 'naive_bayes', 'knn', 'tree')
    auto_optimize : bool
        是否自动优化参数
    **params : 模型超参数
    
    Returns:
    --------
    dict
        包含模型、预测结果和评估指标的字典
    """
    model_map = {
        'logistic': train_logistic_regression,
        'naive_bayes': train_naive_bayes,
        'knn': train_knn_classification,
        'tree': train_tree_classification
    }
    
    train_func = model_map.get(model_type, train_logistic_regression)
    return train_func(X_train, y_train, X_test, y_test, auto_optimize=auto_optimize, **params)


# ==================== 聚类算法 ====================
def perform_kmeans_clustering(X, n_clusters, random_state=42):
    """
    执行K-means聚类（统一接口）
    
    Parameters:
    -----------
    X : pd.DataFrame
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
    return fit_kmeans(X, n_clusters, random_state=random_state)


# find_optimal_k 已在上面导入，直接使用


def perform_dbscan_clustering(X, eps, min_samples):
    """
    执行DBSCAN聚类（统一接口）
    
    Parameters:
    -----------
    X : pd.DataFrame
        特征数据
    eps : float
        eps参数
    min_samples : int
        min_samples参数
    
    Returns:
    --------
    dict
        包含聚类结果的字典
    """
    return fit_dbscan(X, eps, min_samples)


# ==================== 降维算法 ====================
def perform_pca(X, n_components=None):
    """
    执行PCA降维（统一接口）
    
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
    return fit_pca(X, n_components=n_components)


# ==================== 关联规则 ====================
def perform_apriori(df, min_support=0.1, min_confidence=0.5):
    """
    执行Apriori关联规则挖掘（统一接口）
    
    Parameters:
    -----------
    df : pd.DataFrame
        事务数据（每行是一个事务，每列是一个项）
    min_support : float
        最小支持度
    min_confidence : float
        最小置信度
    
    Returns:
    --------
    dict
        包含频繁项集和关联规则的字典
    """
    return fit_apriori(df, min_support=min_support, min_confidence=min_confidence)


def perform_fpgrowth(df, min_support=0.1, min_confidence=0.5):
    """
    执行FP-Growth关联规则挖掘（统一接口）
    
    Parameters:
    -----------
    df : pd.DataFrame
        事务数据（每行是一个事务，每列是一个项）
    min_support : float
        最小支持度
    min_confidence : float
        最小置信度
    
    Returns:
    --------
    dict
        包含频繁项集和关联规则的字典
    """
    return fit_fpgrowth(df, min_support=min_support, min_confidence=min_confidence)


# ==================== 模型解释 ====================
def explain_model_with_shap(model, X, model_type='tree'):
    """
    使用SHAP解释模型（统一接口）
    
    Parameters:
    -----------
    model : 训练好的模型
    X : pd.DataFrame
        特征数据
    model_type : str
        模型类型 ('tree', 'linear', 'other')
    
    Returns:
    --------
    tuple
        (shap_values, explainer) 或 (None, None) 如果失败
    """
    return explain_with_shap(model, X, model_type=model_type)
