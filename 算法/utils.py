"""
算法工具模块
提供算法实现中常用的公共功能
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, silhouette_score
)
from typing import Dict, Any, Optional, Union
import warnings


def validate_train_data(X_train, y_train, X_test, y_test, task_type='classification'):
    """
    验证训练和测试数据的有效性
    
    Parameters:
    -----------
    X_train, y_train, X_test, y_test : 训练和测试数据
    task_type : str
        任务类型 ('classification' 或 'regression')
    
    Returns:
    --------
    None
    
    Raises:
    -------
    ValueError: 如果数据无效
    """
    # 检查输入类型
    if not isinstance(X_train, (pd.DataFrame, np.ndarray)):
        raise ValueError("X_train 必须是 pandas.DataFrame 或 numpy.ndarray")
    
    # 检查数据形状
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("训练集或测试集为空")
    
    if len(X_train) != len(y_train):
        raise ValueError(f"X_train 和 y_train 的长度不匹配: {len(X_train)} vs {len(y_train)}")
    
    if len(X_test) != len(y_test):
        raise ValueError(f"X_test 和 y_test 的长度不匹配: {len(X_test)} vs {len(y_test)}")
    
    # 检查特征数量
    if X_train.shape[1] == 0:
        raise ValueError("特征数量为0")
    
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(f"训练集和测试集的特征数量不匹配: {X_train.shape[1]} vs {X_test.shape[1]}")
    
    # 检查是否有足够的样本
    if len(X_train) < 5:
        raise ValueError(f"训练样本数量过少（{len(X_train)}），至少需要5个样本")
    
    # 检查目标变量
    if task_type == 'classification':
        unique_train = len(np.unique(y_train))
        if unique_train < 2:
            raise ValueError(f"分类任务需要至少2个类别，但训练集中只有{unique_train}个")
        if unique_train == 1:
            raise ValueError("训练集中所有样本都属于同一类别，无法进行分类")


def compute_classification_metrics(y_test, y_pred) -> Dict[str, Any]:
    """
    计算分类任务的评估指标
    
    Parameters:
    -----------
    y_test : array-like
        真实标签
    y_pred : array-like
        预测标签
    
    Returns:
    --------
    dict
        包含准确率、分类报告和混淆矩阵的字典
    """
    try:
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'Accuracy': accuracy,
            'Classification Report': report,
            'confusion_matrix': cm
        }
    except Exception as e:
        warnings.warn(f"计算分类指标时出错: {str(e)}")
        return {
            'Accuracy': 0.0,
            'Classification Report': {},
            'confusion_matrix': np.array([])
        }


def compute_regression_metrics(y_test, y_pred) -> Dict[str, float]:
    """
    计算回归任务的评估指标
    
    Parameters:
    -----------
    y_test : array-like
        真实值
    y_pred : array-like
        预测值
    
    Returns:
    --------
    dict
        包含MSE、RMSE和R²的字典
    """
    try:
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2
        }
    except Exception as e:
        warnings.warn(f"计算回归指标时出错: {str(e)}")
        return {
            'MSE': float('inf'),
            'RMSE': float('inf'),
            'R²': -float('inf')
        }


def extract_feature_importance(model, feature_names, importance_type='auto') -> Optional[Dict[str, float]]:
    """
    提取模型的特征重要性
    
    Parameters:
    -----------
    model : 训练好的模型
    feature_names : list or pd.Index
        特征名称列表
    importance_type : str
        重要性类型 ('auto', 'coef', 'importance')
    
    Returns:
    --------
    dict or None
        特征重要性字典，如果模型不支持则返回None
    """
    if feature_names is None:
        return None
    
    try:
        # 尝试使用 feature_importances_（决策树等）
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            return dict(zip(feature_names, importance))
        
        # 尝试使用 coef_（线性模型）
        if hasattr(model, 'coef_'):
            coef = model.coef_
            # 处理多分类情况（多个系数数组）
            if len(coef.shape) > 1:
                coef_abs = np.abs(coef).mean(axis=0)
            else:
                coef_abs = np.abs(coef)
            return dict(zip(feature_names, coef_abs))
        
        return None
    except Exception as e:
        warnings.warn(f"提取特征重要性时出错: {str(e)}")
        return None


def validate_clustering_data(X, min_samples=2):
    """
    验证聚类数据的有效性
    
    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        特征数据
    min_samples : int
        最小样本数
    
    Returns:
    --------
    None
    
    Raises:
    -------
    ValueError: 如果数据无效
    """
    if not isinstance(X, (pd.DataFrame, np.ndarray)):
        raise ValueError("X 必须是 pandas.DataFrame 或 numpy.ndarray")
    
    if len(X) < min_samples:
        raise ValueError(f"样本数量过少（{len(X)}），至少需要{min_samples}个样本")
    
    if X.shape[1] == 0:
        raise ValueError("特征数量为0")


def safe_silhouette_score(X, labels):
    """
    安全计算轮廓系数（处理边界情况）
    
    Parameters:
    -----------
    X : array-like
        特征数据
    labels : array-like
        聚类标签
    
    Returns:
    --------
    float or None
        轮廓系数，如果无法计算则返回None
    """
    try:
        unique_labels = len(np.unique(labels))
        
        # 需要至少2个聚类才能计算轮廓系数
        if unique_labels < 2:
            return None
        
        # 过滤噪声点（标签为-1）
        mask = labels != -1
        if np.sum(mask) < 2:
            return None
        
        # 检查过滤后是否还有足够的聚类
        filtered_labels = labels[mask]
        if len(np.unique(filtered_labels)) < 2:
            return None
        
        return silhouette_score(X[mask] if hasattr(X, '__getitem__') else X, filtered_labels)
    except Exception as e:
        warnings.warn(f"计算轮廓系数时出错: {str(e)}")
        return None

