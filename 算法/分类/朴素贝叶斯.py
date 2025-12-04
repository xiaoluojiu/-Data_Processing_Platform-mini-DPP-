"""
朴素贝叶斯算法
基于贝叶斯定理的分类模型
"""
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RandomizedSearchCV

# 导入公共工具函数
try:
    from ..utils import validate_train_data, compute_classification_metrics, extract_feature_importance
except (ImportError, ValueError):
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from 算法.utils import validate_train_data, compute_classification_metrics, extract_feature_importance


def train(X_train, y_train, X_test, y_test, auto_optimize=False, **params):
    """
    训练朴素贝叶斯模型
    
    Parameters:
    -----------
    X_train, y_train, X_test, y_test : 训练和测试数据
    auto_optimize : bool
        是否自动优化参数
    **params : 模型超参数
        var_smoothing : float, 默认1e-9, 方差平滑参数
    
    Returns:
    --------
    dict
        包含模型、预测结果和评估指标的字典
    """
    # 验证输入数据
    validate_train_data(X_train, y_train, X_test, y_test, task_type='classification')
    
    model = GaussianNB()
    
    # 参数自动优化
    if auto_optimize:
        param_grid = {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        }
        search = RandomizedSearchCV(
            model, param_grid,
            n_iter=min(20, len(param_grid['var_smoothing'])),
            cv=5, scoring='accuracy',
            n_jobs=-1, random_state=42
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
    else:
        model.fit(X_train, y_train)
        best_params = params
    
    y_pred = model.predict(X_test)
    
    # 计算评估指标
    metrics = compute_classification_metrics(y_test, y_pred)
    
    # 朴素贝叶斯没有特征重要性
    return {
        'model': model,
        'y_pred': y_pred,
        'y_test': y_test,
        'metrics': {
            'Accuracy': metrics['Accuracy'],
            'Classification Report': metrics['Classification Report']
        },
        'confusion_matrix': metrics['confusion_matrix'],
        'feature_importance': None,
        'best_params': best_params if auto_optimize else None
    }

