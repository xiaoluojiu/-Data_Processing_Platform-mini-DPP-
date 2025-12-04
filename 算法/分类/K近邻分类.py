"""
K近邻分类算法
基于距离的分类模型
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
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
    训练K近邻分类模型
    
    Parameters:
    -----------
    X_train, y_train, X_test, y_test : 训练和测试数据
    auto_optimize : bool
        是否自动优化参数
    **params : 模型超参数
        n_neighbors : int, 默认5, K值
    
    Returns:
    --------
    dict
        包含模型、预测结果和评估指标的字典
    """
    # 验证输入数据
    validate_train_data(X_train, y_train, X_test, y_test, task_type='classification')
    
    # 智能初始K值选择：根据训练集大小
    default_k = min(5, max(3, int(np.sqrt(len(X_train)))))
    model = KNeighborsClassifier(n_neighbors=params.get('n_neighbors', default_k))
    
    # 参数自动优化（根据数据规模调整范围）
    if auto_optimize:
        max_k = min(21, max(5, len(X_train) // 10))
        param_grid = {
            'n_neighbors': list(range(3, max_k + 1, 2))  # 只搜索奇数，避免平局
        }
        search = RandomizedSearchCV(
            model, param_grid,
            n_iter=min(20, len(param_grid['n_neighbors'])),
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
    
    # KNN没有特征重要性
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

