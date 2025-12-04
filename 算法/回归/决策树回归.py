"""
决策树回归算法
基于CART算法的决策树回归模型
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV

# 导入公共工具函数
try:
    from ..utils import validate_train_data, compute_regression_metrics, extract_feature_importance
except (ImportError, ValueError):
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from 算法.utils import validate_train_data, compute_regression_metrics, extract_feature_importance


def train(X_train, y_train, X_test, y_test, auto_optimize=False, **params):
    """
    训练决策树回归模型
    
    Parameters:
    -----------
    X_train, y_train, X_test, y_test : 训练和测试数据
    auto_optimize : bool
        是否自动优化参数
    **params : 模型超参数
        max_depth : int, 默认None, 最大深度
        min_samples_split : int, 默认2, 最小分割样本数
    
    Returns:
    --------
    dict
        包含模型、预测结果和评估指标的字典
    """
    # 验证输入数据
    validate_train_data(X_train, y_train, X_test, y_test, task_type='regression')
    
    # 获取特征名称
    feature_names = X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else None
    
    model = DecisionTreeRegressor(
        max_depth=params.get('max_depth', None),
        min_samples_split=params.get('min_samples_split', 2),
        random_state=42
    )
    
    # 参数自动优化
    if auto_optimize:
        param_grid = {
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10]
        }
        search = RandomizedSearchCV(
            model, param_grid,
            n_iter=min(20, np.prod([len(v) for v in param_grid.values()])),
            cv=5, scoring='neg_mean_squared_error',
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
    metrics = compute_regression_metrics(y_test, y_pred)
    
    # 提取特征重要性
    feature_importance = extract_feature_importance(model, feature_names)
    
    return {
        'model': model,
        'y_pred': y_pred,
        'y_test': y_test,
        'metrics': metrics,
        'feature_importance': feature_importance,
        'best_params': best_params if auto_optimize else None
    }

