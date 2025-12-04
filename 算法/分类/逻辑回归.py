"""
逻辑回归算法
基于最大似然估计的分类模型
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

try:
    from ..utils import validate_train_data, compute_classification_metrics, extract_feature_importance
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from 算法.utils import validate_train_data, compute_classification_metrics, extract_feature_importance


def train(X_train, y_train, X_test, y_test, auto_optimize=False, **params):
    """
    训练逻辑回归模型
    
    Parameters:
    -----------
    X_train, y_train, X_test, y_test : 训练和测试数据
    auto_optimize : bool
        是否自动优化参数
    **params : 模型超参数
        C : float, 默认1.0, 正则化强度的倒数
        penalty : str, 默认'l2', 正则化类型
    
    Returns:
    --------
    dict
        包含模型、预测结果和评估指标的字典
    """
    # 验证输入数据
    validate_train_data(X_train, y_train, X_test, y_test, task_type='classification')
    
    # 获取特征名称（如果是DataFrame）
    feature_names = X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else None
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    
    # 参数自动优化
    if auto_optimize:
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2']
        }
        # 对于逻辑回归，需要处理penalty参数
        best_score = -np.inf
        best_model = None
        best_params = {}
        for penalty in param_grid['penalty']:
            try:
                solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
                temp_model = LogisticRegression(
                    penalty=penalty,
                    max_iter=1000,
                    random_state=42,
                    solver=solver
                )
                temp_params = {'C': param_grid['C']}
                search = RandomizedSearchCV(
                    temp_model, temp_params,
                    n_iter=min(10, len(param_grid['C'])),
                    cv=5, scoring='accuracy',
                    n_jobs=-1, random_state=42
                )
                search.fit(X_train, y_train)
                if search.best_score_ > best_score:
                    best_score = search.best_score_
                    best_model = search.best_estimator_
                    best_params = search.best_params_
                    best_params['penalty'] = penalty
            except Exception as e:
                continue
        model = best_model if best_model else model
    else:
        model.fit(X_train, y_train)
        best_params = params
    
    y_pred = model.predict(X_test)
    
    # 计算评估指标
    metrics = compute_classification_metrics(y_test, y_pred)
    
    # 提取特征重要性
    feature_importance = extract_feature_importance(model, feature_names)
    
    return {
        'model': model,
        'y_pred': y_pred,
        'y_test': y_test,
        'metrics': {
            'Accuracy': metrics['Accuracy'],
            'Classification Report': metrics['Classification Report']
        },
        'confusion_matrix': metrics['confusion_matrix'],
        'feature_importance': feature_importance,
        'best_params': best_params if auto_optimize else None
    }

