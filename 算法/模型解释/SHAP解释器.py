"""
SHAP模型解释器
使用SHAP值解释模型预测
"""
import shap
import warnings
import numpy as np
import pandas as pd


def explain(model, X, model_type='tree'):
    """
    使用SHAP解释模型
    
    Parameters:
    -----------
    model : 训练好的模型
    X : pd.DataFrame or np.ndarray
        特征数据
    model_type : str
        模型类型 ('tree', 'linear', 'other')
    
    Returns:
    --------
    tuple
        (shap_values, explainer) 或 (None, None) 如果失败
    """
    # 输入验证
    if model is None:
        warnings.warn("模型为空，无法生成SHAP解释")
        return None, None
    
    if X is None or len(X) == 0:
        warnings.warn("输入数据为空，无法生成SHAP解释")
        return None, None
    
    try:
        # 采样以减少计算量
        sample_size = min(100, len(X))
        if hasattr(X, 'sample'):
            X_sample = X.sample(n=sample_size, random_state=42)
        else:
            X_sample = X[:sample_size]
        
        # 根据模型类型选择合适的解释器
        if model_type == 'tree':
            explainer = shap.TreeExplainer(model)
        elif model_type == 'linear':
            explainer = shap.LinearExplainer(model, X_sample)
        else:
            # 对于其他模型，使用KernelExplainer
            explainer = shap.KernelExplainer(model.predict, X_sample)
        
        # 计算SHAP值
        shap_values = explainer.shap_values(X_sample)
        return shap_values, explainer
    except Exception as e:
        warnings.warn(f"SHAP解释生成失败: {str(e)}")
        return None, None

