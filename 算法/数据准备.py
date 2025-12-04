"""
数据准备模块
用于机器学习前的数据预处理和准备
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings


def prepare_data_for_ml(df, target_column, feature_columns=None, task_type='classification'):
    """
    准备机器学习数据
    
    Parameters:
    -----------
    df : pd.DataFrame
        数据框
    target_column : str
        目标列名
    feature_columns : list, optional
        要使用的特征列列表，如果为None则使用除目标列外的所有列
    task_type : str
        任务类型 ('classification' 或 'regression')
    
    Returns:
    --------
    tuple
        (X, y, feature_names, label_encoders)
    """
    df_clean = df.copy()
    
    # 检查目标列是否存在
    if target_column not in df_clean.columns:
        raise ValueError(f"目标列 '{target_column}' 不存在于数据框中")
    
    # 分离特征和目标
    y = df_clean.pop(target_column)
    
    # 选择特征列
    if feature_columns is None:
        # 使用除目标列外的所有列
        X = df_clean.copy()
    else:
        # 只使用指定的特征列
        missing_cols = [col for col in feature_columns if col not in df_clean.columns]
        if missing_cols:
            raise ValueError(f"以下特征列不存在: {missing_cols}")
        X = df_clean[feature_columns].copy()
    
    # 检查是否有特征
    if X.empty or len(X.columns) == 0:
        raise ValueError("没有可用的特征列。请确保数据框中除了目标列外还有其他列。")
    
    # 处理缺失值
    # 对于特征：删除包含缺失值的行
    # 对于目标变量：也删除缺失值
    valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
    
    if valid_mask.sum() == 0:
        raise ValueError("数据中包含过多缺失值，删除缺失值后没有剩余数据。请先进行数据清洗。")
    
    X = X[valid_mask].copy()
    y = y[valid_mask].copy()
    
    if len(X) == 0:
        raise ValueError("删除缺失值后没有剩余数据。请先进行数据清洗。")
    
    # 检查是否有足够的样本
    if len(X) < 10:
        raise ValueError(f"样本数量过少（{len(X)}个），至少需要10个样本才能进行机器学习。")
    
    # 处理分类特征
    label_encoders = {}
    for col in X.select_dtypes(include=['object', 'category']).columns:
        try:
            le = LabelEncoder()
            # 处理缺失值和特殊值
            col_data = X[col].astype(str).fillna('missing')
            X[col] = le.fit_transform(col_data)
            label_encoders[col] = le
        except Exception as e:
            # 编码失败时跳过此列
            warnings.warn(f"列 '{col}' 编码失败: {str(e)}，将跳过此列")
            X = X.drop(columns=[col])
    
    # 检查编码后是否还有特征
    if X.empty or len(X.columns) == 0:
        raise ValueError("编码后没有可用的特征列。")
    
    # 处理目标变量（如果是分类任务）
    if task_type == 'classification':
        if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
            le_target = LabelEncoder()
            y_encoded = y.astype(str).fillna('missing')
            y = le_target.fit_transform(y_encoded)
            label_encoders['target'] = le_target
        else:
            # 数值型目标变量，检查是否适合分类
            unique_values = y.nunique()
            # 如果唯一值太多，强制转换为分类（使用分箱）
            if unique_values > 50:
                # 使用分位数分箱，最多分成10类
                n_bins = min(10, unique_values // 5)
                if n_bins >= 2:
                    y_binned = pd.cut(y, bins=n_bins, labels=False, duplicates='drop')
                    # 处理NaN值（边界情况）
                    y_binned = y_binned.fillna(0).astype(int)
                    y = y_binned.values
                    # 返回警告信息，由调用者显示
                    warnings.warn(
                        f"目标变量有 {unique_values} 个唯一值，已自动分箱为 {n_bins} 个类别用于分类任务。",
                        UserWarning
                    )
                else:
                    raise ValueError(
                        f"目标变量有 {unique_values} 个唯一值，不适合分类任务。"
                        f"请选择'回归'任务类型，或先对目标变量进行离散化处理。"
                    )
            else:
                # 唯一值较少，直接转换为整数用于分类
                y = y.astype(int)
    
    feature_names = X.columns.tolist()
    
    # 确保所有列都是数值型
    X = X.select_dtypes(include=[np.number])
    
    if X.empty or len(X.columns) == 0:
        raise ValueError("没有可用的数值型特征列。")
    
    # 标准化特征
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=feature_names, index=X.index)
    except Exception as e:
        raise ValueError(f"特征标准化失败: {str(e)}。请检查数据是否包含有效数值。")
    
    return X, y, feature_names, label_encoders

