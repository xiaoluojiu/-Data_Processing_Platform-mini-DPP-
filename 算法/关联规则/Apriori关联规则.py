"""
Apriori关联规则挖掘算法
频繁项集挖掘和关联规则生成
"""
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


def fit(df, min_support=0.1, min_confidence=0.5):
    """
    执行Apriori关联规则挖掘
    
    Parameters:
    -----------
    df : pd.DataFrame
        事务数据（每行是一个事务，每列是一个项）
    min_support : float
        最小支持度（0-1之间）
    min_confidence : float
        最小置信度（0-1之间）
    
    Returns:
    --------
    dict
        包含频繁项集和关联规则的字典
    """
    # 输入验证
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df 必须是 pandas.DataFrame")
    
    if len(df) == 0:
        raise ValueError("输入数据为空")
    
    if not (0 < min_support <= 1):
        raise ValueError(f"min_support 必须在 (0, 1] 范围内，当前值: {min_support}")
    
    if not (0 < min_confidence <= 1):
        raise ValueError(f"min_confidence 必须在 (0, 1] 范围内，当前值: {min_confidence}")
    
    # 将数据转换为事务格式
    # 假设数据已经是二进制格式（0/1）或需要转换
    if df.dtypes.apply(lambda x: x in [int, float, bool]).all():
        # 已经是二进制格式
        te_df = df.astype(bool)
    else:
        # 需要转换：将每个唯一值视为一个项
        transactions = []
        for _, row in df.iterrows():
            transaction = [f"{col}_{val}" for col, val in row.items() if pd.notna(val)]
            transactions.append(transaction)
        
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        te_df = pd.DataFrame(te_ary, columns=te.columns_)
    
    # 生成频繁项集
    try:
        frequent_itemsets = apriori(te_df, min_support=min_support, use_colnames=True)
        
        if len(frequent_itemsets) == 0:
            return {
                'frequent_itemsets': pd.DataFrame(),
                'rules': pd.DataFrame(),
                'message': '未找到满足最小支持度的频繁项集'
            }
        
        # 生成关联规则
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        
        return {
            'frequent_itemsets': frequent_itemsets,
            'rules': rules,
            'message': '成功生成关联规则'
        }
    except Exception as e:
        return {
            'frequent_itemsets': pd.DataFrame(),
            'rules': pd.DataFrame(),
            'message': f'执行失败: {str(e)}'
        }

