"""
数据加载与预处理模块
支持多源数据导入、智能数据概览、交互式数据清洗
"""
import pandas as pd
import numpy as np
import streamlit as st
from io import StringIO
import json


@st.cache_data
def load_data(file, file_type, sep=None):
    """
    加载数据文件
    
    Parameters:
    -----------
    file : UploadedFile
        上传的文件对象
    file_type : str
        文件类型 ('csv', 'excel', 'json')
    sep : str, optional
        分隔符，如果为None则自动检测
    
    Returns:
    --------
    pd.DataFrame
        加载的数据框
    """
    try:
        if file_type == 'csv':
            # 尝试不同的编码和分隔符
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
            separators = [sep] if sep else [',', ';', '\t', '|']
            
            for encoding in encodings:
                for separator in separators:
                    try:
                        file.seek(0)
                        # 先读取前几行来检测分隔符
                        if sep is None:
                            sample = file.read(1024).decode(encoding, errors='ignore')
                            file.seek(0)
                            
                            # 检测最可能的分隔符
                            if separator == ',' and sample.count(';') > sample.count(','):
                                separator = ';'
                            elif separator == ',' and sample.count('\t') > sample.count(','):
                                separator = '\t'
                        
                        df = pd.read_csv(
                            file, 
                            encoding=encoding,
                            sep=separator,
                            quotechar='"',
                            skipinitialspace=True,
                            on_bad_lines='skip'  # 跳过有问题的行
                        )
                        
                        # 检查数据是否有效（至少2列）
                        if len(df.columns) >= 2 or len(df) > 0:
                            # 清理列名：移除引号、空格等
                            df.columns = df.columns.str.strip().str.strip('"').str.strip("'")
                            return df
                    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
                        continue
                    except Exception:
                        continue
            
            # 如果所有方法都失败，尝试使用错误处理
            file.seek(0)
            try:
                df = pd.read_csv(
                    file, 
                    encoding='utf-8', 
                    errors='ignore',
                    sep=sep or ',',
                    on_bad_lines='skip'
                )
                df.columns = df.columns.str.strip().str.strip('"').str.strip("'")
                return df
            except:
                # 最后尝试：读取为文本然后手动解析
                file.seek(0)
                content = file.read().decode('utf-8', errors='ignore')
                lines = content.split('\n')
                if len(lines) > 1:
                    # 尝试检测分隔符
                    first_line = lines[0]
                    if ';' in first_line:
                        sep_char = ';'
                    elif '\t' in first_line:
                        sep_char = '\t'
                    else:
                        sep_char = ','
                    
                    # 手动解析
                    headers = [h.strip().strip('"').strip("'") for h in first_line.split(sep_char)]
                    data = []
                    for line in lines[1:]:
                        if line.strip():
                            row = [v.strip().strip('"').strip("'") for v in line.split(sep_char)]
                            if len(row) == len(headers):
                                data.append(row)
                    
                    if data:
                        df = pd.DataFrame(data, columns=headers)
                        return df
                
                return None
        
        elif file_type == 'excel':
            file.seek(0)
            df = pd.read_excel(file)
            # 清理列名
            df.columns = df.columns.str.strip().str.strip('"').str.strip("'")
            return df
        
        elif file_type == 'json':
            file.seek(0)
            content = file.read().decode('utf-8')
            data = json.loads(content)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                df = pd.DataFrame(data)
            # 清理列名
            df.columns = df.columns.str.strip().str.strip('"').str.strip("'")
            return df
        
    except Exception as e:
        st.error(f"数据加载失败: {str(e)}")
        import traceback
        st.error(f"详细错误: {traceback.format_exc()}")
        return None


def calculate_data_quality_score(df):
    """
    计算数据质量评分
    
    Parameters:
    -----------
    df : pd.DataFrame
        数据框
    
    Returns:
    --------
    dict
        包含各项质量指标和总分的字典
    """
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    completeness = 1 - (missing_cells / total_cells) if total_cells > 0 else 0
    
    # 唯一性评分（基于重复行比例）
    duplicate_rows = df.duplicated().sum()
    uniqueness = 1 - (duplicate_rows / len(df)) if len(df) > 0 else 0
    
    # 一致性评分（基于数据类型一致性）
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    consistency = len(numeric_cols) / len(df.columns) if len(df.columns) > 0 else 0
    
    # 综合评分（加权平均）
    overall_score = (completeness * 0.5 + uniqueness * 0.3 + consistency * 0.2) * 100
    
    return {
        'completeness': completeness * 100,
        'uniqueness': uniqueness * 100,
        'consistency': consistency * 100,
        'overall_score': overall_score,
        'missing_count': missing_cells,
        'duplicate_count': duplicate_rows
    }


def get_data_overview(df):
    """
    生成数据概览信息
    
    Parameters:
    -----------
    df : pd.DataFrame
        数据框
    
    Returns:
    --------
    dict
        包含数据基本信息的字典
    """
    overview = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'numeric_stats': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
        'categorical_stats': {}
    }
    
    # 分类变量统计
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        overview['categorical_stats'][col] = {
            'unique_count': df[col].nunique(),
            'top_values': df[col].value_counts().head(5).to_dict()
        }
    
    return overview


def clean_data(df, cleaning_options):
    """
    根据用户选择的清洗选项清洗数据
    
    Parameters:
    -----------
    df : pd.DataFrame
        原始数据框
    cleaning_options : dict
        清洗选项字典，包含：
        - missing_value_strategy: 缺失值处理策略
        - missing_value_method: 填充方法（当strategy为'fill'时）
        - remove_duplicates: 是否删除重复值
        - outlier_method: 异常值处理方法
        - outlier_threshold: 异常值阈值（Z-score或IQR倍数）
    
    Returns:
    --------
    pd.DataFrame
        清洗后的数据框
    """
    df_cleaned = df.copy()
    changes_log = []
    
    # 处理缺失值
    if cleaning_options.get('missing_value_strategy') == 'drop':
        original_len = len(df_cleaned)
        df_cleaned = df_cleaned.dropna()
        changes_log.append(f"删除了 {original_len - len(df_cleaned)} 行包含缺失值的记录")
    
    elif cleaning_options.get('missing_value_strategy') == 'fill':
        method = cleaning_options.get('missing_value_method', 'mean')
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df_cleaned[col].isnull().sum() > 0:
                if method == 'mean':
                    fill_value = df_cleaned[col].mean()
                elif method == 'median':
                    fill_value = df_cleaned[col].median()
                elif method == 'mode':
                    fill_value = df_cleaned[col].mode()[0] if len(df_cleaned[col].mode()) > 0 else 0
                else:
                    fill_value = 0
                
                missing_count = df_cleaned[col].isnull().sum()
                df_cleaned[col].fillna(fill_value, inplace=True)
                changes_log.append(f"列 '{col}': 使用{method}填充了 {missing_count} 个缺失值")
        
        # 分类变量用众数填充
        categorical_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df_cleaned[col].isnull().sum() > 0:
                mode_value = df_cleaned[col].mode()[0] if len(df_cleaned[col].mode()) > 0 else 'Unknown'
                missing_count = df_cleaned[col].isnull().sum()
                df_cleaned[col].fillna(mode_value, inplace=True)
                changes_log.append(f"列 '{col}': 使用众数填充了 {missing_count} 个缺失值")
    
    # 处理重复值
    if cleaning_options.get('remove_duplicates', False):
        original_len = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        changes_log.append(f"删除了 {original_len - len(df_cleaned)} 行重复记录")
    
    # 处理异常值
    if cleaning_options.get('outlier_method') != 'none':
        method = cleaning_options.get('outlier_method')
        threshold = cleaning_options.get('outlier_threshold', 3.0)
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'zscore':
                z_scores = np.abs((df_cleaned[col] - df_cleaned[col].mean()) / df_cleaned[col].std())
                outliers = df_cleaned[z_scores > threshold]
                if len(outliers) > 0:
                    # 可以选择删除或替换
                    if cleaning_options.get('outlier_action') == 'remove':
                        df_cleaned = df_cleaned[z_scores <= threshold]
                        changes_log.append(f"列 '{col}': 删除了 {len(outliers)} 个异常值 (Z-score > {threshold})")
                    else:
                        # 用边界值替换
                        lower_bound = df_cleaned[col].mean() - threshold * df_cleaned[col].std()
                        upper_bound = df_cleaned[col].mean() + threshold * df_cleaned[col].std()
                        df_cleaned.loc[df_cleaned[col] < lower_bound, col] = lower_bound
                        df_cleaned.loc[df_cleaned[col] > upper_bound, col] = upper_bound
                        changes_log.append(f"列 '{col}': 修正了 {len(outliers)} 个异常值 (Z-score > {threshold})")
            
            elif method == 'iqr':
                Q1 = df_cleaned[col].quantile(0.25)
                Q3 = df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = df_cleaned[(df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)]
                
                if len(outliers) > 0:
                    if cleaning_options.get('outlier_action') == 'remove':
                        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                        changes_log.append(f"列 '{col}': 删除了 {len(outliers)} 个异常值 (IQR方法)")
                    else:
                        df_cleaned.loc[df_cleaned[col] < lower_bound, col] = lower_bound
                        df_cleaned.loc[df_cleaned[col] > upper_bound, col] = upper_bound
                        changes_log.append(f"列 '{col}': 修正了 {len(outliers)} 个异常值 (IQR方法)")
    
    return df_cleaned, changes_log

