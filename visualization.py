"""
可视化模块
提供交互式图表生成和智能可视化推荐
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st


def recommend_charts(df, column):
    """
    根据列的数据类型推荐合适的图表类型
    
    Parameters:
    -----------
    df : pd.DataFrame
        数据框
    column : str
        列名
    
    Returns:
    --------
    list
        推荐的图表类型列表
    """
    dtype = df[column].dtype
    
    if pd.api.types.is_numeric_dtype(dtype):
        # 数值型：直方图、箱线图、小提琴图、散点图
        return ['histogram', 'box_plot', 'violin', 'scatter']
    elif pd.api.types.is_categorical_dtype(dtype) or dtype == 'object':
        unique_count = df[column].nunique()
        if unique_count <= 10:
            return ['bar_chart', 'pie_chart', 'count_plot']
        else:
            return ['bar_chart', 'word_cloud', 'count_plot']
    else:
        return ['line_chart', 'bar_chart', 'scatter']


def create_violin_plot(df, column, by=None, title=None):
    """创建小提琴图（可按分组变量展示分布）"""
    if by and by in df.columns:
        fig = px.violin(
            df,
            x=by,
            y=column,
            box=True,
            points='all',
            title=title or f'{column} 分布小提琴图 (按 {by} 分组)',
            labels={column: column, by: by},
        )
    else:
        fig = px.violin(
            df,
            y=column,
            box=True,
            points='all',
            title=title or f'{column} 分布小提琴图',
            labels={column: column},
        )
    
    fig.update_layout(
        template='plotly_white',
        hovermode='closest',
        height=500
    )
    return fig


def create_histogram(df, column, bins=30, title=None):
    """创建直方图"""
    fig = px.histogram(
        df, 
        x=column, 
        nbins=bins,
        title=title or f'{column} 分布直方图',
        labels={column: column},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(
        template='plotly_white',
        hovermode='closest',
        height=500
    )
    return fig


def create_box_plot(df, column, title=None):
    """创建箱线图"""
    fig = px.box(
        df, 
        y=column,
        title=title or f'{column} 箱线图',
        labels={column: column},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(
        template='plotly_white',
        hovermode='closest',
        height=500
    )
    return fig


def create_scatter_plot(df, x_col, y_col, color_col=None, title=None):
    """创建散点图"""
    if color_col:
        fig = px.scatter(
            df, 
            x=x_col, 
            y=y_col,
            color=color_col,
            title=title or f'{x_col} vs {y_col} 散点图',
            labels={x_col: x_col, y_col: y_col},
            hover_data=df.columns.tolist()
        )
    else:
        fig = px.scatter(
            df, 
            x=x_col, 
            y=y_col,
            title=title or f'{x_col} vs {y_col} 散点图',
            labels={x_col: x_col, y_col: y_col},
            hover_data=df.columns.tolist()
        )
    
    fig.update_layout(
        template='plotly_white',
        hovermode='closest',
        height=500
    )
    return fig


def create_density_contour(df, x_col, y_col, title=None):
    """创建二维密度等高线图"""
    fig = px.density_contour(
        df,
        x=x_col,
        y=y_col,
        title=title or f'{x_col} vs {y_col} 密度等高线图',
        labels={x_col: x_col, y_col: y_col},
        color_continuous_scale='Viridis'
    )
    fig.update_traces(contours_coloring="fill", contours_showlabels=True)
    fig.update_layout(
        template='plotly_white',
        height=500
    )
    return fig


def create_bar_chart(df, column, title=None):
    """创建柱状图"""
    value_counts = df[column].value_counts().head(20)
    fig = px.bar(
        x=value_counts.index,
        y=value_counts.values,
        title=title or f'{column} 频数分布',
        labels={'x': column, 'y': '频数'},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(
        template='plotly_white',
        hovermode='closest',
        height=500,
        xaxis_tickangle=-45
    )
    return fig


def create_correlation_heatmap(df, title=None):
    """创建相关性热力图"""
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) < 2:
        return None
    
    corr_matrix = numeric_df.corr()
    
    fig = px.imshow(
        corr_matrix,
        labels=dict(color="相关系数"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale='RdBu',
        aspect="auto",
        title=title or '特征相关性热力图'
    )
    
    fig.update_layout(
        template='plotly_white',
        height=max(500, len(corr_matrix.columns) * 40)
    )
    
    return fig


def create_scatter_matrix(df, columns=None, color_col=None, title=None):
    """创建散点图矩阵"""
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols[:5]  # 限制最多5个特征
    
    if len(columns) < 2:
        return None
    
    if color_col:
        fig = px.scatter_matrix(
            df,
            dimensions=columns,
            color=color_col,
            title=title or '散点图矩阵',
            labels={col: col for col in columns}
        )
    else:
        fig = px.scatter_matrix(
            df,
            dimensions=columns,
            title=title or '散点图矩阵',
            labels={col: col for col in columns}
        )
    
    fig.update_layout(
        template='plotly_white',
        height=800
    )
    
    return fig


def create_parallel_coordinates(df, columns, color_col=None, title=None):
    """创建平行坐标图（多变量模式探索）"""
    if not columns:
        return None
    
    data = df[columns].copy()
    # 仅保留数值型列
    data = data.select_dtypes(include=[np.number])
    if data.shape[1] < 2:
        return None
    
    if color_col and color_col in df.columns:
        data[color_col] = df[color_col]
        fig = px.parallel_coordinates(
            data,
            color=color_col,
            labels={col: col for col in data.columns},
            title=title or '平行坐标图',
            color_continuous_scale=px.colors.diverging.Tealrose
        )
    else:
        fig = px.parallel_coordinates(
            data,
            labels={col: col for col in data.columns},
            title=title or '平行坐标图',
            color=data.columns[0]
        )
    
    fig.update_layout(
        template='plotly_white',
        height=600
    )
    return fig


def create_distribution_comparison(df, columns, title=None):
    """创建分布对比图"""
    fig = make_subplots(
        rows=1, 
        cols=len(columns),
        subplot_titles=columns
    )
    
    for i, col in enumerate(columns):
        fig.add_trace(
            go.Histogram(x=df[col], name=col, nbinsx=30),
            row=1, col=i+1
        )
    
    fig.update_layout(
        template='plotly_white',
        title=title or '分布对比图',
        height=400,
        showlegend=False
    )
    
    return fig


def create_time_series_plot(df, date_col, value_col, title=None):
    """创建时间序列图"""
    if date_col not in df.columns:
        return None
    
    df_sorted = df.sort_values(date_col)
    fig = px.line(
        df_sorted,
        x=date_col,
        y=value_col,
        title=title or f'{value_col} 时间序列图',
        labels={date_col: date_col, value_col: value_col}
    )
    
    fig.update_layout(
        template='plotly_white',
        hovermode='x unified',
        height=500
    )
    
    return fig


def create_pie_chart(df, column, title=None):
    """创建饼图"""
    value_counts = df[column].value_counts().head(10)
    fig = px.pie(
        values=value_counts.values,
        names=value_counts.index,
        title=title or f'{column} 分布饼图'
    )
    
    fig.update_layout(
        template='plotly_white',
        height=500
    )
    
    return fig


def create_residual_plot(y_true, y_pred, title=None):
    """创建残差图"""
    residuals = y_true - y_pred
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('残差分布', '残差 vs 预测值'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 残差直方图
    fig.add_trace(
        go.Histogram(x=residuals, name='残差', nbinsx=30),
        row=1, col=1
    )
    
    # 残差散点图
    fig.add_trace(
        go.Scatter(x=y_pred, y=residuals, mode='markers', name='残差'),
        row=1, col=2
    )
    
    # 添加零线
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
    
    fig.update_layout(
        template='plotly_white',
        title=title or '残差分析',
        height=400,
        showlegend=False
    )
    
    fig.update_xaxes(title_text="残差值", row=1, col=1)
    fig.update_yaxes(title_text="频数", row=1, col=1)
    fig.update_xaxes(title_text="预测值", row=1, col=2)
    fig.update_yaxes(title_text="残差", row=1, col=2)
    
    return fig


def create_feature_importance_plot(importance_dict, title=None):
    """创建特征重要性图"""
    sorted_items = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    features, importances = zip(*sorted_items[:15])  # 只显示前15个
    
    fig = px.bar(
        x=list(importances),
        y=list(features),
        orientation='h',
        title=title or '特征重要性',
        labels={'x': '重要性', 'y': '特征'},
        color=list(importances),
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        template='plotly_white',
        height=max(400, len(features) * 30),
        showlegend=False
    )
    
    return fig

