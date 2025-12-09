"""
配置和样式工具
"""
import streamlit as st
import pandas as pd


def get_plotly_config():
    """获取Plotly图表渲染配置"""
    return {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'autoScale2d', 'resetScale2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'chart',
            'height': 500,
            'width': 800,
            'scale': 1
        },
        'responsive': True,
        'staticPlot': False
    }


def get_custom_css():
    """获取自定义CSS样式"""
    return """
    <style>
    .stApp {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 25%, #7dd3fc 50%, #38bdf8 75%, #0ea5e9 100%);
        background-attachment: fixed;
        font-family: 'Inter', sans-serif;
        min-height: 100vh;
        will-change: auto;
        transform: translateZ(0);
        -webkit-transform: translateZ(0);
    }
    
    /* 主内容区域 - 使用纯白背景，不再需要半透明 */
    .main .block-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin: 1rem;
    }
    
    /* 确保文字颜色清晰 */
    .main .block-container,
    .main .block-container * {
        color: #1e293b;
    }
    
    /* 标题样式 - 使用渐变文字，容器背景独立设置 */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        padding: 1.5rem;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #3b82f6 0%, #2dd4bf 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    /* 标题容器背景 */
    .main-header-container {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 0.5rem;
        padding: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* 卡片式容器 */
    .css-1r6slb0, .stCard {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
    }
    
    /* 侧边栏美化 */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
        box-shadow: 2px 0 4px rgba(0, 0, 0, 0.05);
    }
    
    /* 按钮美化 */
    .stButton>button {
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric 样式 */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #3b82f6;
    }
    
    /* 隐藏默认菜单和页脚 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* 保留header但使其透明，确保侧边栏按钮正常显示 */
    header [data-testid="stHeader"] {
        background: transparent;
    }
    /* 只隐藏header中的菜单部分，保留侧边栏切换按钮 */
    header [data-testid="stHeader"] > div:first-child {
        display: none;
    }
    /* 确保侧边栏切换按钮始终可见和可点击 */
    header button,
    [data-testid="stHeader"] button {
        visibility: visible !important;
        display: block !important;
        z-index: 999;
    }
    
    /* 自定义进度条颜色 */
    .stProgress > div > div > div > div {
        background-color: #3b82f6;
    }
    
    /* 输入框美化 */
    .stTextInput>div>div>input, .stSelectbox>div>div>select {
        border-radius: 0.5rem;
        border: 1px solid #e2e8f0;
    }
    
    /* 标签页美化 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 0.5rem 0.5rem 0 0;
        padding: 0.75rem 1.5rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #dbeafe;
        color: #1e40af;
    }
    
    /* 警告和成功消息美化 */
    .stAlert {
        border-radius: 0.5rem;
        border-left: 4px solid;
    }
    
    /* 数据框美化 */
    .stDataFrame {
        border-radius: 0.5rem;
    }
    </style>
    """


def init_page_config():
    """初始化页面配置"""
    st.set_page_config(
        page_title="数据分析平台",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': "基于 Streamlit 的交互式数据分析平台"
        }
    )


def get_cached_df_operation(df, operation, *args, **kwargs):
    """
    缓存DataFrame常用操作结果，避免重复计算
    
    Parameters:
    -----------
    df : pd.DataFrame
        数据框
    operation : str
        操作名称 ('head', 'describe', 'columns_list', 'select_dtypes_numeric', 'select_dtypes_object')
    *args, **kwargs
        操作参数
    
    Returns:
    --------
    操作结果
    """
    df_id = id(df)
    cache_key = f'df_cache_{operation}_{df_id}'
    
    # 检查缓存是否存在且有效
    if cache_key in st.session_state:
        cached_result = st.session_state[cache_key]
        cached_df_id = st.session_state.get(f'{cache_key}_df_id')
        if cached_df_id == df_id:
            return cached_result
    
    # 执行操作并缓存
    if operation == 'head':
        result = df.head(*args, **kwargs) if args or kwargs else df.head(50)
    elif operation == 'describe':
        result = df.describe()
    elif operation == 'columns_list':
        result = df.columns.tolist()
    elif operation == 'select_dtypes_numeric':
        result = df.select_dtypes(include=['number']).columns.tolist()
    elif operation == 'select_dtypes_object':
        result = df.select_dtypes(include=['object', 'category']).columns.tolist()
    else:
        result = None
    
    st.session_state[cache_key] = result
    st.session_state[f'{cache_key}_df_id'] = df_id
    
    return result


def clear_df_cache(df_id=None):
    """
    清除DataFrame缓存
    
    Parameters:
    -----------
    df_id : int, optional
        特定DataFrame的ID，如果为None则清除所有缓存
    """
    if df_id is None:
        keys_to_remove = [k for k in st.session_state.keys() if k.startswith('df_cache_')]
        for key in keys_to_remove:
            del st.session_state[key]
    else:
        keys_to_remove = [k for k in st.session_state.keys() 
                         if k.startswith(f'df_cache_') and k.endswith(f'_{df_id}')]
        for key in keys_to_remove:
            del st.session_state[key]
            if f'{key}_df_id' in st.session_state:
                del st.session_state[f'{key}_df_id']


def init_session_state():
    """初始化session state"""
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'df_cleaned' not in st.session_state:
        st.session_state.df_cleaned = None
    if 'data_overview' not in st.session_state:
        st.session_state.data_overview = None
    if 'quality_scores' not in st.session_state:
        st.session_state.quality_scores = None
    if 'ml_results' not in st.session_state:
        st.session_state.ml_results = None
    if 'cleaning_log' not in st.session_state:
        st.session_state.cleaning_log = []
    if 'find_optimal_k' not in st.session_state:
        st.session_state.find_optimal_k = False
    if 'viz_options' not in st.session_state:
        st.session_state.viz_options = {}
    if 'df_metadata' not in st.session_state:
        st.session_state.df_metadata = None
    if 'df_metadata_id' not in st.session_state:
        st.session_state.df_metadata_id = None

