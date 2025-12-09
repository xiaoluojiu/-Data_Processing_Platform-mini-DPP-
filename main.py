import streamlit as st
from streamlit_option_menu import option_menu

from utils import get_custom_css, init_page_config, init_session_state
from page_modules.data_pages import show_data_upload, show_data_overview, show_data_cleaning, show_eda
from page_modules.ml_pages import show_ml_analysis

init_page_config()

if 'css_injected' not in st.session_state:
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    st.session_state.css_injected = True

init_session_state()


def main():
    st.markdown('<div class="main-header-container"><div class="main-header">📊 交互式数据分析平台</div></div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/null/data-configuration.png", width=80)
        st.markdown("### 导航菜单")
        
        menu_index = st.session_state.get('menu_index', 0)
        menu = option_menu(
            menu_title=None,
            options=["数据上传", "数据概览", "数据清洗", "探索性分析", "机器学习"],
            icons=["cloud-upload", "grid", "brush", "bar-chart-line", "cpu"],
            menu_icon="cast",
            default_index=menu_index,
            key='main_menu',
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#3b82f6", "font-size": "1.1rem"},
                "nav-link": {
                    "font-size": "0.95rem", 
                    "text-align": "left", 
                    "margin": "5px 0",
                    "border-radius": "0.5rem",
                    "--hover-color": "#eff6ff"
                },
                "nav-link-selected": {"background-color": "#dbeafe", "color": "#1e40af", "font-weight": "600"},
            }
        )
        
        menu_options = ["数据上传", "数据概览", "数据清洗", "探索性分析", "机器学习"]
        if menu in menu_options:
            st.session_state.menu_index = menu_options.index(menu)
        
        st.markdown("---")

    if menu == "数据上传":
        show_data_upload()
    elif menu == "数据概览":
        show_data_overview()
    elif menu == "数据清洗":
        show_data_cleaning()
    elif menu == "探索性分析":
        show_eda()
    elif menu == "机器学习":
        show_ml_analysis()


if __name__ == "__main__":
    main()
