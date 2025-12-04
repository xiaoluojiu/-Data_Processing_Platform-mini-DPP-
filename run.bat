@echo off
chcp 65001 >nul
echo ========================================
echo 正在启动数据分析平台 (Streamlit)...
echo ========================================
echo.

REM 方法1: 直接使用 conda run（推荐，最可靠）
conda run -n lxt streamlit run main.py 2>nul
if not errorlevel 1 (
    echo.
    echo [信息] 应用已关闭
    pause
    exit /b 0
)

REM 方法2: 如果 conda run 失败，尝试激活环境后运行
echo [信息] 尝试激活环境后运行...
call conda activate lxt 2>nul
if errorlevel 1 (
    echo [错误] 无法激活 conda 环境 'lxt'
    echo.
    echo 请检查:
    echo 1. 环境是否存在: conda env list
    echo 2. 如果不存在，请创建: conda create -n lxt python=3.9
    echo 3. 然后安装依赖: conda activate lxt ^&^& pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM 检查 streamlit
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo [警告] Streamlit 未安装，正在安装依赖...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [错误] 依赖安装失败，请手动运行: pip install -r requirements.txt
        pause
        exit /b 1
    )
)

echo [信息] 正在启动 Streamlit 应用...
echo [提示] 应用将在浏览器中自动打开
echo [提示] 按 Ctrl+C 可停止应用
echo.
echo ========================================
echo.

REM 运行 Streamlit
streamlit run main.py

REM 如果程序退出，暂停以便查看错误
if errorlevel 1 (
    echo.
    echo [错误] Streamlit 启动失败
    echo 请检查上述错误信息
)
pause

