@echo off
setlocal

REM 获取批处理脚本所在的目录
set "SCRIPT_DIR=%~dp0"
REM 切换当前目录到脚本所在目录
cd /d "%SCRIPT_DIR%"

echo Verifying Python installation...
REM 检查 Python 是否安装并配置在 PATH 中
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python 未安装或未在系统 PATH 中找到。
    echo 请安装 Python (例如从 https://www.python.org/downloads/)
    echo 并确保将其添加到 PATH 环境变量中。
    pause
    exit /b 1
)
echo Python 已找到。

echo Verifying pip installation...
REM 检查 pip 是否安装
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: pip 未安装或未找到。pip 通常随 Python 一起安装。
    echo 请尝试重新安装 Python 或确保 Python 的 Scripts 目录在 PATH 中。
    pause
    exit /b 1
)
echo pip 已找到。

REM 检查 requirements.txt 文件是否存在
if not exist "requirements.txt" (
    echo ERROR: 在 %SCRIPT_DIR% 目录下未找到 requirements.txt 文件。
    pause
    exit /b 1
)

REM 检查 libs 目录是否存在且不为空
if not exist "libs" (
    echo WARNING: 'libs' 目录不存在。
    echo 您需要先将所需的依赖包下载到 'libs' 文件夹中。
    echo 请在此目录下的终端中运行以下命令：
    echo.
    echo   pip download -r requirements.txt -d libs
    echo.
    echo 在 'libs' 文件夹填充完毕后，请重新运行此脚本。
    pause
    exit /b 1
)

REM 简单检查 libs 目录是否为空
dir /b libs\*.* >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: 'libs' 目录存在但似乎是空的。
    echo 您需要先将所需的依赖包下载到 'libs' 文件夹中。
    echo 请在此目录下的终端中运行以下命令：
    echo.
    echo   pip download -r requirements.txt -d libs
    echo.
    echo 在 'libs' 文件夹填充完毕后，请重新运行此脚本。
    pause
    exit /b 1
)


echo Installing required libraries from the local 'libs' folder...
REM 从本地 libs 文件夹安装依赖，不访问网络索引
pip install --no-index --find-links=libs -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: 从 'libs' 文件夹安装库失败。
    echo 请检查 'libs' 文件夹的内容，并尝试再次运行：
    echo   pip download -r requirements.txt -d libs
    echo 以确保所有包都已正确下载。
    pause
    exit /b 1
)
echo 库安装成功。

echo.
echo Starting Log Highlighter...
REM 启动 Python 主程序
python log_hightlight.py

echo.
echo Log Highlighter 已退出。
pause
endlocal 