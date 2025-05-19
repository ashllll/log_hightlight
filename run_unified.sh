#!/bin/bash

echo "================================================"
echo "           日志高亮工具 - 统一启动脚本"
echo "================================================"

# 检查Python是否存在
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python. 请先安装Python."
    read -p "按回车键继续..."
    exit 1
fi

# 检查依赖需求文件是否存在
requirements_file="requirements.txt"
if [ ! -f "$requirements_file" ]; then
    echo "创建 requirements.txt 文件..."
    cat > "$requirements_file" << EOL
brotli
easyprocess
entrypoint2
patoolib
psutil
py7zr
PyQt5
PyQt5-Qt5
PyQt5-sip
pyunpack
pyzstd
rarfile
setuptools
setuptools-scm
EOL
    echo "requirements.txt 创建完成."
fi

# 检查libs目录是否存在
libs_dir="libs"
if [ ! -d "$libs_dir" ]; then
    echo "警告: libs目录不存在，将从网络安装依赖."
    use_network=true
else
    echo "检查libs目录是否含有依赖包..."
    if [ -z "$(ls -A $libs_dir/*.whl 2>/dev/null)" ]; then
        echo "警告: libs目录为空，将从网络安装依赖."
        use_network=true
    else
        use_network=false
    fi
fi

# 检查必要依赖是否已安装
echo "检查必要依赖是否已安装..."
if ! python3 -c "import PyQt5, toml" &> /dev/null; then
    echo "缺少必要依赖，开始安装..."
    
    if [ "$use_network" = false ]; then
        echo "从本地libs目录安装依赖..."
        pip3 install --no-index --find-links="$libs_dir" -r "$requirements_file"
        
        if [ $? -ne 0 ]; then
            echo "本地安装失败，尝试从网络安装..."
            pip3 install -r "$requirements_file"
            if [ $? -ne 0 ]; then
                echo "依赖安装失败."
                read -p "按回车键继续..."
                exit 1
            fi
        fi
    else
        echo "从网络安装依赖..."
        pip3 install -r "$requirements_file"
        if [ $? -ne 0 ]; then
            echo "依赖安装失败."
            read -p "按回车键继续..."
            exit 1
        fi
    fi
    echo "依赖安装完成."
else
    echo "必要依赖已安装，继续运行..."
fi

# 设置macOS特定环境变量
if [ "$(uname)" == "Darwin" ]; then
    echo "检测到macOS系统, 设置Qt环境变量..."
    export QT_MAC_WANTS_LAYER=1
    export QT_AUTO_SCREEN_SCALE_FACTOR=0
fi

# 运行平台检查
echo "进行平台兼容性检查..."
python3 platform_check.py
if [ $? -ne 0 ]; then
    echo "警告: 平台检查失败。程序可能无法正常运行。"
    echo -n "是否继续运行？(Y/N) "
    read continue
    if [ "${continue,,}" != "y" ]; then
        echo "用户取消运行。"
        read -p "按回车键继续..."
        exit 1
    fi
fi

# 启动程序
echo "启动日志高亮工具..."
python3 main.py
if [ $? -ne 0 ]; then
    echo "程序异常退出，错误代码: $?"
    read -p "按回车键继续..."
    exit $?
fi

echo "程序正常退出。"
exit 0 