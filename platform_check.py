#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
平台兼容性检测脚本
"""

import sys
import os
import platform
import importlib.util
import tempfile
from typing import Dict, List, Tuple

def check_module(module_name: str) -> bool:
    """检查模块是否可以导入"""
    try:
        if module_name == "patool":
            # 特殊处理patool，因为它的导入名称是patoolib
            module_name = "patoolib"
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return False
        return True
    except (ImportError, ModuleNotFoundError):
        return False

def check_gui() -> bool:
    """检查GUI环境是否可用"""
    try:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import Qt
        # 不显示窗口，仅检查QApplication是否能正常初始化
        app = QApplication([])
        return True
    except Exception as e:
        print(f"GUI环境检查失败: {e}")
        return False

def check_temp_dir() -> Dict:
    """检查临时目录权限并创建必要的子目录"""
    results = {
        "temp_dir_exists": True,
        "temp_dir_writable": True,
        "log_highlighter_dir_created": False,
        "extracted_archives_dir_created": False,
        "error": None
    }
    
    try:
        # 获取系统临时目录
        temp_dir = tempfile.gettempdir()
        
        # 检查临时目录是否存在且可写
        if not os.path.exists(temp_dir):
            results["temp_dir_exists"] = False
            results["error"] = f"系统临时目录 {temp_dir} 不存在"
            return results
        
        if not os.access(temp_dir, os.W_OK):
            results["temp_dir_writable"] = False
            results["error"] = f"系统临时目录 {temp_dir} 不可写"
            return results
        
        # 创建日志高亮器临时目录
        log_highlighter_dir = os.path.join(temp_dir, "log_highlighter_temp")
        try:
            os.makedirs(log_highlighter_dir, exist_ok=True)
            results["log_highlighter_dir_created"] = True
        except Exception as e:
            results["error"] = f"创建日志高亮器临时目录失败: {str(e)}"
            return results
        
        # 创建解压缩临时目录
        extracted_dir = os.path.join(log_highlighter_dir, "extracted_archives")
        try:
            os.makedirs(extracted_dir, exist_ok=True)
            results["extracted_archives_dir_created"] = True
        except Exception as e:
            results["error"] = f"创建解压缩临时目录失败: {str(e)}"
            
        return results
    except Exception as e:
        results["error"] = f"检查临时目录时发生错误: {str(e)}"
        return results

def run_checks() -> Dict:
    """运行所有检查并返回结果"""
    results = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "python_version": sys.version,
        },
        "environment": {
            "QT_MAC_WANTS_LAYER": os.environ.get("QT_MAC_WANTS_LAYER", "未设置"),
            "QT_AUTO_SCREEN_SCALE_FACTOR": os.environ.get("QT_AUTO_SCREEN_SCALE_FACTOR", "未设置"),
            "PYTHONPATH": os.environ.get("PYTHONPATH", "未设置"),
            "PATH": os.environ.get("PATH", "未设置"),
        },
        "modules": {
            "PyQt5": check_module("PyQt5"),
            "toml": check_module("toml"),
            "rarfile": check_module("rarfile"),
            "py7zr": check_module("py7zr"),
            "pyunpack": check_module("pyunpack"),
            "patool": check_module("patool"),
            "psutil": check_module("psutil"),
        },
        "gui": {
            "available": check_gui(),
        },
        "file_system": check_temp_dir()
    }
    return results

def print_results(results: Dict) -> None:
    """格式化并打印检查结果"""
    print("=" * 50)
    print("平台兼容性检测报告")
    print("=" * 50)
    
    print("\n## 平台信息")
    print(f"系统: {results['platform']['system']}")
    print(f"版本: {results['platform']['release']} ({results['platform']['version']})")
    print(f"Python版本: {results['platform']['python_version']}")
    
    print("\n## 环境变量")
    if results['platform']['system'] == "Darwin":  # macOS
        print(f"QT_MAC_WANTS_LAYER: {results['environment']['QT_MAC_WANTS_LAYER']} (macOS: 推荐设置为'1')")
        print(f"QT_AUTO_SCREEN_SCALE_FACTOR: {results['environment']['QT_AUTO_SCREEN_SCALE_FACTOR']} (macOS: 推荐设置为'0')")
    
    print("\n## 模块检查")
    required_modules = ["PyQt5", "toml"]
    optional_modules = ["rarfile", "py7zr", "pyunpack", "patool", "psutil"]
    
    print("必要模块:")
    for module in required_modules:
        status = "✓ 已安装" if results['modules'][module] else "✗ 未安装 (必需)"
        print(f"- {module}: {status}")
    
    print("\n可选模块:")
    for module in optional_modules:
        status = "✓ 已安装" if results['modules'][module] else "- 未安装 (可选)"
        print(f"- {module}: {status}")
    
    print("\n## GUI环境")
    gui_status = "✓ 正常" if results['gui']['available'] else "✗ 不可用"
    print(f"GUI环境: {gui_status}")
    
    print("\n## 文件系统")
    fs = results['file_system']
    if fs.get("error"):
        print(f"✗ 文件系统检查错误: {fs['error']}")
    else:
        temp_status = "✓ 可写" if fs.get("temp_dir_writable", False) else "✗ 不可写"
        print(f"临时目录权限: {temp_status}")
        
        if fs.get("log_highlighter_dir_created", False):
            print("✓ 日志高亮器临时目录已创建")
        else:
            print("✗ 日志高亮器临时目录创建失败")
            
        if fs.get("extracted_archives_dir_created", False):
            print("✓ 解压缩临时目录已创建")
        else:
            print("✗ 解压缩临时目录创建失败")
    
    print("\n## 总结")
    all_required = all(results['modules'][m] for m in required_modules)
    fs_ok = not fs.get("error") and fs.get("temp_dir_writable", False)
    
    if all_required and results['gui']['available'] and fs_ok:
        print("✓ 所有必要条件已满足，程序应该可以正常运行")
    else:
        print("✗ 有必要条件未满足，程序可能无法正常运行")
    
    optional_count = sum(results['modules'][m] for m in optional_modules)
    print(f"- {optional_count}/{len(optional_modules)} 可选模块已安装")

if __name__ == "__main__":
    import tempfile
    results = run_checks()
    print_results(results)
    
    # 返回状态码：如果必要条件未满足，返回非零值
    required_modules = ["PyQt5", "toml"]
    all_required = all(results['modules'][m] for m in required_modules)
    fs_ok = not results['file_system'].get("error") and results['file_system'].get("temp_dir_writable", False)
    
    if all_required and results['gui']['available'] and fs_ok:
        sys.exit(0)
    else:
        sys.exit(1) 