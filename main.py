#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
主程序模块
负责应用程序的主入口、界面初始化和用户交互逻辑。
"""

import sys
import os
import platform
import time

# 平台特定设置
if platform.system() == "Darwin":  # macOS
    os.environ['QT_MAC_WANTS_LAYER'] = '1'
    os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '0'

import json
import toml
import signal
import atexit
import gzip
import shutil
import tempfile
import logging
from typing import List, Dict
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton, QFileDialog,
    QHBoxLayout, QVBoxLayout, QMessageBox, QGroupBox, QComboBox, QCheckBox,
    QTextEdit, QScrollArea, QProgressBar, QListWidget, QSplitter, QSpinBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView, QFrame
)
from PyQt5.QtCore import Qt, QTimer, QThread, QCoreApplication
from PyQt5.QtGui import QTextCharFormat, QColor, QFont, QIcon, QPixmap
from utils import CONFIG_DEFAULTS, generate_color
from file_handler import TempFileManager, CheckpointManager
from memory_manager import MemoryMonitor, MemoryStatusWidget
from keyword_matcher import Keyword
from scan_worker_complete import ScanWorker
import gc
import webbrowser
import zipfile
import tarfile

# 尝试导入 rarfile 库，用于解压 .rar 文件
try:
    import rarfile
    RARFILE_AVAILABLE = True
    logging.info("rarfile 库可用，支持 .rar 压缩包解压")
except ImportError:
    RARFILE_AVAILABLE = False
    logging.warning("无法导入 rarfile 库，.rar 压缩包解压功能不可用")

# 尝试导入 unarr 库，用于纯 Python 解压
try:
    import unrar  # pip install unrar
except ImportError:
    unrar = None

# 尝试导入 pyunpack 和 patool，增强解压功能
try:
    from pyunpack import Archive
    from easyprocess import EasyProcess
    PYUNPACK_AVAILABLE = True
    logging.info("pyunpack 和 patool 库可用，支持多种格式压缩包解压")
except ImportError:
    PYUNPACK_AVAILABLE = False
    logging.warning("无法导入 pyunpack 或 patool 库，解压功能受限")

# 尝试导入 py7zr 库，用于解压 .7z 文件
try:
    import py7zr
    PY7ZR_AVAILABLE = True
    logging.info("py7zr 库可用，支持 .7z 压缩包解压")
except ImportError:
    PY7ZR_AVAILABLE = False
    logging.warning("无法导入 py7zr 库，.7z 压缩包解压功能不可用")


class LogHighlighter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("日志关键词高亮工具")
        self.resize(1200, 800)
        self.settings_path = "settings.json"
        self.config_path = None
        self.history = {"sources": [], "keywords": [], "cores": os.cpu_count() or 1}
        self.group_colors = {}
        self.custom_keyword_checks = []
        
        # 优化：使用共享的临时文件和检查点管理器
        self.temp_manager = TempFileManager()
        self.checkpoint_manager = CheckpointManager()
        
        # 工作线程
        self.worker = None
        self.decompress_worker = None
        
        # 解压后的文件路径
        self.decompressed_files = []
        
        # 初始化参数
        self.config_params = {
            "max_results": CONFIG_DEFAULTS["max_results"],
            "time_range_hours": CONFIG_DEFAULTS["time_range_hours"],
            "chunk_size": CONFIG_DEFAULTS["chunk_size"],
            "thread_timeout": CONFIG_DEFAULTS["thread_timeout"],
            "max_file_size": CONFIG_DEFAULTS["max_file_size"],
            "batch_update_size": CONFIG_DEFAULTS["batch_update_size"],
            "scan_mode": CONFIG_DEFAULTS["scan_mode"],
            "prefilter_enabled": CONFIG_DEFAULTS["prefilter_enabled"],
            "bitmap_filter_enabled": CONFIG_DEFAULTS["bitmap_filter_enabled"],
            "large_file_threshold": CONFIG_DEFAULTS["large_file_threshold"],
            "huge_file_threshold": CONFIG_DEFAULTS["huge_file_threshold"],
            "use_process_pool": True
        }
        self.init_ui()
        QTimer.singleShot(100, self.load_settings)
        # 捕获系统信号优雅退出
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        atexit.register(self.cleanup_temp_files)

    def signal_handler(self, signum: int, frame: object) -> None:
        """处理系统信号"""
        logging.info(f"收到系统信号 {signum}，正在关闭程序")
        self.close()

    def cleanup_temp_files(self) -> None:
        """清理所有临时文件，并回收内存。"""
        # 使用 TempFileManager 清理临时文件
        if hasattr(self, 'temp_manager'):
            self.temp_manager.cleanup_all()
            
        # 清理其他可能的临时文件
        temp_dir = tempfile.gettempdir()
        for f in os.listdir(temp_dir):
            if f.startswith(tempfile.gettempprefix()) and f.endswith('.html'):
                try:
                    os.remove(os.path.join(temp_dir, f))
                    logging.info(f"清理临时文件: {f}")
                except Exception as e:
                    logging.error(f"清理临时文件 {f} 失败: {e}")
                    
        # 回收内存
        gc.collect()

    def init_ui(self) -> None:
        """初始化用户界面。"""
        mainSplitter = QSplitter(Qt.Vertical)
        self.setCentralWidget(mainSplitter)
        topSplitter = QSplitter(Qt.Horizontal)
        mainSplitter.addWidget(topSplitter)

        # 左侧面板
        left = self._init_left_panel()
        topSplitter.addWidget(left)

        # 右侧面板
        right = self._init_right_panel()
        topSplitter.addWidget(right)

        # 调试输出
        dbg_g = self._init_debug_panel()
        mainSplitter.addWidget(dbg_g)

        topSplitter.setStretchFactor(0, 1)
        topSplitter.setStretchFactor(1, 2)
        mainSplitter.setStretchFactor(0, 3)
        mainSplitter.setStretchFactor(1, 1)

    def _init_left_panel(self) -> QWidget:
        """初始化左侧面板。"""
        left = QWidget()
        ll = QVBoxLayout(left)

        cfg_g = QGroupBox("配置文件 (TOML)")
        cfg_l = QHBoxLayout(cfg_g)
        self.cfg_edit = QLineEdit(readOnly=True)
        self.btn_cfg = QPushButton("选择配置文件")
        self.btn_cfg.clicked.connect(self.select_config)
        cfg_l.addWidget(self.cfg_edit)
        cfg_l.addWidget(self.btn_cfg)
        ll.addWidget(cfg_g)

        src_g = QGroupBox("日志源 (目录/文件)")
        src_l = QHBoxLayout(src_g)
        self.src_list = QListWidget()
        self.src_list.setSelectionMode(QListWidget.ExtendedSelection)
        btns = QVBoxLayout()
        self.btn_add_dir = QPushButton("添加目录")
        self.btn_add_dir.clicked.connect(self.add_directory)
        self.btn_add_file = QPushButton("添加文件")
        self.btn_add_file.clicked.connect(self.add_file)
        self.btn_add_archive = QPushButton("添加压缩包")
        self.btn_add_archive.clicked.connect(self.add_archive)
        self.btn_remove = QPushButton("移除所选")
        self.btn_decompress = QPushButton("解压选中文件")
        self.btn_decompress.clicked.connect(self.decompress_selected)
        btns.addWidget(self.btn_decompress)
        self.btn_remove.clicked.connect(self.remove_sources)
        self.btn_clear = QPushButton("清除历史")
        self.btn_clear.clicked.connect(self.clear_history)
        for b in (self.btn_add_dir, self.btn_add_file, self.btn_add_archive, self.btn_remove, self.btn_clear):
            btns.addWidget(b)
        btns.addStretch()
        src_l.addWidget(self.src_list, 4)
        src_l.addLayout(btns, 1)
        ll.addWidget(src_g)

        cpu_g = QGroupBox("CPU 核心数")
        cpu_l = QHBoxLayout(cpu_g)
        cpu_l.addWidget(QLabel("使用核心:"))
        self.spin_cores = QSpinBox()
        maxc = os.cpu_count() or 1
        self.spin_cores.setRange(1, maxc)
        self.spin_cores.setValue(maxc)
        cpu_l.addWidget(self.spin_cores)
        cpu_l.addStretch()
        ll.addWidget(cpu_g)

        params_g = QGroupBox("参数设置")
        params_l = QVBoxLayout(params_g)
        
        max_results_l = QHBoxLayout()
        max_results_label = QLabel("最大结果数:")
        self.spin_max_results = QSpinBox()
        self.spin_max_results.setRange(1000, 100000)
        self.spin_max_results.setValue(self.config_params["max_results"])
        self.spin_max_results.valueChanged.connect(lambda v: self.update_config_param("max_results", v))
        max_results_l.addWidget(max_results_label)
        max_results_l.addWidget(self.spin_max_results)
        max_results_l.addStretch()
        params_l.addLayout(max_results_l)

        scan_mode_l = QHBoxLayout()
        scan_mode_label = QLabel("扫描模式:")
        self.scan_mode_combo = QComboBox()
        self.scan_mode_combo.addItems(["自动", "快速（低内存）", "精确（高内存）", "平衡"])
        self.scan_mode_combo.currentTextChanged.connect(self._on_scan_mode_changed)
        scan_mode_l.addWidget(scan_mode_label)
        scan_mode_l.addWidget(self.scan_mode_combo)
        scan_mode_l.addStretch()
        params_l.addLayout(scan_mode_l)
        
        prefilter_l = QHBoxLayout()
        self.prefilter_check = QCheckBox("启用关键词预过滤")
        self.prefilter_check.setChecked(CONFIG_DEFAULTS["prefilter_enabled"])
        self.prefilter_check.stateChanged.connect(lambda v: self.update_config_param("prefilter_enabled", bool(v)))
        
        self.bitmap_filter_check = QCheckBox("启用位图过滤")
        self.bitmap_filter_check.setChecked(CONFIG_DEFAULTS["bitmap_filter_enabled"])
        self.bitmap_filter_check.stateChanged.connect(lambda v: self.update_config_param("bitmap_filter_enabled", bool(v)))
        
        self.use_process_pool = QCheckBox("使用进程池")
        self.use_process_pool.setChecked(True)
        
        prefilter_l.addWidget(self.prefilter_check)
        prefilter_l.addWidget(self.bitmap_filter_check)
        params_l.addLayout(prefilter_l)
        
        process_l = QHBoxLayout()
        process_l.addWidget(self.use_process_pool)
        process_l.addStretch()
        params_l.addLayout(process_l)

        params_l.addStretch()
        ll.addWidget(params_g)
        ll.addStretch()

        return left

    def _init_right_panel(self) -> QWidget:
        """初始化右侧面板。"""
        right = QWidget()
        rl = QVBoxLayout(right)

        # 分组复选框面板
        grp_g = QGroupBox("关键词分组 (可多选)")
        grp_scroll = QScrollArea()
        grp_scroll.setWidgetResizable(True)
        grp_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        grp_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        grp_inner = QWidget()
        self.group_layout = QVBoxLayout(grp_inner)
        grp_scroll.setWidget(grp_inner)
        grp_l = QHBoxLayout(grp_g)
        grp_l.addWidget(grp_scroll)
        rl.addWidget(grp_g)

        # 自定义关键词面板
        cus_g = QGroupBox("自定义关键词 (可选填)")
        cus_l = QVBoxLayout(cus_g)
        
        # 关键词输入区域
        input_l = QHBoxLayout()
        self.kw_text = QLineEdit()
        self.kw_text.setPlaceholderText("输入关键词...")
        self.kw_annotation = QLineEdit()
        self.kw_annotation.setPlaceholderText("注释 (可选)...")
        self.btn_add = QPushButton("添加")
        self.btn_add.clicked.connect(self.add_custom_keyword)
        input_l.addWidget(self.kw_text, 3)
        input_l.addWidget(self.kw_annotation, 2)
        input_l.addWidget(self.btn_add, 1)
        cus_l.addLayout(input_l)
        
        # 选项区域
        opts_l = QHBoxLayout()
        self.chk_case = QCheckBox("区分大小写")
        self.chk_word = QCheckBox("全词匹配")
        self.chk_regex = QCheckBox("使用正则表达式")
        opts_l.addWidget(self.chk_case)
        opts_l.addWidget(self.chk_word)
        opts_l.addWidget(self.chk_regex)
        opts_l.addStretch()
        cus_l.addLayout(opts_l)
        
        # 自定义关键词列表
        self.custom_scroll = QScrollArea()
        self.custom_scroll.setWidgetResizable(True)
        self.custom_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.custom_inner = QWidget()
        self.custom_layout = QVBoxLayout(self.custom_inner)
        self.custom_layout.setAlignment(Qt.AlignTop)
        self.custom_scroll.setWidget(self.custom_inner)
        cus_l.addWidget(self.custom_scroll)
        
        # 按钮组
        btns_l = QHBoxLayout()
        self.btn_clear_selected = QPushButton("清除选中项")
        self.btn_clear_selected.clicked.connect(self.clear_selected_custom_keywords)
        self.btn_select_all = QPushButton("全选")
        self.btn_select_all.clicked.connect(self.select_all_custom_keywords)
        btns_l.addWidget(self.btn_clear_selected)
        btns_l.addWidget(self.btn_select_all)
        btns_l.addStretch()
        cus_l.addLayout(btns_l)
        
        rl.addWidget(cus_g)
        
        # 操作按钮区
        act_g = QGroupBox("操作")
        act_l = QHBoxLayout(act_g)
        self.btn_analysis = QPushButton("开始分析")
        self.btn_analysis.clicked.connect(self.analyze_combined_keywords)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.cancel_analysis)
        self.btn_cancel.setVisible(False)
        self.btn_preview = QPushButton("查看预览")
        self.btn_preview.clicked.connect(self.preview_results)
        self.btn_preview.setVisible(False)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        act_l.addWidget(self.btn_analysis)
        act_l.addWidget(self.btn_cancel)
        act_l.addWidget(self.btn_preview)
        act_l.addWidget(self.progress)
        rl.addWidget(act_g)

        return right

    def _init_debug_panel(self) -> QGroupBox:
        """初始化调试输出面板。"""
        dbg_g = QGroupBox("调试输出")
        dbg_l = QVBoxLayout(dbg_g)
        self.debug = QTextEdit(readOnly=True)
        self.error_log = QTextEdit(readOnly=True)
        dbg_l.addWidget(self.debug)
        return dbg_g

    def _on_scan_mode_changed(self, mode_text: str) -> None:
        """处理扫描模式变更"""
        mode_map = {
            "自动": "auto",
            "快速（低内存）": "fast", 
            "精确（高内存）": "accurate", 
            "平衡": "balanced"
        }
        
        if mode_text in mode_map:
            self.update_config_param("scan_mode", mode_map[mode_text])
            
    def update_config_param(self, key: str, value) -> None:
        """更新配置参数并保存设置。"""
        self.config_params[key] = value
        self.save_settings()
        
        # 对特定参数的变更进行额外处理
        if key == "scan_mode":
            self._handle_scan_mode_change(value)
        # 对于布尔值参数的特殊处理
        elif isinstance(value, bool):
            self._handle_boolean_param_change(key, value)

    def select_config(self) -> None:
        """选择 TOML 配置文件。"""
        cfg, _ = QFileDialog.getOpenFileName(self, "选择配置文件", "", "TOML (*.toml)")
        if cfg:
            try:
                toml.load(cfg)  # 校验配置文件
                self.config_path = cfg
                self.cfg_edit.setText(cfg)
                self.save_settings()
                self.update_group_checkboxes()
            except Exception as e:
                logging.error(f"加载配置文件失败: {e}")
                QMessageBox.critical(self, "配置文件错误", f"加载配置文件失败: {str(e)}")

    def _is_archive_file(self, filepath: str) -> bool:
        """检查文件是否为支持的压缩格式"""
        extensions = ('.rar', '.zip', '.7z', '.tar', '.tar.gz', '.tgz', '.gz')
        return filepath.lower().endswith(extensions)

    def update_group_checkboxes(self) -> None:
        """根据加载的 TOML 配置文件更新关键词分组的复选框。"""
        # 清除现有的分组复选框
        for i in reversed(range(self.group_layout.count())):
            widget = self.group_layout.itemAt(i).widget()
            if widget:
                self.group_layout.removeWidget(widget)
                widget.deleteLater()
        self.group_colors.clear()  # 清除旧颜色映射

        # 如果有配置文件，加载分组信息
        if self.config_path and os.path.isfile(self.config_path):
            try:
                config = toml.load(self.config_path)
                self.debug.append(f"加载配置文件: {self.config_path}")

                # 检查是否存在顶层 'group' 字典 (toml库会自动创建)
                if 'group' in config and isinstance(config['group'], dict):
                    actual_groups = config['group']
                    self.debug.append(f"找到顶层 'group' 字典，包含子分组: {list(actual_groups.keys())}")
                    total_groups = len(actual_groups)
                    idx = 0
                    # 遍历 'group' 字典下的子分组 ('errors', 'warnings', etc.)
                    for sub_group_key, group_data in actual_groups.items():
                        full_group_name = f"group.{sub_group_key}"  # 构建完整的组名，如 group.errors
                        display_name = sub_group_key  # 使用子分组名作为显示名称
                        # 检查 group_data 是否是字典
                        if not isinstance(group_data, dict):
                            self.debug.append(f"警告: '{full_group_name}' 在配置文件中的值不是字典，跳过.")
                            continue
                        color = generate_color(idx, total_groups)
                        self.group_colors[full_group_name] = color  # 使用完整组名作为键
                        cb = QCheckBox(display_name)
                        cb.setProperty("group_name", full_group_name)  # 存储完整组名
                        
                        # 默认所有关键词组不勾选
                        cb.setChecked(False)
                            
                        self.group_layout.addWidget(cb)
                        # 调试：检查该分组是否包含关键词
                        keyword_count = sum(1 for k, v in group_data.items() if isinstance(v, dict) and "key" in v)
                        self.debug.append(f"创建复选框: 显示='{display_name}', 属性名='{full_group_name}', 包含 {keyword_count} 个关键词")
                        idx += 1
                else:
                    self.debug.append("配置文件中未找到预期的顶层 'group' 字典结构.")
            except Exception as e:
                logging.error(f"更新分组复选框失败: {e}")
                self.debug.append(f"更新分组复选框失败: {str(e)}")
                QMessageBox.critical(self, "分组错误", f"更新分组复选框失败: {str(e)}")

    def add_directory(self) -> None:
        """添加日志目录并异步处理压缩文件解压"""
        d = QFileDialog.getExistingDirectory(self, "添加日志目录")
        if d and d not in self.history["sources"]:
            self.history["sources"].insert(0, d)
            self.src_list.insertItem(0, d)
            self.save_settings()
            self.debug.append(f"添加目录: {d}")

    def add_file(self) -> None:
        """添加单个日志文件。"""
        f, _ = QFileDialog.getOpenFileName(self, "添加日志文件", "", "所有文件 (*)")
        if f and f not in self.history["sources"]:
            self.history["sources"].insert(0, f)
            self.src_list.insertItem(0, f)
            self.save_settings()

    def add_archive(self) -> None:
        """添加压缩包并异步处理解压"""
        f, _ = QFileDialog.getOpenFileName(self, "添加压缩包", "", "所有支持格式 (*.rar *.zip *.7z *.tar *.tar.gz *.tgz);;RAR (*.rar);;ZIP (*.zip)")
        if f and f not in self.history["sources"]:
            self.history["sources"].insert(0, f)
            self.src_list.insertItem(0, f)
            self.save_settings()
            self.debug.append(f"添加压缩包: {f}")
            # 自动解压添加的压缩包，初始递归深度为0
            self.process_archives([f], current_recursion_depth=0)

    def process_archives(self, archive_paths: List[str], current_recursion_depth: int = 0) -> None:
        """处理压缩文件，解压到临时目录"""
        if not archive_paths:
            return

        # 创建临时解压目录 (如果不存在)
        extract_dir_base = os.path.join(self.temp_manager.temp_dir, "extracted_archives")
        os.makedirs(extract_dir_base, exist_ok=True)
        
        for archive_path in archive_paths:
            if not os.path.exists(archive_path):
                self.debug.append(f"错误: 文件 {archive_path} 不存在")
                continue
                
            self.debug.append(f"开始处理压缩包 (层级 {current_recursion_depth}): {os.path.basename(archive_path)}")
            
            # 为每个压缩包创建单独的目录，避免文件名冲突和覆盖
            # 使用更独特的名字，例如 压缩包名_随机后缀 或 压缩包名_层级_序号
            basename = os.path.basename(archive_path)
            name_without_ext, ext_only = os.path.splitext(basename)
            # 尝试创建一个基于 (原名 + 层级 + 时间戳/随机数) 的唯一目录名，防止多层解压时同名文件覆盖
            # 简化：暂时还是用原名，但如果发现问题需要改进
            unique_folder_name = f"{name_without_ext}_{current_recursion_depth}_{int(time.time() * 1000) % 10000}" # 增加唯一性
            specific_extract_dir = os.path.join(extract_dir_base, unique_folder_name)

            # 检查目标解压目录是否已存在（理论上不应该，因为我们加了唯一后缀）
            if os.path.exists(specific_extract_dir):
                # 如果意外存在，尝试添加一个更独特的后缀
                specific_extract_dir = os.path.join(extract_dir_base, f"{unique_folder_name}_{os.urandom(4).hex()}")
            
            os.makedirs(specific_extract_dir, exist_ok=True)
            
            try:
                ext = archive_path.lower()
                extracted_successfully = False
                
                if ext.endswith('.zip'):
                    with zipfile.ZipFile(archive_path, 'r') as zipf:
                        zipf.extractall(path=specific_extract_dir)
                    self.debug.append(f"已解压 ZIP 文件到: {specific_extract_dir}")
                    extracted_successfully = True
                        
                elif ext.endswith('.rar') and RARFILE_AVAILABLE:
                    if self.decompress_rar(archive_path, specific_extract_dir):
                        extracted_successfully = True
                    
                elif ext.endswith('.7z') and PY7ZR_AVAILABLE:
                    with py7zr.SevenZipFile(archive_path, mode='r') as z:
                        z.extractall(path=specific_extract_dir)
                    self.debug.append(f"已解压 7Z 文件到: {specific_extract_dir}")
                    extracted_successfully = True
                        
                elif ext.endswith(('.tar', '.tar.gz', '.tgz')):
                    mode = 'r:gz' if ext.endswith(('.tar.gz', '.tgz')) else 'r'
                    with tarfile.open(archive_path, mode) as tar:
                        tar.extractall(path=specific_extract_dir)
                    self.debug.append(f"已解压 TAR 文件到: {specific_extract_dir}")
                    extracted_successfully = True
                
                elif ext.endswith('.gz') and not ext.endswith('.tar.gz'): # 处理单独的 .gz 文件 (放在 tar.gz 之后确保特异性)
                    uncompressed_filename = os.path.splitext(basename)[0] 
                    uncompressed_filepath = os.path.join(specific_extract_dir, uncompressed_filename)
                    try:
                        with gzip.open(archive_path, 'rb') as f_in:
                            with open(uncompressed_filepath, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        self.debug.append(f"已解压单独 GZ 文件 ({basename}) 到: {uncompressed_filepath}")
                        extracted_successfully = True
                    except gzip.BadGzipFile:
                        self.debug.append(f"文件 {basename} 不是有效的 GZIP 文件或已损坏。跳过。")
                        # extracted_successfully 保持 False，会走到最后的continue
                    except Exception as e_gz:
                        self.debug.append(f"解压单独 GZ 文件 {basename} 失败: {str(e_gz)}。跳过。")
                        # extracted_successfully 保持 False
                
                elif PYUNPACK_AVAILABLE:
                    try:
                        Archive(archive_path).extractall(specific_extract_dir)
                        self.debug.append(f"使用 pyunpack 解压成功: {specific_extract_dir}")
                        extracted_successfully = True
                    except Exception as e_pyunpack:
                        self.debug.append(f"pyunpack 解压失败: {str(e_pyunpack)}")
                        # 不要在这里 raise，允许尝试其他方法或报告不支持
                
                if not extracted_successfully:
                    # 如果前面的特定解压方法都失败或不可用，且 PYUNPACK_AVAILABLE 为 False
                    # 或者 PYUNPACK_AVAILABLE 为 True 但也失败了 (这种情况上面已经处理)
                    if not PYUNPACK_AVAILABLE or (PYUNPACK_AVAILABLE and not extracted_successfully) : # 确认条件
                         self.debug.append(f"不支持的压缩格式或解压失败: {ext}")
                         QMessageBox.warning(self, "不支持的格式或解压失败", f"无法解压此文件: {os.path.basename(archive_path)}")
                         continue # 继续处理下一个压缩包
                
                # 如果解压成功，则添加解压后的文件和目录
                if extracted_successfully:
                    self._add_extracted_files(specific_extract_dir, current_recursion_depth)
                
            except Exception as e:
                self.debug.append(f"处理压缩包 {os.path.basename(archive_path)} 时发生严重错误: {str(e)}")
                QMessageBox.critical(self, "解压错误", f"处理压缩包 {os.path.basename(archive_path)} 失败:\n{str(e)}")
    
    def _add_extracted_files(self, extract_dir: str, current_recursion_depth: int) -> None:
        """添加解压后的文件到扫描列表，并支持递归解压"""
        MAX_RECURSION_DEPTH = 5 
        if current_recursion_depth >= MAX_RECURSION_DEPTH: # 注意是 >=
            self.debug.append(f"警告: 达到最大解压递归深度 ({MAX_RECURSION_DEPTH})，目录 {extract_dir} 内的嵌套压缩包将不再进一步解压。")
            # 即使达到最大深度，仍然需要添加这一层解压出来的非压缩文件
            for root, _, files in os.walk(extract_dir):
                for file_name in files:
                    full_path = os.path.join(root, file_name)
                    if not self._is_archive_file(full_path):
                        if full_path not in self.decompressed_files:
                            self.decompressed_files.append(full_path)
                            self.debug.append(f"(Max Depth) 添加解压文件: {full_path}")
            return

        archives_for_next_level_processing = []
        current_level_files_added = 0
        # 为每个嵌套压缩包生成唯一解压目录
        nested_extraction_dirs = []

        for root, _, files in os.walk(extract_dir):
            for file_name in files:
                full_path = os.path.join(root, file_name)
                
                if self._is_archive_file(full_path):
                    # 检查是否是已知的源文件，或者是否已经被处理过（避免无限循环）
                    # 一个简单的检查方法是：如果这个压缩文件路径和我们最初传入的archive_paths中的任何一个相同，
                    # 且current_recursion_depth > 0，说明可能是个循环。
                    # 但更简单的是依赖 MAX_RECURSION_DEPTH。
                    # 同时，如果一个压缩包解压出自身，会导致问题。
                    # 确保添加到 archives_for_next_level_processing 的路径是新的
                    is_new_archive_to_process = True
                    # 为嵌套压缩包创建唯一解压目录
                    nested_extract_dir = os.path.join(
                        os.path.dirname(full_path),
                        f"{os.path.splitext(os.path.basename(full_path))[0]}_nested_{current_recursion_depth+1}"
                    )
                    # 确保目录唯一性
                    unique_suffix = 1
                    while os.path.exists(nested_extract_dir):
                        nested_extract_dir = f"{nested_extract_dir}_{unique_suffix}"
                        unique_suffix += 1
                    nested_extraction_dirs.append(nested_extract_dir)
                    # 使用新生成的唯一目录进行解压
                    if PYUNPACK_AVAILABLE:
                        try:
                            Archive(full_path).extractall(nested_extract_dir)
                            self.debug.append(f"使用 pyunpack 解压嵌套包成功: {nested_extract_dir}")
                            archives_for_next_level_processing.append(full_path)
                            extracted_successfully = True
                        except Exception as e_pyunpack:
                            self.debug.append(f"pyunpack 解压嵌套包失败: {str(e_pyunpack)}")
                            extracted_successfully = False
                    else:
                        self.debug.append(f"发现嵌套压缩包 {full_path} 但无法处理（pyunpack不可用）")

                    if is_new_archive_to_process:
                        self.debug.append(f"发现嵌套压缩包: {full_path} (当前层级 {current_recursion_depth}, 下一级 {current_recursion_depth + 1})")
                        if full_path not in archives_for_next_level_processing: # 避免重复添加同一文件多次
                             archives_for_next_level_processing.append(full_path)
                else:
                    if full_path not in self.decompressed_files:
                        self.decompressed_files.append(full_path)
                        current_level_files_added += 1
                        # self.debug.append(f"添加解压文件 (层级 {current_recursion_depth}): {full_path}")

        if current_level_files_added > 0:
             self.debug.append(f"在层级 {current_recursion_depth} 添加了 {current_level_files_added} 个非压缩文件。")
        
        if archives_for_next_level_processing:
            self.debug.append(f"准备递归处理 {len(archives_for_next_level_processing)} 个在层级 {current_recursion_depth} 发现的嵌套压缩包...")
            self.process_archives(archives_for_next_level_processing, current_recursion_depth + 1)
        else:
            self.debug.append(f"在层级 {current_recursion_depth} 未发现新的嵌套压缩包需要处理。")

    def remove_sources(self) -> None:
        """移除选中的日志源。"""
        for it in self.src_list.selectedItems():
            p = it.text()
            self.history["sources"].remove(p)
            self.src_list.takeItem(self.src_list.row(it))
        self.save_settings()

    def clear_history(self) -> None:
        """清除历史记录。"""
        self.history["sources"].clear()
        self.src_list.clear()
        self.save_settings()

    def get_log_files(self) -> List[str]:
        """获取所有日志文件路径，包含解压后的文件"""
        paths = []
        if self.decompressed_files:
            paths.extend(self.decompressed_files)
            
        for src in self.history["sources"]:
            if os.path.isfile(src):
                if not self._is_archive_file(src):
                    paths.append(src)
            elif os.path.isdir(src):
                for root, _, files in os.walk(src):
                    for fn in files:
                        if not self._is_archive_file(fn):
                            full_path = os.path.join(root, fn)
                            paths.append(full_path)
        
        # 去重并排序
        paths = list(set(paths))
        paths.sort()
        
        return paths

    def add_custom_keyword(self) -> None:
        """添加自定义关键词"""
        txt = self.kw_text.text().strip()
        if not txt:
            return
        if txt not in self.history["keywords"]:
            self.history["keywords"].insert(0, txt)
            self.save_settings()
        parts = [p.strip() for p in txt.split('|') if p.strip()]
        tot = max(1, len(self.custom_keyword_checks) + len(parts))
        
        for p in parts:
            if p:
                idx = len(self.custom_keyword_checks)
                col = generate_color(idx, tot)
                # 获取注释文本
                annotation = self.kw_annotation.text().strip() or "[自定义]"
                kw = Keyword(p, annotation, self.chk_case.isChecked(), self.chk_word.isChecked(), self.chk_regex.isChecked(), col)
                cb = QCheckBox(p)
                cb.setProperty("keyword_obj", kw)
                cb.setChecked(False)  # 默认不勾选
                self.custom_keyword_checks.append(cb)
                self.custom_layout.addWidget(cb)

    def clear_selected_custom_keywords(self) -> None:
        """清除选中的自定义关键词"""
        for i in reversed(range(len(self.custom_keyword_checks))):
            cb = self.custom_keyword_checks[i]
            if cb.isChecked():
                self.custom_layout.removeWidget(cb)
                cb.deleteLater()
                self.custom_keyword_checks.pop(i)

    def select_all_custom_keywords(self) -> None:
        """选择所有自定义关键词"""
        for cb in self.custom_keyword_checks:
            cb.setChecked(True)

    def _on_scan_progress(self, message: str) -> None:
        """处理扫描进度更新"""
        if self.progress:
            self.progress.setVisible(True)
            # 提取百分比值
            if "%" in message:
                try:
                    percent_str = message.split("%")[0]
                    percent = int(percent_str)
                    self.progress.setValue(percent)
                except (ValueError, IndexError):
                    # 如果无法解析进度百分比，设置为忙碌状态
                    self.progress.setRange(0, 0)
            else:
                # 如果消息中没有百分比，设置为忙碌状态
                self.progress.setRange(0, 0)
                
        self.debug.append(message)
    
    def _on_scan_finished(self, path: str) -> None:
        """扫描完成处理"""
        # 更新按钮和进度条状态
        self.btn_analysis.setEnabled(True)
        self.btn_cancel.setVisible(False)
        self.progress.setVisible(False)
        
        # 立即处理UI事件，确保状态更新
        QApplication.processEvents()
        
        # 添加完成提示
        self.debug.append("分析已完成")
        
        # 打开结果文件
        if path and os.path.isfile(path):
            self._open_result_file(path)
            
        # 清理工作线程
        if self.worker:
            self.worker.deleteLater()
            self.worker = None

    def _on_scan_error(self, message: str) -> None:
        """处理扫描错误"""
        QMessageBox.critical(self, "扫描错误", message)
        if hasattr(self, 'error_log'):
            self.error_log.append(f"[错误] {message}")

    def cancel_analysis(self) -> None:
        """取消正在进行的分析任务。"""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(self.config_params["thread_timeout"])
            if self.worker:
                self.worker.deleteLater()
                self.worker = None
        
        # 无论如何确保界面状态更新
        self.btn_analysis.setEnabled(True)
        self.btn_cancel.setVisible(False)
        self.progress.setVisible(False)
        self.debug.append("分析已取消，可以从上次中断点继续")
        QApplication.processEvents()

    def load_settings(self) -> None:
        """加载设置"""
        try:
            if not os.path.isfile(self.settings_path):
                pass
            else:
                with open(self.settings_path, 'r', encoding='utf-8') as f:
                    obj = json.load(f)
                self.history = obj.get('history', self.history)
                self.config_path = obj.get('config_path', None)
                self.cfg_edit.setText(self.config_path or "")
                
                # 更新源文件列表
                self.src_list.clear()
                for s in self.history["sources"]:
                    self.src_list.addItem(s)
                
                # 不再需要更新关键词下拉框，因为已改为文本框
                # for kw in self.history["keywords"]:
                #     self.kw_text.addItem(kw)
                    
                self.update_group_checkboxes()
                    
        except Exception as e:
            logging.error(f"加载设置失败: {e}")
            QMessageBox.critical(self, "设置错误", f"加载设置失败: {str(e)}")

    def save_settings(self) -> None:
        """保存设置"""
        self.history["cores"] = self.spin_cores.value()
        self.history["max_results"] = self.config_params["max_results"]
        
        obj = {"config_path": self.config_path, "history": self.history}
        try:
            with open(self.settings_path, 'w', encoding='utf-8') as f:
                json.dump(obj, f, ensure_ascii=False)
        except Exception as e:
            logging.error(f"保存设置失败: {e}")
            QMessageBox.critical(self, "设置错误", f"保存设置失败: {str(e)}")
            
    def analyze_combined_keywords(self) -> None:
        """开始分析日志文件中的关键词。"""
        # 防抖：立即禁用按钮
        self.btn_analysis.setEnabled(False)
        should_re_enable_button = True
        try:
            if not self._validate_config():
                return
            if self._is_analysis_running():
                return
            self._clear_debug_logs()
            files = self.get_log_files()
            if not files:
                QMessageBox.warning(self, "提示", "无日志文件可分析")
                return
            all_kws = self._collect_keywords()
            if not all_kws:
                QMessageBox.warning(self, "提示", "请勾选至少一个自定义关键词或分组关键词")
                self.debug.append("DEBUG: Entering 'if not all_kws' block.")
                self.debug.append("DEBUG: After QMessageBox.warning, before return.")
                return
            should_re_enable_button = False
            self._display_keyword_summary(all_kws)
            if self._cancel_existing_worker():
                should_re_enable_button = True
                return
            self._update_ui_for_analysis()
            out_path = self._determine_output_path()
            max_workers = self.spin_cores.value()
            if hasattr(self, 'use_process_pool'):
                self.config_params["use_process_pool"] = self.use_process_pool.isChecked()
            self._start_scan_worker(files, all_kws, out_path, max_workers)
        finally:
            if should_re_enable_button:
                self.btn_analysis.setEnabled(True)
                self.debug.append("DEBUG: Analysis button re-enabled due to early return.")

    def _validate_config(self) -> bool:
        """验证配置文件是否存在且有效。"""
        if not self.config_path or not os.path.isfile(self.config_path):
            QMessageBox.warning(self, "提示", "请选择有效配置文件")
            return False
        return True

    def _is_analysis_running(self) -> bool:
        """检查是否已有分析正在进行。"""
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "提示", "已有分析正在进行，请稍候")
            return True
        return False

    def _clear_debug_logs(self) -> None:
        """清除调试日志。"""
        self.debug.clear()
        if hasattr(self, 'error_log'):
            self.error_log.clear()

    def _collect_keywords(self) -> List[Dict]:
        """收集自定义和分组关键词。"""
        custom_kws = self._collect_custom_keywords()
        group_kws = self._collect_group_keywords()
        all_kws = custom_kws + group_kws
        self.debug.append(f"DEBUG: Just before check, len(all_kws) = {len(all_kws)}")
        return all_kws

    def _collect_custom_keywords(self) -> List[Dict]:
        """收集自定义关键词。"""
        custom_kws = []
        for cb in self.custom_keyword_checks:
            if cb.property("keyword_obj") and cb.isChecked():
                kw = cb.property("keyword_obj")
                custom_kws.append({
                    "raw": kw.raw,
                    "annotation": kw.annotation,
                    "match_case": kw.match_case,
                    "whole_word": kw.whole_word,
                    "use_regex": kw.use_regex,
                    "color": kw.color
                })
        return custom_kws

    def _collect_group_keywords(self) -> List[Dict]:
        """收集分组关键词。"""
        group_kws = []
        for i in range(self.group_layout.count()):
            cb = self.group_layout.itemAt(i).widget()
            if isinstance(cb, QCheckBox) and cb.isChecked():
                group_name = cb.property("group_name")
                if not group_name:
                    self.debug.append(f"警告: 复选框 {cb.text()} 没有关联的 group_name")
                    continue
                self.debug.append(f"处理选中的分组: {group_name}")
                group_kws.extend(self._load_group_keywords(group_name))
        return group_kws

    def _load_group_keywords(self, group_name: str) -> List[Dict]:
        """从配置文件加载指定分组的关键词。"""
        try:
            config_data = toml.load(self.config_path)
            self.debug.append(f"  配置文件加载成功.")
            top_group_dict = config_data.get('group', None)
            if top_group_dict is None or not isinstance(top_group_dict, dict):
                self.debug.append(f"  错误: 配置文件缺少顶层 'group' 字典结构.")
                return []
            subgroup_key = group_name.split('.')[-1] if '.' in group_name else group_name
            grp = top_group_dict.get(subgroup_key, None)
            if grp is None:
                self.debug.append(f"  警告: 在 'group' 字典下未找到子分组 '{subgroup_key}' (来自 {group_name})")
                return []
            if not isinstance(grp, dict):
                self.debug.append(f"  警告: 子分组 '{subgroup_key}' 的值不是字典 (类型: {type(grp)}). 跳过.")
                return []
            self.debug.append(f"  成功获取分组 '{subgroup_key}' (来自 {group_name}) 的数据: {grp}")
            mc = grp.get("match_case", False)
            ww = grp.get("whole_word", False)
            uz = grp.get("use_regex", False)
            color = self.group_colors.get(group_name, "#ffff99")
            self.debug.append(f"  分组 '{subgroup_key}' 设置: match_case={mc}, whole_word={ww}, use_regex={uz}, color={color}")
            entries_count = 0
            group_kws = []
            self.debug.append(f"  开始遍历分组 '{subgroup_key}' (来自 {group_name}) 中的条目...")
            for k, v in grp.items():
                if isinstance(v, dict) and "key" in v:
                    keyword_text = v["key"]
                    annotation_text = v.get("annotation", "")
                    self.debug.append(f"    处理条目: key='{k}', keyword_text='{keyword_text}', annotation='{annotation_text}'")
                    entries_count += 1
                    group_kws.append({
                        "raw": keyword_text,
                        "annotation": annotation_text,
                        "match_case": mc,
                        "whole_word": ww,
                        "use_regex": uz,
                        "color": color
                    })
                elif k not in ["match_case", "whole_word", "use_regex"]:
                    self.debug.append(f"    跳过条目 '{k}' (不是关键词定义，也不是已知配置项: value='{v}')")
            self.debug.append(f"  从分组 '{subgroup_key}' (来自 {group_name}) 中成功添加了 {entries_count} 个关键词到 group_kws列表")
            return group_kws
        except Exception as e:
            self.debug.append(f"  加载或处理分组 '{group_name}' 时发生错误: {str(e)}")
            logging.error(f"加载分组 '{group_name}' 失败: {e}")
            return []

    def _display_keyword_summary(self, all_kws: List[Dict]) -> None:
        """显示关键词摘要。"""
        raw_list = [kw['raw'] for kw in all_kws]
        self.debug.append(f"共选择了 {len(all_kws)} 个关键词:")
        for kw in all_kws[:10]:
            self.debug.append(f"- {kw['raw']}")
        if len(all_kws) > 10:
            self.debug.append(f"... 以及 {len(all_kws) - 10} 个其他关键词")

    def _cancel_existing_worker(self) -> bool:
        """取消现有的工作线程。"""
        if self.worker and self.worker.isRunning():
            self.cancel_analysis()
            return True
        return False

    def _update_ui_for_analysis(self) -> None:
        """更新UI以进行分析。"""
        self.btn_cancel.setVisible(True)
        self.btn_preview.setVisible(True)
        self.progress.setVisible(True)
        QApplication.processEvents()

    def _determine_output_path(self) -> str:
        """确定输出路径。"""
        out_dir = CONFIG_DEFAULTS["output_dir"]
        for s in self.history["sources"]:
            if os.path.isdir(s):
                out_dir = s
                break
        if not out_dir and self.history["sources"]:
            out_dir = os.path.dirname(self.history["sources"][0])
        return os.path.join(out_dir, CONFIG_DEFAULTS["output_filename"])

    def _start_scan_worker(self, files: List[str], all_kws: List[Dict], out_path: str, max_workers: int) -> None:
        """启动扫描工作线程。"""
        self.worker = ScanWorker(
            file_paths=files,
            keywords=all_kws,
            out_path=out_path,
            max_workers=max_workers,
            config_params=self.config_params,
            temp_manager=self.temp_manager,
            parent=self
        )
        self.worker.error.connect(self._on_scan_error)
        self.worker.warning.connect(self._on_scan_warning)
        self.worker.progress.connect(self._on_scan_progress)
        self.worker.debug.connect(self._on_scan_debug_message)
        self.worker.finished.connect(self._on_scan_finished)
        self.worker.start()
        self.debug.append("工作线程已启动")

    def decompress_selected(self) -> None:
        """解压选中的压缩文件。"""
        selected_items = self.src_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "提示", "请选择要解压的压缩文件")
            return
        
        archive_paths = [item.text() for item in selected_items if self._is_archive_file(item.text())]
        if not archive_paths:
            QMessageBox.warning(self, "提示", "所选文件中没有支持的压缩文件格式")
            return
        
        self.debug.append(f"开始解压 {len(archive_paths)} 个选中的压缩文件...")
        self.process_archives(archive_paths, current_recursion_depth=0)
        self.debug.append("解压操作已完成")

    def closeEvent(self, event: 'QCloseEvent') -> None:
        """处理窗口关闭事件，优雅停线程并回收内存。"""
        # 停止扫描工作线程
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(self.config_params["thread_timeout"])
            
        # 停止解压工作线程
        if self.decompress_worker and self.decompress_worker.isRunning():
            self.decompress_worker.stop()
            self.decompress_worker.wait(self.config_params["thread_timeout"])
            
        # 清理临时文件
        self.cleanup_temp_files()
        
        # 退出应用
        QCoreApplication.quit()
        event.accept()

    def decompress_rar(self, rar_path: str, dest_dir: str) -> bool:
        """解压 .rar 文件到指定目录"""
        if not RARFILE_AVAILABLE:
            logging.error("rarfile 库不可用，无法解压 .rar 文件")
            self.debug.append("rarfile 库不可用，无法解压 .rar 文件")
            return False
        try:
            with rarfile.RarFile(rar_path) as rf:
                rf.extractall(dest_dir)
            logging.info(f"成功解压 {rar_path} 到 {dest_dir}")
            self.debug.append(f"成功解压 {os.path.basename(rar_path)} 到 {dest_dir}")
            return True
        except Exception as e:
            logging.error(f"解压 {rar_path} 失败: {e}")
            self.debug.append(f"解压 {os.path.basename(rar_path)} 失败: {str(e)}")
            return False

    def preview_results(self) -> None:
        """预览扫描结果。"""
        if self.worker and self.worker.isRunning():
            self.debug.append("正在获取预览结果...")
            # 这里可以添加逻辑以从工作线程获取部分结果进行预览
            # 例如：self.worker.request_preview()
            QMessageBox.information(self, "预览结果", "此功能暂未完全实现。当前仅显示此消息作为占位符。")
        else:
            QMessageBox.warning(self, "预览结果", "没有正在进行的分析，无法预览结果。")

    def _open_result_file(self, path: str) -> None:
        """尝试打开结果文件，使用多种方法确保兼容性"""
        self.debug.append(f"尝试打开结果文件: {path}")
        try:
            import platform
            import webbrowser
            import subprocess
            
            abs_path = os.path.abspath(path)
            self.debug.append(f"绝对路径: {abs_path}")
            
            # 1. 首先尝试使用系统默认程序打开
            system = platform.system()
            success = False
            
            if system == "Darwin":  # macOS
                self.debug.append("在macOS上使用open命令打开")
                try:
                    subprocess.run(['open', abs_path])
                    success = True
                except Exception as e:
                    self.debug.append(f"open命令失败: {str(e)}")
            
            elif system == "Windows":
                self.debug.append("在Windows上使用start命令打开")
                try:
                    os.startfile(abs_path)
                    success = True
                except Exception as e:
                    self.debug.append(f"os.startfile失败: {str(e)}")
            
            elif system == "Linux":
                self.debug.append("在Linux上使用xdg-open命令打开")
                try:
                    subprocess.run(['xdg-open', abs_path])
                    success = True
                except Exception as e:
                    self.debug.append(f"xdg-open命令失败: {str(e)}")
            
            # 2. 如果系统方法失败，尝试使用webbrowser模块
            if not success:
                self.debug.append("使用webbrowser模块打开")
                file_url = f"file://{abs_path}"
                if not webbrowser.open(file_url):
                    self.debug.append("webbrowser.open失败，尝试使用get方法")
                    try:
                        browser = webbrowser.get()
                        browser.open(file_url)
                        success = True
                    except Exception as e:
                        self.debug.append(f"webbrowser.get()失败: {str(e)}")
            
            # 3. 最后如果依然失败，提示用户手动打开
            if not success:
                QMessageBox.information(
                    self,
                    "结果已生成",
                    f"无法自动打开浏览器查看结果。\n请手动打开文件：\n{abs_path}"
                )
                
        except Exception as e:
            self.debug.append(f"打开结果文件时出错: {str(e)}")
            QMessageBox.information(
                self,
                "结果已生成",
                f"结果文件已生成，但无法自动打开。\n请手动查看：\n{path}"
            )

    def _handle_scan_mode_change(self, value: str) -> None:
        """处理扫描模式变更"""
        # 根据扫描模式调整其他参数
        if value == "fast":
            if hasattr(self, 'prefilter_check'):
                self.prefilter_check.setChecked(True)
            if hasattr(self, 'bitmap_filter_check'):
                self.bitmap_filter_check.setChecked(True)
        elif value == "accurate":
            if hasattr(self, 'prefilter_check'):
                self.prefilter_check.setChecked(False)
            if hasattr(self, 'bitmap_filter_check'):
                self.bitmap_filter_check.setChecked(False)
            
        self.debug.append(f"扫描模式已设置为: {value}")

    def _handle_boolean_param_change(self, key: str, value: bool) -> None:
        """处理布尔值参数变更"""
        if key == "prefilter_enabled":
            self.debug.append(f"{'启用' if value else '禁用'}关键词预过滤")
        elif key == "bitmap_filter_enabled":
            self.debug.append(f"{'启用' if value else '禁用'}位图过滤")

    def _on_scan_warning(self, message: str) -> None:
        QMessageBox.warning(self, "扫描警告", message)
        logging.warning(message)

    def _on_scan_debug_message(self, message: str) -> None:
        """处理来自ScanWorker的调试消息"""
        print(f"[ScanWorker DEBUG]:\n{message}") # Print to console
        logging.debug(f"[ScanWorker DEBUG]: {message}") # Also log to file

    def _reset_ui_for_new_scan(self) -> None:
        """重置UI元素以便进行新的扫描"""
        # ... existing code ...

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LogHighlighter()
    window.show()
    sys.exit(app.exec_())
