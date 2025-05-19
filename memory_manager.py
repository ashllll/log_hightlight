#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
内存管理模块
负责监控和管理程序内存使用情况。
"""

import os
import gc
import platform
import logging
from typing import Optional, Tuple

from PyQt5.QtWidgets import QWidget, QProgressBar, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import QTimer, pyqtSignal, QObject

# 尝试导入psutil库，用于更准确的内存监控
try:
    import psutil
    PSUTIL_AVAILABLE = True
    logging.info("psutil 库可用，将使用更准确的内存监控")
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("无法导入 psutil 库，将使用基本内存监控")

class MemoryMonitor(QObject):
    """内存监控器，用于跟踪和报告内存使用情况"""
    
    memory_usage = pyqtSignal(float, float)  # 当前使用量(MB), 总内存(MB)
    memory_warning = pyqtSignal(str)  # 内存警告信息
    
    def __init__(self, parent: Optional[QObject] = None, 
                 warning_threshold: float = 80.0, 
                 critical_threshold: float = 90.0,
                 update_interval: int = 2000):
        """
        初始化内存监控器
        
        Args:
            parent: 父QObject
            warning_threshold: 内存使用率警告阈值(%)
            critical_threshold: 内存使用率严重警告阈值(%)
            update_interval: 更新间隔(毫秒)
        """
        super().__init__(parent)
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.update_interval = update_interval
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_memory)
        self.last_warning_level = 0  # 0: 无警告, 1: 警告, 2: 严重警告
        
        # 初始化内存基准值
        self.initial_memory = self.get_memory_usage()[0]
        logging.info(f"初始内存使用: {self.initial_memory:.1f} MB")
    
    def start_monitoring(self) -> None:
        """开始内存监控"""
        self.timer.start(self.update_interval)
        logging.info(f"内存监控已启动，更新间隔: {self.update_interval}ms")
    
    def stop_monitoring(self) -> None:
        """停止内存监控"""
        self.timer.stop()
        logging.info("内存监控已停止")
    
    def get_memory_usage(self) -> Tuple[float, float]:
        """
        获取当前内存使用情况
        
        Returns:
            当前使用量(MB), 总内存(MB)的元组
        """
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                used_memory = memory_info.rss / (1024 * 1024)  # 转换为MB
                
                # 获取系统总内存
                system_memory = psutil.virtual_memory().total / (1024 * 1024)  # 转换为MB
                
                return used_memory, system_memory
            except Exception as e:
                logging.error(f"获取内存信息失败: {e}")
        
        # 如果psutil不可用或发生错误，使用基本方法
        try:
            import resource
            used_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # 转换为MB
            # 在某些系统上，ru_maxrss已经是KB单位，需要除以1024转换为MB
            if platform.system() == "Darwin":  # macOS返回的是字节数
                used_memory /= 1024
            
            # 无法准确获取系统总内存，返回估计值
            system_memory = 8 * 1024  # 假设8GB
            
            return used_memory, system_memory
        except Exception:
            # 最后的后备方案
            return 0.0, 8 * 1024  # 无法获取，返回0和假设的8GB
    
    def check_memory(self) -> None:
        """检查内存使用情况并发出信号"""
        used_memory, total_memory = self.get_memory_usage()
        
        # 发送内存使用信号
        self.memory_usage.emit(used_memory, total_memory)
        
        # 计算使用率
        usage_percent = (used_memory / total_memory) * 100 if total_memory > 0 else 0
        
        # 检查是否需要发出警告
        if usage_percent > self.critical_threshold and self.last_warning_level < 2:
            self.memory_warning.emit(f"内存使用率严重警告: {usage_percent:.1f}%")
            self.last_warning_level = 2
            # 尝试释放一些内存
            self.free_memory()
        elif usage_percent > self.warning_threshold and self.last_warning_level < 1:
            self.memory_warning.emit(f"内存使用率警告: {usage_percent:.1f}%")
            self.last_warning_level = 1
        elif usage_percent < self.warning_threshold and self.last_warning_level > 0:
            self.last_warning_level = 0
    
    def free_memory(self) -> None:
        """尝试释放一些内存"""
        logging.info("尝试释放内存...")
        gc.collect()
        
        # 在Windows上，有时需要多次调用才能释放更多内存
        if platform.system() == "Windows":
            gc.collect()
        
        # 记录释放后的内存使用
        used_memory, _ = self.get_memory_usage()
        logging.info(f"内存释放后使用: {used_memory:.1f} MB")

class MemoryStatusWidget(QWidget):
    """内存状态显示小部件"""
    
    def __init__(self, parent: Optional[QWidget] = None):
        """初始化内存状态小部件"""
        super().__init__(parent)
        self.monitor = MemoryMonitor(self)
        self.monitor.memory_usage.connect(self.update_display)
        self.monitor.memory_warning.connect(self.show_warning)
        self.init_ui()
        self.monitor.start_monitoring()
    
    def init_ui(self) -> None:
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 内存使用标签
        self.usage_label = QLabel("内存使用: 0 MB / 0 MB")
        layout.addWidget(self.usage_label)
        
        # 内存使用进度条
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("使用率:"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        hbox.addWidget(self.progress_bar, 1)
        layout.addLayout(hbox)
        
        # 警告标签
        self.warning_label = QLabel("")
        self.warning_label.setStyleSheet("color: red;")
        layout.addWidget(self.warning_label)
        
        layout.addStretch()
    
    def update_display(self, used_memory: float, total_memory: float) -> None:
        """更新显示的内存使用情况"""
        self.usage_label.setText(f"内存使用: {used_memory:.1f} MB / {total_memory:.1f} MB")
        
        # 更新进度条
        usage_percent = (used_memory / total_memory) * 100 if total_memory > 0 else 0
        self.progress_bar.setValue(int(usage_percent))
        
        # 根据使用率设置进度条颜色
        if usage_percent > 90:
            self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: red; }")
        elif usage_percent > 70:
            self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: orange; }")
        else:
            self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: green; }")
    
    def show_warning(self, message: str) -> None:
        """显示内存警告信息"""
        self.warning_label.setText(message)
    
    def closeEvent(self, event) -> None:
        """关闭事件处理"""
        self.monitor.stop_monitoring()
        super().closeEvent(event)
