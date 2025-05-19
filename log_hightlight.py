#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import json
import html
import toml
import webbrowser
import colorsys
import logging
import gc
import signal
import tempfile
import datetime
import atexit
import shutil
import gzip
import time
import re  # 直接使用Python内置re模块
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Generator, TextIO, Set, Any
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton, QFileDialog,
    QHBoxLayout, QVBoxLayout, QMessageBox, QGroupBox, QComboBox, QCheckBox,
    QTextEdit, QScrollArea, QProgressBar, QListWidget, QSplitter, QSpinBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView, QFrame
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QCoreApplication
from PyQt5.QtGui import QTextCharFormat, QColor, QFont, QIcon, QPixmap

# 导入自定义模块
import utils  # 导入utils模块用于统一的时间戳解析和其他功能

# 移除google-re2相关代码，直接使用Python内置re模块
RE_MODULE = re
logging.info("使用Python内置re模块进行正则表达式匹配")

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

# 尝试导入 pyahocorasick 库的代码已移除 - 我们将使用自己实现的优化技术
AHOCORASICK_AVAILABLE = False
logging.info("不使用Aho-Corasick算法，改用优化的分级正则表达式和位图过滤")

# 尝试导入 mmap 库，用于处理超大文件
try:
    import mmap
    MMAP_AVAILABLE = True
    logging.info("mmap 可用，将使用内存映射优化超大文件处理")
except ImportError:
    MMAP_AVAILABLE = False
    logging.warning("无法导入 mmap，超大文件将使用标准方式处理")

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', 
                    handlers=[logging.FileHandler("log_highlighter.log", encoding='utf-8'), logging.StreamHandler()])

# 常量配置（默认值）
CONFIG_DEFAULTS = {
    "output_dir": os.getcwd(),
    "output_filename": "highlight_results",
    "html_style": {
        "font_family": "Consolas",
        "header": "<h2>分析结果（按时间升序）</h2><hr>",
    },
    "batch_update_size": 10,  # 批量更新 UI 的文件数量
    "max_results": 10000,  # 最大结果数量限制
    "chunk_size": 1024 * 1024,  # 文件分块读取大小（1MB）
    "thread_timeout": 5000,  # 线程等待超时时间（毫秒）
    "max_file_size": 1024 * 1024 * 1024,  # 最大文件大小限制（1GB）
    "time_range_hours": 1,  # 每个 HTML 文件包含的时间范围（小时）
    "max_output_files": 100,  # 最大输出文件数量限制
    "scan_mode": "auto",  # 扫描模式: auto, fast, accurate, balanced
    "large_file_threshold": 100 * 1024 * 1024,  # 大文件阈值(100MB)
    "huge_file_threshold": 500 * 1024 * 1024,  # 超大文件阈值(500MB)
    "prefilter_enabled": True,  # 启用预过滤
    "bitmap_filter_enabled": True,  # 启用位图过滤
}

# === 添加临时文件管理类 ===


class TempFileManager:
    """临时文件管理器，负责创建、跟踪和清理临时文件"""

    def __init__(self):
        self.temp_files = {}  # 路径 -> {"file_obj": 文件对象, "metadata": 元数据}
        self.temp_dir = os.path.join(
    tempfile.gettempdir(),
     "log_highlighter_temp")
        os.makedirs(self.temp_dir, exist_ok=True)

    def create_temp_file(self, prefix: str, suffix: str,
                         **metadata) -> Tuple[str, TextIO]:
        """创建临时文件并返回路径和文件对象"""
        fd, path = tempfile.mkstemp(
    prefix=prefix, suffix=suffix, dir=self.temp_dir, text=True)
        file_obj = os.fdopen(fd, 'w', encoding='utf-8')
        self.temp_files[path] = {"file_obj": file_obj, "metadata": metadata}
        return path, file_obj

    def close_file(self, path: str) -> None:
        """安全关闭临时文件"""
        if path in self.temp_files and "file_obj" in self.temp_files[path]:
            try:
                if not self.temp_files[path]["file_obj"].closed:
                    self.temp_files[path]["file_obj"].close()
            except Exception as e:
                logging.error(f"关闭临时文件 {path} 失败: {e}")

    def remove_file(self, path: str) -> bool:
        """删除临时文件并从跟踪中移除"""
        self.close_file(path)
        try:
            if os.path.exists(path):
                os.remove(path)
            if path in self.temp_files:
                del self.temp_files[path]
            return True
        except Exception as e:
            logging.error(f"删除临时文件 {path} 失败: {e}")
            return False

    def cleanup_all(self) -> None:
        """清理所有临时文件"""
        for path in list(self.temp_files.keys()):
            self.remove_file(path)

        # 清理临时目录中的所有剩余文件
        try:
            for filename in os.listdir(self.temp_dir):
                try:
                    os.remove(os.path.join(self.temp_dir, filename))
                except Exception as e:
                    logging.error(f"清理临时文件 {filename} 失败: {e}")
        except Exception as e:
            logging.error(f"清理临时目录失败: {e}")

# === 增加进度监控类 ===


class ProgressMonitor:
    """跟踪扫描任务的进度和错误"""

    def __init__(self, worker=None):
        """
        初始化进度监控器

        Args:
            worker: 关联的工作线程，用于发送进度信号
        """
        self.worker = worker
        self.start_time = time.time()
        self.total_files = 0
        self.processed_files = 0
        self.failed_files = 0
        self.errors = {}
        self.warnings = []
        self.last_updated = 0

    def set_total(self, total: int) -> None:
        """设置待处理文件总数"""
        self.total_files = total

    def update(self, processed: int, current_file: str = "") -> None:
        """
        更新处理进度

        Args:
            processed: 已处理文件数
            current_file: 当前正在处理的文件
        """
        self.processed_files = processed

        # 限制更新频率(最多每0.5秒一次)
        current_time = time.time()
        if current_time - self.last_updated < 0.5:
            return

        self.last_updated = current_time

        # 如果有worker，发送进度信号
        if self.worker and hasattr(self.worker, 'progress'):
            if self.total_files > 0:
                percent = min(int(processed / self.total_files * 100), 100)
                self.worker.progress.emit(
                    f"{percent}% ({processed}/{self.total_files}) - {current_file}")
            else:
                self.worker.progress.emit(
    f"已处理 {processed} 个文件 - {current_file}")

    def record_error(self, error_type: str, message: str) -> None:
        """
        记录错误

        Args:
            error_type: 错误类型
            message: 错误消息
        """
        if error_type not in self.errors:
            self.errors[error_type] = []
        self.errors[error_type].append(message)
        self.failed_files += 1

    def record_warning(self, message: str) -> None:
        """记录警告"""
        self.warnings.append(message)

    def get_summary(self) -> str:
        """获取进度摘要"""
        duration = time.time() - self.start_time
        summary = [
            f"扫描完成！",
            f"总耗时: {duration:.2f}秒",
            f"总文件数: {self.total_files}",
            f"成功处理: {self.processed_files - self.failed_files}",
            f"处理失败: {self.failed_files}"
        ]

        # 添加错误汇总
        if self.errors:
            summary.append("\n错误统计:")
            for error_type, messages in self.errors.items():
                summary.append(f"  {error_type}: {len(messages)}个")

        # 添加警告汇总
        if self.warnings:
            summary.append("\n警告:")
            for warning in self.warnings[:5]:  # 只显示前5个警告
                summary.append(f"  {warning}")
            if len(self.warnings) > 5:
                summary.append(f"  ... 以及 {len(self.warnings) - 5} 个其他警告")

        return "\n".join(summary)

    def get_error_details(self) -> Dict[str, List[str]]:
        """获取错误详情"""
        return self.errors

    def get_progress_percentage(self) -> float:
        """获取进度百分比"""
        if self.total_files > 0:
            return min(self.processed_files / self.total_files * 100, 100.0)
        return 0.0

# === 增加断点续传管理类 ===


class CheckpointManager:
    """管理扫描任务的检查点，用于恢复中断的扫描任务"""

    def __init__(self, checkpoint_dir: Optional[str] = None):
        """
        初始化检查点管理器

        Args:
            checkpoint_dir: 检查点存储目录，默认为临时目录
        """
        self.checkpoint_dir = checkpoint_dir or os.path.join(
            tempfile.gettempdir(), "log_highlighter_checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def get_checkpoint_path(self, task_id: str) -> str:
        """获取检查点文件路径"""
        safe_id = self._sanitize_id(task_id)
        return os.path.join(self.checkpoint_dir, f"{safe_id}.json")

    def _sanitize_id(self, task_id: str) -> str:
        """清理任务ID，确保文件名安全"""
        return re.sub(r'[^\w\-_]', '_', task_id)

    def save_checkpoint(self, task_id: str, data: Dict) -> bool:
        """
        保存检查点数据

        Args:
            task_id: 任务ID
            data: 检查点数据字典

        Returns:
            保存成功返回True，否则返回False
        """
        try:
            checkpoint_path = self.get_checkpoint_path(task_id)
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
            return True
        except Exception as e:
            logging.error(f"保存检查点失败: {e}")
            return False

    def load_checkpoint(self, task_id: str) -> Optional[Dict]:
        """
        加载检查点数据

        Args:
            task_id: 任务ID

        Returns:
            检查点数据字典，如果检查点不存在或加载失败则返回None
        """
        checkpoint_path = self.get_checkpoint_path(task_id)
        if not os.path.exists(checkpoint_path):
            return None

        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"加载检查点失败: {e}")
            return None

    def has_checkpoint(self, task_id: str) -> bool:
        """检查任务是否有检查点"""
        checkpoint_path = self.get_checkpoint_path(task_id)
        return os.path.exists(checkpoint_path)

    def remove_checkpoint(self, task_id: str) -> bool:
        """删除检查点"""
        checkpoint_path = self.get_checkpoint_path(task_id)
        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
                return True
            except Exception as e:
                logging.error(f"删除检查点失败: {e}")
        return False

    def list_checkpoints(self) -> List[str]:
        """列出所有可用的检查点ID"""
        checkpoints = []
        for filename in os.listdir(self.checkpoint_dir):
            if filename.endswith('.json'):
                checkpoints.append(os.path.splitext(filename)[0])
        return checkpoints

    def cleanup_old_checkpoints(self, max_age_hours: int = 24) -> int:
        """
        清理旧检查点

        Args:
            max_age_hours: 最大保留时间(小时)

        Returns:
            已清理的检查点数量
        """
        if not os.path.exists(self.checkpoint_dir):
            return 0

        cleaned = 0
        cutoff_time = time.time() - (max_age_hours * 3600)

        for filename in os.listdir(self.checkpoint_dir):
            if not filename.endswith('.json'):
                continue

            filepath = os.path.join(self.checkpoint_dir, filename)
            try:
                if os.path.getmtime(filepath) < cutoff_time:
                    os.remove(filepath)
                    cleaned += 1
            except Exception as e:
                logging.error(f"清理检查点 {filename} 失败: {e}")

        return cleaned

# === 强化内存监控类 ===


class MemoryMonitor(QThread):
    """监控系统内存使用的线程类"""

    warning = pyqtSignal(str)
    memory_status = pyqtSignal(float, bool, bool)  # 使用率, 是否警告, 是否危险

    def __init__(
    self,
    parent=None,
    warning_threshold=75,
    critical_threshold=90,
     check_interval=2.0):
        """
        初始化内存监控器

        Args:
            parent: 父组件
            warning_threshold: 内存使用率警告阈值(%)
            critical_threshold: 内存使用率临界阈值(%)
            check_interval: 检查间隔(秒)
        """
        super().__init__(parent)
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.check_interval = check_interval
        self.running = False
        self._usage = 0.0
        self._is_warning = False
        self._is_critical = False
        self._peak_usage = 0.0
        self._usage_history = []

        # 尝试导入psutil
        try:
            import psutil
            self.psutil_available = True
        except ImportError:
            self.psutil_available = False
            logging.warning("psutil 库不可用，内存监控功能受限")

    def run(self):
        """监控线程的主循环"""
        self.running = True
        while self.running:
            try:
                self._check_memory()

                # 发送内存状态信号
                self.memory_status.emit(
    self._usage, self._is_warning, self._is_critical)

                # 睡眠指定时间
                for _ in range(int(self.check_interval * 2)):
                    if not self.running:
                        break
                    time.sleep(0.5)  # 分割睡眠以便更快响应停止请求

            except Exception as e:
                logging.error(f"内存监控错误: {e}")
                time.sleep(5)  # 出错时等待更长时间

    def stop(self):
        """停止监控线程"""
        self.running = False
        self.wait(1000)  # 等待线程终止，最多1秒

    def _check_memory(self):
        """检查当前内存使用情况"""
        if not self.psutil_available:
            # 没有psutil时使用简单的方法估计内存
            try:
                import gc
                gc.collect()  # 触发垃圾回收
                self._usage = 50.0  # 无法准确获取，返回保守估计
                self._is_warning = False
                self._is_critical = False
            except:
                pass
            return

        # 使用psutil获取详细内存信息
        import psutil
        try:
            # 获取内存使用率
            memory = psutil.virtual_memory()
            self._usage = memory.percent

            # 更新历史记录
            self._usage_history.append(self._usage)
            if len(self._usage_history) > 30:  # 保留最近30个数据点
                self._usage_history.pop(0)

            # 更新峰值
            self._peak_usage = max(self._peak_usage, self._usage)

            # 更新状态
            old_warning = self._is_warning
            old_critical = self._is_critical

            self._is_warning = self._usage >= self.warning_threshold
            self._is_critical = self._usage >= self.critical_threshold

            # 状态变化时发出警告
            if not old_warning and self._is_warning:
                msg = f"内存使用率达到警告水平: {self._usage:.1f}%"
                logging.warning(msg)
                self.warning.emit(msg)

            if not old_critical and self._is_critical:
                msg = f"内存使用率达到危险水平: {self._usage:.1f}%，将减少并发任务"
                logging.warning(msg)
                self.warning.emit(msg)

                # 触发内存释放
                self._force_memory_cleanup()

        except Exception as e:
            logging.error(f"获取内存使用率失败: {e}")

    def _force_memory_cleanup(self):
        """强制内存清理"""
        try:
            # 触发Python垃圾回收
            gc.collect()

            # 如果psutil可用，尝试更多清理
            if self.psutil_available:
                import psutil
                p = psutil.Process()

                # 如果支持内存紧缩，则执行
                if hasattr(p, 'memory_maps'):
                    logging.info("执行内存紧缩")
        except Exception as e:
            logging.error(f"内存清理失败: {e}")

    def get_usage(self) -> float:
        """获取当前内存使用率"""
        return self._usage

    def get_peak_usage(self) -> float:
        """获取峰值内存使用率"""
        return self._peak_usage

    def get_usage_trend(self) -> List[float]:
        """获取内存使用趋势"""
        return self._usage_history

    def is_warning(self) -> bool:
        """内存使用率是否达到警告阈值"""
        return self._is_warning

    def is_critical(self) -> bool:
        """内存使用率是否达到危险阈值"""
        return self._is_critical

    def suggest_worker_count(self, requested_count: int) -> int:
        """根据内存使用情况建议工作线程数量"""
        if self._is_critical:
            # 内存紧张时减少到1/4
            return max(1, requested_count // 4)
        elif self._is_warning:
            # 内存接近警告线时减少到1/2
            return max(1, requested_count // 2)
        else:
            return requested_count


def generate_color(index: int, total: int) -> str:
    """
    根据索引和总数生成颜色值。
    
    Args:
        index: 当前索引
        total: 总数
    
    Returns:
        颜色值的十六进制表示
    """
    hue = (index / max(total, 1)) % 1.0
    r, g, b = colorsys.hls_to_rgb(hue, 0.6, 0.5)
    return '#{:02x}{:02x}{:02x}'.format(
        int(r * 255), int(g * 255), int(b * 255))


class Keyword:
    def __init__(
    self,
    raw: str,
    annotation: str,
    match_case: bool = False,
    whole_word: bool = False,
    use_regex: bool = False,
     color: str = "#ffff99"):
        self.raw = raw
        self.annotation = annotation
        self.match_case = match_case
        self.whole_word = whole_word
        self.use_regex = use_regex
        self.color = color

    def to_group(self, idx: int) -> Tuple[str, str]:
        """
        将关键词转换为正则表达式组。
        
        Args:
            idx: 关键词索引
        
        Returns:
            正则表达式模式和组名
        """
        pat = self.raw if self.use_regex else RE_MODULE.escape(self.raw)
        if self.whole_word:
            pat = rf'(?<!\w){pat}(?!\w)'
        if not self.match_case:
            pat = f'(?i:{pat})'
        name = f'k{idx}'
        return f'(?P<{name}>{pat})', name


def highlight_line(line: str, regex: 're.Pattern',
                   mapping: Dict[str, Dict[str, str]]) -> str:
    """
    高亮显示一行日志中匹配的关键词。
    
    Args:
        line: 需要高亮的日志行
        regex: 匹配的正则表达式
        mapping: 关键词映射，格式: {group_name: {"annotation": text, "color": color}}
    
    Returns:
        包含HTML高亮标记的行
    """
    result = html.escape(line)
    for match in regex.finditer(line):
        matched_text = match.group(0)
        for name, group in match.groupdict().items():
            if group is not None:
                meta = mapping.get(name, {})
                color = meta.get("color", "#ffff99")
                annotation = meta.get("annotation", "")
                tooltip = f' title="{annotation}"' if annotation else ''

                # 将匹配的文本替换为带高亮的HTML
                safe_text = html.escape(matched_text)
                highlight = f'<span style="background-color: {color};"{tooltip}>{safe_text}</span>'
                result = result.replace(
    html.escape(matched_text), highlight, 1)
                break

    return result


def parse_timestamp(line: str) -> datetime.datetime:
    """
    从日志行中解析时间戳，格式为 MM-DD HH:MM:SS.sss。
    
    Args:
        line: 日志行
    
    Returns:
        解析后的时间戳，如果无法解析则返回 datetime.datetime.min
    """
    # 使用utils.py中的增强版时间戳解析实现
    result = utils.parse_timestamp_safe(line, use_default=False)
    if result is None:
        logging.warning(f"无法解析时间戳: {line[:18] if len(line) >= 18 else line}")
        return datetime.datetime.min
    return result


class ScanWorker(QThread):
    progress = pyqtSignal(str)  # 修改为单个字符串参数
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    warning = pyqtSignal(str)
    debug = pyqtSignal(str)  # 用于传递调试信息

    def __init__(self, file_paths: List[str], keywords: List[Dict[str, Any]],
                 out_path: str, max_workers: int, config_params: Dict[str, Any],
                 temp_manager: Optional[TempFileManager] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.file_paths = file_paths
        self.keywords = keywords  # 关键词信息列表
        self.out_path = out_path
        self.max_workers = max_workers
        self.max_results = config_params["max_results"]
        self.time_range_hours = config_params["time_range_hours"]
        self.max_file_size = config_params["max_file_size"]
        self.chunk_size = config_params["chunk_size"]
        self.max_output_files = CONFIG_DEFAULTS["max_output_files"]
        self._stop_requested = False
        self._processed_files = 0
        self._batch_update_size = config_params["batch_update_size"]
        self._result_truncated = False
        self._executor = None
        self.temp_manager = temp_manager or TempFileManager()
        self.progress_monitor = ProgressMonitor(self)
        self.checkpoint_manager = CheckpointManager()
        self.memory_monitor = None
        self.task_id = f"scan_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

        # 扫描模式和优化配置
        self.scan_mode = config_params.get(
    "scan_mode", CONFIG_DEFAULTS["scan_mode"])
        self.large_file_threshold = config_params.get(
    "large_file_threshold", CONFIG_DEFAULTS["large_file_threshold"])
        self.huge_file_threshold = config_params.get(
    "huge_file_threshold", CONFIG_DEFAULTS["huge_file_threshold"])
        self.prefilter_enabled = config_params.get(
    "prefilter_enabled", CONFIG_DEFAULTS["prefilter_enabled"])
        self.bitmap_filter_enabled = config_params.get(
    "bitmap_filter_enabled", CONFIG_DEFAULTS["bitmap_filter_enabled"])

        # 使用优化的关键词匹配器
        self.keyword_matcher = KeywordMatcher(
    keywords, use_bitmap_filter=self.bitmap_filter_enabled)

        # 收集要匹配的所有原始关键词
        self.raw_list = [kw.get('raw', '') for kw in keywords]

        # 是否使用进程池
        self.use_process_pool = config_params.get("use_process_pool", True)

        # 结果缓存
        self._results_buffer = []
        self._results_buffer_size = 1000  # 每1000条结果写入一次

    def stop(self) -> None:
        """停止扫描任务并清理资源。"""
        self._stop_requested = True
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)

        # 保存检查点以便后续恢复
        self._save_checkpoint()

        if self.memory_monitor:
            self.memory_monitor.stop()

    def _save_checkpoint(self) -> None:
        """保存当前处理状态到检查点"""
        try:
            checkpoint_data = {
                "processed_files": self._processed_files,
                "out_path": self.out_path,
                "timestamp": datetime.datetime.now().isoformat()
            }
            self.checkpoint_manager.save_checkpoint(
                self.task_id, checkpoint_data)
            self.debug.emit(f"已保存检查点: {self.task_id}")
        except Exception as e:
            logging.error(f"保存检查点失败: {e}")

    def run(self) -> None:
        """扫描文件线程的主要执行函数"""
        start_time = time.time()
        self.progress.emit("正在初始化扫描...")

        # 创建并启动内存监控
        self.memory_monitor = MemoryMonitor(self)
        self.memory_monitor.start()

        # 检查是否有需要恢复的任务
        if self.checkpoint_manager.has_checkpoint(self.task_id):
            checkpoint = self.checkpoint_manager.load_checkpoint(self.task_id)
            if checkpoint:
                self.debug.emit(f"恢复之前的任务: {self.task_id}")
                self._processed_files = checkpoint.get("processed_files", 0)

        # 按照扫描模式调整检测策略
        self.debug.emit(f"使用扫描模式: {self.scan_mode}")
        if self.scan_mode == "fast":
            self.debug.emit("启用快速模式 - 使用低内存、高效率策略")
        elif self.scan_mode == "accurate":
            self.debug.emit("启用精确模式 - 使用高内存、高精度策略")

        # 开始文件分析
        worker_count = self.memory_monitor.suggest_worker_count(
            self.max_workers)
        self.debug.emit(
            f"开始分析 {len(self.file_paths)} 个文件，使用 {worker_count} 个工作单元")

        # 按照时间范围分组处理数据
        temp_output_files = []
        try:
            # 创建适当的执行器（进程池或线程池）
            self._executor = self._create_optimized_executor(worker_count)

            # 准备进度报告
            total_files = len(self.file_paths)
            processed = 0
            skipped = 0
            self.progress.emit("0%")

            # 跟踪各个时间范围的数据
            time_groups: Dict[Tuple[datetime.datetime,
                datetime.datetime], List[Tuple[str, str]]] = {}

            # 为了减少内存使用，分批处理文件
            batches = self._prepare_batched_tasks()

            # 分批处理文件
            for batch_idx, batch in enumerate(batches):
                if self._stop_requested:
                    break

                self.debug.emit(
                    f"处理批次 {batch_idx + 1}/{len(batches)}，包含 {len(batch)} 个文件")

                # 并发处理当前批次
                future_to_file = {}
                for file_path in batch:
                    if self._stop_requested:
                        break
                    future = self._executor.submit(self.scan_file, file_path)
                    future_to_file[future] = file_path

                # 收集当前批次结果
                for future in as_completed(future_to_file):
                    if self._stop_requested:
                        break

                    file_path = future_to_file[future]
                    try:
                        filename, matches = future.result()
                        processed += 1

                        # 处理匹配结果
                        if matches:
                            self.debug.emit(f"在文件 {filename} 中找到 {len(matches)} 个匹配")

                            # 按照时间范围分组
                            for timestamp, highlighted in matches:
                                # 尝试解析时间戳
                                try:
                                    ts = parse_timestamp(timestamp)
                                    start_ts = ts.replace(
                                        minute=0, second=0, microsecond=0,
                                        hour=ts.hour - (ts.hour %
                                                        self.time_range_hours)
                                    )
                                    end_ts = start_ts + \
                                        datetime.timedelta(
                                            hours=self.time_range_hours)
                                    if (start_ts, end_ts) not in time_groups:
                                        time_groups[(start_ts, end_ts)] = []
                                    time_groups[(start_ts, end_ts)].append(
                                        (ts, highlighted))

                                    # 如果结果缓冲区太大，执行垃圾回收
                                    if sum(
    len(group) for group in time_groups.values()) % 5000 == 0:
                                        gc.collect()

                                except Exception:
                                    # 如果时间解析失败，放入默认组
                                    default_ts = datetime.datetime.now()
                                    start_ts = default_ts.replace(
                                        hour=0, minute=0, second=0, microsecond=0)
                                    end_ts = start_ts + \
                                        datetime.timedelta(days=1)
                                    if (start_ts, end_ts) not in time_groups:
                                        time_groups[(start_ts, end_ts)] = []
                                    time_groups[(start_ts, end_ts)].append(
                                        (datetime.datetime.now(), highlighted))
                        else:
                            skipped += 1

                        # 更新进度
                        if processed % self._batch_update_size == 0 or processed == total_files:
                            progress_pct = min(
                                int((processed / total_files) * 100), 100)
                            self.progress.emit(f"{progress_pct}% ({processed}/{total_files})")

                            # 定期保存检查点
                            self._processed_files = processed
                            if processed % 50 == 0:
                                self._save_checkpoint()

                                # 尝试释放内存
                                if self.memory_monitor.is_warning():
                                    self._flush_results_for_group(time_groups)
                    except Exception as e:
                        logging.error(f"处理文件 {file_path} 失败: {e}")
                        self.progress_monitor.record_error(
                            "文件处理", f"{file_path}: {str(e)}")

                # 每完成一个批次，清理一次内存
                gc.collect()

            # 处理结果
            if not self._stop_requested and time_groups:
                self.progress.emit("正在生成输出文件...")

                # 检查输出文件数量限制
                if len(time_groups) > self.max_output_files:
                    self.warning.emit(f"时间范围过多，将限制为最多 {self.max_output_files} 个输出文件")
                    # 保留最新的N个时间范围
                    sorted_groups = sorted(
    time_groups.keys(), key=lambda x: x[0], reverse=True)
                    for time_range in sorted_groups[self.max_output_files:]:
                        del time_groups[time_range]

                # 为每个时间范围创建输出文件
                temp_output_files = self._generate_output_files(time_groups)

            # 生成统计报告
            if not self._stop_requested:
                self.progress.emit("正在生成统计报告...")
                self._generate_summary_report(
    temp_output_files, processed, skipped)

        except Exception as e:
            self.error.emit(f"扫描过程中发生错误: {str(e)}")
            logging.error(f"扫描过程中发生错误: {e}", exc_info=True)
        finally:
            # 清理资源
            if self._executor:
                self._executor.shutdown()

            # 停止内存监控
            if self.memory_monitor:
                self.memory_monitor.stop()

            # 输出完成信息
            duration = time.time() - start_time
            self.debug.emit(f"扫描完成，用时: {duration:.2f} 秒")

            if self._result_truncated:
                self.warning.emit(f"部分结果因超出限制被截断。建议增加 '最大结果数' 设置或分割日志文件。")

            if not self._stop_requested:
                # 打开生成的第一个文件
                if temp_output_files:
                    first_file = temp_output_files[0]
                    self.debug.emit(f"正在打开结果文件: {first_file}")
                    webbrowser.open(f"file://{os.path.abspath(first_file)}")
                else:
                    self.debug.emit("未生成任何输出文件")

            # 发送完成信号
            self.progress.emit("已完成")
            self.finished.emit(summary_path if "summary_path" in locals() else "")

    def _flush_results_for_group(self, time_groups):
        """尝试将一部分结果写入临时文件以释放内存"""
        # 由于内存限制，尝试将部分结果写入临时文件以释放内存
        if not time_groups:
            return

        # 选择结果最多的时间组进行写入
        largest_group = max(time_groups.items(), key=lambda x: len(x[1]), default=None)
        if largest_group and len(largest_group[1]) > 1000:  # 只有当结果超过1000条时才写入
            start_ts, end_ts = largest_group[0]
            matches = largest_group[1]
            
            # 创建临时文件
            prefix = f"log_hl_temp_{start_ts.strftime('%Y%m%d%H%M%S')}_"
            suffix = ".html.part"
            temp_path, temp_file = self.temp_manager.create_temp_file(
                prefix=prefix, suffix=suffix,
                time_range=(start_ts, end_ts)
            )
            
            # 写入结果到临时文件
            temp_file.write(f"<!-- 临时结果文件，时间范围: {start_ts} 到 {end_ts} -->\n")
            temp_file.write("<pre>\n")
            for ts, highlighted in matches:
                if isinstance(ts, datetime.datetime):
                    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    ts_str = str(ts)
                temp_file.write(f"<span class=\"timestamp\">[{ts_str}]</span> {highlighted}<br>\n")
            temp_file.write("</pre>\n")
            
            # 关闭临时文件
            self.temp_manager.close_file(temp_path)
            
            # 从内存中移除已写入的数据
            del time_groups[(start_ts, end_ts)]
            self.debug.emit(f"已将时间范围 {start_ts} 到 {end_ts} 的 {len(matches)} 条结果写入临时文件 {temp_path}")
            
            # 触发垃圾回收
            gc.collect()

    def _generate_output_files(self, time_groups) -> List[str]:
        """生成所有输出文件"""
        temp_output_files = []

        for i, ((start_ts, end_ts), matches) in enumerate(
            sorted(time_groups.items(), key=lambda x: x[0][0])):
            if self._stop_requested:
                break

            if not matches:
                continue

            # 创建临时HTML文件
            prefix = f"log_hl_{i:03d}_"
            suffix = ".html.tmp"
            temp_path, temp_file = self.temp_manager.create_temp_file(
                prefix=prefix, suffix=suffix,
                time_range=(start_ts, end_ts)
            )

            # 写入HTML头部
            temp_file.write('<!DOCTYPE html>\n<html>\n<head>\n')
            temp_file.write('<meta charset="utf-8">\n')
            temp_file.write(f'<title>日志分析 {start_ts} - {end_ts}</title>\n')
            temp_file.write('<style>\n')
            temp_file.write(
                'body { font-family: Consolas, monospace; margin: 20px; }\n')
            temp_file.write(
                'pre { line-height: 1.5; white-space: pre-wrap; }\n')
            temp_file.write('.timestamp { color: #666; font-weight: bold; }\n')
            temp_file.write('</style>\n')
            temp_file.write('</head>\n<body>\n')
            temp_file.write(f"<h1>日志分析: {start_ts.strftime('%Y-%m-%d %H:%M')} 到 {end_ts.strftime('%Y-%m-%d %H:%M')}</h1>\n")

            # 按时间排序匹配结果
            sorted_matches = sorted(matches, key=lambda x: x[0])

            # 检查结果数量限制
            if len(sorted_matches) > self.max_results:
                self._result_truncated = True
                temp_file.write(f"<div style=\"color:red;font-weight:bold;\">警告: 结果数量超过 {self.max_results} 条限制，仅显示前 {self.max_results} 条。</div>\n")
                sorted_matches = sorted_matches[:self.max_results]

            # 写入结果
            temp_file.write('<pre>\n')
            for ts, highlighted in sorted_matches:
                if isinstance(ts, datetime.datetime):
                    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    ts_str = str(ts)
                temp_file.write(f"<span class=\"timestamp\">[{ts_str}]</span> {highlighted}<br>\n")
            temp_file.write('</pre>\n')

            # 关闭临时文件
            self.temp_manager.close_file(temp_path)

            # 生成最终HTML文件
            output_file = self._finalize_output_file(
                temp_path, start_ts, end_ts)
            if output_file:
                temp_output_files.append(output_file)
                self.debug.emit(f"已生成输出文件 {i + 1}/{len(time_groups)}: {os.path.basename(output_file)}")

        return temp_output_files

    def _create_optimized_executor(self, worker_count: int):
        """创建优化的执行器，优先使用线程池以避免序列化问题"""
        # 由于进程池可能导致序列化问题，我们强制使用线程池
        self.debug.emit(f"创建线程池，线程数: {worker_count}")
        return ThreadPoolExecutor(max_workers=worker_count)

    def _prepare_batched_tasks(self) -> List[List[str]]:
        """将文件任务分批，优先处理小文件"""
        # 获取文件大小信息
        file_sizes = []
        for path in self.file_paths:
            try:
                size = os.path.getsize(path)
                file_sizes.append((path, size))
            except Exception as e:
                # 文件可能无法访问，使用0大小
                logging.warning(f"无法获取文件大小 {path}: {e}")
                file_sizes.append((path, 0))

        # 按大小排序 (小到大)
        file_sizes.sort(key=lambda x: x[1])

        # 批量划分，平衡每批的文件数量和总大小
        batches = []
        current_batch = []
        current_batch_size = 0
        target_batch_size = sum(size for _,
     size in file_sizes) / (self.max_workers * 2)
        target_batch_size = max(
    target_batch_size,
     10 * 1024 * 1024)  # 最小10MB一批

        for path, size in file_sizes:
            if current_batch_size + size > target_batch_size and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_batch_size = 0

            current_batch.append(path)
            current_batch_size += size

        # 添加最后一批
        if current_batch:
            batches.append(current_batch)

        self.debug.emit(f"任务分批完成: {len(batches)} 批")
        return batches

    def scan_file(self, path: str) -> Tuple[str, List[Tuple[str, str]]]:
        """处理单个文件，根据文件大小选择不同处理策略"""
        out = []
        fname = os.path.basename(path)
        try:
            # 检查文件大小
            file_size = os.path.getsize(path)
            if file_size > self.max_file_size:
                logging.warning(
                    f"文件 {path} 超出大小限制 ({file_size} bytes)，已跳过")
                return fname, out

            # 根据文件大小和扫描模式选择处理策略
            if self.scan_mode == "auto":
                if file_size < self.large_file_threshold:
                    return self._scan_small_file(path)
                elif file_size < self.huge_file_threshold:
                    return self._scan_large_file_streaming(path)
                else:
                    return self._scan_huge_file_mmap(path)
            elif self.scan_mode == "fast":
                return self._scan_large_file_streaming(path)
            elif self.scan_mode == "accurate":
                return self._scan_small_file(path)
            else:  # balanced
                if file_size < self.huge_file_threshold:
                    return self._scan_large_file_streaming(path)
                else:
                    return self._scan_huge_file_mmap(path)
        except Exception as e:
            logging.error(f"扫描文件 {path} 时出错: {e}")
            return fname, out
        
    def _scan_small_file(self, path: str) -> Tuple[str, List[Tuple[str, str]]]:
        """处理小文件，一次性读取整个文件到内存"""
        out = []
        fname = os.path.basename(path)
        try:
            for encoding in ['utf-8', 'gbk', 'gb2312', 'latin-1']:
                try:
                    with open(path, 'r', encoding=encoding) as f:
                        for line in f:
                            if self._stop_requested:
                                break
                            self._process_line_with_matcher(line, out)
                    return fname, out
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logging.error(f"读取文件 {path} 失败: {e}")
                    break
        except Exception as e:
            logging.error(f"读取文件 {path} 失败: {e}")
        return fname, out
        
    def _scan_large_file_streaming(self, path: str) -> Tuple[str, List[Tuple[str, str]]]:
        """流式处理大文件（待实现/补全）"""
        out = []
        fname = os.path.basename(path)
        try:
            for encoding in ['utf-8', 'gbk', 'gb2312', 'latin-1']:
                try:
                    with open(path, 'r', encoding=encoding, errors='ignore') as f:
                        for line in f:
                            if self._stop_requested:
                                break
                            self._process_line_with_matcher(line, out)
                    return fname, out
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logging.error(f"读取文件 {path} 失败: {e}")
                    break
        except Exception as e:
            logging.error(f"读取文件 {path} 失败: {e}")
        return fname, out
    
    def _scan_huge_file_mmap(self, path: str) -> Tuple[str, List[Tuple[str, str]]]:
        """使用内存映射处理超大文件"""
        out = []
        fname = os.path.basename(path)
        if not MMAP_AVAILABLE:
            logging.warning("mmap不可用，回退到流式处理")
            return self._scan_large_file_streaming(path)
        try:
            with open(path, 'r+b') as f:
                # 尝试推断文件编码
                encoding = self._detect_file_encoding(path) if hasattr(self, '_detect_file_encoding') else 'utf-8'
                # 使用内存映射
                mm = mmap.mmap(f.fileno(), 0)
                pos = 0
                while pos < mm.size():
                    if self._stop_requested:
                        break
                    # 查找下一个换行符
                    next_pos = mm.find(b'\n', pos)
                    if next_pos == -1:
                        next_pos = mm.size()
                    # 读取一行
                    line_bytes = mm[pos:next_pos]
                    try:
                        line = line_bytes.decode(encoding, errors='ignore')
                        self._process_line_with_matcher(line, out)
                        # 处理结果太多时截断
                        if len(out) >= self.max_results * 2:
                            out = out[:self.max_results]
                            break
                    except UnicodeDecodeError:
                        # 解码失败则跳过此行
                        pass
                    pos = next_pos + 1
                mm.close()
        except Exception as e:
            logging.error(f"内存映射处理文件 {path} 失败: {e}")
            # 出错时回退到流式处理
            if not out:
                return self._scan_large_file_streaming(path)
        return fname, out
    
    def _process_line_with_matcher(self, line: str, out: List[Tuple[str, str]]) -> None:
        """使用KeywordMatcher处理一行文本"""
        if self._stop_requested:
            return
            
        line = line.rstrip('\n')
        
        # 使用KeywordMatcher进行高效匹配
        has_match, highlighted = self.keyword_matcher.highlight_line(line)
        
        if has_match:
            ts = line[:18] if len(line) >= 18 else line
            out.append((ts, highlighted))

    def _generate_summary_report(self, output_files: List[str], processed: int, skipped: int) -> None:
        """生成扫描结果统计报告"""
        if not output_files:
            return
            
        try:
            # 创建一个摘要HTML文件
            summary_path = f"{self.out_path}_summary.html"
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write('<!DOCTYPE html>\n<html>\n<head>\n')
                f.write('<meta charset="utf-8">\n')
                f.write('<title>日志分析摘要</title>\n')
                f.write('<style>\n')
                f.write('body { font-family: Arial, sans-serif; margin: 20px; }\n')
                f.write('h1 { color: #333; }\n')
                f.write('table { border-collapse: collapse; width: 100%; }\n')
                f.write('th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n')
                f.write('th { background-color: #f2f2f2; }\n')
                f.write('tr:nth-child(even) { background-color: #f9f9f9; }\n')
                f.write('</style>\n')
                f.write('</head>\n<body>\n')
                f.write('<h1>日志分析摘要</h1>\n')
                
                # 写入基本信息
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f'<p>扫描时间: {timestamp}</p>\n')
                f.write(f'<p>处理文件: {processed}个</p>\n')
                f.write(f'<p>无匹配文件: {skipped}个</p>\n')
                f.write(f'<p>生成报告: {len(output_files)}个</p>\n')
                
                # 关键词统计
                f.write('<h2>关键词组</h2>\n')
                f.write('<ul>\n')
                for pattern, meta in self.keyword_matcher.pattern_mapping.items():
                    annotation = meta.get('annotation', '')
                    color = meta.get('color', '#ffff99')
                    f.write(f'<li><span style="background-color: {color};">{pattern}</span>: {annotation}</li>\n')
                f.write('</ul>\n')
                
                # 扫描设置
                f.write('<h2>扫描设置</h2>\n')
                f.write('<table>\n')
                f.write('<tr><th>设置项</th><th>值</th></tr>\n')
                f.write(f'<tr><td>扫描模式</td><td>{self.scan_mode}</td></tr>\n')
                f.write(f'<tr><td>预过滤</td><td>{"启用" if self.prefilter_enabled else "禁用"}</td></tr>\n')
                f.write(f'<tr><td>位图过滤</td><td>{"启用" if self.bitmap_filter_enabled else "禁用"}</td></tr>\n')
                f.write(f'<tr><td>线程数</td><td>{self.max_workers}</td></tr>\n')
                f.write(f'<tr><td>时间范围(小时)</td><td>{self.time_range_hours}</td></tr>\n')
                f.write('</table>\n')
                
                # 结果文件列表
                f.write('<h2>结果文件</h2>\n')
                f.write('<table>\n')
                f.write('<tr><th>文件名</th><th>时间范围</th></tr>\n')
                
                for file_path in sorted(output_files):
                    file_name = os.path.basename(file_path)
                    # 尝试从文件名解析时间范围
                    time_range = "未知"
                    parts = file_name.split('_')
                    if len(parts) >= 6:
                        try:
                            start = parts[-6] + '-' + parts[-5] + '-' + parts[-4]
                            end = parts[-3] + '-' + parts[-2] + '-' + parts[-1].split('.')[0]
                            time_range = f"{start} 到 {end}"
                        except:
                            pass
                            
                    rel_path = os.path.relpath(file_path, os.path.dirname(summary_path))
                    f.write(f'<tr><td><a href="{rel_path}">{file_name}</a></td><td>{time_range}</td></tr>\n')
                
                f.write('</table>\n')
                f.write('</body>\n</html>\n')
                
            self.debug.emit(f"已生成摘要报告: {summary_path}")
            
        except Exception as e:
            logging.error(f"生成摘要报告失败: {e}")
            self.progress_monitor.record_error("报告生成", str(e))

    def _finalize_output_file(self, temp_file: str, start_time: datetime.datetime, end_time: datetime.datetime) -> Optional[str]:
        """将临时HTML文件转换为最终输出文件"""
        try:
            if os.path.exists(temp_file):
                start_str = start_time.strftime("%Y-%m-%d_%H-%M-%S")
                end_str = end_time.strftime("%Y-%m-%d_%H-%M-%S")
                output_filename = f"{self.out_path}_{start_str}_to_{end_str}.html"
                with open(output_filename, 'w', encoding='utf-8') as outf:
                    with open(temp_file, 'r', encoding='utf-8') as tempf:
                        content = tempf.read()
                        outf.write(content)
                        outf.write('</body></html>')
                # 删除临时文件
                self.temp_manager.remove_file(temp_file)
                return output_filename
        except Exception as e:
            logging.error(f"创建最终输出文件失败: {e}")
            self.error.emit(f"创建最终输出文件失败: {str(e)}")
            return None

    def _is_archive_file(self, filepath: str) -> bool:
        """检查文件是否为支持的压缩格式"""
        extensions = ('.rar', '.zip', '.7z', '.tar', '.tar.gz', '.tgz')
        return filepath.lower().endswith(extensions)

    def on_decompress_finished(self, logs: List[str]) -> None:
        """解压完成后自动触发关键词分析"""
        self.debug.append("解压完成，开始自动分析关键词")
        self.decompressed_files.extend(logs)
        self.analyze_combined_keywords()

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
        """获取所有日志文件路径，优先使用异步解压的结果"""
        paths = []
        
        # 优先收集已经解压的文件
        if self.decompressed_files:
            paths.extend(self.decompressed_files)
            
        # 再收集非压缩文件源
        for src in self.history["sources"]:
            if os.path.isfile(src):
                # 不处理压缩文件，它们应该通过 DecompressWorker 处理
                if not self._is_archive_file(src):
                    paths.append(src)
            elif os.path.isdir(src):
                # 仅收集目录中的非压缩文件
                for root, _, files in os.walk(src):
                    for fn in files:
                        if not self._is_archive_file(fn):
                            full_path = os.path.join(root, fn)
                            paths.append(full_path)
        
        # 去重并排序
        paths = list(set(paths))
        paths.sort()
        
        # 调试信息
        self.debug.append(f"收集到 {len(paths)} 个日志文件用于分析")
        
        return paths

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

    def decompress_gz(self, gz_path: str, dest_path: str) -> bool:
        """解压 .gz 文件到指定路径"""
        try:
            with gzip.open(gz_path, 'rb') as f_in, open(dest_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            logging.info(f"成功解压 {gz_path} 到 {dest_path}")
            self.debug.append(f"成功解压 {os.path.basename(gz_path)} 到 {dest_path}")
            return True
        except Exception as e:
            logging.error(f"解压 {gz_path} 失败: {e}")
            self.debug.append(f"解压 {os.path.basename(gz_path)} 失败: {str(e)}")
        return False

    def decompress_archive(self, archive_path: str, dest_dir: str) -> bool:
        """解压各种类型的压缩包到指定目录"""
        if PYUNPACK_AVAILABLE:
            try:
                Archive(archive_path).extractall(dest_dir)
                logging.info(f"成功解压 {archive_path} 到 {dest_dir}")
                self.debug.append(f"成功解压 {os.path.basename(archive_path)} 到 {dest_dir}")
                return True
            except Exception as e:
                logging.error(f"使用 pyunpack 解压 {archive_path} 失败: {e}")
                self.debug.append(f"使用 pyunpack 解压 {os.path.basename(archive_path)} 失败: {str(e)}")
        
        # 如果 pyunpack 不可用或失败，尝试特定格式的解压
        if archive_path.lower().endswith('.rar'):
            return self.decompress_rar(archive_path, dest_dir)
        elif archive_path.lower().endswith('.gz'):
            dest_path = os.path.join(dest_dir, os.path.basename(os.path.splitext(archive_path)[0]))
            return self.decompress_gz(archive_path, dest_path)
        else:
            logging.error(f"不支持的压缩格式: {archive_path}")
            self.debug.append(f"不支持的压缩格式: {os.path.basename(archive_path)}")
            return False

    def find_and_decompress_gz(self, directory: str) -> List[str]:
        """递归查找并解压目录中的所有 .gz 文件"""
        decompressed_paths = []
        for root, _, files in os.walk(directory):
            for f in files:
                if f.endswith('.gz'):
                    gz_path = os.path.join(root, f)
                    dest_path = os.path.splitext(gz_path)[0]
                    self.progress.emit(f"解压 {f} 到 {dest_path}...")
                    if self.decompress_gz(gz_path, dest_path):
                        decompressed_paths.append(dest_path)
        return decompressed_paths

    def recursive_decompress_archives(self, root_dir: str) -> List[str]:
        """递归解压 root_dir 中所有支持的压缩包并处理内部 .gz"""
        decompressed = []
        for curdir, _, files in os.walk(root_dir):
            for f in files:
                if self._is_archive_file(f):
                    archive_path = os.path.join(curdir, f)
                    dest = os.path.join(curdir, os.path.splitext(f)[0])
                    os.makedirs(dest, exist_ok=True)
                    self.progress.emit(f"递归解压 {f} 到 {dest}...")
                    if self.decompress_archive(archive_path, dest):
                        if f.lower().endswith('.rar'):
                            gz_files = self.find_and_decompress_gz(dest)
                            decompressed.extend(gz_files)
                        sub_archives = self.recursive_decompress_archives(dest)
                        decompressed.extend(sub_archives)
                    decompressed.append(dest)
        return decompressed

    def _update_optimization_info(self) -> None:
        """更新优化信息提示"""
        mode_desc = {
            "auto": "自动 - 根据文件大小智能选择处理策略",
            "fast": "快速 - 优先考虑性能和低内存占用，可能错过某些复杂匹配",
            "accurate": "精确 - 优先考虑准确匹配，内存占用较高",
            "balanced": "平衡 - 兼顾性能和准确度"
        }
        
        mode = self.config_params.get("scan_mode", "auto")
        prefilter = self.config_params.get("prefilter_enabled", True)
        bitmap = self.config_params.get("bitmap_filter_enabled", True)
        
        if hasattr(self, 'debug'):
            optimizations = []
            optimizations.append(f"当前模式: {mode_desc.get(mode, mode)}")
            
            if prefilter:
                optimizations.append("已启用关键词预过滤 (提高性能)")
            
            if bitmap:
                optimizations.append("已启用位图过滤 (加速大文件处理)")
                
            optimizations.append("使用分级正则匹配 (优化多关键词匹配)")
            
            if MMAP_AVAILABLE:
                optimizations.append("内存映射可用 (优化超大文件处理)")
                
            self.debug.append("\n优化设置:\n- " + "\n- ".join(optimizations))
    
    def analyze_combined_keywords(self) -> None:
        """开始分析日志文件中的关键词。"""
        # --- 防抖：立即禁用按钮 --- 
        self.btn_analysis.setEnabled(False)
        # Ensure button is re-enabled in case of early return
        should_re_enable_button = True 
        try:
            if not self.config_path or not os.path.isfile(self.config_path):
                QMessageBox.warning(self, "提示", "请选择有效配置文件")
                # self.btn_analysis.setEnabled(True) # Re-enable handled by finally
                return
            if self.worker and self.worker.isRunning():
                QMessageBox.warning(self, "提示", "已有分析正在进行，请稍候")
                # self.btn_analysis.setEnabled(True) # Re-enable handled by finally
            return

            self.debug.clear()
            if hasattr(self, 'error_log'):
                self.error_log.clear()
                
            files = self.get_log_files()
            if not files:
                QMessageBox.warning(self, "提示", "无日志文件可分析")
                # self.btn_analysis.setEnabled(True) # Re-enable handled by finally
                return
    
            # 收集自定义关键词
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

            # 收集分组关键词
            group_kws = []
            for i in range(self.group_layout.count()):
                cb = self.group_layout.itemAt(i).widget()
                if isinstance(cb, QCheckBox) and cb.isChecked():
                    group_name = cb.property("group_name")
                    if not group_name:
                        self.debug.append(f"警告: 复选框 {cb.text()} 没有关联的 group_name")
                        continue
                    self.debug.append(f"处理选中的分组: {group_name}")

                    # 从配置文件加载分组数据
                    try:
                        config_data = toml.load(self.config_path)
                        self.debug.append(f"  配置文件加载成功.")

                        # 正确访问嵌套结构: config['group']['subgroup_key']
                        top_group_dict = config_data.get('group', None)
                        if top_group_dict is None or not isinstance(top_group_dict, dict):
                            self.debug.append(f"  错误: 配置文件缺少顶层 'group' 字典结构.")
                            continue

                        # 从 group_name (e.g., "group.security") 提取子分组键 ("security")
                        subgroup_key = group_name.split('.')[-1] if '.' in group_name else group_name

                        grp = top_group_dict.get(subgroup_key, None)

                        if grp is None:
                            self.debug.append(f"  警告: 在 'group' 字典下未找到子分组 '{subgroup_key}' (来自 {group_name})")
                            continue
                        
                        # Check if grp is actually a dictionary before proceeding
                        if not isinstance(grp, dict):
                            self.debug.append(f"  警告: 子分组 '{subgroup_key}' 的值不是字典 (类型: {type(grp)}). 跳过.")
                            continue
                            
                        self.debug.append(f"  成功获取分组数据: {grp}")

                        mc = grp.get("match_case", False)
                        ww = grp.get("whole_word", False)
                        uz = grp.get("use_regex", False)
                        color = self.group_colors.get(group_name, "#ffff99")
                        self.debug.append(f"  分组设置: match_case={mc}, whole_word={ww}, use_regex={uz}")

                        # 遍历分组中的所有条目
                        entries_count = 0
                        self.debug.append(f"  开始遍历分组 '{subgroup_key}' (来自 {group_name}) 中的条目...")
                        for k, v in grp.items():
                            self.debug.append(f"    检查条目: key='{k}', value='{v}' (类型: {type(v)})")
                            # 只处理值为字典且包含 'key' 键的条目 (即关键词定义)
                            if isinstance(v, dict) and "key" in v:
                                keyword_text = v["key"]
                                self.debug.append(f"      找到关键词定义: '{keyword_text}'")
                                entries_count += 1
                                group_kws.append({
                                    "raw": keyword_text,
                                    "annotation": v.get("annotation", ""),
                                    "match_case": mc, # 使用分组的设置
                                    "whole_word": ww,
                                    "use_regex": uz,
                                    "color": color
                                })
                            else:
                                self.debug.append(f"      跳过条目 '{k}' (非关键词定义)")

                        self.debug.append(f"  从分组 '{subgroup_key}' (来自 {group_name}) 中成功添加了 {entries_count} 个关键词")

                    except Exception as e:
                        self.debug.append(f"  加载或处理分组 '{group_name}' 时发生错误: {str(e)}")
                        logging.error(f"加载分组 '{group_name}' 失败: {e}")
                        continue # 跳过这个有问题的分组
    
            # 合并关键词
            all_kws = custom_kws + group_kws
            self.debug.append(f"DEBUG: Just before check, len(all_kws) = {len(all_kws)}")
    
            # 如果没有任何关键词，就提示并返回
            if not all_kws:
                self.debug.append("DEBUG: Entering 'if not all_kws' block.")
                QMessageBox.warning(self, "提示", "请勾选至少一个自定义关键词或分组关键词")
                self.debug.append("DEBUG: After QMessageBox.warning, before return.")
                # self.btn_analysis.setEnabled(True) # Re-enable handled by finally
                return
            else:
                 self.debug.append("DEBUG: Skipped 'if not all_kws' block.")
    
            # --- Keyword check passed, analysis will proceed --- 
            should_re_enable_button = False # Don't re-enable if analysis starts
            
            # 收集要显示的原始文本作为提示
            raw_list = [kw['raw'] for kw in all_kws]
            self.debug.append(f"共选择了 {len(all_kws)} 个关键词:")
            for kw in all_kws[:10]:  # 只显示前10个
                self.debug.append(f"- {kw['raw']}")
            if len(all_kws) > 10:
                self.debug.append(f"... 以及 {len(all_kws) - 10} 个其他关键词")
    
            # 如果已有工作线程，取消它
            if self.worker and self.worker.isRunning():
                self.cancel_analysis() # cancel_analysis should handle button state
                should_re_enable_button = True # Re-enable after cancel
                return 
    
            # 更新UI (Button already disabled)
            # self.btn_analysis.setEnabled(False) 
            self.btn_cancel.setVisible(True)
            self.progress.setVisible(True)
            
            # 确保UI状态正确更新
            QApplication.processEvents()
            
            # ... (rest of worker creation and starting logic)
            # 设置输出路径
            out_dir = CONFIG_DEFAULTS["output_dir"]
            for s in self.history["sources"]:
                if os.path.isdir(s):
                    out_dir = s
                    break
            if not out_dir and self.history["sources"]:
                out_dir = os.path.dirname(self.history["sources"][0])
            out_path = os.path.join(out_dir, CONFIG_DEFAULTS["output_filename"])

            # 创建工作线程
            max_workers = self.spin_cores.value()
            
            # 更新配置
            if hasattr(self, 'use_process_pool'):
                self.config_params["use_process_pool"] = False  # 禁用进程池以避免序列化问题
            
            # 创建工作线程
            self.worker = ScanWorker(
                file_paths=files,
                keywords=all_kws,
                out_path=out_path,
                max_workers=max_workers,
                config_params=self.config_params,
                temp_manager=self.temp_manager,
                parent=self
            )
            
            # 连接信号 (Ensure _on_scan_finished re-enables the button)
            self.worker.error.connect(self._on_scan_error) # error should re-enable
            self.worker.warning.connect(lambda msg: QMessageBox.warning(self, "结果限制", msg))
            self.worker.progress.connect(self._on_scan_progress)
            self.worker.debug.connect(lambda msg: self.debug.append(msg))
            self.worker.finished.connect(self._on_scan_finished) # finished should re-enable

            # 启动工作线程
            self.worker.start()
            self.debug.append("工作线程已启动")
            
        finally:
            # --- Re-enable button only if we returned early --- 
            if should_re_enable_button:
                self.btn_analysis.setEnabled(True)
                self.debug.append("DEBUG: Analysis button re-enabled due to early return.")

    def _on_scan_error(self, message: str) -> None:
        """处理扫描错误"""
        QMessageBox.critical(self, "扫描错误", message)
        self.error_log.append(f"[错误] {message}")
        self._update_status_value("运行状态", "出错")
    
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
        if os.path.isfile(path):
            webbrowser.open(path)
            
        # 清理工作线程
        if self.worker:
            self.worker.deleteLater()
            self.worker = None

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

    def add_custom_keyword(self) -> None:
        """添加自定义关键词"""
        txt = self.keyword_combo.currentText().strip()
        if not txt:
                    return
        if txt not in self.history["keywords"]:
            self.history["keywords"].insert(0, txt)
            self.keyword_combo.insertItem(0, txt)
            self.save_settings()
        parts = [p.strip() for p in txt.split('|') if p.strip()]
        tot = len(self.custom_keyword_checks) + len(parts)
        for p in parts:
            idx = len(self.custom_keyword_checks)
            col = generate_color(idx, tot)
            kw = Keyword(p, "[自定义]",
                         self.case_box.isChecked(),
                         self.word_box.isChecked(),
                         self.regex_box.isChecked(),
                         col)
            cb = QCheckBox(p)
            cb.setProperty("keyword_obj", kw)
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

    def load_settings(self) -> None:
        """加载设置"""
        if os.path.exists(self.settings_path):
            try:
                with open(self.settings_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.config_path = data.get("config_path")
                if self.config_path:
                    self.cfg_edit.setText(self.config_path)
                h = data.get("history", {})
                self.history["sources"] = h.get("sources", [])
                self.history["keywords"] = h.get("keywords", [])
                cores = h.get("cores", os.cpu_count() or 1)
                self.spin_cores.setValue(cores)
                # 加载基本参数设置
                self.config_params["max_results"] = h.get("max_results", CONFIG_DEFAULTS["max_results"])
                self.spin_max_results.setValue(self.config_params["max_results"])
                self.config_params["time_range_hours"] = h.get("time_range_hours", CONFIG_DEFAULTS["time_range_hours"])
                self.spin_time_range.setValue(self.config_params["time_range_hours"])
                self.config_params["chunk_size"] = h.get("chunk_size", CONFIG_DEFAULTS["chunk_size"])
                self.spin_chunk_size.setValue(self.config_params["chunk_size"] // 1024)
                self.config_params["thread_timeout"] = h.get("thread_timeout", CONFIG_DEFAULTS["thread_timeout"])
                self.spin_thread_timeout.setValue(self.config_params["thread_timeout"])
                self.config_params["max_file_size"] = h.get("max_file_size", CONFIG_DEFAULTS["max_file_size"])
                self.spin_max_file_size.setValue(self.config_params["max_file_size"] // (1024 * 1024))
                self.config_params["batch_update_size"] = h.get("batch_update_size", CONFIG_DEFAULTS["batch_update_size"])
                self.spin_batch_update_size.setValue(self.config_params["batch_update_size"])
                # 加载优化相关设置
                self.config_params["scan_mode"] = h.get("scan_mode", CONFIG_DEFAULTS["scan_mode"])
                self.config_params["prefilter_enabled"] = h.get("prefilter_enabled", CONFIG_DEFAULTS["prefilter_enabled"])
                self.config_params["bitmap_filter_enabled"] = h.get("bitmap_filter_enabled", CONFIG_DEFAULTS["bitmap_filter_enabled"])
                self.config_params["large_file_threshold"] = h.get("large_file_threshold", CONFIG_DEFAULTS["large_file_threshold"])
                self.config_params["huge_file_threshold"] = h.get("huge_file_threshold", CONFIG_DEFAULTS["huge_file_threshold"])
                # 更新扫描模式UI
                mode_reverse_map = {
                    "auto": "自动",
                    "fast": "快速（低内存）", 
                    "accurate": "精确（高内存）", 
                    "balanced": "平衡"
                }
                mode_text = mode_reverse_map.get(self.config_params["scan_mode"], "自动")
                index = self.scan_mode_combo.findText(mode_text)
                if index >= 0:
                    self.scan_mode_combo.setCurrentIndex(index)
                # 更新预过滤设置
                if hasattr(self, 'prefilter_check'):
                    self.prefilter_check.setChecked(self.config_params["prefilter_enabled"])
                if hasattr(self, 'bitmap_filter_check'):
                    self.bitmap_filter_check.setChecked(self.config_params["bitmap_filter_enabled"])
                
                # 加载源和关键词历史
                for s in self.history["sources"]:
                    self.src_list.addItem(s)
                for kw in self.history["keywords"]:
                    self.keyword_combo.addItem(kw)
                    
                self.update_group_checkboxes()
                
                # 增强分组关键词解析
                if self.config_path and os.path.isfile(self.config_path):
                    raw_groups = toml.load(self.config_path).get("groups", {})
                    self.group_mapping: Dict[str, List[str]] = {}
                    for name, entries in raw_groups.items():
                        if isinstance(entries, dict):
                            kws = entries.get("keywords", [])
                        elif isinstance(entries, list):
                            kws = entries
                        elif isinstance(entries, str):
                            kws = [e.strip() for e in entries.split(",") if e.strip()]
                        else:
                            kws = []
                            self.debug.append(f"Unsupported group type for {name}: {type(entries)}")
                        self.group_mapping[name] = kws
                        self.debug.append(f"加载分组关键词 '{name}': {kws}")
                        
                # 更新优化信息
                if hasattr(self, 'debug'):
                    QTimer.singleShot(100, self._update_optimization_info)
            except Exception as e:
                logging.error(f"加载设置失败: {e}")
                QMessageBox.critical(self, "设置错误", f"加载设置失败: {str(e)}")

    def save_settings(self) -> None:
        """保存设置"""
        self.history["cores"] = self.spin_cores.value()
        self.history["max_results"] = self.config_params["max_results"]
        self.history["time_range_hours"] = self.config_params["time_range_hours"]
        self.history["chunk_size"] = self.config_params["chunk_size"]
        self.history["thread_timeout"] = self.config_params["thread_timeout"]
        self.history["max_file_size"] = self.config_params["max_file_size"]
        self.history["batch_update_size"] = self.config_params["batch_update_size"]
        
        # 保存优化设置
        self.history["scan_mode"] = self.config_params.get("scan_mode", CONFIG_DEFAULTS["scan_mode"])
        self.history["prefilter_enabled"] = self.config_params.get("prefilter_enabled", CONFIG_DEFAULTS["prefilter_enabled"])
        self.history["bitmap_filter_enabled"] = self.config_params.get("bitmap_filter_enabled", CONFIG_DEFAULTS["bitmap_filter_enabled"])
        self.history["large_file_threshold"] = self.config_params.get("large_file_threshold", CONFIG_DEFAULTS["large_file_threshold"])
        self.history["huge_file_threshold"] = self.config_params.get("huge_file_threshold", CONFIG_DEFAULTS["huge_file_threshold"])
        
        obj = {"config_path": self.config_path, "history": self.history}
        try:
            # 创建备份文件
            if os.path.exists(self.settings_path):
                backup_path = self.settings_path + ".bak"
                with open(self.settings_path, 'r', encoding='utf-8') as f:
                    with open(backup_path, 'w', encoding='utf-8') as bf:
                        bf.write(f.read())
            with open(self.settings_path, 'w', encoding='utf-8') as f:
                json.dump(obj, f, ensure_ascii=False)
        except Exception as e:
            logging.error(f"保存设置失败: {e}")
            QMessageBox.critical(self, "设置错误", f"保存设置失败: {str(e)}")

# === 添加高效关键词匹配类 ===
class KeywordMatcher:
    """高效的关键词匹配器，使用多级匹配策略"""
    
    def __init__(self, keywords: List[Dict], use_bitmap_filter: bool = True):
        """
        初始化关键词匹配器
        
        Args:
            keywords: 关键词列表，每项包含raw(关键词文本)、use_regex(是否使用正则)等属性
            use_bitmap_filter: 是否使用位图过滤
        """
        self.keywords = keywords
        self.use_bitmap_filter = use_bitmap_filter
        
        # 分类存储关键词
        self.exact_keywords = []  # 精确匹配(不区分大小写、非全词匹配、非正则)
        self.simple_keywords = []  # 简单匹配(可能区分大小写或全词匹配) 
        self.regex_keywords = []  # 复杂正则表达式
        
        # 为快速预过滤准备字符集
        self.char_set = set()
        self.bitmap = [False] * 256
        
        # 编译好的正则模式
        self.combined_pattern = None
        self.pattern_mapping = {}
        
        # 初始化匹配器
        self._categorize_keywords()
        self._build_filters()
        self._compile_patterns()
    
    def _categorize_keywords(self):
        """将关键词分类"""
        special_chars = set('.*+?[](){}|^$\\')
        
        for idx, kw in enumerate(self.keywords):
            raw = kw.get('raw', '')
            use_regex = kw.get('use_regex', False)
            match_case = kw.get('match_case', False)
            whole_word = kw.get('whole_word', False)
            
            # 收集字符到字符集和位图
            for c in raw:
                self.char_set.add(c.lower())
                self.bitmap[ord(c) & 0xFF] = True
                if not match_case:
                    self.bitmap[ord(c.lower()) & 0xFF] = True
                    self.bitmap[ord(c.upper()) & 0xFF] = True
            
            # 根据复杂度分类
            if use_regex or any(c in raw for c in special_chars):
                self.regex_keywords.append((idx, kw))
            elif match_case or whole_word:
                self.simple_keywords.append((idx, kw))
            else:
                self.exact_keywords.append((idx, kw))
    
    def _build_filters(self):
        """构建过滤器"""
        # 已在_categorize_keywords中实现了bitmap构建
        pass
    
    def _compile_patterns(self):
        """编译正则模式"""
        parts = []
        
        # 先处理精确匹配关键词
        for idx, kw in self.exact_keywords:
            raw = kw.get('raw', '')
            pattern = f"(?P<k{idx}>{re.escape(raw)})"
            parts.append(f"(?i:{pattern})")
            self.pattern_mapping[f"k{idx}"] = kw
        
        # 再处理简单匹配关键词 
        for idx, kw in self.simple_keywords:
            raw = kw.get('raw', '')
            match_case = kw.get('match_case', False)
            whole_word = kw.get('whole_word', False)
            
            pattern = re.escape(raw)
            if whole_word:
                pattern = fr'\b{pattern}\b'
                
            pattern = f"(?P<k{idx}>{pattern})"
            if not match_case:
                pattern = f"(?i:{pattern})"
                
            parts.append(pattern)
            self.pattern_mapping[f"k{idx}"] = kw
        
        # 最后处理复杂正则
        for idx, kw in self.regex_keywords:
            raw = kw.get('raw', '')
            match_case = kw.get('match_case', False)
            
            pattern = f"(?P<k{idx}>{raw})"
            if not match_case:
                pattern = f"(?i:{pattern})"
                
            parts.append(pattern)
            self.pattern_mapping[f"k{idx}"] = kw
        
        # 合并所有模式
        if parts:
            self.combined_pattern = re.compile("|".join(parts))
    
    def should_process_line(self, line: str) -> bool:
        """使用过滤器快速判断是否需要处理该行"""
        if not self.use_bitmap_filter:
            return True
            
        for c in line:
            if self.bitmap[ord(c) & 0xFF]:
                return True
        return False
    
    def match_line(self, line: str) -> List[Tuple[str, Dict]]:
        """匹配单行文本，返回匹配结果"""
        if not self.combined_pattern:
            return []
            
        # 快速过滤
        if not self.should_process_line(line):
            return []
            
        results = []
        for match in self.combined_pattern.finditer(line):
            for group_name, matched_text in match.groupdict().items():
                if matched_text is not None:
                    kw = self.pattern_mapping.get(group_name)
                    if kw:
                        results.append((matched_text, kw))
        
        return results
    
    def highlight_line(self, line: str) -> Tuple[bool, str]:
        """高亮显示一行文本中的匹配项"""
        if not self.combined_pattern:
            return False, html.escape(line)
            
        # 快速过滤
        if not self.should_process_line(line):
            return False, html.escape(line)
            
        result = html.escape(line)
        has_match = False
        
        for match in self.combined_pattern.finditer(line):
            matched_text = match.group(0)
            for group_name, group_match in match.groupdict().items():
                if group_match is not None:
                    kw = self.pattern_mapping.get(group_name, {})
                    color = kw.get('color', '#ffff99')
                    annotation = kw.get('annotation', '')
                    tooltip = f' title="{annotation}"' if annotation else ''
                    
                    has_match = True
                    safe_text = html.escape(matched_text)
                    highlight = f'<span style="background-color: {color};"{tooltip}>{safe_text}</span>'
                    result = result.replace(html.escape(matched_text), highlight, 1)
                    break
        
        return has_match, result

# === 内存监控状态小部件 ===
class MemoryStatusWidget(QFrame):
    """显示内存使用状态的小部件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setMinimumWidth(150)
        self.setMaximumWidth(200)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 内存使用率标签
        self.usage_label = QLabel("内存: 0%")
        self.usage_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.usage_label)
        
        # 内存使用进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # 峰值使用标签
        self.peak_label = QLabel("峰值: 0%")
        self.peak_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.peak_label)
        
        # 进程信息
        self.process_label = QLabel("进程: 0")
        self.process_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.process_label)
        
        # 初始化默认颜色
        self.setStyleSheet("QFrame { background-color: #f0f0f0; }")
        
        # 添加状态表
        self.status_table = QTableWidget(self)
        self.status_table.setColumnCount(2)
        self.status_table.setHorizontalHeaderLabels(["状态", "值"])
        self.status_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.status_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.status_table.setSelectionMode(QTableWidget.SingleSelection)
        self.status_table.setStyleSheet("QTableWidget { background-color: #ffffff; }")
        layout.addWidget(self.status_table)
        
        # 添加按钮
        self.update_button = QPushButton("更新状态", self)
        self.update_button.clicked.connect(self.update_status)
        layout.addWidget(self.update_button)
        
        self.setLayout(layout)
        
    def update_memory_status(self, usage: float, is_warning: bool, is_critical: bool):
        """更新内存状态显示"""
        self.usage_label.setText(f"内存: {usage:.1f}%")
        self.progress_bar.setValue(int(usage))
        self.peak_label.setText(f"峰值: {max(self.progress_bar.value(), self.peak_label.text().split('%')[0])}%")
        self.process_label.setText(f"进程: {mp.cpu_count()}")
        
        # 更新状态表
        self._update_status_value("内存使用率", f"{usage:.1f}%")
        self._update_status_value("内存状态", "危险" if is_critical else ("警告" if is_warning else "正常"))
    
    def _update_status_value(self, key: str, value: str):
        """更新状态表中的值"""
        if not hasattr(self, 'status_table'):
            return
            
        # 查找行
        for row in range(self.status_table.rowCount()):
            item = self.status_table.item(row, 0)
            if item and item.text() == key:
                # 更新值
                self.status_table.setItem(row, 1, QTableWidgetItem(value))
                break
    
    def update_status(self):
        """手动更新状态"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            usage = memory.percent
            is_warning = usage >= 75
            is_critical = usage >= 90
            
            self.update_memory_status(usage, is_warning, is_critical)
            
        except (ImportError, Exception) as e:
            # 无法获取内存信息，可能psutil未安装
                pass

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
        left = QWidget()
        ll = QVBoxLayout(left)
        topSplitter.addWidget(left)

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
        self.btn_remove.clicked.connect(self.remove_sources)
        self.btn_clear = QPushButton("清除历史")
        self.btn_clear.clicked.connect(self.clear_history)
        for b in (self.btn_add_dir, self.btn_add_file, self.btn_add_archive, self.btn_remove, self.btn_clear):
            btns.addWidget(b)
        btns.addStretch()
        src_l.addWidget(self.src_list, 4)
        src_l.addLayout(btns, 1)
        ll.addWidget(src_g)

        # CPU 核心数
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

        # 基本设置
        params_g = QGroupBox("参数设置")
        params_l = QVBoxLayout(params_g)
        
        # 最大结果数
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

        # 扫描模式
        scan_mode_l = QHBoxLayout()
        scan_mode_label = QLabel("扫描模式:")
        self.scan_mode_combo = QComboBox()
        self.scan_mode_combo.addItems(["自动", "快速（低内存）", "精确（高内存）", "平衡"])
        self.scan_mode_combo.currentTextChanged.connect(self._on_scan_mode_changed)
        scan_mode_l.addWidget(scan_mode_label)
        scan_mode_l.addWidget(self.scan_mode_combo)
        scan_mode_l.addStretch()
        params_l.addLayout(scan_mode_l)
        
        # 添加预过滤选项
        prefilter_l = QHBoxLayout()
        self.prefilter_check = QCheckBox("启用关键词预过滤")
        self.prefilter_check.setChecked(CONFIG_DEFAULTS["prefilter_enabled"])
        self.prefilter_check.stateChanged.connect(lambda v: self.update_config_param("prefilter_enabled", bool(v)))
        
        self.bitmap_filter_check = QCheckBox("启用位图过滤")
        self.bitmap_filter_check.setChecked(CONFIG_DEFAULTS["bitmap_filter_enabled"])
        self.bitmap_filter_check.stateChanged.connect(lambda v: self.update_config_param("bitmap_filter_enabled", bool(v)))
        
        # 使用进程池
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

        # 右侧面板
        right = QWidget()
        rl = QVBoxLayout(right)
        topSplitter.addWidget(right)

        grp_g = QGroupBox("关键词分组（可多选）")
        grp_l = QVBoxLayout(grp_g)
        self.grp_scroll = QScrollArea()
        self.grp_scroll.setWidgetResizable(True)
        cont_g = QWidget()
        self.group_layout = QVBoxLayout(cont_g)
        self.grp_scroll.setWidget(cont_g)
        grp_l.addWidget(self.grp_scroll)
        rl.addWidget(grp_g)

        cst_g = QGroupBox("自定义关键词")
        cst_l = QHBoxLayout(cst_g)
        self.keyword_combo = QComboBox(editable=True)
        self.case_box = QCheckBox("区分大小写")
        self.word_box = QCheckBox("全字匹配")
        self.regex_box = QCheckBox("使用正则")
        self.btn_add_kw = QPushButton("添加")
        self.btn_add_kw.clicked.connect(self.add_custom_keyword)
        self.btn_clear_kw = QPushButton("清除勾选")
        self.btn_clear_kw.clicked.connect(self.clear_selected_custom_keywords)
        self.btn_sel_all_kw = QPushButton("全选")
        self.btn_sel_all_kw.clicked.connect(self.select_all_custom_keywords)
        for w in (self.keyword_combo, self.case_box, self.word_box, self.regex_box,
                  self.btn_add_kw, self.btn_clear_kw, self.btn_sel_all_kw):
            cst_l.addWidget(w)
        rl.addWidget(cst_g)
        self.custom_scroll = QScrollArea()
        self.custom_scroll.setWidgetResizable(True)
        cont_c = QWidget()
        self.custom_layout = QVBoxLayout(cont_c)
        self.custom_scroll.setWidget(cont_c)
        rl.addWidget(self.custom_scroll)

        ana_g = QGroupBox("分析控制")
        ana_l = QHBoxLayout(ana_g)
        self.btn_analysis = QPushButton("开始分析")
        self.btn_analysis.clicked.connect(self.analyze_combined_keywords)
        self.btn_cancel = QPushButton("取消分析")
        self.btn_cancel.clicked.connect(self.cancel_analysis)
        self.btn_cancel.setVisible(False)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        ana_l.addWidget(self.btn_analysis)
        ana_l.addWidget(self.btn_cancel)
        ana_l.addWidget(self.progress)
        rl.addWidget(ana_g)
        rl.addStretch()

        dbg_g = QGroupBox("调试输出")
        dbg_l = QVBoxLayout(dbg_g)
        self.debug = QTextEdit(readOnly=True)
        self.error_log = QTextEdit(readOnly=True)
        dbg_l.addWidget(self.debug)
        mainSplitter.addWidget(dbg_g)

        topSplitter.setStretchFactor(0, 1)
        topSplitter.setStretchFactor(1, 2)
        mainSplitter.setStretchFactor(0, 3)
        mainSplitter.setStretchFactor(1, 1)

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
            
        # 对于布尔值参数的特殊处理
        if isinstance(value, bool):
            if key == "prefilter_enabled":
                self.debug.append(f"{'启用' if value else '禁用'}关键词预过滤")
            elif key == "bitmap_filter_enabled":
                self.debug.append(f"{'启用' if value else '禁用'}位图过滤")

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
        extensions = ('.rar', '.zip', '.7z', '.tar', '.tar.gz', '.tgz')
        return filepath.lower().endswith(extensions)

    def update_group_checkboxes(self) -> None:
        """根据加载的 TOML 配置文件更新关键词分组的复选框。"""
        # 清除现有的分组复选框
        for i in reversed(range(self.group_layout.count())):
            widget = self.group_layout.itemAt(i).widget()
            if widget:
                self.group_layout.removeWidget(widget)
                widget.deleteLater()
        self.group_colors.clear() # 清除旧颜色映射

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
                        full_group_name = f"group.{sub_group_key}" # 构建完整的组名，如 group.errors
                        display_name = sub_group_key # 使用子分组名作为显示名称
                        # 检查 group_data 是否是字典
                        if not isinstance(group_data, dict):
                            self.debug.append(f"警告: '{full_group_name}' 在配置文件中的值不是字典，跳过.")
                            continue
                        color = generate_color(idx, total_groups)
                        self.group_colors[full_group_name] = color # 使用完整组名作为键
                        cb = QCheckBox(display_name)
                        cb.setProperty("group_name", full_group_name) # 存储完整组名
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
        f, _ = QFileDialog.getOpenFileName(self, "添加压缩包", "", 
            "所有支持格式 (*.rar *.zip *.7z *.tar *.tar.gz *.tgz);;RAR (*.rar);;ZIP (*.zip)")
        if f and f not in self.history["sources"]:
            self.history["sources"].insert(0, f)
            self.src_list.insertItem(0, f)
            self.save_settings()
            self.debug.append(f"添加压缩包: {f}")
            # 自动解压添加的压缩包
            self.process_archives([f])

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
        txt = self.keyword_combo.currentText().strip()
        if not txt:
            return
        if txt not in self.history["keywords"]:
            self.history["keywords"].insert(0, txt)
            self.keyword_combo.insertItem(0, txt)
            self.save_settings()
        parts = [p.strip() for p in txt.split('|') if p.strip()]
        tot = len(self.custom_keyword_checks) + len(parts)
        for p in parts:
            idx = len(self.custom_keyword_checks)
            col = generate_color(idx, tot)
            kw = Keyword(p, "[自定义]",
                         self.case_box.isChecked(),
                         self.word_box.isChecked(),
                         self.regex_box.isChecked(),
                         col)
            cb = QCheckBox(p)
            cb.setProperty("keyword_obj", kw)
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
        if os.path.isfile(path):
            webbrowser.open(path)
            
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
        if os.path.exists(self.settings_path):
            try:
                with open(self.settings_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.config_path = data.get("config_path")
                if self.config_path:
                    self.cfg_edit.setText(self.config_path)
                h = data.get("history", {})
                self.history["sources"] = h.get("sources", [])
                self.history["keywords"] = h.get("keywords", [])
                cores = h.get("cores", os.cpu_count() or 1)
                self.spin_cores.setValue(cores)
                
                # 加载基本参数设置
                self.config_params["max_results"] = h.get("max_results", CONFIG_DEFAULTS["max_results"])
                self.spin_max_results.setValue(self.config_params["max_results"])
                
                # 加载源和关键词历史
                for s in self.history["sources"]:
                    self.src_list.addItem(s)
                for kw in self.history["keywords"]:
                    self.keyword_combo.addItem(kw)
                    
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
        # --- 防抖：立即禁用按钮 --- 
        self.btn_analysis.setEnabled(False)
        # Ensure button is re-enabled in case of early return
        should_re_enable_button = True 
        try:
            if not self.config_path or not os.path.isfile(self.config_path):
                QMessageBox.warning(self, "提示", "请选择有效配置文件")
                # self.btn_analysis.setEnabled(True) # Re-enable handled by finally
                return
            if self.worker and self.worker.isRunning():
                QMessageBox.warning(self, "提示", "已有分析正在进行，请稍候")
                # self.btn_analysis.setEnabled(True) # Re-enable handled by finally
                return
    
            self.debug.clear()
            if hasattr(self, 'error_log'):
                self.error_log.clear()
                
            files = self.get_log_files()
            if not files:
                QMessageBox.warning(self, "提示", "无日志文件可分析")
                # self.btn_analysis.setEnabled(True) # Re-enable handled by finally
                return
    
            # 收集自定义关键词
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

            # 收集分组关键词
            group_kws = []
            for i in range(self.group_layout.count()):
                cb = self.group_layout.itemAt(i).widget()
                if isinstance(cb, QCheckBox) and cb.isChecked():
                    group_name = cb.property("group_name")
                    if not group_name:
                        self.debug.append(f"警告: 复选框 {cb.text()} 没有关联的 group_name")
                        continue
                    self.debug.append(f"处理选中的分组: {group_name}")

                    # 从配置文件加载分组数据
                    try:
                        config_data = toml.load(self.config_path)
                        self.debug.append(f"  配置文件加载成功.")

                        # 正确访问嵌套结构: config['group']['subgroup_key']
                        top_group_dict = config_data.get('group', None)
                        if top_group_dict is None or not isinstance(top_group_dict, dict):
                            self.debug.append(f"  错误: 配置文件缺少顶层 'group' 字典结构.")
                            continue

                        # 从 group_name (e.g., "group.security") 提取子分组键 ("security")
                        subgroup_key = group_name.split('.')[-1] if '.' in group_name else group_name

                        grp = top_group_dict.get(subgroup_key, None)

                        if grp is None:
                            self.debug.append(f"  警告: 在 'group' 字典下未找到子分组 '{subgroup_key}' (来自 {group_name})")
                            continue
                        
                        # Check if grp is actually a dictionary before proceeding
                        if not isinstance(grp, dict):
                            self.debug.append(f"  警告: 子分组 '{subgroup_key}' 的值不是字典 (类型: {type(grp)}). 跳过.")
                            continue
                            
                        self.debug.append(f"  成功获取分组数据: {grp}")

                        mc = grp.get("match_case", False)
                        ww = grp.get("whole_word", False)
                        uz = grp.get("use_regex", False)
                        color = self.group_colors.get(group_name, "#ffff99")
                        self.debug.append(f"  分组设置: match_case={mc}, whole_word={ww}, use_regex={uz}")

                        # 遍历分组中的所有条目
                        entries_count = 0
                        self.debug.append(f"  开始遍历分组 '{subgroup_key}' (来自 {group_name}) 中的条目...")
                        for k, v in grp.items():
                            self.debug.append(f"    检查条目: key='{k}', value='{v}' (类型: {type(v)})")
                            # 只处理值为字典且包含 'key' 键的条目 (即关键词定义)
                            if isinstance(v, dict) and "key" in v:
                                keyword_text = v["key"]
                                self.debug.append(f"      找到关键词定义: '{keyword_text}'")
                                entries_count += 1
                                group_kws.append({
                                    "raw": keyword_text,
                                    "annotation": v.get("annotation", ""),
                                    "match_case": mc, # 使用分组的设置
                                    "whole_word": ww,
                                    "use_regex": uz,
                                    "color": color
                                })
                            else:
                                self.debug.append(f"      跳过条目 '{k}' (非关键词定义)")

                        self.debug.append(f"  从分组 '{subgroup_key}' (来自 {group_name}) 中成功添加了 {entries_count} 个关键词")

                    except Exception as e:
                        self.debug.append(f"  加载或处理分组 '{group_name}' 时发生错误: {str(e)}")
                        logging.error(f"加载分组 '{group_name}' 失败: {e}")
                        continue # 跳过这个有问题的分组
    
            # 合并关键词
            all_kws = custom_kws + group_kws
            self.debug.append(f"DEBUG: Just before check, len(all_kws) = {len(all_kws)}")
    
            # 如果没有任何关键词，就提示并返回
            if not all_kws:
                self.debug.append("DEBUG: Entering 'if not all_kws' block.")
                QMessageBox.warning(self, "提示", "请勾选至少一个自定义关键词或分组关键词")
                self.debug.append("DEBUG: After QMessageBox.warning, before return.")
                # self.btn_analysis.setEnabled(True) # Re-enable handled by finally
                return
            else:
                 self.debug.append("DEBUG: Skipped 'if not all_kws' block.")
    
            # --- Keyword check passed, analysis will proceed --- 
            should_re_enable_button = False # Don't re-enable if analysis starts
            
            # 收集要显示的原始文本作为提示
            raw_list = [kw['raw'] for kw in all_kws]
            self.debug.append(f"共选择了 {len(all_kws)} 个关键词:")
            for kw in all_kws[:10]:  # 只显示前10个
                self.debug.append(f"- {kw['raw']}")
            if len(all_kws) > 10:
                self.debug.append(f"... 以及 {len(all_kws) - 10} 个其他关键词")
    
            # 如果已有工作线程，取消它
            if self.worker and self.worker.isRunning():
                self.cancel_analysis() # cancel_analysis should handle button state
                should_re_enable_button = True # Re-enable after cancel
                return 
    
            # 更新UI (Button already disabled)
            # self.btn_analysis.setEnabled(False) 
            self.btn_cancel.setVisible(True)
            self.progress.setVisible(True)
            
            # 确保UI状态正确更新
            QApplication.processEvents()
            
            # ... (rest of worker creation and starting logic)
            # 设置输出路径
            out_dir = CONFIG_DEFAULTS["output_dir"]
            for s in self.history["sources"]:
                if os.path.isdir(s):
                    out_dir = s
                    break
            if not out_dir and self.history["sources"]:
                out_dir = os.path.dirname(self.history["sources"][0])
            out_path = os.path.join(out_dir, CONFIG_DEFAULTS["output_filename"])

            # 创建工作线程
            max_workers = self.spin_cores.value()
            
            # 更新配置
            if hasattr(self, 'use_process_pool'):
                self.config_params["use_process_pool"] = self.use_process_pool.isChecked()
            
            # 创建工作线程
            self.worker = ScanWorker(
                file_paths=files,
                keywords=all_kws,
                out_path=out_path,
                max_workers=max_workers,
                config_params=self.config_params,
                temp_manager=self.temp_manager,
                parent=self
            )
            
            # 连接信号 (Ensure _on_scan_finished re-enables the button)
            self.worker.error.connect(self._on_scan_error) # error should re-enable
            self.worker.warning.connect(lambda msg: QMessageBox.warning(self, "结果限制", msg))
            self.worker.progress.connect(self._on_scan_progress)
            self.worker.debug.connect(lambda msg: self.debug.append(msg))
            self.worker.finished.connect(self._on_scan_finished) # finished should re-enable

            # 启动工作线程
            self.worker.start()
            self.debug.append("工作线程已启动")
            
        finally:
            # --- Re-enable button only if we returned early --- 
            if should_re_enable_button:
                self.btn_analysis.setEnabled(True)
                self.debug.append("DEBUG: Analysis button re-enabled due to early return.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = LogHighlighter()
    win.show()
    sys.exit(app.exec_())
