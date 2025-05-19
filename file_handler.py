#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件处理模块
负责临时文件管理、进度监控和断点续传功能。
"""

import os
import re
import json
import logging
import time
import tempfile
import platform
import shutil
from typing import Dict, List, Optional, Tuple, TextIO, IO, Any
import datetime


class TempFileManager:
    """临时文件管理器，负责创建、跟踪和清理临时文件"""

    def __init__(self):
        self.temp_files = {}  # 路径 -> {"file_obj": 文件对象, "metadata": 元数据}
        # 使用os.path.join确保跨平台路径兼容性
        self.temp_dir = os.path.join(tempfile.gettempdir(), "log_highlighter_temp")
        os.makedirs(self.temp_dir, exist_ok=True)

        # 记录平台信息，以便于调整特定平台的行为
        self.is_windows = platform.system() == "Windows"
        self.is_mac = platform.system() == "Darwin"
        self.is_linux = platform.system() == "Linux"
        logging.info(f"临时文件目录: {self.temp_dir}")

    def create_temp_file(self, prefix: str = "log_hl_", suffix: str = ".tmp", 
                        time_range: Optional[Tuple[datetime.datetime, datetime.datetime]] = None) -> Tuple[str, IO]:
        """
        创建临时文件
        
        Args:
            prefix: 文件名前缀
            suffix: 文件名后缀
            time_range: 时间范围元组 (开始时间, 结束时间)
            
        Returns:
            临时文件路径和文件对象的元组
        """
        # 使用指定的前缀和后缀创建临时文件
        fd, temp_path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=self.temp_dir)
        
        # 关闭文件描述符，以便后续以文本模式打开
        os.close(fd)
        
        # 以文本模式打开文件
        temp_file = open(temp_path, 'w', encoding='utf-8')
        
        # 存储文件对象以便后续关闭，确保元数据字典存在
        self.temp_files[temp_path] = {"file_obj": temp_file, "metadata": {}}
        
        # 如果提供了时间范围，将其添加到文件元数据
        if time_range:
            start_time, end_time = time_range
            start_str = start_time.strftime("%Y-%m-%d %H:%M:%S") if isinstance(start_time, datetime.datetime) else str(start_time)
            end_str = end_time.strftime("%Y-%m-%d %H:%M:%S") if isinstance(end_time, datetime.datetime) else str(end_time)
            temp_file.write(f"<!-- 时间范围: {start_str} - {end_str} -->\n")
        
        return temp_path, temp_file

    def close_file(self, path: str) -> None:
        """安全关闭临时文件"""
        if path in self.temp_files:
            file_info = self.temp_files[path]
            if isinstance(file_info, dict) and "file_obj" in file_info:
                file_obj = file_info["file_obj"]
                try:
                    if file_obj and not file_obj.closed:
                        file_obj.close()
                except Exception as e:
                    logging.error(f"关闭临时文件 {path} 失败: {e}")
            elif hasattr(file_info, 'closed') and hasattr(file_info, 'close'): # Fallback for older direct assignment
                try:
                    if not file_info.closed:
                        file_info.close()
                except Exception as e:
                    logging.error(f"关闭临时文件 {path} (direct_obj) 失败: {e}")

    def remove_file(self, path: str) -> bool:
        """删除临时文件并从跟踪中移除"""
        self.close_file(path)
        try:
            if os.path.exists(path):
                # 在Windows上，文件可能被占用，所以增加重试逻辑
                if self.is_windows:
                    retry_count = 3
                    while retry_count > 0:
                        try:
                            os.remove(path)
                            break
                        except PermissionError:
                            retry_count -= 1
                            time.sleep(0.1)
                            if retry_count == 0:
                                logging.warning(f"Windows上无法删除文件 {path}，可能被其他进程占用")
                                return False
                else:
                    # 非Windows平台直接删除
                    os.remove(path)
            
            if path in self.temp_files:
                del self.temp_files[path]
            return True
        except Exception as e:
            logging.error(f"删除临时文件 {path} 失败: {e}")
            return False

    def cleanup_all(self) -> None:
        """清理所有临时文件和目录"""
        # 首先关闭所有打开的文件句柄
        for path in list(self.temp_files.keys()):
            self.close_file(path) # Ensure file is closed before attempting removal

        # 清理单个临时文件
        for path in list(self.temp_files.keys()):
            self.remove_file(path)

        # 尝试清理整个临时目录，包括子目录
        # 特别处理 extracted_archives 目录
        extracted_archives_dir = os.path.join(self.temp_dir, "extracted_archives")
        if os.path.exists(extracted_archives_dir):
            try:
                shutil.rmtree(extracted_archives_dir)
                logging.info(f"已成功删除目录: {extracted_archives_dir}")
            except Exception as e:
                logging.warning(f"删除目录 {extracted_archives_dir} 失败: {e}. 可能需要手动清理。")

        # 清理临时目录中的其他剩余文件 (如果temp_dir本身还存在)
        if os.path.exists(self.temp_dir):
            try:
                # 再次列出temp_dir中剩余的内容，确保是在所有tracked files删除尝试之后
                remaining_items = os.listdir(self.temp_dir)
                
                # 先尝试删除所有已知的非目录文件
                for item_name in list(remaining_items): # Iterate over a copy
                    item_path = os.path.join(self.temp_dir, item_name)
                    if os.path.isfile(item_path):
                        try:
                            os.remove(item_path)
                            remaining_items.remove(item_name) # Update list if successful
                            logging.info(f"已清理未被追踪的临时文件: {item_path}")
                        except Exception as e:
                            logging.warning(f"清理未被追踪的临时文件 {item_path} 失败: {e}")
                
                # 现在检查目录是否为空
                if not remaining_items: # 如果目录为空 (或者只剩下无法删除的目录比如extracted_archives如果删除失败)
                    # Check if extracted_archives_dir still exists and is the only thing left
                    if os.path.exists(extracted_archives_dir) and remaining_items == [os.path.basename(extracted_archives_dir)]:
                        logging.info(f"临时目录 {self.temp_dir} 中仅剩 {os.path.basename(extracted_archives_dir)}，将不删除主临时目录。")
                    else:
                        try:
                            os.rmdir(self.temp_dir)
                            logging.info(f"已成功删除空临时目录: {self.temp_dir}")
                        except Exception as e:
                            logging.warning(f"尝试删除空临时目录 {self.temp_dir} 失败 (可能仍有内容或权限问题): {e}")
                else: # 如果目录不为空
                    logging.info(f"临时目录 {self.temp_dir} 中仍有未清理项: {remaining_items}. 可能需要手动清理。")
            except Exception as e:
                logging.error(f"清理临时目录 {self.temp_dir} 时发生错误: {e}")


class ProgressMonitor:
    """跟踪扫描任务的进度和错误"""

    def __init__(self, worker=None):
        """
        初始化进度监控器

        Args:
            worker: 关联的工作线程，用于发送进度信号
        """
        self.worker = worker
        self.start_time = datetime.datetime.now()
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
        current_time = datetime.datetime.now()
        if current_time - self.last_updated < datetime.timedelta(seconds=0.5):
            return

        self.last_updated = current_time

        # 如果有worker，发送进度信号
        if self.worker and hasattr(self.worker, 'progress'):
            if self.total_files > 0:
                percent = min(int(processed / self.total_files * 100), 100)
                self.worker.progress.emit(
                    f"{percent}% ({processed}/{self.total_files}) - {current_file}")
            else:
                self.worker.progress.emit(f"已处理 {processed} 个文件 - {current_file}")

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
        duration = (datetime.datetime.now() - self.start_time).total_seconds()
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

    def save_checkpoint(self, task_id: str, data: Dict[str, Any]) -> bool:
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
                json.dump(data, f, ensure_ascii=False, default=str)
            return True
        except Exception as e:
            logging.error(f"保存检查点失败: {e}")
            return False

    def load_checkpoint(self, task_id: str) -> Optional[Dict[str, Any]]:
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
        cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=max_age_hours)

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
