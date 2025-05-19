#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
扫描工作模块
负责日志文件的扫描、处理和结果生成。
"""

import os
import time
import logging
import datetime
import webbrowser
import gc
import mmap
import tempfile
import re
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from PyQt5.QtWidgets import QWidget, QMessageBox, QApplication
from PyQt5.QtCore import QThread, pyqtSignal
from keyword_matcher import KeywordMatcher
from file_handler import TempFileManager, ProgressMonitor, CheckpointManager
from memory_manager import MemoryMonitor
from utils import parse_timestamp, CONFIG_DEFAULTS

# 尝试导入 mmap 库，用于处理超大文件
try:
    import mmap
    MMAP_AVAILABLE = True
    logging.info("mmap 可用，将使用内存映射优化超大文件处理")
except ImportError:
    MMAP_AVAILABLE = False
    logging.warning("无法导入 mmap，超大文件将使用标准方式处理")


class ScanWorker(QThread):
    progress = pyqtSignal(str)  # 修改为单个字符串参数
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    warning = pyqtSignal(str)
    debug = pyqtSignal(str)  # 用于传递调试信息

    def __init__(self, file_paths: List[str], keywords: List[Dict[str, Any]], out_path: str, max_workers: int, config_params: Dict[str, Any], temp_manager: Optional[TempFileManager] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.file_paths = file_paths
        self.keywords = keywords  # 关键词信息列表 (仅包含GUI中已勾选的关键词)
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
        self.progress_monitor = ProgressMonitor(worker=self)
        self.checkpoint_manager = CheckpointManager()
        self.memory_monitor = None
        self.task_id = f"scan_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.parent_widget = parent  # 存储父窗口引用

        # 扫描模式和优化配置
        self.scan_mode = config_params.get("scan_mode", CONFIG_DEFAULTS["scan_mode"])
        self.large_file_threshold = config_params.get("large_file_threshold", CONFIG_DEFAULTS["large_file_threshold"])
        self.huge_file_threshold = config_params.get("huge_file_threshold", CONFIG_DEFAULTS["huge_file_threshold"])
        self.prefilter_enabled = config_params.get("prefilter_enabled", CONFIG_DEFAULTS["prefilter_enabled"])
        self.bitmap_filter_enabled = config_params.get("bitmap_filter_enabled", CONFIG_DEFAULTS["bitmap_filter_enabled"])

        # ---- START ADDED DEBUG ----
        logging.debug(f"[ScanWorker __init__] Received {len(keywords)} keyword definitions.")
        if keywords:
            for i, kw_def in enumerate(keywords[:5]): # Log first 5 keywords for brevity
                logging.debug(f"  KW {i}: {kw_def}")
            if len(keywords) > 5:
                logging.debug(f"  ... and {len(keywords) - 5} more keywords.")
        else:
            logging.debug("  Keyword list is empty.")
        # ---- END ADDED DEBUG ----

        # 使用优化的关键词匹配器，只使用勾选的关键词
        self.keyword_matcher = KeywordMatcher(keywords, use_bitmap_filter=self.bitmap_filter_enabled)

        if self.keyword_matcher.combined_pattern:
            compiled_pattern_for_debug = self.keyword_matcher.combined_pattern.pattern
            max_len = 500 # 每条消息的最大长度
            if len(compiled_pattern_for_debug) > max_len:
                logging.debug("[KeywordMatcher Compiled Pattern (chunked due to length)]:")
                for i in range(0, len(compiled_pattern_for_debug), max_len):
                    logging.debug(compiled_pattern_for_debug[i:i+max_len])
            else:
                logging.debug(f"[KeywordMatcher Compiled Pattern]: {compiled_pattern_for_debug}")
        else:
            logging.debug("[KeywordMatcher Compiled Pattern]: None (No patterns were compiled)")

        # 收集要匹配的所有原始关键词
        self.raw_list = [kw.get('raw', '') for kw in keywords]

        # 是否使用进程池
        self.use_process_pool = config_params.get("use_process_pool", False)

        # 结果缓存
        self._results_buffer = []
        self._results_buffer_size = 1000  # 每1000条结果写入一次
        self._debug_match_failure_count = 0 # Counter for match failure debugging
        self._max_debug_match_failures = 5   # Max times to emit detailed debug for match failures
        self._total_matches = 0 # 初始化总匹配数

    def _create_optimized_executor(self, worker_count: int):
        """
        根据配置选择使用线程池或进程池执行器。
        进程池适用于 CPU 密集型任务，而线程池适用于 I/O 密集型任务。
        """
        # 强制使用 ThreadPoolExecutor 以避免 pickle 问题
        from concurrent.futures import ThreadPoolExecutor
        self.debug.emit(f"强制使用 ThreadPoolExecutor，工作单元数: {worker_count}，忽略 use_process_pool 设置: {self.use_process_pool}")
        return ThreadPoolExecutor(max_workers=worker_count)

    def stop(self) -> None:
        """停止扫描任务并清理资源。"""
        self._stop_requested = True
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)

        # 保存检查点以便后续恢复
        self._save_checkpoint()

        # 不再需要停止内存监控，因为不再使用MemoryMonitor类
        self.debug.emit("已停止扫描任务")

    def _save_checkpoint(self) -> None:
        """保存当前处理状态到检查点"""
        try:
            checkpoint_data = {
                "processed_files": self._processed_files,
                "out_path": self.out_path,
                "timestamp": datetime.datetime.now().isoformat()
            }
            self.checkpoint_manager.save_checkpoint(self.task_id, checkpoint_data)
            self.debug.emit(f"已保存检查点: {self.task_id}")
        except Exception as e:
            logging.error(f"保存检查点失败: {e}")

    def scan_file(self, file_path: str) -> Tuple[str, List[Dict[str, str]]]:
        """
        扫描单个文件，根据文件大小选择不同的读取策略。
        对于大文件，使用 mmap 读取以减少内存占用。
        """
        if self._stop_requested:
            return os.path.basename(file_path), []

        matches = []
        try:
            file_size = os.path.getsize(file_path)
            use_mmap = MMAP_AVAILABLE and file_size > self.large_file_threshold
            
            if use_mmap:
                self.debug.emit(f"使用 mmap 读取大文件: {file_path}，大小: {file_size/(1024*1024):.2f} MB")
                with open(file_path, 'rb') as f:
                    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                    try:
                        for line in iter(mm.readline, b''):
                            if self._stop_requested:
                                break
                            decoded_line = line.decode('utf-8', errors='ignore')
                            self._process_line_with_matcher(decoded_line, matches, os.path.basename(file_path))
                            if len(matches) >= self.max_results:
                                self._result_truncated = True
                                self.warning.emit(f"文件 {file_path} 的结果已达到最大限制 {self.max_results}，后续结果将被截断。")
                                break
                    finally:
                        mm.close()
            else:
                logging.debug(f"使用标准读取方式: {file_path}，大小: {file_size/(1024*1024):.2f} MB")
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if self._stop_requested:
                            break
                        self._process_line_with_matcher(line, matches, os.path.basename(file_path))
                        if len(matches) >= self.max_results:
                            self._result_truncated = True
                            self.warning.emit(f"文件 {file_path} 的结果已达到最大限制 {self.max_results}，后续结果将被截断。")
                            break
            self.debug.emit(f"完成文件扫描: {file_path}, 找到 {len(matches)} 个匹配")
        except Exception as e:
            self.error.emit(f"扫描文件 {file_path} 出错: {str(e)}")
            logging.error(f"扫描文件 {file_path} 出错: {e}", exc_info=True)
        
        return os.path.basename(file_path), matches

    def run(self) -> None:
        """执行扫描过程"""
        start_time = time.time()
        try:
            self.progress.emit("正在初始化扫描...")
            self.debug.emit(f"开始扫描文件，使用 {len(self.file_paths)} 文件，{len(self.keywords)} 个关键词")
            
            for kw in self.keywords[:10]:  # 只打印前10个，避免日志过长
                self.debug.emit(f"要搜索的关键词: {kw.get('raw', '')}, 大小写敏感: {kw.get('match_case', False)}, 全词匹配: {kw.get('whole_word', False)}")

            # 初始化关键词匹配器
            self.keyword_matcher = KeywordMatcher(self.keywords, use_bitmap_filter=True)
            if self.keyword_matcher.combined_pattern:
                self.debug.emit(f"关键词匹配器模式: {self.keyword_matcher.combined_pattern.pattern}")
            else:
                self.debug.emit("关键词匹配器没有编译任何模式，可能是关键词列表为空")

            if not self.file_paths:
                self.debug.emit("没有要扫描的文件")
                return

            # 用于按时间范围分组的字典
            time_groups = {}
            
            # 检查文件大小总和
            total_size = 0
            for f in self.file_paths:
                if os.path.isfile(f):
                    total_size += os.path.getsize(f)
            self.debug.emit(f"总文件大小: {total_size/(1024*1024):.2f} MB")

            # 创建并启动内存监控
            # 注意：在线程中创建QObject可能导致线程错误
            # 改为使用非QObject的监控方式或在主线程中创建
            try:
                # 尝试使用简化版内存监控，避免QObject问题
                import psutil
                self.debug.emit("使用psutil监控内存使用情况")
                
                def get_memory_usage():
                    try:
                        process = psutil.Process(os.getpid())
                        return process.memory_info().rss / (1024 * 1024)  # MB
                    except:
                        return 0
                
                initial_memory = get_memory_usage()
                self.debug.emit(f"初始内存使用: {initial_memory:.1f}MB")
            except Exception as e:
                self.debug.emit(f"初始化内存监控失败: {str(e)}")

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
            # 使用合理的工作线程数
            worker_count = min(self.max_workers, os.cpu_count() or 1)
            self.debug.emit(f"开始分析 {len(self.file_paths)} 个文件，使用 {worker_count} 个工作单元")

            # 按照时间范围分组处理数据
            temp_output_files = {}
            try:
                # 创建适当的执行器（进程池或线程池）
                self._executor = self._create_optimized_executor(worker_count)

                # 准备进度报告
                total_files = len(self.file_paths)
                processed = 0
                skipped = 0
                self.progress.emit("0%")

                # 跟踪各个时间范围的数据
                time_groups: Dict[Tuple[datetime.datetime, datetime.datetime], List[Dict[str, str]]] = {}
                # 保存当前time_groups的引用，供_finalize_output_file使用
                self._current_time_groups = time_groups

                # 为了减少内存使用，分批处理文件
                batches = self._prepare_batched_tasks()

                # 分批处理文件
                for batch_idx, batch in enumerate(batches):
                    if self._stop_requested:
                        break

                    self.debug.emit(f"处理批次 {batch_idx + 1}/{len(batches)}，包含 {len(batch)} 个文件")

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
                                self._total_matches += len(matches) # 更新总匹配数

                                # 按照时间范围分组
                                for match_item in matches:
                                    # 从字典中获取时间戳
                                    timestamp_str = match_item.get('timestamp', '')
                                    try:
                                        ts = parse_timestamp(timestamp_str)
                                        start_ts = ts.replace(minute=0, second=0, microsecond=0, hour=ts.hour - (ts.hour % self.time_range_hours))
                                        end_ts = start_ts + datetime.timedelta(hours=self.time_range_hours)
                                        if (start_ts, end_ts) not in time_groups:
                                            time_groups[(start_ts, end_ts)] = []
                                        time_groups[(start_ts, end_ts)].append(match_item)  # 直接添加整个匹配项字典
                                    except Exception:
                                        # 如果时间解析失败，放入默认组
                                        default_ts = datetime.datetime.now()
                                        start_ts = default_ts.replace(hour=0, minute=0, second=0, microsecond=0)
                                        end_ts = start_ts + datetime.timedelta(days=1)
                                        if (start_ts, end_ts) not in time_groups:
                                            time_groups[(start_ts, end_ts)] = []
                                        time_groups[(start_ts, end_ts)].append(match_item)  # 直接添加整个匹配项字典

                            else:
                                skipped += 1

                            # 更新进度
                            if processed % self._batch_update_size == 0 or processed == total_files:
                                progress_pct = min(int((processed / total_files) * 100), 100)
                                self.progress.emit(f"{progress_pct}% ({processed}/{total_files})")

                                # 定期保存检查点
                                self._processed_files = processed
                                if processed % 50 == 0:
                                    self._save_checkpoint()

                                    # 尝试释放内存，优化垃圾回收策略
                                    if processed % 100 == 0:
                                        try:
                                            import psutil
                                            process = psutil.Process(os.getpid())
                                            mem_info = process.memory_info()
                                            mem_usage_mb = mem_info.rss / (1024 * 1024)
                                            self.debug.emit(f"内存使用: {mem_usage_mb:.1f}MB")
                                            
                                            # 只有当内存使用超过1.5GB时才进行垃圾回收，减少频繁调用
                                            if mem_usage_mb > 1536:
                                                self.debug.emit("内存使用超过1.5GB，进行垃圾回收并尝试释放更多内存...")
                                                gc.collect()
                                                self._flush_results_for_group(time_groups)
                                        except:
                                            pass
                        except Exception as e:
                            logging.error(f"处理文件 {file_path} 失败: {e}")
                            self.progress_monitor.record_error("文件处理", f"{file_path}: {str(e)}")

                    # 每完成一个批次，检查内存使用情况并分批写入结果
                    if sum(len(group) for group in time_groups.values()) > self._results_buffer_size:
                        self._flush_results_for_group(time_groups)
                    gc.collect()

                # 处理结果
                if not self._stop_requested and time_groups:
                    self.progress.emit("正在生成输出文件...")

                    # 直接生成单一结果文件而不是摘要
                    all_results = []
                    for time_range, results in time_groups.items():
                        all_results.extend(results)
                    
                    # 按时间顺序排序所有结果
                    sorted_results = sorted(all_results, key=lambda x: x.get('timestamp', ''))
                    
                    # 生成单一报告
                    result_file = self._generate_full_results_report(sorted_results)
                    
                    # 计算总耗时
                    duration = time.time() - start_time
                    self.debug.emit(f"扫描完成，用时: {duration:.2f} 秒")
                    
                    if self._result_truncated:
                        self.warning.emit(f"部分结果因超出限制被截断。建议增加 '最大结果数' 设置或分割日志文件。")
                    
                    if not self._stop_requested and result_file:
                        if os.path.exists(result_file):
                            self.debug.emit(f"使用结果文件: {result_file}")
                            self.progress.emit("已完成")
                            self.finished.emit(result_file)
                            return
                
                # 如果没有生成任何结果文件或者过程被中断
                self.debug.emit("未能生成有效的结果文件")
                self.progress.emit("已完成")
                self.finished.emit("")

            except Exception as e:
                self.error.emit(f"扫描过程中发生错误: {str(e)}")
                logging.error(f"扫描过程中发生错误: {e}", exc_info=True)
            finally:
                # 清理资源
                if self._executor:
                    self._executor.shutdown()

                # 计算总耗时
                duration = time.time() - start_time
                self.debug.emit(f"扫描完成，用时: {duration:.2f} 秒")

                if self._result_truncated:
                    self.warning.emit(f"部分结果因超出限制被截断。建议增加 '最大结果数' 设置或分割日志文件。")

                if not self._stop_requested:
                    # 打开生成的结果文件
                    if os.path.exists(f"{self.out_path}_results.html"):
                        result_file = f"{self.out_path}_results.html"
                        self.debug.emit(f"正在打开结果文件: {result_file}")
                        
                        # 确保使用完整的绝对路径和file://协议
                        try:
                            abs_path = os.path.abspath(result_file)
                            file_url = f"file://{abs_path}"
                            self.debug.emit(f"使用URL打开文件: {file_url}")
                            self.debug.emit(f"文件是否存在: {os.path.exists(abs_path)}")
                            self.debug.emit(f"文件大小: {os.path.getsize(abs_path) / 1024:.1f} KB")
                            
                            # 不在线程中尝试打开浏览器，而是通过信号让主线程处理
                            self.debug.emit("将结果文件路径发送到主线程进行处理")
                            self.progress.emit("已完成")
                            self.finished.emit(result_file)
                            return
                        except Exception as e:
                            self.debug.emit(f"准备打开文件时出错: {str(e)}")

                # 如果前面的逻辑未返回
                self.progress.emit("已完成")
                self.finished.emit("")

        except Exception as e:
            self.error.emit(f"扫描过程中发生错误: {str(e)}")
            logging.error(f"扫描过程中发生错误: {e}", exc_info=True)
        finally:
            # 清理资源
            if self._executor:
                self._executor.shutdown()

            # 计算总耗时
            duration = time.time() - start_time
            self.debug.emit(f"扫描完成，用时: {duration:.2f} 秒")

            if self._result_truncated:
                self.warning.emit(f"部分结果因超出限制被截断。建议增加 '最大结果数' 设置或分割日志文件。")

            if not self._stop_requested:
                # 打开生成的结果文件
                if os.path.exists(f"{self.out_path}_results.html"):
                    result_file = f"{self.out_path}_results.html"
                    self.debug.emit(f"正在打开结果文件: {result_file}")
                    
                    # 确保使用完整的绝对路径和file://协议
                    try:
                        abs_path = os.path.abspath(result_file)
                        file_url = f"file://{abs_path}"
                        self.debug.emit(f"使用URL打开文件: {file_url}")
                        self.debug.emit(f"文件是否存在: {os.path.exists(abs_path)}")
                        self.debug.emit(f"文件大小: {os.path.getsize(abs_path) / 1024:.1f} KB")
                        
                        # 不在线程中尝试打开浏览器，而是通过信号让主线程处理
                        self.debug.emit("将结果文件路径发送到主线程进行处理")
                        self.progress.emit("已完成")
                        self.finished.emit(result_file)
                        return
                    except Exception as e:
                        self.debug.emit(f"准备打开文件时出错: {str(e)}")

            # 如果前面的逻辑未返回
            self.progress.emit("已完成")
            self.finished.emit("")

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
            temp_path, temp_file = self.temp_manager.create_temp_file(prefix=prefix, suffix=suffix, time_range=(start_ts, end_ts))
            
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

    def _process_line_with_matcher(self, line: str, out: List[Dict[str, str]], source_filename: str) -> None:
        """使用KeywordMatcher处理一行文本，并将包含关键词等信息的字典添加到out列表"""
        if self._stop_requested:
            return

        original_line = line.rstrip('\n') # Keep original for debug
        line_to_process = original_line

        # 使用KeywordMatcher进行高效匹配
        has_match, highlighted, matched_keyword = self.keyword_matcher.highlight_line(line_to_process)

        if has_match:
            # 精确匹配检查 - 只处理用户勾选的关键词
            exact_match = False
            case_insensitive_match = False
            for kw in self.keywords:
                raw_keyword = kw.get('raw', '')
                match_case = kw.get('match_case', False)

                if match_case:  # 区分大小写
                    if raw_keyword == matched_keyword:
                        exact_match = True
                        break
                else:  # 不区分大小写
                    if raw_keyword.upper() == matched_keyword.upper():
                        case_insensitive_match = True
                        break

            self.debug.emit(f"精确匹配结果: {exact_match}, 不区分大小写匹配结果: {case_insensitive_match}")

            # 只有当找到匹配的关键词在用户选择的列表中时，才添加到结果中
            if exact_match or case_insensitive_match:
                ts_str = original_line[:19] if len(original_line) >= 19 and original_line[4] == '-' and original_line[7] == '-' else "unknown_time"
                # 尝试从行首提取更标准的时间戳，如果匹配特定格式
                match = re.match(r"^(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[,.]\d{3,6})?Z?)", original_line)
                if match:
                    ts_str = match.group(1)

                out.append({
                    'timestamp': ts_str,
                    'highlighted_line': highlighted,
                    'keyword': matched_keyword if matched_keyword else '未知',
                    'source_file': source_filename,
                    'original_line': original_line # 添加原始行用于调试或未来功能
                })
            else:
                # 如果关键词不在用户选择的列表中，则忽略此匹配
                self.debug.emit(f"忽略匹配项: '{matched_keyword}' 不在用户选择的关键词列表中")

    def _generate_full_results_report(self, results: List[Dict]) -> str:
        """生成包含所有匹配结果的单一报告，按时间顺序排序"""
        result_file = f"{self.out_path}_results.html"
        logging.debug(f"正在生成完整结果报告: {result_file}")
        
        try:
            with open(result_file, 'w', encoding='utf-8') as f:
                self._write_html_header(f, len(results))
                
                # 写入匹配结果
                if not results:
                    self._write_no_results_message(f)
                else:
                    self._write_match_results(f, results)

                # 添加简化的JavaScript用于过滤功能
                self._write_filter_javascript(f)
                
                # 写入HTML尾部
                f.write('</div>\n')
                f.write('</body>\n')
                f.write('</html>\n')

            logging.debug(f"完整结果报告生成成功: {result_file}")
            return result_file
        except Exception as e:
            self.error.emit(f"生成完整结果报告失败: {str(e)}")
            return ""

    def _finalize_output_file(self, temp_file: str, start_time: datetime.datetime, end_time: datetime.datetime) -> Optional[str]:
        """完成输出文件，返回最终的文件路径"""
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write('<!DOCTYPE html>\n<html>\n<head>\n')
                f.write('<meta charset="utf-8">\n')
                f.write('<meta name="viewport" content="width=device-width, initial-scale=1.0">\n')
                f.write(f'<title>日志分析结果</title>\n')
                f.write('<style>\n')
                f.write('body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }\n')
                f.write('h1, h2, h3 { color: #333; }\n')
                f.write('pre { background-color: #f5f5f5; padding: 10px; border-radius: 5px; white-space: pre-wrap; word-wrap: break-word; }\n')
                f.write('.timestamp { color: #666; font-size: 0.9em; margin-right: 10px; }\n')
                f.write('.source-file { color: #0066cc; margin-left: 10px; }\n')
                f.write('.match-item { margin-bottom: 20px; border-bottom: 1px solid #ddd; padding-bottom: 10px; }\n')
                f.write('.error-highlight { background-color: #ffdddd; }\n')
                f.write('.warning-highlight { background-color: #ffffcc; }\n')
                f.write('.info-highlight { background-color: #ddffdd; }\n')
                f.write('.debug-highlight { background-color: #ddddff; }\n')
                f.write('.back-link { margin-bottom: 20px; }\n')
                
                # 添加颜色样式
                for keyword_meta in self.keywords:
                    keyword = keyword_meta.get('raw', '')
                    color = keyword_meta.get('color', '#ffff99')
                    f.write(f'.highlight-{hash(keyword) % 10000} {{ background-color: {color}; }}\n')
                
                f.write('</style>\n')
                f.write('</head>\n<body>\n')
                
                # 添加返回摘要的链接
                summary_path = f"{self.out_path}_summary.html"
                summary_filename = os.path.basename(summary_path)
                f.write(f'<div class="back-link"><a href="{summary_filename}">返回摘要</a></div>\n')
                
                # 添加标题和时间范围信息
                f.write('<h1>日志分析结果</h1>\n')
                
                start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S") if isinstance(start_time, datetime.datetime) else str(start_time)
                end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S") if isinstance(end_time, datetime.datetime) else str(end_time)
                
                f.write(f'<p>时间范围: {start_time_str} - {end_time_str}</p>\n')
                
                # 从时间组中查找相应的结果
                time_range_tuple = (start_time, end_time)
                
                # 查找这个时间范围的匹配记录
                matching_results = []
                if hasattr(self, 'time_groups') and isinstance(self.time_groups, dict):
                    matching_results = self.time_groups.get(time_range_tuple, [])
                
                # 写入结果到HTML中
                f.write('<h2>匹配结果</h2>\n')
                
                # 查找此范围在当前执行的time_groups中的记录
                # 由于我们传入的temp_file已经包含了时间信息，尝试匹配
                current_results = []
                for (range_start, range_end), results_group in self._current_time_groups.items():
                    if abs((range_start - start_time).total_seconds()) < 60 and abs((range_end - end_time).total_seconds()) < 60:
                        current_results = results_group
                        break
                
                if current_results:
                    f.write(f'<p>找到 {len(current_results)} 条匹配记录</p>\n')
                    
                    # 创建关键词统计
                    keyword_counts = {}
                    source_file_counts = {}
                    for result in current_results:
                        keyword = result.get('keyword', '未知')
                        source_file = result.get('source_file', '')
                        keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
                        source_file_counts[source_file] = source_file_counts.get(source_file, 0) + 1
                    
                    # 写入统计信息
                    f.write('<div class="stats">\n')
                    f.write('<h3>关键词统计</h3>\n')
                    f.write('<table border="1" cellpadding="5">\n')
                    f.write('<tr><th>关键词</th><th>出现次数</th></tr>\n')
                    
                    for kw, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True):
                        f.write(f'<tr><td>{kw}</td><td>{count}</td></tr>\n')
                    
                    f.write('</table>\n</div>\n')
                    
                    # 写入来源文件统计
                    if len(source_file_counts) > 1:  # 如果有多个来源文件
                        f.write('<div class="stats">\n')
                        f.write('<h3>来源文件统计</h3>\n')
                        f.write('<table border="1" cellpadding="5">\n')
                        f.write('<tr><th>文件名</th><th>匹配数</th></tr>\n')
                        
                        for source_file_val, count in sorted(source_file_counts.items(), key=lambda x: x[1], reverse=True):
                            f.write(f'<tr><td>{source_file_val}</td><td>{count}</td></tr>\n')
                        
                        f.write('</table>\n</div>\n')
                    
                    # 写入具体匹配行
                    f.write('<div class="results">\n')
                    f.write('<h3>匹配详情</h3>\n')
                    
                    # 按时间排序
                    sorted_results = sorted(current_results, key=lambda x: x.get('timestamp', ''))
                    
                    for result in sorted_results:
                        timestamp = result.get('timestamp', '')
                        highlighted_line = result.get('highlighted_line', '')
                        source_file = result.get('source_file', '')
                        keyword = result.get('keyword', '')
                        
                        # 计算CSS类名
                        css_class = f"highlight-{hash(keyword) % 10000}"
                        
                        f.write('<div class="match-item">\n')
                        f.write(f'<p><span class="timestamp">[{timestamp}]</span> <span class="source-file">文件: {source_file}</span></p>\n')
                        f.write(f'<div class="{css_class}">{highlighted_line}</div>\n')
                        f.write('</div>\n')
                    
                    f.write('</div>\n')
                    
                else:
                    f.write('<p>该时间范围内没有发现匹配记录</p>\n')
                
                # 写入HTML尾部
                f.write('</body>\n</html>\n')
            
            return temp_file
        except Exception as e:
            self.error.emit(f"完成输出文件失败: {str(e)}")
            return None

    def _prepare_batched_tasks(self) -> List[List[str]]:
        """准备分批处理的任务"""
        batches = []
        current_batch = []
        current_size = 0

        for file_path in self.file_paths:
            if os.path.isfile(file_path):
                file_size = os.path.getsize(file_path)
                if file_size > self.max_file_size:
                    self.warning.emit(f"文件 {file_path} 超过最大文件大小限制，将被忽略")
                    continue
                if current_size + file_size > self.max_file_size:
                    batches.append(current_batch)
                    current_batch = [file_path]
                    current_size = file_size
                else:
                    current_batch.append(file_path)
                    current_size += file_size
            else:
                self.warning.emit(f"文件 {file_path} 不存在，将被忽略")

        if current_batch:
            batches.append(current_batch)

        return batches

    def _write_html_header(self, f, total_results: int) -> None:
        """写入HTML头部，包括样式和统计信息"""
        f.write('<!DOCTYPE html>\n')
        f.write('<html>\n')
        f.write('<head>\n')
        f.write('    <meta charset="utf-8">\n')
        f.write('    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n')
        f.write('    <title>日志分析结果</title>\n')
        f.write('    <style>\n')
        f.write('        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }\n')
        f.write('        h1, h2 { color: #333; }\n')
        f.write('        .match-item { margin-bottom: 10px; border-bottom: 1px solid #eee; padding-bottom: 10px; }\n')
        f.write('        .timestamp { color: #666; font-size: 0.9em; }\n')
        f.write('        .source-file { color: #0066cc; margin-left: 10px; }\n')
        f.write('        pre { background-color: #f9f9f9; padding: 10px; border-radius: 3px; white-space: pre-wrap; word-wrap: break-word; }\n')
        f.write('        .filter-controls { margin: 15px 0; padding: 15px; background-color: #f5f5f5; border-radius: 3px; }\n')
        f.write('        .filter-controls input, .filter-controls select { padding: 5px; margin-right: 10px; }\n')
        f.write('        .filter-controls button { padding: 5px 10px; background-color: #4CAF50; color: white; border: none; border-radius: 3px; cursor: pointer; }\n')
        f.write('        .filter-controls button:hover { background-color: #45a049; }\n')
        f.write('        .stats { margin: 15px 0; }\n')
        f.write('        .keyword-tag { margin-left: 10px; color: #009688; font-size: 0.9em; }\n')
        f.write('    </style>\n')
        f.write('</head>\n')
        f.write('<body>\n')
        f.write('    <h1>日志分析结果</h1>\n')
        f.write('    <div class="stats">\n')
        f.write('        <p>分析文件数: <strong>%s</strong>, 匹配行数: <strong>%s</strong></p>\n' % (str(self._processed_files), str(total_results)))
        f.write('        <p>生成时间: <strong>%s</strong></p>\n' % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        f.write('    </div>\n')
        f.write('    <div class="filter-controls">\n')
        f.write('        <div>\n')
        f.write('            <input type="text" id="filterText" placeholder="输入文本过滤..." oninput="filterResults()">\n')
        f.write('            <select id="filterSource" onchange="filterResults()">\n')
        f.write('                <option value="">全部文件</option>\n')
        f.write('            </select>\n')
        f.write('            <button onclick="resetFilters()">重置过滤</button>\n')
        f.write('        </div>\n')
        f.write('    </div>\n')
        f.write('    <h2>匹配结果 <span id="visible-count">0</span>/<span id="total-count">%s</span></h2>\n' % str(total_results))
        f.write('    <div id="results">\n')

    def _write_no_results_message(self, f) -> None:
        """写入无匹配结果的提示信息"""
        selected_keywords = [kw.get('raw', '') for kw in self.keywords]
        keyword_items = []
        for keyword in selected_keywords:
            keyword_items.append("        <li>%s</li>" % keyword)
        keyword_list_html = "\n".join(keyword_items)
        
        f.write('<div class="no-results" style="text-align: center; padding: 50px; font-size: 18px; color: #666;">\n')
        f.write('    <p>未找到匹配的关键词。</p>\n')
        f.write('    <p>您勾选了以下关键词：</p>\n')
        f.write('    <ul style="display: inline-block; text-align: left;">\n')
        f.write('%s\n' % keyword_list_html)
        f.write('    </ul>\n')
        f.write('    <p>但在日志文件中没有找到匹配项。</p>\n')
        f.write('    <p>请返回主界面，检查关键词选择或尝试分析其他日志文件。</p>\n')
        f.write('</div>\n')

    def _write_match_results(self, f, results: List[Dict]) -> None:
        """写入匹配结果的HTML内容"""
        # 有匹配结果时展示详细内容
        filtered_results = []
        selected_keywords = set(kw.get('raw', '').lower() for kw in self.keywords)
        for result in results:
            keyword = result.get('keyword', '').lower()
            if any(kw in keyword for kw in selected_keywords):
                filtered_results.append(result)
        
        for i, result in enumerate(filtered_results):
            timestamp = result.get('timestamp', '')
            highlighted_line = result.get('highlighted_line', '')
            source_file = result.get('source_file', '')
            keyword = result.get('keyword', '')
            
            f.write('<div class="match-item" data-source="%s" data-keyword="%s">\n' % (source_file, keyword))
            f.write('    <div>%s</div>\n' % highlighted_line)
            f.write('</div>\n')

    def _write_filter_javascript(self, f) -> None:
        """写入用于过滤结果的JavaScript代码"""
        f.write('</div>\n\n')
        f.write('<script>\n')
        f.write('// 初始化过滤源下拉列表\n')
        f.write('function initSourceFilter() {\n')
        f.write('    const sourceSelect = document.getElementById("filterSource");\n')
        f.write('    const sources = new Set();\n')
        f.write('    \n')
        f.write('    document.querySelectorAll(".match-item").forEach(function(item) {\n')
        f.write('        sources.add(item.getAttribute("data-source"));\n')
        f.write('    });\n')
        f.write('    \n')
        f.write('    Array.from(sources).sort().forEach(function(source) {\n')
        f.write('        const option = document.createElement("option");\n')
        f.write('        option.value = source;\n')
        f.write('        option.textContent = source;\n')
        f.write('        sourceSelect.appendChild(option);\n')
        f.write('    });\n')
        f.write('    \n')
        f.write('    // 初始过滤一次，确保正确计数\n')
        f.write('    filterResults();\n')
        f.write('}\n')
        f.write('\n')
        f.write('// 过滤结果\n')
        f.write('function filterResults() {\n')
        f.write('    const filterText = document.getElementById("filterText").value.toLowerCase();\n')
        f.write('    const filterSource = document.getElementById("filterSource").value;\n')
        f.write('    \n')
        f.write('    var visibleCount = 0;\n')
        f.write('    document.querySelectorAll(".match-item").forEach(function(item) {\n')
        f.write('        const source = item.getAttribute("data-source");\n')
        f.write('        const content = item.textContent.toLowerCase();\n')
        f.write('        \n')
        f.write('        const sourceMatch = filterSource === "" || source === filterSource;\n')
        f.write('        const textMatch = filterText === "" || content.indexOf(filterText) !== -1;\n')
        f.write('        \n')
        f.write('        const isVisible = sourceMatch && textMatch;\n')
        f.write('        item.style.display = isVisible ? "block" : "none";\n')
        f.write('        \n')
        f.write('        if (isVisible) visibleCount++;\n')
        f.write('    });\n')
        f.write('    \n')
        f.write('    // 更新可见计数\n')
        f.write('    document.getElementById("visible-count").textContent = visibleCount;\n')
        f.write('}\n')
        f.write('\n')
        f.write('// 重置过滤\n')
        f.write('function resetFilters() {\n')
        f.write('    document.getElementById("filterText").value = "";\n')
        f.write('    document.getElementById("filterSource").value = "";\n')
        f.write('    filterResults();\n')
        f.write('}\n')
        f.write('\n')
        f.write('// 页面加载完成后初始化\n')
        f.write('window.onload = function() {\n')
        f.write('    initSourceFilter();\n')
        f.write('};\n')
        f.write('</script>\n')
        f.write('</body>\n')
        f.write('</html>\n')