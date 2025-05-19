#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
通用工具函数和配置常量模块
提供日志高亮工具的通用功能和配置参数，供其他模块使用。
"""

import html
import colorsys
import datetime
import logging
import os
import re

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

def highlight_line(line: str, regex: 're.Pattern',
                   mapping: dict) -> str:
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

def parse_timestamp(line: str, custom_formats: list = None) -> datetime.datetime:
    """
    从日志行中解析时间戳，支持多种常见格式。
    
    支持的格式包括:
    - YYYY-MM-DD HH:MM:SS.sss
    - YYYY-MM-DD HH:MM:SS
    - YYYY/MM/DD HH:MM:SS
    - DD-MM-YYYY HH:MM:SS (欧洲格式)
    - DD/MM/YYYY HH:MM:SS (欧洲格式)
    - MM-DD HH:MM:SS.sss
    - MM-DD HH:MM:SS
    - HH:MM:SS.sss
    - HH:MM:SS
    - Unix时间戳 (秒或毫秒)
    - 格式化的ISO 8601时间戳 (包括各种时区格式)
    
    Args:
        line: 日志行
        custom_formats: 用户自定义的时间戳格式列表，每项为(pattern, format_strings)元组
        
    Returns:
        解析后的时间戳，如果无法解析则返回None
    """
    if not line or len(line) < 8:  # 时间戳最短也需要8个字符 (HH:MM:SS)
        return None
        
    # 当前年份，用于不包含年份的时间戳
    current_year = datetime.datetime.now().year
    current_date = datetime.datetime.now().date()
    
    # 尝试多种格式
    timestamp_formats = [
        # 完整日期时间格式 (YYYY-MM-DD或YYYY/MM/DD)
        (r'(\d{4}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)', [
            "%Y-%m-%d %H:%M:%S.%f", 
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S.%f",
            "%Y/%m/%d %H:%M:%S"
        ]),
        # 欧洲格式日期时间 (DD-MM-YYYY或DD/MM/YYYY)
        (r'(\d{2}[-/]\d{2}[-/]\d{4}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)', [
            "%d-%m-%Y %H:%M:%S.%f",
            "%d-%m-%Y %H:%M:%S",
            "%d/%m/%Y %H:%M:%S.%f",
            "%d/%m/%Y %H:%M:%S"
        ]),
        # 点分隔的日期格式 (DD.MM.YYYY)
        (r'(\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)', [
            "%d.%m.%Y %H:%M:%S.%f",
            "%d.%m.%Y %H:%M:%S",
        ]),
        # 月日时间格式
        (r'(\d{2}[-/]\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)', [
            "%m-%d %H:%M:%S.%f",
            "%m-%d %H:%M:%S",
            "%m/%d %H:%M:%S.%f",
            "%m/%d %H:%M:%S"
        ]),
        # 仅时间格式
        (r'(\d{2}:\d{2}:\d{2}(?:\.\d+)?)', [
            "%H:%M:%S.%f",
            "%H:%M:%S"
        ]),
        # ISO 8601格式 (带或不带毫秒和时区)
        (r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)', [
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z"
        ]),
        # 其他常见日志格式 (如Apache日志)
        (r'\[(\d{2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2} [+-]\d{4})\]', [
            "%d/%b/%Y:%H:%M:%S %z"
        ]),
        # Unix时间戳 (秒) - 10位数字
        (r'(\b\d{10}\b)', []),
        # Unix时间戳 (毫秒) - 13位数字
        (r'(\b\d{13}\b)', [])
    ]
    
    # 添加自定义格式
    if custom_formats:
        timestamp_formats.extend(custom_formats)
    
    # 遍历所有格式进行匹配
    for pattern, formats in timestamp_formats:
        match = re.search(pattern, line)
        if match:
            ts_str = match.group(1)
            
            # Unix时间戳特殊处理
            if pattern == r'(\b\d{10}\b)':
                try:
                    return datetime.datetime.fromtimestamp(int(ts_str))
                except (ValueError, OverflowError):
                    continue
            elif pattern == r'(\b\d{13}\b)':
                try:
                    return datetime.datetime.fromtimestamp(int(ts_str) / 1000)
                except (ValueError, OverflowError):
                    continue
            
            # 尝试标准格式
            for fmt in formats:
                try:
                    dt = datetime.datetime.strptime(ts_str, fmt)
                    
                    # 为不包含年份的格式补充当前年份
                    if "%Y" not in fmt:
                        dt = dt.replace(year=current_year)
                    
                    # 为仅包含时间的格式补充当前日期
                    if not any(x in fmt for x in ["%Y", "%m", "%d"]):
                        dt = dt.replace(year=current_date.year, month=current_date.month, day=current_date.day)
                        
                    return dt
                except ValueError:
                    continue
    
    # 无法解析，返回None
    logging.debug(f"无法解析时间戳: {line[:30]}...")
    return None

def get_default_datetime(failed_timestamp=None):
    """
    获取默认的日期时间，用于时间戳解析失败时
    
    Args:
        failed_timestamp: 解析失败的时间戳字符串，用于日志记录
        
    Returns:
        默认的datetime对象
    """
    if failed_timestamp:
        logging.debug(f"使用当前时间作为默认值，无法解析: {failed_timestamp}")
    return datetime.datetime.now()

def parse_timestamp_safe(line: str, custom_formats: list = None, use_default: bool = True) -> datetime.datetime:
    """
    安全的时间戳解析函数，在解析失败时返回当前时间或None
    
    Args:
        line: 日志行
        custom_formats: 用户自定义的时间戳格式列表
        use_default: 解析失败时是否返回默认时间(当前时间)，False则返回None
        
    Returns:
        解析后的时间戳，解析失败时根据use_default返回当前时间或None
    """
    result = parse_timestamp(line, custom_formats)
    if result is None and use_default:
        return get_default_datetime(line[:30] if line else None)
    return result
