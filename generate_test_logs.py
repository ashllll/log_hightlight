#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import datetime
import os
import gzip
import time
import string
import argparse
from typing import List, Dict, Tuple, Optional

def generate_timestamp() -> str:
    """生成一个随机的时间戳"""
    now = datetime.datetime.now()
    delta = datetime.timedelta(
        days=random.randint(-30, 0),
        hours=random.randint(-23, 23),
        minutes=random.randint(-59, 59),
        seconds=random.randint(-59, 59)
    )
    timestamp = now + delta
    return timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def generate_ip() -> str:
    """生成一个随机的IP地址"""
    return f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}"

def generate_uuid() -> str:
    """生成一个随机的UUID"""
    hex_chars = "0123456789abcdef"
    sections = [8, 4, 4, 4, 12]  # UUID格式的各部分长度
    
    uuid_parts = []
    for length in sections:
        part = ''.join(random.choice(hex_chars) for _ in range(length))
        uuid_parts.append(part)
    
    return '-'.join(uuid_parts)

def generate_memory_usage() -> str:
    """生成一个随机的内存使用量字符串"""
    value = random.randint(1, 9999)
    unit = random.choice(["KB", "MB", "GB"])
    return f"{value}{unit}"

def generate_random_text(length=50) -> str:
    """生成随机文本"""
    chars = string.ascii_letters + string.digits + " " * 10
    return ''.join(random.choice(chars) for _ in range(length))

def generate_log_line(include_error_prob=0.05, include_regex_prob=0.1) -> str:
    """生成一行日志"""
    timestamp = generate_timestamp()
    ip = generate_ip() if random.random() < include_regex_prob else "localhost"
    
    log_types = ["INFO", "DEBUG", "WARNING", "ERROR", "TRACE"]
    weights = [0.6, 0.15, 0.15, 0.05, 0.05]  # 控制各种日志类型的概率
    log_type = random.choices(log_types, weights=weights)[0]
    
    random_text = generate_random_text(random.randint(30, 100))
    
    # 随机添加一些特殊内容，如异常信息
    if log_type == "ERROR" or (log_type == "WARNING" and random.random() < 0.3):
        exceptions = [
            "NullPointerException",
            "IndexOutOfBoundsException",
            "IllegalArgumentException",
            "IOException",
            "SQLiteException",
            "RuntimeException",
            "TimeoutException"
        ]
        exception = random.choice(exceptions)
        random_text += f" Exception occurred: {exception} at line {random.randint(1, 1000)}"
    
    # 随机添加一些关键词
    if random.random() < include_error_prob:
        keywords = [
            "Failed to connect", "Connection refused", "Unable to access", 
            "Permission denied", "Resource unavailable", "Timeout waiting for",
            "Access denied", "Authentication failed", "CPU usage high",
            "Memory usage exceeded", "Disk space low", "Network timeout"
        ]
        random_text += f" {random.choice(keywords)}"
    
    # 可能添加中文日志内容
    if random.random() < 0.2:
        cn_keywords = ["信息", "警告", "错误", "调试", "失败", "注意", "超时"]
        cn_actions = ["处理", "连接", "启动", "查询", "认证", "读取", "写入"]
        cn_objects = ["数据库", "文件", "网络", "内存", "设备", "服务", "进程"]
        
        cn_text = f"{random.choice(cn_keywords)}: {random.choice(cn_actions)}{random.choice(cn_objects)}"
        random_text += f" {cn_text}"
        
    # 可能添加UUID和内存使用信息
    if random.random() < include_regex_prob:
        if random.random() < 0.5:
            uuid = generate_uuid()
            random_text += f" ID: {uuid}"
            
        if random.random() < 0.5:
            memory = generate_memory_usage()
            random_text += f" Memory usage: {memory}"
    
    # 可能添加一个处理请求的语句
    if random.random() < 0.15:
        endpoints = ["/api/user", "/api/product", "/auth/login", "/admin/dashboard", "/cart/checkout"]
        methods = ["GET", "POST", "PUT", "DELETE"]
        status_codes = [200, 201, 400, 401, 403, 404, 500]
        
        endpoint = random.choice(endpoints)
        method = random.choice(methods)
        status = random.choice(status_codes)
        
        random_text += f" Request: {method} {endpoint} - Status: {status}"
    
    return f"{timestamp} [{log_type}] {ip} - {random_text}"

def generate_log_file(filename: str, size_mb: int, compressed: bool = False, include_error_prob: float = 0.05, include_regex_prob: float = 0.1) -> None:
    """生成指定大小的日志文件"""
    size_bytes = size_mb * 1024 * 1024
    current_size = 0
    lines_written = 0
    
    if compressed:
        f = gzip.open(filename, 'wt', encoding='utf-8')
    else:
        f = open(filename, 'w', encoding='utf-8')
    
    try:
        while current_size < size_bytes:
            log_line = generate_log_line(include_error_prob, include_regex_prob) + "\n"
            f.write(log_line)
            current_size += len(log_line.encode('utf-8'))
            lines_written += 1
            
            # 每生成100行打印一次进度
            if lines_written % 100 == 0:
                print(f"生成中: {current_size / (1024 * 1024):.2f} MB / {size_mb} MB, {lines_written} 行", end="\r")
    finally:
        f.close()
    
    print(f"已生成日志文件: {filename} ({size_mb} MB, {lines_written} 行)")

def main():
    parser = argparse.ArgumentParser(description="生成测试日志文件")
    parser.add_argument("--output", "-o", default="test_logs", help="输出目录")
    parser.add_argument("--count", "-c", type=int, default=5, help="生成的文件数量")
    parser.add_argument("--size", "-s", type=int, default=10, help="每个文件大小(MB)")
    parser.add_argument("--compress", action="store_true", help="是否使用gzip压缩")
    parser.add_argument("--error-rate", type=float, default=0.05, help="错误日志的概率(0-1)")
    parser.add_argument("--regex-rate", type=float, default=0.1, help="包含正则匹配内容的概率(0-1)")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    for i in range(args.count):
        if args.compress:
            filename = os.path.join(args.output, f"logfile_{i+1}.log.gz")
        else:
            filename = os.path.join(args.output, f"logfile_{i+1}.log")
        
        generate_log_file(filename, args.size, args.compress, args.error_rate, args.regex_rate)

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"总耗时: {elapsed_time:.2f} 秒") 