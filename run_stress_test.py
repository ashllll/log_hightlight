#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import subprocess
import psutil
import argparse
import threading
import json
from datetime import datetime

class MemoryMonitor(threading.Thread):
    """监控内存使用情况的线程"""
    
    def __init__(self, interval=0.5):
        super().__init__()
        self.interval = interval
        self.running = True
        self.memory_usage = []
        self.peak_usage = 0
        self.start_usage = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
    
    def run(self):
        while self.running:
            current = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            self.memory_usage.append(current)
            self.peak_usage = max(self.peak_usage, current)
            time.sleep(self.interval)
    
    def stop(self):
        self.running = False
        
    def get_peak_diff(self):
        """获取峰值内存增长"""
        return self.peak_usage - self.start_usage
    
    def get_stats(self):
        """获取内存统计信息"""
        if not self.memory_usage:
            return {
                "start": self.start_usage,
                "peak": 0,
                "peak_diff": 0,
                "avg": 0,
                "data": []
            }
        
        return {
            "start": self.start_usage,
            "peak": self.peak_usage,
            "peak_diff": self.get_peak_diff(),
            "avg": sum(self.memory_usage) / len(self.memory_usage),
            "data": self.memory_usage
        }

def run_test(python_path, log_highlighter_path, toml_path, log_dir, options=None):
    """
    运行日志高亮工具进行测试
    
    Args:
        python_path: Python解释器路径
        log_highlighter_path: 日志高亮工具路径
        toml_path: TOML配置文件路径
        log_dir: 日志文件目录
        options: 其他选项字典
        
    Returns:
        测试结果字典
    """
    options = options or {}
    monitor = MemoryMonitor()
    
    print(f"正在启动测试: {log_dir}")
    
    # 记录开始状态
    start_time = time.time()
    
    # 准备环境变量 - 设置QT无界面模式
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    
    # 执行命令
    cmd = [python_path, log_highlighter_path]
    
    # 监控进程内存使用
    monitor.start()
    
    # 在PyQt应用中，我们通常无法以命令行方式运行，所以这个测试更多是示例
    # 实际使用时，您可能需要修改日志高亮工具支持命令行参数
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        env=env
    )
    
    # 获取进程输出
    stdout, stderr = process.communicate()
    
    # 停止监控
    monitor.stop()
    monitor.join()
    
    # 记录结束状态
    end_time = time.time()
    duration = end_time - start_time
    
    # 获取内存统计
    memory_stats = monitor.get_stats()
    
    return {
        "duration": duration,
        "memory": memory_stats,
        "exit_code": process.returncode,
        "stdout": stdout.decode("utf-8", errors="ignore"),
        "stderr": stderr.decode("utf-8", errors="ignore"),
        "options": options
    }

def main():
    parser = argparse.ArgumentParser(description="日志高亮工具压力测试")
    parser.add_argument("--python", default="python", help="Python解释器路径")
    parser.add_argument("--tool", default="log_hightlight.py", help="日志高亮工具路径")
    parser.add_argument("--toml", required=True, help="TOML配置文件路径")
    parser.add_argument("--logs", required=True, help="日志文件目录")
    parser.add_argument("--output", default="test_results", help="测试结果输出目录")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.tool):
        print(f"错误: 未找到日志高亮工具 {args.tool}")
        return
        
    if not os.path.exists(args.toml):
        print(f"错误: 未找到TOML配置文件 {args.toml}")
        return
        
    if not os.path.exists(args.logs):
        print(f"错误: 未找到日志目录 {args.logs}")
        return
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 设置测试场景
    test_scenarios = [
        {
            "name": "基本测试",
            "description": "使用默认设置运行测试",
            "options": {}
        },
        {
            "name": "快速模式",
            "description": "使用快速模式运行测试",
            "options": {"scan_mode": "fast"}
        },
        {
            "name": "精确模式",
            "description": "使用精确模式运行测试",
            "options": {"scan_mode": "accurate"}
        },
        {
            "name": "多线程测试",
            "description": "使用8个线程运行测试",
            "options": {"threads": 8}
        },
        {
            "name": "大批量测试",
            "description": "使用大批量处理运行测试",
            "options": {"batch_size": 20}
        }
    ]
    
    print(f"开始压力测试，共 {len(test_scenarios)} 个场景")
    print(f"配置文件: {args.toml}")
    print(f"日志目录: {args.logs}")
    
    results = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n[{i}/{len(test_scenarios)}] 运行场景: {scenario['name']}")
        print(f"描述: {scenario['description']}")
        
        result = run_test(
            args.python,
            args.tool,
            args.toml,
            args.logs,
            scenario["options"]
        )
        
        # 添加场景信息
        result["scenario"] = scenario["name"]
        result["description"] = scenario["description"]
        
        # 打印结果摘要
        print(f"完成: 耗时 {result['duration']:.2f} 秒")
        print(f"内存峰值增长: {result['memory']['peak_diff']:.2f} MB")
        
        results.append(result)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(args.output, f"results_{timestamp}.json")
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n测试完成! 结果已保存至 {result_file}")
    
    # 生成简单报告
    print("\n测试结果摘要:")
    print("-" * 80)
    print(f"{'场景':<20} {'耗时(秒)':<12} {'内存峰值(MB)':<15} {'结果':<10}")
    print("-" * 80)
    
    for result in results:
        status = "成功" if result["exit_code"] == 0 else f"失败({result['exit_code']})"
        print(f"{result['scenario']:<20} {result['duration']:<12.2f} {result['memory']['peak_diff']:<15.2f} {status:<10}")

if __name__ == "__main__":
    main() 