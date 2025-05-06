#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import zipfile
import argparse
import shutil
import random
import gzip
import time
from datetime import datetime

def compress_file_gzip(source_file, dest_file=None):
    """将文件使用gzip压缩"""
    if dest_file is None:
        dest_file = source_file + '.gz'
    
    with open(source_file, 'rb') as f_in:
        with gzip.open(dest_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    return dest_file

def create_zip_archive(source_dir, output_file, compression_level=6, include_pattern=None, exclude_pattern=None):
    """
    创建ZIP归档
    
    Args:
        source_dir: 源目录
        output_file: 输出文件路径
        compression_level: 压缩级别 (0-9)
        include_pattern: 包含的文件模式列表
        exclude_pattern: 排除的文件模式列表
        
    Returns:
        生成的ZIP文件路径
    """
    if include_pattern is None:
        include_pattern = ['*.log']
    
    if exclude_pattern is None:
        exclude_pattern = []
    
    print(f"创建ZIP归档: {output_file}")
    print(f"源目录: {source_dir}")
    
    with zipfile.ZipFile(output_file, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=compression_level) as zipf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, source_dir)
                
                # 检查是否应包含此文件
                should_include = any(file.endswith(pattern.replace('*', '')) for pattern in include_pattern)
                should_exclude = any(file.endswith(pattern.replace('*', '')) for pattern in exclude_pattern)
                
                if should_include and not should_exclude:
                    print(f"  添加: {rel_path}")
                    zipf.write(file_path, rel_path)
    
    return output_file

def create_mixed_archive(source_dir, output_dir, num_archives=3, files_per_archive=5, include_nested=True):
    """
    创建混合结构的归档文件，包括ZIP嵌套ZIP、嵌套GZIP等
    
    Args:
        source_dir: 源日志目录
        output_dir: 输出目录
        num_archives: 创建的归档数量
        files_per_archive: 每个归档的文件数量
        include_nested: 是否包含嵌套归档
        
    Returns:
        创建的归档文件列表
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取源目录中的所有日志文件
    log_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.log'):
                log_files.append(os.path.join(root, file))
    
    if not log_files:
        print("错误: 源目录中没有日志文件")
        return []
    
    created_archives = []
    
    for i in range(num_archives):
        # 为每个归档创建临时目录
        temp_dir = os.path.join(output_dir, f"temp_{i}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        
        # 随机选择日志文件
        selected_files = random.sample(log_files, min(files_per_archive, len(log_files)))
        
        # 复制文件到临时目录
        for file in selected_files:
            shutil.copy2(file, temp_dir)
        
        # 如果需要嵌套归档
        if include_nested and i > 0:
            # 创建内部ZIP
            inner_zip = os.path.join(temp_dir, f"inner_logs_{i}.zip")
            inner_files = random.sample(selected_files, min(2, len(selected_files)))
            
            with zipfile.ZipFile(inner_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
                for file in inner_files:
                    zipf.write(file, os.path.basename(file))
            
            # 随机创建一些GZIP文件
            for file in random.sample(selected_files, min(2, len(selected_files))):
                gzip_file = os.path.join(temp_dir, os.path.basename(file) + '.gz')
                compress_file_gzip(file, gzip_file)
        
        # 创建ZIP归档
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_file = os.path.join(output_dir, f"logs_archive_{i+1}_{timestamp}.zip")
        create_zip_archive(temp_dir, zip_file)
        created_archives.append(zip_file)
        
        # 清理临时目录
        shutil.rmtree(temp_dir)
    
    return created_archives

def main():
    parser = argparse.ArgumentParser(description="创建日志归档文件")
    parser.add_argument("--source", "-s", required=True, help="源日志目录")
    parser.add_argument("--output", "-o", default="archives", help="输出目录")
    parser.add_argument("--count", "-c", type=int, default=3, help="创建的归档数量")
    parser.add_argument("--files", "-f", type=int, default=5, help="每个归档的文件数量")
    parser.add_argument("--nested", "-n", action="store_true", help="包含嵌套归档")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.source):
        print(f"错误: 源目录 {args.source} 不存在")
        return 1
    
    try:
        start_time = time.time()
        
        print(f"开始创建归档文件...")
        print(f"源目录: {args.source}")
        print(f"输出目录: {args.output}")
        print(f"归档数量: {args.count}")
        print(f"每个归档文件数: {args.files}")
        print(f"包含嵌套: {'是' if args.nested else '否'}")
        
        archives = create_mixed_archive(
            args.source, 
            args.output, 
            args.count, 
            args.files, 
            args.nested
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"\n创建完成! 共创建 {len(archives)} 个归档文件:")
        for archive in archives:
            size_mb = os.path.getsize(archive) / (1024 * 1024)
            print(f"  - {os.path.basename(archive)} ({size_mb:.2f} MB)")
        
        print(f"\n总耗时: {elapsed_time:.2f} 秒")
        
        return 0
    
    except Exception as e:
        print(f"错误: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 