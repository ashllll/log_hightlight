#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
关键词匹配模块
负责关键词的定义、分类、匹配和高亮显示。
"""

import re
import html
from typing import List, Dict, Tuple, Optional
import logging


class Keyword:
    def __init__(self, raw: str, annotation: str, match_case: bool = False, whole_word: bool = False, use_regex: bool = False, color: str = "#ffff99"):
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
        pat = self.raw if self.use_regex else re.escape(self.raw)
        if self.whole_word:
            pat = rf'(?<!\w){pat}(?!\w)'
        if not self.match_case:
            pat = f'(?i:{pat})'
        name = f'k{idx}'
        return f'(?P<{name}>{pat})', name


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
            logging.debug(f"Compiled KeywordMatcher pattern: {self.combined_pattern.pattern}")
        else:
            logging.warning("KeywordMatcher: No keyword patterns were compiled.")
            self.combined_pattern = None
    
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
    
    def highlight_line(self, line: str) -> Tuple[bool, str, Optional[str]]:
        """高亮显示一行文本中的匹配项, 并返回匹配的原始关键词"""
        if not self.combined_pattern:
            return False, html.escape(line), None

        # 预过滤可以保留，如果性能需要的话
        # should_process = self.should_process_line(line)
        # if not should_process:
        #     return False, html.escape(line), None

        last_end = 0
        highlighted_parts = []
        has_match = False

        # 使用 finditer 获取所有非重叠匹配
        # 先将 finditer 的结果转换为列表，这样可以检查是否为空
        matches = list(self.combined_pattern.finditer(line))
        
        if not matches: # 如果没有找到任何匹配
            # logging.debug(f"KeywordMatcher: No matches found by finditer for line '{line[:100]}...'")
            return False, html.escape(line), None # 直接转义整行并返回

        # logging.debug(f"KeywordMatcher: Found {len(matches)} potential match(es) for line '{line[:100]}...' by finditer.")
        
        first_matched_keyword_raw = None # 用于存储第一个成功匹配的关键词的原始文本

        for match_obj in matches:
            start, end = match_obj.span() # 获取当前匹配对象的起始和结束位置
            
            group_name_found = None
            actual_matched_text_from_group = None
            kw_config = None

            # 遍历命名捕获组，找到实际捕获当前文本段的那个组
            for name, content in match_obj.groupdict().items():
                if content is not None:
                    # 确保这个命名组的范围与当前match_obj的整体范围一致
                    # 这是因为一个match_obj可能由多个'|'分隔的模式产生，但只有一个命名组会捕获内容
                    if match_obj.start(name) == start and match_obj.end(name) == end:
                        group_name_found = name
                        actual_matched_text_from_group = content
                        # 获取关键词配置
                        kw_config = self.pattern_mapping.get(name, {})
                        break
            
            if group_name_found is None or not kw_config:
                # 这通常不应该发生，如果发生了，说明正则表达式或逻辑可能有问题
                # 安全起见，将这部分视为未匹配内容
                logging.warning(f"KeywordMatcher: Could not determine active group for match spanning ({start}-{end}) in line: {line[:100]}... Treating as non-match for this segment.")
                continue # 跳过这个无法确定命名组的 match_obj

            # 1. 添加从上一个匹配结束到当前匹配开始之间的文本 (进行HTML转义)
            if start > last_end:
                highlighted_parts.append(html.escape(line[last_end:start]))

            # 2. 获取关键词配置，准备高亮处理当前匹配的文本
            color = kw_config.get('color', '#ffff99') # 默认颜色
            annotation = kw_config.get('annotation', '')
            # 获取原始关键词文本（这是配置中的原始关键词，不是匹配的文本片段）
            matched_keyword_raw = kw_config.get('raw', None)
            if first_matched_keyword_raw is None and matched_keyword_raw is not None:
                first_matched_keyword_raw = matched_keyword_raw
            
            # 注解文本放入HTML title属性时需要转义，以防注解本身包含HTML特殊字符
            escaped_annotation = html.escape(annotation) if annotation else ''

            # actual_matched_text_from_group 是从原始行中匹配到的文本
            # 在将其放入span标签内容之前，进行HTML转义，以防止XSS或显示问题
            # 例如，如果关键词是 "<b>"，转义后会显示为 "&lt;b&gt;" 而不是加粗
            escaped_matched_text = html.escape(actual_matched_text_from_group)
            
            # 3. 创建带有关键词高亮的HTML元素，并将其添加到结果中
            highlighted_text = f"<span style=\"background-color: {color};\" title=\"{escaped_annotation}\">{escaped_matched_text}</span>"
            highlighted_parts.append(highlighted_text)
            
            # 更新last_end以处理下一个文本段
            last_end = end
            has_match = True # 因为找到了有效匹配

        # 4. 处理最后一个匹配之后的文本 (进行HTML转义)
        if last_end < len(line):
            highlighted_parts.append(html.escape(line[last_end:]))
        
        # 5. 将所有部分连接起来形成完整的高亮文本
        return has_match, ''.join(highlighted_parts), first_matched_keyword_raw
