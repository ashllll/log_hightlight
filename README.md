# Python日志高亮工具 (Log Highlighter)

一个高效的Python日志文件分析和高亮显示工具，帮助开发人员和系统管理员更快地分析和理解大型日志文件。

![Python版本](https://img.shields.io/badge/Python-3.6+-blue.svg)
![许可证](https://img.shields.io/badge/License-MIT-green.svg)

## 功能特点

### 核心功能
- 🔍 **关键词高亮**：根据自定义关键词或预定义分组自动高亮显示日志内容
- 📦 **压缩文件支持**：直接分析压缩包（zip, gz, tar等）中的日志文件
- 🚀 **高性能处理**：优化的多进程处理支持，可以处理GB级别的日志文件
- 🧠 **智能内存管理**：自适应内存使用监控和优化，防止内存溢出
- 📊 **简洁美观的UI**：基于PyQt5的直观图形界面
- 💡 **一键离线启动**：支持Windows下无需联网自动安装依赖并启动

### 性能优化
- 多级关键词匹配策略
- 使用ProcessPoolExecutor进行多核心并行处理
- 智能文件批处理机制
- 分级文件处理策略（小文件直接读取，大文件流式处理，超大文件内存映射）

### 内存管理
- 内存使用趋势监控
- 自适应线程/进程数控制
- 内存压力下的资源管理
- 结果分块写入减少内存占用

## 安装与启动指南

### 前提条件
- Python 3.6+
- Windows 推荐直接用一键启动脚本，无需手动安装依赖

### 一键离线启动（推荐，适用于Windows）

1. **下载完整项目（含 libs/ 依赖包目录）**
2. 双击 `start.bat`，自动检测/安装所有依赖并启动主程序

> `start.bat` 会自动调用 `bootstrap.py`，优先从 `libs/` 目录离线安装所有依赖，无需联网。

### 手动安装（适用于Linux/MacOS或自定义环境）

1. 克隆仓库
```bash
git clone https://github.com/ashllll/log_hightlight.git
cd log_hightlight
```
2. 创建虚拟环境（推荐）
```bash
python -m venv venv
source venv/bin/activate  # Linux/MacOS
# 或者
venv\Scripts\activate  # Windows
```
3. 安装依赖
```bash
pip install -r requirements.txt
```
4. 启动应用
```bash
python log_hightlight.py
```

## 使用方法

### 启动应用
- Windows 推荐直接运行 `start.bat`
- 其它平台运行 `python log_hightlight.py`

### 基本操作流程
1. **选择日志文件**：点击"浏览"按钮选择单个日志文件或压缩包
2. **配置关键词**：勾选右侧的预定义关键词组或添加自定义关键词
3. **开始分析**：点击"开始分析"按钮
4. **查看结果**：分析完成后会自动打开生成的HTML报告

### 关键词配置
- 可以使用预定义的关键词分组（错误、警告、信息等）
- 添加自定义关键词，支持精确匹配和正则表达式
- 为关键词设置颜色、注释和匹配规则

## 离线依赖包与自动安装

- 所有三方依赖已下载至 `libs/` 目录
- `bootstrap.py` 会自动检测缺失依赖并优先离线安装
- 支持无网络环境下一键部署

## 配置文件

应用程序使用TOML格式的配置文件 (`keywords.toml`) 来管理关键词分组：

```toml
# 示例配置
[group.errors]
match_case = false
whole_word = false
use_regex = false

[group.errors.error1]
key = "ERROR"
annotation = "错误级别日志"

[group.errors.error2]
key = "Exception"
annotation = "异常信息"
```

## 项目结构
```
log_hightlight/
├── log_hightlight.py       # 主程序
├── keywords.toml           # 关键词配置文件
├── create_archive.py       # 创建测试归档文件工具
├── generate_test_logs.py   # 生成测试日志文件
├── run_stress_test.py      # 性能测试脚本
├── requirements.txt        # 依赖列表
├── bootstrap.py            # 自动检测/安装依赖并启动主程序
├── start.bat               # Windows一键启动脚本
├── libs/                   # 离线依赖包目录
├── .gitignore              # Git忽略文件
└── README.md               # 项目说明
```

## .gitignore 说明
- 已配置忽略虚拟环境、依赖包、临时文件、日志、测试输出、IDE配置等常见无关内容
- `libs/` 下的所有 wheel/压缩包也会被忽略

## 示例

### 错误日志分析
![错误日志示例](https://example.com/error_log_demo.png)

### 系统性能监控日志
![系统监控示例](https://example.com/system_log_demo.png)

## 自定义开发

### 添加新的关键词分组
编辑 `keywords.toml` 文件，添加新的分组定义：

```toml
[group.custom_group]
match_case = true
whole_word = false
use_regex = false

[group.custom_group.keyword1]
key = "YourKeyword"
annotation = "自定义注释"
```

### 自定义界面
修改 `log_hightlight.py` 中的 UI 相关代码以满足特定需求。

## 故障排除

### 常见问题
- **分析速度慢**：对于大型日志文件，尝试增加配置中的线程数或进程数
- **内存占用高**：调整内存管理参数，降低批处理大小
- **无法识别压缩文件**：确保已安装所有必要的解压缩库
- **依赖安装失败**：请确认 `libs/` 目录完整，或手动联网安装依赖

## 贡献指南

欢迎贡献代码、报告问题或提出改进建议！请遵循以下步骤：

1. Fork 项目仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启一个 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详情请参阅 [LICENSE](LICENSE) 文件。

## 联系方式

如有问题或建议，请通过GitHub Issues提交。

---

感谢使用Python日志高亮工具！希望它能帮助您更高效地分析日志文件。 