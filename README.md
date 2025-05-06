# 日志高亮工具压力测试套件

这个测试套件用于对Python日志高亮工具进行压力测试，包括生成测试日志、创建压缩归档和运行性能测试。

## 目录结构

```
.
├── README.md                  # 本说明文件
├── keywords.toml              # 关键词配置文件
├── generate_test_logs.py      # 生成测试日志的脚本
├── create_archive.py          # 创建日志归档文件的脚本
├── run_stress_test.py         # 性能压力测试脚本
└── log_hightlight.py          # 日志高亮工具主程序
```

## 安装依赖

在运行测试之前，请确保安装了所需的Python库：

```bash
pip install toml psutil PyQt5
```

## 使用步骤

### 1. 生成测试日志

生成不同大小的测试日志文件：

```bash
# 生成5个各10MB的日志文件
python generate_test_logs.py --output test_logs --count 5 --size 10

# 生成1个100MB的超大日志文件
python generate_test_logs.py --output test_logs --count 1 --size 100

# 生成压缩格式的日志文件
python generate_test_logs.py --output test_logs --count 3 --size 20 --compress
```

参数说明：
- `--output`/-o: 日志文件输出目录
- `--count`/-c: 生成的文件数量
- `--size`/-s: 每个文件大小(MB)
- `--compress`: 使用gzip压缩生成的日志
- `--error-rate`: 错误日志的概率(0-1)
- `--regex-rate`: 包含正则匹配内容的概率(0-1)

### 2. 创建日志归档文件

将生成的日志文件打包成ZIP归档，用于测试解压功能：

```bash
# 创建3个归档，每个包含5个日志文件，含嵌套结构
python create_archive.py --source test_logs --output archives --count 3 --files 5 --nested

# 创建1个大型归档，包含所有日志文件
python create_archive.py --source test_logs --output archives --count 1 --files 20
```

参数说明：
- `--source`/-s: 源日志目录
- `--output`/-o: 输出目录
- `--count`/-c: 创建的归档数量
- `--files`/-f: 每个归档的文件数量
- `--nested`/-n: 包含嵌套归档

### 3. 运行压力测试

测试日志高亮工具在不同场景下的性能表现：

```bash
# 使用默认设置运行测试
python run_stress_test.py --toml keywords.toml --logs test_logs

# 指定Python解释器和工具路径
python run_stress_test.py --python python3 --tool ./log_hightlight.py --toml keywords.toml --logs test_logs
```

参数说明：
- `--python`: Python解释器路径
- `--tool`: 日志高亮工具路径
- `--toml`: TOML配置文件路径
- `--logs`: 日志文件目录
- `--output`: 测试结果输出目录

## 测试策略建议

### 小文件测试

生成多个小型日志文件，测试工具处理多文件的能力：

```bash
python generate_test_logs.py --output test_small --count 20 --size 5
```

### 大文件测试

生成少量大型日志文件，测试内存映射和流式处理功能：

```bash
python generate_test_logs.py --output test_large --count 2 --size 200
```

### 压缩文件测试

创建嵌套的归档文件，测试工具处理复杂压缩结构的能力：

```bash
python generate_test_logs.py --output test_compressed --count 10 --size 10 --compress
python create_archive.py --source test_compressed --output test_archives --count 3 --nested
```

### 关键词匹配测试

根据需要调整`keywords.toml`文件，添加或修改关键词组，特别关注：

1. 正则表达式匹配性能
2. 大量关键词的处理能力
3. 不同匹配模式(大小写敏感、全词匹配)的效果

## 结果分析

测试结果将保存在`test_results`目录下，包含：

- 耗时统计
- 内存使用情况
- 不同场景下的性能比较

通过分析这些结果，可以确定工具在不同场景下的最佳配置参数和性能瓶颈。 