# xDAN-AI数据集质量评估系统

这是一个基于xDAN-AI 的高性能数据质量评估系统，专门用于处理和优化Alpaca格式的数据集。系统支持从HuggingFace直接加载数据集，并允许自定义字段映射。

## 系统概述

这是一个数据质量评估系统，主要用于评估数据集中的数据质量。系统使用LLM（大语言模型）来分析和评估数据的各个质量维度，并提供详细的评估结果和改进建议。

## 主要功能

### 1. 数据加载和处理
- 支持从Hugging Face加载数据集
- 支持自定义字段映射配置
- 支持数据采样和过滤

### 2. 质量评估
- **多维度评估**：
  - reasoning_depth（推理深度）
  - correctness（正确性）
  - clarity（清晰度）
  - context_awareness（上下文意识）
  - engagement（参与度）
- 支持加权评分系统
- 分数归一化处理（0-10分）

### 3. 数据分类
- 支持自定义分类体系
- 当前包含：
  - KNOWLEDGE（知识性回答）
  - CONVERSATION（对话性回答）
  - CREATIVE（创造性回答）

### 4. 批量处理能力
- 异步并发处理
- 支持进度跟踪
- 错误处理和重试机制

### 5. 结果输出和分析
- 详细的评估指标
- 改进建议
- 处理元数据（时间、token使用等）
- 批处理统计（平均分、最高分、最低分等）

### 6. 配置系统
- 支持OpenAI API配置
- 并发和速率限制配置
- 输出配置（本地/HuggingFace Hub）
- 日志配置

## 特性

- 支持HuggingFace数据集加载
- 自定义字段映射
- 高并发异步处理
- 自动重试机制
- 批量处理支持
- 详细的进度显示
- 完整的错误处理
- 灵活的配置系统
- 精确的数据质量评估
- 场景分类标记

## 安装

1. 克隆仓库：
```bash
git clone <repository-url>
cd DataQualityAssessment
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置环境变量：
创建 `.env` 文件并设置：
```
OPENAI_API_KEY=your_api_key_here
```

## 配置

在 `config/default_config.yaml` 中可以自定义：

1. **数据集配置**
```yaml
dataset:
  hf_dataset_name: "tatsu-lab/alpaca"  # HuggingFace数据集名称
  hf_dataset_split: "train"            # 数据集分片
  field_mapping:                       # 字段映射
    instruction: "instruction"
    input: "input"
    output: "output"
```

2. **处理参数**
```yaml
concurrency:
  max_concurrent_requests: 50
  batch_size: 100
```

3. **质量控制**
```yaml
dataset:
  min_quality_score: 6.0
  excluded_categories: []
```

## 使用方法

本系统提供了两种主要的使用方式：

### 1. 使用脚本运行（推荐）

使用 `run_task.sh` 脚本可以在后台运行评估任务，并提供任务管理功能：

#### 基本用法
使用默认配置运行：
```bash
./scripts/run_task.sh
```

#### 指定配置文件
```bash
./scripts/run_task.sh config/test_config.yaml
```

#### 任务管理
查看任务状态：
```bash
./scripts/check_task.sh [任务ID]
```

查看任务日志：
```bash
tail -f outputs/[任务ID]/task.log
```

### 2. 直接运行Python脚本

如果需要更细粒度的控制，可以直接运行Python脚本：

```bash
python scripts/run_assessment.py --config config/default_config.yaml
```

### 输出文件

处理后的数据将保存在 `outputs/[任务ID]` 目录下：
- `successful.jsonl`: 成功处理的数据
- `failed.jsonl`: 处理失败的数据
- `task.log`: 任务日志
- `task.pid`: 任务进程ID
- `task_config.yaml`: 任务使用的配置文件副本

每个处理后的数据项包含：
```json
{
  "id": "unique_id",
  "sources": "dataset_name",
  "instruction": "原始指令",
  "input": "输入内容",
  "output": "输出内容",
  "quality_metrics": {
    "reasoning_depth": 8.5,
    "correctness": 9.0,
    "clarity": 7.5,
    "context_awareness": 8.0,
    "engagement": 7.0
  },
  "score": 8.0,
  "category": "KNOWLEDGE",
  "metadata": {
    "processing_time": 1.5,
    "token_count": 150,
    "improvement_suggestions": [
      "建议1",
      "建议2"
    ],
    "validation_notes": [
      "说明1",
      "说明2"
    ]
  }
}
```

### 任务管理功能

1. **任务状态查看**
   - 使用 `check_task.sh` 脚本查看任务状态
   - 显示任务是否运行中、CPU使用率、内存使用等信息

2. **日志查看**
   - 所有任务日志保存在 `outputs/[任务ID]/task.log`
   - 使用 `tail -f` 命令实时查看日志

3. **配置管理**
   - 每个任务的配置文件会被复制到任务目录
   - 方便追踪和复现任务配置

## 场景分类

系统支持30种细分场景类别，分为8个主要类别：
1. 核心能力场景 (数学、逻辑等)
2. 技术领域场景 (编程、架构等)
3. 业务领域场景 (商业、金融等)
4. 创新思维场景 (创意、策略等)
5. 语言交互场景 (写作、翻译等)
6. 专业领域场景 (法律、工程等)
7. 社会人文场景 (心理、文化等)
8. 实践应用场景 (生活、职业等)

## 性能优化

1. **并发处理**
   - 异步处理
   - 可配置并发数
   - 智能速率限制

2. **批量处理**
   - 自动分批
   - 进度显示
   - 结果汇总

3. **错误处理**
   - 自动重试
   - 错误日志
   - 结果分类

## 数据转换功能

### 配置文件

数据转换的配置在 `config/data_transform.yaml` 中定义：

```yaml
openai:
  model_name: "deepseek-chat"
  max_tokens: 8192
  temperature: 0.7
  api_key: "your-api-key"
  api_base: "your-api-base"

concurrency:
  max_concurrent_requests: 50
  batch_size: 200
  request_timeout: 60
  max_retries: 3
  retry_delay: 1

rate_limits:
  tokens_per_minute: 10000
  requests_per_minute: 1000

datasets:
  - name: "tatsu-lab/alpaca"
    config: "default"
    split: "train"
    subset: null
    num_samples: 5
  - name: "yahma/alpaca-cleaned"
    config: "default"
    split: "train"
    subset: null
    num_samples: 5
```

### 运行数据转换

使用以下命令运行数据转换：

```bash
# 使用默认配置文件
./scripts/transform_data.sh

# 使用自定义配置文件
./scripts/transform_data.sh --config config/custom_transform.yaml
```

### 转换结果

转换后的数据将按照以下结构组织：

```json
{
  "instruction": "原始问题",
  "input": "输入上下文（如果有）",
  "output": "原始答案",
  "refined_output": {
    "<Analyze>": [
      "完整保留原始问题",
      "提取关键信息和概念",
      "明确问题类型和目标",
      "列出已知条件和要求"
    ],
    "<Solve>": [
      "说明解决思路和理由",
      "列出详细的解决步骤",
      "展示完整的推理过程",
      "记录关键的中间结果"
    ],
    "<Verify>": [
      "检查步骤的正确性",
      "验证是否满足条件",
      "评估结果的合理性",
      "考虑优化的空间"
    ],
    "<Solution>": "面向用户的友好回答"
  }
}
```

转换后的数据将保存在 `outputs` 目录下，文件名可在配置文件中指定。

## 注意事项

1. 确保有足够的API配额
2. 根据API限制调整并发参数
3. 对大型数据集先进行小规模测试
4. 定期检查处理日志

## 开发计划

- [ ] 添加更多数据验证规则
- [ ] 实现自适应并发控制
- [ ] 支持更多数据源
- [ ] 添加质量评估可视化
- [ ] 支持自定义评估规则
# xDAN-DataQualityAssessment
