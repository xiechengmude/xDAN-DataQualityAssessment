# Alpaca数据集质量评估系统

这是一个基于xDAN-AI 的高性能数据质量评估系统，专门用于处理和优化Alpaca格式的数据集。系统支持从HuggingFace直接加载数据集，并允许自定义字段映射。

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

1. **基本用法**
```bash
python -m src.cli process-dataset
```

2. **指定数据集**
```bash
python -m src.cli process-dataset \
    --dataset-name "tatsu-lab/alpaca" \
    --dataset-split "train"
```

3. **自定义配置**
```bash
python -m src.cli process-dataset \
    --config-path "path/to/config.yaml" \
    --output-dir "path/to/output"
```

## 输出格式

处理后的数据将分为两个文件：
- `successful.jsonl`: 成功处理的数据
- `failed.jsonl`: 处理失败的数据

每个处理后的数据项包含：
```json
{
  "original": {
    "instruction": "...",
    "input": "...",
    "output": "..."
  },
  "quality_metrics": {
    "clarity": 8.5,
    "correctness": 9.0,
    "completeness": 8.0,
    "educational_value": 8.5
  },
  "category": "PROGRAMMING",
  "processed_output": "...",
  "metadata": {
    "processing_time": 1.23,
    "token_count": 450,
    "improvement_suggestions": [...],
    "validation_notes": [...]
  }
}
```

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
