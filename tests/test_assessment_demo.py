#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import json
from pathlib import Path
import logging
import asyncio
from datetime import datetime
from tqdm import tqdm
from typing import List, Optional
from datasets import load_dataset
import pytest
import pytest_asyncio

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.processor import DataProcessor
from src.models import AlpacaItem, BatchResult

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: Path) -> dict:
    """加载配置文件"""
    logger.info(f"Loading config from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_task_name() -> str:
    """生成任务名称，格式：YYYYMMDD_HHMMSS_test_assessment"""
    now = datetime.now()
    return f"{now.strftime('%Y%m%d_%H%M%S')}_test_assessment"

def save_results(results: List[AlpacaItem], task_name: str, output_dir: Path, config: dict):
    """保存评估结果
    
    Args:
        results: 评估结果列表
        task_name: 任务名称
        output_dir: 输出目录
        config: 配置信息
    """
    timestamp = datetime.now()
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存为Alpaca格式
    alpaca_file = output_dir / f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{task_name}_alpaca.json"
    logger.info(f"Saving Alpaca format results to: {alpaca_file}")
    
    alpaca_data = [item.model_dump() for item in results]
    with open(alpaca_file, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
    
    # 保存原始评估结果
    eval_file = output_dir / f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{task_name}_eval.json"
    logger.info(f"Saving evaluation details to: {eval_file}")
    
    # 计算平均分数
    total_scores = []
    for result in results:
        if hasattr(result, 'quality_metrics'):
            metrics = result.quality_metrics
            score = sum(metrics.values()) / len(metrics) if metrics else 0
            total_scores.append(score)
    
    avg_score = sum(total_scores) / len(total_scores) if total_scores else 0
    
    eval_data = {
        "task_name": task_name,
        "timestamp": timestamp.isoformat(),
        "model_name": config['openai']['model_name'],
        "average_score": avg_score,
        "num_samples": len(results),
        "config": {
            k: v for k, v in config.items()
            if k not in ['openai']  # 排除敏感信息
        }
    }
    
    with open(eval_file, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    
    # 如果配置了推送到Hub，则执行推送
    output_config = config.get('output', {})
    if output_config.get('push_to_hub'):
        hub_config = output_config.get('hub_config', {})
        try:
            from datasets import Dataset, DatasetDict
            
            # 根据数据源分组
            source_groups = {}
            for item in results:
                source = item.sources if hasattr(item, 'sources') and item.sources else 'unknown'
                if source not in source_groups:
                    source_groups[source] = []
                source_groups[source].append(item.model_dump())
            
            # 创建 DatasetDict
            dataset_dict = {}
            for source, items in source_groups.items():
                dataset_dict[source] = Dataset.from_list(items)
            
            dataset = DatasetDict(dataset_dict)
            
            # 推送到Hub
            repo_id = hub_config.get('repository_id')
            if not repo_id:
                logger.error("No repository_id specified in hub_config")
                return
                
            # 构建完整的repository_id
            if not "/" in repo_id:
                repo_id = f"{repo_id}/alpaca_eval"
                
            logger.info(f"Pushing results to Hugging Face Hub: {repo_id}")
            dataset.push_to_hub(
                repo_id,
                private=hub_config.get('private', True),
                token=hub_config.get('token')
            )
            
            logger.info("Successfully pushed results to Hugging Face Hub")
            
        except Exception as e:
            logger.error(f"Failed to push results to Hugging Face Hub: {str(e)}")
            logger.debug("Error details:", exc_info=True)

@pytest_asyncio.fixture
async def test_config():
    """测试配置 fixture"""
    config_path = project_root / "config" / "test_config.yaml"
    return load_config(config_path)

@pytest_asyncio.fixture
async def test_items(test_config):
    """测试数据 fixture"""
    return await load_test_data(test_config)

async def load_hf_dataset(dataset_config: dict) -> List[AlpacaItem]:
    """从HuggingFace加载数据集
    
    Args:
        dataset_config: 数据集配置信息
            {
                'name': 'dataset_name',  # 数据集名称
                'config': 'config_name',  # 可选的配置名称
                'split': 'train',  # 可选的数据集分片
                'num_samples': 10,  # 可选的样本数量
                'field_mapping': {  # 字段映射
                    'instruction': 'instruction',
                    'input': 'input',
                    'output': 'output'
                }
            }
            
    Returns:
        List[AlpacaItem]: 加载的数据项列表
    """
    try:
        # 加载数据集
        dataset = load_dataset(
            dataset_config['name'],
            dataset_config.get('config', None),
            split=dataset_config.get('split', 'train')
        )
        
        # 如果指定了样本数量
        if dataset_config.get('num_samples', -1) > 0:
            dataset = dataset.select(range(min(len(dataset), dataset_config['num_samples'])))
            
        # 获取字段映射
        field_mapping = dataset_config.get('field_mapping', {})
        instruction_field = field_mapping.get('instruction', 'instruction')
        input_field = field_mapping.get('input', 'input')
        output_field = field_mapping.get('output', 'output')
        
        # 转换为AlpacaItem
        items = []
        for item in dataset:
            try:
                alpaca_item = AlpacaItem(
                    instruction=str(item.get(instruction_field, '')),
                    input=str(item.get(input_field, '')),
                    output=str(item.get(output_field, '')),
                    sources=dataset_config['name']  # 添加数据源信息
                )
                items.append(alpaca_item)
            except Exception as e:
                logger.warning(f"跳过无效数据项: {str(e)}")
                continue
                
        logger.info(f"从数据集 {dataset_config['name']} 加载了 {len(items)} 个测试项")
        return items
        
    except Exception as e:
        logger.error(f"加载数据集失败: {str(e)}")
        return []

async def load_test_data(config: dict) -> List[AlpacaItem]:
    """加载测试数据
    
    Args:
        config: 配置信息，必须包含 datasets 字段，格式为：
            {
                'datasets': [
                    {
                        'name': 'dataset_name',
                        'num_samples': 10,
                        'field_mapping': {...}
                    },
                    ...
                ]
            }
        
    Returns:
        List[AlpacaItem]: 加载的数据项列表
    """
    items = []
    
    if 'datasets' not in config:
        logger.error("配置文件中缺少 datasets 字段")
        return items
        
    # 并行加载所有数据集
    tasks = [load_hf_dataset(dataset_config) for dataset_config in config['datasets']]
    results = await asyncio.gather(*tasks)
    
    # 合并结果
    for dataset_items in results:
        items.extend(dataset_items)
    
    return items

@pytest.mark.asyncio
async def test_small_batch(test_config: dict, test_items: list) -> None:
    """测试小批量数据评估"""
    try:
        # 初始化处理器
        processor = DataProcessor(test_config)
        logger.info("DataProcessor initialized")
        
        # 运行评估
        logger.info("Starting assessment of test data...")
        with tqdm(total=len(test_items), desc="Processing items") as pbar:
            results = await processor.process_batch(test_items, progress_callback=lambda: pbar.update(1))
            
        # 输出结果
        logger.info("\nAssessment completed. Results:")
        
        # 处理成功的项目
        if results.successful:
            logger.info("\nSuccessful items:")
            for idx, item in enumerate(results.successful, 1):
                logger.info(f"Result {idx}:")
                logger.info(f"Instruction: {item.instruction}")
                logger.info(f"Input: {item.input}")
                logger.info(f"Output: {item.output}")
                logger.info(f"Quality Metrics: {item.quality_metrics}")
                logger.info(f"Score: {item.score}")
                logger.info(f"Category: {item.category}")
                logger.info(f"Metadata: {item.metadata}\n")
        
        # 处理失败的项目
        if results.failed:
            logger.info("\nFailed items:")
            for i, item in enumerate(results.failed, 1):
                logger.info(f"Failed item {i}: {item.instruction}")
        
        # 批处理统计
        logger.info("\nBatch Statistics:")
        for metric, value in results.metrics.items():
            logger.info(f"{metric}: {value}")
            
        # 保存结果
        task_name = get_task_name()
        output_dir = project_root / "outputs"
        save_results(results.successful, task_name, output_dir, test_config)
        
        assert len(results.successful) > 0, "没有成功处理的数据项"
        assert len(results.failed) == 0, "存在处理失败的数据项"
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        raise

async def main():
    """主函数"""
    try:
        # 加载配置
        config_path = project_root / "config" / "test_config.yaml"
        config = load_config(config_path)
        
        # 加载测试数据
        test_items = await load_test_data(config)
        if not test_items:
            logger.error("没有加载到测试数据")
            return
            
        # 运行测试
        task_name = get_task_name()
        result = await test_small_batch(config, test_items)
        
        # 保存结果
        output_dir = project_root / "outputs"
        save_results(result.successful, task_name, output_dir, config)
        
        logger.info(f"测试完成! 成功处理 {len(result.successful)} 个样本")
        
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")
        logger.debug("Error details:", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
