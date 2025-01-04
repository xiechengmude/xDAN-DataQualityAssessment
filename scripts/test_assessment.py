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

def save_results(results: BatchResult, task_name: str, output_dir: Path, config: dict):
    """保存评估结果
    
    Args:
        results: 批处理结果
        task_name: 任务名称
        output_dir: 输出目录
        config: 配置信息
    """
    timestamp = datetime.now()
    
    # 获取数据集配置
    dataset_configs = config.get('datasets', [])
    dataset_sources = {
        item.get('field_mapping', {}).get('instruction'): item.get('name', '')
        for item in dataset_configs
    }
    
    # 转换为增强的Alpaca格式
    enhanced_items = results.to_enhanced_alpaca_format(
        model_name=config['openai']['model_name'],
        task_name=task_name,
        timestamp=timestamp
    )
    
    # 添加sources信息
    for item in enhanced_items:
        # 根据instruction字段匹配数据源
        for field, source in dataset_sources.items():
            if field and field in item.instruction:
                item.sources = source
                break
    
    # 保存为Alpaca格式
    alpaca_file = output_dir / f"{task_name}_alpaca.json"
    logger.info(f"Saving Alpaca format results to: {alpaca_file}")
    with open(alpaca_file, 'w', encoding='utf-8') as f:
        json.dump(
            [item.model_dump() for item in enhanced_items],
            f,
            ensure_ascii=False,
            indent=2
        )
    
    # 保存原始评估结果（包含更多元信息）
    eval_file = output_dir / f"{task_name}_eval.json"
    logger.info(f"Saving evaluation details to: {eval_file}")
    eval_data = {
        "task_name": task_name,
        "timestamp": timestamp.isoformat(),
        "model_name": config['openai']['model_name'],
        "metrics": results.metrics,
        "metadata": results.metadata.model_dump(),
        "config": {
            k: v for k, v in config.items()
            if k not in ['openai']  # 排除敏感信息
        }
    }
    with open(eval_file, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)

async def test_small_batch(config: dict, test_items: list) -> BatchResult:
    """测试小批量数据评估
    
    Args:
        config: 配置信息
        test_items: 要评估的测试数据列表
    """
    try:
        # 初始化处理器
        processor = DataProcessor(config)
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
            idx = 0
            for item in results.successful:
                logger.info(f"Result {idx + 1}:")
                logger.info(f"Instruction: {item.instruction}")
                logger.info(f"Input: {item.input}")
                logger.info(f"Output: {item.output}")
                logger.info(f"Quality Metrics: {item.quality_metrics}")
                logger.info(f"Score: {item.score}")
                logger.info(f"Category: {item.category}")
                logger.info(f"Metadata: {item.metadata}\n")
                idx += 1
        
        # 处理失败的项目
        if results.failed:
            logger.info("\nFailed items:")
            for i, item in enumerate(results.failed, 1):
                logger.info(f"Failed item {i}: {item.instruction}")
        
        # 批处理统计
        logger.info("\nBatch Statistics:")
        for metric, value in results.metrics.items():
            logger.info(f"{metric}: {value}")

        return results

    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        raise

def load_hf_dataset(dataset_config: dict) -> List[AlpacaItem]:
    """从HuggingFace加载数据集
    
    Args:
        dataset_config: 数据集配置信息
        
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
        
        # 如果指定了子集
        if dataset_config.get('subset'):
            dataset = dataset.filter(lambda x: x[dataset_config['subset']])
            
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
                    output=str(item.get(output_field, ''))
                )
                items.append(alpaca_item)
            except Exception as e:
                logger.warning(f"跳过无效数据项: {str(e)}")
                continue
                
        return items
        
    except Exception as e:
        logger.error(f"加载数据集失败: {str(e)}")
        return []

def load_test_data(config: dict, json_file: Optional[str] = None) -> List[AlpacaItem]:
    """加载测试数据
    
    Args:
        config: 配置信息
        json_file: 可选的JSON文件路径
        
    Returns:
        List[AlpacaItem]: 加载的数据项列表
    """
    items = []
    
    # 如果提供了JSON文件，优先加载
    if json_file:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                
            items.extend([
                AlpacaItem(
                    instruction=item.get('instruction', ''),
                    input=item.get('input', ''),
                    output=item.get('output', '')
                )
                for item in json_data
            ])
            
            logger.info(f"从JSON文件加载了 {len(items)} 个测试项")
            
        except Exception as e:
            logger.error(f"加载JSON文件失败: {str(e)}")
    
    # 从配置的数据集加载
    if not items and 'datasets' in config:
        for dataset_config in config['datasets']:
            dataset_items = load_hf_dataset(dataset_config)
            items.extend(dataset_items)
            logger.info(f"从数据集 {dataset_config['name']} 加载了 {len(dataset_items)} 个测试项")
    
    return items

def main():
    """主函数"""
    try:
        # 加载配置
        config_path = project_root / "config" / "default_config.yaml"
        config = load_config(config_path)
        
        # 获取JSON文件路径（如果提供）
        json_file = sys.argv[1] if len(sys.argv) > 1 else None
        
        # 加载测试数据
        test_items = load_test_data(config, json_file)
        
        if not test_items:
            logger.error("没有找到任何测试数据")
            sys.exit(1)
            
        logger.info(f"总共加载了 {len(test_items)} 个测试项")
        
        # 生成任务名称
        task_name = get_task_name()
        logger.info(f"Task name: {task_name}")
        
        # 运行测试
        results = asyncio.run(test_small_batch(config, test_items))
        
        # 保存结果
        output_dir = project_root / "outputs"
        save_results(results, task_name, output_dir, config)
        
        logger.info("Test completed successfully")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
