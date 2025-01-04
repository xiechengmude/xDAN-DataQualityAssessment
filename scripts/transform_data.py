#!/usr/bin/env python
import argparse
import asyncio
import logging
import os
import yaml
from pathlib import Path
from datetime import datetime
from src.data_transformer import DataTransformer

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """加载配置文件。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_output_path(config: dict) -> Path:
    """根据配置生成输出路径。"""
    output_config = config.get('output', {})
    base_dir = output_config.get('base_dir', 'outputs')
    file_naming = output_config.get('file_naming', {})
    
    # 获取文件命名组件
    task_name = file_naming.get('task_name', 'data_transform')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 构建文件名
    filename_parts = [task_name]
    if file_naming.get('include_timestamp', True):
        filename_parts.append(timestamp)
    
    filename = '_'.join(filename_parts) + '.json'
    return Path(base_dir) / filename

async def main():
    parser = argparse.ArgumentParser(description='Transform dataset using LLM.')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--push-to-hub', action='store_true', help='Push results to HuggingFace Hub')
    args = parser.parse_args()

    try:
        # 加载配置
        config = load_config(args.config)
        
        # 确保输出目录存在
        output_path = get_output_path(config)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 初始化转换器
        transformer = DataTransformer(config)
        
        # 转换数据
        transformed_items = await transformer.transform_dataset()
        
        # 保存结果
        transformer.save_results(transformed_items, output_path)
        
        # 如果指定了--push-to-hub，则推送到Hub
        if args.push_to_hub:
            logger.info("Pushing results to HuggingFace Hub...")
            transformer.push_to_hub(output_path)
            logger.info("Results pushed to HuggingFace Hub successfully!")
        
        logger.info("Data transformation and analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during data transformation: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
