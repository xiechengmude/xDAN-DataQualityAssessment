#!/usr/bin/env python
import argparse
import asyncio
import logging
import os
import yaml
from pathlib import Path
from datetime import datetime
from src.data_transformer import DataTransformer
from src.data_loader import DataLoader

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
    base_dir = output_config.get('base_dir', 'output')  # 修改默认值为 'output'
    file_naming = output_config.get('file_naming', {})
    
    # 获取文件命名组件
    dataset_name = config.get('dataset', {}).get('name', 'unknown')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 构建文件名
    filename = f"{dataset_name}_{timestamp}.json"
    return Path(base_dir) / filename

async def main():
    parser = argparse.ArgumentParser(description='Transform dataset using LLM.')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--no-push-to-hub', action='store_true', help='Do not push results to HuggingFace Hub')
    parser.add_argument('--save-local', action='store_true', help='Save results to local file')
    args = parser.parse_args()

    try:
        # 加载配置
        config = load_config(args.config)
        
        # 加载数据集
        data_loader = DataLoader(config)
        dataset = data_loader.load_and_convert()
        
        # 初始化转换器并传入数据集
        transformer = DataTransformer(config, dataset)
        
        # 转换数据
        transformed_items = await transformer.transform_dataset()
        logger.info(f"Successfully transformed {len(transformed_items)} items")
        
        # 获取输出文件路径
        output_file = transformer._get_output_file_path()
        
        # 根据配置和命令行参数决定是否保存到本地
        save_local = args.save_local or config.get('output', {}).get('save_local', False)
        if save_local:
            transformer._save_to_json(transformed_items, output_file)
            logger.info(f"Saved {len(transformed_items)} transformed items to {output_file}")
        
        # 默认推送到Hub，除非指定了--no-push-to-hub
        if not args.no_push_to_hub:
            logger.info("Pushing results to HuggingFace Hub...")
            await transformer.push_to_hub(output_file)
            logger.info("Results pushed to HuggingFace Hub successfully!")
        
        logger.info("Data transformation and analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during data transformation: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
