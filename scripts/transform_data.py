#!/usr/bin/env python
import asyncio
import argparse
import logging
from pathlib import Path
from src.data_transformer import DataTransformer
from src.utils.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    parser.add_argument('--push-to-hub', action='store_true', help='Push results to HuggingFace Hub')
    args = parser.parse_args()

    try:
        # 加载配置
        config = load_config(args.config)
        
        # 初始化转换器
        transformer = DataTransformer(config)
        
        # 转换数据
        transformed_items = await transformer.transform_dataset()
        
        # 保存结果
        output_path = Path(args.output)
        transformer.save_results(transformed_items, output_path)
        
        # 如果指定了push-to-hub，则上传结果
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
