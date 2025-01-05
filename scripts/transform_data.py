#!/usr/bin/env python
import argparse
import asyncio
import logging
import logging.config
import os
import yaml
from pathlib import Path
from datetime import datetime
from src.data_transformer import DataTransformer
from src.data_loader import DataLoader

# 配置日志记录
def setup_logging():
    """设置日志配置"""
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging_config_path = Path("config/logging.yaml")
    if logging_config_path.exists():
        with open(logging_config_path, 'r') as f:
            config = yaml.safe_load(f)
            # 确保日志目录存在
            Path("logs").mkdir(exist_ok=True)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)

def load_config(config_path: str) -> dict:
    """加载配置文件。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_output_path(config: dict) -> Path:
    """根据配置生成输出路径。"""
    output_config = config.get('output', {})
    base_dir = output_config.get('base_dir', 'output')
    file_naming = output_config.get('file_naming', {})
    
    # 获取文件命名组件
    task_name = config.get('task_name', 'unknown')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 构建文件名
    filename = f"{task_name}_{timestamp}.json"
    return Path(base_dir) / filename

async def main():
    """主函数"""
    # 设置日志
    setup_logging()
    parser = argparse.ArgumentParser(description='Transform dataset using LLM.')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    transformer = None
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 加载数据集
        data_loader = DataLoader(config)
        dataset = data_loader.load_and_convert()
        
        # 初始化转换器并传入数据集
        transformer = DataTransformer(config, dataset)
        
        # 执行转换
        try:
            transformed_items = await transformer.transform_dataset()
            logger = logging.getLogger(__name__)
            logger.info("Data transformation completed successfully")
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error during data transformation: {e}")
            raise
            
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error during data transformation: {e}")
        raise
    finally:
        # 确保关闭连接
        if transformer:
            await transformer.close()

if __name__ == "__main__":
    asyncio.run(main())
