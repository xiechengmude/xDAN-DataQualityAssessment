import asyncio
import logging
import os
import sys
from pathlib import Path
import datasets
from typing import List, Optional
import pytest

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.processor import DataProcessor
from src.models import AlpacaItem
from src.utils.dataset_utils import DatasetManager
from tests.test_assessment_demo import load_test_data, save_results, get_task_name, load_config

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # 强制重新配置日志
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 确保pytest不会捕获日志
@pytest.fixture(autouse=True)
def setup_logging():
    logging.getLogger().handlers = []
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )

@pytest.mark.asyncio
async def test_single_item():
    """测试单个数据项的处理"""
    # 设置日志级别
    logging.basicConfig(level=logging.DEBUG,
                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.DEBUG)
    
    logger.debug("开始单项测试...")
    
    # 加载配置
    config_path = project_root / "config" / "test_config.yaml"
    config = load_config(config_path)
    logger.info(f"已加载配置文件: {config_path}")
    
    processor = DataProcessor(config)
    
    # 从测试配置加载数据
    test_items = await load_test_data(config)
    if not test_items:
        logger.error("没有找到任何测试数据")
        return
        
    test_item = test_items[0]  # 使用第一个测试项
    logger.info(f"测试数据: {test_item.instruction}")
    
    # 处理数据
    logger.info("开始处理测试数据...")
    logger.info(f"测试项内容: {test_item.instruction}, {test_item.input}, {test_item.output}")
    result = await processor.process_single_item(test_item, "test_dataset", 1)
    
    # 打印结果
    logger.info("\n处理结果:")
    logger.info(f"质量指标:\n{result.quality_metrics}")
    logger.info(f"总分: {result.score}")
    logger.info(f"分类: {result.category}")
    logger.info(f"处理后输出:\n{result.processed_output}")
    logger.info(f"处理元数据:\n{result.metadata}")
    
    # 生成任务名称
    task_name = get_task_name()
    
    # 保存结果
    output_dir = project_root / "outputs"
    logger.info(f"保存结果到: {output_dir}")
    save_results([result], task_name, output_dir, config)
    
    # 验证结果
    assert result is not None
    assert result.quality_metrics is not None
    assert result.score >= 0
    assert result.category is not None
    
    logger.info("单项测试完成")
    return result

@pytest.mark.asyncio
async def test_category_classification():
    """测试新的分类系统"""
    logger.info("开始分类系统测试...")
    
    # 加载配置
    config_path = project_root / "config" / "test_config.yaml"
    config = load_config(config_path)
    logger.info(f"已加载配置文件: {config_path}")
    
    processor = DataProcessor(config)
    
    # 从测试配置加载数据
    test_items = await load_test_data(config)
    if not test_items:
        logger.error("没有找到任何测试数据")
        return
        
    logger.info("\n开始测试分类系统...")
    
    results = []
    for i, item in enumerate(test_items, 1):
        try:
            logger.info(f"\n测试用例 {i}:")
            logger.info(f"指令: {item.instruction}")
            logger.info(f"输入: {'无' if not item.input else item.input}")
            logger.info(f"输出: {item.output}\n")
            
            # 处理数据
            result = await processor.process_single_item(item, "test_dataset", i)
            results.append(result)
            
            logger.info("分类结果:")
            logger.info(f"类别: {result.category}")
            logger.info(f"质量评分: {result.score}")
            logger.info(f"评估说明: {result.metadata['validation_notes']}")
            logger.info("-" * 50)
            
        except Exception as e:
            logger.error(f"处理测试用例 {i} 时出错: {str(e)}")
            continue
    
    # 确保至少有一个成功的结果
    assert len(results) > 0, "所有测试用例都失败了"
    
    # 验证每个成功的结果
    for result in results:
        assert result is not None
        assert result.quality_metrics is not None
        assert result.score >= 0
        assert result.category is not None
        assert 'model_name' in result.metadata
        assert 'timestamp' in result.metadata
        assert result.sources == "test_dataset"  # 验证数据源正确
    
    # 生成任务名称
    task_name = get_task_name()
    
    # 保存结果
    output_dir = project_root / "outputs"
    logger.info(f"保存结果到: {output_dir}")
    save_results(results, task_name, output_dir, config)
    
    logger.info("测试完成!")
