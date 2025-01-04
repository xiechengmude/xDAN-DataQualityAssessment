import asyncio
import logging
import os
import sys
from pathlib import Path
import datasets

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.processor import DataProcessor
from src.models import AlpacaItem
from src.utils.dataset_utils import DatasetManager

async def test_single_item():
    """测试单个数据项的处理"""
    # 加载配置
    config_path = project_root / "config" / "default_config.yaml"
    processor = DataProcessor(str(config_path))
    
    # 创建测试数据
    test_item = AlpacaItem(
        instruction="解释什么是机器学习",
        input="",
        output="机器学习是人工智能的一个分支，它使用统计技术让计算机系统能够从数据中学习和改进，而无需被明确编程。机器学习算法通过分析大量的训练数据来识别模式，并使用这些模式做出预测或决策。"
    )
    
    # 处理数据
    print("\n开始处理测试数据...")
    result = await processor.process_single_item(test_item, "test_dataset", 1)
    
    # 打印结果
    print("\n处理结果:")
    print(f"质量指标: {result.quality_metrics}")
    print(f"总分: {result.score}")
    print(f"分类: {result.category}")
    print(f"处理后输出: {result.processed_output}")
    print(f"处理元数据: {result.metadata}")
    
    # 转换结果为字典格式
    result_dict = {
        "id": result.id,
        "source": result.source,
        "instruction": test_item.instruction,
        "input": test_item.input,
        "output": test_item.output,
        "quality_metrics": result.quality_metrics,
        "score": result.score,
        "category": result.category,
        "processed_output": result.processed_output,
        "metadata": result.metadata
    }
    
    # 保存结果
    dataset_manager = DatasetManager(
        output_dir=project_root / "outputs",
        output_config=processor.config.get("output", {})
    )
    
    # 保存结果
    output_path = dataset_manager.save_locally(
        [result_dict],
        prefix="test_assessment",
        dataset_name="alpaca"
    )
    if output_path:
        print(f"\n结果已保存到: {output_path}")
        
    # 推送到Hugging Face Hub
    try:
        dataset = dataset_manager.push_to_hub(
            [result_dict],
            dataset_name="alpaca"
        )
        if dataset:
            print("\n数据已成功推送到 Hugging Face Hub")
    except Exception as e:
        print(f"\n推送到 Hugging Face Hub 失败: {e}")
    
    return result

async def test_category_classification():
    """测试新的分类系统"""
    # 加载配置
    config_path = project_root / "config" / "default_config.yaml"
    processor = DataProcessor(str(config_path))
    
    # 创建数据集管理器并加载数据
    dataset_manager = DatasetManager(output_dir=project_root / "outputs")
    
    # 从配置中加载数据集
    datasets_config = processor.config.get("datasets", [])
    if not datasets_config:
        raise ValueError("配置文件中未找到数据集配置")
    
    print("\n开始测试分类系统...")
    
    for dataset_config in datasets_config:
        dataset_name = dataset_config["name"]
        num_samples = dataset_config["num_samples"]
        field_mapping = dataset_config["field_mapping"]
        
        print(f"\n尝试加载数据集: {dataset_name}")
        try:
            # 加载数据集
            dataset = await dataset_manager.load_dataset(
                dataset_name=dataset_name,
                num_samples=num_samples,
                field_mapping=field_mapping
            )
            
            results = []
            for i, item in enumerate(dataset, 1):
                print(f"\n测试用例 {i} (来自 {dataset_name}):")
                print(f"指令: {item.instruction}")
                print(f"输入: {'无' if not item.input else item.input}")
                print(f"输出: {item.output}\n")
                
                # 处理数据
                result = await processor.process_single_item(item, dataset_name, i)
                results.append({
                    "id": result.id,
                    "source": result.source,
                    "instruction": item.instruction,
                    "input": item.input,
                    "output": item.output,
                    "quality_metrics": result.quality_metrics,
                    "score": result.score,
                    "category": result.category,
                    "processed_output": result.processed_output,
                    "metadata": result.metadata
                })
                
                print("分类结果:")
                print(f"类别: {result.category}")
                print(f"质量评分: {result.score:.2f}")
                print(f"评估说明: {result.metadata['validation_notes']}")
                print("-" * 50)
            
            # 保存结果
            if results:
                output_path = dataset_manager.save_locally(
                    results,
                    prefix="test_assessment",
                    dataset_name=dataset_name.replace("/", "_")
                )
                if output_path:
                    print(f"\n结果已保存到: {output_path}")
                
                # 推送到Hugging Face Hub
                try:
                    dataset = dataset_manager.push_to_hub(
                        results,
                        dataset_name=dataset_name.replace("/", "_")
                    )
                    if dataset:
                        print("\n数据已成功推送到 Hugging Face Hub")
                except Exception as e:
                    print(f"\n推送到 Hugging Face Hub 失败: {e}")
                    
        except Exception as e:
            print(f"处理数据集 {dataset_name} 时出错: {e}")
            continue
            
    print("\n测试完成!")

async def main():
    """主测试函数"""
    print("开始测试数据处理器...")
    await test_single_item()
    await test_category_classification()
    print("\n测试完成!")

if __name__ == "__main__":
    asyncio.run(main())
