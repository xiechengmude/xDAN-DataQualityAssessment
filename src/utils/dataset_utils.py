import os
import json
import logging
from typing import List, Dict, Any
from datasets import Dataset
from ..models import ProcessedItem

logger = logging.getLogger(__name__)

class DatasetUtils:
    def __init__(self, output_dir: str, output_config: Dict[str, Any]):
        """初始化数据集工具
        
        Args:
            output_dir: 输出目录
            output_config: 输出配置
        """
        self.output_dir = output_dir
        self.output_config = output_config
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

    def save_dataset(self, data: List[Dict], task_name: str) -> None:
        """保存数据集到本地和/或Hugging Face Hub"""
        # 构建输出路径
        output_path = os.path.join(self.output_dir, f"{task_name}.json")
        
        # 保存到本地
        if self.output_config["save_local"]:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Dataset saved locally to: {output_path}")
        
        # 推送到Hugging Face Hub
        if self.output_config["push_to_hub"]:
            # 构建完整的repository_id
            base_repo = self.output_config["hub_config"]["repository_id"].split('/')[0]
            repository_id = f"{base_repo}/{task_name}"
            
            try:
                # 创建Dataset对象
                dataset = Dataset.from_list(data)
                
                # 推送到Hub
                dataset.push_to_hub(
                    repository_id,
                    private=self.output_config["hub_config"]["private"],
                    token=self.output_config["hub_config"]["token"]
                )
                logger.info(f"Dataset pushed to Hugging Face Hub: {repository_id}")
            except Exception as e:
                logger.error(f"Failed to push dataset to Hugging Face Hub: {str(e)}")

    def load_dataset(self, dataset_name: str) -> Dataset:
        """从Hugging Face Hub加载数据集"""
        try:
            # 从Hub加载数据集
            dataset = Dataset.from_hub(
                dataset_name,
                token=self.output_config["hub_config"]["token"]
            )
            logger.info(f"Dataset loaded from Hugging Face Hub: {dataset_name}")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load dataset from Hugging Face Hub: {str(e)}")
            raise
