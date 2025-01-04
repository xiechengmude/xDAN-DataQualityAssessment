import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from datasets import Dataset
from ..models import ProcessedItem

logger = logging.getLogger(__name__)

class DatasetManager:
    def __init__(self, output_dir: str, output_config: Dict[str, Any]):
        """初始化数据集管理器
        
        Args:
            output_dir: 输出目录
            output_config: 输出配置
        """
        self.output_dir = output_dir
        self.output_config = output_config
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

    def save_locally(self, data: List[Dict], prefix: str, dataset_name: str) -> str:
        """保存数据集到本地
        
        Args:
            data: 要保存的数据
            prefix: 文件名前缀
            dataset_name: 数据集名称
            
        Returns:
            str: 保存的文件路径
        """
        # 构建输出路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{dataset_name}_{timestamp}.json"
        output_path = os.path.join(self.output_dir, filename)
        
        # 保存到本地
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Dataset saved locally to: {output_path}")
        
        return output_path

    def push_to_hub(self, data: List[Dict], dataset_name: str) -> Optional[Dataset]:
        """推送数据集到Hugging Face Hub
        
        Args:
            data: 要推送的数据
            dataset_name: 数据集名称
            
        Returns:
            Optional[Dataset]: 推送成功返回Dataset对象，失败返回None
        """
        if not self.output_config.get("push_to_hub"):
            return None
            
        try:
            # 构建完整的repository_id
            hub_config = self.output_config.get("hub_config", {})
            base_repo = hub_config.get("repository_id", "").split('/')[0]
            repository_id = f"{base_repo}/{dataset_name}"
            
            # 创建Dataset对象
            dataset = Dataset.from_list(data)
            
            # 推送到Hub
            dataset.push_to_hub(
                repository_id,
                private=hub_config.get("private", True),
                token=hub_config.get("token")
            )
            logger.info(f"Dataset pushed to Hugging Face Hub: {repository_id}")
            return dataset
        except Exception as e:
            logger.error(f"Failed to push dataset to Hugging Face Hub: {str(e)}")
            return None

import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import datasets
from huggingface_hub import HfApi, Repository
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class AlpacaItem:
    instruction: str
    input: str
    output: str

class DatasetManager:
    def __init__(self, 
                 output_dir: str = "outputs",
                 output_config: Optional[Dict] = None):
        """初始化数据集管理器
        
        Args:
            output_dir: 本地输出目录
            output_config: 输出配置字典，包含save_local, push_to_hub等选项
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置默认配置
        self.output_config = output_config or {
            "save_local": True,
            "push_to_hub": False,
            "hub_config": {
                "repository_id": "",
                "private": True,
                "token": ""
            }
        }
        
        # 从环境变量获取token，如果配置中没有提供
        if not self.output_config["hub_config"]["token"]:
            self.output_config["hub_config"]["token"] = os.environ.get("HF_TOKEN")
        
    def save_locally(self, 
                    data: List[Dict[str, Any]], 
                    prefix: str = "assessment",
                    dataset_name: str = None) -> str:
        """保存数据到本地文件
        
        Args:
            data: 要保存的数据
            prefix: 文件名前缀
            dataset_name: 数据集名称
        
        Returns:
            保存的文件路径
        """
        if not self.output_config["save_local"]:
            return None
            
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 构建文件名
        filename = f"{timestamp}_{prefix}"
        if dataset_name:
            filename = f"{filename}_{dataset_name}"
        filename = f"{filename}.json"
        
        # 保存文件
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        return str(output_path)
    
    def push_to_hub(self, 
                   data: List[Dict[str, Any]], 
                   dataset_name: str) -> Optional[datasets.DatasetDict]:
        """将数据推送到 Hugging Face Hub
        
        Args:
            data: 要推送的数据
            dataset_name: 数据集名称，例如 "alpaca_eval"
        
        Returns:
            datasets.DatasetDict 对象，如果不推送则返回None
        """
        if not self.output_config["push_to_hub"]:
            return None
            
        hub_config = self.output_config["hub_config"]
        if not hub_config["token"]:
            raise ValueError("需要在配置中提供 Hugging Face token 或设置 HF_TOKEN 环境变量")
            
        if not hub_config["repository_id"]:
            raise ValueError("需要在配置中提供 repository_id")
            
        # 创建数据集
        dataset_dict = {}
        
        # 根据数据源分组
        sources = set(item.get('sources', 'unknown') for item in data)
        for source in sources:
            source_data = [item for item in data if item.get('sources', 'unknown') == source]
            if source_data:
                dataset_dict[source] = datasets.Dataset.from_list(source_data)
        
        # 创建 DatasetDict
        full_dataset = datasets.DatasetDict(dataset_dict)
        
        # 推送到hub
        repository_id = hub_config["repository_id"]
        if not "/" in repository_id:
            repository_id = f"{repository_id}/{dataset_name}"
            
        full_dataset.push_to_hub(
            repository_id,
            token=hub_config["token"],
            private=hub_config["private"],
            commit_message=f"Update dataset: {dataset_name}"
        )
        
        return full_dataset

    def save_dataset(self, 
                    data: List[Dict], 
                    task_name: str) -> None:
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
                dataset = datasets.Dataset.from_list(data)
                
                # 推送到Hub
                dataset.push_to_hub(
                    repository_id,
                    private=self.output_config["hub_config"]["private"],
                    token=self.output_config["hub_config"]["token"]
                )
                logger.info(f"Dataset pushed to Hugging Face Hub: {repository_id}")
            except Exception as e:
                logger.error(f"Failed to push dataset to Hugging Face Hub: {str(e)}")

    async def load_dataset(self, dataset_name: str, num_samples: int, field_mapping: Dict[str, str]) -> List[AlpacaItem]:
        """从Hugging Face Hub加载数据集
        
        Args:
            dataset_name: 数据集名称
            num_samples: 样本数量
            field_mapping: 字段映射，如 {"instruction": "prompt", "input": "input", "output": "output"}
            
        Returns:
            List[AlpacaItem]: 数据项列表
        """
        try:
            # 加载数据集
            dataset = datasets.load_dataset(
                dataset_name,
                split="train",
                streaming=True
            )
            
            # 获取指定数量的样本
            items = []
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break
                    
                try:
                    # 根据字段映射创建 AlpacaItem
                    instruction = item[field_mapping["instruction"]] if field_mapping["instruction"] else ""
                    input_text = item[field_mapping["input"]] if field_mapping["input"] else ""
                    output = item[field_mapping["output"]] if field_mapping["output"] else ""
                    
                    alpaca_item = AlpacaItem(
                        instruction=instruction,
                        input=input_text,
                        output=output
                    )
                    items.append(alpaca_item)
                except KeyError as e:
                    logger.warning(f"样本 {i} 缺少必要字段: {e}")
                    continue
                    
            return items
            
        except Exception as e:
            logger.error(f"加载数据集 {dataset_name} 时出错: {e}")
            raise
