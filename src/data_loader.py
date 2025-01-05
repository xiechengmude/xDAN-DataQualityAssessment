from typing import Dict, Any, Optional, List, Union
from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd
from pathlib import Path
import yaml
import logging
import os
import random
import time
from datetime import datetime

from .data_types import AlpacaItem

logger = logging.getLogger(__name__)

class DatasetConfig:
    """数据集配置类"""
    def __init__(self, config: Dict[str, Any]):
        self.name = config['name']
        self.config = config.get('config', '')
        self.split = config.get('split', 'train')
        self.subset = config.get('subset', '')
        self.num_samples = config.get('num_samples', -1)
        self.field_mapping = config['field_mapping']

class DataLoader:
    """数据加载器，用于加载和转换数据。"""
    
    def __init__(self, config):
        """初始化加载器。"""
        self.config = config if isinstance(config, dict) else self._load_config(config)
        self.dataset_configs = [DatasetConfig(cfg) for cfg in self.config['datasets']]
        self.common_config = self.config['dataset_common']
        self.current_dataset_name = None
        self._setup_hf_cache()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # 处理环境变量
        if 'hf_cache_dir' in config['dataset_common']:
            config['dataset_common']['hf_cache_dir'] = os.path.expanduser(
                config['dataset_common']['hf_cache_dir']
            )
        return config

    def _setup_hf_cache(self):
        """设置HuggingFace缓存目录"""
        cache_dir = self.common_config.get('hf_cache_dir')
        if cache_dir:
            os.environ['HF_HOME'] = cache_dir
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

    def load_and_convert(self) -> List[AlpacaItem]:
        """加载并转换所有数据集。"""
        all_items = []
        
        # 加载每个配置的数据集
        for dataset_config in self.dataset_configs:
            self.current_dataset_name = dataset_config.name
            items = self._load_dataset(dataset_config)
            converted_items = self._convert_to_alpaca(items, dataset_config)
            all_items.extend(converted_items)
            
        logger.info(f"Total items loaded and converted: {len(all_items)}")
        return all_items
    
    def _load_dataset(self, dataset_config: DatasetConfig) -> Dataset:
        """加载单个数据集。"""
        try:
            logger.info(
                f"Loading dataset: {dataset_config.name}, "
                f"config: {dataset_config.config or 'default'}, "
                f"split: {dataset_config.split}, subset: {dataset_config.subset or 'None'}"
            )

            # 构建加载参数
            load_params = {
                "path": dataset_config.name,
                "split": dataset_config.split,
            }
            
            if dataset_config.config:
                load_params["name"] = dataset_config.config
                
            if dataset_config.subset:
                if dataset_config.config:
                    load_params["name"] = f"{dataset_config.config}/{dataset_config.subset}"
                else:
                    load_params["name"] = dataset_config.subset

            # 加载数据集
            dataset = load_dataset(**load_params, cache_dir=self.common_config.get('hf_cache_dir', None))
            
            # 只有当num_samples > 0时才限制样本数量
            if dataset_config.num_samples > 0:
                dataset = dataset.shuffle(seed=self.common_config.get('shuffle_seed', 42))
                dataset = dataset.select(range(dataset_config.num_samples))
            
            logger.info(f"Successfully loaded {len(dataset)} items from {dataset_config.name}")
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_config.name}: {str(e)}")
            raise
    
    def _convert_to_alpaca(self, dataset: Dataset, dataset_config: DatasetConfig) -> List[AlpacaItem]:
        """将数据集转换为Alpaca格式。"""
        try:
            converted_items = []
            field_mapping = dataset_config.field_mapping
            
            for row in dataset:
                # 创建基础元数据
                metadata = {
                    'dataset_name': dataset_config.name,
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                }
                
                # 添加其他可能的元数据
                for key, value in row.items():
                    if key not in field_mapping.values():
                        metadata[key] = value
                
                # 创建AlpacaItem
                alpaca_item = AlpacaItem(
                    instruction=row[field_mapping['instruction']],
                    input=row.get(field_mapping['input'], '') if field_mapping.get('input') else None,
                    output=row[field_mapping['output']],
                    metadata=metadata
                )
                converted_items.append(alpaca_item)
            
            logger.info(f"Converted {len(converted_items)} items to AlpacaItem format")
            return converted_items
            
        except Exception as e:
            logger.error(f"Error converting dataset to Alpaca format: {str(e)}")
            raise

    def generate_output_filename(self, data_length: int) -> str:
        """生成输出文件名"""
        naming_config = self.config['output']['file_naming']
        parts = []
        
        # 添加任务名称
        parts.append(naming_config['task_name'])
        
        # 添加数据集信息
        if naming_config.get('include_dataset_info', True):
            dataset_names = [
                config.name.split('/')[-1]  # 只取数据集名称的最后一部分
                for config in self.dataset_configs
            ]
            parts.append(f"datasets_{'-'.join(dataset_names)}")
        
        # 添加样本数量
        if naming_config.get('include_sample_count', True):
            parts.append(f"samples_{data_length}")
        
        # 添加时间戳
        if naming_config.get('include_timestamp', True):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            parts.append(timestamp)
        
        # 组合文件名
        filename = "_".join(parts)
        extension = self.config['output']['save_format']
        return f"{filename}.{extension}"

    def save_processed_data(self, data: List[Any], output_dir: str):
        """保存处理后的数据"""
        output_format = self.config['output']['save_format']
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名
        filename = self.generate_output_filename(len(data))
        full_path = output_path / filename
        
        # 保存数据
        if output_format == 'jsonl':
            with open(full_path, 'w') as f:
                for item in data:
                    f.write(item.json() + '\n')
        elif output_format == 'json':
            pd.DataFrame([item.dict() for item in data]).to_json(
                full_path, orient='records', indent=2
            )
        elif output_format == 'parquet':
            pd.DataFrame([item.dict() for item in data]).to_parquet(full_path)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        logger.info(f"Saved {len(data)} items to {full_path}")
