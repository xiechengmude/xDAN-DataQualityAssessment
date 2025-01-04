from typing import Dict, Any, Optional, List, Union
from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd
from pathlib import Path
import yaml
import logging
import os
import random
import time

from .models import AlpacaItem

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
    """加载和处理数据集的类。"""
    
    def __init__(self, config):
        """初始化数据加载器。"""
        self.config = config if isinstance(config, dict) else self._load_config(config)
        self.dataset_configs = [DatasetConfig(cfg) for cfg in self.config['datasets']]
        self.common_config = self.config['dataset_common']
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

    def load_single_dataset(self, config: DatasetConfig) -> Dataset:
        """加载单个数据集"""
        logger.info(
            f"Loading dataset: {config.name}, "
            f"config: {config.config or 'default'}, "
            f"split: {config.split}, subset: {config.subset or 'None'}"
        )

        # 构建加载参数
        load_params = {
            "path": config.name,
            "split": config.split,
        }
        
        if config.config:
            load_params["name"] = config.config
            
        if config.subset:
            if config.config:
                load_params["name"] = f"{config.config}/{config.subset}"
            else:
                load_params["name"] = config.subset

        try:
            dataset = load_dataset(**load_params)
            
            # 验证字段映射
            self._validate_field_mapping(dataset, config.field_mapping)
            
            # 随机抽样
            if config.num_samples > 0 and config.num_samples < len(dataset):
                dataset = dataset.shuffle(seed=self.common_config['shuffle_seed'])
                dataset = dataset.select(range(config.num_samples))
            
            logger.info(f"Successfully loaded {len(dataset)} items from {config.name}")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset {config.name}: {e}")
            raise

    def _validate_field_mapping(self, dataset: Dataset, field_mapping: Dict[str, str]):
        """验证字段映射是否有效"""
        dataset_columns = dataset.column_names
        for field, mapped_name in field_mapping.items():
            if mapped_name and mapped_name not in dataset_columns:
                raise ValueError(
                    f"Mapped field '{mapped_name}' not found in dataset. "
                    f"Available columns: {dataset_columns}"
                )

    def convert_to_alpaca_format(self, 
                               dataset: Dataset, 
                               field_mapping: Dict[str, str]) -> List[AlpacaItem]:
        """将数据集转换为AlpacaItem格式"""
        items = []
        for row in dataset:
            try:
                item = AlpacaItem(
                    instruction=row[field_mapping['instruction']],
                    input=row[field_mapping['input']] if field_mapping.get('input') else "",
                    output=row[field_mapping['output']]
                )
                items.append(item)
            except Exception as e:
                logger.warning(f"Failed to convert row: {row}. Error: {e}")
                continue
        
        logger.info(f"Converted {len(items)} items to AlpacaItem format")
        return items

    def load_and_convert(self) -> List[AlpacaItem]:
        """加载并转换所有配置的数据集"""
        all_items = []
        
        for dataset_config in self.dataset_configs:
            try:
                dataset = self.load_single_dataset(dataset_config)
                items = self.convert_to_alpaca_format(dataset, dataset_config.field_mapping)
                all_items.extend(items)
            except Exception as e:
                logger.error(f"Failed to process dataset {dataset_config.name}: {e}")
                continue
        
        if self.common_config.get('combine_datasets', True):
            # 打乱合并后的数据
            random.seed(self.common_config['shuffle_seed'])
            random.shuffle(all_items)
        
        logger.info(f"Total items loaded and converted: {len(all_items)}")
        return all_items

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
