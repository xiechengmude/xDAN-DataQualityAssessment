"""数据转换模块"""
import json
import logging
import os
from pathlib import Path
import yaml
import asyncio
from openai import AsyncOpenAI
from dataclasses import asdict
from datetime import datetime
from datasets import Dataset, load_dataset, concatenate_datasets
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple

from .data_types import AlpacaItem, RefinedAlpacaItem, TokenInfo
from .data_loader import DataLoader, DatasetConfig

logger = logging.getLogger(__name__)

class DataTransformer:
    """数据转换器类"""
    
    def __init__(self, config: Dict[str, Any], dataset: List[AlpacaItem] = None):
        """初始化转换器
        
        Args:
            config: 配置信息
            dataset: 数据集
        """
        self.config = config
        self.dataset = dataset
        self.current_id = 0  # 添加id计数器
        
        # 初始化OpenAI客户端
        openai_config = config.get('openai', {})
        self._openai_client = AsyncOpenAI(
            base_url=openai_config['api_base'],  # 直接使用配置文件中的api_base
            api_key=openai_config['api_key'],
            timeout=openai_config.get('timeout', 60.0),  # 超时设置
            max_retries=openai_config.get('max_retries', 2)  # 重试次数
        )
        
        # 加载默认模板
        template_name = config.get('transform', {}).get('prompt_template', 'structured_analysis')
        self.template = self._load_template(template_name)
        self.data_loader = DataLoader(self.config)
    
    async def close(self):
        """关闭OpenAI客户端连接"""
        await self._openai_client.close()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from yaml file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_template(self, template_name: str) -> str:
        """加载指定的prompt模板"""
        try:
            template_config = self.config['transform']['prompt_templates'].get(template_name)
            if not template_config:
                raise ValueError(f"Template '{template_name}' not found in configuration")
            
            # 使用相对于项目根目录的路径
            template_path = Path(template_config['path'])
            if not template_path.is_absolute():
                # 获取当前文件所在目录的父目录作为项目根目录
                project_root = Path(__file__).parent.parent
                template_path = project_root / template_path
            
            if not template_path.exists():
                raise FileNotFoundError(f"Template file not found: {template_path}")
            
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading template '{template_name}': {str(e)}")
            raise
    
    def _is_empty_value(self, value: Any) -> bool:
        """检查值是否为空
        
        处理以下情况：
        - None
        - 空字符串 ""
        - 只包含空白字符的字符串 "  \t\n"
        - 空列表 []
        - 空字典 {}
        """
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, (list, dict)):
            return len(value) == 0
        return False

    def validate_item(self, item: Dict[str, Any]) -> bool:
        """验证数据项是否满足要求"""
        if not self.config['transform'].get('validate_input', True):
            return True
            
        # 检查必需字段
        required_fields = self.config['transform'].get('required_fields', [])
        for field in required_fields:
            if field not in item or self._is_empty_value(item.get(field)):
                logger.warning(f"Required field '{field}' is missing or empty")
                return False
                
        # 检查可选字段 - 不影响验证结果
        optional_fields = self.config['transform'].get('optional_fields', [])
        for field in optional_fields:
            if field not in item:
                logger.debug(f"Optional field '{field}' not present")
            elif self._is_empty_value(item.get(field)):
                logger.debug(f"Optional field '{field}' is empty")
                
        return True

    async def refine_output(self, item: AlpacaItem) -> Tuple[str, TokenInfo]:
        """使用LLM优化输出"""
        openai_config = self.config.get('openai', {})
        model = openai_config.get('model_name', 'gpt-3.5-turbo')
        max_tokens = openai_config.get('max_tokens', 2000)
        temperature = openai_config.get('temperature', 0.7)
        
        try:
            # 加载prompt模板
            template_name = self.config.get('transform', {}).get('prompt_template', 'structured_analysis')
            prompt_template = self._load_template(template_name)

            # 准备输入数据，处理空值情况
            input_text = item.input
            if self._is_empty_value(input_text):
                input_text = ""
                logger.debug("Input field is empty, will be treated as no context")
            
            # 构造提示
            prompt = prompt_template.format(
                instruction=item.instruction,
                context=f"Context: {input_text}" if input_text else "",
                output=item.output
            )

            # 调用LLM
            client = self._get_client()
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # 提取回答
            answer = response.choices[0].message.content
            
            # 计算token使用情况
            token_info = TokenInfo(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                estimated_cost=self._calculate_cost({
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens
                })
            )
            
            return answer, token_info

        except Exception as e:
            logger.error(f"Error in refine_output: {str(e)}")
            raise
    
    def _get_client(self) -> AsyncOpenAI:
        """获取或创建OpenAI客户端"""
        return self._openai_client

    def _calculate_cost(self, usage: Dict[str, Any]) -> float:
        """计算预估成本"""
        # 从配置中获取价格信息
        pricing_config = self.config['openai'].get('pricing', {})
        input_price = pricing_config.get('input_price_per_million', 1.0) / 1_000_000
        output_price = pricing_config.get('output_price_per_million', 2.0) / 1_000_000
        currency = pricing_config.get('currency', 'CNY')
        
        # 计算预估成本
        estimated_cost = (usage['prompt_tokens'] * input_price) + (usage['completion_tokens'] * output_price)
        
        return estimated_cost
    
    async def transform_batch(self, items: List[AlpacaItem], start_idx: int = 0, dataset_names: List[str] = None) -> List[RefinedAlpacaItem]:
        """转换一批数据项"""
        tasks = []
        for i, item in enumerate(items, start=start_idx):
            item.id = i  # 设置id
            tasks.append(self._transform_item(item, dataset_name=dataset_names[i] if dataset_names else None))
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return [
                item for item in results 
                if not isinstance(item, Exception)
            ]
        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            raise

    async def _transform_item(self, alpaca_item: AlpacaItem, dataset_name: str = None) -> RefinedAlpacaItem:
        """Transform a single data item to include structured output."""
        try:
            # 验证输入
            item_dict = alpaca_item.dict()
            if not self.validate_item(item_dict):
                raise ValueError("Invalid input data")
            
            # 增加id
            self.current_id += 1
            alpaca_item.id = self.current_id
            
            refined_output, token_info = await self.refine_output(alpaca_item)
            return RefinedAlpacaItem.from_alpaca_item(
                item=alpaca_item,
                refined_output=refined_output,
                token_info=token_info,
                model_name=self.config.get('openai', {}).get('model_name', 'gpt-3.5-turbo'),
                sources=dataset_name if dataset_name else "unknown"
            )
        except Exception as e:
            logger.error(f"Error transforming item: {str(e)}")
            raise
    
    async def _transform_batch(self, batch: List[AlpacaItem]) -> List[RefinedAlpacaItem]:
        """转换一批数据"""
        try:
            # 并行处理所有items
            tasks = [
                self._transform_item(item, item.metadata.get('dataset_name', 'unknown'))
                for item in batch
            ]
            transformed_items = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 过滤出成功的结果
            return [
                item for item in transformed_items 
                if not isinstance(item, Exception)
            ]
        except Exception as e:
            logger.error(f"Error in batch transformation: {str(e)}")
            return []

    def _create_batches(self) -> List[List[AlpacaItem]]:
        """创建数据批次"""
        batch_size = self.config.get('concurrency', {}).get('batch_size', 100)
        batches = []
        current_batch = []
        
        for item in self.dataset:
            current_batch.append(item)
            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []
        
        # 添加最后一个不完整的批次
        if current_batch:
            batches.append(current_batch)
            
        return batches

    def _get_output_file_path(self, suffix: str = "") -> str:
        """获取输出文件路径。
        
        Args:
            suffix: 文件名后缀，用于区分不同类型的输出文件
        """
        output_config = self.config.get('output', {})
        base_dir = output_config.get('base_dir', 'output')
        task_name = self.config.get('task_name', 'unknown')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建输出目录
        os.makedirs(base_dir, exist_ok=True)
        
        # 构建文件名：task_taskName_时间戳_后缀.json
        filename = f"task_{task_name}_{timestamp}{suffix}.json"
        return os.path.join(base_dir, filename)

    def _save_to_json(self, items: List[RefinedAlpacaItem], output_file: str) -> None:
        """保存数据到JSON文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([item.dict() for item in items], f, ensure_ascii=False, indent=2)

    def _load_from_json(self, file_path: str) -> List[RefinedAlpacaItem]:
        """从JSON文件加载数据"""
        if not os.path.exists(file_path):
            return []
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return [RefinedAlpacaItem(**item) for item in data]

    def _merge_items(self, items1: List[RefinedAlpacaItem], items2: List[RefinedAlpacaItem]) -> List[RefinedAlpacaItem]:
        """合并两个数据列表，去除重复项"""
        # 使用id作为唯一标识
        id_map = {item.id: item for item in items1}
        # 更新或添加新项
        for item in items2:
            id_map[item.id] = item
        # 返回排序后的列表
        return sorted(id_map.values(), key=lambda x: x.id)

    async def transform_dataset(self) -> List[RefinedAlpacaItem]:
        """转换数据集"""
        try:
            if not self.dataset:
                self.dataset = self.data_loader.load_dataset()
            
            # 获取已处理的数据数量
            start_offset = 0
            try:
                # 尝试从checkpoint数据集获取已处理的数量
                split = self.dataset[0].metadata.get('split', 'train') if self.dataset else 'train'
                dataset_name = self.dataset[0].metadata.get('dataset_name', 'unknown').split('/')[-1] if self.dataset else 'unknown'
                repo_id = f"{self.config['output']['hub_config']['owner']}/{self.config['output']['hub_config']['repo_prefix']}-{dataset_name}-{self.config['task_name']}_{split}_checkpoint"
                repo_id = repo_id.replace("--", "-")  # 替换双横线
                checkpoint_dataset = load_dataset(repo_id, split="train")
                if checkpoint_dataset is not None:
                    start_offset = len(checkpoint_dataset)
                    self.current_id = start_offset
                    logger.info(f"Found existing checkpoint with {start_offset} items for dataset {dataset_name} split {split}, continuing from there")
            except Exception as e:
                logger.info(f"No existing checkpoint found: {e}")
            
            # 只处理未处理的数据
            remaining_dataset = self.dataset[start_offset:]
            if len(remaining_dataset) == 0:
                logger.info("All data has been processed, nothing to do")
                return []
            
            logger.info(f"Processing {len(remaining_dataset)} items (skipping first {start_offset} items)")
            
            transformed_items = []
            save_interval = self.config.get('output', {}).get('save_interval', 1000)
            push_interval = self.config.get('output', {}).get('push_interval', 5000)
            save_local = self.config.get('output', {}).get('save_local', True)
            push_to_hub = self.config.get('output', {}).get('push_to_hub', False)
            
            # 创建批次
            batch_size = self.config.get('concurrency', {}).get('batch_size', 100)
            batches = [
                remaining_dataset[i:i + batch_size]
                for i in range(0, len(remaining_dataset), batch_size)
            ]
            
            # 创建累积数据文件
            accumulated_file = self._get_output_file_path("_accumulated")
            # 加载已有的累积数据
            accumulated_items = self._load_from_json(accumulated_file)
            
            # 使用tqdm显示进度
            with tqdm(total=len(batches), desc="Transforming data") as pbar:
                for batch_idx, batch in enumerate(batches):
                    # 转换当前批次
                    batch_items = await self._transform_batch(batch)
                    transformed_items.extend(batch_items)
                    
                    # 更新进度条
                    pbar.update(1)
                    pbar.set_postfix({
                        'Items': len(transformed_items) + start_offset,
                        'Current Batch': f"{batch_idx + 1}/{len(batches)}"
                    })
                    
                    # 定期保存本地文件
                    if save_local and save_interval > 0 and len(transformed_items) % save_interval == 0:
                        # 保存分段数据
                        segment_file = self._get_output_file_path(f"_segment_{len(transformed_items)}")
                        segment_items = transformed_items[-save_interval:]
                        self._save_to_json(segment_items, segment_file)
                        logger.info(f"Saved segment: {len(segment_items)} items to {segment_file}")
                        
                        # 更新累积数据
                        accumulated_items = self._merge_items(accumulated_items, transformed_items)
                        self._save_to_json(accumulated_items, accumulated_file)
                        logger.info(f"Updated accumulated file: {len(accumulated_items)} total items")
                    
                    # 定期推送到hub
                    if push_to_hub and push_interval > 0 and len(transformed_items) % push_interval == 0:
                        # 推送所有累积的数据
                        await self._upload_to_hub(accumulated_items, is_checkpoint=True)
                        logger.info(f"Pushed checkpoint: {len(accumulated_items)} items to hub")
            
            # 最终保存
            if save_local:
                # 保存最后一个分段
                if len(transformed_items) % save_interval > 0:
                    segment_file = self._get_output_file_path(f"_segment_{len(transformed_items)}")
                    segment_items = transformed_items[-(len(transformed_items) % save_interval):]
                    self._save_to_json(segment_items, segment_file)
                    logger.info(f"Saved final segment: {len(segment_items)} items to {segment_file}")
                
                # 更新并保存最终的累积数据
                final_items = self._merge_items(accumulated_items, transformed_items)
                final_file = self._get_output_file_path("_final")
                self._save_to_json(final_items, final_file)
                logger.info(f"Saved final file: {len(final_items)} total items to {final_file}")
            
            # 最终推送到hub
            if push_to_hub:
                await self._upload_to_hub(final_items, is_checkpoint=False)
                logger.info(f"Pushed final dataset: {len(final_items)} items to hub")
            
            return transformed_items
        
        except Exception as e:
            logger.error(f"Error in transform_dataset: {e}")
            raise
    
    async def _upload_to_hub(self, items: List[RefinedAlpacaItem], is_checkpoint: bool = False) -> None:
        """上传数据到 HuggingFace Hub，支持增量更新
        
        Args:
            items: 要上传的数据项列表
            is_checkpoint: 是否是检查点数据
        """
        if not items:
            logger.warning("No items to upload")
            return

        try:
            # 获取数据集名称和分片信息
            dataset_name = items[0].metadata.get('dataset_name', 'unknown').split('/')[-1]
            split = items[0].metadata.get('split', 'train')
            
            # 构建repo_id
            hub_config = self.config['output']['hub_config']
            task_name = self.config['task_name']
            
            # 构建repo名称：owner/prefix-datasetname-taskname[-checkpoint]
            repo_name = f"{hub_config['repo_prefix']}-{dataset_name}-{task_name}"
            if is_checkpoint:
                repo_name = f"{repo_name}-{split}_checkpoint"
            
            repo_id = f"{hub_config['owner']}/{repo_name}".replace("--", "-")
            
            # 创建数据集字典列表
            data_list = []
            for item in items:
                data_dict = asdict(item)
                data_dict['task_name'] = task_name
                data_list.append(data_dict)

            # 创建 Dataset 对象
            new_dataset = Dataset.from_list(data_list)
            final_dataset = new_dataset

            try:
                # 尝试加载现有数据集
                existing_dataset = load_dataset(repo_id, split=split)
                if existing_dataset is not None:
                    # 获取现有数据集的id列表
                    existing_ids = set(existing_dataset['id'])
                    # 过滤掉已存在的数据
                    filtered_new_dataset = new_dataset.filter(lambda x: x['id'] not in existing_ids)
                    
                    if len(filtered_new_dataset) > 0:
                        # 合并新旧数据集
                        final_dataset = concatenate_datasets([existing_dataset, filtered_new_dataset])
                        logger.info(f"Successfully merged {len(filtered_new_dataset)} new items with existing {len(existing_dataset)} items")
                    else:
                        final_dataset = existing_dataset
                        logger.info(f"No new data to add, all items already exist in the dataset")
            except Exception as e:
                logger.info(f"Creating new dataset: {e}")
            
            # 上传到 HuggingFace Hub
            final_dataset.push_to_hub(
                repo_id=repo_id,
                split=split,
                token=hub_config['token'],
                private=True
            )

            logger.info(f"Successfully pushed {'checkpoint' if is_checkpoint else 'final'} dataset ({len(final_dataset)} items) to {repo_id}")
            
        except Exception as e:
            logger.error(f"Error uploading to hub: {str(e)}")
            raise

    async def push_to_hub(self, output_file: str) -> None:
        """
        将数据推送到 HuggingFace Hub。
        这是一个公共接口，内部调用 _upload_to_hub 方法。
        
        Args:
            output_file: 要上传的输出文件路径
        """
        await self._upload_to_hub(output_file)
