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
            base_url=openai_config.get('base_url', 'http://35.240.173.116:7220/v1'),
            api_key=openai_config.get('api_key', 'dummy-key'),  # API密钥
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

    async def transform_dataset(self) -> List[RefinedAlpacaItem]:
        """转换数据集"""
        try:
            transformed_items = []
            save_interval = self.config.get('output', {}).get('save_interval', 100)
            save_local = self.config.get('output', {}).get('save_local', True)
            push_to_hub = self.config.get('output', {}).get('push_to_hub', False)
            
            # 创建批次
            batches = self._create_batches()
            total_batches = len(batches)
            
            # 使用tqdm创建进度条
            with tqdm(total=total_batches, desc="Transforming data") as pbar:
                for i, batch in enumerate(batches, 1):
                    # 转换当前批次
                    batch_items = await self._transform_batch(batch)
                    transformed_items.extend(batch_items)
                    
                    # 更新进度条
                    pbar.update(1)
                    pbar.set_postfix({
                        'Items': len(transformed_items),
                        'Current Batch': f"{i}/{total_batches}"
                    })
                    
                    # 检查是否需要间隔保存
                    if len(transformed_items) % save_interval == 0:
                        if save_local:
                            output_file = self._get_output_file_path()
                            self._save_to_json(transformed_items, output_file)
                            logger.info(f"Saved checkpoint: {len(transformed_items)} items to {output_file}")
                        
                        if push_to_hub:
                            await self._upload_to_hub(transformed_items, is_checkpoint=True)
            
            # 最终保存和上传
            if save_local:
                output_file = self._get_output_file_path()
                self._save_to_json(transformed_items, output_file)
                logger.info(f"Final save: {len(transformed_items)} items to {output_file}")
            
            if push_to_hub:
                await self._upload_to_hub(transformed_items, is_checkpoint=False)
            
            return transformed_items
            
        except Exception as e:
            logger.error(f"Error in transform_dataset: {e}")
            raise
    
    def _get_output_file_path(self) -> str:
        """获取输出文件路径。"""
        output_config = self.config.get('output', {})
        base_dir = output_config.get('base_dir', 'output')
        task_name = self.config.get('task_name', 'unknown')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建输出目录
        os.makedirs(base_dir, exist_ok=True)
        
        # 构建文件名：task_taskName_时间戳.json
        filename = f"task_{task_name}_{timestamp}.json"
        return os.path.join(base_dir, filename)
    
    def _save_to_json(self, items: List[RefinedAlpacaItem], output_file: str) -> None:
        """保存数据到JSON文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([item.dict() for item in items], f, ensure_ascii=False, indent=2)
    
    async def _upload_to_hub(self, items: List[RefinedAlpacaItem], is_checkpoint: bool = False) -> None:
        """上传数据到 HuggingFace Hub，支持增量更新
        
        Args:
            items: 要上传的数据项列表
            is_checkpoint: 是否是检查点数据
        """
        try:
            # 获取配置
            hub_config = self.config.get('output', {}).get('hub_config', {})
            owner = hub_config.get('owner')
            repo_prefix = hub_config.get('repo_prefix')
            split = hub_config.get('split', 'train')
            token = hub_config.get('token')

            if not all([owner, repo_prefix, token]):
                raise ValueError("Missing required HuggingFace Hub configuration")

            # 创建数据集字典列表
            data_list = []
            for item in items:
                data_dict = asdict(item)
                # 将 source 字段改为 task_name
                data_dict['task_name'] = data_dict.pop('source', None)
                data_list.append(data_dict)

            # 创建 Dataset 对象
            dataset = Dataset.from_list(data_list)

            # 使用固定的仓库名
            task_name = self.config.get('task_name', 'unknown')
            dataset_name = f"{task_name}"
            if is_checkpoint:
                dataset_name += "_checkpoint"
            repo_id = f"{owner}/{repo_prefix}-{dataset_name}".replace("--", "-")

            try:
                # 尝试加载并合并现有数据集
                existing_dataset = load_dataset(repo_id, split=split)
                
                # 获取现有数据集的id列表
                existing_ids = set(existing_dataset['id'])
                
                # 过滤掉已存在的数据
                new_dataset = dataset.filter(lambda x: x['id'] not in existing_ids)
                
                # 合并新旧数据集
                if len(new_dataset) > 0:
                    dataset = concatenate_datasets([existing_dataset, new_dataset])
                else:
                    dataset = existing_dataset
                    logger.info(f"No new data to add, all {len(existing_ids)} items already exist")
            except Exception as e:
                logger.info(f"Creating new dataset: {e}")
            
            # 上传到 HuggingFace Hub
            dataset.push_to_hub(
                repo_id=repo_id,
                split=split,
                token=token,
                private=True
            )

            logger.info(f"Successfully pushed {'checkpoint' if is_checkpoint else 'final'} dataset ({len(items)} items) to {repo_id}")
            
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
