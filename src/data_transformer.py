"""数据转换模块"""
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from tqdm import tqdm
from datetime import datetime
import yaml
import asyncio
from openai import AsyncOpenAI
from dataclasses import asdict

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
        
        # 初始化OpenAI客户端
        openai_config = config.get('openai', {})
        self.client = AsyncOpenAI(
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
        await self.client.close()

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
            openai_config = self.config.get('openai', {})
            response = await self.client.chat.completions.create(
                model=openai_config.get('model_name', 'gpt-3.5-turbo'),
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=openai_config.get('temperature', 0.7),
                max_tokens=openai_config.get('max_tokens', 2000)
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
            results = await asyncio.gather(*tasks)
            return results
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
            
            refined_output, token_info = await self.refine_output(alpaca_item)
            return RefinedAlpacaItem.from_alpaca_item(
                item=alpaca_item,
                refined_output=refined_output,
                token_info=token_info,
                source=self.config.get('task_name', 'data_transform'),
                model_name=self.config.get('openai', {}).get('model_name', 'gpt-3.5-turbo'),
                sources=dataset_name if dataset_name else "unknown"
            )
        except Exception as e:
            logger.error(f"Error transforming item: {str(e)}")
            raise
    
    def _create_batches(self) -> List[List[AlpacaItem]]:
        """创建数据批次"""
        items = self.dataset
        if not items:
            raise ValueError("No items to transform")
        
        logger.info(f"Starting transformation of {len(items)} items")
        batch_size = self.config.get('batch_size', 10)
        
        # 分批处理数据
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            yield batch
    
    async def _transform_batch(self, batch: List[AlpacaItem]) -> List[RefinedAlpacaItem]:
        """转换一批数据"""
        transformed_items = []
        for item in batch:
            try:
                # 获取数据集名称
                dataset_name = item.metadata.get('dataset_name', 'unknown')
                
                # 转换单个数据项
                transformed_item = await self._transform_item(item, dataset_name)
                transformed_items.append(transformed_item)
            except Exception as e:
                logger.error(f"Error transforming item in batch: {str(e)}")
                continue
        return transformed_items
    
    async def transform_dataset(self) -> List[RefinedAlpacaItem]:
        """转换数据集"""
        try:
            # 获取保存间隔
            save_interval = self.config.get('output', {}).get('save_interval', 500)
            transformed_items = []
            
            # 分批处理数据
            batches = list(self._create_batches())
            with tqdm(total=len(batches), desc="Transforming data") as pbar:
                for batch in batches:
                    # 转换当前批次
                    batch_items = await self._transform_batch(batch)
                    transformed_items.extend(batch_items)
                    
                    # 如果达到保存间隔且需要上传到 HF Hub，则进行保存和上传
                    if len(transformed_items) >= save_interval and self.config.get('output', {}).get('push_to_hub', True):
                        await self._upload_to_hub(transformed_items)
                        logger.info(f"Uploaded {len(transformed_items)} items to HuggingFace Hub")
                        # 清空已保存的数据
                        transformed_items = []
                    
                    pbar.update(1)
            
            # 处理剩余的数据
            if transformed_items and self.config.get('output', {}).get('push_to_hub', True):
                await self._upload_to_hub(transformed_items)
                logger.info(f"Uploaded {len(transformed_items)} items to HuggingFace Hub")
            
            return transformed_items
            
        except Exception as e:
            logger.error(f"Error in transform_dataset: {str(e)}")
            raise
    
    def _get_output_file_path(self) -> str:
        """获取输出文件路径。"""
        output_config = self.config.get('output', {})
        base_dir = output_config.get('base_dir', 'output')
        task_name = self.config.get('task_name', 'unknown_task')
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
    
    async def _upload_to_hub(self, items: List[RefinedAlpacaItem]) -> None:
        """上传数据到 HuggingFace Hub"""
        try:
            # 获取配置
            hub_config = self.config.get('output', {}).get('hub_config', {})
            repo_id = hub_config.get('repo_id')
            split = hub_config.get('split', 'train')
            token = hub_config.get('token')

            if not repo_id or not token:
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

            # 获取当前时间戳作为数据集名称
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = f"task_data_transform_{timestamp}"

            # 上传到 HuggingFace Hub
            dataset.push_to_hub(
                repo_id=repo_id,
                split=split,
                token=token,
                private=True
            )

        except Exception as e:
            logger.error(f"Error uploading to HuggingFace Hub: {str(e)}")
            raise

    async def push_to_hub(self, output_file: str) -> None:
        """
        将数据推送到 HuggingFace Hub。
        这是一个公共接口，内部调用 _upload_to_hub 方法。
        
        Args:
            output_file: 要上传的输出文件路径
        """
        await self._upload_to_hub(output_file)
