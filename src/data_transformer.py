from typing import Dict, List, Optional
from pathlib import Path
import datasets
from tqdm import tqdm
import yaml
import logging
import json
from openai import AsyncOpenAI
from datetime import datetime
from .models import AlpacaItem, RefinedAlpacaItem
from .data_loader import DataLoader, DatasetConfig
from .data_types import AlpacaItem, RefinedAlpacaItem
import asyncio

logger = logging.getLogger(__name__)

class DataTransformer:
    """数据转换器，用于将数据转换为结构化格式。"""
    
    def __init__(self, config):
        """初始化转换器。"""
        self.config = config if isinstance(config, dict) else self._load_config(config)
        self.template = self._load_template()
        self.data_loader = DataLoader(self.config)
        self.client = AsyncOpenAI(
            api_key=self.config['openai']['api_key'],
            base_url=self.config['openai']['api_base']
        )
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from yaml file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_template(self):
        """加载提示模板。"""
        template_path = Path('prompt/convert') / f"{self.config['prompt']['template']}.md"
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    async def refine_output(self, item: AlpacaItem) -> str:
        """将原始输出转换为结构化格式"""
        prompt = f"""请将以下问答对转换为结构化的分析格式，要求简洁明确。

问题：{item.instruction}
{f'上下文：{item.input}' if item.input else ''}
原始答案：{item.output}

请按照以下结构重新组织答案：

<Analyze>
- 完整保留原始问题
- 提取关键信息和概念
- 明确问题类型和目标
- 列出已知条件和要求
</Analyze>

<Solve>
- 说明解决思路和理由
- 列出详细的解决步骤
- 展示完整的推理过程
- 记录关键的中间结果
</Solve>

<Verify>
- 检查步骤的正确性
- 验证是否满足条件
- 评估结果的合理性
- 考虑优化的空间
</Verify>

<Solution>
[这里请直接以对话的方式回答用户，语气友好自然，不需要列举要求。把前面分析的结果转化为一个完整、清晰、易懂的回答。就像您在跟用户面对面交谈一样。]
</Solution>

请确保前三个部分保持分析的严谨性，最后的Solution部分采用对话的方式直接回答用户的问题。
"""

        try:
            # 从配置中获取超时设置
            timeout = self.config.get('concurrency', {}).get('request_timeout', 30)
            
            # 添加重试逻辑
            max_retries = self.config.get('concurrency', {}).get('max_retries', 3)
            retry_delay = self.config.get('concurrency', {}).get('retry_delay', 1)
            
            for attempt in range(max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.config['openai']['model_name'],
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.config['openai']['temperature'],
                        max_tokens=self.config['openai']['max_tokens'],
                        timeout=timeout
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"All retry attempts failed: {str(e)}")
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    await asyncio.sleep(retry_delay)
        except Exception as e:
            logger.error(f"Error in refine_output: {str(e)}")
            raise
    
    async def transform_item(self, alpaca_item: AlpacaItem) -> RefinedAlpacaItem:
        """Transform a single data item to include structured output."""
        try:
            # 验证输入
            if self.config['transform']['validate_input']:
                required_fields = self.config['transform']['required_fields']
                for field in required_fields:
                    if not getattr(alpaca_item, field, None):
                        raise ValueError(f"Required field '{field}' is missing or empty")
            
            refined_output = await self.refine_output(alpaca_item)
            return RefinedAlpacaItem.from_alpaca_item(
                item=alpaca_item,
                refined_output=refined_output,
                source=self.config.get('task_name', 'data_transform'),
                model_name=self.config.get('openai', {}).get('model_name', 'unknown')
            )
        except Exception as e:
            logger.error(f"Error transforming item: {str(e)}")
            raise
    
    async def transform_dataset(self, items: List[AlpacaItem] = None) -> List[RefinedAlpacaItem]:
        """Transform the entire dataset."""
        try:
            # 如果没有提供items，则从数据加载器获取
            if items is None:
                items = self.data_loader.load_and_convert()
            
            logger.info(f"Starting transformation of {len(items)} items")
            
            # 从配置中获取batch_size
            batch_size = self.config.get('concurrency', {}).get('batch_size', 10)
            transformed_items = []
            
            # 使用tqdm显示进度
            for i in tqdm(range(0, len(items), batch_size), desc="Transforming data"):
                batch = items[i:i + batch_size]
                batch_tasks = [self.transform_item(item) for item in batch]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # 处理结果
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Batch processing error: {str(result)}")
                    else:
                        transformed_items.append(result)
                
                logger.info(f"Processed {len(transformed_items)}/{len(items)} items")
            
            return transformed_items
        except Exception as e:
            logger.error(f"Error in transform_dataset: {str(e)}")
            raise
    
    def save_results(self, items: List[RefinedAlpacaItem], output_path: Path) -> None:
        """保存转换结果到文件。"""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存到本地
            if self.config['output']['save_local']:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump([item.dict() for item in items], f, ensure_ascii=False, indent=2)
                logger.info(f"Saved {len(items)} transformed items to {output_path}")
            
            # 如果配置了push_to_hub，则上传结果
            if self.config['output']['push_to_hub']:
                self.push_to_hub(output_path)
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

    def push_to_hub(self, file_path: Path) -> None:
        """推送结果到HuggingFace Hub。"""
        try:
            from datasets import Dataset
            import json
            from datetime import datetime
            
            # 获取hub配置
            hub_config = self.config.get('output', {}).get('hub_config', {})
            if not hub_config:
                raise ValueError("Hub configuration not found in config")
            
            # 读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 将数据转换为Dataset格式
            dataset = Dataset.from_list(data)
            
            # 构建仓库名称
            file_naming = self.config.get('output', {}).get('file_naming', {})
            task_name = file_naming.get('task_name', 'data_transform')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            repository_name = f"{task_name}_{timestamp}"
            
            # 推送到hub
            dataset.push_to_hub(
                f"{hub_config['repository_id']}/{repository_name}",
                private=hub_config.get('private', True),
                token=hub_config.get('token')
            )
            logger.info(f"Successfully pushed dataset to {hub_config['repository_id']}/{repository_name}")
            
        except Exception as e:
            logger.error(f"Error pushing to hub: {str(e)}")
            raise
