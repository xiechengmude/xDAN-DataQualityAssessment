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
import asyncio

logger = logging.getLogger(__name__)

class DataTransformer:
    """Transform Alpaca format data to include structured output."""
    
    def __init__(self, config):
        """Initialize the transformer."""
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
    
    def _load_template(self) -> str:
        """Load the prompt template."""
        template_path = Path('prompt/convert') / f"{self.config['output']['template']}.md"
        with open(template_path, 'r') as f:
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

        # 设置更合理的超时时间
        response = await self.client.chat.completions.create(
            model=self.config['openai']['model_name'],
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config['openai']['temperature'],
            max_tokens=self.config['openai']['max_tokens'],
            timeout=30  # 设置30秒超时
        )
        
        return response.choices[0].message.content
    
    async def transform_item(self, alpaca_item: AlpacaItem) -> RefinedAlpacaItem:
        """Transform a single data item to include structured output."""
        refined_output = await self.refine_output(alpaca_item)
        return RefinedAlpacaItem.from_alpaca_item(alpaca_item, refined_output)
    
    async def transform_dataset(self) -> List[RefinedAlpacaItem]:
        """Transform the entire dataset."""
        alpaca_items = self.data_loader.load_and_convert()
        
        # 使用asyncio.gather并发处理多个项目
        async def process_batch(items: List[AlpacaItem]) -> List[RefinedAlpacaItem]:
            tasks = [self.transform_item(item) for item in items]
            return await asyncio.gather(*tasks)
        
        # 将数据分成10个一批
        batch_size = 10
        transformed_items = []
        
        for i in tqdm(range(0, len(alpaca_items), batch_size), desc="Transforming data"):
            batch = alpaca_items[i:i + batch_size]
            batch_results = await process_batch(batch)
            transformed_items.extend(batch_results)
            
        logger.info(f"Transformed {len(transformed_items)} items")
        return transformed_items
    
    def save_transformed_data(self, items: List[RefinedAlpacaItem], output_path: str):
        """Save transformed data to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 将数据转换为字典列表
        data = [item.model_dump() for item in items]
        
        # 保存为JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(items)} transformed items to {output_path}")

    def save_results(self, items: List[RefinedAlpacaItem], output_path: Path) -> None:
        """Save transformed items to a JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([item.dict() for item in items], f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(items)} transformed items to {output_path}")

    def push_to_hub(self, file_path: Path) -> None:
        """Push results to HuggingFace Hub using datasets."""
        try:
            from datasets import Dataset
            import json
            
            # 获取hub配置
            hub_config = self.config.get('output', {}).get('hub_config', {})
            if not hub_config:
                raise ValueError("Hub configuration not found in config")
            
            # 检查是否需要推送到hub
            if not self.config.get('output', {}).get('push_to_hub', False):
                logger.info("Push to hub is disabled in config")
                return
            
            # 读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 将数据转换为Dataset格式
            dataset = Dataset.from_list(data)
            
            # 推送到hub
            dataset.push_to_hub(
                hub_config['repository_id'],
                private=hub_config.get('private', True),
                token=hub_config.get('token')
            )
            logger.info(f"Successfully pushed dataset to {hub_config['repository_id']}")
            
        except Exception as e:
            logger.error(f"Error pushing to hub: {str(e)}")
            raise
