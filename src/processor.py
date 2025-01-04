import asyncio
import time
import json
import logging
import yaml
from typing import List, Dict, Any, Optional, Tuple
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from pathlib import Path

from .models import (
    AlpacaItem, ProcessedItem, BatchResult, 
    QualityMetrics, ProcessingMetadata,
    ImprovementSuggestion, DataCategory
)

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ProcessingError(Exception):
    """处理错误基类"""
    def __init__(self, message: str, error_type: str):
        super().__init__(message)
        self.error_type = error_type

class JSONParseError(ProcessingError):
    """JSON解析错误"""
    def __init__(self, message: str):
        super().__init__(message, "json_parse_error")

class APIError(ProcessingError):
    """API调用错误"""
    def __init__(self, message: str):
        super().__init__(message, "api_error")

class ValidationError(ProcessingError):
    """数据验证错误"""
    def __init__(self, message: str):
        super().__init__(message, "validation_error")

class DataProcessor:
    def __init__(self, config_path_or_dict):
        """初始化处理器
        
        Args:
            config_path_or_dict: 配置文件路径或配置字典
        """
        # 加载配置
        if isinstance(config_path_or_dict, (str, Path)):
            self.config = self._load_config(config_path_or_dict)
        else:
            self.config = config_path_or_dict
        
        # 加载分类配置
        self.category_config = self._load_category_config()
        
        # 初始化OpenAI客户端
        openai_config = self.config['openai']
        client_params = {
            "api_key": openai_config['api_key'],
        }
        
        # 添加可选参数
        if 'api_base' in openai_config and openai_config['api_base']:
            client_params['base_url'] = openai_config['api_base']
        
        if 'api_version' in openai_config and openai_config['api_version']:
            client_params['api_version'] = openai_config['api_version']
            
        if 'organization' in openai_config and openai_config['organization']:
            client_params['organization'] = openai_config['organization']
        
        self.client = AsyncOpenAI(**client_params)
        
        # 初始化并发控制
        self.semaphore = asyncio.Semaphore(
            self.config['concurrency']['max_concurrent_requests']
        )
        self.progress = Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn()
        )

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置"""
<<<<<<< HEAD
        try:
            logger.info(f"Loading config from: {config_path}")
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            raise

    def _load_category_config(self) -> Dict[str, Any]:
=======
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _load_category_config(self):
>>>>>>> 585f086 (Initial commit: Add data quality assessment functionality)
        """加载分类配置文件
        
        Returns:
            dict: 分类配置
        """
<<<<<<< HEAD
        try:
            category_config_path = self.config.get('paths', {}).get('category_config', 'config/category.yaml')
            with open(category_config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load category config: {str(e)}")
            return {}
=======
        category_path = Path(self.config.get('paths', {}).get(
            'category_config', 
            Path(__file__).parent.parent / 'config' / 'category.yaml'
        ))
        
        try:
            with open(category_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logging.warning(f"无法加载分类配置文件 {category_path}: {e}")
            return {
                "version": "1.0",
                "categories": {
                    "KNOWLEDGE": {"description": "知识性回答"},
                    "CONVERSATION": {"description": "对话性回答"},
                    "CREATIVE": {"description": "创造性回答"}
                }
            }
>>>>>>> 585f086 (Initial commit: Add data quality assessment functionality)

    def _create_default_result(self, item: AlpacaItem, error_msg: str, error_type: str) -> ProcessedItem:
        """创建默认的错误结果"""
        return ProcessedItem(
<<<<<<< HEAD
            id=str(time.time()),
=======
            id="error",
>>>>>>> 585f086 (Initial commit: Add data quality assessment functionality)
            source="error",
            instruction=item.instruction,
            input=item.input,
            output=item.output,
            quality_metrics={
                "reasoning_depth": 0,
                "correctness": 0,
                "clarity": 0,
                "context_awareness": 0,
                "engagement": 0
            },
            score=0,
            category="ERROR",
            processed_output=item.output,
            metadata={
                "error": error_msg,
                "error_type": error_type,
                "processing_time": 0,
                "token_count": 0
            }
        )

    def _build_evaluation_prompt(self, item: AlpacaItem) -> str:
        """构建评估提示
        
        Args:
            item: 待评估的数据项
            
        Returns:
            str: 评估提示
        """
        prompt = f"""你是一个专业的数据质量评估专家。请根据以下维度评估这条数据的质量，并以JSON格式返回评估结果。

评估维度：
{self.config.get('quality_metrics_description', '')}

可选的回答类别：
{self._format_categories()}

数据内容：
指令: {item.instruction}
输入: {item.input or '无'}
输出: {item.output}

请以下面的JSON格式返回评估结果：
{{
    "quality_metrics": {{
        "reasoning_depth": 分数,
        "correctness": 分数,
        "clarity": 分数,
        "context_awareness": 分数,
        "engagement": 分数
    }},
    "category": "选择最合适的类别",
    "processed_output": "处理后的输出",
    "metadata": {{
        "improvement_suggestions": ["改进建议1", "改进建议2"],
        "validation_notes": ["评估说明1", "评估说明2"]
    }}
}}"""
        return prompt

    def _format_categories(self) -> str:
        """格式化类别描述"""
        if not self.category_config or 'categories' not in self.category_config:
            return ""
            
        categories = []
        for category, info in self.category_config['categories'].items():
            detail = info.get('detail', '')
            categories.append(f"- {category}: {detail}")
            
        return "\n".join(categories)

    def _calculate_weighted_score(self, quality_metrics: Dict[str, float]) -> float:
        """计算加权总分
        
        根据quality_metrics.yaml中的权重计算总分
        """
        metrics_config = self.config.get('quality_metrics', {})
        if not metrics_config:
            return sum(quality_metrics.values()) / len(quality_metrics)
            
        total_score = 0.0
        total_weight = 0.0
        
        for metric, value in quality_metrics.items():
            if metric in metrics_config:
                weight = metrics_config[metric].get('weight', 0)
                total_score += value * weight
                total_weight += weight
                
        if total_weight == 0:
            return 0.0
            
        return round(total_score / total_weight, 2)

    def _generate_unique_id(self, dataset_name: str, index: int) -> str:
        """生成唯一的数字ID"""
        # 使用时间戳和索引生成唯一ID
        timestamp = int(time.time())
        return f"{timestamp}{index:06d}"

    async def _call_api_with_retry(self, prompt: str, timeout: Optional[float] = None) -> Tuple[str, Dict[str, int]]:
        """调用API并返回响应"""
        try:
            # 记录请求内容
            logger.debug(f"API Request Prompt: {prompt}")
            
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.config['openai']['model_name'],
                    messages=[
                        {"role": "system", "content": "你是一个专业的数据质量评估专家，擅长分析和评估数据质量。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config['openai']['temperature'],
                    max_tokens=self.config['openai']['max_tokens']
                ),
                timeout or self.config['concurrency']['request_timeout']
            )
            
            # 记录完整的API响应
            logger.debug(f"Raw API Response: {response}")
            response_content = response.choices[0].message.content
            logger.info(f"Response Content: {response_content}")
            
            # 检查响应内容是否为空
            if not response_content or not response_content.strip():
                raise APIError("Empty response from API")
            
            return response_content, {
                "total_tokens": response.usage.total_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
            
        except asyncio.TimeoutError:
            logger.error("API request timed out")
            raise APIError("API request timed out")
        except Exception as e:
            logger.error(f"API Error: {str(e)}")
            logger.error(f"API Error Type: {type(e)}")
            raise APIError(f"API call failed: {str(e)}")

    def _parse_api_response(self, response_text: str) -> Dict[str, Any]:
        """解析API响应"""
        try:
            # 记录原始响应
            logger.debug(f"Parsing response text: {response_text}")
            
            # 移除可能的Markdown代码块标记
            response_text = response_text.strip()
            if response_text.startswith('```'):
                response_text = '\n'.join(response_text.split('\n')[1:-1])
            response_text = response_text.strip('`')
            
            # 尝试直接解析JSON
            try:
                data = json.loads(response_text)
                logger.debug(f"Successfully parsed JSON: {data}")
                return data
            except json.JSONDecodeError as e:
                logger.warning(f"Initial JSON parse failed: {str(e)}")
                
                # 尝试清理文本后再解析
                cleaned_text = response_text.replace('\n', '').replace(' ', '')
                data = json.loads(cleaned_text)
                logger.debug(f"Successfully parsed JSON after cleaning: {data}")
                return data
            
        except Exception as e:
            logger.error(f"解析响应时出错: {str(e)}")
            logger.error(f"原始响应: {response_text}")
            raise JSONParseError(f"无法解析响应: {str(e)}")

    def _normalize_scores(self, metrics: dict) -> dict:
        """Normalize scores to be between 0 and 10."""
        normalized_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                # 如果分数大于10，将其归一化到0-10的范围
                normalized_metrics[key] = min(float(value), 10.0) if value > 0 else value
            else:
                normalized_metrics[key] = value
        return normalized_metrics

    async def process_single_item(self, item: AlpacaItem, dataset_name: str = None, item_index: int = 0) -> ProcessedItem:
        """处理单个数据项"""
        try:
            start_time = time.time()
            
            # 构建评估提示
            prompt = self._build_evaluation_prompt(item)
            
            # 调用API进行评估
            response = await self._call_api_with_retry(prompt)
            
            # 解析API响应
            result = self._parse_api_response(response[0])
            
            # 归一化分数
            result["quality_metrics"] = self._normalize_scores(result["quality_metrics"])
            
            # 计算加权总分
            weighted_score = self._calculate_weighted_score(result['quality_metrics'])
            
            # 生成唯一ID
            unique_id = self._generate_unique_id(dataset_name or "unknown", item_index)
            
            # 构建处理后的数据项
            processed_item = ProcessedItem(
                id=unique_id,
                source=dataset_name or "unknown",
                instruction=item.instruction,
                input=item.input,
                output=item.output,
                quality_metrics=result['quality_metrics'],
                score=weighted_score,
                category=result['category'],
                processed_output=result.get('processed_output', item.output),
                metadata={
                    **result.get('metadata', {}),
                    'processing_time': time.time() - start_time,
                    'token_count': response[1]["total_tokens"],
                    'id': unique_id,
                    'source': dataset_name or "unknown"
                }
            )
            
            return processed_item
            
        except Exception as e:
            logger.error(f"处理数据项时出错: {str(e)}")
            raise

    async def process_batch(self, items: List[AlpacaItem], progress_callback=None) -> BatchResult:
        """批量处理数据"""
        tasks = [self.process_single_item(item) for item in items]
        
        successful = []
        failed = []
        
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                successful.append(result)
            except Exception as e:
                failed.append(items[len(successful) + len(failed)])
                logger.error(f"Error processing item: {e}")
            finally:
                if progress_callback:
                    progress_callback()
        
        # 计算批处理指标
        metrics = self.calculate_batch_metrics(successful)
        
        # 创建处理元数据
        metadata = ProcessingMetadata(
            processing_time=sum(item.metadata["processing_time"] for item in successful) if successful else 0,
            token_count=sum(item.metadata["token_count"] for item in successful) if successful else 0,
            improvement_suggestions=[],
            validation_notes=[]
        )
        
        return BatchResult(
            successful=successful,
            failed=failed,
            metrics=metrics,
            metadata=metadata
        )

    def calculate_batch_metrics(self, items: List[ProcessedItem]) -> Dict[str, float]:
        """计算批处理指标"""
        if not items:
            return {
                "average_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0,
                "success_rate": 0.0
            }
<<<<<<< HEAD
        
        scores = [item.score for item in items]
        return {
            "average_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "success_rate": 1.0  # 所有成功处理的项都被视为成功
=======
            
        quality_scores = [item.score for item in items]
        
        return {
            "average_score": sum(quality_scores) / len(quality_scores),
            "min_score": min(quality_scores),
            "max_score": max(quality_scores),
            "success_rate": float(len(items)) / (len(items) + len([i for i in items if not i.score]))
>>>>>>> 585f086 (Initial commit: Add data quality assessment functionality)
        }

    def filter_results(self, results: List[ProcessedItem]) -> List[ProcessedItem]:
        """根据配置过滤结果"""
        filtered = []
        filter_config = self.config.get('filter', {})
        
        min_score = filter_config.get('min_score', 0)
        categories = filter_config.get('categories', [])
        
        for item in results:
            if item.score >= min_score and (not categories or item.category in categories):
                filtered.append(item)
                
        return filtered
