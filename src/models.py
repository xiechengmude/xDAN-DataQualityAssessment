from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime

class DataCategory(str, Enum):
    """数据场景分类"""
    KNOWLEDGE_QA = "KNOWLEDGE_QA"
    REASONING = "REASONING"
    CODE_GENERATION = "CODE_GENERATION"
    CREATIVE_WRITING = "CREATIVE_WRITING"
    CONVERSATION = "CONVERSATION"
    INSTRUCTION_FOLLOWING = "INSTRUCTION_FOLLOWING"
    ANALYSIS_SUMMARY = "ANALYSIS_SUMMARY"
    MULTIMODAL_UNDERSTANDING = "MULTIMODAL_UNDERSTANDING"

class QualityMetrics(BaseModel):
    """质量评估指标"""
    # 核心评估维度
    reasoning_depth: float = Field(ge=1, le=10)  # 推理深度
    correctness: float = Field(ge=1, le=10)      # 准确性
    clarity: float = Field(ge=1, le=10)          # 清晰度
    context_awareness: float = Field(ge=1, le=10) # 上下文理解
    engagement: float = Field(ge=1, le=10)        # 互动质量

    # 维度权重，从配置文件加载
    _weights: Dict[str, float] = {
        'reasoning_depth': 0.25,
        'correctness': 0.25,
        'clarity': 0.2,
        'context_awareness': 0.15,
        'engagement': 0.15
    }

    @property
    def overall_score(self) -> float:
        """计算总体得分"""
        return sum(getattr(self, k) * v for k, v in self._weights.items())

    def to_dict(self) -> Dict[str, float]:
        """转换为字典格式"""
        return {
            **self.model_dump(),
            'overall_score': self.overall_score
        }

    @classmethod
    def load_weights(cls, config_path: str):
        """从配置文件加载权重"""
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        weights = {}
        # 加载质量评估指标权重
        for metric, data in config['quality_metrics'].items():
            weights[metric] = data['weight']
            
        cls._weights = weights

class EnhancedAlpacaItem(BaseModel):
    """增强的Alpaca数据项，包含评分和元数据"""
    # 原始Alpaca字段
    instruction: str
    input: str = ""
    output: str
    
    # 新增字段
    scores: float = Field(0.0, ge=0, le=10)  # 总体评分
    detailed_scores: Dict[str, float] = Field(default_factory=dict)  # 详细评分
    model_name: str  # 使用的模型名称
    task_name: str  # 任务名称
    timestamp: datetime  # 处理时间戳
    category: str  # 类别标签
    metadata: Dict[str, Any] = Field(default_factory=dict)  # 处理元数据
    sources: str = Field(default="")  # 数据来源

    @validator('instruction')
    def validate_instruction(cls, v):
        if not v.strip():
            raise ValueError("Instruction cannot be empty")
        return v.strip()

    @validator('output')
    def validate_output(cls, v):
        if not v.strip():
            raise ValueError("Output cannot be empty")
        return v.strip()
    
    def model_dump(self, **kwargs) -> dict:
        """自定义序列化方法"""
        data = super().model_dump(**kwargs)
        # 确保时间戳使用ISO格式
        if isinstance(data.get('timestamp'), datetime):
            data['timestamp'] = data['timestamp'].isoformat()
        return data

    @classmethod
    def from_processed_item(cls, 
                          item: 'ProcessedItem', 
                          model_name: str,
                          task_name: str,
                          timestamp: datetime = None) -> 'EnhancedAlpacaItem':
        """从ProcessedItem创建EnhancedAlpacaItem"""
        return cls(
            instruction=item.instruction,
            input=item.input,
            output=item.output,
            scores=item.score,
            detailed_scores=item.quality_metrics,
            model_name=model_name,
            task_name=task_name,
            timestamp=timestamp or datetime.now(),
            category=str(item.category),
            metadata={
                "processing_time": item.metadata.get("processing_time", 0),
                "token_count": item.metadata.get("token_count", 0),
                "processed_output": item.processed_output,
                "improvement_suggestions": item.metadata.get("improvement_suggestions", []),
                "validation_notes": item.metadata.get("validation_notes", [])
            }
        )

class AlpacaItem(BaseModel):
    """原始数据项"""
    instruction: str
    input: str = ""
    output: str

    @validator('instruction')
    def validate_instruction(cls, v):
        if not v.strip():
            raise ValueError("Instruction cannot be empty")
        return v.strip()

    @validator('output')
    def validate_output(cls, v):
        if not v.strip():
            raise ValueError("Output cannot be empty")
        return v.strip()

class ProcessedItem(BaseModel):
    """处理后的数据项"""
    id: str
    source: str
    instruction: str
    input: str
    output: str
    quality_metrics: Dict[str, float]
    score: float
    category: DataCategory
    processed_output: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def quality_score(self) -> float:
        return self.score

class ImprovementSuggestion(BaseModel):
    """改进建议"""
    aspect: str
    suggestion: str
    priority: int = Field(ge=1, le=3)

class ProcessingMetadata(BaseModel):
    """处理元数据"""
    processing_time: float
    token_count: int
    improvement_suggestions: List[ImprovementSuggestion] = []
    validation_notes: List[str] = []

class BatchResult(BaseModel):
    """批处理结果"""
    successful: List[ProcessedItem]
    failed: List[AlpacaItem]
    metrics: Dict[str, float]
    metadata: ProcessingMetadata
    
    def to_enhanced_alpaca_format(self, 
                                model_name: str,
                                task_name: str,
                                timestamp: datetime = None) -> List[EnhancedAlpacaItem]:
        """转换为增强的Alpaca格式"""
        timestamp = timestamp or datetime.now()
        enhanced_items = []
        
        # 处理成功的项目
        for item in self.successful:
            enhanced_items.append(
                EnhancedAlpacaItem.from_processed_item(
                    item=item,
                    model_name=model_name,
                    task_name=task_name,
                    timestamp=timestamp
                )
            )
        
        # 处理失败的项目
        for item in self.failed:
            enhanced_items.append(
                EnhancedAlpacaItem(
                    instruction=item.instruction,
                    input=item.input,
                    output=item.output,
                    scores=0.0,
                    detailed_scores={},
                    model_name=model_name,
                    task_name=task_name,
                    timestamp=timestamp,
                    metadata={"status": "failed"}
                )
            )
        
        return enhanced_items
