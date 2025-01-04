from pydantic import BaseModel, Field, field_validator
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

    @field_validator('instruction')
    @classmethod
    def validate_instruction(cls, v):
        if not v.strip():
            raise ValueError("Instruction cannot be empty")
        return v.strip()

    @field_validator('output')
    @classmethod
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
    sources: str = ""  # 数据来源

    @field_validator('instruction')
    @classmethod
    def validate_instruction(cls, v):
        if not v.strip():
            raise ValueError("Instruction cannot be empty")
        return v.strip()

    @field_validator('output')
    @classmethod
    def validate_output(cls, v):
        if not v.strip():
            raise ValueError("Output cannot be empty")
        return v.strip()

class ProcessedItem(BaseModel):
    """处理后的数据项"""
    id: str
    sources: str  # 改为sources以保持一致性
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
        """计算质量得分"""
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

class StructuredPromptItem(BaseModel):
    """结构化prompt格式的数据项"""
    question: str
    components: Dict[str, str] = Field(
        description="包含Analyze、Solve、Verify、Solution等组件的内容"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="原始数据和转换信息的元数据"
    )

    @classmethod
    def from_alpaca_item(cls, item: AlpacaItem, template: str) -> 'StructuredPromptItem':
        """从AlpacaItem创建StructuredPromptItem"""
        question = item.instruction
        if item.input:
            question += f"\nContext: {item.input}"
            
        return cls(
            question=question,
            components={
                'Analyze': "- Key aspects identified\n- Scope defined\n- Evaluation criteria established",
                'Solve': "- Systematic examination\n- Evidence provided\n- Multiple perspectives considered",
                'Verify': "- Comprehensiveness checked\n- Arguments validated\n- Objectivity assessed",
                'Solution': item.output
            },
            metadata={
                'original_format': 'alpaca',
                'transformation_template': template,
                'original_instruction': item.instruction,
                'original_input': item.input,
                'original_output': item.output,
                'sources': item.sources
            }
        )

class QuestionAnalysis(BaseModel):
    """问题分析结果"""
    original_question: str
    question_quality: Dict[str, float] = Field(
        description="问题质量评分，包括清晰度、具体性、合理性等"
    )
    improvement_suggestions: List[str] = Field(
        description="问题改进建议"
    )
    improved_question: str = Field(
        description="改进后的问题"
    )

class AnswerAnalysis(BaseModel):
    """答案分析结果"""
    original_answer: str
    answer_quality: Dict[str, float] = Field(
        description="答案质量评分，包括完整性、准确性、逻辑性等"
    )
    improvement_suggestions: List[str] = Field(
        description="答案改进建议"
    )
    improved_answer: str = Field(
        description="改进后的答案"
    )

class EnhancedStructuredPromptItem(BaseModel):
    """增强的结构化prompt数据项，包含问题和答案的分析"""
    question: str
    components: Dict[str, str] = Field(
        description="包含Analyze、Solve、Verify、Solution等组件的内容"
    )
    question_analysis: QuestionAnalysis = Field(
        description="问题分析结果"
    )
    answer_analysis: AnswerAnalysis = Field(
        description="答案分析结果"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="原始数据和转换信息的元数据"
    )

    @classmethod
    def from_alpaca_item(cls, 
                        item: AlpacaItem, 
                        template: str,
                        question_analysis: QuestionAnalysis,
                        answer_analysis: AnswerAnalysis) -> 'EnhancedStructuredPromptItem':
        """从AlpacaItem和分析结果创建EnhancedStructuredPromptItem"""
        question = question_analysis.improved_question
            
        return cls(
            question=question,
            components={
                'Analyze': f"""问题分析：
- 原始问题：{question_analysis.original_question}
- 问题质量评估：{', '.join(f'{k}: {v}' for k, v in question_analysis.question_quality.items())}
- 改进建议：{'; '.join(question_analysis.improvement_suggestions)}
- 改进后的问题：{question_analysis.improved_question}

答案分析：
- 原始答案质量评估：{', '.join(f'{k}: {v}' for k, v in answer_analysis.answer_quality.items())}
- 改进建议：{'; '.join(answer_analysis.improvement_suggestions)}""",
                'Solve': "基于改进后的问题和分析，提供解决方案：\n" + answer_analysis.improved_answer,
                'Verify': """验证改进效果：
- 问题改进是否有效
- 答案是否更加完整和准确
- 整体质量是否提升""",
                'Solution': answer_analysis.improved_answer
            },
            question_analysis=question_analysis,
            answer_analysis=answer_analysis,
            metadata={
                'original_format': 'alpaca',
                'transformation_template': template,
                'original_instruction': item.instruction,
                'original_input': item.input,
                'original_output': item.output,
                'sources': item.sources
            }
        )

class RefinedAlpacaItem(BaseModel):
    """带有结构化输出的Alpaca数据项"""
    instruction: str
    input: str = ""
    output: str
    refined_output: str = Field(
        description="包含结构化标签的改进输出，例如<Analyze>...</Analyze>"
    )
    sources: str = Field(default="")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_alpaca_item(cls, item: AlpacaItem, refined_output: str) -> 'RefinedAlpacaItem':
        """从AlpacaItem创建RefinedAlpacaItem"""
        return cls(
            instruction=item.instruction,
            input=item.input,
            output=item.output,
            refined_output=refined_output,
            sources=item.sources,
            metadata={
                'original_format': 'alpaca',
                'transformation_time': datetime.now().isoformat()
            }
        )
