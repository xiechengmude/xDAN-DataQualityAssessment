from dataclasses import dataclass, asdict, field
from typing import Optional, Dict
from datetime import datetime

@dataclass
class TokenInfo:
    """Token使用信息"""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost: float
    currency: str = 'CNY'  # 货币单位，默认为人民币

@dataclass
class AlpacaItem:
    """原始的 Alpaca 数据项"""
    instruction: str
    output: str
    id: Optional[int] = None
    input: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def dict(self):
        """转换为字典格式"""
        return asdict(self)

@dataclass
class RefinedAlpacaItem:
    """经过结构化的 Alpaca 数据项"""
    id: int
    instruction: str
    output: str
    refined_output: str
    token_info: TokenInfo
    input: Optional[str] = None
    model_name: str = "gpt-3.5-turbo"  # 默认模型
    sources: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    @classmethod
    def from_alpaca_item(cls, item: AlpacaItem, refined_output: str, token_info: TokenInfo, 
                        model_name: Optional[str] = None,
                        sources: Optional[str] = None):
        """从原始 AlpacaItem 创建 RefinedAlpacaItem"""
        return cls(
            id=item.id if item.id is not None else 0,
            instruction=item.instruction,
            input=item.input,
            output=item.output,
            refined_output=refined_output,
            token_info=token_info,
            model_name=model_name or "xDAN-L3-Chat",
            sources=sources,
            timestamp=datetime.now().isoformat(),
            metadata=item.metadata
        )

    def dict(self):
        """转换为字典格式"""
        return asdict(self)
