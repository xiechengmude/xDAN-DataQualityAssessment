from dataclasses import dataclass, asdict, field
from typing import Optional, Dict
from datetime import datetime

@dataclass
class AlpacaItem:
    """原始 Alpaca 格式数据项"""
    instruction: str
    input: Optional[str]
    output: str
    metadata: Optional[Dict] = field(default_factory=dict)

    def dict(self):
        """转换为字典格式"""
        return asdict(self)

@dataclass
class RefinedAlpacaItem:
    """经过结构化的 Alpaca 数据项"""
    instruction: str
    input: Optional[str]
    output: str
    refined_output: str
    source: Optional[str] = None
    model_name: Optional[str] = None
    original_dataset: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Optional[Dict] = field(default_factory=dict)

    @classmethod
    def from_alpaca_item(cls, item: AlpacaItem, refined_output: str, source: str = None, 
                        model_name: str = None, original_dataset: str = None):
        """从原始 AlpacaItem 创建 RefinedAlpacaItem"""
        return cls(
            instruction=item.instruction,
            input=item.input,
            output=item.output,
            refined_output=refined_output,
            source=source,
            model_name=model_name,
            original_dataset=original_dataset,
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            metadata=item.metadata
        )

    def dict(self):
        """转换为字典格式"""
        return asdict(self)
