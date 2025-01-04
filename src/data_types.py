from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class AlpacaItem:
    """原始 Alpaca 格式数据项"""
    instruction: str
    input: Optional[str]
    output: str

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

    @classmethod
    def from_alpaca_item(cls, item: AlpacaItem, refined_output: str, source: str = None, model_name: str = None):
        """从原始 AlpacaItem 创建 RefinedAlpacaItem"""
        return cls(
            instruction=item.instruction,
            input=item.input,
            output=item.output,
            refined_output=refined_output,
            source=source,
            model_name=model_name
        )

    def dict(self):
        """转换为字典格式"""
        return asdict(self)
