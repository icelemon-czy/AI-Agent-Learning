from abc import ABC,abstractmethod
from typing import Dict, Any, List
from pydantic import BaseModel

class ToolParameter(BaseModel):
    """工具参数定义"""
    name: str
    type: str
    description: str

class Tool(ABC):
    """
    工具基类
    """

    def __init__(self,name:str,description:str):
        self.name = name,
        self.description = description
    
    @abstractmethod
    def run(self, parameters: Dict[str, Any]) -> str:
        """执行工具"""
        pass
    
    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """获取工具参数定义"""
        pass

    # def to_dict(self) -> Dict[str, Any]:
    #     """转换为字典格式"""
    #     return {
    #         "name": self.name,
    #         "description": self.description,
    #         "parameters": [param.dict() for param in self.get_parameters()]
    #     }

    def __str__(self) -> str:
        return f"Tool(name={self.name})"