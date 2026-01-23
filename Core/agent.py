from abc import ABC, abstractmethod
from .llm import CoreLLM
from typing import Optional

"""
Agent Class
"""
class Agent(ABC):
    def __init__(
        self,
        llm: CoreLLM,
        system_prompt: Optional[str] = None
    ):
        self.llm = llm
        self.system_prompt = system_prompt
    
    @abstractmethod
    def run(self,user_prompt:str,**kwargs)->str:
        pass