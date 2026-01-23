import os
from dotenv import load_dotenv
from typing import Optional,Literal,List,Dict
from langchain_community.chat_models import ChatTongyi, ChatOllama

class CoreLLM:
    """
    兼容多家LLM Supplier 
    """
    # 定义常量
    PROVIDER_OLLAMA = "OLLAMA"
    PROVIDER_TONGYI = "TONGYI"
    SUPPORT_LLMSUPPLIER = [PROVIDER_OLLAMA, PROVIDER_TONGYI]

    def __init__(self,
                provider: Optional[str] = None,  # 修改为 str
                model: Optional[str] = None,
                base_url: Optional[str] = None,
                api_key: Optional[str] = None,
                temperature: float = 0.7,
                **kwargs
                ):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        支持自动检测provider或使用统一的LLM_*环境变量配置。

        Args:
            model: 模型名称，如果未提供则从环境变量LLM_MODEL_ID读取
            api_key: API密钥，如果未提供则从环境变量读取
            base_url: 服务地址，如果未提供则从环境变量LLM_BASE_URL读取
            provider: LLM提供商，如果未提供则自动检测
            temperature: 温度参数
            # max_tokens: 最大token数
            # timeout: 超时时间，从环境变量LLM_TIMEOUT读取，默认60秒
        """
        load_dotenv() # Load environment variables from .env file

        self.provider = self._resolve_provider(provider)
        if not self.provider:
            raise ValueError("Does not find any Provider Information.")
        
        if self.provider == self.PROVIDER_TONGYI:
            self._init_TONGYI_LLM(model=model, api_key=api_key)
        elif self.provider == self.PROVIDER_OLLAMA:
            self._init_OLLAMA_LLM(model=model, base_url=base_url)

    def _resolve_provider(self, provider: Optional[str]):
        return provider or os.getenv("LLM_PROVIDER")

    def _init_OLLAMA_LLM(self, model: Optional[str], base_url: Optional[str]):
        self.model = model or os.getenv("OLLAMA_MODEL")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL")
        self._client = ChatOllama(
            model=self.model,
            base_url=self.base_url or "http://localhost:11434",
        )

    def _init_TONGYI_LLM(self,model:Optional[str],api_key:Optional[str]):
        self.model = model or os.getenv("LLM_TONGYI_MODEL")
        self.api_key = api_key or os.getenv("LLM_TONGYI_API_KEY")
        if not all([self.model, self.api_key]):
            raise ValueError("Model and API key must be provided either as arguments or environment variables.")
        self._client = ChatTongyi(
            model_name=self.model,
            api_key=self.api_key,
            streaming=True
        )

    def think(self, 
            messages: List[Dict[str, str]], 
            temperature: float = 0,
            stream :bool = False) -> str:
        """根据 provider 调用对应的 think 方法"""
        if self.provider == self.PROVIDER_TONGYI:
            return self._TONGYI_think(messages, temperature, stream)
        elif self.provider == self.PROVIDER_OLLAMA:
            return self._OLLAMA_think(messages, temperature,stream)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
    def _TONGYI_think(self, 
                    messages: List[Dict[str, str]], 
                    temperature: float = 0,
                    stream: bool = False) -> str:
        """通义千问的 think 实现"""       
        try:
            if stream:
                response = self._client.stream(input=messages, temperature=temperature)
                collected_content = []
                for chunk in response:
                    content = chunk.content or ""
                    collected_content.append(content)
                return "".join(collected_content)
            else:
                response = self._client.invoke(input=messages, temperature=temperature)
                return response.content
        except Exception as e:
            print(f"TongYi Error: {e}")
            return ""

    def _OLLAMA_think(self, messages: List[Dict[str, str]], temperature: float = 0, stream: bool = False) -> str:
        """本地 Ollama 的 think 实现，使用 ChatOllama"""
        try:
            if stream:
                response = self._client.stream(input=messages, temperature=temperature)
                collected_content = []
                for chunk in response:
                    content = chunk.content or ""
                    print(content, end="", flush=True)
                    collected_content.append(content)
                return "".join(collected_content)
            else:
                response = self._client.invoke(input=messages, temperature=temperature)
                return response.content
        except Exception as e:
            print(f"Local Ollama Error: {e}")
            return ""

# --- 客户端使用示例 ---
if __name__ == '__main__':
    llmClient = CoreLLM()
    messages = [{"role": "user", "content": "请介绍一下你是谁？ 大约200字左右"}]
    result = llmClient.think(messages,stream=False)
    print(result)
    print(" ")
    result = llmClient.think(messages,stream=True)
    print(result)
    