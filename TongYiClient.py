import os
from dotenv import load_dotenv
from typing import List, Dict 
from langchain_community.chat_models import ChatTongyi

class TongyiClient:
    def __init__(self,model: str = None, api_key: str=None):
        load_dotenv() # Load environment variables from .env file
        self.model = model or os.getenv("LLM_TONGYI_MODEL")
        self.api_key = api_key or os.getenv("LLM_TONGYI_API_KEY")
        if not all([self.model, self.api_key]):
            raise ValueError("Model and API key must be provided either as arguments or environment variables.")
        self.llm_client = ChatTongyi(
            model_name=self.model,
            api_key=self.api_key,
            streaming=True
        )

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        try:
            response = self.llm_client.stream(input=messages,temperature=temperature)
            collected_content = []
            for chunk in response:
                content = chunk.content or ""
                collected_content.append(content)
            return "".join(collected_content)
        except Exception as e:
                print(e)
                
# --- 客户端使用示例 ---
if __name__ == '__main__':
    llmClient = TongyiClient()
    
    exampleMessages = [
        {"role": "system", "content": "You are a helpful assistant that writes Python code."},
        {"role": "user", "content": "写一个快速排序算法"}
    ]
    
    print("--- 调用LLM ---")
    responseText = llmClient.think(exampleMessages)
    if responseText:
        print("\n\n--- 完整模型响应 ---")
        print(responseText)

    
    


    