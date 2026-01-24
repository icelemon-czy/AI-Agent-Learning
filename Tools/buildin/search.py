from ..tool import Tool
import os
from typing import Dict, Any

class SearchTool(Tool):
    """
    智能混合搜索工具

    支持多种搜索引擎后端，智能选择最佳搜索源：
    1. 混合模式 (hybrid) - 智能选择TAVILY或SERPAPI
    2. Tavily API (tavily) - 专业AI搜索
    3. SerpApi (serpapi) - 传统Google搜索
    """
    
    # 定义模式常量，避免硬编码
    MODE_HYBRID = "hybrid"
    MODE_TAVILY = "tavily"
    MODE_SERPAPI = "serpapi"
    
    SUPPORTED_MODES = [MODE_HYBRID, MODE_TAVILY, MODE_SERPAPI]
    
    def __init__(self, mode: str = MODE_HYBRID):
        super().__init__(
            name="search",
            description="智能搜索工具，支持多种搜索引擎后端"
        )
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode: {mode}. Supported: {self.SUPPORTED_MODES}")
        
        self.mode = mode
        self._init_clients()
    
    def _init_clients(self):
        """初始化搜索客户端"""
        # Tavily API
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        
        # SerpApi
        self.serpapi_api_key = os.getenv("SERPAPI_API_KEY")
        
        # 检查 API keys
        if self.mode in [self.MODE_HYBRID, self.MODE_TAVILY] and not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY is required for Tavily mode")
        if self.mode in [self.MODE_HYBRID, self.MODE_SERPAPI] and not self.serpapi_api_key:
            raise ValueError("SERPAPI_API_KEY is required for SerpApi mode")
    
    def run(self, parameters: Dict[str, Any]) -> str:
        """执行搜索"""
        query = parameters.get("query", "")
        if not query:
            return "搜索查询不能为空"
        
        if self.mode == self.MODE_HYBRID:
            return self._hybrid_search(query)
        elif self.mode == self.MODE_TAVILY:
            return self._tavily_search(query)
        elif self.mode == self.MODE_SERPAPI:
            return self._serpapi_search(query)
        else:
            return f"不支持的模式: {self.mode}"
    
    def _hybrid_search(self, query: str) -> str:
        """混合搜索：优先 Tavily，失败时回退到 SerpApi"""
        try:
            return self._tavily_search(query)
        except Exception as e:
            print(f"Tavily 搜索失败，回退到 SerpApi: {e}")
            return self._serpapi_search(query)
    
    def _tavily_search(self, query: str) -> str:
        """使用 Tavily API 搜索"""
        # 这里实现 Tavily API 调用
        # 示例：使用 requests 调用 Tavily API
        # import requests
        # response = requests.post("https://api.tavily.com/search", json={"query": query, "api_key": self.tavily_api_key})
        # return response.json().get("results", "No results")
        return f"Tavily 搜索结果 for: {query} (请实现实际 API 调用)"
    
    def _serpapi_search(self, query: str) -> str:
        """使用 SerpApi 搜索"""
        # 这里实现 SerpApi 调用
        # 示例：使用 serpapi 库
        # from serpapi import GoogleSearch
        # search = GoogleSearch({"q": query, "api_key": self.serpapi_api_key})
        # return search.get_dict().get("organic_results", [])
        return f"SerpApi 搜索结果 for: {query} (请实现实际 API 调用)"