import os
from dotenv import load_dotenv
# 配置好同级文件夹下.env中的大模型API
from hello_agents import SimpleAgent, HelloAgentsLLM, ToolRegistry
from hello_agents.tools import MemoryTool, RAGTool

# 加载 .env 环境变量
load_dotenv()

# 创建LLM实例
llm = HelloAgentsLLM(
    api_key=os.getenv("LLM_TONGYI_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", # 通义千问兼容 OpenAI 的 endpoint
    model=os.getenv("LLM_TONGYI_MODEL", "qwen-turbo")
)

# 创建Agent
agent = SimpleAgent(
    name="智能助手",
    llm=llm,
    system_prompt="你是一个有记忆和知识检索能力的AI助手"
)

# 创建工具注册表
tool_registry = ToolRegistry()

# 添加记忆工具
memory_tool = MemoryTool(user_id="user123")
tool_registry.register_tool(memory_tool)

# 添加RAG工具
rag_tool = RAGTool(knowledge_base_path="./knowledge_base",llm=llm)
tool_registry.register_tool(rag_tool)

# 为Agent配置工具
agent.tool_registry = tool_registry

# 开始对话
response = agent.run("你好！请记住我叫张三，我是一名Python开发者")
print(response)


