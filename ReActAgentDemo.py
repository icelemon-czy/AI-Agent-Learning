from serpapi import SerpApiClient
import os
import re
from dotenv import load_dotenv
from typing import Dict, Any, Callable
from TongYiClient import TongyiClient
load_dotenv()

def search(query:str) -> str:
    """
    一个基于SerpApi的实战网页搜索引擎工具。
    它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
    """
    print(f"正在执行[SerpApi] 网页搜索: {query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            raise Exception("SERPAPI_API_KEY environment variable is not set.")
        
        params = {
            "engine": "google",
            "q": query,         
            "api_key": api_key,
            "gl":"us",
            "hl": "zh-cn",
        }

        client = SerpApiClient(params_dict=params)
        results = client.get_dict()

        # 解析答案
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # 如果没有直接答案，则返回前三个有机结果的摘要
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
        
        return f"对不起，没有找到关于 '{query}' 的信息。"
    except Exception as e:
        return f"搜索时出错: {e}"

def addition(int1:str, int2:str) -> str:
    """
    一个简单的加法计算器工具，能够计算数学加法表达式的结果。
    """
    return str(int(int1) + int(int2))

class ToolExcutor:
    """
    一个工具执行器，负责管理和执行工具。
    """
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def register_tool(self, name: str, description: str, func: Callable):
        """
        向工具箱中注册一个新工具。
        """
        if name in self.tools:
            print(f"警告:工具 '{name}' 已存在，将被覆盖。")
        self.tools[name] = {
            "description": description,
            "function": func
        }
        print(f"工具 '{name}' 已成功注册。")
    
    def getAllAvilableTools(self) -> str:
        """
        获取所有已注册工具的描述。
        """
        descriptions = [
            f"{name}: {info['description']}"
            for name, info in self.tools.items()
        ]
        return "\n".join(descriptions)

    def getTool(self, name: str) -> Callable:
        """
        根据名称获取工具函数。
        """
        tool = self.tools.get(name)
        if tool is None:
            print(f"错误: 工具 '{name}' 未找到。")
            return None
        return tool["function"]

# --- 工具注册示例 ---

class ReActAgentDemo:
    def __init__(self):
        self.tool_executor = ToolExcutor()
        self.tool_executor.register_tool(
            name="网页搜索",
            description="用于在互联网上搜索信息，适用于需要查找最新信息或具体事实的场景。 输入参数1: 搜索关键词",
            func=search
        )
        self.tool_executor.register_tool(
            name="加法计算器",
            description="用于计算两个整数的加法结果。 输入参数1: 第一个整数 输入参数2: 第二个整数",
            func=addition
        )
        self.llm_client = TongyiClient()
        self.PromptTemplate = """
        你是一个智能的ReAct风格的AI助手。你可以使用以下工具来帮助用户解决问题：
        {tools} 

        你的工作流程必须严格遵守以下规则：
        1. **单步执行**：每一轮对话中，你**只能**生成**一个** Thought 和 **一个** Action。
        2. **等待反馈**：生成 Action 后，**必须立即停止输出**，等待系统提供 Observation（观察结果）。
        3. **禁止自问自答**：绝对不要在一个回答中捏造 Observation，也不要连续生成多个 Action。

        请严格按照以下格式进行回应:

        Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
        Action: 你决定采取的行动，必须是以下格式之一:
            - `[tool_name][tool_input1,tool_input2,...etc]` (仅当 Reach_Final_Answer 为 False 时使用)
        FinalAnswer: 你的决策结果，格式必须严格为:
            - `[Reach_Final_Answer][Content]`
        特别说明:
        - `Reach_Final_Answer`: 布尔值 (True 或 False)。
            - 如果你需要调用工具，设为 `False`,`Content` 留空或写 "None"。
            - 如果你已经得到最终答案，设为 `True`,`Content` 为最终答案文本。
        - Thought, Action 和 FinalAnswer 每一轮只能进行一次。

        现在，请开始解决以下问题:
        Question: {question}
        History: {history}
        """
        print("ReActAgentDemo 初始化完成。\n")
    
    def _parse_output(self, text: str):
        """解析LLM的输出，提取Thought和Action。"""
        thought_match = re.search(r"Thought: (.*)", text)
        action_match = re.search(r"Action: (.*)", text)
        final_answer_match = re.search(r"FinalAnswer: (.*)", text)

        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        final_answer = final_answer_match.group(1).strip() if final_answer_match else None

        return thought, action, final_answer
    
    def _parse_final_answer(self, final_answer_text: str):
        if not final_answer_text:
            return None, None
        """解析FinalAnswer字符串，提取Reach_Final_Answer和Content。"""
        match = re.match(r"\[(True|False)\]\[(.*)\]", final_answer_text,re.IGNORECASE)
        if match:
            reach_final_answer = match.group(1).lower() == "true"
            content = match.group(2)
            return reach_final_answer, content
        return None, None
    
    def _parse_action(self, action_text: str):
        """解析Action字符串，提取工具名称和输入。"""
        if not action_text:
            return None, None
        
        # 匹配格式: [tool_name][input]
        # 使用非贪婪匹配 (.*?) 来防止第一个括号吃掉后面的内容
        match = re.match(r"\[(.*?)\]\[(.*)\]", action_text)
        
        if match:
            tool_name = match.group(1).strip()
            tool_input_str = match.group(2).strip()
            
            # 处理多参数情况（按逗号分割）
            # 例如: "1, 2" -> ["1", "2"]
            if "," in tool_input_str:
                tool_inputs = [arg.strip() for arg in tool_input_str.split(",")]
                return tool_name, tool_inputs
            elif tool_input_str:
                # 单参数也必须包在列表里！
                return tool_name, [tool_input_str]
            else:
                # 无参数情况
                return tool_name, []
        return None, None
    
    def run(self, query: str):
        print(f"User Prompt: {query}\n")
        StopIteration = 10
        CurrentIteration = 0     
        self.history = []

        while CurrentIteration < StopIteration:
            CurrentIteration += 1
            print( f"--- 第 {CurrentIteration} 轮 ---" )
            
            # 1. 格式化提示词
            tools_desc = self.tool_executor.getAllAvilableTools()
            history_str = "\n".join(self.history)
            prompt = self.PromptTemplate.format(
                tools=tools_desc,
                question=query,
                history=history_str
            )

            # 2. 调用LLM获取响应
            messages = [
                {"role": "user", "content": prompt}
            ]
            response = self.llm_client.think(messages)
            print(messages)
            print(f"LLM响应:\n{response}\n")
            # 3. 解析LLM的输出
            thought, action, final_answer = self._parse_output(response)
            if thought:
                print(f"思考: {thought}")
            
            # 4. 检查是否达成最终答案
            reach_final_answer, content = self._parse_final_answer(final_answer)
            if reach_final_answer is not None:
                if reach_final_answer:
                    print(f"🎉 最终答案: {content}")
                    return content

            # 5. 执行Action          
            if action:
                tool_name, tool_inputs = self._parse_action(action)

                if not tool_name:
                    observation = "不需要调用工具。"
                    print("不需要调用工具，跳过本轮。")
                    continue
                print(f"🔧 执行工具: {tool_name}，输入: {tool_inputs}")
                tool_function = self.tool_executor.getTool(tool_name)
                if not tool_function:
                    observation = f"Error: Tool '{tool_name}' not found."
                else:
                    try:
                        # === 万能调用写法 ===
                        # *tool_inputs 会把列表拆开，按顺序传给函数
                        # 例如: search(*["华为"]) -> search("华为")
                        # 例如: addition(*["1", "2"]) -> addition("1", "2")
                        observation = tool_function(*tool_inputs) 
                    except TypeError as e:
                        observation = f"Tool Argument Error: {str(e)} (Check argument count)"
                    except Exception as e:
                        observation = f"Tool Execution Error: {str(e)}"
            
            # 将本轮的Action和Observation添加到历史记录中
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")
        print("达到最大迭代次数，流程终止。")
        return None
    
if __name__ == "__main__":
    agent = ReActAgentDemo()
    user_query = "249 + 761 + 13785 等于多少？"
    # user_query = "华为最新的手机型号是什么？"
    agent.run(user_query)
