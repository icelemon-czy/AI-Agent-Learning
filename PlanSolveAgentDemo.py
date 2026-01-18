
import ast
import os
from typing import List
from TongYiClient import TongyiClient
from dotenv import load_dotenv
load_dotenv()

class PlanLLM:
    def __init__(self):
        self.client = TongyiClient()
        self.PLANNER_PROMPT_TEMPLATE = """
        你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
        请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
        你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

        问题: {question}

        请严格按照以下格式输出你的计划:
        ```
        ["步骤1", "步骤2", "步骤3", ...]
        ```
        """

    def plan(self, prompt: str) -> List[str]:
        prompt = self.PLANNER_PROMPT_TEMPLATE.format(question=prompt)
        messages = [
            {"role": "user", "content": f"{prompt}"}
        ]
        print("--- 正在生成计划 ---")
        response = self.client.think(messages)
        print(f"✅ 计划已生成.")

        # 解析LLM输出的列表字符串
        try:
            # 找到```和```之间的内容
            plan_str = response.split("```")[1].split("```")[0].strip()
            # 使用ast.literal_eval来安全地执行字符串，将其转换为Python列表
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ 解析计划时出错: {e}")
            print(f"原始响应: {response}")
            return []
        except Exception as e:
            print(f"❌ 解析计划时发生未知错误: {e}")
            return []

class ExcutorLLM:
    def __init__(self):
        self.client = TongyiClient()
        self.EXECUTOR_PROMPT_TEMPLATE = """
        你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
        你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
        请你专注于解决“当前步骤”，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

        # 原始问题:
        {question}

        # 完整计划:
        {plan}

        # 历史步骤与结果:
        {history}

        # 当前步骤:
        {current_step}

        请仅输出针对“当前步骤”的回答:
        """

    def execute(self, query: str, plan: list[str]) -> str:
        """
        根据计划，逐步执行并解决问题。
        """
        history = "" # 用于存储历史步骤和结果的字符串
        
        print("\n--- 正在执行计划 ---")
        
        for i, step in enumerate(plan):
            print(f"\n-> 正在执行步骤 {i+1}/{len(plan)}: {step}")
            
            prompt = self.EXECUTOR_PROMPT_TEMPLATE.format(
                question=query,
                plan=plan,
                history=history if history else "", # 如果是第一步，则历史为空
                current_step=step
            )
            
            messages = [{"role": "user", "content": prompt}]
            
            response_text = self.client.think(messages=messages) or ""
            
            # 更新历史记录，为下一步做准备
            history += f"步骤 {i+1}: {step}\n结果: {response_text}\n\n"
            
            print(f"✅ 步骤 {i+1} 已完成，结果: {response_text}")

        # 循环结束后，最后一步的响应就是最终答案
        final_answer = response_text
        return final_answer

class PlanSolveAgent:
    def __init__(self):
        self.planner = PlanLLM()
        self.executor = ExcutorLLM()

    def solve(self, prompt: str) -> str:
        print("\n=== 开始规划与执行流程 ===")
        
        # 1. 生成计划
        plan_steps = self.planner.plan(prompt)
        if not plan_steps:
            print("❌ 未能生成有效的计划，流程终止。")
            return "未能生成有效的计划。"
        
        print("\n生成的计划步骤:")
        for idx, step in enumerate(plan_steps, 1):
            print(f"  步骤 {idx}: {step}")
        
        # 2. 执行计划
        final_answer = self.executor.execute(prompt, plan_steps)
        
        print("\n=== 流程结束 ===")
        return final_answer
    
# --- 规划器使用示例 ---
if __name__ == '__main__':
    agent = PlanSolveAgent()
    complex_prompt = "问题: 一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"
    final_answer = agent.solve(complex_prompt)
    print("最终答案:")
    print(final_answer)