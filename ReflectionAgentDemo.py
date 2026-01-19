from TongYiClient import TongyiClient
from Memory import Memory

class ReflectionAgentDemo:
    def __init__(self):
        self.llm = TongyiClient()
        self.memory = Memory()
        self.INIT_PROMPT = """
        你是一位资深的Python程序员。请根据以下要求，编写一个Python函数。
        你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。

        要求: {task}

        请直接输出代码，不要包含任何额外的解释。
        """
        

        self.REFLECT_PROMPT = """
        你是一位极其严格的代码评审专家和资深算法工程师，对代码的性能有极致的要求。
        你的任务是审查以下Python代码，并专注于找出其在<strong>算法效率</strong>上的主要瓶颈。

        # 原始任务:
        {task}

        # 待审查的代码:
        ```python
        {code}
        ```

        请分析该代码的时间复杂度，并思考是否存在一种<strong>算法上更优</strong>的解决方案来显著提升性能。
        如果存在，请清晰地指出当前算法的不足，并提出具体的、可行的改进算法建议（例如，使用筛法替代试除法）。
        如果代码在算法层面已经达到最优，才能回答“无需改进”。

        请直接输出你的反馈，不要包含任何额外的解释。   
        """

        self.REFINE_PROMPT = """
        你是一位资深的Python程序员。你正在根据一位代码评审专家的反馈来优化你的代码。

        # 原始任务:
        {task}

        # 你上一轮尝试的代码:
        {last_code_attempt}
        评审员的反馈：
        {feedback}

        请根据评审员的反馈，生成一个优化后的新版本代码。
        你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。
        请直接输出优化后的代码，不要包含任何额外的解释。
        """
    
    def run(self, query: str):
        MAX_ITERATIONS = 3

        # 1. 根据初始query 生成代码
        prompt = self.INIT_PROMPT.format(
            task=query,
        )
        messages = [
            {"role": "user", "content": prompt}
        ]
        print("--- 正在生成初始答案 ---")
        response = self.llm.think(messages)
        self.memory.add_record("execution", response)
        print(f"✅ 初始代码已答案.")

        for iteration in range(MAX_ITERATIONS):
            print(f"\n===== 迭代轮次 {iteration + 1} =====")
            # 2. 反思与评审
            prompt = self.REFLECT_PROMPT.format(
                task=query,
                code=self.memory.get_last_execution()
            )
            messages = [
                {"role": "user", "content": prompt}
            ]
            print("--- 正在进行代码反思与评审 ---")
            response = self.llm.think(messages)
            self.memory.add_record("reflection", response)
            print(f"✅ 反思与评审完成.")

            # 3. 检查是否需要停止
            if "无需改进" in response:
                print("\n✅ 反思认为代码已无需改进，任务完成。")
                break

            # 4. 根据评审反馈优化代码
            prompt = self.REFINE_PROMPT.format(
                task=query,
                last_code_attempt=self.memory.get_last_execution(),
                feedback=response
            )
            messages = [
                {"role": "user", "content": prompt}
            ]
            print("--- 正在根据评审反馈优化代码 ---")
            response = self.llm.think(messages)
            self.memory.add_record("execution", response)
            print(f"✅ 代码优化完成.")

        final_code = self.memory.get_last_execution()
        print(f"\n--- 任务完成 ---\n最终生成的代码:\n```python\n{final_code}\n```")
        return final_code


if __name__ == "__main__":
    agent = ReflectionAgentDemo()
    task_description = "实现一个函数，输入一个整数n，返回小于等于n的所有质数列表。要求算法效率尽可能高。"
    agent.run(task_description)