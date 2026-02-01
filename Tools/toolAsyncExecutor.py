import asyncio
import concurrent.futures
from typing import Dict, Any, List
from .toolManager import ToolManager

class AsyncToolExecutor:
    """异步工具执行器"""
    def __init__(self, toolmanager:ToolManager, max_workers:int = 4):
        self.manager = toolmanager
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    
    async def execute_tool_async(self,toolName:str, toolInput:str)-> str:
        # 异步执行单个工具
        def _execute():
            return self.manager.execute_tool(toolName,toolInput)
        
        loop = asyncio.get_event_loop()

        try:
            result = loop.run_in_executor(self.executor,_execute)
            return result
        except Exception as e:
            return f"❌ 工具 '{toolName}' 异步执行失败: {e}"

    async def execute_tools_parallel(self, tasks:  List[Dict[str, str]])->  List[Dict[str, Any]]:
        """
        并行执行多个工具
        
        Args:
            tasks: 任务列表，每个任务包含 tool_name 和 input_data
            
        Returns:
            执行结果列表，包含任务信息和结果
        """
        print(f"开始并行执行 {len(tasks)} 个工具任务")

        # 创建异步任务
        async_tasks = []
        for i, task in enumerate(tasks):
            tool_name = task.get("tool_name")
            input_data = task.get("input_data", "")

            if not tool_name:
                continue
                
            async_task = self.execute_tool_async(tool_name, input_data)
            async_tasks.append((i, task, async_task))

        # 等待所有任务完成
        results = []
        for i, task, async_task in async_tasks:
            try:
                result = await async_task
                results.append({
                    "task_id": i,
                    "tool_name": task["tool_name"],
                    "input_data": task["input_data"],
                    "result": result,
                    "status": "success"
                })
                print(f"✅ 任务 {i+1} 完成: {task['tool_name']}")
            except Exception as e:
                results.append({
                    "task_id": i,
                    "tool_name": task["tool_name"],
                    "input_data": task["input_data"],
                    "result": str(e),
                    "status": "error"
                })
                print(f"❌ 任务 {i+1} 失败: {task['tool_name']} - {e}")
        return results
    
    async def execute_tools_batch(self, tool_name: str, input_list: List[str]) -> List[Dict[str, Any]]:
        """
        批量执行同一个工具
        
        Args:
            tool_name: 工具名称
            input_list: 输入数据列表
            
        Returns:
            执行结果列表
        """
        tasks = [
            {"tool_name": tool_name, "input_data": input_data}
            for input_data in input_list
        ]
        return await self.execute_tools_parallel(tasks)

    def close(self):
        """关闭执行器"""
        self.executor.shutdown(wait=True)
        print("🔒 异步工具执行器已关闭")
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# 便捷函数
async def run_parallel_tools(toolmanager:ToolManager, tasks: List[Dict[str, str]], max_workers: int = 4) -> List[Dict[str, Any]]:
    """
    便捷函数：并行执行多个工具
    
    Args:
        registry: 工具注册表
        tasks: 任务列表
        max_workers: 最大工作线程数
        
    Returns:
        执行结果列表
    """
    async with AsyncToolExecutor(toolmanager, max_workers) as executor:
        return await executor.execute_tools_parallel(tasks)

async def run_batch_tool(toolmanager:ToolManager, tool_name: str, input_list: List[str], max_workers: int = 4) -> List[Dict[str, Any]]:
    """
    便捷函数：批量执行同一个工具
    
    Args:
        registry: 工具注册表
        tool_name: 工具名称
        input_list: 输入数据列表
        max_workers: 最大工作线程数
        
    Returns:
        执行结果列表
    """
    async with AsyncToolExecutor(toolmanager, max_workers) as executor:
        return await executor.execute_tools_batch(tool_name, input_list)

