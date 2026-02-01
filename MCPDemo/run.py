import sys
import os

# Add root to sys.path to ensure we can import from Tools and Core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MCPDemo.bridge import SyncMCPClient, MCPToolAdapter
from Tools.toolManager import ToolManager
from Tools.toolAsyncExecutor import AsyncToolExecutor
import asyncio
import json

def main():
    # Path to server script
    server_script = os.path.join(os.path.dirname(__file__), 'server.py')
    
    print("Initializing MCP Client...")
    # Use python to run the server script. 
    # Ensure usage of the same python executable
    client = SyncMCPClient(sys.executable, [server_script])
    
    try:
        client.start()
        print("Connected to MCP Server!")
        
        # Initialize ToolManager
        tool_manager = ToolManager()
        
        print("\n--- Listing and Registering Tools ---")
        tools = client.list_tools()
        for t in tools:
            print(f"- Found tool: {t.name}")
            adapter = MCPToolAdapter(client, t)
            tool_manager.registerTool(adapter)
            
        # Verify 'add' tool is registered
        if tool_manager.get_tool("add"):
            print("\n--- Testing Async Parallel Execution with 'add' Tool ---")
            
            # Prepare tasks for AsyncToolExecutor
            # We use JSON string for parameters because ToolManager takes string input
            tasks = [
                {"tool_name": "add", "input_data": json.dumps({"a": 1, "b": 1})},
                {"tool_name": "add", "input_data": json.dumps({"a": 10, "b": 20})},
                {"tool_name": "add", "input_data": json.dumps({"a": 100, "b": 200})},
                {"tool_name": "add", "input_data": json.dumps({"a": 123, "b": 456})},
                {"tool_name": "echo", "input_data": json.dumps({"message": "Hello MCP!"})}
            ]
            
            async def run_async_test():
                with AsyncToolExecutor(tool_manager) as executor:
                    results = await executor.execute_tools_parallel(tasks)
                    return results

            # Run the async test
            results = asyncio.run(run_async_test())
            
            print("\n--- Results ---")
            for res in results:
                status = "✅" if res["status"] == "success" else "❌"
                print(f"{status} Task {res['task_id']} ({res['tool_name']}): {res['result']}")
                
        else:
            print("Tool 'add' not found!")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nDemo finished.")
        # Force exit because the background thread uses run_forever
        os._exit(0) 

if __name__ == "__main__":
    main()
