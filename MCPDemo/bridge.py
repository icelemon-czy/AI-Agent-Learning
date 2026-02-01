import sys
import os
import asyncio
import threading
import json
from typing import Dict, Any, List, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool as MCPToolDef

# Add parent directory to path to allow importing Tools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Tools.tool import Tool, ToolParameter

class SyncMCPClient:
    """
    A synchronous wrapper around the Async MCP Client.
    Allows standard sync Python code to interact with an MCP server running in a background thread.
    """
    def __init__(self, command: str, args: List[str]):
        self.command = command
        self.args = args
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._start_loop, daemon=True)
        self.session: Optional[ClientSession] = None
        self.exit_stack = None
        self._connected = threading.Event()

    def _start_loop(self):
        """Internal method to run the asyncio loop in a separate thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._connect())
        self.loop.run_forever()

    async def _connect(self):
        """Internal async method to establish connection."""
        try:
            from contextlib import AsyncExitStack
            self.exit_stack = AsyncExitStack()
            
            server_params = StdioServerParameters(
                command=self.command,
                args=self.args,
                env=os.environ.copy()
            )
            
            read, write = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await self.session.initialize()
            self._connected.set()
        except Exception as e:
            print(f"Connection failed: {e}", file=sys.stderr)

    def start(self):
        """Starts the background thread and waits for connection."""
        self.thread.start()
        self._connected.wait(timeout=10)
        if not self._connected.is_set():
            raise TimeoutError("Failed to connect to MCP server")

    def list_tools(self) -> List[MCPToolDef]:
        """Synchronously list available tools."""
        if not self.session:
            raise RuntimeError("Client not connected")
        future = asyncio.run_coroutine_threadsafe(self.session.list_tools(), self.loop)
        result = future.result()
        return result.tools

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Synchronously call a tool."""
        if not self.session:
            raise RuntimeError("Client not connected")
        future = asyncio.run_coroutine_threadsafe(
            self.session.call_tool(name, arguments=arguments), 
            self.loop
        )
        return future.result()

class MCPToolAdapter(Tool):
    """
    Adapts an MCP Tool definition to the project's Tool base class.
    """
    def __init__(self, client: SyncMCPClient, mcp_def: MCPToolDef):
        # We access the name from the tuple if the base class has the bug, 
        # but here we just pass it to super().__init__
        super().__init__(mcp_def.name, mcp_def.description or "")
        self.client = client
        self.mcp_def = mcp_def

    def describe_inputs(self) -> List[ToolParameter]:
        """Convert MCP JSON schema to ToolParameter list."""
        params = []
        schema = self.mcp_def.inputSchema
        properties = schema.get("properties", {})
        
        for prop_name, prop_def in properties.items():
            params.append(ToolParameter(
                name=prop_name,
                type=prop_def.get("type", "string"),
                description=prop_def.get("description", "")
            ))
        return params

    def run(self, parameters: Dict[str, Any]) -> str:
        """Execute the MCP tool."""
        try:
            # Handle ToolManager's input wrapping behavior
            # ToolManager wraps input in {"input": "value"}
            # If the tool expects other parameters, we try to parse the input string as JSON
            final_params = parameters.copy()
            if "input" in parameters and len(parameters) == 1:
                input_val = parameters["input"]
                # Only attempt to unwrap if the MCP tool DOESN'T actually have a parameter named 'input'
                schema = self.mcp_def.inputSchema
                props = schema.get("properties", {})
                
                if "input" not in props and isinstance(input_val, str):
                    try:
                        import json
                        parsed = json.loads(input_val)
                        if isinstance(parsed, dict):
                            final_params = parsed
                    except json.JSONDecodeError:
                        # Maybe it's not JSON, just a simple string value. 
                        # If the tool has exactly one parameter, assign it?
                        # For now, let's leave it as is, or maybe handle single-arg tools
                        if len(props) == 1:
                            single_key = next(iter(props))
                            final_params = {single_key: input_val}

            result = self.client.call_tool(self.name, final_params)
            
            # MCP CallToolResult contains a list of contents (TextContent, ImageContent, etc.)
            output_parts = []
            if hasattr(result, 'content'):
                for content in result.content:
                    if content.type == 'text':
                        output_parts.append(content.text)
                    elif content.type == 'image':
                        output_parts.append(f"[Image: {content.mimeType}]")
                    else:
                        output_parts.append(str(content))
            return "\n".join(output_parts)
        except Exception as e:
            return f"Error executing MCP tool: {str(e)}"
