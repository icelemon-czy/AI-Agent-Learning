import sys
from mcp.server.fastmcp import FastMCP

# Initialize the server
mcp = FastMCP("Demo Server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@mcp.tool()
def echo(message: str) -> str:
    """Echo back the message."""
    return f"Echo: {message}"

if __name__ == "__main__":
    # Runs on stdio by default
    print("Starting MCP Demo Server...", file=sys.stderr)
    try:
        mcp.run()
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
