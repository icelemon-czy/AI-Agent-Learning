import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

async def call_amap_mcp():
    load_dotenv()

    # 配置 MCP Server（通过 npx 启动高德官方服务）
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@amap/amap-maps-mcp-server"],
        env={"AMAP_MAPS_API_KEY": os.getenv("AMAP_MAPS_API_KEY")}
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 1. 初始化连接
            await session.initialize()
            
            # 2. 查看可用工具（调试用）
            tools = await session.list_tools()
            print("✅ 可用工具列表:")
            for tool in tools.tools:
                print(f"  • {tool.name}: {tool.description}")
                for key,value in tool.inputSchema.items():
                    print(f"{key} {value}")
                print()



            # 3. 调用具体工具（示例：查询福州天气）
            # result = await session.call_tool(
            #     "maps_weather",
            #     arguments={"city": "福州"}
            # )
            # print("\n🌤️ 天气查询结果:")
            # print(result.content)
            
            # 4. 其他工具调用示例（取消注释使用）
            # 地理编码：地址转坐标
            # geo_result = await session.call_tool("maps_geo", {"address": "北京市朝阳区望京"})
            # 路径规划
            # route_result = await session.call_tool("maps_driving", {
            #     "origin": "116.481488,39.985578",
            #     "destination": "116.397428,39.90923"
            # })

if __name__ == "__main__":
    asyncio.run(call_amap_mcp())