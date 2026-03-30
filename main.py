from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage
from graph import build_graph, make_agent_node, make_tool_node


async def main():
    client = MultiServerMCPClient({
        'research-tools': {
            'command': 'python',
            'args': ['mcp_server.py'],
            'transport': 'stdio'
        }
    })

    agent_node = make_agent_node(client)
    tool_node = await make_tool_node(client)

    app = build_graph(agent_node, tool_node)
    config = {'configurable': {'thread_id': '1'}}

    result = await app.ainvoke(
        {"messages": [HumanMessage(content="Сравни что написано в документах про ML с последними новостями в интернете")]},
        config=config
    )

    print("ANSWER:", result["messages"][-1].content)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())