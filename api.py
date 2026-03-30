from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
import uuid
import os
from pydantic import BaseModel
from contextlib import asynccontextmanager
from graph import make_agent_node, make_tool_node, build_graph
from langgraph.graph.state import CompiledStateGraph


graph_app: CompiledStateGraph | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph_app
    client = MultiServerMCPClient({
        'research-tools': {
            'transport': 'sse',
            'url': os.getenv('MCP_URL', 'http://localhost:8001/sse'),
        }
    })
    agent_node = make_agent_node(client)
    tool_node = await make_tool_node(client)

    graph_app = build_graph(agent_node, tool_node)
    yield

app = FastAPI(lifespan=lifespan)

app.mount('/static', StaticFiles(directory='static'), name='static')


class ChatRequest(BaseModel):
    question: str
    thread_id: str | None = None


@app.get('/')
async def root():
    return FileResponse('static/index.html')

@app.post('/chat')
async def chat(request: ChatRequest):
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {'configurable': {'thread_id': thread_id}}

    result = await graph_app.ainvoke(
        {"messages": [HumanMessage(content=request.question)]},
        config=config
    )

    return {
        "answer": result["messages"][-1].content,
        "thread_id": thread_id
    }