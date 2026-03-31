from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import shutil
import uuid
import os

from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph.state import CompiledStateGraph
from contextlib import asynccontextmanager

from graph import make_agent_node, make_tool_node, build_graph
from mcp_server import ingest_file


graph_app: CompiledStateGraph | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph_app
    os.makedirs("uploads", exist_ok=True)
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

    try:
        result = await graph_app.ainvoke(
            {"messages": [HumanMessage(content=request.question)]},
            config=config
        )
        return {
            "answer": result["messages"][-1].content,
            "thread_id": thread_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка агента: {str(e)}")


@app.post('/upload')
async def upload_file(file: UploadFile = File(...)):
    # 1. Проверить расширение
    allowed = ('.pdf', '.md', '.markdown')
    if not file.filename.endswith(allowed):
        raise HTTPException(status_code=400, detail=f"Неподдерживаемый формат. Разрешены: {allowed}")

    file_path = f"uploads/{file.filename}"

    with open(file_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = ingest_file(file_path)
        os.remove(file_path)
        return {"status": "success", "message": result}
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))
