from typing import TypedDict, Annotated
from dotenv import load_dotenv
import os
from datetime import datetime

from langgraph.prebuilt import ToolNode, tools_condition
from qdrant_client import QdrantClient

from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.types import interrupt, Send
from langgraph.graph import StateGraph

from rag import search_chunks

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model=os.getenv('LLM_MODEL'),
)


# =============================== State class ===============================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    rag_results: list[str]
    web_results: list[str]
    answer: str
    sources: list[str]  # откуда пришёл ответ (для пользователя)
    error: str          # если что-то пошло не так
    route: str # 'rag', 'web' или 'both'

# =============================== Route node ===============================
def route_node(state: AgentState) -> dict:
    question = state['question']
    route_prompt = f'''
    Определи источник поиска для {question}.
    Если вопрос про загруженные документы → ответь "rag"
    Если вопрос про актуальные новости/события → ответь "web"
    Ответь ТОЛЬКО одним словом: rag или web или both
    '''
    response = llm.invoke(route_prompt)
    route = response.content.strip().lower()

    return {'route': route}


# =============================== Route decision ===============================

def route_decision(state: AgentState) -> str:
    route = state['route']
    if route == 'both':
        return [Send('rag', state), Send('web', state)]
    return route


# =============================== Rag node ===============================
db_client = QdrantClient("localhost", port=6333)

def rag_node(state: AgentState) -> dict:
    question = state['question']
    rag_res = search_chunks(client=db_client, collection_name=os.getenv("COLLECTION_NAME"), query=question, model_name=os.getenv('EMBEDDING_MODEL'), top_k=3)
    results = [r.payload['text'] for r in rag_res]
    return {'rag_results': results}


# =============================== Make Web node ===============================

def make_web_node(client):

    async def web_node(state: AgentState) -> dict:
        query = state['question']
        answer = interrupt('Искать в интернете? (да/нет)')
        if answer.lower() != 'да':
            return {"web_results": ['Пользователь отказался от веб-поиска']}
        tools = await client.get_tools()  # не await! это синхронный метод
        web_tool = next(t for t in tools if t.name == "web_search")
        result = await web_tool.ainvoke({"query": query})

        return {'web_results': result}


    return web_node


# =============================== Answer node ===============================

def answer_node(state: AgentState) -> dict:
    question = state['question']
    messages = state.get('messages', [])
    rag_results = state.get('rag_results', [])  # ✅ пустой список если нет
    web_results = state.get('web_results', [])  # ✅ пустой список если нет

    if state.get('route') == 'both':
        result = rag_results + web_results
    elif rag_results:
        result = rag_results
    else:
        result = web_results

    # история разговора
    history = "\n".join([
        f"{'Пользователь' if isinstance(m, HumanMessage) else 'Агент'}: {m.content}"
        for m in messages
    ])
    flat_result = []
    for item in result:
        if isinstance(item, list):
            flat_result.extend(item)
        elif isinstance(item, dict):
            flat_result.append(item.get('text', ''))
        else:
            flat_result.append(str(item))

    prompt = f"""
    Ответь на вопрос используя контекст и историю разговора.
    История разговора: {history}          
    Вопрос: {question}
    Контекст:
    {"\n".join(flat_result)}
    """
    response = llm.invoke(prompt)
    return {
        'answer': response.content,
        'messages': [
            HumanMessage(content=question),
            AIMessage(content=response.content),
        ]
    }

# =============================== Agent node ===============================

def make_agent_node(client):
    async def agent_node(state: AgentState) -> dict:
        tools = await client.get_tools()
        llm_with_tools = llm.bind_tools(tools)

        last_error = None
        for attempt in range(3):
            try:
                system = SystemMessage(content=f"""Ты Research & Analysis Agent. 
                                                У тебя есть инструменты для поиска информации.
                                                Всегда отвечай на русском языке.
                                                Текущее время: {datetime.now().isoformat()}""")
                messages = [system] + state['messages']
                response = await llm_with_tools.ainvoke(messages)
                return {'messages': [response]}
            except Exception as e:
                last_error = e
                continue

        raise last_error

    return agent_node

# =============================== Tool node ===============================

async def make_tool_node(client):
    tools = await client.get_tools()
    return ToolNode(tools=tools)


# =============================== Graph ===============================

def build_graph(agent_node, tool_node):
    checkpointer = MemorySaver()
    graph = StateGraph(AgentState)

    graph.add_node('agent', agent_node)
    graph.add_node('tools', tool_node)

    graph.set_entry_point('agent')

    # Conditional edge - заменяет обычный add_edge от router
    graph.add_conditional_edges('agent', tools_condition)

    graph.add_edge('tools', 'agent')

    app = graph.compile(checkpointer=checkpointer)

    return app
