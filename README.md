# 🔍 Research & Analysis Agent

An AI-powered research agent built on **LangGraph** and **Model Context Protocol (MCP)**.
The agent autonomously decides whether to search the web via Tavily, query a local vector
knowledge base (Qdrant), or both — then synthesizes a comprehensive answer with sources.

Tools are exposed as MCP instruments over SSE transport, keeping the embedding model
hot-loaded in memory. The ReAct loop runs on Groq's `llama-3.3-70b-versatile` with
session memory persisted per `thread_id`.

Built with **LangGraph** · **FastMCP** · **Qdrant** · **Groq** · **FastAPI**

---

## What it does

You ask a question. The agent decides whether to search the web, query your documents, or both — then synthesizes a final answer.

```
User question
     │
     ▼
 ReAct Agent (LangGraph)
     │
     ├──► search_documents()  →  Qdrant vector search
     ├──► web_search()        →  Tavily web search
     ├──► ingest_pdf_file()   →  Add PDF to knowledge base
     ├──► ingest_google_docs() → Add Google Doc to knowledge base
     ├──► read_google_doc()   →  Read Google Doc directly
     └──► list_files()        →  Browse server files
     │
     ▼
 Synthesized answer
```

---

## Stack

| Layer | Technology |
|-------|-----------|
| LLM | Groq — `llama-3.3-70b-versatile` |
| Agent framework | LangGraph (ReAct loop) |
| Tool protocol | Model Context Protocol (MCP) via FastMCP |
| Vector DB | Qdrant |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Web search | Tavily API |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS |
| Deployment | Docker Compose |

---

## Project structure

```
research-agent/
├── api.py              # FastAPI server — chat endpoint
├── graph.py            # LangGraph ReAct agent
├── mcp_server.py       # MCP server with all tools
├── rag.py              # RAG pipeline (chunking, embeddings, Qdrant)
├── main.py             # CLI runner for local testing
├── static/
│   └── index.html      # Chat UI
├── Dockerfile.api      # Docker image for FastAPI
├── Dockerfile.mcp      # Docker image for MCP server
├── docker-compose.yml  # Orchestration
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quickstart

### Prerequisites

- Docker & Docker Compose
- [Groq API key](https://console.groq.com)
- [Tavily API key](https://tavily.com)

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/research-agent.git
cd research-agent
```

### 2. Set up environment

```bash
cp .env.example .env
```

Edit `.env` and fill in your API keys:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### 3. Run with Docker Compose

```bash
docker compose up --build
```

Open **http://localhost:8000** in your browser.

---

## Local development (without Docker)

### Terminal 1 — MCP server

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

fastmcp run mcp_server.py --transport sse --port 8001
```

### Terminal 2 — FastAPI

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

---

## Adding documents to the knowledge base

### Via the agent (recommended)

Just ask the agent:

```
"Ingest this PDF: /path/to/document.pdf"
"Add this Google Doc to the knowledge base: https://docs.google.com/..."
```

### Via API directly

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Ingest /path/to/document.pdf into the knowledge base"}'
```

---

## API

### `POST /chat`

```json
{
  "question": "What does the document say about neural networks?",
  "thread_id": "optional-uuid-for-session-continuity"
}
```

**Response:**

```json
{
  "answer": "According to the document...",
  "thread_id": "a2f59f57-7a2b-41eb-81c4-d5a286a0b63f"
}
```

Pass the returned `thread_id` in subsequent requests to continue the conversation.

---

## How it works

### RAG Pipeline

Documents are chunked (fixed-size with overlap), embedded via `SentenceTransformer`, and stored in Qdrant. At query time, the agent embeds the question and retrieves the most semantically similar chunks.

### ReAct Agent

The agent follows a Reasoning + Acting loop: it decides which tool to call, observes the result, and repeats until it has enough context to answer. The LangGraph graph looks like this:

```
START → agent → tools → agent → ... → END
```

### MCP Protocol

All tools are exposed via the Model Context Protocol over SSE transport. The MCP server runs as a separate process, keeping the embedding model loaded in memory across requests.

### Session Memory

Conversation history is persisted per `thread_id` using LangGraph's `MemorySaver`. Each unique `thread_id` maintains its own context window.

---

## Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key | required |
| `TAVILY_API_KEY` | Tavily search API key | required |
| `QDRANT_URL` | Qdrant connection URL | `http://localhost:6333` |
| `COLLECTION_NAME` | Qdrant collection name | `research` |
| `EMBEDDING_MODEL` | HuggingFace model name | `all-MiniLM-L6-v2` |
| `LLM_MODEL` | Groq model ID | `llama-3.3-70b-versatile` |
| `MCP_URL` | MCP server SSE URL | `http://localhost:8001/sse` |

---

## Roadmap

- [ ] PostgreSQL persistence (`PostgresSaver`)
- [ ] RAG evaluation (RAGAS metrics)
- [ ] Reranking (cross-encoder, MMR)
- [ ] GitHub Actions CI/CD
- [ ] VPS deployment guide

---

## License

MIT