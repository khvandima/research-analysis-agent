import os
from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient
from qdrant_client import QdrantClient
import requests

from rag import ingest_text, search_chunks, ingest_pdf, ingest_markdown
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP('research-tools')
qdrant_client = QdrantClient(host="localhost", port=6333)
embedding_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL"))

# ============================== File reader ==============================
@mcp.tool()
def list_files(path: str) -> str:
    """Возвращает список файлов в указанной папке на сервере. Требует реальный путь к существующей папке, например '/home/user/documents'"""
    try:
        files = os.listdir(path)
        return "\n".join(files) if files else "Папка пуста"
    except FileNotFoundError:
        return f"Ошибка: папка '{path}' не существует"
    except Exception as e:
        return f"Ошибка: {e}"


@mcp.tool()
def read_file(path: str) -> str:
    """Возвращает содержимое файла"""
    with open(path, 'r') as f:
        return f.read()

# ============================== Web search ==============================
@mcp.tool()
def web_search(query: str) -> str:
    """Ищет информацию в интернете по запросу"""
    client = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))
    response = client.search(query)
    res = ''
    for result in response['results']:
        res += result['content']
    return res

# ============================== Google docs ==============================

def create_export_url(url: str) -> str:
    """Функция для создания export_url чтобы по ссылке можно было прочитать файл не загружая"""
    doc_id = url.split('/d/')[1].split('/')[0]
    export_url = f'https://docs.google.com/document/d/{doc_id}/export?format=txt'
    return export_url


@mcp.tool()
def ingest_google_docs(url: str) -> str:
    """Считывает документ по ссылке с google docs, нарезает на чанки делает embidding и добавляет его в базу данных qdrant"""
    export_url = create_export_url(url)

    try:
        response = requests.get(export_url)
        response.raise_for_status()
        if not response.text.strip():
            return "Ошибка: документ пустой или недоступен"
        text = response.text
        ingest_text(text, os.getenv("COLLECTION_NAME"), qdrant_client, model=embedding_model)
        return "Документ успешно загружен в базу знаний"
    except Exception as e:
        return f"Ошибка: {e}"


@mcp.tool()
def read_google_doc(url: str) -> str:
    """Считывает документ по ссылке с google docs и возвращает текст"""
    export_url = create_export_url(url)

    try:
        response = requests.get(export_url)
        response.raise_for_status()
        if not response.text.strip():
            return "Ошибка: документ пустой или недоступен"
        text = response.text.strip()
        return text
    except Exception as e:
        return f"Ошибка: {e}"

# ============================== RAG ==============================

@mcp.tool()
def search_documents(query: str) -> str:
    """Ищет релевантную информацию в загруженных документах по смыслу"""
    results = search_chunks(client=qdrant_client, query=query, collection_name=os.getenv("COLLECTION_NAME"), model=embedding_model)
    return "\n".join([p.payload.get("text", "") for p in results])


@mcp.tool()
def ingest_file(file_path: str) -> str:
    """Загружает документ в базу знаний. Автоматически определяет формат по расширению. Поддерживает PDF (.pdf) и Markdown (.md, .markdown)"""
    try:
        if file_path.endswith('.pdf'):
            ingest_pdf(pdf_path=file_path, collection_name=os.getenv("COLLECTION_NAME"), client=qdrant_client, model=embedding_model)
        elif file_path.endswith(('.md', '.markdown')):
            ingest_markdown(md_path=file_path, collection_name=os.getenv("COLLECTION_NAME"), client=qdrant_client, model=embedding_model)
        else:
            return f"Неподдерживаемый формат: {file_path}. Поддерживаются .pdf и .md"
        return 'Файл успешно добавлен в базу знаний'
    except FileNotFoundError:
        return f"Файл не найден: {file_path}"
    except Exception as e:
        return str(e)


if __name__ == '__main__':
    mcp.run()
