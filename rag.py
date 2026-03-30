from sentence_transformers import SentenceTransformer

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# PDF Reader
from pypdf import PdfReader

# ==================================== Chunking ====================================

def fixed_size_chunks(text: str, chunk_size: int, overlap: int) -> list:
    if not text or overlap >= chunk_size:
        raise ValueError('Overlap must be smaller than chunk size')
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start: start + chunk_size])
        start += chunk_size - overlap

    return chunks

# ==================================== Recorsive chunks ====================================

def recursive_chunks(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    # separators = ["\n\n", "\n", ". ", " "]
    if not separators:
        return [text]
    result = []
    chunks = text.split(separators[0])
    for chunk in chunks:
        if len(chunk) > chunk_size:
            result += recursive_chunks(chunk, chunk_size, separators[1:])
        else:
            result.append(chunk)

    return result

# ==================================== Embedding ====================================

def embed_chunks(chunks: list[str], model: SentenceTransformer) -> list:

    if not chunks:
        raise ValueError('Empty chunks')
    if not model:
        raise ValueError('Empty model name')

    vectors = model.encode(chunks).tolist()

    return vectors

# ==================================== Upsert chunks ====================================

def upsert_chunks(client: QdrantClient, collection_name: str, chunks: list[str], vectors: list) -> None:

    points = []
    for i, chunk in enumerate(chunks):
        points.append(PointStruct(
            id=i,
            vector=vectors[i],
            payload={
                "text": chunk,
                "chunk_order": i
            })
        )

    client.upsert(collection_name=collection_name, points=points)

# ==================================== Search chunks ====================================

def search_chunks(client: QdrantClient, collection_name: str, query: str, model: SentenceTransformer, top_k: int=3) -> list:

    vector = embed_chunks([query], model)[0]
    result = client.query_points(collection_name=collection_name, query=vector, limit=top_k)

    return result.points

# ==================================== Extract text from PDF ====================================

def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    return text

# ==================================== Pipeline ====================================
# Pipeline соединяем все вместе

def ingest_pdf(pdf_path: str, collection_name: str, client: QdrantClient, model: SentenceTransformer, chunk_size: int=500, overlap=50) -> None:
    text = extract_text_from_pdf(pdf_path)
    chunks = fixed_size_chunks(text, chunk_size, overlap)
    vectors = embed_chunks(chunks, model)
    upsert_chunks(client, collection_name, chunks, vectors)


def ingest_text(text: str, collection_name: str, client: QdrantClient, model: SentenceTransformer, chunk_size: int=500, overlap=50) -> None:
    chunks = fixed_size_chunks(text, chunk_size, overlap)
    vectors = embed_chunks(chunks, model)
    upsert_chunks(client, collection_name, chunks, vectors)