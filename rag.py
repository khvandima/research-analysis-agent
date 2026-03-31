from sentence_transformers import SentenceTransformer

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector, Prefetch, FusionQuery, Fusion
from qdrant_client.models import VectorParams, Distance, SparseVectorParams, SparseIndexParams
from fastembed import SparseTextEmbedding

# PDF Reader
from pypdf import PdfReader
import hashlib

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

# ==================================== Markdown chunking ====================================

def extract_chunks_from_markdown(md_path: str, chunk_size: int=500, overlap: int=50) -> list[str]:
    with open(md_path, 'r', encoding='utf-8') as f:
        text = f.read()

    separators = [
        '\n# ',    # заголовок 1 уровня
        '\n## ',   # заголовок 2 уровня
        '\n### ',  # заголовок 3 уровня
        '\n\n',    # параграфы
        '\n',      # строки
        '. ',      # предложения
    ]

    return recursive_chunks(text, chunk_size, separators)


# ==================================== Embedding ====================================

def embed_chunks(chunks: list[str], model: SentenceTransformer) -> list:

    if not chunks:
        raise ValueError('Empty chunks')
    if not model:
        raise ValueError('Empty model name')

    vectors = model.encode(chunks).tolist()

    return vectors

# ==================================== Upsert chunks ====================================

def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name,
            vectors_config={
                "dense": VectorParams(size=vector_size, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams())
            }
        )


# ==================================== Upsert chunks ====================================

sparse_encoder = SparseTextEmbedding(model_name="Qdrant/bm25")

def upsert_chunks(client: QdrantClient, collection_name: str, chunks: list[str], vectors: list) -> None:
    ensure_collection(client, collection_name, len(vectors[0]))
    sparse_embeddings = list(sparse_encoder.embed(chunks))
    points = []
    for i, chunk in enumerate(chunks):
        points.append(PointStruct(
            id=int(hashlib.md5(chunk.encode()).hexdigest(), 16) % (10**18),
            vector={
                "dense": vectors[i],
                "sparse": SparseVector(
                    indices=sparse_embeddings[i].indices.tolist(),
                    values=sparse_embeddings[i].values.tolist()
                )
            },
            payload={
                "text": chunk,
                "chunk_order": i
            })
        )

    client.upsert(collection_name=collection_name, points=points)

# ==================================== Search chunks ====================================

def search_chunks(client: QdrantClient, collection_name: str, query: str, model: SentenceTransformer, top_k: int=3) -> list:
    # Dense вектор
    dense_vector = embed_chunks([query], model)[0]

    # Sparse вектор
    sparse_embedding = list(sparse_encoder.embed([query]))[0]

    sparse_vector = SparseVector(
        indices=sparse_embedding.indices.tolist(),
        values=sparse_embedding.values.tolist()
    )
    result = client.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(query=dense_vector, using="dense", limit=top_k * 2),
            Prefetch(query=sparse_vector, using="sparse", limit=top_k * 2),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k
    )

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

# Pipeline для pdf файлов
def ingest_pdf(pdf_path: str, collection_name: str, client: QdrantClient, model: SentenceTransformer, chunk_size: int=500, overlap=100) -> None:
    text = extract_text_from_pdf(pdf_path)
    chunks = fixed_size_chunks(text, chunk_size, overlap)
    vectors = embed_chunks(chunks, model)
    upsert_chunks(client, collection_name, chunks, vectors)

# Pipeline для текстовых файлов
def ingest_text(text: str, collection_name: str, client: QdrantClient, model: SentenceTransformer, chunk_size: int=500, overlap=100) -> None:
    chunks = fixed_size_chunks(text, chunk_size, overlap)
    vectors = embed_chunks(chunks, model)
    upsert_chunks(client, collection_name, chunks, vectors)

# Pipeline для MD(Markdown) файлов
def ingest_markdown(md_path: str, collection_name: str, client: QdrantClient, model: SentenceTransformer, chunk_size: int=500, overlap: int=100) -> None:
    chunks = extract_chunks_from_markdown(md_path, chunk_size, overlap)
    vectors = embed_chunks(chunks, model)
    upsert_chunks(client, collection_name, chunks, vectors)