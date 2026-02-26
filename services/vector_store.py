import os
import pickle
import re
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np

from config import EMBEDDING_MODEL_NAME

DATA_PATH = "data/plant_knowledge.txt"
EMBEDDING_PATH = "embeddings"
INDEX_PATH = os.path.join(EMBEDDING_PATH, "index.faiss")
DOCS_PATH = os.path.join(EMBEDDING_PATH, "documents.pkl")


@dataclass
class Document:
    page_content: str
    metadata: dict[str, str] | None = None


@dataclass
class SearchResult:
    document: Document
    score: float


class SimpleVectorStore:
    def __init__(
        self, index: faiss.Index, documents: list[Document], embedder: Any
    ):
        self.index = index
        self.documents = documents
        self.embedder = embedder

    def similarity_search(self, query: str, k: int = 3) -> list[Document]:
        return [result.document for result in self.similarity_search_with_score(query, k=k)]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 3,
        crop: str = "Any",
        stage: str = "Any",
        region: str = "Any",
    ) -> list[SearchResult]:
        if not query.strip():
            return []
        query_vector = self.embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

        top_n = min(max(k * 4, 8), len(self.documents))
        scores, indices = self.index.search(query_vector, top_n)
        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc = self.documents[idx]
            if not _metadata_match(doc.metadata or {}, crop=crop, stage=stage, region=region):
                continue
            # IndexFlatIP gives cosine-like similarity in [-1, 1] with normalized vectors.
            confidence = float(max(0.0, min(1.0, (score + 1.0) / 2.0)))
            results.append(SearchResult(document=doc, score=confidence))
            if len(results) >= k:
                break
        return results


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(0, end - overlap)
    return [chunk for chunk in chunks if chunk]


def _get_embedder() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "sentence-transformers is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def _extract_metadata(chunk: str) -> dict[str, str]:
    metadata = {"crop": "Any", "stage": "Any", "region": "Any"}
    patterns = {
        "crop": r"Crop:\s*([^\n]+)",
        "stage": r"Stage:\s*([^\n]+)",
        "region": r"Region:\s*([^\n]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, chunk, flags=re.IGNORECASE)
        if match:
            metadata[key] = match.group(1).strip()
    return metadata


def _metadata_match(
    metadata: dict[str, str], crop: str = "Any", stage: str = "Any", region: str = "Any"
) -> bool:
    checks = [("crop", crop), ("stage", stage), ("region", region)]
    for key, selected in checks:
        if not selected or selected == "Any":
            continue
        value = (metadata.get(key) or "").lower()
        if selected.lower() not in value:
            return False
    return True


def build_vector_store() -> None:
    if not os.path.exists(EMBEDDING_PATH):
        os.makedirs(EMBEDDING_PATH)

    with open(DATA_PATH, "r", encoding="utf-8") as file:
        text = file.read()

    chunks = _chunk_text(text)
    if not chunks:
        raise ValueError(f"No text found in {DATA_PATH}.")

    embedder = _get_embedder()
    vectors = embedder.encode(
        chunks, convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")

    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    documents = [Document(page_content=chunk, metadata=_extract_metadata(chunk)) for chunk in chunks]

    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "wb") as file:
        pickle.dump(documents, file)


def load_vector_store() -> SimpleVectorStore:
    if not os.path.exists(INDEX_PATH) or not os.path.exists(DOCS_PATH):
        raise FileNotFoundError(
            "Vector index not found. Build the knowledge base index first."
        )

    index = faiss.read_index(INDEX_PATH)
    with open(DOCS_PATH, "rb") as file:
        documents = pickle.load(file)

    embedder = _get_embedder()
    return SimpleVectorStore(index=index, documents=documents, embedder=embedder)
