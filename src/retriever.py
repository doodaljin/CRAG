import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
from flashrank import Ranker, RerankRequest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.schema import DocumentChunk


class HybridRetriever:
    """Hybrid Retriever combining Dense (Qdrant), Sparse (BM25), RRF, and FlashRank re-ranking."""

    @classmethod
    def check_index_exists(cls, storage_dir: Optional[Path | str] = None) -> bool:
        """Checks if persistent index files exist on disk without loading models."""
        dir_path = Path(storage_dir) if storage_dir else settings.STORAGE_DIR
        bm25_file = dir_path / "bm25.pkl"
        chunks_file = dir_path / "chunks.pkl"
        qdrant_folder = dir_path / "qdrant"
        return bm25_file.exists() and chunks_file.exists() and qdrant_folder.exists()

    def __init__(self, storage_dir: Optional[Path | str] = None, auto_load: bool = True):
        # 1. Configure persistent storage paths
        self.storage_dir = Path(storage_dir) if storage_dir else settings.STORAGE_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.qdrant_path = self.storage_dir / "qdrant"
        self.bm25_path = self.storage_dir / "bm25.pkl"
        self.chunks_path = self.storage_dir / "chunks.pkl"

        # 2. Local Dense Embedding Model
        print(f"[*] Initializing Embedding model: {settings.EMBEDDING_MODEL}")
        self.encoder = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.encoder.max_seq_length = 512
        self.vector_dim = self.encoder.get_embedding_dimension()

        # 3. Qdrant Client (persistent on-disk or remote/in-memory)
        self.collection_name = settings.QDRANT_COLLECTION
        if settings.QDRANT_LOCATION:
            print(f"[*] Connecting to Qdrant at: {settings.QDRANT_LOCATION}")
            self.qdrant = QdrantClient(location=settings.QDRANT_LOCATION)
        else:
            print(f"[*] Initializing persistent Qdrant at: {self.qdrant_path}")
            self.qdrant = QdrantClient(path=str(self.qdrant_path))

        # 4. FlashRank Cross-Encoder Reranker
        print(f"[*] Initializing FlashRank Reranker: {settings.RERANKER_MODEL}")
        self.ranker = Ranker(model_name=settings.RERANKER_MODEL, cache_dir="./tmp/flashrank")

        # 5. Sparse Storage (BM25) and Chunks Cache
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[DocumentChunk] = []

        # 6. Auto-load index if available
        if auto_load and self.is_indexed():
            self.load_index()

    def _ensure_collection(self, recreate: bool = False):
        """Ensures Qdrant collection exists, recreating it if recreate=True."""
        collections = [c.name for c in self.qdrant.get_collections().collections]
        if self.collection_name in collections:
            if recreate:
                print(f"[*] Recreating collection '{self.collection_name}' in Qdrant...")
                self.qdrant.delete_collection(self.collection_name)
                self.qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
                )
        else:
            print(f"[*] Creating collection '{self.collection_name}' in Qdrant...")
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
            )

    def is_indexed(self) -> bool:
        """Returns True if both Qdrant vectors and BM25 index files exist and are non-empty."""
        if not (self.bm25_path.exists() and self.chunks_path.exists()):
            return False
        try:
            collections = [c.name for c in self.qdrant.get_collections().collections]
            if self.collection_name not in collections:
                return False
            info = self.qdrant.get_collection(self.collection_name)
            return (info.points_count or 0) > 0
        except Exception:
            return False

    def index_documents(self, chunks: List[DocumentChunk], recreate: bool = True):
        """Indexes DocumentChunks into Qdrant and BM25, persisting them to disk."""
        if not chunks:
            print("[!] Warning: No chunks provided for indexing.")
            return

        self.chunks = chunks
        print(f"[*] Indexing {len(chunks)} chunks into Qdrant & BM25...")

        # 1. Prepare Qdrant collection
        self._ensure_collection(recreate=recreate)

        # 2. Build BM25 index
        tokenized_corpus = [c.text.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 3. Compute Dense Embeddings & upsert to Qdrant
        texts = [c.text for c in chunks]
        embeddings = self.encoder.encode(texts, show_progress_bar=False)

        points = []
        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            points.append(
                PointStruct(
                    id=idx,
                    vector=vector.tolist(),
                    payload={"text": chunk.text, **chunk.metadata}
                )
            )

        self.qdrant.upsert(collection_name=self.collection_name, points=points)

        # 4. Persist BM25 and chunks to disk
        print(f"[*] Saving BM25 and chunks index to: {self.storage_dir}...")
        with open(self.chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
        with open(self.bm25_path, "wb") as f:
            pickle.dump(self.bm25, f)

        print(f"[✓] Successfully indexed and persisted {len(chunks)} chunks.")

    def load_index(self):
        """Loads persisted BM25 and document chunks from disk."""
        if not self.chunks_path.exists() or not self.bm25_path.exists():
            raise FileNotFoundError(
                f"[!] Offline index not found at '{self.storage_dir}'. "
                "Please run offline ingestion first (e.g., python ingest.py)."
            )

        print(f"[*] Loading offline index from: {self.storage_dir}...")
        with open(self.chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        with open(self.bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)

        collections = [c.name for c in self.qdrant.get_collections().collections]
        if self.collection_name not in collections:
            raise RuntimeError(
                f"[!] Collection '{self.collection_name}' does not exist in Qdrant. "
                "Please run offline ingestion first."
            )

        print(f"[✓] Successfully loaded offline index: {len(self.chunks)} chunks ready for querying.")

    def _dense_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        query_vector = self.encoder.encode(query).tolist()
        response = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k
        )
        return [
            {
                "id": hit.id, 
                "text": hit.payload["text"], 
                "metadata": hit.payload
            } 
            for hit in response.points
        ]

    def _sparse_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if not self.bm25:
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "id": idx,
                "text": self.chunks[idx].text,
                "metadata": self.chunks[idx].metadata
            })
        return results

    def _reciprocal_rank_fusion(
        self, 
        dense_results: List[Dict[str, Any]], 
        sparse_results: List[Dict[str, Any]], 
        k: int = 60
    ) -> List[Dict[str, Any]]:
        rrf_scores: Dict[int, float] = {}
        doc_map: Dict[int, Dict[str, Any]] = {}

        for rank, doc in enumerate(dense_results, 1):
            doc_id = doc["id"]
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

        for rank, doc in enumerate(sparse_results, 1):
            doc_id = doc["id"]
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return [doc_map[doc_id] for doc_id in sorted_ids]

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        if not self.chunks or not self.bm25:
            if self.is_indexed():
                self.load_index()
            else:
                raise ValueError("No indexed documents found. Please run offline ingestion first.")

        # 1. Parallel Dense & Sparse search
        dense_docs = self._dense_search(query, top_k=settings.TOP_K_RETRIEVAL)
        sparse_docs = self._sparse_search(query, top_k=settings.TOP_K_RETRIEVAL)

        # 2. Reciprocal Rank Fusion
        fused_docs = self._reciprocal_rank_fusion(dense_docs, sparse_docs)
        candidates = fused_docs[:settings.TOP_K_RETRIEVAL]

        # 3. FlashRank Cross-Encoder Re-ranking
        passages = [{"id": d["id"], "text": d["text"]} for d in candidates]
        rerank_req = RerankRequest(query=query, passages=passages)
        ranked_passages = self.ranker.rerank(rerank_req)

        top_k_reranked = []
        for item in ranked_passages[:settings.TOP_K_RERANK]:
            original_doc = next(d for d in candidates if d["id"] == item["id"])
            top_k_reranked.append({
                "text": item["text"],
                "score": item["score"],
                "metadata": original_doc["metadata"]
            })

        return top_k_reranked

    def close(self):
        """Closes Qdrant connection and releases resource locks."""
        if hasattr(self.qdrant, "close"):
            try:
                self.qdrant.close()
            except Exception:
                pass
