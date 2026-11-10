import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import settings
from src.parser import DocumentParser
from src.retriever import HybridRetriever
from src.schema import DocumentChunk


def ingest_documents(
    input_path: Path | str,
    recreate: bool = True,
    storage_dir: Optional[Path | str] = None
) -> Dict[str, Any]:
    """Offline document ingestion pipeline:
    1. Extracts text and markdown tables via Docling.
    2. Enriches chunks with hierarchical breadcrumbs.
    3. Indexes dense vectors into persistent Qdrant on disk.
    4. Serializes BM25 model and chunks metadata to disk.
    """
    target_path = Path(input_path)
    if not target_path.exists():
        raise FileNotFoundError(f"[!] Path not found: {target_path}")

    start_time = time.time()
    print("=" * 65)
    print("       OFFLINE DOCUMENT INGESTION (CRAG PIPELINE)        ")
    print("=" * 65)
    print(f"[*] Source: {target_path.resolve()}")
    print(f"[*] Storage: {storage_dir or settings.STORAGE_DIR}")
    print(f"[*] Mode: {'Overwrite' if recreate else 'Append'}")
    print("-" * 65)

    # 1. Document Parsing (Docling)
    parser = DocumentParser()
    chunks: List[DocumentChunk] = parser.parse_path(target_path)

    if not chunks:
        print("[!] No chunks extracted from source.")
        return {"status": "empty", "num_chunks": 0, "elapsed_seconds": time.time() - start_time}

    num_tables = sum(1 for c in chunks if c.metadata.get("has_table", False))
    print(f"[✓] Created {len(chunks)} chunks ({num_tables} containing markdown tables).")

    # 2. Indexing & Persistence (Qdrant + BM25)
    retriever = HybridRetriever(storage_dir=storage_dir, auto_load=False)
    try:
        retriever.index_documents(chunks, recreate=recreate)
    finally:
        retriever.close()

    elapsed = time.time() - start_time
    print("-" * 65)
    print("   OFFLINE INGESTION COMPLETE   ")
    print(f"   • Total chunks: {len(chunks)}")
    print(f"   • Table chunks: {num_tables}")
    print(f"   • Elapsed time: {elapsed:.2f}s")
    print(f"   • Storage path: {storage_dir or settings.STORAGE_DIR}")
    print("=" * 65)

    return {
        "status": "success",
        "num_chunks": len(chunks),
        "num_tables": num_tables,
        "elapsed_seconds": elapsed,
        "storage_dir": str(storage_dir or settings.STORAGE_DIR),
    }
