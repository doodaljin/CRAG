from pathlib import Path
from typing import Any, Dict, List

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from transformers import AutoTokenizer

from src.config import settings
from src.schema import DocumentChunk

__all__ = ["DocumentChunk", "DocumentParser"]


class DocumentParser:
    """Parses documents with Docling, preserving layout and markdown table structures."""

    def __init__(self, tokenizer_name: str = settings.EMBEDDING_MODEL, max_tokens: int = settings.MAX_CHUNK_TOKENS):
        self.converter = DocumentConverter()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=max_tokens,
            merge_peers=True
        )

    def parse_pdf(self, file_path: Path | str) -> List[DocumentChunk]:
        """Parses a single PDF and applies contextual breadcrumb enrichment."""
        target_path = Path(file_path)
        if not target_path.exists():
            raise FileNotFoundError(f"File not found: {target_path}")

        print(f"[*] Parsing document with Docling: {target_path.name}")
        doc = self.converter.convert(target_path).document

        raw_chunks = list(self.chunker.chunk(doc))
        parsed_chunks: List[DocumentChunk] = []

        for idx, chunk in enumerate(raw_chunks):
            headings = [
                h if isinstance(h, str) else getattr(h, "text", str(h)) 
                for h in chunk.meta.headings
            ] if chunk.meta.headings else []

            breadcrumb = " > ".join(headings) if headings else "Root"

            # Prepend hierarchical breadcrumbs to avoid losing context in isolated retrieval
            enriched_text = f"Context: {breadcrumb}\n\n{chunk.text}" if breadcrumb != "Root" else chunk.text

            metadata = {
                "source": target_path.name,
                "chunk_id": idx,
                "heading_path": breadcrumb,
                "has_table": "\n|" in chunk.text
            }
            parsed_chunks.append(DocumentChunk(text=enriched_text, metadata=metadata))

        print(f"[✓] Created {len(parsed_chunks)} chunks from {target_path.name}")
        return parsed_chunks

    def parse_directory(self, dir_path: Path | str, pattern: str = "*.pdf") -> List[DocumentChunk]:
        """Parses all matching files in a directory."""
        target_dir = Path(dir_path)
        if not target_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {target_dir}")

        files = sorted(target_dir.glob(pattern))
        if not files:
            print(f"[!] No {pattern} files found in {target_dir}")
            return []

        all_chunks: List[DocumentChunk] = []
        print(f"[*] Found {len(files)} files to process in {target_dir.name}")
        for file in files:
            chunks = self.parse_pdf(file)
            all_chunks.extend(chunks)

        print(f"[✓] Finished directory parsing: {len(all_chunks)} chunks from {len(files)} files.")
        return all_chunks

    def parse_path(self, path: Path | str) -> List[DocumentChunk]:
        """Automatically parses either a single file or an entire directory."""
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Path not found: {target}")
        if target.is_dir():
            return self.parse_directory(target)
        return self.parse_pdf(target)
