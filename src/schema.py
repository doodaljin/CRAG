from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class DocumentChunk:
    """Represents an extracted and enriched document text chunk."""
    text: str
    metadata: Dict[str, Any]
