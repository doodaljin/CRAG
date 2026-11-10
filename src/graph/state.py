from typing import Any, Dict, List, TypedDict


class GraphState(TypedDict):
    """Represents the state of the CRAG execution workflow."""
    question: str                   # Current query (may be reformulated by rewrite node)
    raw_question: str               # Original user question
    documents: List[Dict[str, Any]] # Candidate documents after grading
    generation: str                 # LLM-generated answer
    retry_count: int                # Number of retries executed
    max_retries: int                # Maximum allowed retries (default: 2)
    is_grounded: bool               # Fact-checking hallucination verification flag
