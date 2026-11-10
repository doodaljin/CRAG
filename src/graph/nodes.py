import time
from typing import Any, Callable, Dict, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from src.config import settings
from src.graph.state import GraphState
from src.retriever import HybridRetriever

# Shared Gemini LLM instance
llm = ChatGoogleGenerativeAI(
    model=settings.GEMINI_MODEL,
    google_api_key=settings.GOOGLE_API_KEY,
    temperature=0,
)


# ==========================================
# 1. Structured Output Schemas
# ==========================================

class GradeDocument(BaseModel):
    """Schema for evaluating document relevance against a query."""
    binary_score: str = Field(
        description="Relevance assessment: 'yes' or 'no'",
        enum=["yes", "no"]
    )
    explanation: str = Field(description="Brief reason for the decision")


class GradeHallucination(BaseModel):
    """Schema for evaluating answer faithfulness to context."""
    binary_score: str = Field(
        description="Whether the answer is grounded in context: 'yes' or 'no'",
        enum=["yes", "no"]
    )
    explanation: str = Field(description="Brief reason for the decision")


# ==========================================
# 2. StateGraph Nodes
# ==========================================

def make_retrieve_node(retriever: HybridRetriever) -> Callable[[GraphState], Dict[str, Any]]:
    """Node factory creating the retrieve step bound to a HybridRetriever."""
    def retrieve(state: GraphState) -> Dict[str, Any]:
        query = state["question"]
        print(f"\n--- [NODE: RETRIEVE] Query: '{query}' ---")
        docs = retriever.retrieve(query=query)
        print(f" -> Retrieved {len(docs)} chunks post reranking.")
        return {"documents": docs}
    return retrieve


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """Evaluates retrieved document relevance to filter out noise."""
    print("\n--- [NODE: GRADE DOCUMENTS] ---")
    question = state["question"]
    documents = state.get("documents", [])

    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert document relevance grader assessing retrieved content against a user question.\n"
                   "- Return 'yes' if the document contains keywords, semantics, table data, or relevant clues.\n"
                   "- Return 'no' if the document is irrelevant."),
        ("human", "Retrieved Document:\n\n{document}\n\nQuestion: {question}")
    ])
    grader_chain = grader_prompt | llm.with_structured_output(GradeDocument)

    filtered_docs: List[Dict[str, Any]] = []
    for idx, doc in enumerate(documents, 1):
        try:
            res: GradeDocument = grader_chain.invoke({
                "question": question, 
                "document": doc["text"]
            })
            time.sleep(0.3)  # Brief pause to avoid burst rate limits

            if res.binary_score.lower() == "yes":
                print(f" -> Chunk #{idx} [{doc['metadata'].get('heading_path', 'Root')}]: ACCEPTED")
                filtered_docs.append(doc)
            else:
                print(f" -> Chunk #{idx} [{doc['metadata'].get('heading_path', 'Root')}]: REJECTED ({res.explanation})")
        except Exception as e:
            print(f" [!] Error grading chunk #{idx}: {e}. Retaining chunk by default.")
            filtered_docs.append(doc)

    return {"documents": filtered_docs}


def rewrite_query(state: GraphState) -> Dict[str, Any]:
    """Reformulates query when no relevant context was found."""
    print("\n--- [NODE: REWRITE QUERY] ---")
    current_query = state["question"]
    retry_count = state.get("retry_count", 0) + 1

    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI specialized in optimizing search queries for hybrid dense and sparse retrieval.\n"
                   "Expand semantics, clarify acronyms, and focus on core technical keywords.\n"
                   "Output ONLY the revised query text without quotes or explanations."),
        ("human", "Query to optimize: {question}")
    ])
    chain = rewrite_prompt | llm
    new_query = chain.invoke({"question": current_query}).content.strip()
    print(f" -> Rewritten Query (Attempt #{retry_count}): '{new_query}'")

    return {"question": new_query, "retry_count": retry_count}


def generate(state: GraphState) -> Dict[str, Any]:
    """Synthesizes the answer strictly based on graded context."""
    print("\n--- [NODE: GENERATE ANSWER] ---")
    raw_question = state["raw_question"]
    documents = state.get("documents", [])

    if not documents:
        fallback_msg = "The provided documents do not contain sufficient information to answer your question."
        print(f" -> {fallback_msg}")
        return {"generation": fallback_msg}

    context_blocks = []
    for d in documents:
        src = d["metadata"].get("source", "Document")
        heading = d["metadata"].get("heading_path", "Root")
        context_blocks.append(f"[Source: {src} | Section: {heading}]\n{d['text']}")

    context_str = "\n\n---\n\n".join(context_blocks)

    gen_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional AI assistant. Answer the question strictly based on the provided context.\n"
                   "Rules:\n"
                   "1. Do not hallucinate or extrapolate facts not directly supported by the context.\n"
                   "2. Accurately cite tabular numbers and metrics.\n"
                   "3. Reference source files and section headings when possible."),
        ("human", "Reference Context:\n{context}\n\nQuestion: {question}\n\nDetailed Answer:")
    ])
    chain = gen_prompt | llm
    response = chain.invoke({"context": context_str, "question": raw_question})

    if isinstance(response.content, str):
        answer_text = response.content
    elif isinstance(response.content, list):
        answer_text = "".join(
            block.get("text", "") if isinstance(block, dict) else str(getattr(block, "text", block))
            for block in response.content
        )
    else:
        answer_text = str(response.content)

    return {"generation": answer_text}


def hallucination_grader(state: GraphState) -> Dict[str, Any]:
    """Fact-checks the generated answer against the reference context."""
    print("\n--- [NODE: HALLUCINATION GRADER] ---")
    documents = state.get("documents", [])
    generation = state.get("generation", "")

    if not documents or "do not contain sufficient information" in generation:
        return {"is_grounded": True}

    context_str = "\n\n".join([d["text"] for d in documents])

    hal_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a fact-checking evaluator.\n"
                   "Compare the answer against the provided context:\n"
                   "- Return 'yes' if all facts and numbers in the answer are strictly supported by the context.\n"
                   "- Return 'no' if the answer contains unsupported, contradictory, or fabricated claims."),
        ("human", "Context:\n{context}\n\nAnswer:\n{generation}")
    ])
    hal_chain = hal_prompt | llm.with_structured_output(GradeHallucination)

    try:
        res: GradeHallucination = hal_chain.invoke({
            "context": context_str, 
            "generation": generation
        })
        is_grounded = res.binary_score.lower() == "yes"
        print(f" -> Grounded: {is_grounded} (Reason: {res.explanation})")
    except Exception as e:
        print(f" [!] Error in hallucination check: {e}. Defaulting to grounded.")
        is_grounded = True

    return {"is_grounded": is_grounded}
