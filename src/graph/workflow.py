from langgraph.graph import END, StateGraph

from src.graph.nodes import (
    generate,
    grade_documents,
    hallucination_grader,
    make_retrieve_node,
    rewrite_query,
)
from src.graph.state import GraphState
from src.retriever import HybridRetriever


def route_after_grading(state: GraphState) -> str:
    """Routes execution based on document grading results."""
    docs = state.get("documents", [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if not docs:
        if retry_count < max_retries:
            print(" [ROUTER] No relevant chunks found -> Routing to REWRITE QUERY")
            return "rewrite_query"
        else:
            print(" [ROUTER] Max retries reached -> Routing to GENERATE (fallback)")
            return "generate"

    print(" [ROUTER] Relevant chunks confirmed -> Routing to GENERATE")
    return "generate"


def route_after_hallucination_check(state: GraphState) -> str:
    """Routes execution based on hallucination verification."""
    is_grounded = state.get("is_grounded", True)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if not is_grounded and retry_count < max_retries:
        print(" [ROUTER] Hallucination detected -> Regenerating response")
        return "generate"

    return END


def create_crag_workflow(retriever: HybridRetriever):
    """Initializes and compiles the LangGraph StateGraph for Corrective RAG."""
    workflow = StateGraph(GraphState)

    # 1. Register Nodes
    workflow.add_node("retrieve", make_retrieve_node(retriever))
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("generate", generate)
    workflow.add_node("check_hallucination", hallucination_grader)

    # 2. Entry point and static edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate", "check_hallucination")

    # 3. Conditional edges
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "rewrite_query": "rewrite_query",
            "generate": "generate"
        }
    )

    workflow.add_conditional_edges(
        "check_hallucination",
        route_after_hallucination_check,
        {
            "generate": "generate",
            END: END
        }
    )

    return workflow.compile()
