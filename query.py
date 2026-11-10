import argparse
import sys
from pathlib import Path
from typing import Any, Dict


def execute_query(app: Any, query: str) -> Dict[str, Any]:
    """Executes a query through the CRAG StateGraph."""
    initial_state = {
        "question": query,
        "raw_question": query,
        "documents": [],
        "generation": "",
        "retry_count": 0,
        "max_retries": 2,
        "is_grounded": False
    }

    print("\n[*] Processing via CRAG StateGraph (Retrieve -> Grade -> Generate -> Validate)...")
    result = app.invoke(initial_state)
    return result


def display_result(result: Dict[str, Any]):
    """Displays the final answer and reference sources."""
    print("\n" + "=" * 25 + " ANSWER " + "=" * 25)
    print(result.get("generation", "No response generated."))
    print("=" * 58)

    docs = result.get("documents", [])
    if docs:
        print("\n[*] Reference sources used:")
        for idx, doc in enumerate(docs, 1):
            meta = doc.get("metadata", {})
            src = meta.get("source", "Document")
            heading = meta.get("heading_path", "Root")
            score = doc.get("score")
            score_str = f" | Rerank Score: {score:.4f}" if score is not None else ""
            print(f"   [{idx}] {src} -> {heading}{score_str}")
    print()


def interactive_loop(app: Any):
    """Terminal-based interactive Q&A session."""
    print("-" * 65)
    print("Enter your question (Type 'exit', 'quit', or 'q' to quit):")
    print("-" * 65)

    while True:
        try:
            query = input("\n[You]: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("[*] Exiting session. Goodbye!")
                break

            result = execute_query(app, query)
            display_result(result)

        except KeyboardInterrupt:
            print("\n[*] Session interrupted.")
            break
        except Exception as e:
            print(f"\n[!] Error processing query: {e}")


def main():
    cli_parser = argparse.ArgumentParser(
        description="CRAG Online Question Engine CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  # Interactive mode:
  python query.py

  # Single question:
  python query.py --question "What dataset does Docling evaluate on for layout analysis?"
        """
    )
    cli_parser.add_argument(
        "--question", "-q",
        type=str,
        default=None,
        help="Question to answer (opens interactive CLI if omitted)"
    )
    cli_parser.add_argument(
        "--storage-dir",
        type=str,
        default=None,
        help="Path to database directory (defaults to settings.STORAGE_DIR)"
    )

    args = cli_parser.parse_args()

    from src.config import check_index_exists, settings

    print("=" * 65)
    print("       CRAG ONLINE QUESTION ENGINE        ")
    print("=" * 65)

    # 1. Check if offline index exists
    if not check_index_exists(args.storage_dir):
        print("\n" + "!" * 65)
        print("[!] OFFLINE INDEX NOT FOUND!")
        print("!" * 65)
        print("No indexed data found. Please run offline ingestion first:")
        print("   python ingest.py --pdf data/docling_technical_report.pdf")
        print("or:")
        print("   python ingest.py --dir data/")
        print("!" * 65 + "\n")
        sys.exit(1)

    # 2. Initialize Retriever from stored index
    from src.graph.workflow import create_crag_workflow
    from src.retriever import HybridRetriever
    retriever = HybridRetriever(storage_dir=args.storage_dir, auto_load=True)

    # 3. Compile StateGraph
    print("\n[*] Compiling StateGraph...")
    app = create_crag_workflow(retriever)
    print("[✓] StateGraph ready for online queries!\n")

    # 4. Process single query or launch interactive CLI
    if args.question:
        result = execute_query(app, args.question)
        display_result(result)
    else:
        interactive_loop(app)


if __name__ == "__main__":
    main()
