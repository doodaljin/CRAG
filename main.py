import argparse
import sys
from pathlib import Path

from src.config import settings


def run_ingest(args):
    from src.ingester import ingest_documents
    target = Path(args.input or args.pdf or args.dir or (settings.DATA_DIR / "docling_technical_report.pdf"))
    if not target.exists():
        print(f"[!] Path not found: {target}")
        sys.exit(1)
    ingest_documents(
        input_path=target,
        recreate=not args.append,
        storage_dir=args.storage_dir
    )


def run_query(args):
    import query
    sys.argv = [sys.argv[0]]
    if args.question:
        sys.argv.extend(["--question", args.question])
    if args.storage_dir:
        sys.argv.extend(["--storage-dir", args.storage_dir])
    query.main()


def main():
    parser = argparse.ArgumentParser(
        description="Corrective RAG (CRAG) Engine - Unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The system is divided into two decoupled stages:
  1. Offline Ingestion (parse documents, build Qdrant vector & BM25 indices):
     python ingest.py --pdf data/docling_technical_report.pdf
     or: python main.py ingest --pdf data/docling_technical_report.pdf

  2. Online Question (retrieve, grade, rewrite, generate, validate):
     python query.py
     or: python main.py query --question "What is TableBench?"
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run: 'ingest' or 'query'")

    # Subcommand: Ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents offline into database")
    ingest_parser.add_argument("--input", "-i", type=str, default=None, help="File or directory path")
    ingest_parser.add_argument("--pdf", type=str, default=None, help="PDF file path")
    ingest_parser.add_argument("--dir", "-d", type=str, default=None, help="Directory path")
    ingest_parser.add_argument("--append", action="store_true", help="Append to index without overwriting")
    ingest_parser.add_argument("--storage-dir", type=str, default=None, help="Custom storage directory")

    # Subcommand: Query
    query_parser = subparsers.add_parser("query", help="Run online question interface")
    query_parser.add_argument("--question", "-q", type=str, default=None, help="Specific question to answer")
    query_parser.add_argument("--storage-dir", type=str, default=None, help="Custom storage directory")

    # Root arguments (backward compatibility)
    parser.add_argument("--pdf", type=str, default=None, help="Ingest PDF then launch query (legacy mode)")
    parser.add_argument("--question", "-q", type=str, default=None, help="Direct question to answer")

    args = parser.parse_args()

    if args.command == "ingest":
        run_ingest(args)
    elif args.command == "query":
        run_query(args)
    elif args.pdf:
        # Legacy mode: ingest PDF then start query
        from src.ingester import ingest_documents
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            print(f"[!] File not found: {pdf_path}")
            sys.exit(1)
        ingest_documents(input_path=pdf_path, recreate=True)
        import query
        query.main()
    else:
        # Default behavior: run query if index exists, else print guidance
        from src.config import check_index_exists
        if check_index_exists():
            import query
            query.main()
        else:
            print("=" * 65)
            print("         CORRECTIVE RAG (CRAG) ENGINE           ")
            print("=" * 65)
            print("[!] No indexed data detected.")
            print("\nThe system is decoupled into two independent stages:")
            print(" 1. Offline Document Ingestion:")
            print("    > python ingest.py --pdf data/docling_technical_report.pdf")
            print("    > python ingest.py --dir data/")
            print("\n 2. Online Question Answering:")
            print("    > python query.py")
            print("    > python query.py --question \"Your question here\"")
            print("=" * 65)


if __name__ == "__main__":
    main()
