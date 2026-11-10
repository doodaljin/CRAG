import argparse
import sys
from pathlib import Path

from src.config import settings


def main():
    cli_parser = argparse.ArgumentParser(
        description="CRAG Offline Document Ingestion CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python ingest.py --pdf data/docling_technical_report.pdf
  python ingest.py --dir data/
  python ingest.py --input path/to/document.pdf
        """
    )
    cli_parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to a PDF file or directory to ingest"
    )
    cli_parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Path to a specific PDF file"
    )
    cli_parser.add_argument(
        "--dir", "-d",
        type=str,
        default=None,
        help="Path to a directory containing PDF files"
    )
    cli_parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing index instead of overwriting"
    )
    cli_parser.add_argument(
        "--storage-dir",
        type=str,
        default=None,
        help="Custom storage directory (defaults to settings.STORAGE_DIR)"
    )

    args = cli_parser.parse_args()

    # Resolve target input path
    target_path = None
    if args.input:
        target_path = Path(args.input)
    elif args.dir:
        target_path = Path(args.dir)
    elif args.pdf:
        target_path = Path(args.pdf)
    else:
        default_pdf = settings.DATA_DIR / "docling_technical_report.pdf"
        target_path = default_pdf if default_pdf.exists() else settings.DATA_DIR

    if not target_path.exists():
        print(f"[!] Path not found: {target_path}")
        sys.exit(1)

    recreate = not args.append
    try:
        from src.ingester import ingest_documents
        ingest_documents(
            input_path=target_path,
            recreate=recreate,
            storage_dir=args.storage_dir
        )
    except Exception as e:
        print(f"\n[!] Ingestion error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
