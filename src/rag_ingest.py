"""CLI to ingest the curated knowledge base into Qdrant."""
from __future__ import annotations

import argparse
import json
import sys

from src.rag import RAGError, index_knowledge_base


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Index curated logistics documentation for the RAG assistant."
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Do not delete the existing dedicated collection before indexing.",
    )
    args = parser.parse_args()
    try:
        result = index_knowledge_base(force_recreate=not args.append)
    except (RAGError, OSError, ValueError) as exc:
        print(f"RAG indexing failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
