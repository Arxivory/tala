
"""
TALA - Knowledge Base Builder
=============================

Builds the persistent vector knowledge base from the source documents
configured in configs/rag.yaml.

Run from the project root:

    python scripts/build_knowledge_base.py
"""

import logging

from tala.rag.ingestion import ingest_knowledge_base


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    report = ingest_knowledge_base()

    print("\nKnowledge Base Ingestion Report")
    print("=" * 40)
    print(report.to_dict())


if __name__ == "__main__":
    main()
