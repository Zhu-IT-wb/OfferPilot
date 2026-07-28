import argparse
from pathlib import Path

from app.core.config import settings
from app.repositories.sqlite_interview_knowledge_repository import (
    SQLiteInterviewKnowledgeRepository,
)
from app.services.knowledge_corpus import KnowledgeCorpus


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync local Markdown interview notes into OfferPilot SQLite."
    )
    parser.add_argument(
        "--source",
        default=settings.knowledge_source_path,
        help="Directory containing the private Markdown corpus.",
    )
    parser.add_argument(
        "--database",
        default=settings.sqlite_path,
        help="OfferPilot SQLite database path.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow a verified sync to remove or replace existing questions.",
    )
    args = parser.parse_args()
    source_root = Path(args.source).expanduser().resolve()
    if not source_root.is_dir():
        parser.error(f"Knowledge source directory does not exist: {source_root}")
    repository = SQLiteInterviewKnowledgeRepository(
        args.database,
        initialize_questions=False,
    )
    report = KnowledgeCorpus(repository, source_root).sync_markdown(force=args.force)
    print(
        "Knowledge sync completed: "
        f"files={report.discovered_files} "
        f"questions={report.imported_questions} "
        f"new={report.new_questions} "
        f"updated={report.updated_questions} "
        f"unchanged={report.unchanged_questions} "
        f"removed={report.removed_questions} "
        f"skipped_files={report.skipped_files}"
    )


if __name__ == "__main__":
    main()
