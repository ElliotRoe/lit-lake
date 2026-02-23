from __future__ import annotations


def get_documentation_text(section: str | None) -> str:
    if section == "workflows":
        return (
            "# Supported Workflows\n\n"
            "## Workflow 1: Semantic Search\n"
            "Use `sql_search` with `embed()` + `rerank_score()` over `vec_documents` and `documents`.\n\n"
            "## Workflow 2: Citation Assistance\n"
            "Break claims into search statements, retrieve with vector match, rerank, then inspect full text chunks. "
            "For citations, use `json_extract(documents.metadata_json, '$.loc.page_start')` and "
            "`json_extract(documents.metadata_json, '$.loc.page_end')` when available.\n\n"
            "## Workflow 3: Full Text Verification\n"
            "Use `document_files.extracted_text` and `documents.kind='fulltext_chunk'` for precise evidence checks.\n"
            "Zotero notes and annotations are also searchable via `documents.kind IN ('note','annotation')`.\n"
        )

    if section == "schema":
        return (
            "# Schema Overview\n\n"
            "Core tables: `reference_items`, `document_files`, `documents`, `vec_documents`.\n"
            "Chunk- and source-specific metadata is stored in `documents.metadata_json`.\n"
            "File-level extraction metadata is stored in `document_files.metadata_json`.\n"
            "Queue tables: `jobs`, `job_attempts`, `worker_runs`.\n"
            "Semantic search uses `vec_documents` with `MATCH embed('query') AND k = N`.\n"
        )

    if section == "examples":
        return (
            "# SQL Examples\n\n"
            "```sql\n"
            "SELECT r.title, d.kind, d.content\n"
            "FROM vec_documents v\n"
            "JOIN documents d ON v.rowid = d.id\n"
            "JOIN reference_items r ON d.reference_id = r.id\n"
            "WHERE v.embedding MATCH embed('your topic')\n"
            "  AND k = 30\n"
            "ORDER BY rerank_score('your topic', d.content) DESC\n"
            "LIMIT 10;\n"
            "```\n"
        )

    return (
        "Lit Lake documentation sections: `workflows`, `schema`, `examples`.\n"
        "Start with `workflows` before advanced SQL queries."
    )


__all__ = ["get_documentation_text"]
