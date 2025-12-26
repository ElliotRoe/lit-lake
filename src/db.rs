use rusqlite::{Connection, Result};
use std::time::Duration;

pub fn init_db(conn: &Connection) -> Result<()> {
    // Concurrency-friendly settings (worker + sync operations).
    let _mode: String = conn.query_row("PRAGMA journal_mode=WAL;", [], |row| row.get(0))?;
    conn.busy_timeout(Duration::from_millis(5000))?;
    conn.pragma_update(None, "foreign_keys", &1i64)?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS reference_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_key TEXT UNIQUE,
            title TEXT,
            authors TEXT,
            year TEXT,
            source_system TEXT NOT NULL,
            source_id TEXT NOT NULL,
            metadata TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source_system, source_id)
        )",
        [],
    )?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS reference_external_ids (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reference_id INTEGER NOT NULL,
            scheme TEXT NOT NULL,
            value TEXT NOT NULL,
            UNIQUE(reference_id, scheme, value),
            FOREIGN KEY(reference_id) REFERENCES reference_items(id) ON DELETE CASCADE
        )",
        [],
    )?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reference_id INTEGER,
            document_file_id INTEGER,
            kind TEXT NOT NULL,
            content TEXT,
            chunk_index INTEGER,
            embedding_status TEXT DEFAULT 'pending',
            embedding_updated_at DATETIME,
            embedding_error TEXT,
            source_system TEXT,
            source_id TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(reference_id) REFERENCES reference_items(id) ON DELETE CASCADE,
            FOREIGN KEY(document_file_id) REFERENCES document_files(id) ON DELETE CASCADE
        )",
        [],
    )?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS document_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            file_path TEXT NOT NULL,
            mime_type TEXT,
            label TEXT,
            extracted_text TEXT,
            extraction_status TEXT DEFAULT 'pending',
            extraction_error TEXT,
            source_system TEXT,
            source_id TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(document_id, file_path),
            FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
        )",
        [],
    )?;

    // Helpful indexes
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_reference_id ON documents(reference_id)",
        [],
    )?;
    conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_kind ON documents(kind)", [])?;
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_embedding_status ON documents(embedding_status)",
        [],
    )?;
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_reference_external_ids_scheme_value ON reference_external_ids(scheme, value)",
        [],
    )?;
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_document_files_extraction_status ON document_files(extraction_status)",
        [],
    )?;
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_document_file_id ON documents(document_file_id)",
        [],
    )?;

    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS vec_documents USING vec0(embedding float[384]);",
        [],
    )?;

    Ok(())
}
