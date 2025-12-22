mod db;
mod embeddings;
mod preview;
mod zotero;

use anyhow::Result;
use base64::{engine::general_purpose, Engine as _};
use fastembed::{EmbeddingModel, InitOptions, RerankInitOptions, RerankerModel, TextEmbedding, TextRerank};
use rusqlite::functions::FunctionFlags;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::sync::mpsc;
use std::thread;

use crate::db::init_db;
use crate::embeddings::{EmbeddingWorker, WorkerConfig, WorkerSignal};
use crate::zotero::ZoteroReader;

/// Paths for the LitLake application data directory.
struct LitLakePaths {
    /// Root directory (e.g., ~/LitLake)
    #[allow(dead_code)]
    root: PathBuf,
    /// Path to the SQLite database
    db: PathBuf,
    /// Path to the models cache directory
    models: PathBuf,
}

impl LitLakePaths {
    /// Initialize LitLake paths, creating directories as needed.
    /// Priority: LIT_LAKE_DIR env var, otherwise ~/LitLake
    fn init() -> Result<Self> {
        let root = if let Ok(dir) = env::var("LIT_LAKE_DIR") {
            PathBuf::from(dir)
        } else if let Some(home) = dirs::home_dir() {
            home.join("LitLake")
        } else {
            // Fallback to current directory
            PathBuf::from(".")
        };

        // Create root directory if it doesn't exist
        if !root.exists() {
            fs::create_dir_all(&root)?;
            eprintln!("[main] Created LitLake directory at {:?}", root);
        }

        let db = root.join("lit_lake.db");
        let models = root.join("models");

        // Create models directory if it doesn't exist
        if !models.exists() {
            fs::create_dir_all(&models)?;
            eprintln!("[main] Created models directory at {:?}", models);
        }

        Ok(Self { root, db, models })
    }
}

fn text_tool_result(text: String) -> Value {
    json!({ "content": [{ "type": "text", "text": text }] })
}

fn tool_preview_document_pdf_pages(state: Arc<AppState>, args: Value) -> Result<Value> {
    let document_file_id = args
        .get("document_file_id")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| anyhow::anyhow!("Missing required argument: document_file_id (integer)"))?;

    let size: u32 = args
        .get("size")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
        .unwrap_or(1024)
        .clamp(64, 4096);

    let start_page: u32 = args
        .get("start_page")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
        .unwrap_or(1);
    let end_page: u32 = args
        .get("end_page")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
        .unwrap_or(start_page);

    if start_page == 0 || end_page == 0 {
        return Err(anyhow::anyhow!(
            "Pages are 1-based; start_page/end_page must be >= 1"
        ));
    }
    if start_page > end_page {
        return Err(anyhow::anyhow!("start_page must be <= end_page"));
    }
    let page_count_requested = (end_page - start_page + 1) as usize;
    if page_count_requested > 10 {
        return Err(anyhow::anyhow!(
            "Requested {} pages ({}..={}) which is too many for a single tool response; max is 10.",
            page_count_requested,
            start_page,
            end_page
        ));
    }

    let (file_path, mime_type): (String, Option<String>) = {
        let conn = state.conn.lock().unwrap();
        conn.query_row(
            "SELECT file_path, mime_type FROM document_files WHERE id = ?",
            [document_file_id],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?)),
        )
        .map_err(|_| anyhow::anyhow!("document_files row not found for id={}", document_file_id))?
    };

    if let Some(mt) = &mime_type {
        if mt != "application/pdf" {
            return Err(anyhow::anyhow!(
                "Only application/pdf is supported for PDF page previews (got {})",
                mt
            ));
        }
    }

    let batch = crate::preview::PdfPreviewRenderer::page_range_png(
        &file_path,
        start_page,
        end_page,
        size,
    )?;

    let mut content: Vec<Value> = Vec::new();
    for p in batch.pages {
        content.push(json!({
            "type": "text",
            "text": format!("Page {}", p.page_number)
        }));
        content.push(json!({
            "type": "image",
            "data": general_purpose::STANDARD.encode(p.png_bytes),
            "mimeType": "image/png",
            "annotations": {
                "audience": ["user", "assistant"],
                "priority": 1.0
            }
        }));
    }

    Ok(json!({ "content": content }))
}

#[derive(Deserialize, Debug)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    method: String,
    params: Option<Value>,
    id: Option<Value>,
}

#[derive(Serialize, Debug)]
struct JsonRpcResponse {
    jsonrpc: String,
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<Value>,
}

struct AppState {
    conn: Arc<Mutex<Connection>>,
    worker_tx: mpsc::Sender<WorkerSignal>,
}

/// Register `embed(text)` and `rerank_score(query, doc)` scalar functions on the connection.
fn register_ai_functions(
    conn: &Connection,
    embedder: Arc<Mutex<TextEmbedding>>,
    reranker: Arc<Mutex<TextRerank>>,
) -> Result<()> {
    // embed(text) -> BLOB (384 floats as little-endian bytes)
    let emb = embedder.clone();
    conn.create_scalar_function(
        "embed",
        1,
        FunctionFlags::SQLITE_UTF8 | FunctionFlags::SQLITE_DETERMINISTIC,
        move |ctx| {
            let text: String = ctx.get(0)?;
            let mut model = emb.lock().unwrap();
            let embeddings = model
                .embed(vec![text], None)
                .map_err(|e| rusqlite::Error::UserFunctionError(e.into()))?;
            let vec = &embeddings[0];
            let bytes: Vec<u8> = vec.iter().flat_map(|f| f.to_le_bytes()).collect();
            Ok(bytes)
        },
    )?;

    // rerank_score(query, document) -> REAL
    let rr = reranker.clone();
    conn.create_scalar_function(
        "rerank_score",
        2,
        FunctionFlags::SQLITE_UTF8,
        move |ctx| {
            let query: String = ctx.get(0)?;
            let doc: String = ctx.get(1)?;
            let mut model = rr.lock().unwrap();
            let results = model
                .rerank(query, vec![doc], false, None)
                .map_err(|e| rusqlite::Error::UserFunctionError(e.into()))?;
            let score = results.first().map(|r| r.score as f64).unwrap_or(0.0);
            Ok(score)
        },
    )?;

    Ok(())
}

fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    // Initialize LitLake paths (creates directories as needed)
    let paths = LitLakePaths::init()?;
    let db_path = paths.db.to_string_lossy().to_string();
    let models_path = paths.models.clone();

    eprintln!("[main] Using database: {}", db_path);
    eprintln!("[main] Using models directory: {:?}", models_path);

    // Register sqlite-vec extension once per process.
    unsafe {
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
            sqlite_vec::sqlite3_vec_init as *const (),
        )));
    }

    // Spawn embedding worker (background doc indexing).
    let (worker_tx, worker_rx) = mpsc::channel::<WorkerSignal>();
    {
        let cfg = WorkerConfig {
            db_path: db_path.clone(),
            models_path: models_path.clone(),
            ..Default::default()
        };
        thread::spawn(move || {
            let worker = EmbeddingWorker::new(cfg, worker_rx);
            if let Err(e) = worker.run() {
                eprintln!("[embeddings] Worker exited with error: {e:?}");
            }
        });
    }

    // Load AI models eagerly (like original script).
    eprintln!("[main] Loading embedding model...");
    let embedder = Arc::new(Mutex::new(TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::BGESmallENV15)
            .with_cache_dir(models_path.clone())
            .with_show_download_progress(true),
    )?));

    eprintln!("[main] Loading reranker model...");
    let reranker = Arc::new(Mutex::new(TextRerank::try_new(
        RerankInitOptions::new(RerankerModel::BGERerankerBase)
            .with_cache_dir(models_path)
            .with_show_download_progress(true),
    )?));

    // Open main connection and register scalar functions.
    let conn = Connection::open(&db_path)?;
    init_db(&conn)?;
    register_ai_functions(&conn, embedder.clone(), reranker.clone())?;

    let app_state = Arc::new(AppState {
        conn: Arc::new(Mutex::new(conn)),
        worker_tx,
    });

    // Kick the worker once on startup to process any backlog.
    let _ = app_state.worker_tx.send(WorkerSignal::Wake);

    eprintln!("[main] Server ready. Waiting for messages...");

    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    while let Some(line) = lines.next() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let req: JsonRpcRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Failed to parse JSON: {}", e);
                continue;
            }
        };

        eprintln!("Received: {} ({:?})", req.method, req.id);

        let response = match req.method.as_str() {
            "initialize" => handle_initialize(&req),
            "notifications/initialized" => None,
            "tools/list" => handle_list_tools(&req),
            "tools/call" => handle_call_tool(&req, app_state.clone()),
            _ => handle_unknown(&req),
        };

        if let Some(resp) = response {
            let json_str = serde_json::to_string(&resp)?;
            let mut stdout = io::stdout();
            stdout.write_all(json_str.as_bytes())?;
            stdout.write_all(b"\n")?;
            stdout.flush()?;
        }
    }

    Ok(())
}

fn handle_initialize(req: &JsonRpcRequest) -> Option<JsonRpcResponse> {
    Some(JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        id: req.id.clone(),
        error: None,
        result: Some(json!({
            "protocolVersion": "2025-06-18",
            "capabilities": { "tools": {} },
            "serverInfo": { "name": "lit-lake-mcp", "version": "0.4.0" }
        })),
    })
}

fn handle_list_tools(req: &JsonRpcRequest) -> Option<JsonRpcResponse> {
    Some(JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        id: req.id.clone(),
        error: None,
        result: Some(json!({
            "tools": [
                {
                    "name": "sync_zotero",
                    "description": "Import/update references from Zotero into the local library. Creates title and abstract documents for each reference, which are then embedded asynchronously for semantic search. Call this first if the library seems empty or out of date.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                },
                {
                    "name": "sql_search",
                    "description": r#"Query the research library with full SQL flexibility. Supports semantic search via embed() and rerank_score() scalar functions.

**First time?** Call get_documentation with section='workflows' to understand typical workflows for this tool before writing queries.

## Quick Reference

embed(text) → BLOB: Compute embedding for semantic similarity.
rerank_score(query, doc) → REAL: Cross-encoder relevance score.

```sql
SELECT r.title, r.authors, r.year, d.content,
       rerank_score('your question', d.content) AS relevance
FROM vec_documents v
JOIN documents d ON v.rowid = d.id
JOIN reference_items r ON d.reference_id = r.id
WHERE v.embedding MATCH embed('your question')
  AND k = 30 AND d.embedding_status = 'ready'
ORDER BY relevance DESC LIMIT 10;
```

Start broad, then refine. Don't over-specify on first attempt."#,
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": { "type": "string", "description": "A read-only SQL query. Use embed() for vector search and rerank_score() for relevance ordering." }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "get_documentation",
                    "description": "Get detailed documentation including supported workflows, SQL examples, and database schema. Call with section='workflows' for step-by-step guidance on citation assistance and library research. Sections: 'workflows', 'schema', 'examples', or omit for overview.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "section": {
                                "type": "string",
                                "description": "Section to retrieve: 'workflows' (recommended first read), 'schema', 'examples', or omit for overview."
                            }
                        }
                    }
                },
                {
                    "name": "library_status",
                    "description": "Get an overview of the library: how many references exist, what document types are available per reference, and embedding status. Use this to understand library scope before crafting queries, or to check if embeddings are ready after a sync.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                },
                {
                    "name": "preview_document_pdf_pages",
                    "description": "Render PNG images of PDF pages to visually inspect full-text content. Use this to validate claims, check methodology, or verify specific details that aren't in the abstract. Requires document_file_id from the document_files table.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "document_file_id": { "type": "integer", "description": "The document_files.id for the PDF to preview." },
                            "size": { "type": "integer", "description": "Max dimension in pixels (64-4096). Default 1024." },
                            "start_page": { "type": "integer", "description": "First page to render (1-based). Default 1." },
                            "end_page": { "type": "integer", "description": "Last page to render (1-based). Default = start_page. Max 10 pages per call." }
                        },
                        "required": ["document_file_id"]
                    }
                }
            ]
        })),
    })
}

fn handle_call_tool(req: &JsonRpcRequest, state: Arc<AppState>) -> Option<JsonRpcResponse> {
    let params = req.params.as_ref()?;
    let name = params.get("name").and_then(|n| n.as_str())?;
    let args = params.get("arguments").cloned().unwrap_or(json!({}));

    let result: Result<Value> = match name {
        "sync_zotero" => tool_sync_zotero(state).map(text_tool_result),
        "sql_search" => tool_sql_search(state, args).map(text_tool_result),
        "get_documentation" => tool_get_documentation(args).map(text_tool_result),
        "library_status" => tool_library_status(state).map(text_tool_result),
        "preview_document_pdf_pages" => tool_preview_document_pdf_pages(state, args),
        _ => Err(anyhow::anyhow!("Tool not found")),
    };

    match result {
        Ok(content) => Some(JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: req.id.clone(),
            error: None,
            result: Some(content),
        }),
        Err(e) => Some(JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: req.id.clone(),
            error: None,
            result: Some(json!({
                "isError": true,
                "content": [{ "type": "text", "text": format!("Error: {}", e) }]
            })),
        }),
    }
}

fn tool_library_status(state: Arc<AppState>) -> Result<String> {
    let conn = state.conn.lock().unwrap();

    let reference_count: i64 =
        conn.query_row("SELECT COUNT(*) FROM reference_items", [], |row| row.get(0))?;
    let document_count: i64 =
        conn.query_row("SELECT COUNT(*) FROM documents", [], |row| row.get(0))?;

    let mut stmt = conn.prepare(
        "SELECT
            r.id,
            r.title,
            r.authors,
            r.year,
            r.source_system,
            r.source_id,
            d.id,
            d.kind,
            d.embedding_status,
            d.updated_at
         FROM reference_items r
         LEFT JOIN documents d ON d.reference_id = r.id
         ORDER BY r.updated_at DESC, r.id DESC, d.kind ASC, d.id ASC",
    )?;

    #[derive(Debug)]
    struct Row {
        reference_id: i64,
        title: Option<String>,
        authors: Option<String>,
        year: Option<String>,
        source_system: String,
        source_id: String,
        document_id: Option<i64>,
        kind: Option<String>,
        embedding_status: Option<String>,
        updated_at: Option<String>,
    }

    let rows = stmt.query_map([], |row| {
        Ok(Row {
            reference_id: row.get(0)?,
            title: row.get(1)?,
            authors: row.get(2)?,
            year: row.get(3)?,
            source_system: row.get(4)?,
            source_id: row.get(5)?,
            document_id: row.get(6)?,
            kind: row.get(7)?,
            embedding_status: row.get(8)?,
            updated_at: row.get(9)?,
        })
    })?;

    let mut refs: Vec<Value> = Vec::new();
    let mut current_ref_id: Option<i64> = None;
    let mut current_ref: Option<serde_json::Map<String, Value>> = None;
    let mut current_docs: Vec<Value> = Vec::new();

    for r in rows {
        let r = r?;
        if current_ref_id != Some(r.reference_id) {
            if let Some(mut cr) = current_ref.take() {
                cr.insert("documents".to_string(), json!(current_docs));
                refs.push(Value::Object(cr));
                current_docs = Vec::new();
            }
            current_ref_id = Some(r.reference_id);
            let mut m = serde_json::Map::new();
            m.insert("reference_id".to_string(), json!(r.reference_id));
            m.insert("title".to_string(), json!(r.title));
            m.insert("authors".to_string(), json!(r.authors));
            m.insert("year".to_string(), json!(r.year));
            m.insert("source_system".to_string(), json!(r.source_system));
            m.insert("source_id".to_string(), json!(r.source_id));
            current_ref = Some(m);
        }
        if let Some(doc_id) = r.document_id {
            current_docs.push(json!({
                "document_id": doc_id,
                "kind": r.kind,
                "embedding_status": r.embedding_status,
                "updated_at": r.updated_at,
            }));
        }
    }
    if let Some(mut cr) = current_ref.take() {
        cr.insert("documents".to_string(), json!(current_docs));
        refs.push(Value::Object(cr));
    }

    // Embedding counts
    let mut count_stmt = conn.prepare(
        "SELECT kind, embedding_status, COUNT(*) as count
         FROM documents
         GROUP BY kind, embedding_status
         ORDER BY kind, embedding_status",
    )?;
    let count_rows = count_stmt.query_map([], |row| {
        Ok(json!({
            "kind": row.get::<_, Option<String>>(0)?,
            "embedding_status": row.get::<_, Option<String>>(1)?,
            "count": row.get::<_, i64>(2)?
        }))
    })?;
    let embedding_counts: Vec<Value> = count_rows.filter_map(|r| r.ok()).collect();

    // Recent errors
    let mut err_stmt = conn.prepare(
        "SELECT id, kind, embedding_error, embedding_updated_at
         FROM documents
         WHERE embedding_status = 'error'
         ORDER BY embedding_updated_at DESC
         LIMIT 5",
    )?;
    let err_rows = err_stmt.query_map([], |row| {
        Ok(json!({
            "document_id": row.get::<_, i64>(0)?,
            "kind": row.get::<_, Option<String>>(1)?,
            "error": row.get::<_, Option<String>>(2)?,
            "updated_at": row.get::<_, Option<String>>(3)?
        }))
    })?;
    let recent_errors: Vec<Value> = err_rows.filter_map(|r| r.ok()).collect();

    let out = json!({
        "summary": {
            "reference_count": reference_count,
            "document_count": document_count
        },
        "references": refs,
        "embedding_counts": embedding_counts,
        "recent_errors": recent_errors
    });

    Ok(serde_json::to_string_pretty(&out)?)
}

fn tool_sync_zotero(state: Arc<AppState>) -> Result<String> {
    let reader = ZoteroReader::new()?;
    let items = reader.get_items()?;

    let mut conn = state.conn.lock().unwrap();
    let tx = conn.transaction()?;

    let mut existing_map = std::collections::HashMap::new();
    {
        let mut key_stmt = tx.prepare(
            "SELECT value, reference_id FROM reference_external_ids WHERE scheme = 'zotero_key'",
        )?;
        let keys_iter = key_stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?;
        for k in keys_iter {
            let (val, id) = k?;
            existing_map.insert(val, id);
        }
    }

    let mut refs_added = 0usize;
    let mut refs_updated = 0usize;
    let mut docs_upserted = 0usize;
    let mut files_upserted = 0usize;

    for item in items {
        let reference_id: i64 = if let Some(existing_id) = existing_map.get(&item.key) {
            tx.execute(
                "UPDATE reference_items
                 SET title = ?, authors = ?, year = ?, updated_at = CURRENT_TIMESTAMP
                 WHERE id = ?",
                (&item.title, &item.authors, &item.date, existing_id),
            )?;
            refs_updated += 1;
            *existing_id
        } else {
            tx.execute(
                "INSERT INTO reference_items (title, authors, year, source_system, source_id)
                 VALUES (?, ?, ?, 'zotero', ?)",
                (&item.title, &item.authors, &item.date, &item.key),
            )?;
            let new_ref_id = tx.last_insert_rowid();
            tx.execute(
                "INSERT INTO reference_external_ids (reference_id, scheme, value)
                 VALUES (?, 'zotero_key', ?)",
                (new_ref_id, &item.key),
            )?;
            refs_added += 1;
            existing_map.insert(item.key.clone(), new_ref_id);
            new_ref_id
        };

        // Title document
        let title_content = item.title.as_deref().map(|t| t.trim()).filter(|t| !t.is_empty());
        let title_status = if title_content.is_some() { "pending" } else { "skipped" };
        {
            let existing: Option<(i64, Option<String>)> = tx.query_row(
                "SELECT id, content FROM documents
                 WHERE reference_id = ? AND kind = 'title' AND source_system = 'zotero' AND source_id = ?",
                (reference_id, &item.key),
                |row| Ok((row.get(0)?, row.get(1)?)),
            ).ok();

            if let Some((doc_id, old_content)) = existing {
                let old_trimmed = old_content.as_deref().map(|s| s.trim());
                let new_trimmed = title_content;
                let content_changed = old_trimmed != new_trimmed;
                let embedding_status = if content_changed && new_trimmed.is_some() {
                    "pending"
                } else if new_trimmed.is_none() {
                    "skipped"
                } else {
                    // no change; leave status alone
                    ""
                };
                if content_changed {
                    if embedding_status.is_empty() {
                        tx.execute(
                            "UPDATE documents SET content = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                            (new_trimmed, doc_id),
                        )?;
                    } else {
                        tx.execute(
                            "UPDATE documents
                             SET content = ?,
                                 embedding_status = ?,
                                 embedding_error = NULL,
                                 embedding_updated_at = NULL,
                                 updated_at = CURRENT_TIMESTAMP
                             WHERE id = ?",
                            (new_trimmed, embedding_status, doc_id),
                        )?;
                    }
                }
            } else {
                tx.execute(
                    "INSERT INTO documents (reference_id, kind, content, embedding_status, source_system, source_id)
                     VALUES (?, 'title', ?, ?, 'zotero', ?)",
                    (reference_id, title_content, title_status, &item.key),
                )?;
            }
            docs_upserted += 1;
        }

        // Abstract document
        let abs_content = item.abstract_note.as_deref().map(|t| t.trim()).filter(|t| !t.is_empty());
        let abs_status = if abs_content.is_some() { "pending" } else { "skipped" };
        {
            let existing: Option<(i64, Option<String>)> = tx.query_row(
                "SELECT id, content FROM documents
                 WHERE reference_id = ? AND kind = 'abstract' AND source_system = 'zotero' AND source_id = ?",
                (reference_id, &item.key),
                |row| Ok((row.get(0)?, row.get(1)?)),
            ).ok();

            if let Some((doc_id, old_content)) = existing {
                let old_trimmed = old_content.as_deref().map(|s| s.trim());
                let new_trimmed = abs_content;
                let content_changed = old_trimmed != new_trimmed;
                let embedding_status = if content_changed && new_trimmed.is_some() {
                    "pending"
                } else if new_trimmed.is_none() {
                    "skipped"
                } else {
                    ""
                };
                if content_changed {
                    if embedding_status.is_empty() {
                        tx.execute(
                            "UPDATE documents SET content = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                            (new_trimmed, doc_id),
                        )?;
                    } else {
                        tx.execute(
                            "UPDATE documents
                             SET content = ?,
                                 embedding_status = ?,
                                 embedding_error = NULL,
                                 embedding_updated_at = NULL,
                                 updated_at = CURRENT_TIMESTAMP
                             WHERE id = ?",
                            (new_trimmed, embedding_status, doc_id),
                        )?;
                    }
                }
            } else {
                tx.execute(
                    "INSERT INTO documents (reference_id, kind, content, embedding_status, source_system, source_id)
                         VALUES (?, 'abstract', ?, ?, 'zotero', ?)",
                    (reference_id, abs_content, abs_status, &item.key),
                )?;
            }
            docs_upserted += 1;
        }

        // PDF document
        if let Some(pdf_path) = &item.pdf_path {
            let document_id: i64 = match tx.query_row(
                "SELECT id FROM documents
                 WHERE reference_id = ? AND kind = 'full_text_pdf' AND source_system = 'zotero' AND source_id = ?
                 LIMIT 1",
                (reference_id, &item.key),
                |row| row.get::<_, i64>(0),
            ) {
                Ok(id) => id,
                Err(_) => {
                    tx.execute(
                        "INSERT INTO documents (reference_id, kind, content, embedding_status, source_system, source_id)
                         VALUES (?, 'full_text_pdf', NULL, 'skipped', 'zotero', ?)",
                        (reference_id, &item.key),
                    )?;
                    docs_upserted += 1;
                    tx.last_insert_rowid()
                }
            };

            tx.execute(
                "INSERT INTO document_files (document_id, file_path, mime_type, label, source_system, source_id)
                 VALUES (?, ?, 'application/pdf', 'main_pdf', 'zotero', ?)
                 ON CONFLICT(document_id, file_path) DO UPDATE SET
                    mime_type=excluded.mime_type,
                    label=excluded.label,
                    source_system=excluded.source_system,
                    source_id=excluded.source_id",
                (document_id, pdf_path, &item.key),
            )?;
            files_upserted += 1;
        }
    }

    tx.commit()?;

    let _ = state.worker_tx.send(WorkerSignal::Wake);

    Ok(format!(
        "Zotero sync complete. References added: {}, updated: {}. Documents upserted: {}. Document files upserted: {}.",
        refs_added, refs_updated, docs_upserted, files_upserted
    ))
}

fn tool_get_documentation(args: Value) -> Result<String> {
    let section = args.get("section").and_then(|s| s.as_str());

    let docs = match section {
        Some("workflows") => {
            r#"
# Supported Workflows

## Workflow 1: Semantic Search (Finding Relevant Papers)

**Use when**: You have text — a claim, question, paragraph, or topic — and need to find papers that support or relate to it.

**Covers**: Adding citations to uncited text, finding papers for a research question, building a bibliography.

**Core Pattern**:
```sql
SELECT r.title, r.authors, r.year, d.content,
       rerank_score('your claim or question', d.content) AS relevance
FROM vec_documents v
JOIN documents d ON v.rowid = d.id
JOIN reference_items r ON d.reference_id = r.id
WHERE v.embedding MATCH embed('your claim or question')
  AND k = 30 AND d.embedding_status = 'ready'
  AND d.kind IN ('title', 'abstract')
ORDER BY relevance DESC LIMIT 10;
```

**Steps**:
1. If input is a paragraph with multiple claims, break it into separate searchable statements
2. For each claim/question, run a semantic search (embed + rerank)
3. Add hard filters as needed: year range, author patterns, document kind
4. Review abstracts to assess relevance
5. For deeper validation (methodology, data), use preview_document_pdf_pages

**Tips**:
- Different claims need different searches — don't try to find one paper that covers everything
- Start with k=30, increase if results seem incomplete
- Use rerank_score() in ORDER BY for better relevance than raw vector distance

---

## Workflow 2: Citation QA (Verifying Existing Citations)

**Use when**: Text already has citations, and you need to verify they actually support the claims they're attached to.

**Key difference from Workflow 1**: You're NOT searching semantically. You already know which papers are cited — you're looking them up by bibliographic data and verifying appropriateness.

**Steps**:
1. Parse the text to identify each citation and the claim it's attached to
2. For each citation, look up the reference by title/author/year:
   ```sql
   SELECT r.id, r.title, r.authors, r.year, d.content AS abstract
   FROM reference_items r
   JOIN documents d ON d.reference_id = r.id AND d.kind = 'abstract'
   WHERE r.title LIKE '%key words from cited title%'
     AND r.year = '2023';
   ```
3. Read the abstract to assess whether it genuinely supports the claim
4. For rigorous QA, find the PDF and visually inspect relevant sections:
   ```sql
   SELECT df.id AS document_file_id, df.file_path
   FROM reference_items r
   JOIN documents d ON d.reference_id = r.id AND d.kind = 'full_text_pdf'
   JOIN document_files df ON df.document_id = d.id
   WHERE r.id = <reference_id>;
   ```
   Then call preview_document_pdf_pages with that document_file_id.
5. Flag citations that don't adequately support their claims

**When to do rigorous PDF review**:
- High-stakes documents (publications, grants)
- Claims about specific data, numbers, or methodology
- Citations that seem tangentially related based on abstract

---

## Workflow 3: Iterative Exploration (Deep Research)

**Use when**: You have a vague topic and need to understand what exists before targeting specific papers. Common for literature reviews or entering a new research area.

**CRITICAL PRINCIPLE**: Start broad, then narrow. Don't over-specify on first attempt.

**Steps**:
1. Start with a broad semantic query, no hard filters, large k:
   ```sql
   SELECT r.title, r.authors, r.year, d.content
   FROM vec_documents v
   JOIN documents d ON v.rowid = d.id
   JOIN reference_items r ON d.reference_id = r.id
   WHERE v.embedding MATCH embed('your broad topic')
     AND k = 50 AND d.embedding_status = 'ready'
   ORDER BY v.distance LIMIT 20;
   ```
2. Observe patterns in results: common terminology, author names, year ranges, themes
3. Progressively add filters based on what you learn:
   - Add year constraints if you see relevant work clusters in certain periods
   - Add author patterns if key researchers emerge
   - Refine the semantic query using vocabulary from relevant abstracts
4. Use rerank_score() once you've narrowed to a reasonable candidate set
5. Final query might look very different from first query

**Why this matters**: A hyper-specific first query may return nothing, missing papers that use different terminology for the same concepts.

---

## Query Design Tips

1. **k parameter**: Controls how many candidates sqlite-vec retrieves. Start with k=30-50. Increase if results seem incomplete.

2. **embed() vs rerank_score()**: 
   - embed() is fast, good for initial retrieval
   - rerank_score() is slower but more accurate for final ordering
   - Pattern: broad retrieval with embed(), then rerank the top candidates

3. **Combining hard and soft filters**: 
   - Hard: year, author LIKE, kind IN (...)
   - Soft: rerank_score() for relevance
   - Apply hard filters in WHERE, soft in ORDER BY

4. **Document kinds**:
   - 'title': Short, good for quick relevance
   - 'abstract': Detailed, best for substantive matching
   - 'full_text_pdf': Content is NULL, but PDF is available via preview_document_pdf_pages
"#
        }
        Some("schema") => {
            r#"
# Database Schema

## Core Tables

### reference_items
Canonical citeable references imported from Zotero.
- `id`: Primary key
- `title`, `authors`, `year`: Bibliographic metadata
- `source_system`: Origin (e.g., 'zotero')
- `source_id`: ID in source system

### documents
Text artifacts linked to references. Each reference typically has:
- A 'title' document (embedded)
- An 'abstract' document (embedded)
- Optionally a 'full_text_pdf' document (not embedded, but PDF available)

Key columns:
- `id`: Primary key (also used as rowid in vec_documents)
- `reference_id`: FK to reference_items
- `kind`: 'title', 'abstract', or 'full_text_pdf'
- `content`: The actual text (NULL for file-backed docs)
- `embedding_status`: 'pending', 'embedding', 'ready', 'error', 'skipped'

### vec_documents
sqlite-vec virtual table for vector similarity search.
- `rowid`: Matches documents.id
- `embedding`: 384-dim float32 vector
- Query with: `WHERE embedding MATCH embed('query') AND k = N`

### document_files
File attachments (PDFs) linked to documents.
- `id`: Primary key (use this for preview_document_pdf_pages)
- `document_id`: FK to documents
- `file_path`: Absolute path to the file
- `mime_type`: Usually 'application/pdf'

### reference_external_ids
Additional identifiers per reference (DOI, ISBN, Zotero key, etc.).
- `reference_id`: FK to reference_items
- `scheme`: Identifier type (e.g., 'zotero_key', 'doi')
- `value`: The identifier value

## Useful Joins

Reference with its documents:
```sql
SELECT r.*, d.id as doc_id, d.kind, d.embedding_status
FROM reference_items r
LEFT JOIN documents d ON d.reference_id = r.id;
```

Document with its PDF file:
```sql
SELECT d.*, df.id as file_id, df.file_path
FROM documents d
JOIN document_files df ON df.document_id = d.id
WHERE d.kind = 'full_text_pdf';
```
"#
        }
        Some("examples") => {
            r#"
# SQL Examples

## Basic Semantic Search (start here)
```sql
SELECT r.title, r.authors, r.year, d.kind, d.content
FROM vec_documents v
JOIN documents d ON v.rowid = d.id
JOIN reference_items r ON d.reference_id = r.id
WHERE v.embedding MATCH embed('your search topic')
  AND k = 30
  AND d.embedding_status = 'ready'
ORDER BY v.distance
LIMIT 15;
```

## Semantic Search with Reranking (better relevance)
```sql
SELECT r.title, r.authors, r.year, d.kind, d.content,
       rerank_score('your specific question', d.content) AS relevance
FROM vec_documents v
JOIN documents d ON v.rowid = d.id
JOIN reference_items r ON d.reference_id = r.id
WHERE v.embedding MATCH embed('your specific question')
  AND k = 40
  AND d.embedding_status = 'ready'
  AND d.kind IN ('title', 'abstract')
ORDER BY relevance DESC
LIMIT 10;
```

## Filtered by Year and Author
```sql
SELECT r.title, r.authors, r.year, d.content,
       rerank_score('immune response mechanisms', d.content) AS relevance
FROM vec_documents v
JOIN documents d ON v.rowid = d.id
JOIN reference_items r ON d.reference_id = r.id
WHERE v.embedding MATCH embed('immune response mechanisms')
  AND k = 50
  AND d.embedding_status = 'ready'
  AND d.kind = 'abstract'
  AND r.year >= '2020'
  AND r.authors LIKE '%Smith%'
ORDER BY relevance DESC
LIMIT 10;
```

## Find PDFs for Full-Text Preview
```sql
SELECT r.title, df.id AS document_file_id, df.file_path
FROM reference_items r
JOIN documents d ON d.reference_id = r.id AND d.kind = 'full_text_pdf'
JOIN document_files df ON df.document_id = d.id
WHERE df.mime_type = 'application/pdf'
LIMIT 20;
```
Then use preview_document_pdf_pages with the document_file_id.

## Check Library Readiness
```sql
SELECT kind, embedding_status, COUNT(*) AS count
FROM documents
GROUP BY kind, embedding_status
ORDER BY kind, embedding_status;
```

## Search Titles Only (fast, for quick scans)
```sql
SELECT r.title, r.authors, r.year
FROM vec_documents v
JOIN documents d ON v.rowid = d.id
JOIN reference_items r ON d.reference_id = r.id
WHERE v.embedding MATCH embed('machine learning')
  AND k = 20
  AND d.embedding_status = 'ready'
  AND d.kind = 'title'
ORDER BY v.distance
LIMIT 20;
```
"#
        }
        _ => {
            r#"
# Lit Lake MCP Documentation

## What This Server Does
Lit Lake connects your Zotero library to AI-powered semantic search. It syncs your references, generates embeddings for titles and abstracts, and lets you query with natural language via SQL.

## Quick Start
1. Call `sync_zotero` to import your library
2. Wait a moment for embeddings to be generated (check with `library_status`)
3. Use `sql_search` with embed() and rerank_score() to find relevant papers
4. Use `preview_document_pdf_pages` to visually inspect full PDFs when needed

## Tools Summary

| Tool | Purpose |
|------|---------|
| sync_zotero | Import/update references from Zotero |
| sql_search | Query with SQL + semantic search functions |
| library_status | Check library scope and embedding readiness |
| preview_document_pdf_pages | Visual PDF inspection for validation |
| get_documentation | This documentation |

## Key Concepts

**embed(text)**: Computes a semantic embedding. Use for similarity search:
```sql
WHERE v.embedding MATCH embed('your query') AND k = 30
```

**rerank_score(query, doc)**: Cross-encoder relevance score. Use for final ordering:
```sql
ORDER BY rerank_score('your query', d.content) DESC
```

**Document kinds**: 'title' (short), 'abstract' (detailed), 'full_text_pdf' (file only)

## Recommended Reading Order
1. `get_documentation` with section='workflows' — understand how to use the tools together
2. `get_documentation` with section='examples' — see SQL patterns
3. `get_documentation` with section='schema' — understand the data model

## Important Principle
**Start broad, then narrow.** Begin with simple queries that return results. Observe patterns. Progressively add filters based on what you learn. Don't over-specify on first attempt.
"#
        }
    };

    Ok(docs.to_string())
}

fn tool_sql_search(state: Arc<AppState>, args: Value) -> Result<String> {
    let query = args["query"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing query"))?;

    // Basic safety check: read-only
    let q_upper = query.to_uppercase();
    if q_upper.contains("INSERT ")
        || q_upper.contains("UPDATE ")
        || q_upper.contains("DELETE ")
        || q_upper.contains("DROP ")
        || q_upper.contains("ALTER ")
        || q_upper.contains("CREATE ")
    {
        return Err(anyhow::anyhow!("Only read-only queries allowed"));
    }

    let conn = state.conn.lock().unwrap();
    let mut stmt = conn.prepare(query)?;
    let col_count = stmt.column_count();
    let col_names: Vec<String> = stmt
        .column_names()
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    let rows = stmt.query_map([], |row| {
        let mut map = serde_json::Map::new();
        for i in 0..col_count {
            let val = match row.get_ref(i)? {
                rusqlite::types::ValueRef::Null => Value::Null,
                rusqlite::types::ValueRef::Integer(i) => json!(i),
                rusqlite::types::ValueRef::Real(f) => json!(f),
                rusqlite::types::ValueRef::Text(t) => json!(String::from_utf8_lossy(t)),
                rusqlite::types::ValueRef::Blob(_) => json!("<blob>"),
            };
            map.insert(col_names[i].clone(), val);
        }
        Ok(Value::Object(map))
    })?;

    let mut output = Vec::new();
    for r in rows {
        output.push(r?);
    }

    Ok(serde_json::to_string_pretty(&output)?)
}

fn handle_unknown(req: &JsonRpcRequest) -> Option<JsonRpcResponse> {
    if req.id.is_some() {
        Some(JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: req.id.clone(),
            result: None,
            error: Some(json!({
                "code": -32601,
                "message": "Method not found"
            })),
        })
    } else {
        None
    }
}
