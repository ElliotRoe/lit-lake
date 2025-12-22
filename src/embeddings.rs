use anyhow::Result;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use rusqlite::{params, Connection};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::Duration;

pub const EMBEDDING_DIM: usize = 384;
pub const EMBEDDING_MODEL_NAME: &str = "BGESmallENV15";

#[derive(Debug, Clone)]
pub struct WorkerConfig {
    pub db_path: String,
    pub models_path: PathBuf,
    pub batch_size: usize,
    pub poll_interval: Duration,
    pub busy_timeout_ms: u64,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            db_path: "lit_lake.db".to_string(),
            models_path: PathBuf::from("."),
            batch_size: 64,
            poll_interval: Duration::from_millis(1500),
            busy_timeout_ms: 5000,
        }
    }
}

pub enum WorkerSignal {
    Wake,
    Shutdown,
}

pub struct EmbeddingWorker {
    cfg: WorkerConfig,
    rx: Receiver<WorkerSignal>,
}

impl EmbeddingWorker {
    pub fn new(cfg: WorkerConfig, rx: Receiver<WorkerSignal>) -> Self {
        Self { cfg, rx }
    }

    pub fn run(mut self) -> Result<()> {
        // Model lives on this dedicated thread (keeps complexity isolated).
        eprintln!("[embeddings] Loading embedding model from {:?}...", self.cfg.models_path);
        let mut model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::BGESmallENV15)
                .with_cache_dir(self.cfg.models_path.clone()),
        )?;

        let mut conn = Connection::open(&self.cfg.db_path)?;
        conn.busy_timeout(Duration::from_millis(self.cfg.busy_timeout_ms))?;
        let _mode: String = conn.query_row("PRAGMA journal_mode=WAL;", [], |row| row.get(0))?;

        loop {
            // Wait to be nudged, or poll periodically as a safety net.
            match self.rx.recv_timeout(self.cfg.poll_interval) {
                Ok(WorkerSignal::Shutdown) => {
                    eprintln!("[embeddings] Shutdown received.");
                    return Ok(());
                }
                Ok(WorkerSignal::Wake) => { /* fall through */ }
                Err(RecvTimeoutError::Timeout) => { /* poll */ }
                Err(RecvTimeoutError::Disconnected) => {
                    // If sender dropped, exit quietly.
                    return Ok(());
                }
            }

            // Drain backlog in batches each time we wake/poll.
            loop {
                let did_work = self.process_pending_batch(&mut conn, &mut model)?;
                if !did_work {
                    break;
                }
            }
        }
    }

    fn process_pending_batch(
        &mut self,
        conn: &mut Connection,
        model: &mut TextEmbedding,
    ) -> Result<bool> {
        // 1) Read a batch of pending docs.
        let (ids, texts) = {
            let mut stmt = conn.prepare(
                "SELECT id, content
                 FROM documents
                 WHERE embedding_status = 'pending'
                   AND kind IN ('title', 'abstract')
                   AND content IS NOT NULL
                   AND TRIM(content) <> ''
                 ORDER BY updated_at ASC
                 LIMIT ?",
            )?;

            let mut rows = stmt.query([self.cfg.batch_size as i64])?;
            let mut ids: Vec<i64> = Vec::new();
            let mut texts: Vec<String> = Vec::new();
            while let Some(row) = rows.next()? {
                let id: i64 = row.get(0)?;
                let content: String = row.get(1)?;
                ids.push(id);
                texts.push(content);
            }
            (ids, texts)
        };

        if ids.is_empty() {
            return Ok(false);
        }

        // 2) Claim them and keep only successfully claimed rows.
        // Note: we keep this tx short to reduce lock time.
        let mut claimed_ids: Vec<i64> = Vec::with_capacity(ids.len());
        let mut claimed_texts: Vec<String> = Vec::with_capacity(texts.len());
        {
            let tx = conn.transaction()?;
            for (id, text) in ids.into_iter().zip(texts.into_iter()) {
                let affected = tx.execute(
                    "UPDATE documents
                     SET embedding_status = 'embedding', embedding_error = NULL
                     WHERE id = ? AND embedding_status = 'pending'",
                    [id],
                )?;
                if affected == 1 {
                    claimed_ids.push(id);
                    claimed_texts.push(text);
                }
            }
            tx.commit()?;
        }

        if claimed_ids.is_empty() {
            return Ok(true);
        }

        // 3) Embed outside a write transaction.
        let embeddings = match model.embed(claimed_texts.clone(), None) {
            Ok(e) => e,
            Err(e) => {
                // Mark batch as error.
                let msg = format!("{e}");
                let tx = conn.transaction()?;
                for id in &claimed_ids {
                    tx.execute(
                        "UPDATE documents
                         SET embedding_status = 'error',
                             embedding_error = ?,
                             embedding_updated_at = NULL
                         WHERE id = ?",
                        params![msg, id],
                    )?;
                }
                tx.commit()?;
                return Ok(true);
            }
        };

        // 4) Write vectors + mark ready in one short transaction.
        let tx = conn.transaction()?;
        for (id, vec) in claimed_ids.iter().zip(embeddings.iter()) {
            // sqlite-vec expects float bytes. We store as little-endian f32 bytes.
            let vec_bytes: Vec<u8> = vec
                .iter()
                .flat_map(|f| f.to_le_bytes().to_vec())
                .collect();

            tx.execute("DELETE FROM vec_documents WHERE rowid = ?", [id])?;
            tx.execute(
                "INSERT INTO vec_documents(rowid, embedding) VALUES (?, ?)",
                params![id, vec_bytes],
            )?;
            tx.execute(
                "UPDATE documents
                 SET embedding_status = 'ready',
                     embedding_updated_at = CURRENT_TIMESTAMP,
                     embedding_error = NULL
                 WHERE id = ?",
                params![id],
            )?;
        }
        tx.commit()?;

        Ok(true)
    }
}


