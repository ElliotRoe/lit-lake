//! PDF text extraction with pluggable backends.
//!
//! This module provides a trait-based system for extracting text from PDFs,
//! allowing different extraction methods (Docling, Gemini, etc.) to be swapped.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      ExtractionWorker                           │
//! │  - Polls document_files for pending PDFs                        │
//! │  - Calls extractor.extract(path) -> Result<String>              │
//! │  - Stores result, chunks text, creates document rows            │
//! └─────────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     PdfExtractor trait                          │
//! │  fn extract(&self, pdf_path: &Path) -> Result<String>           │
//! └─────────────────────────────────────────────────────────────────┘
//!          ▲                    ▲                    ▲
//!          │                    │                    │
//!    ┌─────┴─────┐       ┌──────┴──────┐      ┌──────┴──────┐
//!    │  Docling  │       │   Gemini    │      │   Custom    │
//!    │ Extractor │       │  Extractor  │      │  Extractor  │
//!    └───────────┘       └─────────────┘      └─────────────┘
//! ```
//!
//! ## Usage
//!
//! Implement `PdfExtractor` for your extraction method:
//!
//! ```ignore
//! struct MyExtractor { /* config */ }
//!
//! impl PdfExtractor for MyExtractor {
//!     fn name(&self) -> &str { "my-extractor" }
//!     
//!     fn extract(&self, pdf_path: &Path) -> Result<String> {
//!         // Your extraction logic here
//!         // Returns markdown text
//!     }
//! }
//! ```

use anyhow::{Context, Result};
use futures::stream::{FuturesUnordered, StreamExt};
use rusqlite::{params, Connection};
use std::path::{Path, PathBuf};
use std::{fs, sync::Arc};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::time::Duration;

use crate::gemini::{GeminiClient, GeminiConfig, GeminiModel};
use tokio::sync::Semaphore;

// ============================================================================
// Extractor Trait
// ============================================================================

/// Trait for PDF text extraction backends.
///
/// Implementations convert a PDF file to markdown text. The extraction method
/// can be anything: local OCR, cloud API, embedded models, etc.
///
/// # Contract
///
/// - **Input**: Absolute path to a PDF file on the local filesystem
/// - **Output**: Extracted text as markdown (or plain text)
/// - **Errors**: Should return `Err` only for unrecoverable failures
///
/// # Examples
///
/// ```ignore
/// // Using a local VLM like Docling
/// let extractor = DoclingExtractor::new(model_path)?;
/// let markdown = extractor.extract(Path::new("/path/to/paper.pdf"))?;
///
/// // Using a cloud API like Gemini
/// let extractor = GeminiExtractor::new(api_key);
/// let markdown = extractor.extract(Path::new("/path/to/paper.pdf"))?;
/// ```
pub trait PdfExtractor: Send + Sync {
    /// Human-readable name of this extractor (for logging/debugging).
    fn name(&self) -> &str;

    /// Whether this extractor can actually process files.
    /// Returns false for noop/placeholder extractors.
    fn is_enabled(&self) -> bool {
        true
    }

    /// Extract text from a PDF file.
    ///
    /// # Arguments
    ///
    /// * `pdf_path` - Absolute path to the PDF file
    ///
    /// # Returns
    ///
    /// * `Ok(String)` - The extracted text as markdown
    /// * `Err` - If extraction fails irrecoverably
    fn extract(&self, pdf_path: &Path) -> Result<String>;

    /// Extract text from multiple PDFs concurrently.
    ///
    /// Default implementation processes sequentially.
    /// Override for concurrent processing (e.g., GeminiExtractor).
    fn extract_batch(&self, pdf_paths: &[PathBuf]) -> Vec<Result<String>> {
        pdf_paths.iter().map(|p| self.extract(p)).collect()
    }

    /// Extract PDFs and send results as they complete.
    ///
    /// Each result is sent via the channel immediately after extraction,
    /// allowing the caller to process/store results incrementally.
    ///
    /// The tuple is (index, file_path, result) where index maps back to the input slice.
    fn extract_streaming(
        &self,
        pdf_paths: &[PathBuf],
        result_tx: Sender<(usize, PathBuf, Result<String>)>,
    ) {
        // Default: extract sequentially and send each result immediately
        for (idx, path) in pdf_paths.iter().enumerate() {
            let result = self.extract(path);
            // If receiver is dropped, stop processing
            if result_tx.send((idx, path.clone(), result)).is_err() {
                break;
            }
        }
    }
}

// ============================================================================
// Placeholder/Noop Extractor
// ============================================================================

/// A no-op extractor that skips all PDFs.
///
/// Used as a fallback when no real extractor is configured.
/// PDFs remain in 'pending' state and are not processed.
pub struct NoopExtractor;

impl NoopExtractor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for NoopExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl PdfExtractor for NoopExtractor {
    fn name(&self) -> &str {
        "noop"
    }

    fn is_enabled(&self) -> bool {
        false // Noop extractor doesn't process anything
    }

    fn extract(&self, _pdf_path: &Path) -> Result<String> {
        // Should never be called since is_enabled() returns false
        Err(anyhow::anyhow!("NoopExtractor cannot extract PDFs"))
    }
}

// ============================================================================
// extract-pdf Extractor (local, pure Rust)
// ============================================================================

/// PDF extractor using the `pdf-extract` crate (local, no external services).
pub struct ExtractPdfExtractor;

impl ExtractPdfExtractor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ExtractPdfExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl PdfExtractor for ExtractPdfExtractor {
    fn name(&self) -> &str {
        "extract-pdf"
    }

    fn extract(&self, pdf_path: &Path) -> Result<String> {
        let bytes = fs::read(pdf_path)
            .with_context(|| format!("Failed to read PDF bytes from {}", pdf_path.display()))?;
        pdf_extract::extract_text_from_mem(&bytes)
            .with_context(|| format!("pdf-extract failed for {}", pdf_path.display()))
    }
}

// ============================================================================
// Gemini Extractor
// ============================================================================

/// PDF extractor using Google's Gemini API.
///
/// This extractor uploads PDFs to Gemini and uses the model's vision
/// capabilities to convert them to markdown. Supports concurrent processing
/// with rate limiting via semaphore.
///
/// # Requirements
///
/// - `GEMINI_API_KEY` environment variable must be set
/// - Network access to Google's API
pub struct GeminiExtractor {
    client: GeminiClient,
    runtime: tokio::runtime::Runtime,
    concurrency: usize,
}

impl GeminiExtractor {
    /// Default concurrency limit for parallel PDF processing.
    const DEFAULT_CONCURRENCY: usize = 10;

    /// Create a new Gemini extractor with the given client.
    pub fn new(client: GeminiClient) -> Result<Self> {
        Self::with_concurrency(client, Self::DEFAULT_CONCURRENCY)
    }

    /// Create a new Gemini extractor with custom concurrency limit.
    pub fn with_concurrency(client: GeminiClient, concurrency: usize) -> Result<Self> {
        let runtime = tokio::runtime::Runtime::new()?;
        Ok(Self {
            client,
            runtime,
            concurrency: concurrency.max(1),
        })
    }

    /// Create a new Gemini extractor using environment variables.
    pub fn from_env() -> Result<Self> {
        let client = GeminiClient::from_env()?;
        Self::new(client)
    }

    /// Create from environment with custom concurrency.
    pub fn from_env_with_concurrency(concurrency: usize) -> Result<Self> {
        let client = GeminiClient::from_env()?;
        Self::with_concurrency(client, concurrency)
    }

    /// Create a new Gemini extractor with custom configuration.
    pub fn with_config(config: GeminiConfig) -> Result<Self> {
        let client = GeminiClient::new(config)?;
        Self::new(client)
    }

    /// Create with a specific model.
    pub fn with_model(model: GeminiModel) -> Result<Self> {
        let config = GeminiConfig::from_env()?.with_model(model);
        Self::with_config(config)
    }
}

impl GeminiExtractor {
    /// The prompt used for PDF to markdown conversion.
    const EXTRACTION_PROMPT: &'static str = r#"Convert this PDF document to Markdown format.

Instructions:
- Preserve all headers, using appropriate markdown heading levels (# ## ###)
- Maintain document structure and hierarchy
- Convert tables to markdown tables
- Preserve lists (ordered and unordered)
- Include figure captions as italic text
- For equations, use LaTeX notation within $...$ or $$...$$
- Preserve emphasis (bold, italic) where present
- Output only the markdown content, no preamble or explanation"#;

    /// Extract a single PDF asynchronously.
    async fn extract_one_async(client: &GeminiClient, pdf_path: &Path) -> Result<String> {
        let file_info = client.upload_file(pdf_path).await?;

        let result = client
            .generate_from_file(&file_info.uri, "application/pdf", Self::EXTRACTION_PROMPT)
            .await;

        // Clean up uploaded file (best effort)
        let _ = client.delete_file(&file_info.name).await;

        result
    }
}

impl PdfExtractor for GeminiExtractor {
    fn name(&self) -> &str {
        "gemini"
    }

    fn extract(&self, pdf_path: &Path) -> Result<String> {
        self.runtime
            .block_on(Self::extract_one_async(&self.client, pdf_path))
    }

    fn extract_batch(&self, pdf_paths: &[PathBuf]) -> Vec<Result<String>> {
        // Use the streaming implementation and collect results
        let (tx, rx) = std::sync::mpsc::channel();
        self.extract_streaming(pdf_paths, tx);

        // Collect results and sort by index to maintain order
        let mut indexed_results: Vec<(usize, Result<String>)> = rx
            .into_iter()
            .map(|(idx, _path, result)| (idx, result))
            .collect();
        indexed_results.sort_by_key(|(idx, _)| *idx);
        indexed_results.into_iter().map(|(_, r)| r).collect()
    }

    fn extract_streaming(
        &self,
        pdf_paths: &[PathBuf],
        result_tx: Sender<(usize, PathBuf, Result<String>)>,
    ) {
        if pdf_paths.is_empty() {
            return;
        }

        // Clone paths for use in async block
        let paths: Vec<(usize, PathBuf)> = pdf_paths
            .iter()
            .enumerate()
            .map(|(i, p)| (i, p.clone()))
            .collect();

        self.runtime.block_on(async {
            let semaphore = Arc::new(Semaphore::new(self.concurrency));
            let mut futures = FuturesUnordered::new();

            for (idx, path) in paths {
                let client = self.client.clone();
                let permit = semaphore.clone();
                let path_clone = path.clone();

                futures.push(async move {
                    // Acquire permit before starting work
                    let _permit = permit.acquire().await.unwrap();
                    let result = Self::extract_one_async(&client, &path_clone).await;
                    (idx, path, result)
                });
            }

            // Send results as they complete
            while let Some((idx, path, result)) = futures.next().await {
                if result_tx.send((idx, path, result)).is_err() {
                    // Receiver dropped, stop processing
                    break;
                }
            }
        });
    }
}

// ============================================================================
// Text Chunking Utilities
// ============================================================================

/// Target chunk size in tokens (matches BGE model context window).
pub const TARGET_CHUNK_TOKENS: usize = 512;

/// Approximate tokens per character for English text.
const CHARS_PER_TOKEN: f64 = 4.0;

/// Split text into chunks of approximately `target_tokens` size.
///
/// Uses a simple paragraph-aware splitting strategy:
/// 1. Split on double newlines (paragraphs)
/// 2. Accumulate paragraphs until target size
/// 3. If a single paragraph exceeds target, split on sentences
pub fn chunk_text(text: &str, target_tokens: usize) -> Vec<String> {
    let target_chars = (target_tokens as f64 * CHARS_PER_TOKEN) as usize;

    if text.len() <= target_chars {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut current_chunk = String::new();

    // Split on double newlines (paragraphs)
    for paragraph in text.split("\n\n") {
        let paragraph = paragraph.trim();
        if paragraph.is_empty() {
            continue;
        }

        // If adding this paragraph would exceed target, flush current chunk
        if !current_chunk.is_empty()
            && current_chunk.len() + paragraph.len() + 2 > target_chars
        {
            chunks.push(current_chunk.trim().to_string());
            current_chunk = String::new();
        }

        // If single paragraph exceeds target, split on sentences
        if paragraph.len() > target_chars {
            // Flush anything we have
            if !current_chunk.is_empty() {
                chunks.push(current_chunk.trim().to_string());
                current_chunk = String::new();
            }

            // Split large paragraph on sentence boundaries
            let sentences = split_sentences(paragraph);
            for sentence in sentences {
                if current_chunk.len() + sentence.len() + 1 > target_chars {
                    if !current_chunk.is_empty() {
                        chunks.push(current_chunk.trim().to_string());
                        current_chunk = String::new();
                    }
                }
                if !current_chunk.is_empty() {
                    current_chunk.push(' ');
                }
                current_chunk.push_str(&sentence);
            }
        } else {
            // Normal paragraph, add to current chunk
            if !current_chunk.is_empty() {
                current_chunk.push_str("\n\n");
            }
            current_chunk.push_str(paragraph);
        }
    }

    // Don't forget the last chunk
    if !current_chunk.is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }

    // Filter out empty chunks
    chunks.into_iter().filter(|c| !c.is_empty()).collect()
}

/// Simple sentence splitter (English-focused).
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for c in text.chars() {
        current.push(c);
        if c == '.' || c == '!' || c == '?' {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current = String::new();
        }
    }

    if !current.trim().is_empty() {
        sentences.push(current.trim().to_string());
    }

    sentences
}

/// Light post-processing to improve chunking on raw PDF text.
///
/// - Normalizes line endings
/// - Joins wrapped lines within paragraphs
/// - Preserves paragraph breaks (blank lines)
/// - Repairs simple hyphen line breaks (e.g., "lin-\n guistic")
pub fn normalize_extracted_text(text: &str) -> String {
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    let mut out = String::new();

    for block in normalized.split("\n\n") {
        let block = block.trim();
        if block.is_empty() {
            continue;
        }

        let mut paragraph = String::new();
        let mut pending_hyphen = false;

        for line in block.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if pending_hyphen {
                paragraph.push_str(line);
                pending_hyphen = false;
            } else if paragraph.is_empty() {
                paragraph.push_str(line);
            } else {
                paragraph.push(' ');
                paragraph.push_str(line);
            }

            if paragraph.ends_with('-') {
                paragraph.pop();
                pending_hyphen = true;
            }
        }

        let collapsed = paragraph
            .split_whitespace()
            .collect::<Vec<&str>>()
            .join(" ");

        if !collapsed.is_empty() {
            if !out.is_empty() {
                out.push_str("\n\n");
            }
            out.push_str(&collapsed);
        }
    }

    out
}

// ============================================================================
// Extraction Worker
// ============================================================================

/// Configuration for the extraction worker.
#[derive(Debug, Clone)]
pub struct ExtractionConfig {
    /// Path to the SQLite database.
    pub db_path: String,
    /// How often to poll for new work when idle.
    pub poll_interval: Duration,
    /// SQLite busy timeout in milliseconds.
    pub busy_timeout_ms: u64,
    /// Number of PDFs to process concurrently.
    pub concurrency: usize,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            db_path: "lit_lake.db".to_string(),
            poll_interval: Duration::from_millis(5000),
            busy_timeout_ms: 5000,
            concurrency: 10,
        }
    }
}

/// Signals for controlling the extraction worker.
pub enum ExtractionSignal {
    /// Wake the worker to check for pending work.
    Wake,
    /// Shut down the worker gracefully.
    Shutdown,
}

/// Background worker that processes pending PDF extractions.
///
/// The worker:
/// 1. Polls `document_files` for rows with `extraction_status = 'pending'`
/// 2. Claims a file by setting status to `'extracting'`
/// 3. Calls the configured `PdfExtractor` to get markdown text
/// 4. Stores the result in `extracted_text`
/// 5. Chunks the text and creates `documents` rows for embedding
pub struct ExtractionWorker<E: PdfExtractor> {
    cfg: ExtractionConfig,
    rx: Receiver<ExtractionSignal>,
    extractor: E,
}

impl<E: PdfExtractor> ExtractionWorker<E> {
    /// Create a new extraction worker.
    ///
    /// # Arguments
    ///
    /// * `cfg` - Worker configuration
    /// * `rx` - Channel receiver for control signals
    /// * `extractor` - The PDF extractor implementation to use
    pub fn new(cfg: ExtractionConfig, rx: Receiver<ExtractionSignal>, extractor: E) -> Self {
        Self { cfg, rx, extractor }
    }

    /// Run the worker loop.
    ///
    /// This blocks until a `Shutdown` signal is received or the channel disconnects.
    pub fn run(self) -> Result<()> {
        eprintln!(
            "[extraction] Starting extraction worker with {} extractor...",
            self.extractor.name()
        );

        // If extractor isn't enabled (e.g., NoopExtractor), just idle
        if !self.extractor.is_enabled() {
            eprintln!("[extraction] Extractor disabled, worker will idle (PDFs won't be processed)");
            loop {
                match self.rx.recv() {
                    Ok(ExtractionSignal::Shutdown) | Err(_) => {
                        eprintln!("[extraction] Shutdown received.");
                        return Ok(());
                    }
                    Ok(ExtractionSignal::Wake) => {
                        // Ignore wake signals when disabled
                    }
                }
            }
        }

        eprintln!(
            "[extraction] Processing up to {} PDFs concurrently",
            self.cfg.concurrency
        );

        let mut conn = Connection::open(&self.cfg.db_path)?;
        conn.busy_timeout(Duration::from_millis(self.cfg.busy_timeout_ms))?;
        let _mode: String = conn.query_row("PRAGMA journal_mode=WAL;", [], |row| row.get(0))?;

        loop {
            // Wait for signal or poll interval
            match self.rx.recv_timeout(self.cfg.poll_interval) {
                Ok(ExtractionSignal::Shutdown) => {
                    eprintln!("[extraction] Shutdown received.");
                    return Ok(());
                }
                Ok(ExtractionSignal::Wake) => { /* fall through to process */ }
                Err(RecvTimeoutError::Timeout) => { /* poll for work */ }
                Err(RecvTimeoutError::Disconnected) => {
                    eprintln!("[extraction] Channel disconnected, shutting down.");
                    return Ok(());
                }
            }

            // Process a batch of PDFs concurrently
            match self.process_batch(&mut conn) {
                Ok(count) if count > 0 => {
                    eprintln!("[extraction] Processed {} PDFs, checking for more...", count);
                }
                Ok(_) => {
                    // No pending PDFs, wait for next signal
                }
                Err(e) => {
                    eprintln!("[extraction] Error processing batch: {:?}", e);
                }
            }
        }
    }

    /// Process all pending PDF files concurrently.
    ///
    /// Returns the number of PDFs processed.
    fn process_batch(&self, conn: &mut Connection) -> Result<usize> {
        let pending: Vec<(i64, String, i64)> = {
            let mut stmt = conn.prepare(
                "SELECT df.id, df.file_path, df.reference_id
                 FROM document_files df
                 WHERE df.extraction_status = 'pending'
                   AND df.mime_type = 'application/pdf'
                 ORDER BY df.id",
            )?;

            let rows = stmt.query_map([], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })?;
            rows.filter_map(|r| r.ok()).collect()
        };

        if pending.is_empty() {
            return Ok(0);
        }

        eprintln!(
            "[extraction] Found {} pending PDFs, processing with {} extractor...",
            pending.len(),
            self.extractor.name()
        );

        // Claim all files atomically
        let mut claimed: Vec<(i64, String, i64)> = Vec::new();
        {
            let tx = conn.transaction()?;
            for (file_id, file_path, reference_id) in &pending {
                let affected = tx.execute(
                    "UPDATE document_files 
                     SET extraction_status = 'extracting', extraction_error = NULL
                     WHERE id = ? AND (extraction_status = 'pending' OR extraction_status = 'error')",
                    [file_id],
                )?;
                if affected > 0 {
                    claimed.push((*file_id, file_path.clone(), *reference_id));
                }
            }
            tx.commit()?;
        }

        if claimed.is_empty() {
            return Ok(0);
        }

        // Build lookup map: path -> (file_id, reference_id)
        let path_to_info: std::collections::HashMap<PathBuf, (i64, i64)> = claimed
            .iter()
            .map(|(file_id, path, reference_id)| (PathBuf::from(path), (*file_id, *reference_id)))
            .collect();

        let paths: Vec<PathBuf> = claimed.iter().map(|(_, p, _)| PathBuf::from(p)).collect();

        // Create channel for receiving results as they complete
        let (tx, rx) = std::sync::mpsc::channel();

        // Spawn extraction in a separate thread so we can receive results on this thread
        let extractor = &self.extractor;
        let extractor_name = extractor.name().to_string();
        std::thread::scope(|s| {
            s.spawn(|| {
                extractor.extract_streaming(&paths, tx);
            });

            // Process results as they arrive - each is stored immediately
            let mut processed = 0;
            for (_idx, path, result) in rx {
                let (file_id, reference_id) = path_to_info[&path];

                match result {
                    Ok(extracted_text) => {
                        // Store extracted text immediately
                        if let Err(e) = conn.execute(
                            "UPDATE document_files 
                             SET extracted_text = ?, extraction_status = 'ready', extraction_error = NULL
                             WHERE id = ?",
                            params![&extracted_text, file_id],
                        ) {
                            eprintln!("[extraction] Failed to store result for {:?}: {:?}", path, e);
                            continue;
                        }

                        // Chunk the text and create document rows
                        let normalized_text = if self.extractor.name() == "extract-pdf" {
                            normalize_extracted_text(&extracted_text)
                        } else {
                            extracted_text.clone()
                        };
                        let chunks = chunk_text(&normalized_text, TARGET_CHUNK_TOKENS);
                        eprintln!(
                            "[extraction] Extracted {} chars (normalized {}), {} chunks from {:?}",
                            extracted_text.len(),
                            normalized_text.len(),
                            chunks.len(),
                            path
                        );

                        // Delete any existing chunks for this file
                        let _ = conn.execute(
                            "DELETE FROM documents WHERE document_file_id = ? AND kind = 'pdf_chunk'",
                            [file_id],
                        );

                        // Insert new chunks
                        if let Ok(tx) = conn.transaction() {
                            for (idx, chunk) in chunks.iter().enumerate() {
                                let _ = tx.execute(
                                    "INSERT INTO documents (reference_id, document_file_id, kind, content, chunk_index, embedding_status)
                                     VALUES (?, ?, 'pdf_chunk', ?, ?, 'pending')",
                                    params![reference_id, file_id, chunk, idx as i64],
                                );
                            }
                            let _ = tx.commit();
                        }
                    }
                    Err(e) => {
                        let error_msg = format!("{:?}", e);
                        eprintln!("[extraction] Failed to extract {:?}: {}", path, error_msg);
                        let _ = conn.execute(
                            "UPDATE document_files 
                             SET extraction_status = 'error', extraction_error = ?
                             WHERE id = ?",
                            params![&error_msg, file_id],
                        );
                    }
                }
                processed += 1;

                // Log progress periodically
                if processed % 10 == 0 {
                    eprintln!(
                        "[extraction] Progress: {}/{} PDFs processed with {} extractor",
                        processed,
                        claimed.len(),
                        extractor_name
                    );
                }
            }
        });

        Ok(claimed.len())
    }
}
