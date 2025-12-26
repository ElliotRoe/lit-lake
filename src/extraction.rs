//! PDF extraction worker for processing document_files.
//!
//! This module handles Phase 1 of the pipeline:
//! 1. Find pending PDFs in document_files
//! 2. Render pages to images
//! 3. Run through Docling OCR
//! 4. Parse DocTags to plain text
//! 5. Chunk text and create document rows

use anyhow::Result;
use rusqlite::{params, Connection};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::Duration;

use crate::docling::{chunk_text, doctags_to_text, DoclingProcessor};
use crate::preview::PdfPreviewRenderer;

/// Target chunk size in tokens (matches BGE model context window)
const TARGET_CHUNK_TOKENS: usize = 512;

/// Configuration for the extraction worker
#[derive(Debug, Clone)]
pub struct ExtractionConfig {
    pub db_path: String,
    pub models_path: PathBuf,
    pub poll_interval: Duration,
    pub busy_timeout_ms: u64,
    /// Size of rendered PDF pages for OCR (512 is optimal for Docling)
    pub render_size: u32,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            db_path: "lit_lake.db".to_string(),
            models_path: PathBuf::from("."),
            poll_interval: Duration::from_millis(5000),
            busy_timeout_ms: 5000,
            render_size: 512,
        }
    }
}

/// Signal types for the extraction worker
pub enum ExtractionSignal {
    Wake,
    Shutdown,
}

/// Worker that extracts text from PDFs using Docling
pub struct ExtractionWorker {
    cfg: ExtractionConfig,
    rx: Receiver<ExtractionSignal>,
}

impl ExtractionWorker {
    pub fn new(cfg: ExtractionConfig, rx: Receiver<ExtractionSignal>) -> Self {
        Self { cfg, rx }
    }

    pub fn run(self, processor: DoclingProcessor) -> Result<()> {
        eprintln!("[extraction] Starting extraction worker...");

        let mut conn = Connection::open(&self.cfg.db_path)?;
        conn.busy_timeout(Duration::from_millis(self.cfg.busy_timeout_ms))?;
        let _mode: String = conn.query_row("PRAGMA journal_mode=WAL;", [], |row| row.get(0))?;

        loop {
            // Wait to be nudged, or poll periodically
            match self.rx.recv_timeout(self.cfg.poll_interval) {
                Ok(ExtractionSignal::Shutdown) => {
                    eprintln!("[extraction] Shutdown received.");
                    return Ok(());
                }
                Ok(ExtractionSignal::Wake) => { /* fall through */ }
                Err(RecvTimeoutError::Timeout) => { /* poll */ }
                Err(RecvTimeoutError::Disconnected) => {
                    return Ok(());
                }
            }

            // Process one PDF at a time (they're slow)
            match self.process_one_pdf(&mut conn, &processor) {
                Ok(true) => {
                    eprintln!("[extraction] Processed a PDF, checking for more...");
                }
                Ok(false) => {
                    // No pending PDFs
                }
                Err(e) => {
                    eprintln!("[extraction] Error processing PDF: {:?}", e);
                }
            }
        }
    }

    /// Process one pending PDF file.
    /// Returns true if a PDF was processed, false if none pending.
    fn process_one_pdf(&self, conn: &mut Connection, processor: &DoclingProcessor) -> Result<bool> {
        // Find a pending PDF
        let pending: Option<(i64, i64, String, i64)> = conn
            .query_row(
                "SELECT df.id, df.document_id, df.file_path, d.reference_id
                 FROM document_files df
                 JOIN documents d ON df.document_id = d.id
                 WHERE df.extraction_status = 'pending'
                   AND df.mime_type = 'application/pdf'
                 LIMIT 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .ok();

        let (file_id, _document_id, file_path, reference_id) = match pending {
            Some(p) => p,
            None => return Ok(false),
        };

        eprintln!(
            "[extraction] Processing PDF: {} (file_id={})",
            file_path, file_id
        );

        // Claim the file
        let affected = conn.execute(
            "UPDATE document_files 
             SET extraction_status = 'extracting', extraction_error = NULL
             WHERE id = ? AND extraction_status = 'pending'",
            [file_id],
        )?;

        if affected == 0 {
            // Someone else claimed it
            return Ok(true);
        }

        // Try to extract text
        match self.extract_pdf_text(&file_path, processor) {
            Ok(extracted_text) => {
                // Store extracted text
                conn.execute(
                    "UPDATE document_files 
                     SET extracted_text = ?, extraction_status = 'ready', extraction_error = NULL
                     WHERE id = ?",
                    params![&extracted_text, file_id],
                )?;

                // Chunk the text and create document rows
                let chunks = chunk_text(&extracted_text, TARGET_CHUNK_TOKENS);
                eprintln!(
                    "[extraction] Extracted {} characters, {} chunks from {}",
                    extracted_text.len(),
                    chunks.len(),
                    file_path
                );

                // Delete any existing chunks for this file
                conn.execute(
                    "DELETE FROM documents WHERE document_file_id = ? AND kind = 'pdf_chunk'",
                    [file_id],
                )?;

                // Insert new chunks
                let tx = conn.transaction()?;
                for (idx, chunk) in chunks.iter().enumerate() {
                    tx.execute(
                        "INSERT INTO documents (reference_id, document_file_id, kind, content, chunk_index, embedding_status)
                         VALUES (?, ?, 'pdf_chunk', ?, ?, 'pending')",
                        params![reference_id, file_id, chunk, idx as i64],
                    )?;
                }
                tx.commit()?;

                eprintln!(
                    "[extraction] Created {} pdf_chunk documents for file_id={}",
                    chunks.len(),
                    file_id
                );
            }
            Err(e) => {
                let error_msg = format!("{:?}", e);
                eprintln!("[extraction] Failed to extract {}: {}", file_path, error_msg);
                conn.execute(
                    "UPDATE document_files 
                     SET extraction_status = 'error', extraction_error = ?
                     WHERE id = ?",
                    params![&error_msg, file_id],
                )?;
            }
        }

        Ok(true)
    }

    /// Extract text from a PDF file using Docling.
    fn extract_pdf_text(&self, file_path: &str, processor: &DoclingProcessor) -> Result<String> {
        // Get page count first
        let pdf_bytes = std::fs::read(file_path)?;
        let pdf = hayro::Pdf::new(std::sync::Arc::new(pdf_bytes))
            .map_err(|e| anyhow::anyhow!("Failed to parse PDF: {:?}", e))?;
        let page_count = pdf.pages().len() as u32;
        drop(pdf); // Release the PDF before rendering

        if page_count == 0 {
            return Ok(String::new());
        }

        eprintln!(
            "[extraction] PDF has {} pages, rendering at {}px",
            page_count, self.cfg.render_size
        );

        let mut all_doctags = String::new();

        // Process each page
        for page_num in 1..=page_count {
            // Render page to PNG
            let batch = PdfPreviewRenderer::page_range_png(
                file_path,
                page_num,
                page_num,
                self.cfg.render_size,
            )?;

            if let Some(page) = batch.pages.first() {
                // Load PNG bytes as image
                let img = image::load_from_memory(&page.png_bytes)?;

                // Run through Docling
                match processor.process_image(&img) {
                    Ok(doctags) => {
                        all_doctags.push_str(&doctags);
                        all_doctags.push('\n');
                    }
                    Err(e) => {
                        // Log but continue with other pages
                        eprintln!(
                            "[extraction] Warning: Failed to process page {}: {:?}",
                            page_num, e
                        );
                    }
                }
            }

            // Progress indicator
            if page_num % 5 == 0 {
                eprintln!("[extraction] Processed {}/{} pages", page_num, page_count);
            }
        }

        // If Docling failed (not yet fully implemented), fall back to empty
        // In a full implementation, we'd have actual DocTags here
        if all_doctags.is_empty() {
            // For now, return a placeholder indicating the PDF was scanned
            // This allows the pipeline to continue even without full OCR
            return Ok(format!(
                "[PDF with {} pages - OCR pending full Docling implementation]",
                page_count
            ));
        }

        // Parse DocTags to plain text
        let plain_text = doctags_to_text(&all_doctags);
        Ok(plain_text)
    }
}

/// Helper function to process a single PDF synchronously (for testing or manual extraction)
pub fn extract_single_pdf(
    file_path: &str,
    processor: &DoclingProcessor,
    render_size: u32,
) -> Result<String> {
    let worker = ExtractionWorker {
        cfg: ExtractionConfig {
            render_size,
            ..Default::default()
        },
        rx: std::sync::mpsc::channel().1, // Dummy receiver
    };
    worker.extract_pdf_text(file_path, processor)
}

