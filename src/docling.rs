//! Docling module for PDF text extraction using granite-docling-258M ONNX model.
//!
//! This module provides:
//! - Model download from HuggingFace
//! - Image preprocessing (resize, normalize)
//! - ONNX inference with autoregressive generation
//! - DocTags parsing to extract plain text

use anyhow::{Context, Result};
use hf_hub::api::sync::Api;
use image::DynamicImage;
use ort::session::{builder::GraphOptimizationLevel, Session};
use regex::Regex;
use std::path::PathBuf;
use std::sync::OnceLock;
use tokenizers::Tokenizer;

/// HuggingFace repository for the ONNX model
const HF_REPO: &str = "lamco-development/granite-docling-258M-onnx";

/// Image preprocessing constants (SigLIP2 normalization)
const IMAGE_SIZE: u32 = 512;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Maximum tokens to generate (safety limit)
const MAX_TOKENS: usize = 4096;

/// Regex for stripping location tags
static LOC_REGEX: OnceLock<Regex> = OnceLock::new();
/// Regex for extracting text content from elements
static ELEMENT_REGEX: OnceLock<Regex> = OnceLock::new();

fn loc_regex() -> &'static Regex {
    LOC_REGEX.get_or_init(|| Regex::new(r"<loc_\d+>").unwrap())
}

fn element_regex() -> &'static Regex {
    ELEMENT_REGEX.get_or_init(|| {
        Regex::new(r"<(title|text|section_header|caption|list_item|footnote|page_header|page_footer|picture|table|formula|checkbox_selected|checkbox_unselected|ched|fcel|ecel|srow|nl)>(.*?)</\1>")
            .unwrap()
    })
}

/// Downloads model files from HuggingFace if not already cached.
pub fn download_model(cache_dir: &PathBuf) -> Result<(PathBuf, PathBuf)> {
    eprintln!("[docling] Downloading model from {}...", HF_REPO);
    
    let api = Api::new().context("Failed to create HuggingFace API client")?;
    let repo = api.model(HF_REPO.to_string());
    
    // Download model.onnx
    let model_path = repo
        .get("model.onnx")
        .context("Failed to download model.onnx")?;
    eprintln!("[docling] Model downloaded to {:?}", model_path);
    
    // Download tokenizer.json
    let tokenizer_path = repo
        .get("tokenizer.json")
        .context("Failed to download tokenizer.json")?;
    eprintln!("[docling] Tokenizer downloaded to {:?}", tokenizer_path);
    
    // Note: hf-hub caches to its own directory, we just return the paths
    // The cache_dir parameter is kept for API consistency with fastembed
    let _ = cache_dir;
    
    Ok((model_path, tokenizer_path))
}

/// Docling processor for running inference on document images.
pub struct DoclingProcessor {
    session: Session,
    tokenizer: Tokenizer,
    prompt_tokens: Vec<u32>,
}

impl DoclingProcessor {
    /// Create a new DoclingProcessor by loading the ONNX model and tokenizer.
    pub fn new(model_path: &PathBuf, tokenizer_path: &PathBuf) -> Result<Self> {
        eprintln!("[docling] Loading ONNX model from {:?}...", model_path);
        
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)
            .context("Failed to load ONNX model")?;
        
        eprintln!("[docling] Loading tokenizer from {:?}...", tokenizer_path);
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        
        // Pre-tokenize the prompt
        let prompt = "Convert this page to docling.";
        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Failed to encode prompt: {}", e))?;
        let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
        
        eprintln!("[docling] Model loaded successfully. Prompt tokens: {:?}", prompt_tokens.len());
        
        Ok(Self {
            session,
            tokenizer,
            prompt_tokens,
        })
    }
    
    /// Preprocess an image for the model.
    /// Resizes to 512x512 and applies SigLIP2 normalization.
    pub fn preprocess_image(&self, img: &DynamicImage) -> Vec<f32> {
        // Resize to 512x512
        let resized = img.resize_exact(IMAGE_SIZE, IMAGE_SIZE, image::imageops::FilterType::Lanczos3);
        let rgb = resized.to_rgb8();
        
        // Create tensor in CHW format (channels, height, width)
        let mut pixels = vec![0.0f32; 3 * (IMAGE_SIZE as usize) * (IMAGE_SIZE as usize)];
        
        for y in 0..IMAGE_SIZE {
            for x in 0..IMAGE_SIZE {
                let pixel = rgb.get_pixel(x, y);
                let idx = (y * IMAGE_SIZE + x) as usize;
                
                // Normalize: (pixel / 255.0 - mean) / std
                // Store in CHW format
                let hw = (IMAGE_SIZE * IMAGE_SIZE) as usize;
                pixels[idx] = (pixel[0] as f32 / 255.0 - MEAN[0]) / STD[0];           // R channel
                pixels[hw + idx] = (pixel[1] as f32 / 255.0 - MEAN[1]) / STD[1];      // G channel
                pixels[2 * hw + idx] = (pixel[2] as f32 / 255.0 - MEAN[2]) / STD[2];  // B channel
            }
        }
        
        pixels
    }
    
    /// Process a document image and return DocTags markup.
    /// 
    /// This runs autoregressive generation until </doctag> is produced or max tokens reached.
    pub fn process_image(&self, img: &DynamicImage) -> Result<String> {
        let pixel_values = self.preprocess_image(img);
        
        // For now, we'll use a simplified approach since full autoregressive
        // generation with ONNX requires careful handling of KV cache.
        // 
        // TODO: Implement full autoregressive generation loop
        // For initial implementation, we'll note this as a placeholder
        // that needs the actual model's input/output specifications.
        
        eprintln!("[docling] Processing image ({}x{})...", img.width(), img.height());
        
        // Placeholder: In a full implementation, we would:
        // 1. Run the encoder on the image
        // 2. Initialize decoder with prompt tokens
        // 3. Loop: run decoder, get next token, append, repeat until </doctag>
        
        // For now, return an error indicating this needs proper ONNX model integration
        // This will be refined once we can test with the actual model
        Err(anyhow::anyhow!(
            "Full ONNX inference not yet implemented. \
             The model requires autoregressive generation which needs \
             specific handling based on the model's architecture. \
             Image preprocessed with {} pixels.", 
            pixel_values.len()
        ))
    }
    
    /// Get the tokenizer for external use (e.g., token counting for chunking).
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}

/// Parse DocTags markup to extract plain text in reading order.
/// 
/// The VLM generates DocTags in reading order, so we trust the sequence
/// and simply extract text content from each element.
pub fn doctags_to_text(doctags: &str) -> String {
    // Remove location tags
    let no_locs = loc_regex().replace_all(doctags, "");
    
    // Extract text from elements in order
    let mut text_parts = Vec::new();
    
    for cap in element_regex().captures_iter(&no_locs) {
        let tag = &cap[1];
        let content = cap[2].trim();
        
        if content.is_empty() {
            continue;
        }
        
        match tag {
            // Block-level elements get their own paragraph
            "title" | "text" | "section_header" | "caption" | "list_item" | 
            "footnote" | "page_header" | "page_footer" | "formula" => {
                text_parts.push(content.to_string());
            }
            // Table cells get joined with tabs
            "ched" | "fcel" | "ecel" => {
                // If previous part was also a cell, append with tab
                if let Some(last) = text_parts.last_mut() {
                    if !last.ends_with('\n') {
                        last.push('\t');
                        last.push_str(content);
                    } else {
                        text_parts.push(content.to_string());
                    }
                } else {
                    text_parts.push(content.to_string());
                }
            }
            // Row breaks
            "nl" | "srow" => {
                if let Some(last) = text_parts.last_mut() {
                    if !last.ends_with('\n') {
                        last.push('\n');
                    }
                }
            }
            // Skip structural elements without text
            "picture" | "table" | "checkbox_selected" | "checkbox_unselected" => {}
            _ => {
                text_parts.push(content.to_string());
            }
        }
    }
    
    text_parts.join("\n\n")
}

/// Section structure for chunking
#[derive(Debug, Clone)]
pub struct Section {
    pub header: Option<String>,
    pub paragraphs: Vec<String>,
}

/// Parse DocTags into sections for chunking.
pub fn doctags_to_sections(doctags: &str) -> Vec<Section> {
    let no_locs = loc_regex().replace_all(doctags, "");
    
    let mut sections = Vec::new();
    let mut current_section = Section {
        header: None,
        paragraphs: Vec::new(),
    };
    
    for cap in element_regex().captures_iter(&no_locs) {
        let tag = &cap[1];
        let content = cap[2].trim();
        
        if content.is_empty() {
            continue;
        }
        
        match tag {
            "section_header" => {
                // Save current section if it has content
                if !current_section.paragraphs.is_empty() || current_section.header.is_some() {
                    sections.push(current_section);
                }
                // Start new section
                current_section = Section {
                    header: Some(content.to_string()),
                    paragraphs: Vec::new(),
                };
            }
            "title" => {
                // Title goes in its own section at the start
                if current_section.header.is_none() && current_section.paragraphs.is_empty() {
                    current_section.header = Some(content.to_string());
                } else {
                    current_section.paragraphs.push(content.to_string());
                }
            }
            "text" | "caption" | "list_item" | "footnote" | "formula" => {
                current_section.paragraphs.push(content.to_string());
            }
            // Accumulate table cells into a single paragraph
            "ched" | "fcel" | "ecel" => {
                if let Some(last) = current_section.paragraphs.last_mut() {
                    last.push('\t');
                    last.push_str(content);
                } else {
                    current_section.paragraphs.push(content.to_string());
                }
            }
            "nl" | "srow" => {
                // Row break in table
                if let Some(last) = current_section.paragraphs.last_mut() {
                    if !last.ends_with('\n') {
                        last.push('\n');
                    }
                }
            }
            _ => {}
        }
    }
    
    // Don't forget the last section
    if !current_section.paragraphs.is_empty() || current_section.header.is_some() {
        sections.push(current_section);
    }
    
    sections
}

/// Chunk sections into ~512 token chunks.
/// Each chunk is prefixed with its section header for context.
pub fn chunk_sections(sections: Vec<Section>, max_tokens: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    
    for section in sections {
        let header_prefix = section.header
            .as_ref()
            .map(|h| format!("[{}]\n\n", h))
            .unwrap_or_default();
        
        let header_token_estimate = estimate_tokens(&header_prefix);
        
        let mut current_chunk = header_prefix.clone();
        let mut current_tokens = header_token_estimate;
        
        for para in section.paragraphs {
            let para_tokens = estimate_tokens(&para);
            
            if current_tokens + para_tokens > max_tokens && current_tokens > header_token_estimate {
                // Flush current chunk
                let trimmed = current_chunk.trim();
                if !trimmed.is_empty() {
                    chunks.push(trimmed.to_string());
                }
                // Start new chunk with section header for context
                current_chunk = format!("{}{}", header_prefix, para);
                current_tokens = header_token_estimate + para_tokens;
            } else {
                if current_tokens > header_token_estimate {
                    current_chunk.push_str("\n\n");
                }
                current_chunk.push_str(&para);
                current_tokens += para_tokens;
            }
        }
        
        // Flush remaining content
        let trimmed = current_chunk.trim();
        if !trimmed.is_empty() {
            chunks.push(trimmed.to_string());
        }
    }
    
    chunks
}

/// Estimate token count using a simple heuristic.
/// Roughly 4 characters per token for English text.
fn estimate_tokens(text: &str) -> usize {
    (text.len() + 3) / 4
}

/// Process extracted text into chunks ready for embedding.
/// 
/// This is the main entry point for chunking:
/// 1. Parse DocTags into sections
/// 2. Chunk each section respecting max_tokens limit
/// 3. Return chunks with section headers for context
pub fn chunk_text(doctags: &str, max_tokens: usize) -> Vec<String> {
    let sections = doctags_to_sections(doctags);
    chunk_sections(sections, max_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_doctags_to_text() {
        let doctags = r#"<doctag>
            <title><loc_50><loc_20><loc_450><loc_60>Document Title</title>
            <text><loc_50><loc_80><loc_450><loc_200>First paragraph of content.</text>
            <section_header><loc_50><loc_220><loc_450><loc_240>Methods</section_header>
            <text><loc_50><loc_260><loc_450><loc_400>We conducted an experiment.</text>
        </doctag>"#;
        
        let text = doctags_to_text(doctags);
        assert!(text.contains("Document Title"));
        assert!(text.contains("First paragraph"));
        assert!(text.contains("Methods"));
        assert!(text.contains("experiment"));
    }
    
    #[test]
    fn test_doctags_to_sections() {
        let doctags = r#"<doctag>
            <title><loc_50><loc_20>Introduction</title>
            <text><loc_50><loc_80>Some intro text.</text>
            <section_header><loc_50><loc_220>Methods</section_header>
            <text><loc_50><loc_260>Method description.</text>
            <text><loc_50><loc_300>More methods.</text>
        </doctag>"#;
        
        let sections = doctags_to_sections(doctags);
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].header.as_deref(), Some("Introduction"));
        assert_eq!(sections[1].header.as_deref(), Some("Methods"));
        assert_eq!(sections[1].paragraphs.len(), 2);
    }
    
    #[test]
    fn test_chunk_sections() {
        let sections = vec![
            Section {
                header: Some("Introduction".to_string()),
                paragraphs: vec![
                    "Short intro.".to_string(),
                    "Another paragraph.".to_string(),
                ],
            },
        ];
        
        let chunks = chunk_sections(sections, 100);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].starts_with("[Introduction]"));
    }
    
    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("test"), 1);
        assert_eq!(estimate_tokens("hello world"), 3); // 11 chars -> ~3 tokens
    }
}

