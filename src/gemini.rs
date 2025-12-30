//! Gemini API client for general-purpose LLM operations.

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::Path;
use std::time::Duration;
use tokio::fs::File;
use tokio_util::codec::{BytesCodec, FramedRead};

/// Available Gemini models.
#[derive(Debug, Clone)]
pub enum GeminiModel {
    /// Gemini 3 Flash - balanced speed/quality
    Flash,
    /// Gemini 3 Pro - highest quality, slower
    Pro,
    /// Custom model name
    Custom(String),
}

impl GeminiModel {
    pub fn as_str(&self) -> &str {
        match self {
            GeminiModel::Flash => "gemini-3-flash-preview",
            GeminiModel::Pro => "gemini-3-pro-preview",
            GeminiModel::Custom(name) => name,
        }
    }
}

impl Default for GeminiModel {
    fn default() -> Self {
        GeminiModel::Flash
    }
}

/// Thinking level controls the depth of the model's internal reasoning.
/// Higher levels produce more carefully reasoned outputs but increase latency.
#[derive(Debug, Clone, Copy, Default, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
    /// Matches "no thinking" for most queries. Flash only.
    /// Model may think minimally for complex coding tasks.
    Minimal,
    /// Minimizes latency and cost. Best for simple instruction following.
    Low,
    /// Balanced thinking for most tasks. Flash only.
    Medium,
    /// Maximizes reasoning depth (default). Model may take longer for first token.
    #[default]
    High,
}

/// Configuration for the Gemini client.
#[derive(Debug, Clone)]
pub struct GeminiConfig {
    pub api_key: String,
    pub model: GeminiModel,
    /// Temperature for generation. Gemini 3 recommends keeping at 1.0.
    pub temperature: f32,
    /// Thinking level for reasoning depth.
    pub thinking_level: ThinkingLevel,
    pub timeout: Duration,
    pub max_file_wait_attempts: u32,
}

impl GeminiConfig {
    pub fn from_env() -> Result<Self> {
        let api_key =
            std::env::var("GEMINI_API_KEY").context("GEMINI_API_KEY environment variable not set")?;

        Ok(Self {
            api_key,
            model: GeminiModel::default(),
            temperature: 1.0, // Gemini 3 recommends 1.0
            thinking_level: ThinkingLevel::default(),
            timeout: Duration::from_secs(300), // Thinking can take a while
            max_file_wait_attempts: 30,
        })
    }

    pub fn with_model(mut self, model: GeminiModel) -> Self {
        self.model = model;
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn with_thinking_level(mut self, level: ThinkingLevel) -> Self {
        self.thinking_level = level;
        self
    }
}

// API Request/Response Types

#[derive(Serialize)]
struct GenerateContentRequest {
    contents: Vec<Content>,
    #[serde(rename = "generationConfig", skip_serializing_if = "Option::is_none")]
    generation_config: Option<GenerationConfig>,
}

#[derive(Serialize)]
struct Content {
    role: String,
    parts: Vec<Part>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum Part {
    Text { text: String },
    FileData { file_data: FileData },
}

#[derive(Serialize)]
struct FileData {
    mime_type: String,
    file_uri: String,
}

#[derive(Serialize)]
struct GenerationConfig {
    temperature: f32,
    #[serde(rename = "thinkingConfig", skip_serializing_if = "Option::is_none")]
    thinking_config: Option<ThinkingConfig>,
    #[serde(rename = "responseMimeType", skip_serializing_if = "Option::is_none")]
    response_mime_type: Option<String>,
    #[serde(rename = "responseSchema", skip_serializing_if = "Option::is_none")]
    response_schema: Option<Value>,
}

#[derive(Serialize)]
struct ThinkingConfig {
    #[serde(rename = "thinkingLevel")]
    thinking_level: ThinkingLevel,
}

#[derive(Deserialize, Debug)]
struct FileUploadResponse {
    file: FileInfo,
}

#[derive(Deserialize, Debug, Clone)]
pub struct FileInfo {
    pub uri: String,
    #[serde(rename = "mimeType")]
    pub mime_type: String,
    pub state: String,
    pub name: String,
}

/// Client for interacting with the Gemini API.
#[derive(Clone)]
pub struct GeminiClient {
    client: Client,
    config: GeminiConfig,
}

impl GeminiClient {
    pub fn new(config: GeminiConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self { client, config })
    }

    pub fn from_env() -> Result<Self> {
        Self::new(GeminiConfig::from_env()?)
    }

    pub fn model(&self) -> &str {
        self.config.model.as_str()
    }

    /// Upload a file to Gemini's file storage.
    pub async fn upload_file(&self, path: &Path) -> Result<FileInfo> {
        let file_name = path
            .file_name()
            .context("Invalid file path")?
            .to_string_lossy()
            .to_string();

        let mime_type = Self::detect_mime_type(path);
        let file_size = path.metadata()?.len();

        let start_url = "https://generativelanguage.googleapis.com/upload/v1beta/files";

        let metadata = json!({
            "file": {
                "display_name": file_name
            }
        });

        let res = self
            .client
            .post(start_url)
            .header("x-goog-api-key", &self.config.api_key)
            .header("X-Goog-Upload-Protocol", "resumable")
            .header("X-Goog-Upload-Command", "start")
            .header("X-Goog-Upload-Header-Content-Length", file_size.to_string())
            .header("X-Goog-Upload-Header-Content-Type", mime_type)
            .header("Content-Type", "application/json")
            .json(&metadata)
            .send()
            .await?;

        if !res.status().is_success() {
            let status = res.status();
            let body = res.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "Failed to start upload session: {} - {}",
                status,
                body
            ));
        }

        let upload_url = res
            .headers()
            .get("x-goog-upload-url")
            .context("No upload URL in response")?
            .to_str()?
            .to_string();

        let file = File::open(path).await?;
        let stream = FramedRead::new(file, BytesCodec::new());
        let body = reqwest::Body::wrap_stream(stream);

        let res = self
            .client
            .post(&upload_url)
            .header("Content-Length", file_size.to_string())
            .header("X-Goog-Upload-Offset", "0")
            .header("X-Goog-Upload-Command", "upload, finalize")
            .body(body)
            .send()
            .await?;

        if !res.status().is_success() {
            let status = res.status();
            let body = res.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "Failed to upload file: {} - {}",
                status,
                body
            ));
        }

        let upload_resp: FileUploadResponse = res.json().await?;
        let mut file_info = upload_resp.file;

        let mut attempts = 0;
        while file_info.state == "PROCESSING" {
            if attempts >= self.config.max_file_wait_attempts {
                return Err(anyhow::anyhow!(
                    "File processing timed out after {} attempts",
                    attempts
                ));
            }

            tokio::time::sleep(Duration::from_secs(1)).await;

            let get_url = format!(
                "https://generativelanguage.googleapis.com/v1beta/{}",
                file_info.name
            );

            let res = self
                .client
                .get(&get_url)
                .header("x-goog-api-key", &self.config.api_key)
                .send()
                .await?;
            if res.status().is_success() {
                let wrapper: FileUploadResponse = res.json().await?;
                file_info = wrapper.file;
            }
            attempts += 1;
        }

        if file_info.state == "FAILED" {
            return Err(anyhow::anyhow!("File processing failed"));
        }

        Ok(file_info)
    }

    pub async fn delete_file(&self, file_name: &str) -> Result<()> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/{}",
            file_name
        );

        let res = self
            .client
            .delete(&url)
            .header("x-goog-api-key", &self.config.api_key)
            .send()
            .await?;

        if !res.status().is_success() {
            let status = res.status();
            let body = res.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "Failed to delete file: {} - {}",
                status,
                body
            ));
        }

        Ok(())
    }

    /// Generate text from a prompt using default thinking level.
    pub async fn generate_text(&self, prompt: &str) -> Result<String> {
        self.generate_text_with_thinking(prompt, None).await
    }

    /// Generate text with explicit thinking level override.
    pub async fn generate_text_with_thinking(
        &self,
        prompt: &str,
        thinking_level: Option<ThinkingLevel>,
    ) -> Result<String> {
        self.generate_content(
            vec![Part::Text {
                text: prompt.to_string(),
            }],
            thinking_level,
            None,
            None,
        )
        .await
    }

    /// Generate content from an uploaded file with a prompt.
    pub async fn generate_from_file(
        &self,
        file_uri: &str,
        mime_type: &str,
        prompt: &str,
    ) -> Result<String> {
        self.generate_from_file_with_thinking(file_uri, mime_type, prompt, None)
            .await
    }

    /// Generate from file with explicit thinking level.
    pub async fn generate_from_file_with_thinking(
        &self,
        file_uri: &str,
        mime_type: &str,
        prompt: &str,
        thinking_level: Option<ThinkingLevel>,
    ) -> Result<String> {
        let parts = vec![
            Part::FileData {
                file_data: FileData {
                    mime_type: mime_type.to_string(),
                    file_uri: file_uri.to_string(),
                },
            },
            Part::Text {
                text: prompt.to_string(),
            },
        ];

        self.generate_content(parts, thinking_level, None, None)
            .await
    }

    /// Generate structured JSON output.
    pub async fn generate_json(&self, prompt: &str, schema: Option<Value>) -> Result<Value> {
        self.generate_json_with_thinking(prompt, schema, None).await
    }

    /// Generate JSON with explicit thinking level.
    pub async fn generate_json_with_thinking(
        &self,
        prompt: &str,
        schema: Option<Value>,
        thinking_level: Option<ThinkingLevel>,
    ) -> Result<Value> {
        let text = self
            .generate_content(
                vec![Part::Text {
                    text: prompt.to_string(),
                }],
                thinking_level,
                Some("application/json".to_string()),
                schema,
            )
            .await?;

        serde_json::from_str(&text).context("Failed to parse JSON response")
    }

    /// Generate structured JSON from an uploaded file.
    pub async fn generate_json_from_file(
        &self,
        file_uri: &str,
        mime_type: &str,
        prompt: &str,
        schema: Option<Value>,
    ) -> Result<Value> {
        self.generate_json_from_file_with_thinking(file_uri, mime_type, prompt, schema, None)
            .await
    }

    /// Generate JSON from file with explicit thinking level.
    pub async fn generate_json_from_file_with_thinking(
        &self,
        file_uri: &str,
        mime_type: &str,
        prompt: &str,
        schema: Option<Value>,
        thinking_level: Option<ThinkingLevel>,
    ) -> Result<Value> {
        let parts = vec![
            Part::FileData {
                file_data: FileData {
                    mime_type: mime_type.to_string(),
                    file_uri: file_uri.to_string(),
                },
            },
            Part::Text {
                text: prompt.to_string(),
            },
        ];

        let text = self
            .generate_content(parts, thinking_level, Some("application/json".to_string()), schema)
            .await?;

        serde_json::from_str(&text).context("Failed to parse JSON response")
    }

    async fn generate_content(
        &self,
        parts: Vec<Part>,
        thinking_level_override: Option<ThinkingLevel>,
        response_mime_type: Option<String>,
        response_schema: Option<Value>,
    ) -> Result<String> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
            self.config.model.as_str()
        );

        let thinking_level = thinking_level_override.unwrap_or(self.config.thinking_level);

        let request = GenerateContentRequest {
            contents: vec![Content {
                role: "user".to_string(),
                parts,
            }],
            generation_config: Some(GenerationConfig {
                temperature: self.config.temperature,
                thinking_config: Some(ThinkingConfig { thinking_level }),
                response_mime_type,
                response_schema,
            }),
        };

        let res = self
            .client
            .post(&url)
            .header("x-goog-api-key", &self.config.api_key)
            .json(&request)
            .send()
            .await?;

        if !res.status().is_success() {
            let status = res.status();
            let body = res.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("Gemini API error: {} - {}", status, body));
        }

        let response_json: Value = res.json().await?;

        let candidates = response_json["candidates"]
            .as_array()
            .context("No candidates in response")?;

        if candidates.is_empty() {
            return Err(anyhow::anyhow!("Empty candidates list in response"));
        }

        let text = candidates[0]["content"]["parts"][0]["text"]
            .as_str()
            .context("No text content in response")?;

        Ok(text.to_string())
    }

    fn detect_mime_type(path: &Path) -> &'static str {
        match path.extension().and_then(|e| e.to_str()) {
            Some("pdf") => "application/pdf",
            Some("png") => "image/png",
            Some("jpg") | Some("jpeg") => "image/jpeg",
            Some("gif") => "image/gif",
            Some("webp") => "image/webp",
            Some("txt") => "text/plain",
            Some("md") => "text/markdown",
            Some("json") => "application/json",
            _ => "application/octet-stream",
        }
    }
}
