use anyhow::Result;
use hayro::{render, FontQuery, InterpreterSettings, Pdf, RenderSettings};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct PdfPreviewPage {
    pub page_number: u32,
    pub png_bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct PdfPreviewBatch {
    pub pages: Vec<PdfPreviewPage>,
}

pub struct PdfPreviewRenderer;

impl PdfPreviewRenderer {
    pub fn page_range_png(
        pdf_path: &str,
        start_page: u32,
        end_page: u32,
        size: u32,
    ) -> Result<PdfPreviewBatch> {
        let pdf_bytes = std::fs::read(pdf_path)?;
        let pdf = Pdf::new(Arc::new(pdf_bytes))
            .map_err(|e| anyhow::anyhow!("Failed to parse PDF: {e:?}"))?;
        let pages = pdf.pages();

        let page_count = pages.len() as u32;
        if page_count == 0 {
            return Ok(PdfPreviewBatch { pages: vec![] });
        }
        if start_page == 0 || end_page == 0 {
            anyhow::bail!("Pages are 1-based; start_page/end_page must be >= 1");
        }
        if start_page > end_page {
            anyhow::bail!("start_page must be <= end_page");
        }
        if end_page > page_count {
            anyhow::bail!(
                "Page out of bounds: PDF has {} pages but end_page={}",
                page_count,
                end_page
            );
        }

        // TODO: Fetch fonts lazily / cache across calls.
        let interpreter_settings = InterpreterSettings {
            font_resolver: Arc::new(|query| match query {
                FontQuery::Standard(s) => Some(s.get_font_data()),
                FontQuery::Fallback(f) => Some(f.pick_standard_font().get_font_data()),
            }),
            ..Default::default()
        };

        let mut out_pages = Vec::with_capacity((end_page - start_page + 1) as usize);

        for page_number in start_page..=end_page {
            let page_idx = (page_number - 1) as usize;
            let page = pages
                .get(page_idx)
                .ok_or_else(|| anyhow::anyhow!("Page out of bounds: {}", page_number))?;

            let (w, h) = page.render_dimensions();
            let max_dim = w.max(h).max(1.0);
            let scale = (size as f32) / max_dim;

            let pixmap = render(
                page,
                &interpreter_settings,
                &RenderSettings {
                    x_scale: scale,
                    y_scale: scale,
                    ..Default::default()
                },
            );

            out_pages.push(PdfPreviewPage {
                page_number,
                png_bytes: pixmap.take_png(),
            });
        }

        Ok(PdfPreviewBatch { pages: out_pages })
    }
}