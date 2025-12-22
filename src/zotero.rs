use rusqlite::{Connection, OpenFlags};
use serde::{Deserialize, Serialize};
use std::env;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoteroItem {
    pub item_id: i64,
    pub key: String,
    pub title: Option<String>,
    pub authors: Option<String>,
    pub date: Option<String>,
    pub abstract_note: Option<String>,
    pub pdf_path: Option<String>,
}

pub struct ZoteroReader {
    db_path: PathBuf,
    storage_dir: PathBuf,
}

impl ZoteroReader {
    /// Try to find the Zotero database path.
    /// Priority:
    /// 1. ZOTERO_DB_PATH environment variable (if set)
    /// 2. Default location: ~/Zotero/zotero.sqlite
    fn find_db_path() -> anyhow::Result<PathBuf> {
        // First, check if ZOTERO_DB_PATH is explicitly set
        if let Ok(db_path_str) = env::var("ZOTERO_DB_PATH") {
            let db_path = PathBuf::from(&db_path_str);
            if db_path.exists() {
                return Ok(db_path);
            }
        }

        if let Some(home_dir) = dirs::home_dir() {
            let default_path = home_dir.join("Zotero").join("zotero.sqlite");
            if default_path.exists() {
                return Ok(default_path);
            }
        }

        Err(anyhow::anyhow!(
            "Could not find Zotero database. Either set ZOTERO_DB_PATH environment variable or ensure Zotero is installed at the default location (~/Zotero/zotero.sqlite)"
        ))
    }

    pub fn new() -> anyhow::Result<Self> {
        let db_path = Self::find_db_path()?;

        // Storage dir is in the same parent directory as the DB, under "storage"
        let storage_dir = db_path.parent().unwrap().join("storage");

        Ok(Self {
            db_path,
            storage_dir,
        })
    }

    pub fn get_items(&self) -> anyhow::Result<Vec<ZoteroItem>> {
        // Open in read-only mode
        let conn = Connection::open_with_flags(
            &self.db_path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
        )?;

        // Query to get items. We'll join to get basic metadata.
        // Zotero field IDs: 1=title, 2=abstractNote, 14=date (publication date)
        let mut stmt = conn.prepare(
            "SELECT 
                i.itemID,
                i.key,
                title_val.value as title,
                abstract_val.value as abstract,
                date_val.value as date,
                (
                    SELECT GROUP_CONCAT(
                        CASE 
                            WHEN c.firstName IS NOT NULL AND c.lastName IS NOT NULL 
                            THEN c.lastName || ', ' || c.firstName
                            WHEN c.lastName IS NOT NULL 
                            THEN c.lastName
                            ELSE NULL
                        END, '; '
                    )
                    FROM itemCreators ic
                    JOIN creators c ON ic.creatorID = c.creatorID
                    WHERE ic.itemID = i.itemID
                ) as authors
            FROM items i
            JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
            -- Get title (fieldID = 1)
            LEFT JOIN itemData title_data ON i.itemID = title_data.itemID AND title_data.fieldID = 1
            LEFT JOIN itemDataValues title_val ON title_data.valueID = title_val.valueID
            -- Get abstract (fieldID = 2)
            LEFT JOIN itemData abstract_data ON i.itemID = abstract_data.itemID AND abstract_data.fieldID = 2
            LEFT JOIN itemDataValues abstract_val ON abstract_data.valueID = abstract_val.valueID
            -- Get publication date (fieldID = 14)
            LEFT JOIN itemData date_data ON i.itemID = date_data.itemID AND date_data.fieldID = 14
            LEFT JOIN itemDataValues date_val ON date_data.valueID = date_val.valueID
            
            WHERE it.typeName NOT IN ('attachment', 'note', 'annotation')
            ORDER BY i.dateModified DESC"
        )?;

        let item_iter = stmt.query_map([], |row| {
            Ok(ZoteroItem {
                item_id: row.get(0)?,
                key: row.get(1)?,
                title: row.get(2).ok(),
                abstract_note: row.get(3).ok(),
                date: row.get(4).ok(),
                authors: row.get(5).ok(),
                pdf_path: None, // Will be filled later
            })
        })?;

        let mut items = Vec::new();
        for item in item_iter {
            items.push(item?);
        }

        // Now find attachments for each item
        // We can do this in a separate query or loop.
        // For efficiency, maybe we query all PDF attachments and map them.
        let mut attachment_stmt = conn.prepare(
            "SELECT ia.parentItemID, ia.path, att.key 
             FROM itemAttachments ia
             JOIN items att ON att.itemID = ia.itemID
             WHERE ia.contentType = 'application/pdf'",
        )?;

        let attachment_iter = attachment_stmt.query_map([], |row| {
            Ok((
                row.get::<_, Option<i64>>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?;

        let mut pdf_map = std::collections::HashMap::new();
        for res in attachment_iter {
            let (parent_id_opt, path_opt, key) = res?;
            let Some(parent_id) = parent_id_opt else {
                // Zotero can contain standalone/linked attachments without a parent item.
                // Those cannot be associated to a bibliographic item, so we skip them.
                continue;
            };
            if let Some(path_str) = path_opt {
                pdf_map.insert(parent_id, (path_str, key));
            }
        }

        for item in &mut items {
            if let Some((path_str, key)) = pdf_map.get(&item.item_id) {
                // Resolve path
                if path_str.starts_with("storage:") {
                    let rel_path = path_str.trim_start_matches("storage:");
                    let full_path = self.storage_dir.join(key).join(rel_path);
                    if full_path.exists() {
                        item.pdf_path = Some(full_path.to_string_lossy().to_string());
                    }
                } else {
                    // Absolute path?
                    let p = PathBuf::from(path_str);
                    if p.exists() {
                        item.pdf_path = Some(path_str.clone());
                    }
                }
            }
        }

        Ok(items)
    }
}
