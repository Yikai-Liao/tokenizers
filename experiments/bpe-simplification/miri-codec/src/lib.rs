extern crate self as tk_encode;
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
#[path = "../../../../tokenizers/tk-train/src/trainers/bpe/engine/positions.rs"]
mod positions;
