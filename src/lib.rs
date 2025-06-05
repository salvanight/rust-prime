use tokenizers::Tokenizer;
use tokenizers::models::bpe::BPE;
use std::path::Path; // Added for path checking
use std::fs::File; // For load_pre_tokenized_from_json
use std::io::Read; // For load_pre_tokenized_from_json

// Main library exports

pub mod config;
#[cfg(feature = "ndarray_backend")]
pub mod common;
#[cfg(feature = "ndarray_backend")]
pub mod accelerator;
#[cfg(feature = "ndarray_backend")]
pub mod attention;
#[cfg(feature = "ndarray_backend")]
pub mod mlp;
#[cfg(feature = "ndarray_backend")]
pub mod model;

pub mod tokenizer; // This will now load src/tokenizer.rs
pub mod cache_tier;
pub mod system_resources;
pub mod repl_feedback;
pub mod native;
#[cfg(feature = "ndarray_backend")]
pub mod ndarray_specific;
pub mod ui;
