use tokenizers::Tokenizer;
use tokenizers::models::bpe::BPE;
use std::path::Path; // Added for path checking
use std::fs::File; // For load_pre_tokenized_from_json
use std::io::Read; // For load_pre_tokenized_from_json

// This is the main library file.
// We will add functions and structures here later.

pub mod config;
pub mod common;
pub mod accelerator;
pub mod attention;
pub mod mlp;
pub mod model;
pub mod tokenizer; // This will now load src/tokenizer.rs
pub mod ui;
