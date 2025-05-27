use tokenizers::Tokenizer;
use tokenizers::models::bpe::BPE;
use std::path::Path; // Added for path checking
use std::fs::File; // For load_pre_tokenized_from_json
use std::io::Read; // For load_pre_tokenized_from_json

// This is the main library file.
// We will add functions and structures here later.

pub mod native;
pub mod config;
pub mod cache_tier;
// Conditionally compiled ndarray-specific modules are now under ndarray_specific
#[cfg(feature = "ndarray_backend")]
pub mod ndarray_specific;

// The tokenizer module defined below does not depend on ndarray_backend directly in its own code,
// but its usage might be tied to one backend or the other in main.rs or other parts of the application.
// Based on previous subtasks, it was not moved to ndarray_specific.
// If it's truly common or backend-agnostic, it stays here.
// If its utility is only when ndarray_backend is active, it should also be cfg-guarded.
// For now, assuming it's independent as per previous decisions.
// However, the subtask description said:
// "Remove all the old individual pub mod declarations for attention, common, gating (if it exists), mlp, model, moe, and orchestrator.
// These were previously wrapped with #[cfg(feature = "ndarray_backend")]"
// AND "In src/lib.rs, the declarations for these modules should now point to the ndarray_specific module itself...
// replace all the individual pub mod ndarray_specific::...; lines with a single line:
// #[cfg(feature = "ndarray_backend")] pub mod ndarray_specific;"
// This implies that the tokenizer module, which was previously cfg-guarded, should remain so if it's not part of ndarray_specific.
// The previous turn had `#[cfg(feature = "ndarray_backend")] pub mod tokenizer { ... }`
// If it's meant to be always available, the cfg flag should be removed from it.
// If it's only for ndarray_backend, the cfg flag should remain.
// Given the subtask's focus on refactoring ndarray-specific code, and the previous state of tokenizer,
// I will keep it cfg-guarded for now.
// Correction: Tokenizer was found to be backend-agnostic. Removing cfg flag.
// Re-applying cfg guard as per new instruction.
#[cfg(feature = "ndarray_backend")]
pub mod tokenizer {
    use super::*; // To bring BPE, Tokenizer, Path, File, Read into this module's scope
    // serde_json is already in Cargo.toml, so it can be used here.

    pub struct GPT2Tokenizer {
        tokenizer: Tokenizer,
    }

    impl GPT2Tokenizer {
        pub fn new(vocab_path: &str, merges_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
            // Check if files exist
            if !Path::new(vocab_path).exists() {
                return Err(format!("Vocabulary file not found at: {}", vocab_path).into());
            }
            if !Path::new(merges_path).exists() {
                return Err(format!("Merges file not found at: {}", merges_path).into());
            }

            // Attempt to load BPE model from files
            let bpe_model = BPE::from_file(vocab_path, merges_path)
                .build()
                .map_err(|e| format!("Failed to build BPE model: {}", e))?;

            // Create a Tokenizer instance with the BPE model
            let tokenizer_instance = Tokenizer::new(bpe_model);
            
            Ok(Self { tokenizer: tokenizer_instance })
        }

        pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
            let encoding_result = self.tokenizer.encode(text, add_special_tokens)
                .map_err(|e| format!("Failed to encode text: {}", e))?;
            
            let ids: Vec<u32> = encoding_result.get_ids().iter().map(|&id| id as u32).collect();
            Ok(ids)
        }

        pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, Box<dyn std::error::Error>> {
            // The tokenizer's decode method directly takes &[u32]
            self.tokenizer.decode(ids, skip_special_tokens)
                .map_err(|e| format!("Failed to decode IDs: {}", e).into())
        }

        pub fn load_pre_tokenized_from_json(&self, json_path: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
            if !Path::new(json_path).exists() {
                return Err(format!("JSON file not found at: {}", json_path).into());
            }

            let mut file = File::open(json_path)
                .map_err(|e| format!("Failed to open JSON file {}: {}", json_path, e))?;
            
            let mut contents = String::new();
            file.read_to_string(&mut contents)
                .map_err(|e| format!("Failed to read JSON file {}: {}", json_path, e))?;
            
            let ids: Vec<u32> = serde_json::from_str(&contents)
                .map_err(|e| format!("Failed to deserialize JSON from {}: {}", json_path, e))?;
            
            Ok(ids)
        }
    }
}