use rust_transformers_gpt2::tokenizer::GPT2Tokenizer;
use rust_transformers_gpt2::config::GPT2Config; // Added for GPT2Config

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting tokenizer test...");

    // Define paths to tokenizer files
    let vocab_path = "resources/tokenizer_data/gpt2/gpt2-vocab.json";
    let merges_path = "resources/tokenizer_data/gpt2/merges.txt";

    // Create a new GPT2Tokenizer instance
    println!("Loading tokenizer from: {} and {}", vocab_path, merges_path);
    let tokenizer = match GPT2Tokenizer::new(vocab_path, merges_path) {
        Ok(tk) => tk,
        Err(e) => {
            eprintln!("Failed to create tokenizer: {}", e);
            if !std::path::Path::new(vocab_path).exists() {
                eprintln!("Error: vocab_path '{}' does not exist. Current dir: {:?}", vocab_path, std::env::current_dir()?);
            }
            if !std::path::Path::new(merges_path).exists() {
                eprintln!("Error: merges_path '{}' does not exist. Current dir: {:?}", merges_path, std::env::current_dir()?);
            }
            return Err(e);
        }
    };
    println!("Tokenizer loaded successfully.");

    // --- Test encode and decode ---
    let sample_text = "Hello, world!";
    println!("\n--- Testing encode/decode ---");
    println!("Encoding text: '{}'", sample_text);

    let encoded_ids = tokenizer.encode(sample_text, true)?;
    println!("Encoded IDs: {:?}", encoded_ids);

    println!("Decoding IDs: {:?}", encoded_ids);
    let decoded_text = tokenizer.decode(&encoded_ids, true)?;
    println!("Decoded text: '{}'", decoded_text);
    println!("Encode/decode test completed.");

    // --- Test load_pre_tokenized_from_json ---
    println!("\n--- Testing load_pre_tokenized_from_json ---");
    let json_path = "resources/tokenizer_data/gpt2/sample_token_ids.json";
    println!("Loading pre-tokenized IDs from: '{}'", json_path);

    if !std::path::Path::new(json_path).exists() {
        eprintln!("Error: JSON file '{}' does not exist. Current dir: {:?}", json_path, std::env::current_dir()?);
        return Err(format!("JSON file not found: {}", json_path).into());
    }
    
    let loaded_ids = tokenizer.load_pre_tokenized_from_json(json_path)?;
    println!("Loaded IDs from JSON: {:?}", loaded_ids);

    let expected_ids: &[u32] = &[15496, 11, 6894, 0];
    assert_eq!(&loaded_ids, expected_ids, "Loaded IDs do not match expected IDs!");
    println!("Assertion successful: Loaded IDs match expected IDs.");

    println!("Decoding loaded IDs: {:?}", loaded_ids);
    let decoded_text_from_json = tokenizer.decode(&loaded_ids, true)?;
    println!("Decoded text from JSON-loaded IDs: '{}'", decoded_text_from_json);
    println!("Load pre-tokenized test completed.");

    println!("\nAll tokenizer tests completed successfully.");

    // --- Test GPT2Config loading ---
    println!("\n--- Testing GPT2Config loading ---");
    let config_path = "resources/model_config/gpt2/config.json";
    println!("Loading GPT2Config from: '{}'", config_path);

    if !std::path::Path::new(config_path).exists() {
        eprintln!("Error: Config file '{}' does not exist. Current dir: {:?}", config_path, std::env::current_dir()?);
        return Err(format!("Config file not found: {}", config_path).into());
    }

    let gpt2_config = match GPT2Config::load(config_path) {
        Ok(config) => config,
        Err(e) => {
            eprintln!("Failed to load GPT2Config: {}", e);
            return Err(e);
        }
    };
    println!("GPT2Config loaded successfully.");
    println!("Loaded config: {:?}", gpt2_config);
    println!("Config loading test completed.");
    
    println!("\nAll tests completed successfully.");

    // --- Token Generation Setup ---
    // Constants and configurations are now part of the function or passed as arguments.
    const EOS_TOKEN_ID_MAIN: u32 = 50256; // Placeholder EOS token ID for main execution
    let initial_prompt_tokens_main: Vec<u32> = vec![0]; 
    let max_new_tokens_main = 10;
    let theta_hat_main = 0.0f32;

    let model_config_path = "resources/model_config/gpt2/config.json";
    let model_config = GPT2Config::load(model_config_path)
        .expect("Failed to load model config for token generation loop");
    
    // Re-use model_config and model from GPT2Config loading test.
    // Ensure model is mutable for run_repl_loop.
    let mut model = crate::model::GPT2Model::new(&gpt2_config) // Use gpt2_config loaded earlier
        .expect("Failed to create a GPT2Model instance for REPL execution");

    // --- REPL Setup ---
    println!("\n--- Starting Interactive REPL ---");
    // Import REPL functions
    use rust_transformers_gpt2::repl::{run_repl_loop, get_user_prompt};

    // Get initial prompt from user
    let prompt_string = get_user_prompt();
    if prompt_string.is_empty() {
        println!("Prompt is empty. Exiting REPL.");
        return Ok(());
    }

    // Tokenize the prompt
    let initial_prompt_tokens_repl = tokenizer.encode(&prompt_string, true)
        .map_err(|e| format!("Failed to encode prompt string: {}", e))?;
    
    if initial_prompt_tokens_repl.is_empty() {
        println!("Tokenized prompt is empty (e.g. input was only special tokens not kept). Exiting REPL.");
        return Ok(());
    }

    // REPL parameters
    let max_new_tokens_repl = 10; // Max tokens to generate in one REPL session
    let initial_theta_hat_repl = 0.2f32; // Initial theta_hat value
    let eos_token_id_repl = gpt2_config.eos_token_id.unwrap_or(50256); // Use EOS from config or default

    // Call the REPL loop
    if let Err(e) = run_repl_loop(
        &mut model,
        &gpt2_config, // Use gpt2_config loaded earlier
        &tokenizer,
        initial_prompt_tokens_repl,
        max_new_tokens_repl,
        initial_theta_hat_repl,
        eos_token_id_repl,
    ) {
        eprintln!("REPL loop exited with error: {}", e);
    }
    
    println!("--- REPL Ended ---");

    Ok(())
}

// Extracted Token Generation Logic (generate_tokens_fn) - Commented out as REPL is primary now
/*
use crate::common::ModelKVCache;
use crate::model::GPT2Model;
use ndarray::Array2; 

pub fn generate_tokens_fn(
    // ... (implementation was here) ...
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    // ... (implementation was here) ...
    unimplemented!("generate_tokens_fn is currently commented out in favor of REPL loop.");
}
*/

#[cfg(test)]
mod tests {
    // Comment out the test for the old generate_tokens_fn
    /*
    use super::*; 
    use crate::config::GPT2Config;
    use crate::model::GPT2Model;

    #[test]
    fn test_token_generation_loop_basic() {
        // ... (test implementation was here) ...
    }
    */
}