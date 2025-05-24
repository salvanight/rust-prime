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
    
    let mut model = crate::model::GPT2Model::new(&model_config)
        .expect("Failed to create a GPT2Model instance for main execution");

    println!("\n--- Starting Token Generation (from main) ---");
    match generate_tokens_fn(
        &mut model, 
        &model_config, 
        initial_prompt_tokens_main.clone(), 
        max_new_tokens_main, 
        EOS_TOKEN_ID_MAIN, 
        theta_hat_main
    ) {
        Ok(generated_ids) => {
            println!("Generated token IDs (from main): {:?}", generated_ids);
            // Optional: Decode and print the full generated sequence
            // let final_text = tokenizer.decode(&generated_ids, true)?;
            // println!("Final generated text (from main): {}", final_text);
        }
        Err(e) => {
            eprintln!("Token generation failed (from main): {}", e);
        }
    }
    println!("--- Token Generation (from main) Ended ---");

    Ok(())
}

// Extracted Token Generation Logic
use crate::common::ModelKVCache;
use crate::model::GPT2Model;
use ndarray::Array2; // For Array2

pub fn generate_tokens_fn(
    model: &mut GPT2Model,
    model_config: &GPT2Config, // Pass config for vocab_size, n_layer
    initial_prompt_tokens: Vec<u32>,
    max_new_tokens: usize,
    eos_token_id: u32,
    theta_hat: f32,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let mut current_token_ids = initial_prompt_tokens.clone();
    let mut model_cache: ModelKVCache = vec![Vec::new(); model_config.n_layer as usize];

    println!("Initial tokens (fn): {:?}", current_token_ids);

    for i in 0..max_new_tokens {
        let tokens_for_model_vec = if i == 0 {
            current_token_ids.clone()
        } else {
            vec![*current_token_ids.last().unwrap()]
        };

        // Convert Vec<u32> to Array2<i32> for model.forward
        // Assuming batch_size = 1 for generation.
        // Model expects i32, so cast u32 to i32. This might lose info if token IDs are > i32::MAX.
        let tokens_for_model_arr: Vec<i32> = tokens_for_model_vec.iter().map(|&x| x as i32).collect();
        let input_array = Array2::from_shape_vec((1, tokens_for_model_arr.len()), tokens_for_model_arr)?;
        
        println!("Iteration {} (fn): Tokens for model (shape {:?}): {:?}", i, input_array.dim(), tokens_for_model_vec);

        let model_output = model.forward(&input_array, &mut model_cache, theta_hat)
            .map_err(|e| format!("Model forward pass failed: {}", e))?; // More specific error mapping

        // Placeholder for logit extraction from model_output (ArrayD<f32>)
        // Assuming model_output is [batch_size, seq_len, vocab_size]
        // We need the logits for the last token: output.slice(s![0, -1, ..])
        // For now, using dummy logits as before.
        let vocab_size = model_config.vocab_size as usize;
        
        // This is a placeholder. In reality, you'd get this from model_output.
        // e.g. by taking the slice corresponding to the last token's logits.
        // let last_token_logits_view = model_output.slice(s![0, -1, ..]);
        // let logits: Vec<f32> = last_token_logits_view.to_vec();
        // For testing, if model_output is just hidden states, its last dim might not be vocab_size.
        // The dummy logits ensure the loop runs.
        let logits: Vec<f32> = (0..vocab_size).map(|idx| idx as f32).collect();


        if logits.is_empty() { // Should not happen with dummy logits
            return Err("Logits vector was empty.".into());
        }

        let next_token_id = logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)) // Handle NaN safely
            .map(|(idx, _)| idx as u32)
            .ok_or("Failed to determine next token from logits.")?; // Handle case where max_by returns None

        println!("Iteration {} (fn): Next token ID: {}", i, next_token_id);

        if next_token_id == eos_token_id {
            println!("EOS token encountered (fn). Stopping generation.");
            break;
        }
        current_token_ids.push(next_token_id);
    }

    println!("Generated token IDs (fn): {:?}", current_token_ids);
    Ok(current_token_ids)
}


#[cfg(test)]
mod tests {
    use super::*; // To import generate_tokens_fn
    use crate::config::GPT2Config;
    use crate::model::GPT2Model;
    // ModelKVCache is not directly used by the test function itself, but by generate_tokens_fn

    #[test]
    fn test_token_generation_loop_basic() {
        // 1. Setup configuration
        // Using a default config or loading a minimal one.
        // For this test, many specific config values don't matter as the model.forward() is a placeholder.
        let config = GPT2Config {
            vocab_size: 50257, // Standard for GPT-2
            n_layer: 2,        // Minimal number of layers
            n_head: 2,         // Minimal number of heads
            n_embd: 128,       // Minimal embedding size
            n_positions: 1024, // Max sequence length
            n_ctx: 1024,
            block_size: 1024,  // Often same as n_positions/n_ctx
            embd_pdrop: 0.1,
            resid_pdrop: 0.1,
            attn_pdrop: 0.1,
            layer_norm_epsilon: 1e-5,
            initializer_range: 0.02,
            n_inner: None,     // Will default in MLP::new if needed
            activation_function: "gelu_new".to_string(), // Example
            bos_token_id: Some(50256),
            eos_token_id: Some(50256),
            scale_attn_weights: true,
            scale_attn_by_inverse_layer_idx: false,
            reorder_and_upcast_attn: false,
            // Add other fields if GPT2Model::new or its components require them
            // For instance, if there are new fields not covered by a simple default.
        };

        // 2. Create a dummy model
        let mut model = GPT2Model::new(&config).expect("Failed to create dummy GPT2Model for test.");

        // 3. Define test parameters
        let initial_tokens: Vec<u32> = vec![0]; // E.g., BOS token
        let num_new_tokens_to_generate = 5;
        let test_eos_token_id = 50256; // Standard EOS token ID
        let test_theta_hat = 0.0f32;   // Placeholder temperature

        // 4. Call the token generation function
        let result = generate_tokens_fn(
            &mut model,
            &config, // Pass the config
            initial_tokens.clone(),
            num_new_tokens_to_generate,
            test_eos_token_id,
            test_theta_hat,
        );

        // 5. Assertions
        assert!(result.is_ok(), "Token generation failed: {:?}", result.err());
        let generated_ids = result.unwrap();

        // Check that some tokens were generated.
        // If no EOS is hit, it should generate initial_tokens.len() + num_new_tokens_to_generate tokens.
        // If EOS is hit early, it could be less.
        // A basic check: at least the initial tokens should be there.
        assert!(!generated_ids.is_empty(), "Generated token IDs vector should not be empty.");
        assert!(generated_ids.len() >= initial_tokens.len(), "Generated IDs should be at least as long as initial tokens.");

        // If max_new_tokens > 0 and the first generated token is not EOS, 
        // then more than initial_tokens.len() should be present.
        if num_new_tokens_to_generate > 0 {
            // This assertion might be too strict if EOS is generated as the first token.
            // However, with dummy logits (0..vocab_size), token 0 is always picked.
            // If test_eos_token_id is 0, it will stop immediately.
            if test_eos_token_id != 0 || initial_tokens.contains(&0) { // if 0 is not EOS or already in prompt
                 assert!(generated_ids.len() > initial_tokens.len(), "Should generate new tokens if max_new_tokens > 0 and EOS is not immediately hit.");
                 assert_eq!(generated_ids.len(), initial_tokens.len() + num_new_tokens_to_generate, "Expected to generate all requested tokens as EOS is not 0.");
            } else if test_eos_token_id == 0 { // EOS is 0, so it stops after 0 tokens.
                 assert_eq!(generated_ids.len(), initial_tokens.len(), "Should not generate new tokens if EOS is 0 and generated first.");
            }
        }
        println!("Test test_token_generation_loop_basic completed successfully.");
    }
}