use std::io::{self, Write};
use std::error::Error; // For Box<dyn Error>
use std::path::PathBuf; // For feedback store path

// Module for feedback mechanism
mod repl_feedback;
use repl_feedback::{ResonanceFeedbackStore, ExperienceEntry};

use std::path::Path; // For tokenizer path

// Module for feedback mechanism
mod repl_feedback;
use repl_feedback::{ResonanceFeedbackStore, ExperienceEntry};

// Module for tokenizer
mod tokenizer;
use tokenizer::TokenizerWrapper; // Use the new TokenizerWrapper

// Imports for token generation logic
use ndarray::{ArrayD, Array2}; // s is not used yet, but Array2 is crucial
use crate::model::{GPT2Model, GPT2Config};
use crate::common::ModelKVCache;
// Removed: use crate::tokenizer::GPT2Tokenizer; 

pub fn get_user_prompt() -> String {
    print!("Enter your prompt: ");
    io::stdout().flush().expect("Failed to flush stdout.");
    let mut buffer = String::new();
    io::stdin().read_line(&mut buffer).expect("Failed to read line from stdin.");
    buffer.trim().to_string()
}

pub fn prefill_prompt(
    model: &mut GPT2Model,
    _config: &GPT2Config,
    prompt_tokens: &[u32],
    model_cache: &mut ModelKVCache,
    theta_hat: f32,
) -> Result<ArrayD<f32>, String> {
    if prompt_tokens.is_empty() {
        return Err("Prompt tokens cannot be empty for prefill.".to_string());
    }
    let tokens_array_vec: Vec<i32> = prompt_tokens.iter().map(|&x| x as i32).collect();
    let input_array = Array2::from_shape_vec((1, tokens_array_vec.len()), tokens_array_vec)
        .map_err(|e| format!("Failed to create Array2 from prompt_tokens: {}", e))?;
    model.forward(&input_array, model_cache, theta_hat)
        .map_err(|e| format!("Model forward pass failed during prefill: {}", e))
}

pub fn generate_next_token(
    model: &mut GPT2Model,
    config: &GPT2Config,
    last_token_id: u32,
    model_cache: &mut ModelKVCache,
    theta_hat: f32,
) -> Result<(u32, ArrayD<f32>), String> {
    let token_array = Array2::from_shape_vec((1, 1), vec![last_token_id as i32])
        .map_err(|e| format!("Failed to create Array2 from last_token_id: {}", e))?;
    let model_output = model.forward(&token_array, model_cache, theta_hat)
        .map_err(|e| format!("Model forward pass failed during next token generation: {}", e))?;
    
    let vocab_size = config.vocab_size as usize;
    let dummy_logits_vec: Vec<f32> = (0..vocab_size).map(|idx| idx as f32).collect();
    if dummy_logits_vec.is_empty() {
        return Err("Dummy logits vector was empty (vocab_size might be 0).".to_string());
    }
    let next_token_id = dummy_logits_vec.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as u32)
        .ok_or_else(|| "Failed to determine next token from dummy logits.".to_string())?;
    Ok((next_token_id, model_output))
}

fn get_greedy_token_from_arrayd(
    _logits_array: &ArrayD<f32>, 
    vocab_size: usize,
) -> Result<u32, String> {
    let dummy_logits_vec: Vec<f32> = (0..vocab_size).map(|idx| idx as f32).collect();
    if dummy_logits_vec.is_empty() {
        return Err("Dummy logits vector was empty in get_greedy_token (vocab_size might be 0).".to_string());
    }
    dummy_logits_vec.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as u32)
        .ok_or_else(|| "Failed to determine token from dummy logits in get_greedy_token.".to_string())
}

/// Adjusts theta_hat based on user feedback and clamps it to the [0.0, 1.0] range.
pub(crate) fn adjust_theta_hat(initial_theta: f32, user_feedback: &str) -> f32 {
    let mut new_theta = initial_theta;
    match user_feedback {
        "s" => new_theta += 0.05,
        "n" => new_theta -= 0.05,
        _ => {} // No change for other inputs, including "q" or invalid
    }
    new_theta.max(0.0).min(1.0)
}

#[allow(clippy::too_many_arguments)] 
pub fn run_repl_loop(
    model: &mut GPT2Model,
    config: &GPT2Config,
    tokenizer: &TokenizerWrapper, // Changed from GPT2Tokenizer to TokenizerWrapper
    initial_prompt_tokens: Vec<u32>, 
    max_new_tokens: usize,
    initial_theta_hat: f32,
    eos_token_id: u32,
    store: &mut ResonanceFeedbackStore, // New parameter for feedback store
) -> Result<(), String> {
    if initial_prompt_tokens.is_empty() {
        return Err("Initial prompt tokens cannot be empty for REPL loop.".to_string());
    }
    let mut model_cache: ModelKVCache = vec![Vec::new(); config.n_layer as usize];
    let mut current_token_ids = initial_prompt_tokens.clone();
    let mut current_theta_hat = initial_theta_hat;

    println!("> prompt: (tokens) {:?}", initial_prompt_tokens);
    
    let prefill_output_array = prefill_prompt(
        model, config, &current_token_ids, &mut model_cache, current_theta_hat
    )?;
    let mut next_token_id = get_greedy_token_from_arrayd(
        &prefill_output_array, config.vocab_size as usize
    )?;
    current_token_ids.push(next_token_id);
    
    for i in 0..max_new_tokens {
        let token_display_id = next_token_id;
        print!("Validate token {}: (s)atisfied, (n)ot satisfied, (q)uit? ", token_display_id);
        io::stdout().flush().expect("Failed to flush stdout.");
        let mut user_feedback = String::new();
        io::stdin().read_line(&mut user_feedback).expect("Failed to read user feedback.");
        user_feedback = user_feedback.trim().to_lowercase();

        let pre_adjustment_theta = current_theta_hat; 
        let mut validation_status_str: String; 

        // Decode the token for display before asking for validation
        let token_str_display = tokenizer.decode(&[token_display_id], true) // true = skip special tokens
            .unwrap_or_else(|e| {
                eprintln!("\n[Warning: Failed to decode token ID {}: {}]", token_display_id, e);
                format!("[ID:{}]", token_display_id) // Fallback
            });

        if user_feedback == "q" {
            println!("Quitting generation loop.");
            validation_status_str = "quit".to_string();
            println!(
                "⟳ [{}] token: \"{}\" (ID: {}) | θ̂: {:.2} | validado: {}",
                i, token_str_display, token_display_id, pre_adjustment_theta, validation_status_str
            );
            break;
        }
        
        current_theta_hat = adjust_theta_hat(current_theta_hat, &user_feedback);

        match user_feedback.as_str() {
            "s" | "n" => {
                let validation_bool = user_feedback == "s";
                validation_status_str = if validation_bool { "sí".to_string() } else { "no".to_string() };

                let context_tokens_for_exp = if current_token_ids.len() > 1 {
                    current_token_ids[..current_token_ids.len()-1].to_vec()
                } else {
                    Vec::new() 
                };

                let entry = ExperienceEntry {
                    prompt_tokens: context_tokens_for_exp,
                    generated_token_id: token_display_id, 
                    validation_status: validation_bool,
                    theta_hat_at_generation: current_theta_hat, 
                };
                store.add_experience(entry);
                if let Err(e) = store.save() {
                    eprintln!("\n[Error saving experience: {}]", e);
                }
            }
            _ => { 
                validation_status_str = "n/a".to_string();
            }
        }
        
        println!(
            "⟳ [{}] token: \"{}\" (ID: {}) | θ̂: {:.2} | validado: {}",
            i, token_str_display, token_display_id, current_theta_hat, validation_status_str
        );

        if token_display_id == eos_token_id {
            println!("EOS token ({}) encountered. Stopping generation.", eos_token_id);
            break;
        }
        
        if i < max_new_tokens - 1 {
            let last_token_id_for_gen = token_display_id;
            let (generated_token_id, _generated_output_array) = generate_next_token(
                model, config, last_token_id_for_gen, &mut model_cache, current_theta_hat
            )?;
            next_token_id = generated_token_id;
            current_token_ids.push(next_token_id);
        }
    }

    println!("\n--- Generation Ended ---");
    println!("Final generated token IDs: {:?}", current_token_ids);
    let final_decoded_text = tokenizer.decode(&current_token_ids, true)
        .unwrap_or_else(|e| format!("Failed to decode final sequence: {}", e));
    println!("Final decoded text: {}", final_decoded_text);
    Ok(())
}

// Main function for standalone REPL execution
pub fn main() -> Result<(), Box<dyn Error>> {
    println!("--- Standalone REPL Initializing ---");

    // 1. Initialize GPT2Config (default)
    let config = GPT2Config::default(); // Using default for simplicity here
    println!("Default GPT2Config loaded: {:?}", config);

    // 2. Initialize GPT2Model
    let mut model = GPT2Model::new(&config)
        .map_err(|e| format!("Failed to create GPT2Model: {}", e))?;
    println!("GPT2Model initialized.");

    // 3. Initialize TokenizerWrapper
    // Using test_tokenizer.json as it's known to exist and be compatible.
    // Users should replace "test_tokenizer.json" with their actual HuggingFace tokenizer.json path.
    let tokenizer_path_str = "test_tokenizer.json"; 
    let tokenizer_path = Path::new(tokenizer_path_str);
    let tokenizer = TokenizerWrapper::new(tokenizer_path)
        .map_err(|e| format!("Failed to load TokenizerWrapper from '{}': {}. Ensure the file exists and is a valid HuggingFace tokenizer JSON.", tokenizer_path_str, e))?;
    println!("TokenizerWrapper initialized from '{}'. Vocab size: {}", tokenizer_path_str, tokenizer.get_vocab_size());

    // 4. Get User Prompt
    let prompt_string = get_user_prompt();
    if prompt_string.is_empty() {
        println!("Prompt is empty. Exiting.");
        return Ok(());
    }

    // 5. Convert Prompt to Tokens using TokenizerWrapper
    let initial_prompt_tokens = match tokenizer.encode(&prompt_string, false) { // false = don't add special tokens for prompt
        Ok(ids) => ids,
        Err(e) => {
            eprintln!("Failed to encode prompt: {}. Exiting.", e);
            return Ok(()); // Return Ok here as main expects Result<(), Box<dyn Error>>
        }
    };
    
    if initial_prompt_tokens.is_empty() {
        println!("Tokenized prompt is empty (e.g., input was only special tokens not kept, or tokenizer issue). Exiting.");
        return Ok(());
    }
    println!("Successfully tokenized prompt: {:?}", initial_prompt_tokens);

    // 6. Set REPL parameters
    let max_new_tokens = 10;
    // let initial_theta_hat = 0.75_f32; // Default will be set after prediction attempt
    let eos_token_id = config.eos_token_id.unwrap_or(50256); // Use EOS from config or default

    // 7. Initialize ResonanceFeedbackStore
    let feedback_store_path = PathBuf::from("resonance_feedback.json");
    let mut store = ResonanceFeedbackStore::new(feedback_store_path);
    println!("ResonanceFeedbackStore initialized. Loaded {} existing experiences.", store.experiences.len());

    // Predict initial_theta_hat based on experiences
    let mut initial_theta_hat: f32; // Declare, will be set below
    let default_initial_theta = 0.75_f32; // Define default

    match store.predict_initial_theta(&initial_prompt_tokens) {
        Some(predicted_theta) => {
            initial_theta_hat = predicted_theta;
            println!("Intuition from past experiences suggests an initial theta_hat of: {:.2}", initial_theta_hat);
        }
        None => {
            initial_theta_hat = default_initial_theta;
            println!("No similar past experiences found to predict initial theta_hat. Starting with default: {:.2}", initial_theta_hat);
        }
    }
    
    println!("Starting REPL loop with max_new_tokens={}, initial_theta_hat={:.2}, eos_token_id={}", 
        max_new_tokens, initial_theta_hat, eos_token_id);


    // 8. Call run_repl_loop
    // The run_repl_loop returns Result<(), String>, convert its error to Box<dyn Error>
    run_repl_loop(
        &mut model, 
        &config, 
        &tokenizer, 
        initial_prompt_tokens, 
        max_new_tokens, 
        initial_theta_hat, 
        eos_token_id,
        &mut store, // Pass the feedback store
    ).map_err(|e| -> Box<dyn Error> { Box::from(e) })?;

    println!("--- Standalone REPL Finished ---");
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
    // use crate::tokenizer::GPT2Tokenizer; // No longer used directly in tests here
    // TokenizerWrapper might be needed if we had a test for run_repl_loop that needed specific tokenizer interactions

    #[test]
    #[ignore] 
    fn test_get_user_prompt_manual_input() {
        // ... (existing test code) ...
    }

    #[test]
    fn test_get_user_prompt_exists() {
        // ... (existing test code) ...
    }

    #[cfg(test)]
    mod repl_logic_tests {
        use super::*;
        use crate::model::{GPT2Model, GPT2Config};
        use crate::common::ModelKVCache;

        fn create_test_config() -> GPT2Config {
            GPT2Config {
                vocab_size: 50257, n_layer: 2, n_head: 2, n_embd: 128,
                n_positions: 1024, n_ctx: 1024, block_size: 1024,
                embd_pdrop: 0.1, resid_pdrop: 0.1, attn_pdrop: 0.1,
                layer_norm_epsilon: 1e-5, initializer_range: 0.02,
                n_inner: None, activation_function: "gelu_new".to_string(),
                bos_token_id: Some(50256), eos_token_id: Some(50256),
                scale_attn_weights: true, scale_attn_by_inverse_layer_idx: false,
                reorder_and_upcast_attn: false,
            }
        }

        #[test]
        fn test_prefill_prompt_basic() {
            let config = create_test_config();
            let mut model = GPT2Model::new(&config).expect("Failed to create test model.");
            let mut cache: ModelKVCache = vec![Vec::new(); config.n_layer as usize];
            let prompt_tokens: Vec<u32> = vec![0, 1, 2];
            let theta = 0.0f32;
            let result = prefill_prompt(&mut model, &config, &prompt_tokens, &mut cache, theta);
            assert!(result.is_ok());
            let output_array = result.unwrap();
            let expected_shape: Vec<usize> = vec![1, prompt_tokens.len(), config.n_embd as usize];
            assert_eq!(output_array.shape(), expected_shape.as_slice());
        }

        #[test]
        fn test_prefill_prompt_empty_tokens() {
            let config = create_test_config();
            let mut model = GPT2Model::new(&config).expect("Failed to create test model.");
            let mut cache: ModelKVCache = vec![Vec::new(); config.n_layer as usize];
            let prompt_tokens: Vec<u32> = vec![];
            let theta = 0.0f32;
            let result = prefill_prompt(&mut model, &config, &prompt_tokens, &mut cache, theta);
            assert!(result.is_err());
        }

        #[test]
        fn test_generate_next_token_basic() {
            let config = create_test_config();
            let mut model = GPT2Model::new(&config).expect("Failed to create test model.");
            let mut cache: ModelKVCache = vec![Vec::new(); config.n_layer as usize];
            let last_token_id: u32 = 0;
            let theta = 0.0f32;
            let result = generate_next_token(&mut model, &config, last_token_id, &mut cache, theta);
            assert!(result.is_ok());
            let (next_token, output_array) = result.unwrap();
            assert_eq!(next_token, 0);
            let expected_shape: Vec<usize> = vec![1, 1, config.n_embd as usize];
            assert_eq!(output_array.shape(), expected_shape.as_slice());
        }
        
        #[test]
        fn test_get_greedy_token_from_arrayd_basic() {
            let dummy_array = ArrayD::zeros(IxDyn(&[1,1,128]));
            let vocab_size = 50257;
            let token_id = get_greedy_token_from_arrayd(&dummy_array, vocab_size).unwrap();
            assert_eq!(token_id, 0); 
            let vocab_size_small = 10;
            let token_id_small = get_greedy_token_from_arrayd(&dummy_array, vocab_size_small).unwrap();
            assert_eq!(token_id_small, 0);
        }

        #[test]
        fn test_get_greedy_token_from_arrayd_empty_vocab() {
            let dummy_array = ArrayD::zeros(IxDyn(&[1,1,128]));
            let vocab_size = 0;
            let result = get_greedy_token_from_arrayd(&dummy_array, vocab_size);
            assert!(result.is_err());
        }

        #[test]
        #[ignore] 
        fn test_run_repl_loop_no_input_smoke_test() {
            // ... (This test requires specific setup or manual input, kept ignored) ...
        }

        #[test]
        fn test_adjust_theta_hat_logic() {
            // Test 's' (satisfied)
            assert_eq!(adjust_theta_hat(0.5, "s"), 0.55);
            
            // Test 'n' (not satisfied)
            assert_eq!(adjust_theta_hat(0.5, "n"), 0.45);

            // Test clamping (upper bound from 's')
            // Using f32::abs for float comparisons due to potential precision issues.
            // A small epsilon is often used, but for increments of 0.05, direct comparison should be mostly fine.
            // Let's be explicit with a small tolerance if needed, or ensure results are exact.
            assert!((adjust_theta_hat(0.98, "s") - 1.0).abs() < f32::EPSILON);
            assert!((adjust_theta_hat(0.97, "s") - 1.0).abs() < f32::EPSILON); // 0.97 + 0.05 = 1.02 -> 1.0

            // Test clamping (lower bound from 'n')
            assert!((adjust_theta_hat(0.02, "n") - 0.0).abs() < f32::EPSILON);
            assert!((adjust_theta_hat(0.03, "n") - 0.0).abs() < f32::EPSILON); // 0.03 - 0.05 = -0.02 -> 0.0


            // Test invalid input char (should not change theta)
            assert_eq!(adjust_theta_hat(0.5, "x"), 0.5);
            assert_eq!(adjust_theta_hat(0.5, ""), 0.5); // Empty feedback

            // Test 's' at max
            assert!((adjust_theta_hat(1.0, "s") - 1.0).abs() < f32::EPSILON);

            // Test 'n' at min
            assert!((adjust_theta_hat(0.0, "n") - 0.0).abs() < f32::EPSILON);
            
            // Test 's' that would exceed 1.0
            assert!((adjust_theta_hat(0.99, "s") - 1.0).abs() < f32::EPSILON); // 0.99 + 0.05 = 1.04 -> 1.0

            // Test 'n' that would go below 0.0
            assert!((adjust_theta_hat(0.01, "n") - 0.0).abs() < f32::EPSILON); // 0.01 - 0.05 = -0.04 -> 0.0
        }
    }
}
