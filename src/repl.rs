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
use ndarray::{s, ArrayD, Array2, Axis, ArrayView1, Array1}; // Added s and Axis, ArrayView1, Array1
use crate::model::{GPT2Model, GPT2Config};
use crate::common::ModelKVCache;
// Removed: use crate::tokenizer::GPT2Tokenizer; 

// Added for MoE Orchestrator
use crate::moe::{MLPExpert, SymbolicExpert}; 
use crate::orchestrator::MoEOrchestrator;
use crate::gating::GatingLayer; 
use crate::cache_tier::CacheTier; 
// use crate::system_resources::SystemResources; 

// Helper function for dynamic tier adjustment
fn determine_and_set_allowed_tiers(
    orchestrator: &mut MoEOrchestrator,
    ram_available_gb: f32,
    theta_hat_value: f32
) {
    if ram_available_gb < 2.0 {
        orchestrator.current_allowed_tiers = vec![CacheTier::L1];
        // println!("[REPL Policy] RAM < 2GB, setting tiers to L1 only");
    } else if theta_hat_value > 0.85 && ram_available_gb > 6.0 {
        orchestrator.current_allowed_tiers = vec![CacheTier::L1, CacheTier::L2, CacheTier::L3];
        // println!("[REPL Policy] High Theta & RAM, setting tiers to L1, L2, L3");
    } else if theta_hat_value < 0.5 {
        orchestrator.current_allowed_tiers = vec![CacheTier::L1];
        // println!("[REPL Policy] Low Theta (Safe Mode), setting tiers to L1 only");
    } else {
        orchestrator.current_allowed_tiers = vec![CacheTier::L1, CacheTier::L2];
        // println!("[REPL Policy] Default, setting tiers to L1, L2");
    }
}

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
) -> Result<ArrayD<f32>, String> { // Changed return type
    let token_array = Array2::from_shape_vec((1, 1), vec![last_token_id as i32])
        .map_err(|e| format!("Failed to create Array2 from last_token_id: {}", e))?;
    // model.forward is assumed to return raw logits (or hidden_states as placeholder)
    model.forward(&token_array, model_cache, theta_hat)
        .map_err(|e| format!("Model forward pass failed during next token generation: {}", e))
}

// Helper to get a 1D view of the last sequence element's logits
fn get_last_token_logits_slice(logits: &ArrayD<f32>) -> Option<ArrayView1<f32>> {
    match logits.ndim() {
        1 => Some(logits.view().into_dimensionality().unwrap()), // Already 1D
        3 => { // Assuming [batch, seq_len, vocab_size]
            if logits.shape()[0] == 0 || logits.shape()[1] == 0 || logits.shape()[2] == 0 {
                None // Empty dimension
            } else {
                Some(logits.slice(s![logits.shape()[0]-1, logits.shape()[1]-1, ..]))
            }
        }
        _ => None, // Unexpected number of dimensions
    }
}

/// Returns the top k tokens and their logit values from a logits array.
/// Logits are assumed to be for a single token position (e.g., last token in a sequence).
pub fn get_top_k_tokens(logits: &ArrayD<f32>, k: usize) -> Vec<(u32, f32)> {
    if k == 0 {
        return Vec::new();
    }

    let logits_slice_option = get_last_token_logits_slice(logits);
    if logits_slice_option.is_none() {
        return Vec::new();
    }
    let logits_slice = logits_slice_option.unwrap();

    if logits_slice.is_empty() {
        return Vec::new();
    }
    
    let mut indexed_logits: Vec<(u32, f32)> = logits_slice.iter()
        .enumerate()
        .map(|(id, &logit_val)| (id as u32, logit_val))
        .collect();

    // Sort by logit value in descending order.
    // Using partial_cmp for f32, handling potential NaNs by treating them as less than regular values.
    indexed_logits.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Less)
    });

    indexed_logits.truncate(k);
    indexed_logits
}


/// Gets the token ID and logit value for the token with the highest logit score.
/// Logits are assumed to be for a single token position.
pub fn get_greedy_token_and_logit(logits: &ArrayD<f32>) -> Result<(u32, f32), String> {
    let top_k = get_top_k_tokens(logits, 1);
    if let Some(result) = top_k.first() {
        Ok(*result)
    } else {
        Err("Failed to get greedy token: Logits processing resulted in empty list or k=0.".to_string())
    }
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

/// Runs the main Read-Eval-Print Loop (REPL) for interactive token generation.
///
/// This loop handles user input, token generation, and user feedback.
/// It now integrates with an `MoEOrchestrator` to dynamically select experts
/// based on cache tier policies, system resources, and intentionality score (θ̂).
///
/// Key operations within the loop:
/// 1.  **Dynamic Tier Policy Adjustment**: Before each token generation, the
///     `MoEOrchestrator`'s `current_allowed_tiers` policy is updated based on
///     available system RAM and the current `theta_hat` value. This logic is
///     encapsulated in the `determine_and_set_allowed_tiers` helper function.
/// 2.  **Token Generation via Orchestrator**: Uses `orchestrator.forward()` to generate
///     the next token's logits. This call also returns information about which
///     experts were activated.
/// 3.  **User Validation**: Prompts the user to validate the generated token ("s"atisfied,
///     "n"ot satisfied, "q"uit).
/// 4.  **Theta Hat Adjustment**: Adjusts `theta_hat` based on user feedback.
/// 5.  **Feedback Storage**: Records the generation experience (prompt context,
///     generated token, validation, theta_hat) to a JSON file.
/// 6.  **Information Display**: Shows the generated token, its ID, logit value,
///     the current `theta_hat`, validation status, and a list of activated experts
///     (name and cache tier) for the generated token.
///
/// # Arguments
/// * `model`: A mutable reference to the `GPT2Model` (used for embedding lookup via `get_embeddings`).
/// * `config`: A reference to the `GPT2Config`.
/// * `tokenizer`: A reference to the `TokenizerWrapper` for encoding/decoding tokens.
/// * `initial_prompt_tokens`: A vector of token IDs for the initial prompt.
/// * `max_new_tokens`: The maximum number of new tokens to generate in this session.
/// * `initial_theta_hat`: The starting value for the intentionality score.
/// * `eos_token_id`: The end-of-sequence token ID to stop generation.
/// * `store`: A mutable reference to the `ResonanceFeedbackStore` for saving experiences.
/// * `orchestrator`: A mutable reference to the `MoEOrchestrator` to be used for expert selection and generation.
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
    orchestrator: &mut MoEOrchestrator, // Added orchestrator
) -> Result<(), String> {
    if initial_prompt_tokens.is_empty() {
        return Err("Initial prompt tokens cannot be empty for REPL loop.".to_string());
    }
    // model_cache is not directly used here anymore, orchestrator might manage its own or model does.
    // For now, we remove direct model_cache management from REPL if orchestrator.forward handles it.
    // Based on orchestrator.forward not taking model_cache, model itself must be handling it with its own state.
    // Let's assume GPT2Model::forward and its sub-components like TransformerBlock manage their own cache state
    // or that orchestrator.forward implicitly passes it if needed.
    // The current GPT2Model::forward takes a ModelKVCache, so we still need it.
    let mut model_cache: ModelKVCache = vec![Vec::new(); config.n_layer as usize]; // Keep if model.forward needs it
    
    let mut current_token_ids = initial_prompt_tokens.clone();
    let mut current_theta_hat = initial_theta_hat;

    println!("> prompt: (tokens) {:?}", initial_prompt_tokens);
    
    // Initial tier determination before prefill
    orchestrator.system_status.refresh(); // Refresh status for fresh RAM info
    let ram_gb_prefill = orchestrator.system_status.ram_available_gb;
    determine_and_set_allowed_tiers(orchestrator, ram_gb_prefill, current_theta_hat);
    
    // Prefill using orchestrator
    let prompt_embeddings = model.get_embeddings(&current_token_ids)?;
    let last_prompt_embedding_slice = prompt_embeddings.slice(s![0, prompt_embeddings.shape()[1] - 1, ..]);
    let input_1d_features_prefill = Array1::from_vec(last_prompt_embedding_slice.to_vec());

    // orchestrator.forward might need the model_cache if experts use it.
    // The current setup has model.forward inside expert.forward in some cases.
    // For now, orchestrator.forward does not take model_cache.
    // This implies that if experts call model.forward, the model's internal state or cache handling needs to be robust.
    // Or, the experts themselves get the cache. This is slightly ambiguous.
    // Let's assume for now that experts *don't* directly interact with ModelKVCache passed at this level.
    // The model itself in GPT2Model::forward takes the cache.
    // If `orchestrator.forward` calls `expert.forward` which *then* calls `model.forward` (unlikely for typical MoE),
    // then `model` would need to be part of the expert or passed to `expert.forward`.
    // The current `Expert` trait's `forward` method does not take `ModelKVCache`.
    // This means the placeholder `model.get_embeddings` and `orchestrator.forward` are the main path.
    // The `model_cache` is thus NOT directly used by `orchestrator.forward`.
    // However, the `GPT2Model::forward` (which *is* called by current Experts) *does* use `ModelKVCache`.
    // This is a contradiction. The prompt for this step implies replacing model.forward calls with orchestrator.forward.
    // The `Expert::forward` methods (MLPExpert, SymbolicExpert) in `src/moe.rs` *do not* call `model.forward`. They are simulated.
    // So, `model_cache` is not used by the `orchestrator.forward` path with current experts.
    // I will remove `model_cache` from the REPL loop if it's not used by the new path.
    // Re-checking `GPT2Model::forward` signature in `src/model.rs`: `fn forward(&mut self, input_ids: &Array2<i32>, model_cache: &mut ModelKVCache, theta_hat: f32)`
    // Re-checking `Expert::forward` in `src/moe.rs`: `fn forward(&self, input: &ArrayD<f32>, theta_hat: f32) -> Result<ArrayD<f32>, String>`
    // The `input` to `Expert::forward` is `full_input_tensor` from `orchestrator.forward`, which are the embeddings.
    // The experts *do not* call `model.forward`. They operate on embeddings.
    // So, `model_cache` is not needed for the `orchestrator.forward` path.
    // I will comment out `model_cache` usage in `run_repl_loop`.

    // let prefill_output_array = prefill_prompt(
    //     model, config, &current_token_ids, &mut model_cache, current_theta_hat
    // )?;
    let (prefill_output_array_logits, activated_experts_info_prefill) = orchestrator.forward(
        &input_1d_features_prefill,
        &prompt_embeddings,
        current_theta_hat
    ).map_err(|e| format!("Orchestrator forward pass failed during prefill: {}", e))?;


    // Use the new get_greedy_token_and_logit function
    // Process prefill output to get the first generated token
    let (first_generated_token_id, first_dominant_logit) = get_greedy_token_and_logit(&prefill_output_array_logits)?;
    let first_top_k_alternatives = get_top_k_tokens(&prefill_output_array_logits, 4); // Get top 4 for up to 3 alternatives
    
    let mut next_token_id = first_generated_token_id;
    current_token_ids.push(next_token_id);

    // Variables to hold current iteration's logit info and expert info for display
    let mut dominant_logit_value_for_display = first_dominant_logit;
    let mut top_k_alternatives_for_display = first_top_k_alternatives;
    let mut experts_info_for_display: Vec<(String, CacheTier)> = activated_experts_info_prefill;
    
    for i in 0..max_new_tokens {
        let token_display_id = next_token_id; // This is the token generated in the previous step (or from prefill)
        
        // Decode the token for display
        let token_str_display = tokenizer.decode(&[token_display_id], true)
            .unwrap_or_else(|e| {
                eprintln!("\n[Warning: Failed to decode token ID {}: {}]", token_display_id, e);
                format!("[ID:{}]", token_display_id)
            });

        // --- User Validation for `token_display_id` ---
        print!("Validate token {}: (s)atisfied, (n)ot satisfied, (q)uit? ", token_display_id);
        io::stdout().flush().expect("Failed to flush stdout.");
        let mut user_feedback = String::new();
        io::stdin().read_line(&mut user_feedback).expect("Failed to read user feedback.");
        user_feedback = user_feedback.trim().to_lowercase();

        let pre_adjustment_theta = current_theta_hat; 
        let mut validation_status_str: String; 

        if user_feedback == "q" {
            println!("Quitting generation loop.");
            validation_status_str = "quit".to_string();
            // Display info for the token that led to quit
            println!(
                "⟳ [{}] token: \"{}\" (ID: {}, Logit: {:.2}) | θ̂: {:.2} | validado: {}",
                i, token_str_display, token_display_id, dominant_logit_value_for_display, pre_adjustment_theta, validation_status_str
            );
            // Display activated experts for the quit token
            if !experts_info_for_display.is_empty() {
                let experts_display_str = experts_info_for_display
                    .iter()
                    .map(|(name, tier)| format!("{} ({:?})", name, tier))
                    .collect::<Vec<String>>()
                    .join(", ");
                println!("  Expertos activados (quit): {}", experts_display_str);
            } else {
                println!("  Expertos activados (quit): None");
            }
            // Alternatives for the quit token
            println!("  Alternatives for quit token (ID: {}):", token_display_id);
            let mut alternatives_shown_quit = 0;
            for (alt_id, alt_logit) in top_k_alternatives_for_display.iter() {
                if *alt_id != token_display_id && alternatives_shown_quit < 3 {
                    let alt_token_str = tokenizer.decode(&[*alt_id], true).unwrap_or_else(|_e| format!("[ID:{}]", alt_id));
                    println!("    Alt {}: \"{}\" (ID: {}, Logit: {:.2})", alternatives_shown_quit + 1, alt_token_str, alt_id, alt_logit);
                    alternatives_shown_quit += 1;
                }
                if alternatives_shown_quit >= 3 { break; }
            }
            break; // Exit the loop
        }
        
        current_theta_hat = adjust_theta_hat(current_theta_hat, &user_feedback);

        match user_feedback.as_str() {
            "s" | "n" => {
                let validation_bool = user_feedback == "s";
                validation_status_str = if validation_bool { "sí".to_string() } else { "no".to_string() };

                let context_tokens_for_exp = if current_token_ids.len() > 1 {
                    current_token_ids[..current_token_ids.len()-1].to_vec()
                } else { Vec::new() };

                let entry = ExperienceEntry {
                    prompt_tokens: context_tokens_for_exp,
                    generated_token_id: token_display_id, 
                    validation_status: validation_bool,
                    theta_hat_at_generation: pre_adjustment_theta, // Theta used to generate this token
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
        
        // Display validated token info
        println!(
            "⟳ [{}] token: \"{}\" (ID: {}, Logit: {:.2}) | θ̂: {:.2} | validado: {}",
            i, token_str_display, token_display_id, dominant_logit_value_for_display, current_theta_hat, validation_status_str
        );
        // Display activated experts
        if !experts_info_for_display.is_empty() {
            let experts_display_str = experts_info_for_display
                .iter()
                .map(|(name, tier)| format!("{} ({:?})", name, tier))
                .collect::<Vec<String>>()
                .join(", ");
            println!("  Expertos activados: {}", experts_display_str);
        } else {
            println!("  Expertos activados: None");
        }
        // Display alternatives
        println!("  Alternatives:");
        let mut alternatives_shown = 0;
        for (alt_id, alt_logit) in top_k_alternatives_for_display.iter() {
            if *alt_id != token_display_id && alternatives_shown < 3 {
                let alt_token_str = tokenizer.decode(&[*alt_id], true).unwrap_or_else(|_e| format!("[ID:{}]", alt_id));
                println!("    Alt {}: \"{}\" (ID: {}, Logit: {:.2})", alternatives_shown + 1, alt_token_str, alt_id, alt_logit);
                alternatives_shown += 1;
            }
            if alternatives_shown >= 3 { break; }
        }


        if token_display_id == eos_token_id {
            println!("EOS token ({}) encountered. Stopping generation.", eos_token_id);
            break;
        }
        
        // Prepare for next iteration (if not the last one)
        if i < max_new_tokens - 1 {
            let last_token_id_for_gen = token_display_id; // The token we just processed
            
            // Dynamic tier adjustment before next token generation
            orchestrator.system_status.refresh(); // Refresh status for fresh RAM info
            let ram_gb_next_token = orchestrator.system_status.ram_available_gb;
            determine_and_set_allowed_tiers(orchestrator, ram_gb_next_token, current_theta_hat);

            // Generate logits for the *next* token using orchestrator
            let token_embedding_next = model.get_embeddings(&[last_token_id_for_gen])?;
            let current_token_embedding_slice = token_embedding_next.slice(s![0, 0, ..]);
            let input_1d_features_next = Array1::from_vec(current_token_embedding_slice.to_vec());
            
            // let next_logits = generate_next_token(
            //     model, config, last_token_id_for_gen, &mut model_cache, current_theta_hat
            // )?;
            let (next_logits_output, activated_experts_info_next) = orchestrator.forward(
                &input_1d_features_next,
                &token_embedding_next,
                current_theta_hat
            ).map_err(|e| format!("Orchestrator forward pass failed for next token: {}", e))?;

            // Determine the next token and its logit info based on these new logits
            let (chosen_next_token_id, next_dominant_logit) = get_greedy_token_and_logit(&next_logits_output)?;
            let next_top_k_alternatives = get_top_k_tokens(&next_logits_output, 4);

            next_token_id = chosen_next_token_id;
            current_token_ids.push(next_token_id);

            // Update logit info and expert info for the display in the *next* iteration
            dominant_logit_value_for_display = next_dominant_logit;
            top_k_alternatives_for_display = next_top_k_alternatives;
            experts_info_for_display = activated_experts_info_next;
        }
    }

    println!("\n--- Generation Ended ---");
    println!("Final generated token IDs: {:?}", current_token_ids);
    let final_decoded_text = tokenizer.decode(&current_token_ids, true)
        .unwrap_or_else(|e| format!("Failed to decode final sequence: {}", e));
    println!("Final decoded text: {}", final_decoded_text);
    Ok(())
}

/// Main entry point for the standalone REPL application.
///
/// This function orchestrates the setup of all necessary components for the REPL session:
/// 1.  Initializes `GPT2Config` and `GPT2Model`.
/// 2.  Initializes the `TokenizerWrapper`.
/// 3.  Initializes the `MoEOrchestrator` with a predefined set of experts (e.g., `SymbolicExpert`, `MLPExpert`)
///     and a `GatingLayer`. This orchestrator will manage expert activation during token generation.
/// 4.  Retrieves the user's initial prompt and tokenizes it.
/// 5.  Initializes the `ResonanceFeedbackStore` for recording user feedback.
/// 6.  Predicts or sets an initial `theta_hat` (intentionality score).
/// 7.  Calls `run_repl_loop` to start the interactive session, passing all initialized components,
///     including the `orchestrator`.
///
/// # Returns
/// `Ok(())` on successful execution, or a `Box<dyn Error>` if any critical setup or runtime error occurs.
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


    // Instantiate MoEOrchestrator
    let experts: Vec<Box<dyn crate::moe::Expert>> = vec![
        Box::new(SymbolicExpert::new("SymbolicL1", 0.5)), // L1
        Box::new(MLPExpert::new("MLPL2")),                // L2
    ];
    let num_features_for_gating = config.n_embd as usize;
    let gating_layer = GatingLayer::new(num_features_for_gating, experts.len());
    
    let mut orchestrator = MoEOrchestrator::new(
        experts,
        gating_layer,
        None,    // max_concurrent_experts_override
        0.5,     // min_ram_gb_per_expert
        0.8,     // high_cpu_load_threshold
        1,       // num_experts_in_high_load
        2        // default_top_k_experts (can be small as tier logic will filter first)
    );
    // orchestrator.system_status.refresh(); // Initial refresh before REPL loop if needed by first policy check


    // 8. Call run_repl_loop
    // The run_repl_loop returns Result<(), String>, convert its error to Box<dyn Error>
    run_repl_loop(
        &mut model, 
        &config, // config is still needed for EOS token, and potentially by Gpt2Model methods if not fully via orchestrator
        &tokenizer, 
        initial_prompt_tokens, 
        max_new_tokens, 
        initial_theta_hat, 
        eos_token_id,
        &mut store, // Pass the feedback store
        &mut orchestrator, // Pass the orchestrator
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
        use crate::model::{GPT2Model, GPT2Config}; // Keep GPT2Model and Config for existing tests
        use crate::common::ModelKVCache; // Keep for existing tests
        // Added for new tier logic tests
        use crate::orchestrator::MoEOrchestrator;
        use crate::gating::GatingLayer;
        use crate::moe::{SymbolicExpert, MLPExpert, Expert}; // And Expert trait
        use crate::cache_tier::CacheTier;


        // Helper for existing tests
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
        fn test_get_greedy_token_and_logit_basic() {
            // Test with 1D array
            let logits_1d = ArrayD::from_shape_vec(IxDyn(&[5]), vec![0.1, 0.2, 0.5, 0.1, 0.1]).unwrap();
            let (token_id, logit_val) = get_greedy_token_and_logit(&logits_1d).unwrap();
            assert_eq!(token_id, 2);
            assert!((logit_val - 0.5).abs() < f32::EPSILON);

            // Test with 3D array (e.g., model output [1,1,V])
            let logits_3d = ArrayD::from_shape_vec(IxDyn(&[1, 1, 5]), vec![0.1, 0.8, 0.5, 0.1, 0.1]).unwrap();
            let (token_id_3d, logit_val_3d) = get_greedy_token_and_logit(&logits_3d).unwrap();
            assert_eq!(token_id_3d, 1);
            assert!((logit_val_3d - 0.8).abs() < f32::EPSILON);

            // Test with another 3D array, last element in seq
             let logits_3d_long_seq = ArrayD::from_shape_vec(IxDyn(&[1, 3, 5]), 
                vec![0.1, 0.2, 0.3, 0.4, 0.0, // seq 0
                     0.0, 0.0, 0.0, 0.0, 0.0, // seq 1
                     0.1, 0.8, 0.5, 0.1, 0.1  // seq 2 (last one)
                    ]).unwrap();
            let (token_id_3d_ls, logit_val_3d_ls) = get_greedy_token_and_logit(&logits_3d_long_seq).unwrap();
            assert_eq!(token_id_3d_ls, 1); // from last sequence element
            assert!((logit_val_3d_ls - 0.8).abs() < f32::EPSILON);
        }

        #[test]
        fn test_get_greedy_token_and_logit_empty_or_invalid() {
            let empty_logits_1d = ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap();
            assert!(get_greedy_token_and_logit(&empty_logits_1d).is_err());
            
            let empty_logits_3d = ArrayD::from_shape_vec(IxDyn(&[1,1,0]), vec![]).unwrap();
            assert!(get_greedy_token_and_logit(&empty_logits_3d).is_err());

            let invalid_shape_logits = ArrayD::zeros(IxDyn(&[1,1,1,1])); // 4D
             assert!(get_greedy_token_and_logit(&invalid_shape_logits).is_err());
        }

        #[test]
        fn test_get_top_k_tokens_basic() {
            let logits_1d = ArrayD::from_shape_vec(IxDyn(&[5]), vec![0.1, 0.7, 0.5, 0.9, 0.3]).unwrap();
            // Expected sorted: (3, 0.9), (1, 0.7), (2, 0.5), (4, 0.3), (0, 0.1)
            
            let top_1 = get_top_k_tokens(&logits_1d, 1);
            assert_eq!(top_1.len(), 1);
            assert_eq!(top_1[0].0, 3); // token id
            assert!((top_1[0].1 - 0.9).abs() < f32::EPSILON); // logit value

            let top_3 = get_top_k_tokens(&logits_1d, 3);
            assert_eq!(top_3.len(), 3);
            assert_eq!(top_3[0].0, 3);
            assert_eq!(top_3[1].0, 1);
            assert_eq!(top_3[2].0, 2);
            assert!((top_3[0].1 - 0.9).abs() < f32::EPSILON);
            assert!((top_3[1].1 - 0.7).abs() < f32::EPSILON);
            assert!((top_3[2].1 - 0.5).abs() < f32::EPSILON);

            // K larger than vocab size
            let top_10 = get_top_k_tokens(&logits_1d, 10);
            assert_eq!(top_10.len(), 5); // Should return all available, sorted
            assert_eq!(top_10[0].0, 3);
            assert_eq!(top_10[4].0, 0);

            // K = 0
            let top_0 = get_top_k_tokens(&logits_1d, 0);
            assert!(top_0.is_empty());
        }

        #[test]
        fn test_get_top_k_tokens_3d() {
            let logits_3d = ArrayD::from_shape_vec(IxDyn(&[1, 1, 5]), vec![0.1, 0.7, 0.5, 0.9, 0.3]).unwrap();
            // Expected sorted: (3, 0.9), (1, 0.7), (2, 0.5), (4, 0.3), (0, 0.1)
            let top_2 = get_top_k_tokens(&logits_3d, 2);
            assert_eq!(top_2.len(), 2);
            assert_eq!(top_2[0].0, 3);
            assert_eq!(top_2[1].0, 1);
        }

        #[test]
        fn test_get_top_k_tokens_empty_or_invalid() {
            let empty_logits_1d = ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap();
            assert!(get_top_k_tokens(&empty_logits_1d, 3).is_empty());

            let empty_logits_3d = ArrayD::from_shape_vec(IxDyn(&[1,1,0]), vec![]).unwrap();
            assert!(get_top_k_tokens(&empty_logits_3d, 3).is_empty());
            
            let invalid_shape_logits = ArrayD::zeros(IxDyn(&[2,2])); // 2D
            assert!(get_top_k_tokens(&invalid_shape_logits, 3).is_empty());
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

        // --- Tests for determine_and_set_allowed_tiers ---

        // Helper to create a minimal orchestrator for these tests
        fn create_test_orchestrator_for_tier_logic() -> MoEOrchestrator {
            let experts: Vec<Box<dyn Expert>> = vec![
                Box::new(SymbolicExpert::new("S",0.0)), Box::new(MLPExpert::new("M"))
            ];
            let gating = GatingLayer::new(10, experts.len()); // Dummy values
            // Ensure MoEOrchestrator::new parameters match its definition
            MoEOrchestrator::new(
                experts, 
                gating, 
                None,    // max_concurrent_experts_override
                0.5,     // min_ram_gb_per_expert
                0.8,     // high_cpu_load_threshold
                1,       // num_experts_in_high_load
                2        // default_top_k_experts - this is num_to_activate, not directly tier related for this test
            )
        }

        #[test]
        fn test_determine_allowed_tiers_ram_low() {
            let mut orchestrator = create_test_orchestrator_for_tier_logic();
            // RAM is low, theta irrelevant for this rule
            determine_and_set_allowed_tiers(&mut orchestrator, 1.5, 0.7); 
            assert_eq!(orchestrator.current_allowed_tiers, vec![CacheTier::L1]);
        }

        #[test]
        fn test_determine_allowed_tiers_high_theta_high_ram() {
            let mut orchestrator = create_test_orchestrator_for_tier_logic();
            determine_and_set_allowed_tiers(&mut orchestrator, 7.0, 0.9);
            assert_eq!(orchestrator.current_allowed_tiers, vec![CacheTier::L1, CacheTier::L2, CacheTier::L3]);
        }

        #[test]
        fn test_determine_allowed_tiers_low_theta_safe_mode() {
            let mut orchestrator = create_test_orchestrator_for_tier_logic();
            // RAM is high, but theta is low
            determine_and_set_allowed_tiers(&mut orchestrator, 7.0, 0.4); 
            assert_eq!(orchestrator.current_allowed_tiers, vec![CacheTier::L1]);
        }

        #[test]
        fn test_determine_allowed_tiers_default_case() {
            let mut orchestrator = create_test_orchestrator_for_tier_logic();
            // RAM is moderate, theta is moderate
            determine_and_set_allowed_tiers(&mut orchestrator, 4.0, 0.7); 
            assert_eq!(orchestrator.current_allowed_tiers, vec![CacheTier::L1, CacheTier::L2]);
        }

        #[test]
        fn test_determine_allowed_tiers_ram_boundary() {
            let mut orchestrator = create_test_orchestrator_for_tier_logic();
            // RAM just at 2.0GB, should not trigger < 2.0 rule
            determine_and_set_allowed_tiers(&mut orchestrator, 2.0, 0.7); 
            assert_eq!(orchestrator.current_allowed_tiers, vec![CacheTier::L1, CacheTier::L2]); // Default
        }

        #[test]
        fn test_determine_allowed_tiers_theta_ram_boundaries_for_l1_l2_l3() {
            let mut orchestrator = create_test_orchestrator_for_tier_logic();
            // Theta just above 0.85, RAM just above 6.0
            determine_and_set_allowed_tiers(&mut orchestrator, 6.1, 0.86);
            assert_eq!(orchestrator.current_allowed_tiers, vec![CacheTier::L1, CacheTier::L2, CacheTier::L3]);

            // Theta at 0.85 (not >0.85), RAM high -> default
            determine_and_set_allowed_tiers(&mut orchestrator, 6.1, 0.85);
            assert_eq!(orchestrator.current_allowed_tiers, vec![CacheTier::L1, CacheTier::L2]);
            
            // Theta high, RAM at 6.0 (not >6.0) -> default
            determine_and_set_allowed_tiers(&mut orchestrator, 6.0, 0.86);
            assert_eq!(orchestrator.current_allowed_tiers, vec![CacheTier::L1, CacheTier::L2]);
        }

        #[test]
        fn test_determine_allowed_tiers_theta_boundary_for_safe_mode() {
            let mut orchestrator = create_test_orchestrator_for_tier_logic();
            // Theta just at 0.5 (not <0.5), RAM high -> default
            determine_and_set_allowed_tiers(&mut orchestrator, 7.0, 0.5);
            assert_eq!(orchestrator.current_allowed_tiers, vec![CacheTier::L1, CacheTier::L2]);
        }
    }
}
