// src/runtime_interface.rs

use clap::Parser;
use std::error::Error;

use super::model_loader;
use super::tensor_engine; // Not directly used here, but good to have if errors bubble up
use super::text_generator;
use super::tokenizer_core;
use super::transformer_core;
// use crate::resonance_feedback::{ResonanceFeedbackStore, ExperienceEntry, ValidationStatus}; // Added
// use uuid; // Added for direct UUID usage if ExperienceEntry::new() doesn't set it (it does)

// 2. Define CLI Arguments
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct CliArgs {
    #[clap(long, value_parser)]
    model_path: String,
    #[clap(long, value_parser)]
    vocab_path: String,
    #[clap(long, value_parser)]
    merges_path: String,
    #[clap(long, value_parser)]
    prompt: String,

    #[clap(long, value_parser, default_value_t = 50)]
    max_length: usize,
    #[clap(long, value_parser, default_value_t = 50256)] // Default EOS for GPT-2
    eos_token_id: u32,

    // Model Configuration Arguments
    #[clap(long, value_parser, default_value_t = 12)]
    config_n_layer: usize,
    #[clap(long, value_parser, default_value_t = 12)]
    config_n_head: usize,
    #[clap(long, value_parser, default_value_t = 768)]
    config_n_embd: usize,
    #[clap(long, value_parser, default_value_t = 50257)]
    config_vocab_size: usize,
    #[clap(long, value_parser, default_value_t = 1024)]
    config_block_size: usize,
    // Note: The 'bias' field in Config (true/false) is not easily set via CLI flag without more complex parsing.
    // For now, we'll assume 'bias: true' as is typical for GPT-2.
    // A production CLI might use `--no-bias` or parse "true"/"false".
}

// Custom error wrapper to combine various error types
#[derive(Debug)]
enum RuntimeError {
    Tokenizer(tokenizer_core::TokenizerError),
    ModelLoader(model_loader::ModelLoaderError),
    Transformer(transformer_core::TransformerError),
    TextGenerator(text_generator::TextGeneratorError),
    Io(std::io::Error), // For other IO errors if any
    Message(String), // For general messages
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuntimeError::Tokenizer(e) => write!(f, "Tokenizer error: {:?}", e),
            RuntimeError::ModelLoader(e) => write!(f, "ModelLoader error: {:?}", e),
            RuntimeError::Transformer(e) => write!(f, "Transformer error: {:?}", e),
            RuntimeError::TextGenerator(e) => write!(f, "TextGenerator error: {:?}", e),
            RuntimeError::Io(e) => write!(f, "IO error: {}", e),
            RuntimeError::Message(s) => write!(f, "Runtime error: {}", s),
        }
    }
}

impl Error for RuntimeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            RuntimeError::Tokenizer(e) => Some(e),
            RuntimeError::ModelLoader(e) => Some(e),
            RuntimeError::Transformer(e) => Some(e),
            RuntimeError::TextGenerator(e) => Some(e),
            RuntimeError::Io(e) => Some(e),
            RuntimeError::Message(_) => None,
        }
    }
}

// Implement From for each error type to simplify error handling with `?`
impl From<tokenizer_core::TokenizerError> for RuntimeError {
    fn from(err: tokenizer_core::TokenizerError) -> Self { RuntimeError::Tokenizer(err) }
}
impl From<model_loader::ModelLoaderError> for RuntimeError {
    fn from(err: model_loader::ModelLoaderError) -> Self { RuntimeError::ModelLoader(err) }
}
impl From<transformer_core::TransformerError> for RuntimeError {
    fn from(err: transformer_core::TransformerError) -> Self { RuntimeError::Transformer(err) }
}
impl From<text_generator::TextGeneratorError> for RuntimeError {
    fn from(err: text_generator::TextGeneratorError) -> Self { RuntimeError::TextGenerator(err) }
}
impl From<tensor_engine::TensorError> for RuntimeError {
    fn from(err: tensor_engine::TensorError) -> Self { 
        // Wrap TensorError into a higher-level error, e.g. TransformerError or a new variant
        RuntimeError::Transformer(transformer_core::TransformerError::TensorError(err))
    }
}
impl From<std::io::Error> for RuntimeError {
    fn from(err: std::io::Error) -> Self { RuntimeError::Io(err) }
}


// 3. `run_cli` Function
pub fn run_cli() -> Result<(), Box<dyn Error>> {
    // 1. Parse CLI arguments
    let args = CliArgs::parse();

    // 2. Load Tokenizer components
    println!("Loading vocabulary from: {}", args.vocab_path);
    let vocab = tokenizer_core::load_vocab(&args.vocab_path).map_err(RuntimeError::from)?;
    println!("Loading merges from: {}", args.merges_path);
    let merges = tokenizer_core::load_merges(&args.merges_path).map_err(RuntimeError::from)?;
    println!("Tokenizer components loaded.");

    // 3. Load Model
    println!("Loading model weights from: {}", args.model_path);
    let weights_map = model_loader::load_safetensors(&args.model_path).map_err(RuntimeError::from)?;
    println!("Model weights loaded.");

    let config = transformer_core::Config {
        n_layer: args.config_n_layer,
        n_head: args.config_n_head,
        n_embd: args.config_n_embd,
        vocab_size: args.config_vocab_size,
        block_size: args.config_block_size,
        bias: true, // Assuming bias is true, as typical for GPT-2.
    };
    println!("Model configuration prepared: {:?}", config);

    let model = transformer_core::GPT2Model::new(config, weights_map).map_err(RuntimeError::from)?;
    println!("GPT-2 Model instantiated.");

    // 4. Tokenize Prompt
    println!("Tokenizing prompt: \"{}\"", args.prompt);
    let input_ids = tokenizer_core::encode(&args.prompt, &vocab, &merges).map_err(RuntimeError::from)?;
    if input_ids.is_empty() && !args.prompt.trim().is_empty() {
        // This can happen if all tokens in the prompt are unknown.
        return Err(Box::new(RuntimeError::Message(format!(
            "Prompt '{}' resulted in empty token sequence. Check if prompt tokens are in vocabulary.",
            args.prompt
        ))));
    }
    if args.prompt.trim().is_empty() && input_ids.is_empty() {
         return Err(Box::new(RuntimeError::Message(
            "Prompt is empty or only whitespace, resulting in no tokens to generate from.".to_string()
        )));
    }
    println!("Prompt token IDs: {:?}", input_ids);
    
    // 5. Generate Text
    println!("Generating text (max_length: {}, eos_token_id: {})...", args.max_length, args.eos_token_id);
    let generated_ids = text_generator::generate(
        &model, 
        input_ids.clone(), 
        args.max_length, 
        args.eos_token_id,
        None
        // Some(&feedback_store) // Pass the feedback store
    ).map_err(RuntimeError::from)?;
    println!("Generated token IDs: {:?}", generated_ids);

    // 6. Decode Output
    let output_text = tokenizer_core::decode(&generated_ids, &vocab).map_err(RuntimeError::from)?;
    println!("Decoding complete.");

    // 7. Print Result
    println!("\n--- Prompt ---");
    println!("{}", args.prompt);
    println!("\n--- Generated Text (including prompt) ---");
    println!("{}", output_text);
    // To print only the newly generated part:
    // let prompt_decoded_len = tokenizer_core::decode(&input_ids, &vocab)?.len();
    // if output_text.len() >= prompt_decoded_len {
    //     let newly_generated_text = &output_text[prompt_decoded_len..];
    //     println!("\n--- Newly Generated Text ---");
    //     println!("{}", newly_generated_text.trim_start());
    // }

    // --- Feedback Collection ---
    println!("\n--- Feedback ---");
    println!("Was this response helpful/good?");
    println!("1: Accepted");
    println!("2: Rejected");
    println!("Any other key to skip/unvalidated.");

    let mut user_feedback_input = String::new();
    // std::io::stdin().read_line(&mut user_feedback_input).map_err(|e| RuntimeError::Io(e))?; // Propagate IO error

    // let validation_status = match user_feedback_input.trim() {
    //     "1" => ValidationStatus::Accepted,
    //     "2" => ValidationStatus::Rejected,
    //     _ => ValidationStatus::Unvalidated,
    // };

    // let mut notes_opt = None; // Renamed to avoid conflict with resonance_feedback::notes field
    // if validation_status != ValidationStatus::Unvalidated {
    //     println!("Optional notes/comments for this feedback (press Enter to skip):");
    //     let mut notes_input_str = String::new(); // Renamed
    //     std::io::stdin().read_line(&mut notes_input_str).map_err(|e| RuntimeError::Io(e))?;
    //     if !notes_input_str.trim().is_empty() {
    //         notes_opt = Some(notes_input_str.trim().to_string());
    //     }
    // }

    // let mut experience_entry = ExperienceEntry::new(
    //     args.prompt.clone(), // Original prompt
    //     output_text.clone()  // Decoded output text
    // );
    // experience_entry.validation_status = validation_status;
    // experience_entry.notes = notes_opt;
    // // resonance_score and symbolic_theta_hat remain None for now

    // feedback_store.add_experience(experience_entry);
    // println!("Feedback recorded. Thank you!");

    // // Save the feedback store
    // if let Err(e) = feedback_store.save_to_file(feedback_file_path) {
    //     eprintln!("Warning: Could not save feedback store to '{}': {}", feedback_file_path, e);
    // } else {
    //     println!("Feedback store saved to '{}'.", feedback_file_path);
    // }

    Ok(())
}
