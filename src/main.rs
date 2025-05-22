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
    Ok(())
}