# Rust Transformers GPT-2 & Native CLI

## Overview

This project is a pure Rust implementation for GPT-2 model inference, built from scratch. It aims to provide a foundational understanding of transformer architectures and their implementation without relying on external C/C++ bindings (like PyTorch or TensorFlow).

The project consists of two main parts:

1.  **`rust_transformers_gpt2` (library)**: Located in the `src/` directory, this library provides the core functionalities for working with GPT-2 models, including:
    *   Tokenization (BPE)
    *   Model loading (from `.safetensors` files)
    *   GPT-2 model architecture and inference logic.
2.  **`rust-native-transformer` (CLI)**: Located in the `rust-native-transformer/` directory, this is a command-line application built using the `rust_transformers_gpt2` library. It allows users to run GPT-2 text generation inference directly.

The philosophy behind this project is to explore and implement transformer models in pure Rust, fostering a deeper understanding of their inner workings.

## Features

The `rust-native-transformer` CLI and underlying library support:

*   **`.safetensors` Model Loading**: Securely loads GPT-2 model weights.
*   **Native BPE Tokenizer**: Implements the Byte Pair Encoding tokenizer used by GPT-2.
*   **GPT-2 Architecture**: Core components of the GPT-2 model (attention, MLP, layer normalization) are implemented.
*   **Greedy Decoding**: Basic text generation using greedy sampling (argmax).
*   **Pure Rust**: No external C/C++ dependencies for core model operations.

For more details on CLI features, see the `rust-native-transformer/README.md`.

## Project Structure

```
.
├── DEPLOY.md                       # Detailed CLI deployment and usage instructions
├── LICENSE                         # Project license file
├── README.md                       # This file: Main project overview
├── resources/                      # Tokenizer data (vocab, merges) and model configs
│   ├── config/
│   │   └── gpt2/
│   │       └── config.json         # Example GPT-2 model configuration
│   └── tokenizer_data/
│       └── gpt2/
│           ├── gpt2-vocab.json     # GPT-2 vocabulary
│           ├── merges.txt          # GPT-2 merges rules
│           └── sample_token_ids.json # Sample token IDs for testing
├── rust-native-transformer/        # CLI application source code
│   ├── README.md                   # Detailed CLI-specific README
│   ├── Cargo.toml
│   └── src/                        # CLI source files (main.rs, etc.)
└── src/                            # `rust_transformers_gpt2` library source code
    ├── lib.rs
    ├── model_loader.rs
    ├── tensor_engine.rs
    ├── text_generator.rs
    ├── tokenizer_core.rs
    └── transformer_core.rs
```

## Getting Started

### CLI Application

The primary way to use this project is through the `rust_native_transformer_cli`. This allows you to load a GPT-2 model and generate text based on a prompt.

**For full build, setup, model preparation, and execution instructions, please refer to [DEPLOY.md](DEPLOY.md).**

### Library Usage (Optional)

The `rust_transformers_gpt2` library (located in `src/`) can be used independently in other Rust projects.

A minimal example of using the tokenizer:

```rust
// Add to your Cargo.toml:
// rust_transformers_gpt2 = { path = "path/to/this/repo/src" } // Adjust path as needed

// In your Rust code:
// Note: This is a conceptual example. Actual library structure might differ.
// Refer to `rust-native-transformer/src/main.rs` for a more complete usage example.

/*
use rust_transformers_gpt2::tokenizer_core;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vocab_path = Path::new("resources/tokenizer_data/gpt2/gpt2-vocab.json");
    let merges_path = Path::new("resources/tokenizer_data/gpt2/merges.txt");

    println!("Loading vocabulary from: {:?}", vocab_path);
    let vocab = tokenizer_core::load_vocab(vocab_path)?;
    println!("Loading merges from: {:?}", merges_path);
    let merges = tokenizer_core::load_merges(merges_path)?;
    println!("Tokenizer components loaded.");

    let text = "Hello world!";
    let token_ids = tokenizer_core::encode(text, &vocab, &merges)?;
    println!("'{}' tokenized to: {:?}", text, token_ids);

    let decoded_text = tokenizer_core::decode(&token_ids, &vocab)?;
    println!("Token IDs {:?} decoded to: '{}'", token_ids, decoded_text);

    Ok(())
}
*/
```

For more comprehensive examples of how to use the library components (model loading, inference), please refer to the `rust-native-transformer` CLI application code, particularly its `main.rs` file.

## Current Status & Future Work

This project is currently a minimally viable GPT-2 inference engine and an ongoing research effort. It demonstrates the core concepts of loading and running GPT-2 models in a pure Rust environment.

Key functionalities are in place, but there are many areas for improvement and expansion. For a detailed list of current limitations and planned future work, please see the "Limitations" and "Future Work" sections in the [rust-native-transformer/README.md](rust-native-transformer/README.md).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```
