# Rust Native Transformer

A 100% native Rust implementation of a Transformer inference engine, built from scratch without Python bindings or external ML framework dependencies.

## 🧬 Project Philosophy

*   **Pure Rust:** Zero C/C++ bindings, zero Python interop.
*   **From Scratch:** Core components (tokenizer, tensor operations, model architecture) implemented natively.
*   **Transparency & Introspection:** Aims to provide a clear, understandable, and modifiable Transformer implementation.
*   **Computational Sovereignty:** Foundational work towards AI systems that are independent of non-Rust ecosystems.

## 🚀 Project Status (Phase 1 - Minima Viable Product)

This project has successfully completed its first phase, delivering a minimally viable GPT-2 inference engine capable of:

*   Loading GPT-2 models from `.safetensors` files.
*   Tokenizing input text using a native BPE tokenizer.
*   Performing inference via a native Transformer core implementation.
*   Generating text using greedy decoding.
*   Providing a command-line interface for interaction.

**Note:** While largely functional, this is an early-stage research project. Some components, like the tokenizer, have known limitations with specific complex inputs (3 unit tests failing), and the tensor engine is basic (no SIMD/batching optimizations yet).

## 🛠️ Core Modules

*   `tokenizer_core.rs`: Native BPE (Byte Pair Encoding) tokenizer, loads `vocab.json` and `merges.txt`.
*   `tensor_engine.rs`: Generic `Tensor<T>` structure and basic tensor operations (matmul, softmax, layernorm, gelu) for `f32` on CPU.
*   `model_loader.rs`: Parser for `.safetensors` model weight files (supports F32 tensors).
*   `transformer_core.rs`: Implementation of the GPT-2 model architecture (MHA, MLP, Blocks, etc.).
*   `text_generator.rs`: Logic for text generation (currently greedy decoding).
*   `runtime_interface.rs`: Command-line interface (CLI) for running inference.

## 📦 Build Instructions

1.  Ensure you have Rust and Cargo installed (see [rustup.rs](https://rustup.rs/)).
2.  Clone the repository:
    ```bash
    git clone <repository_url>
    cd rust-native-transformer
    ```
3.  Build the release binary:
    ```bash
    cargo build --release
    ```
    The executable will be located at `target/release/rust_native_transformer_cli`.

## ⚙️ Usage (CLI)

To run inference, you need:
1.  A GPT-2 model in `.safetensors` format.
2.  The corresponding `vocab.json` and `merges.txt` for the tokenizer.

You can obtain tokenizer files and convert Hugging Face pre-trained models (e.g., "gpt2") to `.safetensors` using the `transformers` and `safetensors` Python libraries. (Refer to the guidance in previous steps for a conversion script example).

**Example Command:**

```bash
./target/release/rust_native_transformer_cli \
    --model-path /path/to/your/model.safetensors \
    --vocab-path /path/to/your/vocab.json \
    --merges-path /path/to/your/merges.txt \
    --prompt "The Rust programming language is" \
    --max-length 50 \
    --config-n-layer 12 \
    --config-n-head 12 \
    --config-n-embd 768 \
    --config-vocab-size 50257 \
    --config-block-size 1024 \
    --eos-token-id 50256
```
*Adjust `--config-*` parameters to match your specific GPT-2 model variant (the example uses values for "gpt2" base).*

## ✨ Key Features (Phase 1)

*   Complete GPT-2 architecture implementation in pure Rust.
*   Loading of `.safetensors` weights (F32).
*   Native BPE Tokenizer.
*   Greedy decoding for text generation.
*   Functional CLI for inference.
*   No external C/Python library dependencies for core logic.

## ⚠️ Current Limitations & Future Work

This project is under active development. Key areas for improvement include:

*   **`tokenizer_core.rs`:**
    *   Resolve 3 failing unit tests for specific complex string tokenizations.
    *   Investigate BPE merge application order and handling of unknown tokens more deeply.
*   **`tensor_engine.rs`:**
    *   Implement advanced broadcasting capabilities.
    *   Add true batched matrix multiplication (`matmul_batch`).
    *   Introduce SIMD optimizations (`std::simd` or `core::arch`) for CPU performance.
    *   Support for other data types (e.g., F16).
    *   Develop a basic internal profiler for benchmarking operations.
*   **`transformer_core.rs`:**
    *   Implement KV Caching for efficient generation with long contexts.
*   **`text_generator.rs`:**
    *   Refactor to support multiple sampling strategies (Top-K, Top-P).
    *   Implement `TopKDecoder`, `TopPDecoder` traits.
*   **Build & Deployment:**
    *   Explore WebAssembly (WASM) compilation target using `wasm-bindgen`.
*   **General:**
    *   Further performance profiling and optimization.
    *   Expanded test coverage, including more integration tests with real model data.

## 📄 License

This project is dual-licensed under your choice of the Apache License, Version 2.0 or the MIT license. See LICENSE-APACHE and LICENSE-MIT for details.
