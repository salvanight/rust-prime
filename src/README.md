# Experimental Transformer Library (ndarray-based)

## Overview

This directory contains an experimental Rust library for working with transformer models, specifically focusing on GPT-2. Unlike the `rust-native-transformer` project also in this repository (which is a pure-Rust, from-scratch implementation), this library utilizes external crates for some core functionalities.

**Key Characteristics:**
- **Tensor Operations:** Uses the `ndarray` crate for multi-dimensional arrays and tensor operations.
- **Tokenization:** Leverages the `tokenizers` crate (specifically `tokenizers::models::bpe::BPE`), which is a Rust library often used with Hugging Face models. This is distinct from the native BPE tokenizer found in `rust-native-transformer`.
- **Development Status:** This library appears to be more experimental and less complete than the `rust-native-transformer`. For instance, the `model.rs` contains `todo!` placeholders in its forward pass implementations.

## Modules

The main components of this library include:

- **`lib.rs`**: Main library file, declares modules and might expose public APIs.
- **`config.rs`**: Likely contains configuration structures for models (e.g., `GPT2Config`).
- **`tokenizer.rs`**: Contains `GPT2Tokenizer` which wraps the `tokenizers::Tokenizer` for BPE tokenization.
- **`model.rs`**: Defines `GPT2Model` and `TransformerBlock` using `ndarray` for tensors.
    - *Note: As of the last review, the `forward` methods in `model.rs` were not fully implemented.*
- **`attention.rs`**: Likely contains implementations for attention mechanisms (e.g., Multi-Head Attention) using `ndarray`.
- **`mlp.rs`**: Likely contains implementations for Multi-Layer Perceptrons (feed-forward networks) using `ndarray`.
- **`common.rs`**: May contain shared utilities, such as `LayerNorm`.
- **`cache_tier.rs`, `gating.rs`, `moe.rs`, `orchestrator.rs`, `repl.rs`, `repl_feedback.rs`, `system_resources.rs`**: These modules suggest explorations into more advanced topics like caching, mixture of experts, and REPL interfaces, but their completeness would need further review.

## Purpose

This library seems to serve as an alternative exploration or an earlier developmental stage for building transformer models in Rust, opting to use established crates like `ndarray` and `tokenizers` rather than implementing everything from scratch.

It is distinct from the more developed, pure-Rust `rust-native-transformer` CLI and library found in the `rust-native-transformer/` directory. For a runnable GPT-2 inference engine, please refer to that project.
