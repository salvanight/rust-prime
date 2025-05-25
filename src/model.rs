use ndarray::{ArrayD, IxDyn, Array2, s, Axis, ArrayView2}; // Consolidated use statements, added ArrayView2
use crate::config::GPT2Config;
use crate::common::{LayerNorm, ModelKVCache}; // Import ModelKVCache
use crate::attention::MultiHeadAttention;
use crate::mlp::MLP;

#[derive(Debug)]
pub struct TransformerBlock {
    ln_1: LayerNorm,
    attn: MultiHeadAttention,
    ln_2: LayerNorm,
    mlp: MLP,
}

impl TransformerBlock {
    pub fn new(config: &GPT2Config) -> Result<Self, Box<dyn std::error::Error>> {
        let ln_1 = LayerNorm::new()?;
        let attn = MultiHeadAttention::new(config.n_head, config.n_embd)?;
        let ln_2 = LayerNorm::new()?;
        let n_inner = config.n_inner.unwrap_or(4 * config.n_embd);
        let mlp = MLP::new(config.n_embd, n_inner)?;
        
        Ok(Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        })
    }

    pub fn forward(
        &mut self, // Changed to &mut self
        hidden_states: &ArrayD<f32>, 
        layer_kv_cache: &mut Vec<f32>, // Added layer_kv_cache
        _theta_hat: f32 // Added theta_hat, underscore if not used immediately
    ) -> Result<ArrayD<f32>, Box<dyn std::error::Error>> {
        // For now, make it a pass-through so the model can "run" structurally.
        // TODO: Implement actual TransformerBlock forward logic including attention and MLP,
        // using layer_kv_cache and potentially theta_hat.
        Ok(hidden_states.clone())
    }
}

/// Represents the GPT-2 model architecture and its parameters.
///
/// This struct holds the model's layers (attention, MLP, layer normalization)
/// and provides methods for forward passes and other model-specific operations.
/// It now also stores its `GPT2Config`.
#[derive(Debug)]
pub struct GPT2Model {
    config: GPT2Config, // Added config field
    wte_weight: ArrayD<f32>, // Token embeddings
    wpe_weight: ArrayD<f32>, // Positional embeddings
    h: Vec<TransformerBlock>,
    ln_f: LayerNorm,
}

impl GPT2Model {
    /// Creates a new instance of `GPT2Model` based on the provided configuration.
    ///
    /// Initializes all layers and parameters of the model according to `GPT2Config`.
    /// The provided `config` is stored within the model instance for later reference (e.g., by `get_embeddings`).
    ///
    /// # Arguments
    /// * `config`: A reference to the `GPT2Config` specifying the model's architecture.
    ///
    /// # Returns
    /// A `Result` containing the initialized `GPT2Model` or an error string if initialization fails.
    pub fn new(config: &GPT2Config) -> Result<Self, Box<dyn std::error::Error>> { // Return type kept as Box<dyn Error> from existing code
        let wte_weight = ArrayD::zeros((config.vocab_size as usize, config.n_embd as usize).into_dyn());
        let wpe_weight = ArrayD::zeros((config.n_positions as usize, config.n_embd as usize).into_dyn());
        
        let mut h = Vec::with_capacity(config.n_layer as usize);
        for _i in 0..config.n_layer {
            h.push(TransformerBlock::new(config)?);
        }
        
        let ln_f = LayerNorm::new()?;
        
        Ok(Self {
            config: config.clone(), // Store a clone of the config
            wte_weight,
            wpe_weight,
            h,
            ln_f,
        })
    }

    /// Retrieves the token embeddings for a given sequence of token IDs.
    ///
    /// **Note:** This is currently a placeholder implementation and returns zeros
    /// of the expected shape. A full implementation would use the model's
    /// word token embeddings (`wte`) and word position embeddings (`wpe`).
    ///
    /// # Arguments
    /// * `tokens`: A slice of `u32` token IDs for which to retrieve embeddings.
    ///
    /// # Returns
    /// A `Result` containing an `ArrayD<f32>` with the shape `[1, num_tokens, n_embd]`
    /// representing the embeddings, or an error string if `tokens` is empty or
    /// another issue occurs.
    pub fn get_embeddings(&self, tokens: &[u32]) -> Result<ArrayD<f32>, String> {
        if tokens.is_empty() {
            return Err("Input token list cannot be empty.".to_string());
        }

        let batch_size = 1; // Assuming batch size of 1 for this context
        let seq_len = tokens.len();
        let n_embd = self.config.n_embd as usize;

        // --- 1. Token Embeddings (WTE) ---
        let wte_view: ArrayView2<f32> = self.wte_weight.view().into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| format!("Failed to view wte_weight as 2D array: {}", e))?;

        let mut token_embedding_data = Vec::with_capacity(seq_len * n_embd);
        for &token_id in tokens {
            if (token_id as usize) < wte_view.shape()[0] {
                let embedding_vector = wte_view.row(token_id as usize);
                token_embedding_data.extend(embedding_vector.iter());
            } else {
                return Err(format!("Token ID {} is out of vocab size {}.", token_id, wte_view.shape()[0]));
            }
        }
        let token_embeddings = ArrayD::from_shape_vec(IxDyn(&[batch_size, seq_len, n_embd]), token_embedding_data)
            .map_err(|e| format!("Failed to create token_embeddings ArrayD: {}", e))?;
        
        // --- 2. Positional Embeddings (WPE) ---
        if seq_len > self.wpe_weight.shape()[0] {
            return Err(format!(
                "Sequence length ({}) exceeds maximum positional embeddings ({})",
                seq_len, self.wpe_weight.shape()[0]
            ));
        }
        // Slice wpe_weight: take rows from 0 to seq_len-1
        let positional_embeddings_slice = self.wpe_weight.slice(s![..seq_len, ..]);
        // Convert to owned ArrayD and add batch axis
        let positional_embeddings_broadcastable = positional_embeddings_slice
            .to_owned()
            .into_dyn()
            .insert_axis(Axis(0)); // Shape: [1, seq_len, n_embd]

        if positional_embeddings_broadcastable.shape() != [batch_size, seq_len, n_embd] {
             return Err(format!(
                "Shape mismatch for positional embeddings. Expected: {:?}, Got: {:?}",
                [batch_size, seq_len, n_embd], positional_embeddings_broadcastable.shape()
            ));
        }
        if token_embeddings.shape() != [batch_size, seq_len, n_embd] {
             return Err(format!(
                "Shape mismatch for token embeddings. Expected: {:?}, Got: {:?}",
                [batch_size, seq_len, n_embd], token_embeddings.shape()
            ));
        }

        // --- 3. Combine Embeddings ---
        Ok(&token_embeddings + &positional_embeddings_broadcastable)
    }

    pub fn forward(
        &mut self,
        input_ids: &Array2<i32>, 
        model_cache: &mut ModelKVCache,
        theta_hat: f32
    ) -> Result<ArrayD<f32>, Box<dyn std::error::Error>> {
        let batch_size = input_ids.shape()[0];
        // let seq_len = input_ids.shape()[1]; // Not directly used after refactor to get_embeddings

        // 1. Get initial hidden states using get_embeddings
        // Convert Array2<i32> to Vec<u32> or &[u32] for get_embeddings.
        // Assuming batch_size is 1 for now as get_embeddings expects &[u32] (a single sequence).
        if batch_size != 1 {
            // TODO: Support batch_size > 1 in get_embeddings or handle here.
            return Err(Box::from("GPT2Model::forward currenty only supports batch_size = 1 due to get_embeddings input type."));
        }
        let tokens_for_embedding: Vec<u32> = input_ids.iter().map(|&id| id as u32).collect();
        
        let mut hidden_states = self.get_embeddings(&tokens_for_embedding)
            .map_err(|e| -> Box<dyn std::error::Error> { Box::from(e) })?;
        // Expected shape from get_embeddings: [1, seq_len, n_embd]

        // 2. Pass through Transformer Blocks
        // Ensure model_cache has the correct number of layers
        if model_cache.len() != self.h.len() {
            return Err(Box::from(format!(
                "model_cache length ({}) does not match number of transformer blocks ({}).",
                model_cache.len(), self.h.len()
            )));
        }

        for (i, block) in self.h.iter_mut().enumerate() {
            // Each block updates hidden_states and uses/updates its part of model_cache.
            hidden_states = block.forward(&hidden_states, &mut model_cache[i], theta_hat)?;
        }

        // 3. Final Layer Normalization
        hidden_states = self.ln_f.forward(&hidden_states)
            .map_err(|e| -> Box<dyn std::error::Error> { Box::from(e) })?;
            // Assuming ln_f.forward also returns Result<ArrayD<f32>, Box<dyn Error>>
            // If it returns Result<ArrayD<f32>, String>, adapt error mapping.

        // 4. Language Model Head (Placeholder)
        // TODO: Implement lm_head to map hidden_states (n_embd) to vocab_size logits.
        // For now, returning the processed hidden_states.
        // The REPL currently expects logits from orchestrator.forward, which uses this model's output.
        Ok(hidden_states) 
    }
}