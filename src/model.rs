use ndarray::{ArrayD, IxDyn, Array2, s, Axis}; // Consolidated use statements
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
        // The attention_mask parameter is removed for now.
        // Cache interaction will implicitly handle attention context.
        println!("TransformerBlock forward called with hidden_states shape: {:?}, layer_kv_cache length: {}", hidden_states.shape(), layer_kv_cache.len());
        // Actual implementation will use layer_kv_cache and theta_hat.
        todo!("Implement TransformerBlock forward pass with cache and theta_hat");
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
        // Placeholder implementation - actual embedding lookup would happen here.
        // For now, returns zeros of the expected shape [1, num_tokens, n_embd].
        if tokens.is_empty() {
            return Err("Cannot get embeddings for empty token list".to_string());
        }
        let num_tokens = tokens.len();
        let n_embd = self.config.n_embd as usize;
        // In a real scenario, you would use self.wte (word token embeddings)
        // and self.wpe (word position embeddings) here.
        Ok(ArrayD::zeros(IxDyn(&[1, num_tokens, n_embd])))
    }

    pub fn forward(
        &mut self, // Changed to &mut self
        input_ids: &Array2<i32>, 
        model_cache: &mut ModelKVCache, // Added model_cache
        theta_hat: f32 // Added theta_hat
    ) -> Result<ArrayD<f32>, Box<dyn std::error::Error>> {
        let batch_size = input_ids.shape()[0];
        let seq_len = input_ids.shape()[1];
        
        let n_embd = self.wte_weight.shape()[1]; 

        // 1. Token Embeddings (Placeholder: creating zeros for simplicity)
        // In a real implementation, this would use self.wte_weight.embedding(input_ids)
        let token_embeddings = ArrayD::zeros((batch_size, seq_len, n_embd).into_dyn());

        // 2. Positional Embeddings
        if seq_len > self.wpe_weight.shape()[0] {
            return Err(format!(
                "Sequence length ({}) exceeds maximum positional embeddings ({})",
                seq_len, self.wpe_weight.shape()[0]
            ).into());
        }
        let positional_embeddings_slice = self.wpe_weight.slice(s![..seq_len, ..]);
        let positional_embeddings_owned: ArrayD<f32> = positional_embeddings_slice.to_owned().into_dyn();
        let positional_embeddings_broadcastable = positional_embeddings_owned.insert_axis(Axis(0));
        
        let mut hidden_states = token_embeddings + positional_embeddings_broadcastable;
        // println!("Initial hidden_states shape: {:?}", hidden_states.shape());

        // 3. Pass through Transformer Blocks
        for (i, block) in self.h.iter_mut().enumerate() {
            // Each block updates hidden_states.
            // It also uses/updates its corresponding part of the model_cache.
            // model_cache is Vec<Vec<f32>>, so model_cache[i] is Vec<f32>
            hidden_states = block.forward(&hidden_states, &mut model_cache[i], theta_hat)?;
            // println!("Hidden_states shape after block {}: {:?}", i, hidden_states.shape());
        }
        
        // 4. Final Layer Normalization
        hidden_states = self.ln_f.forward(&hidden_states)?;
        // println!("Hidden_states shape after ln_f: {:?}", hidden_states.shape());

        // 5. Language Model Head (Placeholder)
        // The actual lm_head would be a linear layer mapping hidden_states (n_embd) to vocab_size.
        // For now, returning the processed hidden_states.
        // This needs to be updated to return logits of shape [batch_size, seq_len, vocab_size].
        // Example: let logits = self.lm_head.forward(&hidden_states)?;
        
        // For now, we return hidden_states. The caller in main.rs expects logits.
        // This will be addressed when lm_head is implemented.
        Ok(hidden_states) 
    }
}