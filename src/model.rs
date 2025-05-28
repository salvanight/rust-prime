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
        _layer_kv_cache: &mut Vec<f32>, // Prefixed as it's not used in placeholder
        _theta_hat: f32 
    ) -> Result<ArrayD<f32>, Box<dyn std::error::Error>> {
        // Placeholder implementation for TransformerBlock::forward
        // In a real scenario, this would involve:
        // 1. hidden_states_norm1 = self.ln_1.forward(hidden_states)?
        // 2. attn_output = self.attn.forward(hidden_states_norm1, attention_mask, layer_kv_cache, theta_hat)?
        // 3. hidden_states_attn_added = hidden_states + attn_output 
        // 4. hidden_states_norm2 = self.ln_2.forward(hidden_states_attn_added)?
        // 5. mlp_output = self.mlp.forward(hidden_states_norm2)?
        // 6. final_output = hidden_states_attn_added + mlp_output
        // For now, just pass through the attention and mlp placeholders which clone the input.
        
        // Simulate attention pass (currently clones input)
        let attn_output = self.attn.forward(hidden_states, None)?; 
        // Simulate residual connection
        let x = hidden_states + &attn_output; // Element-wise add if shapes match, or broadcasting. Ndarray handles basic add.
                                              // Ensure hidden_states and attn_output are compatible for addition.
                                              // For placeholder, they are clones, so shapes match.

        // Simulate MLP pass (currently clones input)
        let mlp_output = self.mlp.forward(&x)?;
        // Simulate residual connection
        let final_output = x + &mlp_output;

        Ok(final_output)
        // todo!("Implement TransformerBlock forward pass with cache and theta_hat");
    }
}

#[derive(Debug)]
pub struct GPT2Model {
    wte_weight: ArrayD<f32>, // Token embeddings
    wpe_weight: ArrayD<f32>, // Positional embeddings
    h: Vec<TransformerBlock>,
    ln_f: LayerNorm,
}

impl GPT2Model {
    pub fn new(config: &GPT2Config) -> Result<Self, Box<dyn std::error::Error>> {
        let wte_weight = ArrayD::zeros(IxDyn(&[config.vocab_size as usize, config.n_embd as usize]));
        let wpe_weight = ArrayD::zeros(IxDyn(&[config.n_positions as usize, config.n_embd as usize]));
        
        let mut h = Vec::with_capacity(config.n_layer as usize);
        for _i in 0..config.n_layer {
            h.push(TransformerBlock::new(config)?);
        }
        
        let ln_f = LayerNorm::new()?;
        
        Ok(Self {
            wte_weight,
            wpe_weight,
            h,
            ln_f,
        })
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
        let token_embeddings = ArrayD::zeros(IxDyn(&[batch_size, seq_len, n_embd]));

        // 2. Positional Embeddings
        if seq_len > self.wpe_weight.shape()[0] {
            return Err(format!(
                "Sequence length ({}) exceeds maximum positional embeddings ({})",
                seq_len, self.wpe_weight.shape()[0]
            ).into());
        }
        let positional_embeddings_slice = self.wpe_weight.slice(s![..seq_len, ..]);
        let positional_embeddings_owned: ArrayD<f32> = positional_embeddings_slice.to_owned().into_dyn(); // into_dyn() restored
        let positional_embeddings_broadcastable = positional_embeddings_owned.insert_axis(Axis(0));
        
        let mut hidden_states = token_embeddings + positional_embeddings_broadcastable;
        // println!("Initial hidden_states shape: {:?}", hidden_states.shape());

        // 3. Pass through Transformer Blocks
        for (i, block) in self.h.iter_mut().enumerate() {
            if i >= model_cache.len() {
                // This case should ideally be handled by ensuring model_cache is pre-sized.
                // For safety in this placeholder, one might return an error or skip.
                // For now, assuming model_cache is correctly sized (e.g., in a real scenario,
                // it would be initialized to config.n_layer elements).
                return Err(format!("model_cache does not have enough entries for layer {}", i).into());
            }
            hidden_states = block.forward(&hidden_states, &mut model_cache[i], theta_hat)?;
        }
        
        // 4. Final Layer Normalization
        hidden_states = self.ln_f.forward(&hidden_states)?;

        // 5. Language Model Head (Placeholder)
        // For testing, we need to ensure the output shape is [batch_size, seq_len, vocab_size].
        // The current hidden_states is [batch_size, seq_len, n_embd].
        // We'll create a dummy projection to vocab_size.
        // A real lm_head often shares weights with wte or is a separate Linear layer.
        let vocab_size = self.wte_weight.shape()[0]; // vocab_size from wte
        // Create a dummy weight for projection: [n_embd, vocab_size]
        // let lm_head_weight = ArrayD::zeros(IxDyn(&[n_embd, vocab_size]));
        // Perform a dot product for each token embedding.
        // This is a simplified linear projection.
        // hidden_states [B, S, E] x lm_head_weight [E, V] -> logits [B, S, V]
        
        // For now, to avoid implementing full linear layer here for testing,
        // we'll construct a zero tensor of the correct logit shape.
        // This makes the smoke test for GPT2Model.forward focus on flow and shapes,
        // not on correctness of lm_head projection values.
        let logits = ArrayD::zeros(IxDyn(&[batch_size, seq_len, vocab_size]));
        
        Ok(logits) 
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GPT2Config;
    use ndarray::Array2;

    // Updated create_test_config to align with src/config.rs::GPT2Config
    fn create_test_config(n_embd: i32, n_head: i32, n_layer: i32, vocab_size: i32, n_positions: i32) -> GPT2Config {
        GPT2Config {
            vocab_size,
            n_layer,
            n_head,
            n_embd,
            n_positions, 
            eos_token_id: 0, 
            bos_token_id: 0, 
            layer_norm_epsilon: 1e-5f32, // Changed to f32
            n_inner: Some(4 * n_embd), 
            // Fill in other mandatory fields from src/config.rs::GPT2Config with defaults
            activation_function: "gelu".to_string(),
            resid_pdrop: 0.1,
            embd_pdrop: 0.1,
            attn_pdrop: 0.1,
            initializer_range: 0.02,
            summary_type: "cls_index".to_string(),
            summary_use_proj: true,
            summary_activation: None,
            summary_proj_to_labels: None,
            summary_first_dropout: None,
            scale_attn_weights: Some(true),
            use_cache: Some(true),
            model_type: "gpt2".to_string(),
        }
    }

    #[test]
    fn test_transformer_block_new() {
        let config = create_test_config(4, 2, 1, 10, 10); // n_embd=4, n_head=2
        let block_result = TransformerBlock::new(&config);
        assert!(block_result.is_ok(), "TransformerBlock::new failed: {:?}", block_result.err());
    }

    #[test]
    fn test_transformer_block_forward_shape() {
        let config = create_test_config(4, 2, 1, 10, 10);
        let mut block = TransformerBlock::new(&config).unwrap();
        
        let batch_size = 1;
        let seq_len = 5;
        let n_embd = config.n_embd as usize;
        let hidden_states = ArrayD::zeros(IxDyn(&[batch_size, seq_len, n_embd]));
        
        // Dummy cache for one layer. The actual structure of Vec<f32> for cache is simplified.
        // A real cache might be more complex (e.g., storing K and V tensors).
        // For the placeholder `block.forward`, its content doesn't matter much.
        let mut layer_kv_cache: Vec<f32> = Vec::new(); 
        let theta_hat = 1.0;

        let output_result = block.forward(&hidden_states, &mut layer_kv_cache, theta_hat);
        assert!(output_result.is_ok(), "TransformerBlock::forward failed: {:?}", output_result.err());
        let output = output_result.unwrap();
        assert_eq!(output.shape(), &[batch_size, seq_len, n_embd], "TransformerBlock forward output shape mismatch");
    }

    #[test]
    fn test_gpt2model_new() {
        let config = create_test_config(4, 2, 1, 10, 10); // 1 layer
        let model_result = GPT2Model::new(&config);
        assert!(model_result.is_ok(), "GPT2Model::new failed: {:?}", model_result.err());
        let model = model_result.unwrap();
        assert_eq!(model.h.len(), config.n_layer as usize, "Incorrect number of transformer blocks");
        assert_eq!(model.wte_weight.shape(), &[config.vocab_size as usize, config.n_embd as usize]);
        assert_eq!(model.wpe_weight.shape(), &[config.n_positions as usize, config.n_embd as usize]);
    }

    #[test]
    fn test_gpt2model_forward_smoke_test() {
        let config = create_test_config(4, 2, 1, 10, 10); // n_embd=4, n_head=2, n_layer=1
        let mut model = GPT2Model::new(&config).unwrap();
        
        let batch_size = 1;
        let seq_len = 5;
        let input_ids_data = vec![0i32; batch_size * seq_len]; // Dummy token IDs
        let input_ids = Array2::from_shape_vec((batch_size, seq_len), input_ids_data).unwrap();
        
        // Initialize a dummy model_cache. It's Vec<Vec<f32>>.
        // One inner Vec<f32> per layer.
        let mut model_cache: ModelKVCache = vec![Vec::new(); config.n_layer as usize];
        let theta_hat = 1.0;

        let output_result = model.forward(&input_ids, &mut model_cache, theta_hat);
        assert!(output_result.is_ok(), "GPT2Model::forward failed: {:?}", output_result.err());
        let output = output_result.unwrap();
        
        // Expected output shape: [batch_size, seq_len, vocab_size]
        assert_eq!(output.shape(), &[batch_size, seq_len, config.vocab_size as usize], "GPT2Model forward output shape mismatch (logits)");
    }
}