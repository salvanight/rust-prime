use ndarray::{ArrayD, IxDyn, Array2, s, Axis}; // Consolidated use statements
use crate::config::GPT2Config;
use crate::common::LayerNorm;
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
        &self, 
        hidden_states: &ArrayD<f32>, 
        attention_mask: Option<&ArrayD<f32>>
    ) -> Result<ArrayD<f32>, Box<dyn std::error::Error>> {
        // To avoid unused variable warning if not used yet:
        if attention_mask.is_some() {} 
        println!("TransformerBlock forward called with hidden_states shape: {:?}", hidden_states.shape());
        todo!("Implement TransformerBlock forward pass");
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
        let wte_weight = ArrayD::zeros((config.vocab_size as usize, config.n_embd as usize).into_dyn());
        let wpe_weight = ArrayD::zeros((config.n_positions as usize, config.n_embd as usize).into_dyn());
        
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
        &self, 
        input_ids: &Array2<i32>, 
        _attention_mask: Option<&ArrayD<f32>> // Underscore to avoid unused warning for now
    ) -> Result<ArrayD<f32>, Box<dyn std::error::Error>> {
        let batch_size = input_ids.shape()[0];
        let seq_len = input_ids.shape()[1];
        
        // n_embd is the dimensionality of the embeddings.
        // self.wte_weight is ArrayD but used as 2D [vocab_size, n_embd].
        let n_embd = self.wte_weight.shape()[1]; 

        // 1. Token Embeddings (Placeholder: creating zeros)
        let token_embeddings = ArrayD::zeros((batch_size, seq_len, n_embd).into_dyn());
        // println!("Token embeddings shape: {:?}", token_embeddings.shape());

        // 2. Positional Embeddings
        if seq_len > self.wpe_weight.shape()[0] {
            return Err(format!(
                "Sequence length ({}) exceeds maximum positional embeddings ({})",
                seq_len, self.wpe_weight.shape()[0]
            ).into());
        }
        // Slice wpe_weight to get embeddings for the current sequence length.
        // wpe_weight is conceptually [n_positions, n_embd]. Slice is [seq_len, n_embd].
        let positional_embeddings_slice = self.wpe_weight.slice(s![..seq_len, ..]);
        
        // Convert ArrayView to ArrayD for addition and broadcasting.
        // The slice is 2D [seq_len, n_embd]. We need to make it [1, seq_len, n_embd] to broadcast.
        let positional_embeddings_owned: ArrayD<f32> = positional_embeddings_slice.to_owned().into_dyn();
        let positional_embeddings_broadcastable = positional_embeddings_owned.insert_axis(Axis(0));
        // println!("Positional embeddings broadcastable shape: {:?}", positional_embeddings_broadcastable.shape());
        
        // 3. Add token and positional embeddings
        // token_embeddings: [batch_size, seq_len, n_embd]
        // positional_embeddings_broadcastable: [1, seq_len, n_embd]
        // Resulting inputs_embeds: [batch_size, seq_len, n_embd]
        let inputs_embeds = token_embeddings + positional_embeddings_broadcastable;
        // println!("Combined input embeddings shape: {:?}", inputs_embeds.shape());
        
        // As per subtask: "Return Ok(inputs_embeds) for now, or a placeholder for the final hidden state."
        Ok(inputs_embeds) 
    }
}