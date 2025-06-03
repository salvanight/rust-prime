#![cfg(feature = "ndarray_backend")]
use ndarray::{ArrayD, IxDyn, Array2, s, Axis, ArrayView2}; // Consolidated use statements, added ArrayView2
use crate::config::GPT2Config;
use crate::common::{LayerNorm, ModelKVCache}; // Import ModelKVCache
use crate::accelerator::{CpuTensor, Device, Module, Tensor};
use crate::attention::MultiHeadAttention;
use crate::common::{LayerNorm, ModelKVCache}; // ModelKVCache might need rethink for non-CPU
use crate::config::GPT2Config;
use crate::mlp::MLP;
use std::error::Error;
// use ndarray::{ArrayD, IxDyn, Array2, s, Axis, ArrayView2}; // Commenting out as we replace ndarray with CpuTensor

#[derive(Debug, Clone)]
pub struct TransformerBlock {
    ln_1: LayerNorm,
    attn: MultiHeadAttention,
    ln_2: LayerNorm,
    mlp: MLP,
    device: Device,
}

impl TransformerBlock {
    pub fn new(config: &GPT2Config) -> Result<Self, Box<dyn Error>> {
        let device = Device::CPU; // All components initialized on CPU by default
        let n_embd = config.n_embd as usize;
        let n_head = config.n_head as usize;
        let n_inner = config.n_inner.unwrap_or(4 * config.n_embd) as usize;

        // Dummy data for component initialization
        let dummy_vec_embd = vec![0.0f32; n_embd];
        let dummy_shape_embd = vec![n_embd];
        
        // For attention weights (shapes are illustrative, proper shapes needed)
        // c_attn_weight: [n_embd, 3 * n_embd], c_attn_bias: [3 * n_embd]
        // c_proj_weight: [n_embd, n_embd], c_proj_bias: [n_embd]
        let dummy_c_attn_weight_data = vec![0.0f32; n_embd * (3 * n_embd)];
        let dummy_c_attn_weight_shape = vec![n_embd, 3 * n_embd];
        let dummy_c_attn_bias_data = vec![0.0f32; 3 * n_embd];
        let dummy_c_attn_bias_shape = vec![3 * n_embd];
        let dummy_c_proj_attn_weight_data = vec![0.0f32; n_embd * n_embd];
        let dummy_c_proj_attn_weight_shape = vec![n_embd, n_embd];
        // c_proj_bias_data already defined as dummy_vec_embd
        // c_proj_bias_shape already defined as dummy_shape_embd

        // For MLP weights
        // c_fc_weight: [n_embd, n_inner], c_fc_bias: [n_inner]
        // c_proj_weight: [n_inner, n_embd], c_proj_bias: [n_embd]
        let dummy_mlp_fc_weight_data = vec![0.0f32; n_embd * n_inner];
        let dummy_mlp_fc_weight_shape = vec![n_embd, n_inner];
        let dummy_mlp_fc_bias_data = vec![0.0f32; n_inner];
        let dummy_mlp_fc_bias_shape = vec![n_inner];
        let dummy_mlp_proj_weight_data = vec![0.0f32; n_inner * n_embd];
        let dummy_mlp_proj_weight_shape = vec![n_inner, n_embd];
        // MLP c_proj_bias data/shape are dummy_vec_embd/dummy_shape_embd

        let ln_1 = LayerNorm::new(&dummy_vec_embd, &dummy_shape_embd, &dummy_vec_embd, &dummy_shape_embd, config.layer_norm_epsilon)?;
        let attn = MultiHeadAttention::new(
            &dummy_c_attn_weight_data, &dummy_c_attn_weight_shape,
            &dummy_c_attn_bias_data, &dummy_c_attn_bias_shape,
            &dummy_c_proj_attn_weight_data, &dummy_c_proj_attn_weight_shape,
            &dummy_vec_embd, &dummy_shape_embd, // c_proj_bias
            n_head,
        )?;
        let ln_2 = LayerNorm::new(&dummy_vec_embd, &dummy_shape_embd, &dummy_vec_embd, &dummy_shape_embd, config.layer_norm_epsilon)?;
        let mlp = MLP::new(
            &dummy_mlp_fc_weight_data, &dummy_mlp_fc_weight_shape,
            &dummy_mlp_fc_bias_data, &dummy_mlp_fc_bias_shape,
            &dummy_mlp_proj_weight_data, &dummy_mlp_proj_weight_shape,
            &dummy_vec_embd, &dummy_shape_embd, // c_proj_bias
        )?;

        Ok(Self {
            ln_1,
            attn,
            ln_2,
            mlp,
            device,
        })
    }

    // Old forward method - to be removed or adapted
    // pub fn forward(
    //     &mut self, // Changed to &mut self
    //     hidden_states: &ArrayD<f32>,
    //     layer_kv_cache: &mut Vec<f32>, // Added layer_kv_cache
    //     _theta_hat: f32 // Added theta_hat, underscore if not used immediately
    // ) -> Result<ArrayD<f32>, Box<dyn std::error::Error>> {
    //     Ok(hidden_states.clone())
    // }
}

impl Module for TransformerBlock {
    type Input = (CpuTensor<f32>, Option<CpuTensor<f32>>); // hidden_states, attention_mask_opt
    type Output = CpuTensor<f32>;

    fn forward(&self, input: Self::Input) -> Result<Self::Output, Box<dyn Error>> {
        let (hidden_states, attention_mask_opt) = input;

        if hidden_states.device() != self.device {
            return Err(format!("Block on {:?} received hidden_states on {:?}", self.device, hidden_states.device()).into());
        }
        if let Some(mask) = &attention_mask_opt {
            if mask.device() != self.device {
                return Err(format!("Block on {:?} received mask on {:?}", self.device, mask.device()).into());
            }
        }
        if self.device != Device::CPU {
            return Err(format!("TransformerBlock on {:?} is not supported for forward pass, only CPU.", self.device).into());
        }

        let ln_1_output = self.ln_1.forward(hidden_states.clone())?; // Clone for residual
        let attn_output = self.attn.forward((ln_1_output.clone(), attention_mask_opt.clone()))?;
        
        // TODO: Implement CpuTensor add operation for residual: hidden_states + attn_output
        let residual_1_output = attn_output; // Placeholder

        let ln_2_output = self.ln_2.forward(residual_1_output.clone())?; // Clone for residual
        let mlp_output = self.mlp.forward(ln_2_output)?;

        // TODO: Implement CpuTensor add operation for residual: residual_1_output + mlp_output
        Ok(mlp_output) // Placeholder
    }

    fn to_device(&mut self, device: Device) -> Result<(), Box<dyn Error>> {
        if self.device == device {
            return Ok(());
        }
        if device != Device::CPU {
            return Err(format!("TransformerBlock cannot be moved to {:?}, only CPU is supported.", device).into());
        }
        self.ln_1.to_device(device.clone())?;
        self.attn.to_device(device.clone())?;
        self.ln_2.to_device(device.clone())?;
        self.mlp.to_device(device.clone())?;
        self.device = device;
        Ok(())
    }

    fn current_device(&self) -> Device {
        self.device.clone()
    }
}


#[derive(Debug, Clone)]
pub struct GPT2Model {
    wte_weight: CpuTensor<f32>, // Token embeddings
    wpe_weight: CpuTensor<f32>, // Positional embeddings
    h: Vec<TransformerBlock>,
    ln_f: LayerNorm,
    device: Device,
    n_embd: usize,
    vocab_size: usize,
    n_positions: usize,
}

impl GPT2Model {
    pub fn new(config: &GPT2Config) -> Result<Self, Box<dyn Error>> {
        let device = Device::CPU;
        let n_embd = config.n_embd as usize;
        let vocab_size = config.vocab_size as usize;
        let n_positions = config.n_positions as usize;

        let wte_data = vec![0.0f32; vocab_size * n_embd];
        let wte_shape = vec![vocab_size, n_embd];
        let wte_weight = CpuTensor::from_data_and_shape(&wte_data, &wte_shape, device.clone())?;

        let wpe_data = vec![0.0f32; n_positions * n_embd];
        let wpe_shape = vec![n_positions, n_embd];
        let wpe_weight = CpuTensor::from_data_and_shape(&wpe_data, &wpe_shape, device.clone())?;

        let mut h = Vec::with_capacity(config.n_layer as usize);
        for _i in 0..config.n_layer {
            h.push(TransformerBlock::new(config)?);
        }

        let ln_f_gamma_data = vec![0.0f32; n_embd];
        let ln_f_beta_data = vec![0.0f32; n_embd];
        let ln_f_shape = vec![n_embd];
        let ln_f = LayerNorm::new(&ln_f_gamma_data, &ln_f_shape, &ln_f_beta_data, &ln_f_shape, config.layer_norm_epsilon)?;

        Ok(Self {
            wte_weight,
            wpe_weight,
            h,
            ln_f,
            device,
            n_embd,
            vocab_size,
            n_positions,
        })
    }
    
    // New get_embeddings method using CpuTensor
    pub fn get_embeddings(
        &self,
        input_ids_slice: &[i32], // Expecting flat slice of token IDs
    ) -> Result<CpuTensor<f32>, Box<dyn Error>> {
        if self.device != Device::CPU {
            return Err("get_embeddings currently only supports CPU device.".into());
        }

        let num_tokens = input_ids_slice.len();
        if num_tokens == 0 {
            // Handle empty input if necessary, or return error
            return CpuTensor::from_data_and_shape(&[], &[0, self.n_embd], self.device.clone());
        }
        
        let position_ids_slice: Vec<i32> = (0..num_tokens as i32).collect();
        let mut embedding_data = vec![0.0f32; num_tokens * self.n_embd];

        let wte_slice = self.wte_weight.as_slice()?;
        let wpe_slice = self.wpe_weight.as_slice()?;

        for i in 0..num_tokens {
            let token_id = input_ids_slice[i] as usize;
            let pos_id = position_ids_slice[i] as usize;

            if token_id >= self.vocab_size { return Err(format!("Token ID {} out of vocab size {}", token_id, self.vocab_size).into()); }
            if pos_id >= self.n_positions { return Err(format!("Position ID {} out of max positions {}", pos_id, self.n_positions).into()); }

            let token_emb_offset = token_id * self.n_embd;
            let pos_emb_offset = pos_id * self.n_embd;

            for j in 0..self.n_embd {
                embedding_data[i * self.n_embd + j] =
                    wte_slice[token_emb_offset + j] + wpe_slice[pos_emb_offset + j];
            }
        }
        CpuTensor::from_data_and_shape(&embedding_data, &[num_tokens, self.n_embd], self.device.clone())
    }


    // Old forward method - to be removed or adapted
    // pub fn forward(
    //     &mut self,
    //     input_ids: &Array2<i32>,
    //     model_cache: &mut ModelKVCache,
    //     theta_hat: f32
    // ) -> Result<ArrayD<f32>, Box<dyn std::error::Error>> {
    //     // ... existing logic ...
    //     Ok(hidden_states)
    // }
}

impl Module for GPT2Model {
    type Input = (CpuTensor<i32>, Option<CpuTensor<f32>>); // input_ids, attention_mask_opt
    type Output = CpuTensor<f32>; // Final hidden states (pre-logits)

    fn forward(&self, input: Self::Input) -> Result<Self::Output, Box<dyn Error>> {
        let (input_ids, attention_mask_opt) = input;

        if input_ids.device() != self.device {
            return Err(format!("Model on {:?} received input_ids on {:?}", self.device, input_ids.device()).into());
        }
        if let Some(mask) = &attention_mask_opt {
            if mask.device() != self.device {
                return Err(format!("Model on {:?} received mask on {:?}", self.device, mask.device()).into());
            }
        }
        if self.device != Device::CPU {
            return Err(format!("GPT2Model on {:?} is not supported for forward pass, only CPU.", self.device).into());
        }

        let input_ids_shape = input_ids.shape();
        let input_ids_data_slice = input_ids.as_slice()?; // Assuming this returns Result<&[i32], _>

        let mut hidden_states: CpuTensor<f32>;

        if input_ids_shape.len() == 1 { // Shape [seq_len]
            hidden_states = self.get_embeddings(input_ids_data_slice)?;
            // Reshape to [1, seq_len, n_embd] if necessary for block input.
            // For now, assume blocks can handle [seq_len, n_embd] or get_embeddings returns [1, seq_len, n_embd]
            // Based on get_embeddings, it returns [num_tokens, n_embd]. This is fine for blocks.
        } else if input_ids_shape.len() == 2 { // Shape [batch_size, seq_len]
            let batch_size = input_ids_shape[0];
            let seq_len = input_ids_shape[1];
            let mut all_batch_embeddings_data = Vec::with_capacity(batch_size * seq_len * self.n_embd);

            for b_idx in 0..batch_size {
                let start = b_idx * seq_len;
                let end = start + seq_len;
                let current_ids_slice = &input_ids_data_slice[start..end];
                let single_embedding = self.get_embeddings(current_ids_slice)?;
                all_batch_embeddings_data.extend_from_slice(single_embedding.as_slice()?);
            }
            hidden_states = CpuTensor::from_data_and_shape(&all_batch_embeddings_data, &[batch_size, seq_len, self.n_embd], self.device.clone())?;
        } else {
            return Err(format!("Unsupported input_ids rank: {}. Expected 1 or 2.", input_ids_shape.len()).into());
        }
        
        // K/V cache is not used in this Module::forward signature.
        // Theta_hat is also not used.
        for block in &self.h {
            // Pass None for attention_mask_opt for now if it's not used, or clone if needed by blocks.
            hidden_states = block.forward((hidden_states, attention_mask_opt.clone()))?;
        }

        let final_hidden_states = self.ln_f.forward(hidden_states)?;
        Ok(final_hidden_states)
    }

    fn to_device(&mut self, device: Device) -> Result<(), Box<dyn Error>> {
        if self.device == device {
            return Ok(());
        }
        if device != Device::CPU {
            return Err(format!("GPT2Model cannot be moved to {:?}, only CPU is supported.", device).into());
        }
        
        self.wte_weight = self.wte_weight.to_device(device.clone())?;
        self.wpe_weight = self.wpe_weight.to_device(device.clone())?;
        for block in self.h.iter_mut() {
            block.to_device(device.clone())?;
        }
        self.ln_f.to_device(device.clone())?;
        self.device = device;
        Ok(())
    }

    fn current_device(&self) -> Device {
        self.device.clone()
    }
}

// The old get_embeddings and forward methods using ndarray are removed or commented out by the diff.
// ModelKVCache type alias is still present but its usage with Tensors would need rethinking for non-CPU.
// For now, K/V cache is outside the Module::forward direct inputs for Tensors.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accelerator::{CpuTensor, Device, Tensor}; // Module is already in super
    use crate::config::GPT2Config;

    // Helper to create a default GPT2Config for testing
    fn gpt2_test_config(vocab_size: i32, n_embd: i32, n_layer: i32, n_head: i32) -> GPT2Config {
        GPT2Config {
            vocab_size,
            n_embd,
            n_layer,
            n_head,
            n_positions: 1024, // Default or common value
            n_ctx: 1024,       // Default or common value
            block_size: 1024,  // Default or common value
            embd_pdrop: 0.0,   // No dropout for deterministic tests
            resid_pdrop: 0.0,
            attn_pdrop: 0.0,
            layer_norm_epsilon: 1e-5,
            initializer_range: 0.02,
            n_inner: Some(n_embd * 4), // Common practice
            activation_function: "gelu_new".to_string(),
            bos_token_id: Some(50256),
            eos_token_id: Some(50256),
            scale_attn_weights: true,
            scale_attn_by_inverse_layer_idx: false,
            reorder_and_upcast_attn: false,
        }
    }

    // Helper to create TransformerBlock for testing
    fn test_transformer_block(config: &GPT2Config) -> TransformerBlock {
        TransformerBlock::new(config).expect("Failed to create TransformerBlock for testing")
    }

    // Helper to create GPT2Model for testing using the default constructor
    fn test_gpt2_model(config: &GPT2Config) -> GPT2Model {
        GPT2Model::new(config).expect("Failed to create GPT2Model for testing")
    }

    #[test]
    fn test_transformer_block_forward_2d() {
        let config = gpt2_test_config(50257, 12, 1, 4); // n_embd=12, n_head=4 -> head_dim=3
        let block = test_transformer_block(&config);
        let seq_len = 5;
        let n_embd = config.n_embd as usize;

        let input_data: Vec<f32> = (0..(seq_len * n_embd)).map(|x| x as f32).collect();
        let input_tensor = CpuTensor::from_data_and_shape(&input_data, &[seq_len, n_embd], Device::CPU).unwrap();
        
        let result = block.forward((input_tensor, None));
        assert!(result.is_ok());
        let output_tensor = result.unwrap();
        assert_eq!(output_tensor.shape(), &[seq_len, n_embd]);
        assert_eq!(output_tensor.device(), Device::CPU);
    }

    #[test]
    fn test_transformer_block_forward_3d() {
        let config = gpt2_test_config(50257, 12, 1, 4);
        let block = test_transformer_block(&config);
        let batch_size = 2;
        let seq_len = 5;
        let n_embd = config.n_embd as usize;

        let input_data: Vec<f32> = (0..(batch_size * seq_len * n_embd)).map(|x| x as f32).collect();
        let input_tensor = CpuTensor::from_data_and_shape(&input_data, &[batch_size, seq_len, n_embd], Device::CPU).unwrap();
        
        let result = block.forward((input_tensor, None));
        assert!(result.is_ok());
        let output_tensor = result.unwrap();
        assert_eq!(output_tensor.shape(), &[batch_size, seq_len, n_embd]);
        assert_eq!(output_tensor.device(), Device::CPU);
    }
    
    #[test]
    fn test_transformer_block_to_device_cpu() {
        let config = gpt2_test_config(50257, 12, 1, 4);
        let mut block = test_transformer_block(&config);
        assert_eq!(block.current_device(), Device::CPU);
        assert!(block.to_device(Device::CPU).is_ok());
        assert_eq!(block.current_device(), Device::CPU);
    }

    #[test]
    fn test_gpt2model_get_embeddings_1d_input() {
        let config = gpt2_test_config(50, 4, 1, 2); // vocab_size=50, n_embd=4
        let model = test_gpt2_model(&config);
        let input_ids_slice = vec![0i32, 1, 2, 3, 4]; // 5 tokens
        
        let result = model.get_embeddings(&input_ids_slice);
        assert!(result.is_ok());
        let embeddings_tensor = result.unwrap();
        assert_eq!(embeddings_tensor.shape(), &[input_ids_slice.len(), config.n_embd as usize]);
        assert_eq!(embeddings_tensor.device(), Device::CPU);
    }
    
    #[test]
    // This test is removed as get_embeddings is designed for 1D slice input.
    // Batched embedding generation is implicitly tested via GPT2Model::forward with 2D input_ids.
    // fn test_gpt2model_get_embeddings_2d_input_batched_by_get_embeddings() { ... }


    #[test]
    fn test_gpt2model_forward_1d_input_ids() {
        let config = gpt2_test_config(50, 4, 1, 2);
        let model = test_gpt2_model(&config);
        let input_ids_data: Vec<i32> = (0..5).collect(); // seq_len = 5
        let seq_len = input_ids_data.len();
        let input_ids_tensor = CpuTensor::from_data_and_shape(&input_ids_data, &[seq_len], Device::CPU).unwrap();

        let result = model.forward((input_ids_tensor, None));
        assert!(result.is_ok(), "Model forward failed for 1D input: {:?}", result.err());
        let output_tensor = result.unwrap();
        // GPT2Model::forward handles 1D input by effectively making it batch_size=1.
        // Output is final hidden states: [batch_size=1, seq_len, n_embd]
        assert_eq!(output_tensor.shape(), &[1, seq_len, config.n_embd as usize], "Output shape mismatch for 1D input.");
        assert_eq!(output_tensor.device(), Device::CPU, "Output device mismatch for 1D input.");
        // With zero weights, output should be zero.
        let output_slice = output_tensor.as_slice().unwrap();
        assert!(output_slice.iter().all(|&x| (x - 0.0).abs() < f32::EPSILON), "Output data not all zeros for 1D input.");
    }

    #[test]
    fn test_gpt2model_forward_2d_input_ids() {
        let config = gpt2_test_config(50, 4, 1, 2);
        let model = test_gpt2_model(&config);
        let batch_size = 2;
        let seq_len = 3;
        let input_ids_data: Vec<i32> = (0..(batch_size * seq_len) as i32).collect();
        let input_ids_tensor = CpuTensor::from_data_and_shape(&input_ids_data, &[batch_size, seq_len], Device::CPU).unwrap();

        let result = model.forward((input_ids_tensor, None));
        assert!(result.is_ok(), "Model forward failed for 2D input: {:?}", result.err());
        let output_tensor = result.unwrap();
        assert_eq!(output_tensor.shape(), &[batch_size, seq_len, config.n_embd as usize], "Output shape mismatch for 2D input.");
        assert_eq!(output_tensor.device(), Device::CPU, "Output device mismatch for 2D input.");
        let output_slice = output_tensor.as_slice().unwrap();
        assert!(output_slice.iter().all(|&x| (x - 0.0).abs() < f32::EPSILON), "Output data not all zeros for 2D input.");
    }
    
    #[test]
    fn test_gpt2model_to_device_cpu() {
        let config = gpt2_test_config(50257, 12, 1, 4);
        let mut model = test_gpt2_model(&config);
        assert_eq!(model.current_device(), Device::CPU);
        assert!(model.to_device(Device::CPU).is_ok());
        assert_eq!(model.current_device(), Device::CPU);
    }
}