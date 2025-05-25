use ndarray::{Array, ArrayD, Array2, Ix1, Ix2, ArrayView1, IxDyn, s, Axis, ShapeError}; // Added Ix1, ArrayView1, IxDyn
use safetensors::SafeTensors;
use tokenizers::Tokenizer;
// BPE import moved to test module below
use std::convert::TryInto;
use std::fs::File;
use std::io::Read;
use crate::config::GPT2Config;
use crate::common::LayerNorm;
use crate::attention::MultiHeadAttention;
use crate::mlp::MLP;

#[derive(Debug)]
pub struct TransformerBlock {
    pub(crate) ln_1: LayerNorm,
    #[allow(dead_code)] // Will be used when attention weights are loaded
    pub(crate) attn: MultiHeadAttention,
    pub(crate) ln_2: LayerNorm,
    pub(crate) mlp: MLP,
}

impl TransformerBlock {
    pub fn new(config: &GPT2Config) -> Result<Self, Box<dyn std::error::Error>> {
        let ln_1 = LayerNorm::new(config.n_embd, config.layer_norm_epsilon)?;
        let attn = MultiHeadAttention::new(config.n_head, config.n_embd)?;
        let ln_2 = LayerNorm::new(config.n_embd, config.layer_norm_epsilon)?;
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
        // Self-Attention Path
        let ln_1_output = self.ln_1.forward(hidden_states)?;
        let attn_output = self.attn.forward(&ln_1_output, attention_mask)?;
        // Add the first residual connection
        // hidden_states is &ArrayD<f32>, attn_output is ArrayD<f32>
        // The + operator for &ArrayD + &ArrayD produces an owned ArrayD.
        let residual_1 = hidden_states + &attn_output;

        // Feed-Forward Path
        let ln_2_output = self.ln_2.forward(&residual_1)?;
        let mlp_output = self.mlp.forward(&ln_2_output)?;
        // Add the second residual connection
        // residual_1 is ArrayD<f32>, mlp_output is ArrayD<f32>
        let output = &residual_1 + &mlp_output;
        
        Ok(output)
    }
}

#[derive(Debug)]
pub struct GPT2Model {
    pub(crate) wte_weight: ArrayD<f32>, // Token embeddings
    pub(crate) wpe_weight: ArrayD<f32>, // Positional embeddings
    pub(crate) h: Vec<TransformerBlock>,
    pub(crate) ln_f: LayerNorm,
}

impl GPT2Model {
    pub fn new(config: &GPT2Config) -> Result<Self, Box<dyn std::error::Error>> {
        let wte_weight = Array::zeros((config.vocab_size as usize, config.n_embd as usize)).into_dyn();
        let wpe_weight = Array::zeros((config.n_positions as usize, config.n_embd as usize)).into_dyn();
        
        let mut h = Vec::with_capacity(config.n_layer as usize);
        for _i in 0..config.n_layer {
            h.push(TransformerBlock::new(config)?);
        }
        
        let ln_f = LayerNorm::new(config.n_embd, config.layer_norm_epsilon)?;
        
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
        let n_embd = self.wte_weight.shape()[1]; // n_embd is usize here

        // 1. Token Embeddings
        let mut token_embeddings_arr3 = Array::zeros((batch_size, seq_len, n_embd)); // Array3<f32>
        for b in 0..batch_size {
            for s_idx in 0..seq_len {
                let token_id = input_ids[[b, s_idx]] as usize;
                if token_id >= self.wte_weight.shape()[0] {
                    return Err(format!("Token ID {} at [{},{}] is out of vocab size {}", 
                                       token_id, b, s_idx, self.wte_weight.shape()[0]).into());
                }
                let embedding_vector_view = self.wte_weight.slice(s![token_id, ..]);
                token_embeddings_arr3.slice_mut(s![b, s_idx, ..]).assign(&embedding_vector_view);
            }
        }
        let mut hidden_states = token_embeddings_arr3.into_dyn(); // Now ArrayD<f32>

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
        hidden_states = hidden_states + positional_embeddings_broadcastable;
        
        // Process Through Transformer Blocks
        for block in &self.h {
            hidden_states = block.forward(&hidden_states, _attention_mask)?;
        }

        // Apply Final Layer Normalization
        hidden_states = self.ln_f.forward(&hidden_states)?;
        
        Ok(hidden_states)
    }

    pub fn lm_head(&self, final_hidden_states: &ArrayD<f32>) -> Result<ArrayD<f32>, Box<dyn std::error::Error>> {
        let initial_shape = final_hidden_states.shape();
        if initial_shape.len() != 3 {
            return Err(format!(
                "Expected final_hidden_states to be 3D (batch, seq_len, n_embd), got shape: {:?}",
                initial_shape
            )
            .into());
        }
        let batch_size = initial_shape[0];
        let seq_len = initial_shape[1];
        let n_embd = initial_shape[2]; // This is usize

        let vocab_size = self.wte_weight.shape()[0]; // This is usize

        // Prepare wte_weight for dot product: view as 2D and transpose
        // self.wte_weight has shape [vocab_size, n_embd]
        // wte_transposed_view will have shape [n_embd, vocab_size]
        let wte_2d_view = self.wte_weight.view().into_dimensionality::<Ix2>()
            .map_err(|e| format!("Failed to view wte_weight as 2D: {}. Shape was {:?}", e, self.wte_weight.shape()))?;
        let wte_transposed_view = wte_2d_view.t();

        // Reshape final_hidden_states to [batch_size * seq_len, n_embd]
        let reshaped_hidden_states = final_hidden_states
            .view()
            .into_shape((batch_size * seq_len, n_embd))
            .map_err(|e: ShapeError| format!("Error reshaping hidden_states: {}. Original shape: {:?}, Target shape: ({}, {})", 
                                             e, initial_shape, batch_size * seq_len, n_embd))?;

        // Perform Dot Product: [B*S, E] @ [E, V] -> [B*S, V]
        let logits_2d = reshaped_hidden_states.dot(&wte_transposed_view);
        let original_logits_2d_shape = logits_2d.shape().to_vec(); // Clone shape before move

        // Reshape logits back to [batch_size, seq_len, vocab_size]
        let logits = logits_2d
            .into_shape((batch_size, seq_len, vocab_size))
            .map_err(|e: ShapeError| format!("Error reshaping logits: {}. Original shape: {:?}, Target shape: ({}, {}, {})", 
                                             e, original_logits_2d_shape, batch_size, seq_len, vocab_size))?
            .into_dyn();

        Ok(logits)
    }
}

// Helper function for argmax (remains unchanged, placed here for context)
fn argmax(array_view: ArrayView1<f32>) -> usize {
    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;
    for (idx, &val) in array_view.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = idx;
        }
    }
    max_idx
}


// Helper function for loading a specific tensor
fn load_specific_tensor(
    tensors: &SafeTensors, 
    name: &str, 
    expected_shape_usize: &[usize]
) -> Result<ArrayD<f32>, Box<dyn std::error::Error>> {
    let tensor_view = tensors.tensor(name)
        .map_err(|e| format!("Tensor '{}' not found: {}. Available tensors: {:?}", name, e, tensors.names()))?;

    if tensor_view.shape() != expected_shape_usize {
        return Err(format!(
            "Shape mismatch for tensor '{}'. Expected {:?}, got {:?}.",
            name, expected_shape_usize, tensor_view.shape()
        ).into());
    }

    if tensor_view.dtype() != safetensors::Dtype::F32 {
        return Err(format!(
            "Data type mismatch for tensor '{}'. Expected F32, got {:?}.",
            name, tensor_view.dtype()
        ).into());
    }

    let data_bytes = tensor_view.data();
    if data_bytes.len() != expected_shape_usize.iter().product::<usize>() * std::mem::size_of::<f32>() {
        return Err(format!(
            "Data size mismatch for tensor '{}'. Byte length {} does not match expected elements {} * size_of(f32).",
            name, data_bytes.len(), expected_shape_usize.iter().product::<usize>()
        ).into());
    }
    
    let data_f32: Vec<f32> = data_bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("Chunk size should be size_of(f32)")))
        .collect();
    
    ArrayD::from_shape_vec(IxDyn(expected_shape_usize), data_f32)
        .map_err(|e| format!("Failed to create ArrayD for tensor '{}': {}", name, e).into())
}


impl GPT2Model {
    // ... existing new, forward, lm_head methods ...

    pub fn load_weights(&mut self, model_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::open(model_path)
            .map_err(|e| format!("Failed to open model file {}: {}", model_path, e))?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| format!("Failed to read model file {}: {}", model_path, e))?;
        
        let tensors_map = SafeTensors::deserialize(&buffer)
            .map_err(|e| format!("Failed to deserialize safetensors from buffer (path: {}): {}", model_path, e))?;

        println!("Available tensors in the file:");
        for name in tensors_map.names() {
            if let Ok(tensor_view) = tensors_map.tensor(name) {
                println!("- {}: shape={:?}, dtype={:?}", name, tensor_view.shape(), tensor_view.dtype());
            } else {
                println!("- {}: Error retrieving tensor view", name);
            }
        }
        
        // Load wte.weight
        let wte_expected_shape = &[self.wte_weight.shape()[0], self.wte_weight.shape()[1]];
        println!("Attempting to load 'transformer.wte.weight' with expected shape: {:?}", wte_expected_shape);
        self.wte_weight = load_specific_tensor(&tensors_map, "transformer.wte.weight", wte_expected_shape)?;
        println!("Successfully loaded 'transformer.wte.weight'.");

        // Load wpe.weight
        let wpe_expected_shape = &[self.wpe_weight.shape()[0], self.wpe_weight.shape()[1]];
        println!("Attempting to load 'transformer.wpe.weight' with expected shape: {:?}", wpe_expected_shape);
        self.wpe_weight = load_specific_tensor(&tensors_map, "transformer.wpe.weight", wpe_expected_shape)?;
        println!("Successfully loaded 'transformer.wpe.weight'.");

        let n_embd_usize = self.wte_weight.shape()[1]; // n_embd is consistent

        for i in 0..self.h.len() {
            // LayerNorm 1
            let ln1_w_name = format!("transformer.h.{}.ln_1.weight", i);
            let ln1_b_name = format!("transformer.h.{}.ln_1.bias", i);
            self.h[i].ln_1._weight = load_specific_tensor(&tensors_map, &ln1_w_name, &[n_embd_usize])?;
            self.h[i].ln_1._bias = load_specific_tensor(&tensors_map, &ln1_b_name, &[n_embd_usize])?;
            println!("Loaded layer_norm ln_1 for block {}", i);

            // MLP
            let n_inner_usize = self.h[i].mlp.c_fc_b.shape()[0]; // Get n_inner from existing initialized tensor

            let fc_w_name = format!("transformer.h.{}.mlp.c_fc.weight", i);
            let loaded_fc_w = load_specific_tensor(&tensors_map, &fc_w_name, &[n_inner_usize, n_embd_usize])?;
            self.h[i].mlp.c_fc_w = loaded_fc_w.into_dimensionality::<Ix2>()
                .map_err(|e| format!("Failed to view loaded_fc_w as Ix2 for transpose: {}", e))?
                .t().into_owned().into_dyn();
            
            let fc_b_name = format!("transformer.h.{}.mlp.c_fc.bias", i);
            self.h[i].mlp.c_fc_b = load_specific_tensor(&tensors_map, &fc_b_name, &[n_inner_usize])?;
            println!("Loaded mlp.c_fc for block {}", i);

            let proj_w_name = format!("transformer.h.{}.mlp.c_proj.weight", i);
            let loaded_proj_w = load_specific_tensor(&tensors_map, &proj_w_name, &[n_embd_usize, n_inner_usize])?;
            self.h[i].mlp.c_proj_w = loaded_proj_w.into_dimensionality::<Ix2>()
                .map_err(|e| format!("Failed to view loaded_proj_w as Ix2 for transpose: {}", e))?
                .t().into_owned().into_dyn();

            let proj_b_name = format!("transformer.h.{}.mlp.c_proj.bias", i);
            self.h[i].mlp.c_proj_b = load_specific_tensor(&tensors_map, &proj_b_name, &[n_embd_usize])?;
            println!("Loaded mlp.c_proj for block {}", i);

            // LayerNorm 2
            let ln2_w_name = format!("transformer.h.{}.ln_2.weight", i);
            let ln2_b_name = format!("transformer.h.{}.ln_2.bias", i);
            self.h[i].ln_2._weight = load_specific_tensor(&tensors_map, &ln2_w_name, &[n_embd_usize])?;
            self.h[i].ln_2._bias = load_specific_tensor(&tensors_map, &ln2_b_name, &[n_embd_usize])?;
            println!("Loaded layer_norm ln_2 for block {}", i);

            // Attention weights
            let attn_c_attn_w_name = format!("transformer.h.{}.attn.c_attn.weight", i);
            let loaded_attn_c_attn_w = load_specific_tensor(&tensors_map, &attn_c_attn_w_name, &[3 * n_embd_usize, n_embd_usize])?;
            self.h[i].attn.c_attn_w = loaded_attn_c_attn_w.into_dimensionality::<Ix2>()
                .map_err(|e| format!("Failed to view loaded_attn_c_attn_w as Ix2 for transpose: {}", e))?
                .t().into_owned().into_dyn();

            let attn_c_attn_b_name = format!("transformer.h.{}.attn.c_attn.bias", i);
            self.h[i].attn.c_attn_b = load_specific_tensor(&tensors_map, &attn_c_attn_b_name, &[3 * n_embd_usize])?;
            
            let attn_c_proj_w_name = format!("transformer.h.{}.attn.c_proj.weight", i);
            self.h[i].attn.c_proj_w = load_specific_tensor(&tensors_map, &attn_c_proj_w_name, &[n_embd_usize, n_embd_usize])?;
            
            let attn_c_proj_b_name = format!("transformer.h.{}.attn.c_proj.bias", i);
            self.h[i].attn.c_proj_b = load_specific_tensor(&tensors_map, &attn_c_proj_b_name, &[n_embd_usize])?;
            println!("Loaded attention weights for block {}", i);
        }

        // Final LayerNorm
        self.ln_f._weight = load_specific_tensor(&tensors_map, "transformer.ln_f.weight", &[n_embd_usize])?;
        self.ln_f._bias = load_specific_tensor(&tensors_map, "transformer.ln_f.bias", &[n_embd_usize])?;
        println!("Loaded final layer_norm ln_f.");

        Ok(())
    }


    pub fn generate(
        &self, 
        tokenizer: &Tokenizer, 
        prompt_ids: &[i32], 
        max_length: usize, 
        eos_token_id: i32
    ) -> Result<String, Box<dyn std::error::Error>> {
        let mut current_token_sequence: Vec<i32> = prompt_ids.to_vec();

        if prompt_ids.is_empty() {
            return Err("Prompt cannot be empty.".into());
        }
        if max_length <= prompt_ids.len() {
            // Just decode the prompt if max_length is not greater
            let token_ids_u32: Vec<u32> = current_token_sequence.iter().map(|&id| id as u32).collect();
            let decoded_text = tokenizer.decode(&token_ids_u32, true).map_err(|e| e.to_string())?;
            return Ok(decoded_text);
        }

        let max_new_tokens = max_length - prompt_ids.len();

        for _ in 0..max_new_tokens {
            let current_seq_len = current_token_sequence.len();
            let input_array = Array2::from_shape_vec((1, current_seq_len), current_token_sequence.clone())
                .map_err(|e| format!("Failed to create input array: {}", e))?;
            
            let final_hidden_states = self.forward(&input_array, None)?;
            let all_logits_dyn = self.lm_head(&final_hidden_states)?; // Shape: [1, current_seq_len, vocab_size]

            // Get logits for the very last token position
            let next_token_logits_view_dyn = all_logits_dyn.slice(s![0, current_seq_len - 1, ..]);
            let next_token_logits_view_1d = next_token_logits_view_dyn.view().into_dimensionality::<Ix1>()
                .map_err(|e| format!("Failed to convert next_token_logits to 1D: {}. Shape was {:?}", e, next_token_logits_view_dyn.shape()))?;
            
            let predicted_token_idx = argmax(next_token_logits_view_1d);
            let predicted_token_id = predicted_token_idx as i32;

            current_token_sequence.push(predicted_token_id);

            if predicted_token_id == eos_token_id {
                break;
            }
        }

        let token_ids_u32: Vec<u32> = current_token_sequence.iter().map(|&id| id as u32).collect();
        let decoded_text = tokenizer.decode(&token_ids_u32, true).map_err(|e| e.to_string())?;
        
        Ok(decoded_text)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn}; // IxDyn for ArrayD::from_shape_vec in lm_head test
    use approx::assert_abs_diff_eq;
    use tokenizers::models::bpe::BPE; // Moved BPE import here

    #[test]
    fn test_gpt2_lm_head() -> Result<(), Box<dyn std::error::Error>> {
        let config = GPT2Config {
            vocab_size: 50,
            n_embd: 10,
            n_layer: 1, // Minimal layers for model creation
            n_head: 1,  // Minimal heads
            n_positions: 100, // Max sequence length
            layer_norm_epsilon: 1e-5,
            n_inner: Some(20), // Intermediate MLP size
            // Other fields like resid_pdrop, embd_pdrop, attn_pdrop can be defaults (0.0) if not used
            // For this test, only vocab_size and n_embd are critical for lm_head
            ..Default::default() 
        };

        let mut model = GPT2Model::new(&config)?;

        // Override wte_weight with ones for predictable test results
        model.wte_weight = Array::ones((config.vocab_size as usize, config.n_embd as usize)).into_dyn();

        let batch_size = 2;
        let seq_len = 3;
        let n_embd_usize = config.n_embd as usize;

        // Create sample hidden states (e.g., all ones)
        let sample_hidden_states_vec: Vec<f32> = vec![1.0; batch_size * seq_len * n_embd_usize];
        let sample_hidden_states = ArrayD::from_shape_vec(
            vec![batch_size, seq_len, n_embd_usize].into_dyn().shape(), 
            sample_hidden_states_vec
        )?;
        
        let logits = model.lm_head(&sample_hidden_states)?;

        // Assert output shape
        assert_eq!(logits.shape(), &[batch_size, seq_len, config.vocab_size as usize]);

        // Assert logit values
        // If hidden_states are all 1.0 and wte_weight is all 1.0,
        // each logit value should be sum_k (1.0 * 1.0) for k from 0 to n_embd-1,
        // which is n_embd.
        for val in logits.iter() {
            assert_abs_diff_eq!(*val, config.n_embd as f32, epsilon = 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_gpt2_generate() -> Result<(), Box<dyn std::error::Error>> {
        // Config similar to a small GPT-2 model for testing purposes
        let config = GPT2Config {
            vocab_size: 50257, // Standard GPT-2 vocab size
            n_embd: 48,      // Small embedding size for faster test
            n_layer: 2,      // Minimal layers
            n_head: 2,       // Minimal heads
            n_positions: 100,
            layer_norm_epsilon: 1e-5,
            n_inner: Some(48 * 2), // Intermediate MLP size
            eos_token_id: 50256, // Standard EOS token for GPT-2
            bos_token_id: 50256, // Often same as EOS for GPT-2
            activation_function: "gelu".to_string(), // Ensure this matches expected
            // Other fields can be default or minimal
            ..Default::default()
        };

        let model = GPT2Model::new(&config)?;

        // Initialize tokenizer
        let vocab_path = "../../resources/tokenizer_data/gpt2/gpt2-vocab.json";
        let merges_path = "../../resources/tokenizer_data/gpt2/merges.txt";
        
        let bpe_builder = BPE::from_file(vocab_path, merges_path)
            .map_err(|e| format!("Failed to load BPE model from files: {} & {}. Error: {}", vocab_path, merges_path, e))?;
        let bpe_model = bpe_builder.build()
            .map_err(|e| format!("Failed to build BPE model. Error: {}", e))?;
        let tokenizer = Tokenizer::new(Box::new(bpe_model));


        let prompt_text = "Hello";
        let prompt_encoding = tokenizer.encode(prompt_text, false).map_err(|e| e.to_string())?;
        let prompt_ids_u32: Vec<u32> = prompt_encoding.get_ids().to_vec();
        let prompt_ids_i32: Vec<i32> = prompt_ids_u32.iter().map(|&id| id as i32).collect();


        let max_length = prompt_ids_i32.len() + 5; // Generate 5 new tokens
        
        // This test might be slow if the model is large or if file I/O for tokenizer is slow
        // For now, it's a functional test. If it fails due to missing model file, that's an env issue.
        // If the model file exists but loading fails, that's a code issue.
        // If generation is nonsensical, that's okay for now as weights are zeros or ones.
        // For a real test of generation quality, pretrained weights would be needed.
        // Here, we are primarily testing if the generate() method runs and produces some output.
        // Since weights are zeros, it will likely produce a repetitive sequence or EOS quickly if vocab_size is small
        // and EOS is one of the first few indices.
        // If `model.load_weights` was available and used, this test would be more meaningful.
        
        let generated_text = model.generate(&tokenizer, &prompt_ids_i32, max_length, config.eos_token_id)?;

        println!("Test Generate - Prompt: '{}'", prompt_text);
        println!("Test Generate - Generated: '{}'", generated_text);

        assert!(!generated_text.is_empty(), "Generated text should not be empty");
        assert!(generated_text.len() >= prompt_text.len(), "Generated text should be at least as long as the prompt");
        
        let generated_ids_u32: Vec<u32> = tokenizer.encode(generated_text.as_str(), false).map_err(|e| e.to_string())?.get_ids().to_vec();
        assert!(generated_ids_u32.len() <= max_length, "Generated sequence too long");
        assert!(generated_ids_u32.len() >= prompt_ids_i32.len(), "Generated sequence shorter than prompt");

        let empty_prompt: Vec<i32> = Vec::new();
        let empty_prompt_result = model.generate(&tokenizer, &empty_prompt, 5, config.eos_token_id);
        assert!(empty_prompt_result.is_err(), "Generate with empty prompt should error");

        let short_max_length = prompt_ids_i32.len();
        let short_gen_text = model.generate(&tokenizer, &prompt_ids_i32, short_max_length, config.eos_token_id)?;
        let decoded_prompt_text = tokenizer.decode(&prompt_ids_u32, true).map_err(|e| e.to_string())?;
        assert_eq!(short_gen_text, decoded_prompt_text, "Generated text should be same as prompt if max_length is not greater");

        Ok(())
    }

    #[test]
    fn test_load_weights() -> Result<(), Box<dyn std::error::Error>> {
        let config = GPT2Config {
            vocab_size: 50257,
            n_embd: 768,
            n_positions: 1024,
            n_layer: 12,
            n_head: 12,
            layer_norm_epsilon: 1e-5,
            // Fill other necessary fields from a typical gpt2 config or use Default if comprehensive
            ..Default::default() 
        };

        let mut model = GPT2Model::new(&config)?;
        let initial_wte_sum = model.wte_weight.sum(); // Should be 0.0 if initialized with zeros

        // The path is relative to the crate root.
        // In the test environment, this file needs to exist.
        let model_path = "../../resources/model_data/gpt2/model.safetensors";
        
        // Check if the model file exists to provide a better error message if it doesn't
        if !std::path::Path::new(model_path).exists() {
            eprintln!("Warning: Test model file not found at {}. Skipping weight loading assertions.", model_path);
            // Optionally, one might choose to panic or return Ok(()) if the file is considered essential for the test.
            // For CI/CD where the file might not be present, printing a warning and succeeding can be an option.
            // However, for a test named `test_load_weights`, it should arguably fail if it cannot load weights.
            // For now, let it proceed and fail at SafeTensors::deserialize_from_file if file not found.
        }


        match model.load_weights(model_path) {
            Ok(_) => {
                println!("Successfully called load_weights.");
                // If weights were initialized to zeros, they should not be zeros after loading (for most tensors)
                assert_ne!(model.wte_weight.sum(), initial_wte_sum, "wte_weight sum should change after loading weights.");
                assert_ne!(model.wpe_weight.sum(), 0.0, "wpe_weight sum should not be zero after loading real weights.");
                
                // Verify shapes again post-loading (should match config)
                assert_eq!(model.wte_weight.shape(), &[config.vocab_size as usize, config.n_embd as usize]);
                assert_eq!(model.wpe_weight.shape(), &[config.n_positions as usize, config.n_embd as usize]);

                // Check sums of weights to ensure they changed from their initial state.
                // Weights initialized to zeros should not be zero after loading.
                // LayerNorm weights initialized to ones should change from n_embd.
                assert_ne!(model.wte_weight.sum(), 0.0, "wte_weight sum should change from 0");
                assert_ne!(model.wpe_weight.sum(), 0.0, "wpe_weight sum should change from 0");

                let initial_ln_weight_sum = config.n_embd as f32; // LayerNorm weights are initialized to 1.0

                if !model.h.is_empty() {
                    // First Transformer Block
                    let block = &model.h[0];
                    // LayerNorm 1
                    assert_ne!(block.ln_1._weight.sum(), initial_ln_weight_sum, "h[0].ln_1.weight sum should change from initial ones-sum");
                    assert_ne!(block.ln_1._bias.sum(), 0.0, "h[0].ln_1.bias sum should generally change from 0 (actual value could be 0)");
                    
                    // Attention
                    assert_ne!(block.attn.c_attn_w.sum(), 0.0, "h[0].attn.c_attn_w sum should change from 0");
                    assert_ne!(block.attn.c_attn_b.sum(), 0.0, "h[0].attn.c_attn_b sum should generally change from 0");
                    assert_ne!(block.attn.c_proj_w.sum(), 0.0, "h[0].attn.c_proj_w sum should change from 0");
                    assert_ne!(block.attn.c_proj_b.sum(), 0.0, "h[0].attn.c_proj_b sum should generally change from 0");

                    // MLP
                    assert_ne!(block.mlp.c_fc_w.sum(), 0.0, "h[0].mlp.c_fc_w sum should change from 0");
                    assert_ne!(block.mlp.c_fc_b.sum(), 0.0, "h[0].mlp.c_fc_b sum should generally change from 0");
                    assert_ne!(block.mlp.c_proj_w.sum(), 0.0, "h[0].mlp.c_proj_w sum should change from 0");
                    assert_ne!(block.mlp.c_proj_b.sum(), 0.0, "h[0].mlp.c_proj_b sum should generally change from 0");

                    // LayerNorm 2
                    assert_ne!(block.ln_2._weight.sum(), initial_ln_weight_sum, "h[0].ln_2.weight sum should change from initial ones-sum");
                    assert_ne!(block.ln_2._bias.sum(), 0.0, "h[0].ln_2.bias sum should generally change from 0");
                }
                
                // Final LayerNorm
                assert_ne!(model.ln_f._weight.sum(), initial_ln_weight_sum, "ln_f.weight sum should change from initial ones-sum");
                assert_ne!(model.ln_f._bias.sum(), 0.0, "ln_f.bias sum should generally change from 0");
            }
            Err(e) => {
                // If the file doesn't exist, this error will be about file not found.
                // If the file exists but is malformed, or tensors are missing/mismatched, other errors will occur.
                if !std::path::Path::new(model_path).exists() {
                     // This specific error message is useful for diagnosing test environment issues.
                    return Err(format!("Test model file not found at '{}'. This test requires the GPT-2 safetensors model file. Error: {}", model_path, e).into());
                }
                // If the file exists but loading failed for other reasons (e.g. tensor mismatch), propagate the error.
                return Err(e);
            }
        }
        Ok(())
    }
}