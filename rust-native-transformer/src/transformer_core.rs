// src/transformer_core.rs

use crate::tensor_engine::{Tensor, TensorError};
use std::collections::HashMap;
use std::sync::Arc;

// 0. Basic Setup
#[derive(Debug)]
pub enum TransformerError {
    TensorError(TensorError),
    WeightNotFound(String),
    InvalidWeightShape(String),
    ConfigError(String),
    UnsupportedOperation(String), // For features like KV caching if not implemented
}

impl std::fmt::Display for TransformerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransformerError::TensorError(e) => write!(f, "Tensor error: {:?}", e),
            TransformerError::WeightNotFound(s) => write!(f, "Weight not found: {}", s),
            TransformerError::InvalidWeightShape(s) => write!(f, "Invalid weight shape: {}", s),
            TransformerError::ConfigError(s) => write!(f, "Configuration error: {}", s),
            TransformerError::UnsupportedOperation(s) => write!(f, "Unsupported operation: {}", s),
        }
    }
}

impl std::error::Error for TransformerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TransformerError::TensorError(ref e) => Some(e), 
            _ => None,
        }
    }
}

impl From<TensorError> for TransformerError {
    fn from(err: TensorError) -> TransformerError {
        TransformerError::TensorError(err)
    }
}

// 1. Config Struct
#[derive(Debug, Clone)]
pub struct Config {
    pub n_layer: usize,    
    pub n_head: usize,     
    pub n_embd: usize,     
    pub vocab_size: usize, 
    pub block_size: usize, 
    pub bias: bool, 
}

impl Config {
    pub fn head_dim(&self) -> usize {
        if self.n_embd == 0 || self.n_head == 0 { 
            return 0;
        }
        if self.n_embd % self.n_head != 0 {
            eprintln!("Warning: n_embd {} is not divisible by n_head {}. Head dimension may be incorrect.", self.n_embd, self.n_head);
        }
        self.n_embd / self.n_head
    }
}

#[allow(dead_code)]
pub(crate) mod tensor_ops {
    use super::*;

    // matmul is now primarily called as a method: a.matmul_simd(b)
    // This static version can be kept for compatibility or removed if not used.
    pub fn matmul(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        a.matmul_simd(b) // Defaulting to SIMD version
    }

    pub fn softmax(a: &Tensor<f32>, axis: usize) -> Result<Tensor<f32>, TensorError> {
        a.softmax(axis)
    }

    pub fn layernorm(a: &Tensor<f32>, gamma: &Tensor<f32>, beta: &Tensor<f32>, epsilon: f32) -> Result<Tensor<f32>, TensorError> {
        a.layernorm(gamma, beta, epsilon)
    }

    // gelu is now primarily called as a method: a.gelu_simd()
    // This static version can be kept for compatibility or removed if not used.
    pub fn gelu(a: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        a.gelu_simd() // Defaulting to SIMD version
    }
    
    pub fn add(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        if a.shape != b.shape {
            if b.rank() == 1 && a.shape.last() == b.shape.last() && a.rank() > 1 {
                let mut out_data = a.data.clone();
                let last_dim_size = b.shape[0];
                let num_vectors = a.data.len() / last_dim_size;
                for i in 0..num_vectors {
                    for j in 0..last_dim_size {
                        out_data[i * last_dim_size + j] += b.data[j];
                    }
                }
                return Tensor::new(out_data, a.shape.clone());
            }
            return Err(TensorError::ShapeMismatch(format!(
                "Element-wise add requires identical shapes or broadcastable bias. Got {:?} and {:?}",
                a.shape, b.shape
            )));
        }
        let data = a.data.iter().zip(b.data.iter()).map(|(av, bv)| av + bv).collect();
        Tensor::new(data, a.shape.clone())
    }
    
    pub fn split_last_dim(tensor: &Tensor<f32>, num_chunks: usize) -> Result<Vec<Tensor<f32>>, TensorError> {
        if tensor.rank() == 0 {
            return Err(TensorError::InvalidDimension("Cannot split scalar tensor".to_string()));
        }
        let last_dim_idx = tensor.rank() - 1;
        let last_dim_size = tensor.shape[last_dim_idx];

        if last_dim_size % num_chunks != 0 {
            return Err(TensorError::InvalidDimension(format!(
                "Last dimension size {} cannot be evenly split into {} chunks",
                last_dim_size, num_chunks
            )));
        }
        let chunk_size_last_dim = last_dim_size / num_chunks;
        
        let mut result_tensors = Vec::with_capacity(num_chunks);
        let mut new_shape_template = tensor.shape.clone();
        new_shape_template[last_dim_idx] = chunk_size_last_dim;
        
        let num_elements_per_chunk: usize = new_shape_template.iter().product();
        let num_outer_elements: usize = tensor.shape[..last_dim_idx].iter().product();

        for chunk_idx in 0..num_chunks {
            let mut chunk_data = Vec::with_capacity(num_elements_per_chunk);
            for outer_idx in 0..num_outer_elements {
                let original_data_start_offset = outer_idx * last_dim_size + chunk_idx * chunk_size_last_dim;
                chunk_data.extend_from_slice(&tensor.data[original_data_start_offset .. original_data_start_offset + chunk_size_last_dim]);
            }
            result_tensors.push(Tensor::new(chunk_data, new_shape_template.clone())?);
        }
        Ok(result_tensors)
    }
    
    pub fn linear(input: &Tensor<f32>, weight: &Tensor<f32>, bias: Option<&Tensor<f32>>) -> Result<Tensor<f32>, TransformerError> {
        let din = weight.shape[0];
        let dout = weight.shape[1];
        if input.shape.last().unwrap_or(&0) != &din {
            return Err(TransformerError::TensorError(TensorError::IncompatibleShapes(format!(
                "Linear input last dim {} != weight first dim {}",
                input.shape.last().unwrap_or(&0), din
            ))));
        }

        let mut out_shape = input.shape.clone();
        if out_shape.is_empty() && input.num_elements() == din { 
             out_shape = vec![1, din]; 
        } else if out_shape.is_empty() { 
            return Err(TransformerError::TensorError(TensorError::InvalidDimension("Scalar input not directly usable in linear layer without proper shape".to_string())));
        }
        let last_dim_idx = out_shape.len() - 1;
        out_shape[last_dim_idx] = dout;
        
        let original_rank = input.rank();
        let input_reshaped = if original_rank > 2 {
            let new_rows: usize = input.shape[..original_rank-1].iter().product();
            input.reshape(vec![new_rows, din])?
        } else if original_rank == 1 && input.shape[0] == din { 
            input.reshape(vec![1, din])? 
        }
         else {
            input.clone() 
        };

        let mut output = input_reshaped.matmul_simd(weight)?; // Using matmul_simd

        if let Some(b) = bias {
            if b.rank() != 1 || b.shape[0] != dout {
                 return Err(TransformerError::TensorError(TensorError::IncompatibleShapes(format!("Bias shape {:?} incompatible with output dim {}", b.shape, dout))));
            }
            for r in 0..output.shape[0] { 
                for c in 0..output.shape[1] { 
                    let flat_idx = r * output.shape[1] + c;
                    output.data[flat_idx] += b.data[c];
                }
            }
        }
        
        if original_rank > 2 || (original_rank == 1 && input.shape[0] == din) { 
            output = output.reshape(out_shape)?;
        }
        Ok(output)
    }

    pub fn embedding(ids: &Tensor<u32>, weight: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        if weight.rank() != 2 {
            return Err(TensorError::InvalidDimension("Embedding weight matrix must be 2D".to_string()));
        }
        let vocab_size = weight.shape[0];
        let emb_dim = weight.shape[1];

        let mut out_shape = ids.shape.clone();
        out_shape.push(emb_dim);

        let mut out_data = Vec::with_capacity(ids.num_elements() * emb_dim);

        for id_val in &ids.data {
            let id_idx = *id_val as usize;
            if id_idx >= vocab_size {
                return Err(TensorError::OutOfBounds(format!("Token ID {} out of vocab size {}", id_idx, vocab_size)));
            }
            let embedding_vector_slice_start = id_idx * emb_dim;
            let embedding_vector_slice_end = embedding_vector_slice_start + emb_dim;
            out_data.extend_from_slice(&weight.data[embedding_vector_slice_start..embedding_vector_slice_end]);
        }
        Tensor::new(out_data, out_shape)
    }
}

pub struct MultiHeadAttention {
    c_attn_w: Tensor<f32>, 
    c_attn_b: Tensor<f32>, 
    c_proj_w: Tensor<f32>, 
    c_proj_b: Tensor<f32>, 
    config: Arc<Config>,
}

impl MultiHeadAttention {
    pub fn new(
        c_attn_w: Tensor<f32>, 
        c_attn_b: Tensor<f32>, 
        c_proj_w: Tensor<f32>, 
        c_proj_b: Tensor<f32>, 
        config: Arc<Config>
    ) -> Result<Self, TransformerError> {
        let n_embd = config.n_embd;
        if c_attn_w.shape != vec![n_embd, 3 * n_embd] {
            return Err(TransformerError::InvalidWeightShape(format!("c_attn_w shape mismatch: expected [{}, {}], got {:?}", n_embd, 3 * n_embd, c_attn_w.shape)));
        }
        if c_attn_b.shape != vec![3 * n_embd] {
            return Err(TransformerError::InvalidWeightShape(format!("c_attn_b shape mismatch: expected [{}], got {:?}", 3 * n_embd, c_attn_b.shape)));
        }
        if c_proj_w.shape != vec![n_embd, n_embd] {
            return Err(TransformerError::InvalidWeightShape(format!("c_proj_w shape mismatch: expected [{}, {}], got {:?}", n_embd, n_embd, c_proj_w.shape)));
        }
        if c_proj_b.shape != vec![n_embd] {
            return Err(TransformerError::InvalidWeightShape(format!("c_proj_b shape mismatch: expected [{}], got {:?}", n_embd, c_proj_b.shape)));
        }

        Ok(MultiHeadAttention {
            c_attn_w,
            c_attn_b,
            c_proj_w,
            c_proj_b,
            config,
        })
    }

    pub fn forward(&self, x: &Tensor<f32>, mask: Option<&Tensor<f32>>) -> Result<Tensor<f32>, TransformerError> {
        let batch_size = x.shape[0];
        let seq_len = x.shape[1];
        let n_embd = self.config.n_embd;
        let n_head = self.config.n_head;
        let head_dim = self.config.head_dim();

        let qkv_combined = tensor_ops::linear(x, &self.c_attn_w, Some(&self.c_attn_b))?;
        
        let mut qkv_parts = tensor_ops::split_last_dim(&qkv_combined, 3)?;
        let v_all = qkv_parts.pop().unwrap();
        let k_all = qkv_parts.pop().unwrap();
        let q_all = qkv_parts.pop().unwrap();

        let q = q_all.reshape(vec![batch_size, seq_len, n_head, head_dim])?
                     .permute_mha_qkv()?; 
        let k = k_all.reshape(vec![batch_size, seq_len, n_head, head_dim])?
                     .permute_mha_qkv()?;
        let v = v_all.reshape(vec![batch_size, seq_len, n_head, head_dim])?
                     .permute_mha_qkv()?;
        
        let k_t = k.permute_mha_kt()?;
        
        let mut att_scores_parts = Vec::with_capacity(batch_size * n_head);
        let scale = (head_dim as f32).sqrt();

        for b_idx in 0..batch_size {
            for h_idx in 0..n_head {
                let q_slice = q.slice_mha(b_idx, h_idx)?; 
                let k_t_slice = k_t.slice_mha(b_idx, h_idx)?; 
                
                let mut scores_s = q_slice.matmul_simd(&k_t_slice)?; // Using matmul_simd
                for val in scores_s.data.iter_mut() {
                    *val /= scale;
                }
                att_scores_parts.push(scores_s);
            }
        }
        let mut att_scores_data_flat = Vec::new();
        for t in att_scores_parts { att_scores_data_flat.extend(t.data); }
        let mut att_scores = Tensor::new(att_scores_data_flat, vec![batch_size, n_head, seq_len, seq_len])?;

        if let Some(m) = mask { 
            for b in 0..batch_size {
                for h in 0..n_head {
                    for s1 in 0..seq_len {
                        for s2 in 0..seq_len {
                            if *m.get(&[s1, s2])? == 0.0 { 
                                let val_ref = att_scores.get_mut(&[b,h,s1,s2])?;
                                *val_ref = f32::NEG_INFINITY;
                            }
                        }
                    }
                }
            }
        }

        let att_probs = tensor_ops::softmax(&att_scores, 3)?;

        let mut out_att_parts = Vec::with_capacity(batch_size * n_head);
        for b_idx in 0..batch_size {
            for h_idx in 0..n_head {
                let probs_slice = att_probs.slice_mha(b_idx, h_idx)?; 
                let v_slice = v.slice_mha(b_idx, h_idx)?; 
                out_att_parts.push(probs_slice.matmul_simd(&v_slice)?); // Using matmul_simd
            }
        }
        let mut out_att_data_flat = Vec::new();
        for t in out_att_parts { out_att_data_flat.extend(t.data); }
        let out_att = Tensor::new(out_att_data_flat, vec![batch_size, n_head, seq_len, head_dim])?;
        
        let out_reshaped = out_att.permute_mha_output()?
                                  .reshape(vec![batch_size, seq_len, n_embd])?;
        
        let final_output = tensor_ops::linear(&out_reshaped, &self.c_proj_w, Some(&self.c_proj_b))?;
        
        Ok(final_output)
    }
}

pub struct FeedForward {
    c_fc_w: Tensor<f32>,   
    c_fc_b: Tensor<f32>,   
    c_proj_w: Tensor<f32>, 
    c_proj_b: Tensor<f32>, 
}

impl FeedForward {
    pub fn new(
        c_fc_w: Tensor<f32>, 
        c_fc_b: Tensor<f32>, 
        c_proj_w: Tensor<f32>, 
        c_proj_b: Tensor<f32>,
        config: &Config 
    ) -> Result<Self, TransformerError> {
        let n_embd = config.n_embd;
        let n_hidden = 4 * n_embd; 

        if c_fc_w.shape != vec![n_embd, n_hidden] {
            return Err(TransformerError::InvalidWeightShape(format!("c_fc_w shape mismatch: expected [{}, {}], got {:?}", n_embd, n_hidden, c_fc_w.shape)));
        }
        if c_fc_b.shape != vec![n_hidden] {
            return Err(TransformerError::InvalidWeightShape(format!("c_fc_b shape mismatch: expected [{}], got {:?}", n_hidden, c_fc_b.shape)));
        }
        if c_proj_w.shape != vec![n_hidden, n_embd] {
            return Err(TransformerError::InvalidWeightShape(format!("c_proj_w shape mismatch: expected [{}, {}], got {:?}", n_hidden, n_embd, c_proj_w.shape)));
        }
        if c_proj_b.shape != vec![n_embd] {
            return Err(TransformerError::InvalidWeightShape(format!("c_proj_b shape mismatch: expected [{}], got {:?}", n_embd, c_proj_b.shape)));
        }
        
        Ok(FeedForward { c_fc_w, c_fc_b, c_proj_w, c_proj_b })
    }

    pub fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>, TransformerError> {
        let mut h = tensor_ops::linear(x, &self.c_fc_w, Some(&self.c_fc_b))?;
        h = h.gelu_simd()?; // Using gelu_simd
        let output = tensor_ops::linear(&h, &self.c_proj_w, Some(&self.c_proj_b))?;
        Ok(output)
    }
}

pub struct Block {
    attn: MultiHeadAttention,
    mlp: FeedForward,
    ln_1_g: Tensor<f32>, 
    ln_1_b: Tensor<f32>, 
    ln_2_g: Tensor<f32>, 
    ln_2_b: Tensor<f32>, 
}

impl Block {
    pub fn new(
        attn: MultiHeadAttention, 
        mlp: FeedForward, 
        ln_1_g: Tensor<f32>, 
        ln_1_b: Tensor<f32>, 
        ln_2_g: Tensor<f32>, 
        ln_2_b: Tensor<f32>,
        config: &Config 
    ) -> Result<Self, TransformerError> {
        let n_embd = config.n_embd;
        if ln_1_g.shape != vec![n_embd] || ln_1_b.shape != vec![n_embd] ||
           ln_2_g.shape != vec![n_embd] || ln_2_b.shape != vec![n_embd] {
            return Err(TransformerError::InvalidWeightShape("LayerNorm weight shape mismatch".to_string()));
        }

        Ok(Block { attn, mlp, ln_1_g, ln_1_b, ln_2_g, ln_2_b })
    }

    pub fn forward(&self, x: &Tensor<f32>, mask: Option<&Tensor<f32>>) -> Result<Tensor<f32>, TransformerError> {
        let x_norm1 = tensor_ops::layernorm(x, &self.ln_1_g, &self.ln_1_b, 1e-5)?;
        let attn_output = self.attn.forward(&x_norm1, mask)?;
        let x_plus_attn = tensor_ops::add(x, &attn_output)?; 
        
        let x_norm2 = tensor_ops::layernorm(&x_plus_attn, &self.ln_2_g, &self.ln_2_b, 1e-5)?;
        let mlp_output = self.mlp.forward(&x_norm2)?;
        let final_output = tensor_ops::add(&x_plus_attn, &mlp_output)?;
        
        Ok(final_output)
    }
}

pub struct GPT2Model {
    pub config: Arc<Config>, 
    wte: Tensor<f32>,    
    wpe: Tensor<f32>,    
    blocks: Vec<Block>,
    ln_f_g: Tensor<f32>, 
    ln_f_b: Tensor<f32>, 
}

impl GPT2Model {
    fn get_weight(weights: &mut HashMap<String, Tensor<f32>>, name: &str, expected_shape: Option<&[usize]>) -> Result<Tensor<f32>, TransformerError> {
        let weight_name = name.to_string(); 
        let weight = weights.remove(&weight_name).ok_or_else(|| TransformerError::WeightNotFound(name.to_string()))?;
        if let Some(shape) = expected_shape {
            if weight.shape != shape {
                return Err(TransformerError::InvalidWeightShape(format!(
                    "Weight {} shape mismatch: expected {:?}, got {:?}",
                    name, shape, weight.shape
                )));
            }
        }
        Ok(weight)
    }
    
    pub fn new(config: Config, mut weights: HashMap<String, Tensor<f32>>) -> Result<Self, TransformerError> {
        let conf = Arc::new(config);
        let n_layer = conf.n_layer;
        let n_embd = conf.n_embd;
        let vocab_size = conf.vocab_size;
        let block_size = conf.block_size;

        let wte = Self::get_weight(&mut weights, "wte.weight", Some(&[vocab_size, n_embd]))?;
        let wpe = Self::get_weight(&mut weights, "wpe.weight", Some(&[block_size, n_embd]))?;
        
        let mut blocks_vec = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let c_attn_w = Self::get_weight(&mut weights, &format!("h.{}.attn.c_attn.weight", i), Some(&[n_embd, 3 * n_embd]))?;
            let c_attn_b = Self::get_weight(&mut weights, &format!("h.{}.attn.c_attn.bias", i), Some(&[3 * n_embd]))?;
            let c_proj_w = Self::get_weight(&mut weights, &format!("h.{}.attn.c_proj.weight", i), Some(&[n_embd, n_embd]))?;
            let c_proj_b = Self::get_weight(&mut weights, &format!("h.{}.attn.c_proj.bias", i), Some(&[n_embd]))?;
            let attn = MultiHeadAttention::new(c_attn_w, c_attn_b, c_proj_w, c_proj_b, Arc::clone(&conf))?;

            let mlp_c_fc_w = Self::get_weight(&mut weights, &format!("h.{}.mlp.c_fc.weight", i), Some(&[n_embd, 4 * n_embd]))?;
            let mlp_c_fc_b = Self::get_weight(&mut weights, &format!("h.{}.mlp.c_fc.bias", i), Some(&[4 * n_embd]))?;
            let mlp_c_proj_w = Self::get_weight(&mut weights, &format!("h.{}.mlp.c_proj.weight", i), Some(&[4 * n_embd, n_embd]))?;
            let mlp_c_proj_b = Self::get_weight(&mut weights, &format!("h.{}.mlp.c_proj.bias", i), Some(&[n_embd]))?;
            let mlp = FeedForward::new(mlp_c_fc_w, mlp_c_fc_b, mlp_c_proj_w, mlp_c_proj_b, &conf)?;

            let ln_1_g = Self::get_weight(&mut weights, &format!("h.{}.ln_1.weight", i), Some(&[n_embd]))?;
            let ln_1_b = Self::get_weight(&mut weights, &format!("h.{}.ln_1.bias", i), Some(&[n_embd]))?;
            let ln_2_g = Self::get_weight(&mut weights, &format!("h.{}.ln_2.weight", i), Some(&[n_embd]))?;
            let ln_2_b = Self::get_weight(&mut weights, &format!("h.{}.ln_2.bias", i), Some(&[n_embd]))?;
            
            blocks_vec.push(Block::new(attn, mlp, ln_1_g, ln_1_b, ln_2_g, ln_2_b, &conf)?);
        }

        let ln_f_g = Self::get_weight(&mut weights, "ln_f.weight", Some(&[n_embd]))?;
        let ln_f_b = Self::get_weight(&mut weights, "ln_f.bias", Some(&[n_embd]))?;

        if !weights.is_empty() {
            eprintln!("Warning: Unused weights: {:?}", weights.keys().collect::<Vec<_>>());
        }

        Ok(GPT2Model {
            config: conf,
            wte,
            wpe,
            blocks: blocks_vec,
            ln_f_g,
            ln_f_b,
        })
    }

    pub fn forward(&self, token_ids: &Tensor<u32>, _past_kv_caches: Option<()>) -> Result<Tensor<f32>, TransformerError> {
        let _batch_size = token_ids.shape[0];
        let seq_len = token_ids.shape[1];

        if seq_len > self.config.block_size {
            return Err(TransformerError::ConfigError(format!(
                "Input sequence length {} exceeds model block size {}",
                seq_len, self.config.block_size
            )));
        }

        let token_embed = tensor_ops::embedding(token_ids, &self.wte)?; 
        
        let pos_ids_data: Vec<u32> = (0..seq_len as u32).collect();
        let pos_ids_tensor = Tensor::new(pos_ids_data, vec![seq_len])?;
        
        let pos_embed_full = tensor_ops::embedding(&pos_ids_tensor, &self.wpe)?; 

        let pos_embed_batched_data = pos_embed_full.data.repeat(token_ids.shape[0]);
        let pos_embed_batched = Tensor::new(pos_embed_batched_data, token_embed.shape.clone())?;

        let mut x = tensor_ops::add(&token_embed, &pos_embed_batched)?;

        let mut mask_data = vec![1.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j > i { 
                    mask_data[i * seq_len + j] = 0.0;
                }
            }
        }
        let causal_mask = Tensor::new(mask_data, vec![seq_len, seq_len])?;

        for block in &self.blocks {
            x = block.forward(&x, Some(&causal_mask))?;
        }

        x = tensor_ops::layernorm(&x, &self.ln_f_g, &self.ln_f_b, 1e-5)?;
        
        let wte_t_data = Tensor::<f32>::transpose_data_generic(&self.wte.data, self.wte.shape[0], self.wte.shape[1]);
        let wte_t = Tensor::new(wte_t_data, vec![self.config.n_embd, self.config.vocab_size])?;
        
        let logits = tensor_ops::linear(&x, &wte_t, None)?;

        Ok(logits)
    }
}

trait TensorExtMHA {
    fn permute_mha_qkv(&self) -> Result<Tensor<f32>, TransformerError>;      
    fn permute_mha_kt(&self) -> Result<Tensor<f32>, TransformerError>;       
    fn permute_mha_output(&self) -> Result<Tensor<f32>, TransformerError>;  
    fn slice_mha(&self, batch_idx: usize, head_idx: usize) -> Result<Tensor<f32>, TransformerError>; 
}

impl TensorExtMHA for Tensor<f32> {
    fn permute_mha_qkv(&self) -> Result<Tensor<f32>, TransformerError> {
        if self.rank() != 4 { return Err(TransformerError::TensorError(TensorError::InvalidDimension("permute_mha_qkv expects 4D tensor".into()))); }
        let b = self.shape[0]; let s = self.shape[1]; let h = self.shape[2]; let d = self.shape[3];
        let mut new_data = vec![0.0; self.data.len()];
        let new_shape = vec![b, h, s, d];
        for b_i in 0..b {
            for s_i in 0..s {
                for h_i in 0..h {
                    for d_i in 0..d {
                        let old_idx = b_i*s*h*d + s_i*h*d + h_i*d + d_i;
                        let new_idx = b_i*h*s*d + h_i*s*d + s_i*d + d_i;
                        new_data[new_idx] = self.data[old_idx];
                    }
                }
            }
        }
        Tensor::new(new_data, new_shape).map_err(TransformerError::from)
    }

    fn permute_mha_kt(&self) -> Result<Tensor<f32>, TransformerError> {
        if self.rank() != 4 { return Err(TransformerError::TensorError(TensorError::InvalidDimension("permute_mha_kt expects 4D tensor".into()))); }
        let b = self.shape[0]; let h = self.shape[1]; let s = self.shape[2]; let d = self.shape[3];
        let mut new_data = vec![0.0; self.data.len()];
        let new_shape = vec![b, h, d, s];
        for b_i in 0..b {
            for h_i in 0..h {
                for s_i in 0..s {
                    for d_i in 0..d {
                        let old_idx = b_i*h*s*d + h_i*s*d + s_i*d + d_i;
                        let new_idx = b_i*h*d*s + h_i*d*s + d_i*s + s_i;
                        new_data[new_idx] = self.data[old_idx];
                    }
                }
            }
        }
        Tensor::new(new_data, new_shape).map_err(TransformerError::from)
    }
    
    fn permute_mha_output(&self) -> Result<Tensor<f32>, TransformerError> {
        if self.rank() != 4 { return Err(TransformerError::TensorError(TensorError::InvalidDimension("permute_mha_output expects 4D tensor".into()))); }
        let b = self.shape[0]; let h = self.shape[1]; let s = self.shape[2]; let d = self.shape[3];
        let mut new_data = vec![0.0; self.data.len()];
        let new_shape = vec![b, s, h, d];
         for b_i in 0..b {
            for h_i in 0..h {
                for s_i in 0..s {
                    for d_i in 0..d {
                        let old_idx = b_i*h*s*d + h_i*s*d + s_i*d + d_i;
                        let new_idx = b_i*s*h*d + s_i*h*d + h_i*d + d_i;
                        new_data[new_idx] = self.data[old_idx];
                    }
                }
            }
        }
        Tensor::new(new_data, new_shape).map_err(TransformerError::from)
    }

    fn slice_mha(&self, batch_idx: usize, head_idx: usize) -> Result<Tensor<f32>, TransformerError> {
        if self.rank() != 4 { return Err(TransformerError::TensorError(TensorError::InvalidDimension("slice_mha expects 4D tensor".into()))); }
        let s_dim = self.shape[2]; 
        let d_dim = self.shape[3]; 
        
        let mut slice_data = Vec::with_capacity(s_dim * d_dim);
        let h_total = self.shape[1]; // Total number of heads in this tensor

        let offset = batch_idx * h_total * s_dim * d_dim + head_idx * s_dim * d_dim;

        for i in 0..(s_dim * d_dim) {
            slice_data.push(self.data[offset + i]);
        }
        Tensor::new(slice_data, vec![s_dim, d_dim]).map_err(TransformerError::from)
    }
}

impl Tensor<f32> {
    pub fn transpose_data_generic(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut new_data = vec![0.0; data.len()];
        for r in 0..rows {
            for c in 0..cols {
                new_data[c * rows + r] = data[r * cols + c];
            }
        }
        new_data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_dummy_config() -> Config {
        Config {
            n_layer: 1, 
            n_head: 2,
            n_embd: 4, 
            vocab_size: 10,
            block_size: 8,
            bias: true,
        }
    }
    
    fn create_dummy_weights_for_model(config: &Config) -> HashMap<String, Tensor<f32>> {
        let mut weights = HashMap::new();
        let n_embd = config.n_embd;
        let vocab_size = config.vocab_size;
        let block_size = config.block_size;

        weights.insert("wte.weight".to_string(), Tensor::zeros(vec![vocab_size, n_embd]));
        weights.insert("wpe.weight".to_string(), Tensor::zeros(vec![block_size, n_embd]));

        for i in 0..config.n_layer {
            weights.insert(format!("h.{}.attn.c_attn.weight", i), Tensor::zeros(vec![n_embd, 3 * n_embd]));
            weights.insert(format!("h.{}.attn.c_attn.bias", i), Tensor::zeros(vec![3 * n_embd]));
            weights.insert(format!("h.{}.attn.c_proj.weight", i), Tensor::zeros(vec![n_embd, n_embd]));
            weights.insert(format!("h.{}.attn.c_proj.bias", i), Tensor::zeros(vec![n_embd]));
            
            weights.insert(format!("h.{}.mlp.c_fc.weight", i), Tensor::zeros(vec![n_embd, 4 * n_embd]));
            weights.insert(format!("h.{}.mlp.c_fc.bias", i), Tensor::zeros(vec![4 * n_embd]));
            weights.insert(format!("h.{}.mlp.c_proj.weight", i), Tensor::zeros(vec![4 * n_embd, n_embd]));
            weights.insert(format!("h.{}.mlp.c_proj.bias", i), Tensor::zeros(vec![n_embd]));

            weights.insert(format!("h.{}.ln_1.weight", i), Tensor::zeros(vec![n_embd])); 
            weights.insert(format!("h.{}.ln_1.bias", i), Tensor::zeros(vec![n_embd]));   
            weights.insert(format!("h.{}.ln_2.weight", i), Tensor::zeros(vec![n_embd])); 
            weights.insert(format!("h.{}.ln_2.bias", i), Tensor::zeros(vec![n_embd]));   
        }
        weights.insert("ln_f.weight".to_string(), Tensor::zeros(vec![n_embd])); 
        weights.insert("ln_f.bias".to_string(), Tensor::zeros(vec![n_embd]));   
        weights
    }

    #[test]
    fn test_config_creation() {
        let config = create_dummy_config();
        assert_eq!(config.n_layer, 1);
        assert_eq!(config.head_dim(), 2); 
    }

    #[test]
    fn test_mha_creation_valid() {
        let config = Arc::new(create_dummy_config());
        let n_embd = config.n_embd;
        let c_attn_w = Tensor::zeros(vec![n_embd, 3 * n_embd]);
        let c_attn_b = Tensor::zeros(vec![3 * n_embd]);
        let c_proj_w = Tensor::zeros(vec![n_embd, n_embd]);
        let c_proj_b = Tensor::zeros(vec![n_embd]);
        let mha = MultiHeadAttention::new(c_attn_w, c_attn_b, c_proj_w, c_proj_b, config);
        assert!(mha.is_ok());
    }
    
    #[test]
    fn test_mha_creation_invalid_shape() {
        let config = Arc::new(create_dummy_config());
        let n_embd = config.n_embd;
        let c_attn_w_bad = Tensor::zeros(vec![n_embd, 1 * n_embd]); 
        let c_attn_b = Tensor::zeros(vec![3 * n_embd]);
        let c_proj_w = Tensor::zeros(vec![n_embd, n_embd]);
        let c_proj_b = Tensor::zeros(vec![n_embd]);
        let mha = MultiHeadAttention::new(c_attn_w_bad, c_attn_b, c_proj_w, c_proj_b, config);
        assert!(mha.is_err());
        match mha.err().unwrap() {
            TransformerError::InvalidWeightShape(s) => assert!(s.contains("c_attn_w shape mismatch")),
            e => panic!("Unexpected error type: {:?}", e),
        }
    }

    #[test]
    fn test_ff_creation_valid() {
        let config = create_dummy_config(); 
        let n_embd = config.n_embd;
        let c_fc_w = Tensor::zeros(vec![n_embd, 4*n_embd]);
        let c_fc_b = Tensor::zeros(vec![4*n_embd]);
        let c_proj_w = Tensor::zeros(vec![4*n_embd, n_embd]);
        let c_proj_b = Tensor::zeros(vec![n_embd]);
        let ff = FeedForward::new(c_fc_w, c_fc_b, c_proj_w, c_proj_b, &config);
        assert!(ff.is_ok());
    }

    #[test]
    fn test_block_creation_valid() {
        let config = Arc::new(create_dummy_config()); 
        let n_embd = config.n_embd;

        let c_attn_w = Tensor::zeros(vec![n_embd, 3 * n_embd]);
        let c_attn_b = Tensor::zeros(vec![3 * n_embd]);
        let c_proj_w_attn = Tensor::zeros(vec![n_embd, n_embd]);
        let c_proj_b_attn = Tensor::zeros(vec![n_embd]);
        let attn = MultiHeadAttention::new(c_attn_w, c_attn_b, c_proj_w_attn, c_proj_b_attn, Arc::clone(&config)).unwrap();
        
        let mlp_c_fc_w = Tensor::zeros(vec![n_embd, 4*n_embd]);
        let mlp_c_fc_b = Tensor::zeros(vec![4*n_embd]);
        let mlp_c_proj_w = Tensor::zeros(vec![4*n_embd, n_embd]);
        let mlp_c_proj_b = Tensor::zeros(vec![n_embd]);
        let mlp = FeedForward::new(mlp_c_fc_w, mlp_c_fc_b, mlp_c_proj_w, mlp_c_proj_b, &config).unwrap(); 

        let ln_1_g = Tensor::zeros(vec![n_embd]);
        let ln_1_b = Tensor::zeros(vec![n_embd]);
        let ln_2_g = Tensor::zeros(vec![n_embd]);
        let ln_2_b = Tensor::zeros(vec![n_embd]);

        let block = Block::new(attn, mlp, ln_1_g, ln_1_b, ln_2_g, ln_2_b, &config); 
        assert!(block.is_ok());
    }

    #[test]
    fn test_gpt2model_creation_valid() {
        let config = create_dummy_config();
        let weights = create_dummy_weights_for_model(&config);
        let model = GPT2Model::new(config, weights);
        assert!(model.is_ok(), "Model creation failed: {:?}", model.err());
    }

    #[test]
    fn test_gpt2model_creation_missing_weight_error() {
        let config = create_dummy_config();
        let mut weights = create_dummy_weights_for_model(&config);
        weights.remove("wte.weight"); 
        let model = GPT2Model::new(config, weights);
        assert!(model.is_err());
        match model.err().unwrap() {
            TransformerError::WeightNotFound(s) => assert_eq!(s, "wte.weight"),
            e => panic!("Unexpected error type for missing weight: {:?}", e),
        }
    }
    
    #[test]
    fn test_gpt2model_creation_wrong_weight_shape_error() {
        let config = create_dummy_config();
        let mut weights = create_dummy_weights_for_model(&config);
        weights.insert("ln_f.bias".to_string(), Tensor::zeros(vec![config.n_embd + 1])); 
        let model = GPT2Model::new(config, weights);
        assert!(model.is_err());
        match model.err().unwrap() {
            TransformerError::InvalidWeightShape(s) => assert!(s.contains("ln_f.bias shape mismatch")),
            e => panic!("Unexpected error type for wrong weight shape: {:?}", e),
        }
    }

    #[test]
    fn test_gpt2model_forward_pass_mocked() {
        let config_obj = create_dummy_config();
        let weights = create_dummy_weights_for_model(&config_obj);
        let model = GPT2Model::new(config_obj.clone(), weights).expect("Model creation should succeed");

        let batch_size = 1;
        let seq_len = config_obj.block_size / 2; 
        
        let token_ids_data: Vec<u32> = (0..(batch_size * seq_len) as u32)
                                        .map(|i| i % (config_obj.vocab_size as u32))
                                        .collect();
        let token_ids = Tensor::new(token_ids_data, vec![batch_size, seq_len]).unwrap();

        let result = model.forward(&token_ids, None);

        if let Ok(logits) = result {
            assert_eq!(logits.shape, vec![batch_size, seq_len, config_obj.vocab_size]);
        } else {
            panic!("Forward pass failed: {:?}", result.err());
        }
    }
}
