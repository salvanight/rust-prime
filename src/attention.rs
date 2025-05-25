use ndarray::{Array, ArrayD, Axis, Ix1, Ix2, Ix4, Slice, ShapeError, s}; // Removed Zip, Div
use std::error::Error;
// Removed: use std::ops::Div; 

// Helper function for softmax
fn softmax(input: &ArrayD<f32>, axis_index: usize) -> Result<ArrayD<f32>, Box<dyn Error>> {
    let axis = Axis(axis_index);
    // fold_axis reduces dimensionality, so insert_axis is needed for broadcasting
    let max_val = input.fold_axis(axis, f32::NEG_INFINITY, |&a, &b| a.max(b));
    let max_val_broadcastable = max_val.insert_axis(axis);
    
    let exp_values = (input - &max_val_broadcastable).mapv(f32::exp);
    
    // sum_axis reduces dimensionality
    let sum_exp_values = exp_values.sum_axis(axis);
    let sum_exp_values_broadcastable = sum_exp_values.insert_axis(axis);

    Ok(&exp_values / &sum_exp_values_broadcastable)
}


#[derive(Debug)]
pub struct MultiHeadAttention {
    pub(crate) n_head: i32,
    pub(crate) n_embd: i32,
    pub(crate) head_dim: i32,
    pub(crate) c_attn_w: ArrayD<f32>, // Shape: [n_embd, 3 * n_embd]
    pub(crate) c_attn_b: ArrayD<f32>, // Shape: [3 * n_embd]
    pub(crate) c_proj_w: ArrayD<f32>, // Shape: [n_embd, n_embd]
    pub(crate) c_proj_b: ArrayD<f32>, // Shape: [n_embd]
}

impl MultiHeadAttention {
    pub fn new(n_head: i32, n_embd: i32) -> Result<Self, Box<dyn Error>> {
        if n_embd % n_head != 0 {
            return Err(format!(
                "n_embd ({}) must be divisible by n_head ({})",
                n_embd, n_head
            )
            .into());
        }
        let head_dim = n_embd / n_head;

        let c_attn_w = Array::zeros((n_embd as usize, 3 * n_embd as usize)).into_dyn();
        let c_attn_b = Array::zeros((3 * n_embd as usize,)).into_dyn();
        let c_proj_w = Array::zeros((n_embd as usize, n_embd as usize)).into_dyn();
        let c_proj_b = Array::zeros((n_embd as usize,)).into_dyn();

        Ok(Self {
            n_head,
            n_embd,
            head_dim,
            c_attn_w,
            c_attn_b,
            c_proj_w,
            c_proj_b,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &ArrayD<f32>,
        _attention_mask: Option<&ArrayD<f32>>, // Placeholder for attention mask
    ) -> Result<ArrayD<f32>, Box<dyn Error>> {
        let initial_shape = hidden_states.shape();
        if initial_shape.len() != 3 {
            return Err(format!(
                "Expected hidden_states to be 3D (batch, seq_len, n_embd), got shape: {:?}",
                initial_shape
            )
            .into());
        }
        let batch_size = initial_shape[0];
        let seq_len = initial_shape[1];
        // let current_n_embd = initial_shape[2]; // Should match self.n_embd

        // Reshape hidden_states for matrix multiplication: [B, S, E] -> [B*S, E]
        let hidden_states_reshaped_view = hidden_states
            .view()
            .into_shape((batch_size * seq_len, self.n_embd as usize))
            .map_err(|e: ShapeError| e.to_string())?; // Convert ShapeError to String error

        // QKV Projection: (B*S, E) @ (E, 3E) -> (B*S, 3E)
        // c_attn_w is ArrayD, needs to be viewed as Ix2 for dot product
        let c_attn_w_view = self.c_attn_w.view().into_dimensionality::<Ix2>()
            .map_err(|e| format!("Failed to view c_attn_w as Ix2: {}", e))?;
        let qkv_projected = hidden_states_reshaped_view.dot(&c_attn_w_view);

        // Add bias: (B*S, 3E) + (3E) -> (B*S, 3E)
        // c_attn_b is ArrayD, needs to be viewed as Ix1 for broadcasting
        let c_attn_b_view = self.c_attn_b.view().into_dimensionality::<Ix1>()
            .map_err(|e| format!("Failed to view c_attn_b as Ix1: {}", e))?;
        let qkv_biased = qkv_projected + &c_attn_b_view; // qkv_projected is owned Array<_, Ix2>

        // Reshape back to [B, S, 3E]
        let qkv = qkv_biased
            .into_shape((batch_size, seq_len, 3 * self.n_embd as usize))
            .map_err(|e: ShapeError| e.to_string())? // Convert ShapeError to String error
            .into_dyn(); // Convert Array<_, Ix3> to ArrayD<_>

        // Split Q, K, V
        // Each will have shape [B, S, E]
        let q = qkv.slice_axis(Axis(2), Slice::from(0..(self.n_embd as usize))).to_owned();
        let k = qkv.slice_axis(Axis(2), Slice::from((self.n_embd as usize)..(2 * self.n_embd as usize))).to_owned();
        let v = qkv.slice_axis(Axis(2), Slice::from((2 * self.n_embd as usize)..(3 * self.n_embd as usize))).to_owned();

        // Reshape and transpose Q, K, V for multi-head attention
        // Q: [B, S, E] -> [B, S, n_head, head_dim] -> [B, n_head, S, head_dim]
        // Reshape and transpose Q, K, V for multi-head attention
        // Q: [B, S, E] -> [B, S, n_head, head_dim] -> [B, n_head, S, head_dim]
        let q_split = q
            .into_shape((batch_size, seq_len, self.n_head as usize, self.head_dim as usize))
            .map_err(|e: ShapeError| format!("Error reshaping Q: {}", e.to_string()))?
            .permuted_axes([0, 2, 1, 3]); // Shape: [B, H, S, Dh]

        let k_split = k
            .into_shape((batch_size, seq_len, self.n_head as usize, self.head_dim as usize))
            .map_err(|e: ShapeError| format!("Error reshaping K: {}", e.to_string()))?
            .permuted_axes([0, 2, 1, 3]); // Shape: [B, H, S, Dh]

        let v_split = v
            .into_shape((batch_size, seq_len, self.n_head as usize, self.head_dim as usize))
            .map_err(|e: ShapeError| format!("Error reshaping V: {}", e.to_string()))?
            .permuted_axes([0, 2, 1, 3]); // Shape: [B, H, S, Dh]

        // Scaled Dot-Product Attention
        // Transpose K for QK^T: [B, H, S, Dh] -> [B, H, Dh, S]
        let k_split_transposed = k_split.permuted_axes([0, 1, 3, 2]);

        // Batched matrix multiplication for attn_scores = q_split @ k_split_transposed
        // Result shape: [B, H, S, S]
        let mut attn_scores_arr = Array::zeros((batch_size, self.n_head as usize, seq_len, seq_len));
        for b in 0..batch_size {
            for h_idx in 0..(self.n_head as usize) {
                let q_slice = q_split.slice(s![b, h_idx, .., ..]);
                let k_t_slice = k_split_transposed.slice(s![b, h_idx, .., ..]);
                let score_slice = q_slice.dot(&k_t_slice);
                attn_scores_arr.slice_mut(s![b, h_idx, .., ..]).assign(&score_slice);
            }
        }
        
        let mut attn_scores = attn_scores_arr.into_dyn(); // Convert to ArrayD for further processing

        // Scale attention scores
        attn_scores = attn_scores / (self.head_dim as f32).sqrt();

        // Causal Mask Application
        if let Some(mask) = _attention_mask {
             // Ensure mask is broadcastable. Example: mask might be [B, 1, 1, S] or [B, 1, S, S]
            attn_scores = attn_scores + mask;
        } else {
            if seq_len > 1 { // Causal mask only needed if seq_len > 1
                let mut causal_mask_slice = Array::from_elem((seq_len, seq_len), 0.0f32);
                for i in 0..seq_len {
                    for j in (i + 1)..seq_len {
                        causal_mask_slice[[i,j]] = f32::NEG_INFINITY;
                    }
                }
                let causal_mask = causal_mask_slice
                    .into_dyn()
                    .insert_axis(Axis(0)) // Shape [1, S, S] - Removed ?
                    .insert_axis(Axis(0)); // Shape [1, 1, S, S] - Removed ?
                attn_scores = attn_scores + &causal_mask;
            }
        }

        // Softmax
        let attn_probs = softmax(&attn_scores, 3)?; // Softmax along the last axis (dim 3: key_seq_len)

        // Apply to V (P@V)
        // attn_probs shape: [B, H, S, S], v_split shape: [B, H, S, Dh]
        // Result context_layer_arr shape: [B, H, S, Dh]
        let v_split_arr4 = v_split.view().into_dimensionality::<Ix4>()
            .map_err(|e| format!("Failed to view v_split as Ix4: {}", e))?;
        let attn_probs_arr4 = attn_probs.view().into_dimensionality::<Ix4>()
             .map_err(|e| format!("Failed to view attn_probs as Ix4: {}", e))?;
        
        let mut context_layer_arr = Array::zeros((batch_size, self.n_head as usize, seq_len, self.head_dim as usize));
        for b in 0..batch_size {
            for h_idx in 0..(self.n_head as usize) {
                let prob_slice = attn_probs_arr4.slice(s![b, h_idx, .., ..]);
                let v_slice = v_split_arr4.slice(s![b, h_idx, .., ..]);
                let ctx_slice = prob_slice.dot(&v_slice);
                context_layer_arr.slice_mut(s![b, h_idx, .., ..]).assign(&ctx_slice);
            }
        }

        // Concatenate Heads
        // [B, H, S, Dh] -> [B, S, H, Dh]
        let context_transposed = context_layer_arr.permuted_axes([0, 2, 1, 3]);
        // [B, S, H, Dh] -> [B, S, E=H*Dh]
        let context_reshaped = context_transposed
            .as_standard_layout() // Ensure memory layout is correct for reshape
            .into_shape((batch_size, seq_len, self.n_embd as usize))
            .map_err(|e: ShapeError| format!("Error reshaping context layer: {}", e.to_string()))?;

        // Final Linear Projection
        // [B, S, E] -> [B*S, E]
        let context_reshaped_2d = context_reshaped
            .into_shape((batch_size * seq_len, self.n_embd as usize))
            .map_err(|e: ShapeError| format!("Error reshaping context for final projection: {}", e.to_string()))?;

        let c_proj_w_view = self.c_proj_w.view().into_dimensionality::<Ix2>()
            .map_err(|e| format!("Failed to view c_proj_w as Ix2: {}", e))?;
        let c_proj_b_view = self.c_proj_b.view().into_dimensionality::<Ix1>()
            .map_err(|e| format!("Failed to view c_proj_b as Ix1: {}", e))?;

        let output_2d = context_reshaped_2d.dot(&c_proj_w_view) + &c_proj_b_view;
        
        let output = output_2d
            .into_shape((batch_size, seq_len, self.n_embd as usize))
            .map_err(|e: ShapeError| format!("Error reshaping output: {}", e.to_string()))?
            .into_dyn(); // Convert to ArrayD as the final output type

        Ok(output)
    }
}
