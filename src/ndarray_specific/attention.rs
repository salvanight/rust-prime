use crate::accelerator::{CpuTensor, Device, Module, Tensor};
use std::error::Error;

#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    c_attn_weight: CpuTensor<f32>,
    c_attn_bias: CpuTensor<f32>,
    c_proj_weight: CpuTensor<f32>,
    c_proj_bias: CpuTensor<f32>,
    num_heads: usize,
    device: Device,
}

impl MultiHeadAttention {
    pub fn new(
        c_attn_weight_data: &[f32],
        c_attn_weight_shape: &[usize],
        c_attn_bias_data: &[f32],
        c_attn_bias_shape: &[usize],
        c_proj_weight_data: &[f32],
        c_proj_weight_shape: &[usize],
        c_proj_bias_data: &[f32],
        c_proj_bias_shape: &[usize],
        num_heads: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let c_attn_weight =
            CpuTensor::from_data_and_shape(c_attn_weight_data, c_attn_weight_shape, Device::CPU)?;
        let c_attn_bias =
            CpuTensor::from_data_and_shape(c_attn_bias_data, c_attn_bias_shape, Device::CPU)?;
        let c_proj_weight =
            CpuTensor::from_data_and_shape(c_proj_weight_data, c_proj_weight_shape, Device::CPU)?;
        let c_proj_bias =
            CpuTensor::from_data_and_shape(c_proj_bias_data, c_proj_bias_shape, Device::CPU)?;

        Ok(Self {
            c_attn_weight,
            c_attn_bias,
            c_proj_weight,
            c_proj_bias,
            num_heads,
            device: Device::CPU,
        })
    }
}

use ndarray::{Array, ArrayD, Axis, IxDyn, s, Zip};
use std::f32::consts::SQRT_2; // For GELU, though not used in MHA

// Helper function for softmax
fn softmax(array: &mut ArrayD<f32>, axis: Axis) {
    if array.ndim() == 0 { return } // Cannot apply softmax to a scalar
    array.axis_iter_mut(axis).for_each(|mut lane| {
        let max_val = lane.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        if max_val.is_finite() {
            lane.mapv_inplace(|x| (x - max_val).exp());
            let sum = lane.sum();
            if sum != 0.0 { // Avoid division by zero if all exps are zero (e.g. after extreme values)
                lane.mapv_inplace(|x| x / sum);
            }
        } else {
            // Handle cases where max_val is -inf (all elements are -inf) or +inf or NaN
            // For -inf, all exps will be 0. For +inf/NaN, behavior might be undefined or lead to NaNs.
            // A simple approach is to set to uniform or zero, depending on desired outcome for such edge cases.
            let uniform = 1.0 / lane.len() as f32;
            lane.fill(uniform); // Or fill(0.0) if that's more appropriate
        }
    });
}


impl Module for MultiHeadAttention {
    type Input = (CpuTensor<f32>, Option<CpuTensor<f32>>); // (hidden_states, attention_mask)
    type Output = CpuTensor<f32>;

    fn forward(&self, input: Self::Input) -> Result<Self::Output, Box<dyn Error>> {
        let (hidden_states_tensor, attention_mask_opt_tensor) = input;

        if hidden_states_tensor.device() != self.device {
            return Err(format!("MHA on {:?} received hidden_states on {:?}", self.device, hidden_states_tensor.device()).into());
        }
        if let Some(mask_tensor) = &attention_mask_opt_tensor {
            if mask_tensor.device() != self.device {
                return Err(format!("MHA on {:?} received mask on {:?}", self.device, mask_tensor.device()).into());
            }
        }
        if self.device != Device::CPU {
             return Err(format!("MHA on {:?} is not supported for forward pass, only CPU.", self.device).into());
        }

        let hidden_states_slice = hidden_states_tensor.as_slice()?;
        let original_shape = hidden_states_tensor.shape(); // e.g. [batch, seq_len, n_embd] or [seq_len, n_embd]

        let (batch_size, seq_len, n_embd) = match original_shape.len() {
            2 => (1, original_shape[0], original_shape[1]), // [seq_len, n_embd]
            3 => (original_shape[0], original_shape[1], original_shape[2]), // [batch, seq_len, n_embd]
            _ => return Err(format!("Unsupported hidden_states rank: {}. Expected 2 or 3.", original_shape.len()).into()),
        };
        
        if n_embd % self.num_heads != 0 {
            return Err(format!("n_embd ({}) must be divisible by num_heads ({})", n_embd, self.num_heads).into());
        }
        let head_dim = n_embd / self.num_heads;

        // Wrap hidden_states in ndarray::ArrayView
        let hidden_states_arr = Array::from_shape_vec((batch_size, seq_len, n_embd), hidden_states_slice.to_vec())?
                                .into_dyn(); // Convert to ArrayD for flexibility if needed, or keep typed

        // --- Compute Q, K, V ---
        let c_attn_w_slice = self.c_attn_weight.as_slice()?;
        let c_attn_b_slice = self.c_attn_bias.as_slice()?;
        
        // Reshape c_attn_weight for matmul: [n_embd, 3 * n_embd]
        let c_attn_w_arr = Array::from_shape_vec(self.c_attn_weight.shape().to_vec(), c_attn_w_slice.to_vec())?;
        let c_attn_b_arr = Array::from_shape_vec(self.c_attn_bias.shape().to_vec(), c_attn_b_slice.to_vec())?;

        // hidden_states_arr: [batch, seq_len, n_embd]
        // c_attn_w_arr: [n_embd, 3 * n_embd]
        // qkv: [batch, seq_len, 3 * n_embd]
        let mut qkv = hidden_states_arr.dot(&c_attn_w_arr);
        qkv = qkv + &c_attn_b_arr; // Broadcasting bias

        // Split Q, K, V
        // qkv has shape [batch_size, seq_len, 3 * n_embd]
        let mut qkv_parts = qkv.into_shape((batch_size, seq_len, 3, n_embd))?.permuted_axes([2,0,1,3]);
        // Now qkv_parts has shape [3, batch_size, seq_len, n_embd]
        
        let q = qkv_parts.slice(s![0, .., .., ..]).to_owned(); // [batch, seq_len, n_embd]
        let k = qkv_parts.slice(s![1, .., .., ..]).to_owned(); // [batch, seq_len, n_embd]
        let v = qkv_parts.slice(s![2, .., .., ..]).to_owned(); // [batch, seq_len, n_embd]

        // --- Reshape for Multi-Head ---
        // q,k,v: [batch, seq_len, n_embd] -> [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        let q_multi_head = q.into_shape((batch_size, seq_len, self.num_heads, head_dim))?.permuted_axes([0,2,1,3]);
        let k_multi_head = k.into_shape((batch_size, seq_len, self.num_heads, head_dim))?.permuted_axes([0,2,1,3]);
        let v_multi_head = v.into_shape((batch_size, seq_len, self.num_heads, head_dim))?.permuted_axes([0,2,1,3]);
        
        // --- Scaled Dot-Product Attention ---
        // k_multi_head_t: [batch, num_heads, head_dim, seq_len]
        let k_multi_head_t = k_multi_head.permuted_axes([0,1,3,2]);
        // scores: [batch, num_heads, seq_len, seq_len]
        let mut scores = q_multi_head.dot(&k_multi_head_t);
        scores.mapv_inplace(|x| x / (head_dim as f32).sqrt());

        if let Some(mask_tensor) = attention_mask_opt_tensor {
            let mask_slice = mask_tensor.as_slice()?;
            // Assuming mask is [batch_size, 1, 1, seq_len] or [1, 1, seq_len, seq_len] for broadcasting
            // A common mask has 0 for attended, -1e9 (or similar large negative) for masked.
            let mask_arr = Array::from_shape_vec(mask_tensor.shape().to_vec(), mask_slice.to_vec())?;
            scores = scores + mask_arr; // Broadcasting mask
        }
        
        softmax(&mut scores.into_dyn(), Axis(scores.ndim() - 1)); // Softmax along the last dim (seq_len)
        
        // attn_values: [batch, num_heads, seq_len, head_dim]
        let attn_values = scores.dot(&v_multi_head);

        // --- Concatenate Heads ---
        // Transpose: [batch, seq_len, num_heads, head_dim] then reshape to [batch, seq_len, n_embd]
        let cat_heads = attn_values.permuted_axes([0,2,1,3]).as_standard_layout();
        let reshaped_cat_heads = cat_heads.into_shape((batch_size, seq_len, n_embd))?;
        
        // --- Final Projection ---
        let c_proj_w_slice = self.c_proj_weight.as_slice()?;
        let c_proj_b_slice = self.c_proj_bias.as_slice()?;
        let c_proj_w_arr = Array::from_shape_vec(self.c_proj_weight.shape().to_vec(), c_proj_w_slice.to_vec())?;
        let c_proj_b_arr = Array::from_shape_vec(self.c_proj_bias.shape().to_vec(), c_proj_b_slice.to_vec())?;

        let mut final_output = reshaped_cat_heads.dot(&c_proj_w_arr);
        final_output = final_output + &c_proj_b_arr; // Broadcasting bias

        // Reshape back to original rank if it was 2D
        let final_output_data = final_output.into_raw_vec();
        let output_shape = if original_shape.len() == 2 {
            vec![seq_len, n_embd]
        } else {
            vec![batch_size, seq_len, n_embd]
        };
        
        CpuTensor::from_data_and_shape(&final_output_data, &output_shape, self.device.clone())
    }

    fn to_device(&mut self, device: Device) -> Result<(), Box<dyn Error>> {
        if self.device == device {
            return Ok(());
        }
        if device != Device::CPU {
            return Err(format!(
                "MultiHeadAttention with CpuTensor weights cannot be moved to {:?}. Only CPU is supported.",
                device
            )
            .into());
        }

        self.c_attn_weight = self.c_attn_weight.to_device(device.clone())?;
        self.c_attn_bias = self.c_attn_bias.to_device(device.clone())?;
        self.c_proj_weight = self.c_proj_weight.to_device(device.clone())?;
        self.c_proj_bias = self.c_proj_bias.to_device(device.clone())?;
        self.device = device;
        Ok(())
    }

    fn current_device(&self) -> Device {
        self.device.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accelerator::{CpuTensor, Device, Tensor}; 
    use crate::config::GPT2Config; // Not strictly needed for MHA test_mha, but good for consistency

    // Helper to create MultiHeadAttention for testing
    fn test_mha(n_embd: usize, num_heads: usize) -> MultiHeadAttention {
        // Ensure n_embd is divisible by num_heads for valid head_dim
        assert_eq!(n_embd % num_heads, 0, "n_embd must be divisible by num_heads");
        
        let head_dim = n_embd / num_heads;

        // Initialize weights and biases with zeros for simplicity
        // c_attn: combines Q, K, V projections. Output is 3 * n_embd
        let c_attn_weight_data = vec![0.0f32; n_embd * (3 * n_embd)];
        let c_attn_weight_shape = vec![n_embd, 3 * n_embd];
        let c_attn_bias_data = vec![0.0f32; 3 * n_embd];
        let c_attn_bias_shape = vec![3 * n_embd];

        // c_proj: projects concatenated heads back to n_embd
        let c_proj_weight_data = vec![0.0f32; n_embd * n_embd];
        let c_proj_weight_shape = vec![n_embd, n_embd];
        let c_proj_bias_data = vec![0.0f32; n_embd];
        let c_proj_bias_shape = vec![n_embd];

        MultiHeadAttention::new(
            &c_attn_weight_data, &c_attn_weight_shape,
            &c_attn_bias_data, &c_attn_bias_shape,
            &c_proj_weight_data, &c_proj_weight_shape,
            &c_proj_bias_data, &c_proj_bias_shape,
            num_heads,
        ).expect("Failed to create MultiHeadAttention for test")
    }

    #[test]
    fn test_mha_forward_2d_input() {
        let n_embd = 12;
        let num_heads = 4;
        let seq_len = 5;
        let attention = test_mha(n_embd, num_heads);

        let hidden_states_data: Vec<f32> = (0..(seq_len * n_embd)).map(|x| x as f32 * 0.1).collect();
        let hidden_states_shape = vec![seq_len, n_embd];
        let hidden_states = CpuTensor::from_data_and_shape(&hidden_states_data, &hidden_states_shape, Device::CPU).unwrap();

        let result = attention.forward((hidden_states, None));
        assert!(result.is_ok(), "MHA forward failed for 2D input: {:?}", result.err());
        let output_tensor = result.unwrap();
        assert_eq!(output_tensor.shape(), &[seq_len, n_embd], "Output shape mismatch for 2D input.");
        assert_eq!(output_tensor.device(), Device::CPU, "Output device mismatch for 2D input.");
        // With zero weights and biases, the output of matmuls and additions should be zero.
        let output_slice = output_tensor.as_slice().unwrap();
        assert!(output_slice.iter().all(|&x| (x - 0.0).abs() < f32::EPSILON), "Output data is not all zeros for 2D input.");
    }

    #[test]
    fn test_mha_forward_3d_input() {
        let n_embd = 12;
        let num_heads = 4;
        let batch_size = 2;
        let seq_len = 5;
        let attention = test_mha(n_embd, num_heads);

        let hidden_states_data: Vec<f32> = (0..(batch_size * seq_len * n_embd)).map(|x| x as f32 * 0.1).collect();
        let hidden_states_shape = vec![batch_size, seq_len, n_embd];
        let hidden_states = CpuTensor::from_data_and_shape(&hidden_states_data, &hidden_states_shape, Device::CPU).unwrap();

        let result = attention.forward((hidden_states, None));
        assert!(result.is_ok(), "MHA forward failed for 3D input: {:?}", result.err());
        let output_tensor = result.unwrap();
        assert_eq!(output_tensor.shape(), &[batch_size, seq_len, n_embd], "Output shape mismatch for 3D input.");
        assert_eq!(output_tensor.device(), Device::CPU, "Output device mismatch for 3D input.");
        let output_slice = output_tensor.as_slice().unwrap();
        assert!(output_slice.iter().all(|&x| (x - 0.0).abs() < f32::EPSILON), "Output data is not all zeros for 3D input.");
    }

    #[test]
    fn test_mha_forward_with_mask_3d() {
        let n_embd = 12;
        let num_heads = 4;
        let batch_size = 1; // Simpler for mask testing
        let seq_len = 5;
        let attention = test_mha(n_embd, num_heads);

        let hidden_states_data: Vec<f32> = (0..(batch_size * seq_len * n_embd)).map(|x| x as f32 * 0.1).collect();
        let hidden_states_shape = vec![batch_size, seq_len, n_embd];
        let hidden_states = CpuTensor::from_data_and_shape(&hidden_states_data, &hidden_states_shape, Device::CPU).unwrap();

        // Mask shape: [batch_size, num_heads, query_seq_len, key_seq_len]
        // Here, query_seq_len = key_seq_len = seq_len
        let mask_data = vec![0.0f32; batch_size * num_heads * seq_len * seq_len];
        let mask_shape = vec![batch_size, num_heads, seq_len, seq_len];
        let attention_mask = CpuTensor::from_data_and_shape(&mask_data, &mask_shape, Device::CPU).unwrap();

        let result = attention.forward((hidden_states, Some(attention_mask)));
        assert!(result.is_ok(), "MHA forward with mask failed: {:?}", result.err());
        let output_tensor = result.unwrap();
        assert_eq!(output_tensor.shape(), &[batch_size, seq_len, n_embd], "Output shape mismatch with mask.");
        assert_eq!(output_tensor.device(), Device::CPU, "Output device mismatch with mask.");
        let output_slice = output_tensor.as_slice().unwrap();
        assert!(output_slice.iter().all(|&x| (x - 0.0).abs() < f32::EPSILON), "Output data is not all zeros with zero mask.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accelerator::{CpuTensor, Device, Tensor}; // Module is already in super
    use crate::config::GPT2Config; // For default config, though we'll specify params

    // Helper to create MultiHeadAttention for testing
    fn test_mha(n_embd: usize, num_heads: usize) -> MultiHeadAttention {
        let c_attn_weight_data = vec![0.0f32; n_embd * (3 * n_embd)];
        let c_attn_weight_shape = vec![n_embd, 3 * n_embd];
        let c_attn_bias_data = vec![0.0f32; 3 * n_embd];
        let c_attn_bias_shape = vec![3 * n_embd];

        let c_proj_weight_data = vec![0.0f32; n_embd * n_embd];
        let c_proj_weight_shape = vec![n_embd, n_embd];
        let c_proj_bias_data = vec![0.0f32; n_embd];
        let c_proj_bias_shape = vec![n_embd];

        MultiHeadAttention::new(
            &c_attn_weight_data, &c_attn_weight_shape,
            &c_attn_bias_data, &c_attn_bias_shape,
            &c_proj_weight_data, &c_proj_weight_shape,
            &c_proj_bias_data, &c_proj_bias_shape,
            num_heads,
        ).unwrap()
    }

    #[test]
    fn test_mha_forward_2d_input() {
        let n_embd = 12;
        let num_heads = 4;
        let seq_len = 5;
        let attention = test_mha(n_embd, num_heads);

        let hidden_states_data: Vec<f32> = (0..(seq_len * n_embd)).map(|x| x as f32).collect();
        let hidden_states_shape = vec![seq_len, n_embd];
        let hidden_states = CpuTensor::from_data_and_shape(&hidden_states_data, &hidden_states_shape, Device::CPU).unwrap();

        let result = attention.forward((hidden_states, None));
        assert!(result.is_ok());
        let output_tensor = result.unwrap();
        assert_eq!(output_tensor.shape(), &[seq_len, n_embd]);
        assert_eq!(output_tensor.device(), Device::CPU);
    }

    #[test]
    fn test_mha_forward_3d_input() {
        let n_embd = 12;
        let num_heads = 4;
        let batch_size = 2;
        let seq_len = 5;
        let attention = test_mha(n_embd, num_heads);

        let hidden_states_data: Vec<f32> = (0..(batch_size * seq_len * n_embd)).map(|x| x as f32).collect();
        let hidden_states_shape = vec![batch_size, seq_len, n_embd];
        let hidden_states = CpuTensor::from_data_and_shape(&hidden_states_data, &hidden_states_shape, Device::CPU).unwrap();

        let result = attention.forward((hidden_states, None));
        assert!(result.is_ok());
        let output_tensor = result.unwrap();
        assert_eq!(output_tensor.shape(), &[batch_size, seq_len, n_embd]);
        assert_eq!(output_tensor.device(), Device::CPU);
    }

    #[test]
    fn test_mha_forward_with_mask_3d() {
        let n_embd = 12;
        let num_heads = 4;
        let batch_size = 1; // Simpler for mask testing
        let seq_len = 5;
        let attention = test_mha(n_embd, num_heads);

        let hidden_states_data: Vec<f32> = (0..(batch_size * seq_len * n_embd)).map(|x| x as f32).collect();
        let hidden_states_shape = vec![batch_size, seq_len, n_embd];
        let hidden_states = CpuTensor::from_data_and_shape(&hidden_states_data, &hidden_states_shape, Device::CPU).unwrap();

        // Mask shape: [batch_size, num_heads, seq_len, seq_len]
        // For this test, a simple broadcastable mask can also be [1, 1, seq_len, seq_len]
        // Or even [batch_size, 1, 1, seq_len] if the broadcasting rules in the implementation handle it.
        // The current implementation adds the mask directly, so needs full shape.
        let mask_data = vec![0.0f32; batch_size * num_heads * seq_len * seq_len];
        let mask_shape = vec![batch_size, num_heads, seq_len, seq_len];
        let attention_mask = CpuTensor::from_data_and_shape(&mask_data, &mask_shape, Device::CPU).unwrap();

        let result = attention.forward((hidden_states, Some(attention_mask)));
        assert!(result.is_ok());
        let output_tensor = result.unwrap();
        assert_eq!(output_tensor.shape(), &[batch_size, seq_len, n_embd]);
        assert_eq!(output_tensor.device(), Device::CPU);
    }
}
