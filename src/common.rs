use ndarray::{ArrayD, Array1, Axis};

#[derive(Debug)]
pub struct LayerNorm {
    pub(crate) _weight: ArrayD<f32>, // gamma
    pub(crate) _bias: ArrayD<f32>,   // beta
    epsilon: f32,
}

impl LayerNorm {
    pub fn new(n_embd: i32, epsilon: f32) -> Result<Self, Box<dyn std::error::Error>> {
        let weight = Array1::ones(n_embd as usize).into_dyn();
        let bias = Array1::zeros(n_embd as usize).into_dyn();
        
        Ok(Self { 
            _weight: weight,
            _bias: bias,
            epsilon,
        })
    }

    pub fn forward(&self, x: &ArrayD<f32>) -> Result<ArrayD<f32>, Box<dyn std::error::Error>> {
feat/gpt2-core-logic-and-weights
        let last_dim_idx = x.ndim() - 1;
        let axis = Axis(last_dim_idx);

        // Calculate mean and variance along the last dimension.
        // These will have one less dimension than x (e.g., [B, S] if x is [B, S, E])
        let mean = x.mean_axis(axis).ok_or("Failed to compute mean")?;
        let variance = x.var_axis(axis, 0.0); // ddof = 0 for population variance

        // To keep dimensions for broadcasting, insert the reduced axis back.
        // mean_kept_dims and variance_kept_dims will have shape e.g. [B, S, 1]
        let mean_kept_dims = mean.insert_axis(axis);
        let variance_kept_dims = variance.insert_axis(axis);

        // Normalize x
        // (x - mean_kept_dims)
        let x_minus_mean = x - &mean_kept_dims;
        // sqrt(variance_kept_dims + epsilon)
        let std_dev_inv = (&variance_kept_dims + self.epsilon).mapv(f32::sqrt).mapv(|v| 1.0 / v);
        
        let normalized_x = x_minus_mean * std_dev_inv;

        // Apply scale and shift
        // self._weight and self._bias are 1D [E]
        // ndarray should broadcast them correctly over normalized_x [B, S, E]
        let y = normalized_x * &self._weight + &self._bias;
        
        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, s}; // Import s macro, remove arr3, Ix3
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_layer_norm_forward() -> Result<(), Box<dyn std::error::Error>> {
        let n_embd = 4;
        let epsilon = 1e-5;
        let layer_norm = LayerNorm::new(n_embd, epsilon)?;

        // Input tensor: batch_size=2, seq_len=3, n_embd=4
        let x_data = Array::from_shape_vec((2, 3, 4), vec![
            // Batch 1
            1.0, 2.0, 3.0, 4.0,  // Seq 1
            5.0, 6.0, 7.0, 8.0,  // Seq 2
            9.0, 10.0, 11.0, 12.0, // Seq 3
            // Batch 2
            -1.0, -2.0, -3.0, -4.0, // Seq 1
            -5.0, -6.0, -7.0, -8.0, // Seq 2
            -9.0, -10.0, -11.0, -12.0 // Seq 3
        ])?.into_dyn();

        let y = layer_norm.forward(&x_data)?;

        // 1. Check output shape
        assert_eq!(y.shape(), x_data.shape(), "Output shape mismatch");

        // 2. Check mean of the output over the last dimension is close to 0
        //    mean_axis now correctly returns an Option and drops the axis.
        let y_mean_last_dim = y.mean_axis(Axis(y.ndim() - 1)).expect("Mean calculation failed for y");
        for val in y_mean_last_dim.iter() {
            assert_abs_diff_eq!(*val, 0.0, epsilon = 1e-6);
        }

        // 3. Check variance of the output over the last dimension is close to 1
        //    var_axis also drops the axis.
        let y_var_last_dim = y.var_axis(Axis(y.ndim() - 1), 0.0); 
        for val in y_var_last_dim.iter() {
            assert_abs_diff_eq!(*val, 1.0, epsilon = 1e-6);
        }
        
        // Specific value checks (optional, but good for sanity)
        let first_val_output = y[[0_usize, 0_usize, 0_usize]];
        
        // Slice x_data to get the first vector [1.0, 2.0, 3.0, 4.0]
        // s!(0_usize, 0_usize, ..) defines a slice for the first batch, first sequence, all embeddings.
        // This results in a 1D ArrayView.
        let x_slice_00 = x_data.slice(s![0_usize, 0_usize, ..]); 
        
        // Calculate mean and variance for this specific 1D slice.
        // .mean() on a 1D array returns Option<f32>
        // .var(0.0) on a 1D array returns f32 (population variance)
        let mean_00 = x_slice_00.mean().expect("Mean calculation for slice failed"); 
        let var_00 = x_slice_00.var(0.0); 
        
        // Access the first element of the 1D slice for the formula
        let x_000 = x_slice_00[0_usize]; 
        
        let expected_first_val = (x_000 - mean_00) / (var_00 + epsilon).sqrt();
        assert_abs_diff_eq!(first_val_output, expected_first_val, epsilon = 1e-6);

        Ok(())
    }

    #[test]
    fn test_layer_norm_new() -> Result<(), Box<dyn std::error::Error>> {
        let n_embd = 10;
        let epsilon = 1e-6;
        let ln = LayerNorm::new(n_embd, epsilon)?;

        assert_eq!(ln._weight.shape(), &[n_embd as usize]);
        assert_eq!(ln._bias.shape(), &[n_embd as usize]);
        assert_eq!(ln.epsilon, epsilon);

        assert!(ln._weight.iter().all(|&x| (x - 1.0).abs() < f32::EPSILON));
        assert!(ln._bias.iter().all(|&x| (x - 0.0).abs() < f32::EPSILON));
        Ok(())
=======
        // Placeholder for LayerNorm forward pass
        // Actual implementation would normalize x using self.weight and self.bias
        // println!("LayerNorm forward called with tensor of shape: {:?}", x.shape());
        // For now, just return a clone for shape testing.
        Ok(x.clone()) // Simplest placeholder
        // todo!("Implement LayerNorm forward pass");
 main
    }
}

pub type ModelKVCache = Vec<Vec<f32>>;
