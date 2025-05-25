use ndarray::{Array, ArrayD, Ix1, Ix2, ShapeError};
use std::error::Error;
use std::f32::consts::PI;

// GELU Activation Function (approximation from GPT-2 "gelu_new")
fn gelu(x: &ArrayD<f32>) -> ArrayD<f32> {
    let x_cubed = x.mapv(|v| v.powi(3));
    let coeff = (2.0 / PI).sqrt();
    // Intermediate calculations are element-wise
    let inner_term_values = (x + &(0.044715 * &x_cubed)).mapv(|val| val * coeff);
    let tanh_values = inner_term_values.mapv(|val| val.tanh());
    0.5 * x * (1.0 + &tanh_values)
}

#[derive(Debug)]
pub struct MLP {
    pub(crate) n_embd: i32, // Made pub(crate) for consistency if needed, though not directly set by loader
    pub(crate) n_inner: i32,// Made pub(crate) for consistency if needed, though not directly set by loader
    pub(crate) c_fc_w: ArrayD<f32>,   // Shape: [n_embd, n_inner]
    pub(crate) c_fc_b: ArrayD<f32>,   // Shape: [n_inner]
    pub(crate) c_proj_w: ArrayD<f32>, // Shape: [n_inner, n_embd]
    pub(crate) c_proj_b: ArrayD<f32>, // Shape: [n_embd]
}

impl MLP {
    pub fn new(n_embd: i32, n_inner: i32) -> Result<Self, Box<dyn Error>> {
        let c_fc_w = Array::zeros((n_embd as usize, n_inner as usize)).into_dyn();
        let c_fc_b = Array::zeros((n_inner as usize,)).into_dyn();
        let c_proj_w = Array::zeros((n_inner as usize, n_embd as usize)).into_dyn();
        let c_proj_b = Array::zeros((n_embd as usize,)).into_dyn();

        Ok(Self {
            n_embd,
            n_inner,
            c_fc_w,
            c_fc_b,
            c_proj_w,
            c_proj_b,
        })
    }

    pub fn forward(&self, hidden_states: &ArrayD<f32>) -> Result<ArrayD<f32>, Box<dyn Error>> {
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

        // Reshape hidden_states for matrix multiplication: [B, S, E] -> [B*S, E]
        let reshaped_hs = hidden_states
            .view()
            .into_shape((batch_size * seq_len, self.n_embd as usize))
            .map_err(|e: ShapeError| format!("Error reshaping hidden_states: {}", e.to_string()))?;

        // First Linear Transformation: (B*S, E) @ (E, I) + (I) -> (B*S, I)
        let c_fc_w_view = self.c_fc_w.view().into_dimensionality::<Ix2>()
            .map_err(|e| format!("Failed to view c_fc_w as Ix2: {}", e))?;
        let c_fc_b_view = self.c_fc_b.view().into_dimensionality::<Ix1>()
            .map_err(|e| format!("Failed to view c_fc_b as Ix1: {}", e))?;
        let h_fc = reshaped_hs.dot(&c_fc_w_view) + &c_fc_b_view; // h_fc is Array<f32, Ix2>

        // Apply GELU Activation
        let h_activated = gelu(&h_fc.into_dyn()); // gelu expects &ArrayD, returns ArrayD

        // Second Linear Transformation (Projection): (B*S, I) @ (I, E) + (E) -> (B*S, E)
        let h_activated_2d = h_activated.view().into_dimensionality::<Ix2>()
            .map_err(|e| format!("Failed to view h_activated as Ix2: {}", e))?;
        let c_proj_w_view = self.c_proj_w.view().into_dimensionality::<Ix2>()
            .map_err(|e| format!("Failed to view c_proj_w as Ix2: {}", e))?;
        let c_proj_b_view = self.c_proj_b.view().into_dimensionality::<Ix1>()
            .map_err(|e| format!("Failed to view c_proj_b as Ix1: {}", e))?;
        let output_2d = h_activated_2d.dot(&c_proj_w_view) + &c_proj_b_view; // output_2d is Array<f32, Ix2>

        // Reshape Output back to [B, S, E]
        let output = output_2d
            .into_shape((batch_size, seq_len, self.n_embd as usize))
            .map_err(|e: ShapeError| format!("Error reshaping output: {}", e.to_string()))?
            .into_dyn(); // Convert to ArrayD for the final output type

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr0, arr1, arr3, ArrayD, IxDyn}; // IxDyn for ArrayD::from_shape_vec
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_mlp_new() -> Result<(), Box<dyn Error>> {
        let n_embd = 768;
        let n_inner = 4 * n_embd;
        let mlp = MLP::new(n_embd, n_inner)?;

        assert_eq!(mlp.n_embd, n_embd);
        assert_eq!(mlp.n_inner, n_inner);
        assert_eq!(mlp.c_fc_w.shape(), &[n_embd as usize, n_inner as usize]);
        assert_eq!(mlp.c_fc_b.shape(), &[n_inner as usize]);
        assert_eq!(mlp.c_proj_w.shape(), &[n_inner as usize, n_embd as usize]);
        assert_eq!(mlp.c_proj_b.shape(), &[n_embd as usize]);
        Ok(())
    }

    #[test]
    fn test_gelu_fn() {
        // Test scalar 0.0
        let x0 = arr0(0.0f32).into_dyn();
        let y0 = gelu(&x0);
        assert_abs_diff_eq!(y0.first().unwrap(), &0.0, epsilon = 1e-6);

        // Test scalar 1.0
        let x1 = arr0(1.0f32).into_dyn();
        let y1 = gelu(&x1);
        // Expected: 0.5 * 1.0 * (1.0 + tanh(sqrt(2.0/PI) * (1.0 + 0.044715)))
        // sqrt(2.0/PI) approx 0.79788456
        // 1.0 + 0.044715 = 1.044715
        // arg_tanh = 0.79788456 * 1.044715 = 0.833518
        // tanh(0.833518) approx 0.68272
        // result = 0.5 * (1.0 + 0.68272) = 0.5 * 1.68272 = 0.84136
        assert_abs_diff_eq!(y1.first().unwrap(), &0.84136, epsilon = 1e-5);

        // Test with a small array
        let x_arr = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, 0.0, 1.0]).unwrap();
        let y_arr = gelu(&x_arr);
        let y_expected_vec = vec![
            0.5 * -1.0 * (1.0 + ((2.0/PI).sqrt() * (-1.0 + 0.044715 * (-1.0f32).powi(3)) ).tanh()), // approx -0.15865
            0.0,
            0.84136,
        ];
        for (val, expected) in y_arr.iter().zip(y_expected_vec.iter()) {
            assert_abs_diff_eq!(val, expected, epsilon = 1e-5);
        }
    }
    
    #[test]
    fn test_mlp_forward() -> Result<(), Box<dyn Error>> {
        let n_embd = 4; // Small embedding size for test
        let n_inner = 2 * n_embd; // 8
        let batch_size = 2;
        let seq_len = 3;

        let mut mlp = MLP::new(n_embd, n_inner)?;
        
        // Initialize weights and biases to something non-zero for a more robust test
        // For simplicity, let's make c_fc_w and c_proj_w identity-like if possible, and biases zero.
        // c_fc_w: [4, 8], c_proj_w: [8, 4]
        // This is a bit complex to make identity. Let's use ones for now.
        mlp.c_fc_w = Array::ones((n_embd as usize, n_inner as usize)).into_dyn();
        mlp.c_fc_b = Array::zeros((n_inner as usize,)).into_dyn();
        mlp.c_proj_w = Array::ones((n_inner as usize, n_embd as usize)).into_dyn();
        mlp.c_proj_b = Array::zeros((n_embd as usize,)).into_dyn();


        let hidden_states_vec: Vec<f32> = (0..(batch_size * seq_len * n_embd as usize))
            .map(|x| x as f32 * 0.1)
            .collect();
        let hidden_states = ArrayD::from_shape_vec(IxDyn(&[batch_size, seq_len, n_embd as usize]), hidden_states_vec)?;

        let output = mlp.forward(&hidden_states)?;

        assert_eq!(output.shape(), &[batch_size, seq_len, n_embd as usize]);
        
        // Optional: Check some values.
        // For example, if input is all 0.1, then reshaped_hs.dot(c_fc_w) where c_fc_w is ones
        // will be 0.1 * n_embd for each element in the intermediate layer (before GELU).
        // Example: hidden_states_reshaped_view[0] = [0.0, 0.1, 0.2, 0.3] (n_embd=4)
        // c_fc_w is all 1s, shape [4, 8].
        // h_fc[0,j] = sum(hidden_states_reshaped_view[0,i] * 1) = 0.0+0.1+0.2+0.3 = 0.6 for all j
        // So h_fc[0] = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6] (n_inner=8)
        // Then gelu([0.6, ...]) = [gelu(0.6), ...]
        // gelu(0.6) = 0.5 * 0.6 * (1.0 + tanh(sqrt(2/PI)*(0.6 + 0.044715 * 0.6^3)))
        // gelu(0.6) approx 0.506
        // h_activated[0] = [0.506, ..., 0.506]
        // c_proj_w is all 1s, shape [8,4]
        // output_2d[0,k] = sum(h_activated[0,j] * 1) = 0.506 * n_inner = 0.506 * 8 = 4.048
        // So, output first element's vector should be [4.048, 4.048, 4.048, 4.048]
        
        let first_output_vector = output.slice(s![0,0,..]);
        for val in first_output_vector.iter() {
             // This expected value needs to be precise based on the actual first input vector
             // hidden_states_vec[0..4] = [0.0, 0.1, 0.2, 0.3]
             // sum = 0.6
             // gelu_of_sum = gelu(0.6) which is approx 0.50601 (from test_gelu_fn, 0.5 * 0.6 * (1.0 + tanh(0.79788 * (0.6 + 0.044715 * 0.216))))
             // gelu(0.6) = 0.5 * 0.6 * (1.0 + tanh(0.7978845608028654 * (0.6 + 0.044715 * 0.216)))
             //           = 0.3 * (1.0 + tanh(0.7978845608028654 * (0.6 + 0.00965844)))
             //           = 0.3 * (1.0 + tanh(0.7978845608028654 * 0.60965844))
             //           = 0.3 * (1.0 + tanh(0.4864218)) = 0.3 * (1.0 + 0.45116) = 0.3 * 1.45116 = 0.435348
             let expected_val_after_gelu = gelu(&arr0(0.6f32 * n_embd as f32 / n_embd as f32).into_dyn()).first().unwrap().clone(); // No, this is wrong.
                                                                                                                                // Each element of h_fc is sum_of_input_elements_times_1 (if weights are 1)
                                                                                                                                // So, h_fc[b*s_idx + s_idx, inner_idx] = sum(input_emb_vector_for_that_token)
             let sum_first_input_emb_vector = hidden_states.slice(s![0,0,..]).sum(); // This is 0.6 for [0.0, 0.1, 0.2, 0.3]
             let gelu_of_sum = gelu(&arr0(sum_first_input_emb_vector).into_dyn()).first().unwrap().clone();
             let final_expected_val = gelu_of_sum * n_inner as f32; // Each element of output_2d will be this
            assert_abs_diff_eq!(*val, final_expected_val, epsilon = 1e-4);
        }


        Ok(())
    }
}