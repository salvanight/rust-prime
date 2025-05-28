use ndarray::{ArrayD, IxDyn};

#[derive(Debug)]
pub struct MLP {
    // Placeholder fields for weights/biases
    _c_fc_weight: ArrayD<f32>, // Weight for the first fully connected layer
    _c_fc_bias: ArrayD<f32>,   // Bias for the first fully connected layer
    _c_proj_weight: ArrayD<f32>, // Weight for the projection layer
    _c_proj_bias: ArrayD<f32>,   // Bias for the projection layer
    // activation_function: String, // Could be stored if it varies, or assumed (e.g., GELU)
}

impl MLP {
    pub fn new(
        n_embd: i32, 
        n_inner: i32 
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if n_embd <= 0 || n_inner <= 0 {
            return Err("n_embd and n_inner must be positive".into());
        }
        // c_fc_weight shape: [n_embd, n_inner]
        // c_fc_bias shape: [n_inner]
        // c_proj_weight shape: [n_inner, n_embd]
        // c_proj_bias shape: [n_embd]
        let c_fc_weight_shape = IxDyn(&[n_embd as usize, n_inner as usize]);
        let c_fc_bias_shape = IxDyn(&[n_inner as usize]);
        let c_proj_weight_shape = IxDyn(&[n_inner as usize, n_embd as usize]);
        let c_proj_bias_shape = IxDyn(&[n_embd as usize]);

        Ok(Self {
            _c_fc_weight: ArrayD::zeros(c_fc_weight_shape),
            _c_fc_bias: ArrayD::zeros(c_fc_bias_shape),
            _c_proj_weight: ArrayD::zeros(c_proj_weight_shape),
            _c_proj_bias: ArrayD::zeros(c_proj_bias_shape),
        })
    }

    pub fn forward(&self, hidden_states: &ArrayD<f32>) -> Result<ArrayD<f32>, Box<dyn std::error::Error>> {
        // Placeholder for MLP forward pass.
        // For testing, this should return a tensor of the same shape as the input,
        // as an MLP block (n_embd -> n_inner -> n_embd) preserves the last dimension's size.
        // println!("MLP forward called with hidden_states shape: {:?}", hidden_states.shape());

        if hidden_states.ndim() < 1 { // Should be at least 1D (e.g. [n_embd])
             return Err(format!("Expected at least 1D input, got {}D", hidden_states.ndim()).into());
        }
        // let n_embd_input = hidden_states.shape().last().unwrap_or(&0);
        // let n_embd_config = self._c_proj_bias.shape()[0]; // Infer n_embd from bias shape
        // if *n_embd_input != n_embd_config {
        //      return Err(format!("Input embedding dimension ({}) does not match model n_embd ({})", n_embd_input, n_embd_config).into());
        // }
        
        // For now, to allow testing other parts, return a clone.
        // This is NOT a correct MLP implementation.
        Ok(hidden_states.clone())
        // todo!("Implement MLP forward pass with linear layers and activation");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array; // For arr3

    #[test]
    fn test_mlp_new_valid_params() {
        let n_embd = 768;
        let n_inner = 4 * n_embd; // Typical GPT-2 inner dimension
        let mlp_result = MLP::new(n_embd, n_inner);
        assert!(mlp_result.is_ok());
        let mlp = mlp_result.unwrap();

        assert_eq!(mlp._c_fc_weight.shape(), &[n_embd as usize, n_inner as usize]);
        assert_eq!(mlp._c_fc_bias.shape(), &[n_inner as usize]);
        assert_eq!(mlp._c_proj_weight.shape(), &[n_inner as usize, n_embd as usize]);
        assert_eq!(mlp._c_proj_bias.shape(), &[n_embd as usize]);
    }

    #[test]
    fn test_mlp_new_invalid_params() {
        assert!(MLP::new(0, 3072).is_err());
        assert!(MLP::new(768, 0).is_err());
        assert!(MLP::new(0, 0).is_err());
    }

    #[test]
    fn test_mlp_forward_shape() {
        let n_embd = 4;
        let n_inner = 4 * n_embd;
        let mlp = MLP::new(n_embd, n_inner).unwrap();

        let batch_size = 1;
        let seq_len = 3;
        // Input shape: [batch_size, seq_len, n_embd]
        let input_hidden_states = ArrayD::zeros(IxDyn(&[batch_size, seq_len, n_embd as usize]));
        
        let forward_result = mlp.forward(&input_hidden_states);
        assert!(forward_result.is_ok(), "MLP forward failed: {:?}", forward_result.err());
        let output = forward_result.unwrap();
        
        // Placeholder forward returns clone, so shape is the same.
        // A real MLP would also typically result in the same output shape [batch_size, seq_len, n_embd].
        assert_eq!(output.shape(), &[batch_size, seq_len, n_embd as usize], "Output shape mismatch");
    }
    
    // Value-based test is not feasible until the forward pass is implemented with actual logic.
    // #[test]
    // #[ignore] // Ignored because forward is not fully implemented
    // fn test_mlp_forward_values_simple() {
    //     // This test would require setting specific weights and biases
    //     // and manually calculating the expected output for a very small input.
    //     // For now, it's a placeholder.
    //     todo!("Implement value-based test for MLP forward pass");
    // }
}