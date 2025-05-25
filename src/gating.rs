use ndarray::{Array1, Array2, ArrayD, Axis, IxDyn}; // IxDyn for ArrayD if GatingLayer handles general ArrayD
use ndarray_stats::QuantileExt; // For arg_max if needed, though softmax output is weights

// Helper for softmax (can be placed in common.rs or here if specific)
pub fn softmax(input: &Array1<f32>) -> Array1<f32> {
    let max_val = *input.max().unwrap_or(&0.0);
    let exp_values = input.mapv(|x| (x - max_val).exp());
    let sum_exp_values = exp_values.sum();
    if sum_exp_values == 0.0 {
        // Avoid division by zero if all exp_values are zero (e.g., very negative inputs)
        // Return uniform distribution or handle as error/specific case
        let len = input.len();
        if len > 0 {
            Array1::from_elem(len, 1.0 / len as f32)
        } else {
            Array1::zeros(0)
        }
    } else {
        exp_values / sum_exp_values
    }
}

#[derive(Debug)]
pub struct GatingLayer {
    // Weights: Assumed to transform input to a score per expert.
    // If input is [batch_size, n_embd], and we have n_experts,
    // weights matrix W might be [n_embd, n_experts].
    // Bias b might be [n_experts].
    weights: Array2<f32>, // Shape: [input_features, num_experts]
    biases: Array1<f32>,  // Shape: [num_experts]
}

impl GatingLayer {
    // num_input_features: e.g., n_embd
    // num_experts: the number of experts this gating layer decides between
    pub fn new(num_input_features: usize, num_experts: usize) -> Self {
        // Initialize weights and biases (e.g., randomly or with zeros/ones for placeholder)
        // For simplicity, using zeros for now. Real init would use a distribution.
        let weights = Array2::zeros((num_input_features, num_experts));
        let biases = Array1::zeros(num_experts);
        Self { weights, biases }
    }

    // Input x: Assumed to be 2D [batch_size, num_input_features] or 1D [num_input_features]
    // For now, let's assume input `x` is 1D [num_input_features] for simplicity,
    // representing the features for a single item to be routed.
    // Output: 1D Array [num_experts] with softmax scores.
    pub fn forward(&self, input_features: &Array1<f32>) -> Result<Array1<f32>, String> {
        if input_features.len() != self.weights.shape()[0] {
            return Err(format!(
                "Input feature dimension {} does not match GatingLayer weight input dimension {}",
                input_features.len(),
                self.weights.shape()[0]
            ));
        }

        // Linear transformation: scores = input_features * W + b
        let scores = input_features.dot(&self.weights) + &self.biases;

        // Apply softmax to get probabilities/weights
        Ok(softmax(&scores))
    }

    // Optional: A forward method that takes ArrayD for more general cases,
    // but this often requires knowledge of which axis represents features.
    // For MoE, input to gating is typically the hidden state of a token.
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1; // For creating Array1 instances easily in tests

    #[test]
    fn test_softmax_basic() {
        let input = arr1(&[1.0, 2.0, 3.0]);
        let output = softmax(&input);
        // Expected values (approx): exp(1-3)/(exp(1-3)+exp(2-3)+exp(3-3)) = e^-2 / (e^-2 + e^-1 + 1) ~= 0.09003
        // exp(2-3)/(...) = e^-1 / (e^-2 + e^-1 + 1) ~= 0.24473
        // exp(3-3)/(...) = 1 / (e^-2 + e^-1 + 1) ~= 0.66524
        assert!((output[0] - 0.09003057).abs() < 1e-5);
        assert!((output[1] - 0.24472847).abs() < 1e-5);
        assert!((output[2] - 0.66524096).abs() < 1e-5);
        assert!((output.sum() - 1.0).abs() < 1e-5); // Sum of softmax should be 1
    }

    #[test]
    fn test_softmax_all_zeros() {
        let input = arr1(&[0.0, 0.0, 0.0]);
        let output = softmax(&input);
        // Should be uniform distribution
        assert!((output[0] - 1.0/3.0).abs() < 1e-5);
        assert!((output[1] - 1.0/3.0).abs() < 1e-5);
        assert!((output[2] - 1.0/3.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_very_negative_inputs() {
        let input = arr1(&[-1000.0, -1001.0, -1002.0]);
        let output = softmax(&input);
        // Max val is -1000. x-max_val = [0, -1, -2].
        // exp_values are [1, e^-1, e^-2]. Sum is approx 1 + 0.3678 + 0.1353 = 1.5031
        // output approx [1/1.5031, 0.3678/1.5031, 0.1353/1.5031]
        // = [0.6652, 0.2447, 0.0900]
        assert!((output[0] - 0.66524096).abs() < 1e-5); // exp(0)/(...)
        assert!((output[1] - 0.24472847).abs() < 1e-5); // exp(-1)/(...)
        assert!((output[2] - 0.09003057).abs() < 1e-5); // exp(-2)/(...)
    }
    
    #[test]
    fn test_softmax_empty_input() {
        let input = arr1(&<[f32; 0]>[]); // Empty Array1<f32>
        let output = softmax(&input);
        assert!(output.is_empty());
    }

    #[test]
    fn test_gating_layer_new() {
        let layer = GatingLayer::new(10, 4); // 10 input features, 4 experts
        assert_eq!(layer.weights.shape(), &[10, 4]);
        assert_eq!(layer.biases.shape(), &[4]);
        // Check if initialized to zeros (as per current simple init)
        assert!(layer.weights.iter().all(|&x| x == 0.0));
        assert!(layer.biases.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_gating_layer_forward_basic() {
        let num_features = 3;
        let num_experts = 2;
        let mut layer = GatingLayer::new(num_features, num_experts);
        
        // Manually set some non-zero weights/biases for a meaningful test
        // W = [[1, 0.5], [0.5, 1], [0, 1]]
        // b = [0.1, -0.1]
        layer.weights = Array2::from_shape_vec((num_features, num_experts), vec![1.0, 0.5, 0.5, 1.0, 0.0, 1.0]).unwrap();
        layer.biases = arr1(&[0.1, -0.1]);

        let input_features = arr1(&[1.0, 1.0, 1.0]); // Example input
        
        // Expected scores:
        // input * W = [1,1,1] * [[1, 0.5], [0.5, 1], [0, 1]] = [1*1+1*0.5+1*0, 1*0.5+1*1+1*1] = [1.5, 2.5]
        // scores + b = [1.5, 2.5] + [0.1, -0.1] = [1.6, 2.4]
        // softmax([1.6, 2.4]): max_val = 2.4
        // x-max_val = [1.6-2.4, 2.4-2.4] = [-0.8, 0]
        // exp_values = [exp(-0.8), exp(0)] = [0.4493, 1.0]
        // sum_exp = 1.4493
        // output = [0.4493/1.4493, 1.0/1.4493] = [0.3100, 0.6900] (approx)

        let result = layer.forward(&input_features);
        assert!(result.is_ok());
        let expert_weights = result.unwrap();
        
        assert_eq!(expert_weights.len(), num_experts);
        assert!((expert_weights.sum() - 1.0).abs() < 1e-5); // Sum to 1

        assert!((expert_weights[0] - 0.3100255).abs() < 1e-5);
        assert!((expert_weights[1] - 0.6899745).abs() < 1e-5);
    }

    #[test]
    fn test_gating_layer_forward_dim_mismatch() {
        let layer = GatingLayer::new(5, 3);
        let input_features_wrong_dim = arr1(&[1.0, 2.0, 3.0]); // Only 3 features, expected 5
        let result = layer.forward(&input_features_wrong_dim);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.contains("Input feature dimension 3 does not match GatingLayer weight input dimension 5"));
        }
    }
}
