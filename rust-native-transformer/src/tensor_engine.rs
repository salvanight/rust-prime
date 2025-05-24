// src/tensor_engine.rs

use std::fmt::Debug;

// 4. Error Handling
#[derive(Debug, PartialEq)]
pub enum TensorError {
    ShapeMismatch(String),
    InvalidDimension(String),
    OutOfBounds(String),
    UnsupportedAxis(String),
    IncompatibleShapes(String), // For operations like matmul
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::ShapeMismatch(s) => write!(f, "Shape mismatch: {}", s),
            TensorError::InvalidDimension(s) => write!(f, "Invalid dimension: {}", s),
            TensorError::OutOfBounds(s) => write!(f, "Out of bounds: {}", s),
            TensorError::UnsupportedAxis(s) => write!(f, "Unsupported axis: {}", s),
            TensorError::IncompatibleShapes(s) => write!(f, "Incompatible shapes: {}", s),
        }
    }
}

impl std::error::Error for TensorError {} // Simple implementation, no source needed for these variants

// 1. Define Tensor<T> Struct
#[derive(Debug, Clone)]
pub struct Tensor<T> {
    pub data: Vec<T>,
    pub shape: Vec<usize>,
}

// 2. Implement Basic Tensor Creation and Manipulation
impl<T> Tensor<T> {
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Result<Self, TensorError> {
        let num_elements_shape: usize = shape.iter().product();
        if data.len() != num_elements_shape {
            return Err(TensorError::ShapeMismatch(format!(
                "Data length {} does not match shape product {}",
                data.len(),
                num_elements_shape
            )));
        }
        // The check data.len() != num_elements_shape (where num_elements_shape is 1 for shape=[])
        // correctly handles:
        // - Tensor::new(vec![scalar], vec![]) -> Ok
        // - Tensor::new(vec![val1, val2], vec![]) -> ShapeMismatch (data.len=2 != num_elements_shape=1)
        // - Tensor::new(vec![], vec![]) -> ShapeMismatch (data.len=0 != num_elements_shape=1)
        // The specific block causing InvalidDimension for scalars has been removed.
        Ok(Tensor { data, shape })
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    // Helper for flat indexing
    fn _flat_index(&self, indices: &[usize]) -> Result<usize, TensorError> {
        if indices.len() != self.rank() {
            return Err(TensorError::InvalidDimension(format!(
                "Expected {} indices, got {}",
                self.rank(),
                indices.len()
            )));
        }

        let mut flat_idx = 0;
        let mut multiplier = 1;
        for (i, &dim_idx) in indices.iter().rev().enumerate() {
            let dim_size = self.shape[self.rank() - 1 - i];
            if dim_idx >= dim_size {
                return Err(TensorError::OutOfBounds(format!(
                    "Index {} out of bounds for dimension {} with size {}",
                    dim_idx,
                    self.rank() - 1 - i,
                    dim_size
                )));
            }
            flat_idx += dim_idx * multiplier;
            multiplier *= dim_size;
        }
        Ok(flat_idx)
    }

    pub fn get(&self, indices: &[usize]) -> Result<&T, TensorError> {
        if self.shape.is_empty() && indices.is_empty() && self.data.is_empty() {
             return Err(TensorError::OutOfBounds("Cannot get from empty tensor with empty shape and no data".to_string()));
        }
        if self.shape.is_empty() && indices.is_empty() && self.data.len() == 1 { // Scalar case, shape []
             return Ok(&self.data[0]);
        }
        let flat_idx = self._flat_index(indices)?;
        self.data.get(flat_idx).ok_or_else(|| TensorError::OutOfBounds("Calculated flat index out of bounds".to_string()))
    }

    pub fn get_mut(&mut self, indices: &[usize]) -> Result<&mut T, TensorError> {
        if self.shape.is_empty() && indices.is_empty() && self.data.len() == 1 { // Scalar case, shape []
             return Ok(&mut self.data[0]);
        }
        let flat_idx = self._flat_index(indices)?;
        self.data.get_mut(flat_idx).ok_or_else(|| TensorError::OutOfBounds("Calculated flat index out of bounds".to_string()))
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, TensorError>
    where T: Clone 
    {
        let new_num_elements: usize = new_shape.iter().product();
        if self.num_elements() != new_num_elements {
            return Err(TensorError::ShapeMismatch(format!(
                "Cannot reshape tensor with {} elements into shape {:?} ({} elements)",
                self.num_elements(),
                new_shape,
                new_num_elements
            )));
        }
        Ok(Tensor {
            data: self.data.clone(), // Data is cloned
            shape: new_shape,
        })
    }
}

impl<T: Default + Clone> Tensor<T> {
    pub fn zeros(shape: Vec<usize>) -> Self {
        let num_elements = shape.iter().product();
        Tensor {
            data: vec![T::default(); num_elements],
            shape,
        }
    }
}

// 3. Implement Mathematical Operations (for f32)
impl Tensor<f32> {
    pub fn matmul(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        if a.rank() != 2 || b.rank() != 2 {
            return Err(TensorError::InvalidDimension(
                "Matmul currently only supports 2D tensors".to_string(),
            ));
        }
        let m = a.shape[0];
        let k_a = a.shape[1];
        let k_b = b.shape[0];
        let n = b.shape[1];

        if k_a != k_b {
            return Err(TensorError::IncompatibleShapes(format!(
                "Incompatible shapes for matmul: A has shape [{}, {}], B has shape [{}, {}]",
                m, k_a, k_b, n
            )));
        }

        let mut result_data = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for k_idx in 0..k_a {
                    sum += a.get(&[i, k_idx]).unwrap() * b.get(&[k_idx, j]).unwrap();
                }
                result_data[i * n + j] = sum;
            }
        }
        Tensor::new(result_data, vec![m, n])
    }

    pub fn softmax(&self, axis: usize) -> Result<Tensor<f32>, TensorError> {
        if axis >= self.rank() {
            return Err(TensorError::UnsupportedAxis(format!(
                "Axis {} is out of bounds for tensor with rank {}",
                axis, self.rank()
            )));
        }

        let mut result_data = self.data.clone();
        let axis_size = self.shape[axis];
        let outer_dims_product: usize = self.shape[..axis].iter().product();
        let inner_dims_product: usize = self.shape[axis + 1..].iter().product();

        for i in 0..outer_dims_product {
            for j in 0..inner_dims_product {
                // Extract the slice along the axis
                let mut current_slice = Vec::with_capacity(axis_size);
                for k in 0..axis_size {
                    // Calculate multi-dimensional index for the current element
                    // let mut temp_i = i;
                    // for l in 0..axis { // Indices for outer dimensions
                    //     indices.push(temp_i % self.shape[l]);
                    //     temp_i /= self.shape[l];
                    // }
                    // indices.push(k); // Index for the current axis
                    // let mut temp_j = j;
                    // for l in (axis + 1)..self.rank() { // Indices for inner dimensions
                    //     indices.push(temp_j % self.shape[l]);
                    //     temp_j /= self.shape[l];
                    // }
                    // This indexing logic is incorrect if not careful.
                    // A simpler way is to calculate flat_start_index and stride.
                    // let _flat_idx_start = i * axis_size * inner_dims_product + j; // unused
                    // let _stride = inner_dims_product; // This is the stride for the axis dimension if axis is not the last. // unused
                                                    // This requires careful recalculation of flat_idx
                    let current_flat_idx = self._flat_index_for_softmax(i, k, j, axis, inner_dims_product).unwrap();
                    current_slice.push(self.data[current_flat_idx]);
                }
                
                // 1. Find max for numerical stability
                let max_val = current_slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                
                // 2. Subtract max and exponentiate
                let mut exp_values = Vec::with_capacity(axis_size);
                for &val in &current_slice {
                    exp_values.push((val - max_val).exp());
                }
                
                // 3. Sum exponentiated values
                let sum_exp_values: f32 = exp_values.iter().sum();
                
                // 4. Divide by sum and update result_data
                for k in 0..axis_size {
                     let current_flat_idx = self._flat_index_for_softmax(i, k, j, axis, inner_dims_product).unwrap();
                    result_data[current_flat_idx] = exp_values[k] / sum_exp_values;
                }
            }
        }
        Tensor::new(result_data, self.shape.clone())
    }
    
    // Helper for softmax indexing, needs to be correct for arbitrary axis
    fn _flat_index_for_softmax(&self, outer_idx: usize, axis_idx: usize, inner_idx: usize, axis: usize, _inner_dims_product: usize) -> Result<usize, TensorError> {
        // Reconstruct the full multi-dimensional index
        let mut md_indices = vec![0; self.rank()];
        let mut current_outer = outer_idx;
        for d in (0..axis).rev() {
            md_indices[d] = current_outer % self.shape[d];
            current_outer /= self.shape[d];
        }
        md_indices[axis] = axis_idx;
        let mut current_inner = inner_idx;
        for d in ((axis + 1)..self.rank()).rev() {
            md_indices[d] = current_inner % self.shape[d];
            current_inner /= self.shape[d];
        }
        self._flat_index(&md_indices)
    }


    pub fn layernorm(&self, gamma: &Tensor<f32>, beta: &Tensor<f32>, epsilon: f32) -> Result<Tensor<f32>, TensorError> {
        if self.rank() == 0 {
            return Err(TensorError::InvalidDimension("LayerNorm not supported for scalar tensors".to_string()));
        }
        let last_dim_size = *self.shape.last().unwrap();
        if gamma.rank() != 1 || gamma.shape[0] != last_dim_size {
            return Err(TensorError::IncompatibleShapes(format!(
                "Gamma shape {:?} incompatible with input's last dimension {}",
                gamma.shape, last_dim_size
            )));
        }
        if beta.rank() != 1 || beta.shape[0] != last_dim_size {
            return Err(TensorError::IncompatibleShapes(format!(
                "Beta shape {:?} incompatible with input's last dimension {}",
                beta.shape, last_dim_size
            )));
        }

        let mut result_data = vec![0.0; self.data.len()];
        let num_vectors = self.data.len() / last_dim_size;

        for i in 0..num_vectors {
            let start = i * last_dim_size;
            let end = start + last_dim_size;
            let current_slice = &self.data[start..end];

            // 1. Calculate mean
            let mean: f32 = current_slice.iter().sum::<f32>() / (last_dim_size as f32);

            // 2. Calculate variance
            let variance: f32 = current_slice.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / (last_dim_size as f32);

            // 3. Normalize, scale, and shift
            for j in 0..last_dim_size {
                let normalized_x = (current_slice[j] - mean) / (variance + epsilon).sqrt();
                result_data[start + j] = normalized_x * gamma.data[j] + beta.data[j];
            }
        }
        Tensor::new(result_data, self.shape.clone())
    }

    pub fn gelu(&self) -> Result<Tensor<f32>, TensorError> {
        let mut result_data = Vec::with_capacity(self.data.len());
        for &x_val in &self.data {
            let x = x_val as f64; // Use f64 for intermediate calculations for precision if needed, though f32 is likely fine
            let result = 0.5 * x * (1.0 + (x / std::f64::consts::SQRT_2).tanh());
            result_data.push(result as f32);
        }
        Tensor::new(result_data, self.shape.clone())
    }
}


// 5. Unit Tests
#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::SQRT_2; // For GELU test comparison

    fn assert_f32_slice_eq(a: &[f32], b: &[f32], tolerance: f32) {
        assert_eq!(a.len(), b.len(), "Slice lengths differ");
        for (i, (val_a, val_b)) in a.iter().zip(b.iter()).enumerate() {
            assert!((val_a - val_b).abs() < tolerance, "Mismatch at index {}: {} vs {}", i, val_a, val_b);
        }
    }

    #[test]
    fn test_tensor_new() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(t.shape, vec![2, 2]);
        assert_eq!(t.rank(), 2);
        assert_eq!(t.num_elements(), 4);
    }

    #[test]
    fn test_tensor_new_shape_mismatch() {
        let result = Tensor::new(vec![1.0, 2.0, 3.0], vec![2, 2]);
        assert_eq!(result.err(), Some(TensorError::ShapeMismatch("Data length 3 does not match shape product 4".to_string())));
    }
    
    #[test]
    fn test_tensor_new_empty_shape_non_empty_data() {
        // Test case: shape is empty, but data has more than 1 element (which is what num_elements_shape would be for scalar)
        // This should be a ShapeMismatch.
        let result = Tensor::new(vec![1.0, 2.0], vec![]);
         assert_eq!(result.err(), Some(TensorError::ShapeMismatch(
            "Data length 2 does not match shape product 1".to_string()
        )));
    }

    #[test]
    fn test_tensor_new_scalar_empty_shape() {
        // If shape is [], num_elements is 1. data.len() should be 1.
        let t = Tensor::new(vec![5.0], vec![]).unwrap(); 
        assert_eq!(t.data, vec![5.0]);
        assert_eq!(t.shape, Vec::<usize>::new());
        assert_eq!(t.rank(), 0);
        assert_eq!(t.num_elements(), 1); // Product of empty shape is 1
    }


    #[test]
    fn test_tensor_zeros() {
        let t: Tensor<f32> = Tensor::zeros(vec![2, 3]);
        assert_eq!(t.data, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(t.shape, vec![2, 3]);
    }

    #[test]
    fn test_tensor_get() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        assert_eq!(t.get(&[0, 0]), Ok(&1.0));
        assert_eq!(t.get(&[0, 1]), Ok(&2.0));
        assert_eq!(t.get(&[1, 0]), Ok(&3.0));
        assert_eq!(t.get(&[1, 1]), Ok(&4.0));
    }
    
    #[test]
    fn test_tensor_get_scalar() {
        let t = Tensor::new(vec![42.0], vec![]).unwrap();
        assert_eq!(t.get(&[]), Ok(&42.0));
    }


    #[test]
    fn test_tensor_get_out_of_bounds() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        assert!(t.get(&[2, 0]).is_err());
        assert!(t.get(&[0, 2]).is_err());
    }
    
    #[test]
    fn test_tensor_get_wrong_rank() {
        let t = Tensor::new(vec![1.0,2.0], vec![2]).unwrap();
        assert!(t.get(&[0,0]).is_err()); // too many indices
        assert!(t.get(&[]).is_err()); // too few indices
    }

    #[test]
    fn test_tensor_get_mut() {
        let mut t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        *t.get_mut(&[0, 1]).unwrap() = 5.0;
        assert_eq!(t.get(&[0, 1]), Ok(&5.0));
    }

    #[test]
    fn test_tensor_reshape() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let t_reshaped = t.reshape(vec![3, 2]).unwrap();
        assert_eq!(t_reshaped.shape, vec![3, 2]);
        assert_eq!(t_reshaped.data, t.data); // Data is the same

        let t_reshaped_flat = t.reshape(vec![6]).unwrap();
        assert_eq!(t_reshaped_flat.shape, vec![6]);
    }

    #[test]
    fn test_tensor_reshape_incompatible() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = t.reshape(vec![1, 3]); // 4 elements vs 3 elements
        assert!(result.is_err());
        match result.err().unwrap() {
            TensorError::ShapeMismatch(_) => {} // Expected error
            _ => panic!("Unexpected error type for reshape incompatibility"),
        }
    }

    #[test]
    fn test_matmul_2x2_2x2() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //         = [[19, 22], [43, 50]]
        let result = Tensor::matmul(&a, &b).unwrap();
        assert_eq!(result.shape, vec![2, 2]);
        assert_eq!(result.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_2x3_3x2() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]).unwrap();
        // Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        //         = [[7+18+33, 8+20+36], [28+45+66, 32+50+72]]
        //         = [[58, 64], [139, 154]]
        let result = Tensor::matmul(&a, &b).unwrap();
        assert_eq!(result.shape, vec![2, 2]);
        assert_eq!(result.data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_matmul_incompatible_shapes() {
        let a = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap(); // Should be 2xN
        let result = Tensor::matmul(&a, &b);
        assert!(result.is_err());
        match result.err().unwrap() {
            TensorError::IncompatibleShapes(_) => {} // Expected
            _ => panic!("Unexpected error type"),
        }
    }
    
    #[test]
    fn test_matmul_non_2d() {
        let a = Tensor::new(vec![1.0,2.0], vec![2]).unwrap();
        let b = Tensor::new(vec![1.0,2.0], vec![2]).unwrap();
        let result = Tensor::matmul(&a, &b);
        assert!(matches!(result, Err(TensorError::InvalidDimension(_))));
    }

    #[test]
    fn test_softmax_simple_vector() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let result = t.softmax(0).unwrap(); // Axis 0 for a 1D tensor
        // Manual calculation:
        // max = 3.0
        // exps = [exp(1-3), exp(2-3), exp(3-3)] = [exp(-2), exp(-1), exp(0)]
        //      = [0.13533528, 0.36787944, 1.0]
        // sum_exps = 1.5032147
        // softmax = [0.09003057, 0.24472847, 0.66524096]
        let expected = vec![0.09003057, 0.24472847, 0.66524096];
        assert_eq!(result.shape, vec![3]);
        assert_f32_slice_eq(&result.data, &expected, 1e-6);
        assert!((result.data.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_on_matrix_axis_1() { // Softmax over columns for each row
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0], vec![2, 3]).unwrap();
        let result = t.softmax(1).unwrap(); // Axis 1 (columns)
        
        // Row 0: [1.0, 2.0, 3.0] -> [0.09003057, 0.24472847, 0.66524096] (from previous test)
        // Row 1: [1.0, 1.0, 1.0]
        // max = 1.0
        // exps = [exp(0), exp(0), exp(0)] = [1.0, 1.0, 1.0]
        // sum_exps = 3.0
        // softmax = [0.33333333, 0.33333333, 0.33333333]
        let expected = vec![
            0.09003057, 0.24472847, 0.66524096,
            0.33333333, 0.33333333, 0.33333333,
        ];
        assert_eq!(result.shape, vec![2, 3]);
        assert_f32_slice_eq(&result.data, &expected, 1e-6);
        assert!((result.data[0..3].iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!((result.data[3..6].iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_on_matrix_axis_0() { // Softmax over rows for each column
        let t = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        // Transposed view for easier manual calculation:
        // Col 0: [1.0, 2.0] -> max=2, exps=[exp(-1), exp(0)]=[0.3678, 1.0], sum=1.3678, sm=[0.2689, 0.7311]
        // Col 0: [1,5] -> [0.01798621, 0.9820138]
        // Col 1: [4,3] -> [0.7310586, 0.26894143]
        // Col 2: [2,6] -> [0.01798621, 0.9820138]
        let result = t.softmax(0).unwrap();
        let expected = vec![
            0.01798621, 0.7310586, 0.01798621, // Row 0
            0.9820138,  0.26894143, 0.9820138   // Row 1
        ];
         assert_eq!(result.shape, vec![2, 3]);
         assert_f32_slice_eq(&result.data, &expected, 1e-6);
    }


    #[test]
    fn test_layernorm_simple() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let gamma = Tensor::new(vec![1.0, 1.0, 1.0], vec![3]).unwrap(); // No scaling
        let beta = Tensor::new(vec![0.0, 0.0, 0.0], vec![3]).unwrap();  // No shift
        let epsilon = 1e-5;

        // Row 0: [1.0, 2.0, 3.0]
        // mean = (1+2+3)/3 = 2.0
        // variance = ((1-2)^2 + (2-2)^2 + (3-2)^2)/3 = (1+0+1)/3 = 2/3 = 0.6666667
        // std_dev = sqrt(variance + epsilon) = sqrt(0.6666667 + 1e-5) = sqrt(0.6666767) = 0.8165027
        // norm_x = [(1-2)/0.8165027, (2-2)/0.8165027, (3-2)/0.8165027]
        //        = [-1.22474, 0.0, 1.22474]
        // Output (row 0) = norm_x * 1.0 + 0.0 -> [-1.22474, 0.0, 1.22474]

        // Row 1: [4.0, 5.0, 6.0]
        // mean = (4+5+6)/3 = 5.0
        // variance = ((4-5)^2 + (5-5)^2 + (6-5)^2)/3 = (1+0+1)/3 = 2/3 = 0.6666667
        // std_dev = 0.8165027 (same as above)
        // norm_x = [(4-5)/0.8165027, (5-5)/0.8165027, (6-5)/0.8165027]
        //        = [-1.22474, 0.0, 1.22474]
        // Output (row 1) = norm_x * 1.0 + 0.0 -> [-1.22474, 0.0, 1.22474]
        
        let result = input.layernorm(&gamma, &beta, epsilon).unwrap();
        let expected_data = vec![
            -1.2247448, 0.0, 1.2247448,
            -1.2247448, 0.0, 1.2247448,
        ];
        assert_eq!(result.shape, input.shape);
        assert_f32_slice_eq(&result.data, &expected_data, 1e-5);
    }
    
    #[test]
    fn test_layernorm_with_gamma_beta() {
        let input_data = vec![0.0, 1.0, 2.0];
        let input_shape = vec![1,3];
        let input = Tensor::new(input_data.clone(), input_shape.clone()).unwrap();
        let gamma_data = vec![1.5, 0.5, 2.0];
        let gamma = Tensor::new(gamma_data.clone(), vec![3]).unwrap();
        let beta_data = vec![0.1, 0.2, 0.3];
        let beta = Tensor::new(beta_data.clone(), vec![3]).unwrap();
        let epsilon = 1e-5_f32;

        // Input: [0.0, 1.0, 2.0]
        // Mean: (0.0 + 1.0 + 2.0) / 3.0 = 1.0
        // Variance: ((0-1)^2 + (1-1)^2 + (2-1)^2) / 3.0 = (1 + 0 + 1) / 3.0 = 2.0/3.0
        // StdDev: sqrt(2.0/3.0 + epsilon) = sqrt(0.6666666 + 1e-5) = sqrt(0.6666766) = 0.8165027
        // Normalized:
        // (0.0 - 1.0) / 0.8165027 = -1.2247448
        // (1.0 - 1.0) / 0.8165027 = 0.0
        // (2.0 - 1.0) / 0.8165027 = 1.2247448
        // Scaled and Shifted:
        // -1.2247448 * 1.5 + 0.1 = -1.8371172 + 0.1 = -1.7371172
        //  0.0 * 0.5 + 0.2 = 0.0 + 0.2 = 0.2
        //  1.2247448 * 2.0 + 0.3 = 2.4494896 + 0.3 = 2.7494896

        let expected_output = vec![-1.7371172, 0.2, 2.7494896];
        let result = input.layernorm(&gamma, &beta, epsilon).unwrap();
        assert_f32_slice_eq(&result.data, &expected_output, 2e-5); // Adjusted tolerance
    }


    #[test]
    fn test_gelu() {
        let input = Tensor::new(vec![0.0, 1.0, -1.0, 2.0, -2.0], vec![5]).unwrap();
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) - This is GPT-2's GELU approx
        // The problem description gives: 0.5 * x * (1.0 + (x / sqrt(2.0)).tanh())
        // Let's use the one from the problem description.
        // x = 0: 0.5 * 0 * (1 + tanh(0)) = 0
        // x = 1: 0.5 * 1 * (1 + tanh(1/sqrt(2))) = 0.5 * (1 + tanh(0.7071)) = 0.5 * (1 + 0.6095) = 0.5 * 1.6095 = 0.80475
        // x = -1: 0.5 * -1 * (1 + tanh(-1/sqrt(2))) = -0.5 * (1 - 0.6095) = -0.5 * 0.3905 = -0.19525
        // x = 2: 0.5 * 2 * (1 + tanh(2/sqrt(2))) = 1 * (1 + tanh(sqrt(2))) = 1 * (1 + tanh(1.4142)) = 1 * (1 + 0.8884) = 1.8884
        // x = -2: 0.5 * -2 * (1 + tanh(-2/sqrt(2))) = -1 * (1 - 0.8884) = -1 * 0.1116 = -0.1116
        
        let expected_data = vec![
            0.0,
            0.5 * 1.0 * (1.0 + (1.0 / SQRT_2).tanh()),
            0.5 * -1.0 * (1.0 + (-1.0 / SQRT_2).tanh()),
            0.5 * 2.0 * (1.0 + (2.0 / SQRT_2).tanh()),
            0.5 * -2.0 * (1.0 + (-2.0 / SQRT_2).tanh()),
        ];
        let result = input.gelu().unwrap();
        assert_eq!(result.shape, input.shape);
        assert_f32_slice_eq(&result.data, &expected_data, 1e-6);
    }
}
