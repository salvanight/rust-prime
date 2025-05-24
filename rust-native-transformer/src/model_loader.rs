// src/model_loader.rs

use crate::tensor_engine::Tensor;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

// 1. Define Data Structures
#[derive(Deserialize, Debug)]
struct TensorMetadata {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (usize, usize),
}

// Type alias for the header, which is a map from tensor names to their metadata
type SafeTensorHeader = HashMap<String, TensorMetadata>;

// 3. Error Handling
#[derive(Debug)]
pub enum ModelLoaderError {
    IoError(io::Error),
    JsonError(serde_json::Error),
    UnsupportedDtype(String),
    DataCorruption(String),
    HeaderTooShort,
    InvalidHeaderLength,
    TensorNotFound(String), // Though not strictly used if we iterate through header
}

impl From<io::Error> for ModelLoaderError {
    fn from(err: io::Error) -> ModelLoaderError {
        ModelLoaderError::IoError(err)
    }
}

impl From<serde_json::Error> for ModelLoaderError {
    fn from(err: serde_json::Error) -> ModelLoaderError {
        ModelLoaderError::JsonError(err)
    }
}

// 2. Implement .safetensors Parser
pub fn load_safetensors(file_path: &str) -> Result<HashMap<String, Tensor<f32>>, ModelLoaderError> {
    let path = Path::new(file_path);
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // 1. Read the header length (u64)
    let mut header_len_bytes = [0u8; 8];
    reader.read_exact(&mut header_len_bytes)?;
    let header_length = u64::from_le_bytes(header_len_bytes) as usize;

    if header_length == 0 {
        return Err(ModelLoaderError::InvalidHeaderLength);
    }

    // 2. Read N bytes of JSON header
    let mut json_header_bytes = vec![0u8; header_length];
    reader.read_exact(&mut json_header_bytes)?;
    let header: SafeTensorHeader = serde_json::from_slice(&json_header_bytes)?;

    let mut tensors_map: HashMap<String, Tensor<f32>> = HashMap::new();
    let header_and_size_len = 8 + header_length;

    // 3. For each tensor in the header
    for (tensor_name, metadata) in header {
        if metadata.dtype != "F32" {
            // For this project, we only support F32.
            // In a more general library, you might skip or handle other types.
            return Err(ModelLoaderError::UnsupportedDtype(format!(
                "Unsupported dtype '{}' for tensor '{}'. Only F32 is supported.",
                metadata.dtype, tensor_name
            )));
        }

        let expected_elements: usize = metadata.shape.iter().product();
        let expected_bytes = expected_elements * std::mem::size_of::<f32>();

        let data_start_offset = metadata.data_offsets.0;
        let data_end_offset = metadata.data_offsets.1;
        let tensor_data_len = data_end_offset - data_start_offset;

        if tensor_data_len != expected_bytes {
            return Err(ModelLoaderError::DataCorruption(format!(
                "Tensor '{}': expected {} bytes based on shape {:?} and dtype {}, but metadata indicates {} bytes.",
                tensor_name, expected_bytes, metadata.shape, metadata.dtype, tensor_data_len
            )));
        }

        // Seek to the start of this tensor's data in the file
        // Offsets in metadata are relative to the end of the JSON header
        reader.seek(SeekFrom::Start((header_and_size_len + data_start_offset) as u64))?;

        let mut tensor_bytes = vec![0u8; tensor_data_len];
        reader.read_exact(&mut tensor_bytes)?;

        // Convert bytes to Vec<f32>
        let mut tensor_f32_data = Vec::with_capacity(expected_elements);
        for chunk in tensor_bytes.chunks_exact(std::mem::size_of::<f32>()) {
            tensor_f32_data.push(f32::from_le_bytes(chunk.try_into().unwrap())); // unwrap is safe due to chunks_exact
        }

        let tensor = Tensor::new(tensor_f32_data, metadata.shape.clone())
            .map_err(|e| ModelLoaderError::DataCorruption(format!("Failed to create tensor '{}': {:?}", tensor_name, e)))?;
        
        tensors_map.insert(tensor_name, tensor);
    }

    Ok(tensors_map)
}


// 4. Unit Tests
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    use crate::tensor_engine::TensorError; // For checking tensor creation errors

    // Helper to create a dummy .safetensors file in memory (as Vec<u8>)
    fn create_dummy_safetensors_bytes(
        json_str: &str,
        tensor_data_f32: &[f32],
    ) -> Vec<u8> {
        let json_bytes = json_str.as_bytes();
        let header_len = json_bytes.len() as u64;

        let mut file_bytes = Vec::new();
        file_bytes.write_all(&header_len.to_le_bytes()).unwrap(); // Header length (u64)
        file_bytes.write_all(json_bytes).unwrap(); // JSON header

        // Tensor data (f32s as little-endian bytes)
        for &val in tensor_data_f32 {
            file_bytes.write_all(&val.to_le_bytes()).unwrap();
        }
        file_bytes
    }

    #[test]
    fn test_load_simple_safetensors() {
        let tensor_name = "test_tensor";
        let shape = vec![2, 2];
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let data_bytes_len = data.len() * std::mem::size_of::<f32>(); // 4 * 4 = 16

        // JSON header for one tensor
        let json_header = format!(
            r#"{{"{}": {{"dtype": "F32", "shape": [{}, {}], "data_offsets": [0, {}]}}}}"#,
            tensor_name, shape[0], shape[1], data_bytes_len
        );

        let file_bytes = create_dummy_safetensors_bytes(&json_header, &data);

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&file_bytes).unwrap();
        let temp_file_path = temp_file.path().to_str().unwrap().to_string();

        let result = load_safetensors(&temp_file_path);
        assert!(result.is_ok(), "Failed to load safetensors: {:?}", result.err());

        let tensors_map = result.unwrap();
        assert!(tensors_map.contains_key(tensor_name));

        let loaded_tensor = tensors_map.get(tensor_name).unwrap();
        assert_eq!(loaded_tensor.shape, shape);
        assert_eq!(loaded_tensor.data, data);
    }

    #[test]
    fn test_load_two_tensors() {
        let tensor1_name = "tensor_a";
        let tensor1_shape = vec![1, 2];
        let tensor1_data = vec![1.1f32, 2.2];
        let tensor1_bytes_len = tensor1_data.len() * std::mem::size_of::<f32>(); // 2 * 4 = 8

        let tensor2_name = "tensor_b";
        let tensor2_shape = vec![3];
        let tensor2_data = vec![3.3f32, 4.4, 5.5];
        let tensor2_bytes_len = tensor2_data.len() * std::mem::size_of::<f32>(); // 3 * 4 = 12

        // data_offsets are relative to the start of the binary data block
        // tensor1: [0, 8)
        // tensor2: [8, 8+12) = [8, 20)
        let json_header = format!(
            r#"{{
                "{}": {{"dtype": "F32", "shape": [{}, {}], "data_offsets": [0, {}]}},
                "{}": {{"dtype": "F32", "shape": [{}], "data_offsets": [{}, {}]}}
            }}"#,
            tensor1_name, tensor1_shape[0], tensor1_shape[1], tensor1_bytes_len,
            tensor2_name, tensor2_shape[0], tensor1_bytes_len, tensor1_bytes_len + tensor2_bytes_len
        );
        
        let mut combined_data = tensor1_data.clone();
        combined_data.extend_from_slice(&tensor2_data);

        let file_bytes = create_dummy_safetensors_bytes(&json_header, &combined_data);
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&file_bytes).unwrap();
        let temp_file_path = temp_file.path().to_str().unwrap().to_string();

        let result = load_safetensors(&temp_file_path);
        assert!(result.is_ok(), "Failed to load safetensors: {:?}", result.err());
        let tensors_map = result.unwrap();

        assert_eq!(tensors_map.len(), 2);

        let loaded_tensor1 = tensors_map.get(tensor1_name).unwrap();
        assert_eq!(loaded_tensor1.shape, tensor1_shape);
        assert_eq!(loaded_tensor1.data, tensor1_data);

        let loaded_tensor2 = tensors_map.get(tensor2_name).unwrap();
        assert_eq!(loaded_tensor2.shape, tensor2_shape);
        assert_eq!(loaded_tensor2.data, tensor2_data);
    }

    #[test]
    fn test_file_not_found() {
        let result = load_safetensors("non_existent_file.safetensors");
        assert!(matches!(result, Err(ModelLoaderError::IoError(_))));
    }

    #[test]
    fn test_corrupted_header_length_too_short() {
        let file_bytes = vec![1, 0, 0, 0, 0, 0, 0]; // Only 7 bytes, expected 8 for u64
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&file_bytes).unwrap();
        let result = load_safetensors(temp_file.path().to_str().unwrap());
        // This will likely result in an IoError(Kind::UnexpectedEof) when reading header_len_bytes
        assert!(matches!(result, Err(ModelLoaderError::IoError(e)) if e.kind() == io::ErrorKind::UnexpectedEof));
    }
    
    #[test]
    fn test_zero_header_length() {
        let file_bytes = (0u64).to_le_bytes().to_vec(); // Header length is 0
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&file_bytes).unwrap();
        let result = load_safetensors(temp_file.path().to_str().unwrap());
        assert!(matches!(result, Err(ModelLoaderError::InvalidHeaderLength)));
    }


    #[test]
    fn test_malformed_json() {
        let json_header = r#"{"test_tensor": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]]}"#; // Extra ']'
        let file_bytes = create_dummy_safetensors_bytes(json_header, &[]); // No data needed as JSON parsing fails first
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&file_bytes).unwrap();
        let result = load_safetensors(temp_file.path().to_str().unwrap());
        assert!(matches!(result, Err(ModelLoaderError::JsonError(_))));
    }

    #[test]
    fn test_unsupported_dtype() {
        let json_header = r#"{"test_tensor": {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]}}"#;
        let dummy_f16_data_as_f32 = vec![0.0f32, 0.0]; // 2*4=8 bytes, pretending it's F16 data
        let file_bytes = create_dummy_safetensors_bytes(json_header, &dummy_f16_data_as_f32);
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&file_bytes).unwrap();
        let result = load_safetensors(temp_file.path().to_str().unwrap());
        assert!(matches!(result, Err(ModelLoaderError::UnsupportedDtype(s)) if s.contains("F16")));
    }

    #[test]
    fn test_data_size_mismatch_metadata_vs_shape() {
        // Shape [2,2] means 4 * f32 = 16 bytes. Metadata says data_offsets are [0, 10] (10 bytes).
        let json_header = r#"{"test_tensor": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 10]}}"#;
        let dummy_data = vec![1.0f32, 2.0, 3.0]; // Some data, length doesn't matter as check is on metadata
        let file_bytes = create_dummy_safetensors_bytes(json_header, &dummy_data);
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&file_bytes).unwrap();
        let result = load_safetensors(temp_file.path().to_str().unwrap());
        assert!(matches!(result, Err(ModelLoaderError::DataCorruption(s)) if s.contains("expected 16 bytes") && s.contains("indicates 10 bytes")));
    }
    
    #[test]
    fn test_data_corruption_not_enough_bytes_in_file_for_tensor() {
        let tensor_name = "test_tensor";
        let shape = vec![2, 2]; // Requires 16 bytes
        let data_bytes_len = 16;
        let json_header = format!(
            r#"{{"{}": {{"dtype": "F32", "shape": [{}, {}], "data_offsets": [0, {}]}}}}"#,
            tensor_name, shape[0], shape[1], data_bytes_len
        );

        let mut file_bytes = create_dummy_safetensors_bytes(&json_header, &[]); // Empty actual tensor data
        // Actual tensor data part is missing/truncated. create_dummy_safetensors_bytes adds header + json.
        // We need to simulate a file where tensor data is expected but shorter than specified by data_offsets.
        
        let json_only_bytes = json_header.as_bytes();
        let header_len_u64 = json_only_bytes.len() as u64;
        let mut file_bytes_manual = Vec::new();
        file_bytes_manual.write_all(&header_len_u64.to_le_bytes()).unwrap();
        file_bytes_manual.write_all(json_only_bytes).unwrap();
        file_bytes_manual.write_all(&[1.0f32.to_le_bytes(), 2.0f32.to_le_bytes()].concat()).unwrap(); // Only 8 bytes of data

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&file_bytes_manual).unwrap();
        let temp_file_path = temp_file.path().to_str().unwrap().to_string();

        let result = load_safetensors(&temp_file_path);
        // This should be an IoError(UnexpectedEof) when trying to read_exact for the tensor data.
        assert!(matches!(result, Err(ModelLoaderError::IoError(e)) if e.kind() == io::ErrorKind::UnexpectedEof));
    }
}
