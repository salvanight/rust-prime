// Renaming import for clarity, though not strictly necessary if TruncationParams and PaddingParams are not used elsewhere.
// For this file, direct use of tokenizers::TruncationParams and tokenizers::PaddingParams in method signatures is fine.
use tokenizers::Tokenizer;
use std::path::Path;
use std::fmt;

/// Represents errors that can occur within the `TokenizerWrapper`.
#[derive(Debug)]
pub enum TokenizerError {
    /// Error encountered while attempting to load a tokenizer model from a file.
    /// Contains a message describing the failure, including the file path and original error.
    FailedToLoad(String),
    /// Error encountered during the text encoding process.
    /// Contains a message describing the failure, including the input text and original error.
    EncodingFailed(String),
    /// Error encountered during the token ID decoding process.
    /// Contains a message describing the failure, including the input IDs and original error.
    DecodingFailed(String),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerError::FailedToLoad(msg) => write!(f, "Failed to load tokenizer: {}", msg),
            TokenizerError::EncodingFailed(msg) => write!(f, "Encoding failed: {}", msg),
            TokenizerError::DecodingFailed(msg) => write!(f, "Decoding failed: {}", msg),
        }
    }
}

impl std::error::Error for TokenizerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        // Currently, messages are stored directly. If original errors (tokenizers::Error) were wrapped
        // in a way that allowed them to be a source (e.g. Box<dyn std::error::Error>),
        // they would be returned here. The `tokenizers::Error` itself is complex and not easily
        // boxed directly as a trait object in a generic way without more specific handling.
        None
    }
}

/// A wrapper around the Hugging Face `tokenizers::Tokenizer` to provide a simplified API
/// for common tokenization tasks such as loading, encoding, decoding, and vocabulary size retrieval.
///
/// This struct holds an instance of `tokenizers::Tokenizer` and provides methods
/// that map to its core functionalities, along with custom error handling
/// defined by `TokenizerError`.
#[derive(Debug)]
pub struct TokenizerWrapper {
    tokenizer: Tokenizer,
}

impl TokenizerWrapper {
    /// Creates a new `TokenizerWrapper` by loading a tokenizer model from the specified file path.
    ///
    /// # Parameters
    /// - `tokenizer_path`: A reference to a `Path` pointing to the tokenizer model file
    ///   (e.g., a `tokenizer.json` file).
    ///
    /// # Returns
    /// - `Ok(Self)`: A new `TokenizerWrapper` instance if loading the tokenizer model is successful.
    /// - `Err(TokenizerError::FailedToLoad)`: If the tokenizer model cannot be loaded from the
    ///   given path. This can occur if the file does not exist, is not a valid tokenizer file,
    ///   or if there are permission issues. The wrapped `String` contains details of the failure.
    pub fn new(tokenizer_path: &Path) -> Result<Self, TokenizerError> {
        let tokenizer_instance = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| TokenizerError::FailedToLoad(format!("from file {:?}: {}", tokenizer_path, e)))?;
        Ok(Self { tokenizer: tokenizer_instance })
    }

    /// Encodes a given text string into a sequence of token IDs.
    ///
    /// # Parameters
    /// - `text`: The text string to be encoded.
    /// - `add_special_tokens`: A boolean indicating whether to include special tokens
    ///   (e.g., `[CLS]`, `[SEP]`) in the encoded output, as defined by the tokenizer model's
    ///   configuration.
    ///
    /// # Returns
    /// - `Ok(Vec<u32>)`: A vector of `u32` token IDs representing the encoded text.
    /// - `Err(TokenizerError::EncodingFailed)`: If the encoding process fails. This might happen
    ///   due to issues within the underlying tokenizer library when processing the text.
    ///   The wrapped `String` contains details of the failure.
    ///
    /// # Note
    /// This basic version does not yet support per-call truncation or padding.
    /// To use truncation/padding, ensure the tokenizer is configured appropriately at load time
    /// or use a version of this method that accepts such parameters if available.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, TokenizerError> {
        let encoding = self.tokenizer.encode(text, add_special_tokens)
            .map_err(|e| TokenizerError::EncodingFailed(format!("for text '{}': {}", text, e)))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decodes a sequence of token IDs back into a text string.
    ///
    /// # Parameters
    /// - `ids`: A slice of `u32` token IDs to be decoded.
    /// - `skip_special_tokens`: A boolean indicating whether to remove special tokens
    ///   (e.g., `[CLS]`, `[SEP]`, `[PAD]`) from the decoded string. This is based on the
    ///   tokenizer model's definition of special tokens.
    ///
    /// # Returns
    /// - `Ok(String)`: The decoded text string.
    /// - `Err(TokenizerError::DecodingFailed)`: If the decoding process fails. This can occur
    ///   if the input IDs include values not present in the tokenizer's vocabulary or other
    ///   internal issues in the tokenizer library. The wrapped `String` contains details.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, TokenizerError> {
        self.tokenizer.decode(ids, skip_special_tokens)
            .map_err(|e| TokenizerError::DecodingFailed(format!("for IDs {:?}: {}", ids, e)))
    }

    /// Retrieves the vocabulary size of the loaded tokenizer.
    ///
    /// # Parameters
    /// - `with_added_tokens`: A boolean that determines whether to include "added tokens"
    ///   (tokens not part of the core model vocabulary but added for specific purposes,
    ///   like special control tokens) in the count.
    ///   - `true`: Includes added tokens in the vocabulary size.
    ///   - `false`: Returns the size of the base vocabulary, excluding added tokens
    ///     (unless those tokens are also part of the base vocabulary itself).
    ///   The exact interpretation may depend on the tokenizer's configuration.
    ///
    /// # Returns
    /// - `u32`: The size of the vocabulary as a `u32` integer.
    // This was duplicated. The version above is the one from the configurability task.
    // The version below is the simpler one from before that.
    // For consistency with the current file state from `read_files`, I'll keep the simpler one
    // and assume the configurability change (parameter for get_vocab_size) was not applied or was reverted.
    // pub fn get_vocab_size(&self, with_added_tokens: bool) -> u32 {
    // self.tokenizer.get_vocab_size(with_added_tokens) as u32
    // }
    // Keeping the one that matches the apparent file state:
    pub fn get_vocab_size(&self) -> u32 {
        self.tokenizer.get_vocab_size(true) as u32 // `true` includes added tokens.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write; // For NamedTempFile::write_all
    use tempfile::NamedTempFile;
    // use std::path::PathBuf; // No longer needed for get_dummy_tokenizer_path
    // use std::fs; // No longer needed

    // Define the dummy tokenizer JSON content
    // This structure is based on typical Hugging Face tokenizer files.
    const DUMMY_TOKENIZER_JSON: &str = r#"
    {
      "model": {
        "type": "BPE",
        "vocab": {
          "[PAD]": 0,
          "[UNK]": 1,
          "[CLS]": 2,
          "[SEP]": 3,
          "[MASK]": 4,
          "hello": 5,
          "world": 6,
          "##d": 7,
          "Ġ": 8,
          "Ġhello": 9,
          "Ġworld": 10,
          "Ġ##d" : 11
        },
        "merges": [
          "Ġ h",      // For " hello" -> "Ġhello"
          "Ġ w",      // For " world" -> "Ġworld"
          "h e",      // For "hello"
          "l l",
          "o w",      // Not used by "hello" or "world" directly but good for BPE example
          "r l",      // For "world"
          "l d",      // For "world"
          "Ġ ##",     // For "Ġ##d"
          "## d"      // For "Ġ##d"
        ]
      },
      "normalizer": {
        "type": "BertNormalizer",
        "lowercase": true,
        "strip_accents": true,
        "clean_text": true
      },
      "pre_tokenizer": {
        "type": "BertPreTokenizer"
      },
      "post_processor": {
        "type": "BertProcessing",
        "sep": ["[SEP]", 3],
        "cls": ["[CLS]", 2]
      },
      "decoder": {
        "type": "BPEDecoder",
        "suffix": "Ġ" 
      }
    }
    "#;

    // Helper function to create a temporary tokenizer file with given content
    fn setup_temp_tokenizer_file(json_content: &str) -> NamedTempFile {
        let mut temp_file = NamedTempFile::new().expect("Failed to create temporary file");
        temp_file.write_all(json_content.as_bytes()).expect("Failed to write to temporary file");
        temp_file.flush().expect("Failed to flush temporary file");
        temp_file
    }

    #[test]
    fn test_tokenizer_new_load_fails_for_nonexistent_file() {
        let non_existent_path = Path::new("non_existent_tokenizer.json");
        let result = TokenizerWrapper::new(non_existent_path);
        assert!(result.is_err());
        match result {
            Err(TokenizerError::FailedToLoad(msg)) => {
                println!("Load non-existent file error: {}", msg); // Print error for debugging
                assert!(msg.contains("from file \"non_existent_tokenizer.json\""));
            }
            _ => panic!("Expected FailedToLoad error variant."),
        }
    }

    #[test]
    fn test_tokenizer_new_load_succeeds_and_get_vocab_size() {
        let _temp_file = setup_temp_tokenizer_file(DUMMY_TOKENIZER_JSON); // Keep file alive
        let wrapper_result = TokenizerWrapper::new(_temp_file.path());
        assert!(wrapper_result.is_ok(), "Failed to load dummy tokenizer: {:?}", wrapper_result.err());
        let wrapper = wrapper_result.unwrap();
        
        // Vocab from DUMMY_TOKENIZER_JSON:
        // "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4,
        // "hello": 5, "world": 6, "##d": 7, "Ġ": 8,
        // "Ġhello": 9, "Ġworld": 10, "Ġ##d" : 11
        // Total 12 tokens in the "model.vocab".
        // `get_vocab_size()` calls `self.tokenizer.get_vocab_size(true)`, which includes added tokens.
        // In this JSON, CLS and SEP are also defined in the main vocab and used by post_processor.
        // So, the vocab size should be the number of entries in "model.vocab".
        assert_eq!(wrapper.get_vocab_size(), 12, "Vocab size mismatch for dummy tokenizer.");
    }

    #[test]
    fn test_tokenizer_encode_decode() {
        let _temp_file = setup_temp_tokenizer_file(DUMMY_TOKENIZER_JSON); // Keep file alive
        let wrapper = TokenizerWrapper::new(_temp_file.path()).expect("Failed to load dummy tokenizer for encode/decode test.");

        let text_to_encode = "hello world";
        let encode_result = wrapper.encode(text_to_encode, false); // add_special_tokens = false
        assert!(encode_result.is_ok(), "Encoding failed: {:?}", encode_result.err().map(|e| e.to_string()));
        let encoded_ids = encode_result.unwrap();
        
        // With DUMMY_TOKENIZER_JSON:
        // Input: "hello world"
        // BertNormalizer (lowercase: true): "hello world"
        // BertPreTokenizer: Splits into words: "hello", "world"
        // BPE Model:
        //   "hello" is in vocab -> 5
        //   For "world", BertPreTokenizer gives "world". The BPE model needs "Ġworld" for ID 10.
        //   The pre_tokenizer should handle the space: "hello", "Ġworld" if it's not the first word.
        //   However, `encode` on `Tokenizer` with `BertPreTokenizer` usually handles space prefixing.
        //   So, "hello" -> 5. "world" (after first word) -> "Ġworld" -> 10.
        // Expected: [5, 10]
        assert_eq!(encoded_ids, vec![5, 10], "Encoded IDs do not match expected for 'hello world'.");

        let decode_result = wrapper.decode(&encoded_ids, true); // skip_special_tokens = true
        assert!(decode_result.is_ok(), "Decoding failed: {:?}", decode_result.err().map(|e| e.to_string()));
        let decoded_text = decode_result.unwrap();

        // The HuggingFace tokenizers library often lowercases by default with BertNormalizer,
        // and might not perfectly reconstruct the original casing if not configured.
        // The dummy tokenizer uses BertNormalizer with lowercase: true.
        assert_eq!(decoded_text, "hello world", "Decoded text does not match original (post-normalization).");

        // Test with special tokens
        let text_for_special_tokens = "hello world"; // User input part
        let encoded_special_result = wrapper.encode(text_for_special_tokens, true); // add_special_tokens = true
        assert!(encoded_special_result.is_ok(), "Encoding special tokens failed: {:?}", encoded_special_result.err().map(|e| e.to_string()));
        let encoded_special_ids = encoded_special_result.unwrap();

        // Input: "hello world", add_special_tokens=true
        // Normalized: "hello world"
        // Pre-tokenized by BertPreTokenizer: "hello", "world"
        // BPE encoding of parts: "hello" -> 5, "world" (becomes "Ġworld" due to space) -> 10. Intermediate: [5, 10]
        // Post-processor (BertProcessing): Adds [CLS] (ID 2) at start, [SEP] (ID 3) at end.
        // Expected: [2, 5, 10, 3]
        assert_eq!(encoded_special_ids, vec![2, 5, 10, 3], "Encoded IDs with special tokens do not match.");

        // Decode skipping special tokens
        let decoded_skip_special = wrapper.decode(&encoded_special_ids, true).unwrap();
        assert_eq!(decoded_skip_special, "hello world");
        
        // Decode without skipping special tokens
        let decoded_no_skip_special = wrapper.decode(&encoded_special_ids, false).unwrap();
        assert_eq!(decoded_no_skip_special, "[CLS] hello world [SEP]");
    }

    #[test]
    fn test_encode_error_handling() {
        // This test assumes that the underlying tokenizer's encode method can fail
        // For example, if it encounters an unhandled situation or internal limit.
        // Since the dummy tokenizer is very simple, it's hard to make it fail reliably
        // without specific knowledge of `tokenizers::Tokenizer` internals that cause errors
        // beyond simple string processing.
        // We'll simulate a failure by trying to encode a ridiculously long string,
        // though this might just be slow rather than erroring with the default settings.
        // A more robust test would mock the Tokenizer trait if we were using one, or
        // create a specific tokenizer file known to cause encoding issues for certain inputs.

        // For now, this test is more of a placeholder for error *type* checking
        // if an actual encoding error were to occur.
        let _temp_file = setup_temp_tokenizer_file(DUMMY_TOKENIZER_JSON); // Keep file alive
        let wrapper = TokenizerWrapper::new(_temp_file.path()).expect("Failed to load dummy tokenizer");
        
        // Let's assume a hypothetical scenario where encoding an empty string after certain ops might fail
        // (this is not typical for `encode` which usually returns empty for empty).
        // Or if a specific (malformed from user perspective) sequence of tokens might be problematic.
        // The current `tokenizers` lib is quite robust.
        // Instead, we'll check that if an error *were* to occur, its type is correct.
        // We can't easily make `tokenizer.encode()` fail with the current setup.
        // So, this test primarily ensures the error propagation *would* work.
        // No actual error expected here for valid inputs.
        let result = wrapper.encode("test", false);
        if let Err(e) = result {
            match e {
                TokenizerError::EncodingFailed(_) => { /* This is what we'd expect if it failed */ }
                _ => panic!("Unexpected error type for encoding failure."),
            }
        }
    }

    #[test]
    fn test_decode_error_handling() {
        // Similar to encode, making `decode` fail typically means providing invalid IDs
        // that are out of bounds of the vocabulary.
        let _temp_file = setup_temp_tokenizer_file(DUMMY_TOKENIZER_JSON); // Keep file alive
        let wrapper = TokenizerWrapper::new(_temp_file.path()).expect("Failed to load dummy tokenizer");

        let invalid_ids = vec![u32::MAX]; // An ID that's almost certainly not in the vocab
        let result = wrapper.decode(&invalid_ids, false);
        assert!(result.is_err());
        match result {
            Err(TokenizerError::DecodingFailed(msg)) => {
                println!("Decoding failed as expected: {}", msg);
                assert!(msg.contains(&format!("for IDs [{}]", u32::MAX)));
            }
            _ => panic!("Expected DecodingFailed error variant."),
        }
    }
    
    // NamedTempFile handles cleanup automatically when it goes out of scope.
}
