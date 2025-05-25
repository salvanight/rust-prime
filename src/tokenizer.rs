// Renaming import for clarity, though not strictly necessary if TruncationParams and PaddingParams are not used elsewhere.
// For this file, direct use of tokenizers::TruncationParams and tokenizers::PaddingParams in method signatures is fine.
use tokenizers::{Tokenizer, Error as TokenizersLibraryError, PaddingParams, TruncationParams, AddedToken}; // Import Error for From trait
use std::path::{Path, PathBuf}; // Added PathBuf for error context
use std::fmt;
use std::io; // For std::io::Error

#[cfg(feature = "tokenizer-debug-logs")]
use log::{debug, trace, warn};

/// Represents errors that can occur within the `TokenizerWrapper`.
#[derive(Debug)]
pub enum TokenizerError {
    /// An I/O error occurred.
    Io(io::Error),
    /// An error originating from the underlying `tokenizers` library.
    /// Contains the string representation of the library error.
    Library(String),
    /// Error encountered while attempting to load a tokenizer model from a file.
    /// This variant can be used if specific context beyond the library error is needed.
    /// For direct `?` usage with `tokenizers::Error`, `Library` variant will be used.
    FailedToLoad { path: PathBuf, source_message: String },
    /// Error encountered during the text encoding process.
    EncodingFailed { text: String, source_message: String },
    /// Error encountered during the token ID decoding process.
    DecodingFailed { ids: Vec<u32>, source_message: String },
}

/// Options for configuring the encoding process, specifically truncation and padding.
#[derive(Debug, Clone, Default)]
pub struct EncodeOptions {
    /// Optional truncation parameters. If `None`, no truncation is applied beyond tokenizer defaults.
    pub truncation: Option<TruncationParams>,
    /// Optional padding parameters. If `None`, no padding is applied.
    pub padding: Option<PaddingParams>,
}

impl EncodeOptions {
    /// Creates new `EncodeOptions` with both truncation and padding set to `None`.
    pub fn new() -> Self {
        Self::default()
    }
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerError::Io(err) => write!(f, "I/O error: {}", err),
            TokenizerError::Library(msg) => write!(f, "Tokenizer library error: {}", msg),
            TokenizerError::FailedToLoad{ path, source_message } => write!(f, "Failed to load tokenizer from {:?}: {}", path, source_message),
            TokenizerError::EncodingFailed{ text, source_message } => write!(f, "Encoding failed for text '{}': {}", text, source_message),
            TokenizerError::DecodingFailed{ ids, source_message } => write!(f, "Decoding failed for IDs {:?}: {}", ids, source_message),
        }
    }
}

impl std::error::Error for TokenizerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TokenizerError::Io(ref err) => Some(err),
            // Library error string is stored, so no deeper source.
            // If TokenizersLibraryError was boxed, it could be returned.
            TokenizerError::Library(_) => None, 
            TokenizerError::FailedToLoad { .. } => None, // source_message is String
            TokenizerError::EncodingFailed { .. } => None, // source_message is String
            TokenizerError::DecodingFailed { .. } => None, // source_message is String
        }
    }
}

// Implement From<std::io::Error> for TokenizerError
impl From<io::Error> for TokenizerError {
    fn from(err: io::Error) -> Self {
        TokenizerError::Io(err)
    }
}

// Implement From<tokenizers::Error> for TokenizerError
impl From<TokenizersLibraryError> for TokenizerError {
    fn from(err: TokenizersLibraryError) -> Self {
        TokenizerError::Library(err.to_string())
    }
}

/// A wrapper around the Hugging Face `tokenizers::Tokenizer` to provide a simplified API
/// for common tokenization tasks such as loading, encoding, decoding, and vocabulary size retrieval.
///
/// This struct holds an instance of `tokenizers::Tokenizer` and provides methods
/// that map to its core functionalities, along with custom error handling
/// defined by `TokenizerError`.
///
/// ## Optional Debug Logging
/// This crate includes optional debug logging via the `log` crate. To enable it, compile
/// with the `tokenizer-debug-logs` feature:
///
/// ```bash
/// cargo build --features tokenizer-debug-logs
/// ```
///
/// You will also need to use a logger implementation (e.g., `env_logger`, `simple_logger`)
/// in your application to see the log output.
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
    /// - `Err(TokenizerError::FailedToLoad)`: If the tokenizer model cannot be loaded.
    ///   The error includes the path and the underlying error message from the library.
    pub fn new(tokenizer_path: &Path) -> Result<Self, TokenizerError> {
        #[cfg(feature = "tokenizer-debug-logs")]
        debug!("Attempting to load tokenizer from path: {:?}", tokenizer_path);

        let tokenizer_instance = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| {
                #[cfg(feature = "tokenizer-debug-logs")]
                warn!("Failed to load tokenizer from {:?}: {}", tokenizer_path, e);
                TokenizerError::FailedToLoad {
                    path: tokenizer_path.to_path_buf(),
                    source_message: e.to_string(),
                }
            })?;

        #[cfg(feature = "tokenizer-debug-logs")]
        trace!("Successfully loaded tokenizer from path: {:?}", tokenizer_path);
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
    /// - `Err(TokenizerError::EncodingFailed)`: If encoding fails.
    ///   The error includes the input text and the underlying error message from the library.
    ///
    /// # Parameters
    /// - `text`: The text string to be encoded.
    /// - `add_special_tokens`: A boolean indicating whether to include special tokens.
    /// - `options`: An optional `EncodeOptions` struct to specify truncation and padding.
    ///   If `None`, default encoding behavior (no specific truncation or padding for this call) is used.
    pub fn encode(&self, text: &str, add_special_tokens: bool, options: Option<EncodeOptions>) -> Result<Vec<u32>, TokenizerError> {
        #[cfg(feature = "tokenizer-debug-logs")]
        debug!(
            "Encoding text: '{}', add_special_tokens: {}, options: {:?}",
            text, add_special_tokens, options
        );

        let mut tokenizer_instance = self.tokenizer.clone(); // Clone to apply per-call settings

        let (truncation_params, padding_params) = match options {
            Some(opts) => (opts.truncation, opts.padding),
            None => (None, None),
        };

        #[cfg(feature = "tokenizer-debug-logs")]
        if let Some(ref params) = truncation_params {
            trace!("Applying truncation parameters: {:?}", params);
        }
        tokenizer_instance.set_truncation(truncation_params)
            .map_err(|e| TokenizerError::EncodingFailed {
                text: format!("Failed to set truncation for text '{}'", if text.len() > 50 { &text[..50] } else { text }),
                source_message: e.to_string(),
            })?;

        #[cfg(feature = "tokenizer-debug-logs")]
        if let Some(ref params) = padding_params {
            trace!("Applying padding parameters: {:?}", params);
        }
        tokenizer_instance.set_padding(padding_params)
            .map_err(|e| TokenizerError::EncodingFailed {
                text: format!("Failed to set padding for text '{}'", if text.len() > 50 { &text[..50] } else { text }),
                source_message: e.to_string(),
            })?;

        let encoding_result = tokenizer_instance.encode(text, add_special_tokens)
            .map_err(|e| TokenizerError::EncodingFailed {
                text: text.to_string(), // Full text for this error context
                source_message: e.to_string(),
            })?;
        
        let encoded_ids = encoding_result.get_ids().to_vec();
        #[cfg(feature = "tokenizer-debug-logs")]
        trace!("Encoded IDs: {:?}", encoded_ids);
        Ok(encoded_ids)
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
    /// - `Err(TokenizerError::DecodingFailed)`: If decoding fails.
    ///   The error includes the token IDs and the underlying error message from the library.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, TokenizerError> {
        #[cfg(feature = "tokenizer-debug-logs")]
        debug!(
            "Decoding IDs: {:?}, skip_special_tokens: {}",
            ids, skip_special_tokens
        );

        let decoded_string = self.tokenizer.decode(ids, skip_special_tokens)
            .map_err(|e| TokenizerError::DecodingFailed {
                ids: ids.to_vec(),
                source_message: e.to_string(),
            })?;

        #[cfg(feature = "tokenizer-debug-logs")]
        trace!("Decoded text: '{}'", decoded_string);
        Ok(decoded_string)
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
    // The duplicated get_vocab_size was an artifact from a previous incorrect merge.
    // This is the correct single definition.
    pub fn get_vocab_size(&self) -> u32 {
        // This method is simple enough that perhaps only a trace log is needed.
        // The parameter `with_added_tokens` was part of a previous iteration but removed.
        // The current signature takes no parameters and defaults to `true`.
        let vocab_size = self.tokenizer.get_vocab_size(true) as u32;
        #[cfg(feature = "tokenizer-debug-logs")]
        trace!(
            "Getting vocab size (with_added_tokens=true). Returned size: {}",
            vocab_size
        );
        vocab_size
    }

    /// Adds new tokens to the tokenizer's vocabulary at runtime.
    ///
    /// This method allows extending the existing vocabulary with new tokens.
    /// The added tokens are typically treated as special tokens or whole words,
    /// depending on their configuration and the underlying tokenizer model.
    /// For example, to add a token that should be treated as a single unit and not
    /// be split by the model, you might use `AddedToken::from("new_word", true).single_word(true)`.
    ///
    /// Note: This modifies the internal tokenizer instance (`&mut self`).
    ///
    /// # Parameters
    /// - `tokens`: A slice of `AddedToken` structs representing the new tokens to add.
    ///
    /// # Returns
    /// - `Ok(usize)`: The number of tokens effectively added to the vocabulary.
    /// - `Err(TokenizerError::Library)`: If adding tokens fails (e.g., due to internal tokenizer issues).
    pub fn add_new_tokens(&mut self, tokens: &[AddedToken]) -> Result<usize, TokenizerError> {
        #[cfg(feature = "tokenizer-debug-logs")]
        debug!("Adding new tokens: {:?}", tokens);

        let num_added = self.tokenizer.add_tokens(tokens)
            .map_err(|e| TokenizerError::Library(format!("Failed to add new tokens: {}", e)))?;
        
        #[cfg(feature = "tokenizer-debug-logs")]
        trace!("Successfully added {} tokens. New vocab size: {}", num_added, self.tokenizer.get_vocab_size(true));
        Ok(num_added)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write; // For NamedTempFile::write_all
    use tempfile::NamedTempFile;
    use tokenizers::AddedToken; // Required for add_new_tokens test

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

    // This helper function is defined at the `tests` module level to be shared.
    // It ensures the NamedTempFile lives as long as the TokenizerWrapper instance.
    fn setup_temp_tokenizer_file_and_wrapper(json_content: &str) -> (NamedTempFile, TokenizerWrapper) {
        let mut temp_file = NamedTempFile::new().expect("Failed to create temporary file for test");
        temp_file.write_all(json_content.as_bytes()).expect("Failed to write to temporary test file");
        temp_file.flush().expect("Failed to flush temporary test file");
        let wrapper = TokenizerWrapper::new(temp_file.path())
            .expect("Failed to load TokenizerWrapper from temporary test file");
        (temp_file, wrapper)
    }

    #[test]
    fn test_tokenizer_new_load_fails_for_nonexistent_file() {
        let non_existent_path = Path::new("non_existent_tokenizer.json");
        let result = TokenizerWrapper::new(non_existent_path);
        assert!(result.is_err());
        match result {
            Err(TokenizerError::FailedToLoad{ path, source_message }) => {
                println!("Load non-existent file error (FailedToLoad): Path: {:?}, Source: {}", path, source_message);
                assert_eq!(path, non_existent_path.to_path_buf());
                // The source_message from tokenizers::Error for a non-existent file is usually specific.
                assert!(source_message.contains("No such file or directory") || source_message.contains("failed to read file"));
            }
            other_err => panic!("Expected FailedToLoad error variant, got {:?}", other_err),
        }
    }

    #[test]
    fn test_tokenizer_new_load_succeeds_and_get_vocab_size() {
        let (_temp_file, wrapper) = setup_temp_tokenizer_file_and_wrapper(DUMMY_TOKENIZER_JSON); 
        // No need to assert wrapper_result.is_ok(), as setup_..._and_wrapper panics on failure.
        
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
        let (_temp_file, wrapper) = setup_temp_tokenizer_file_and_wrapper(DUMMY_TOKENIZER_JSON);

        let text_to_encode = "hello world";
        // Pass None for options to keep original test behavior
        let encode_result = wrapper.encode(text_to_encode, false, None); 
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
        // Pass None for options
        let encoded_special_result = wrapper.encode(text_for_special_tokens, true, None); 
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
        let (_temp_file, wrapper) = setup_temp_tokenizer_file_and_wrapper(DUMMY_TOKENIZER_JSON);
        
        // Let's assume a hypothetical scenario where encoding an empty string after certain ops might fail
        // (this is not typical for `encode` which usually returns empty for empty).
        // Or if a specific (malformed from user perspective) sequence of tokens might be problematic.
        // The current `tokenizers` lib is quite robust.
        // Instead, we'll check that if an error *were* to occur, its type is correct.
        // We can't easily make `tokenizer.encode()` fail with the current setup.
        // So, this test primarily ensures the error propagation *would* work.
        // No actual error expected here for valid inputs.
        // Pass None for options
        let result = wrapper.encode("test", false, None); 
        if let Err(e) = result {
            match e {
                TokenizerError::EncodingFailed{ text: _, source_message: _ } => { /* This is what we'd expect if it failed */ }
                // If for some reason it was a Library error (e.g. direct ? from another source)
                TokenizerError::Library(_) => { /* Potentially acceptable if source_message is good */ }
                _ => panic!("Unexpected error type for encoding failure: {:?}", e),
            }
        }
    }

    #[test]
    fn test_decode_error_handling() {
        // Similar to encode, making `decode` fail typically means providing invalid IDs
        // that are out of bounds of the vocabulary.
        let (_temp_file, wrapper) = setup_temp_tokenizer_file_and_wrapper(DUMMY_TOKENIZER_JSON);

        let invalid_ids = vec![u32::MAX]; // An ID that's almost certainly not in the vocab
        let result = wrapper.decode(&invalid_ids, false);
        assert!(result.is_err());
        match result {
            Err(TokenizerError::DecodingFailed{ ids, source_message }) => {
                println!("Decoding failed as expected (DecodingFailed): IDs: {:?}, Source: {}", ids, source_message);
                assert_eq!(ids, &invalid_ids);
                // The source_message from tokenizers::Error for invalid ID might be like "TokenId `X` out of vocabulary bounds"
                assert!(source_message.contains("out of vocabulary bounds") || source_message.contains("invalid id"));
            }
            other_err => panic!("Expected DecodingFailed error variant, got {:?}", other_err),
        }
    }
    
    // NamedTempFile handles cleanup automatically when it goes out of scope.
}


    // NamedTempFile handles cleanup automatically when it goes out of scope.
}

#[cfg(test)]
mod proptests {
    use crate::tokenizer::{TokenizerWrapper, EncodeOptions, TokenizerError, TruncationParams, PaddingParams};
    use super::{setup_temp_tokenizer_file_and_wrapper, DUMMY_TOKENIZER_JSON}; 
    use proptest::prelude::*;

    // Strategy for pt_encode_decode_invariant:
    // Generates strings that are more likely to be handled well by the dummy tokenizer.
    // Focuses on known vocabulary words and simple combinations.
    fn reversible_string_strategy() -> impl Strategy<Value = String> {
        prop_oneof![
            Just("hello").boxed(),
            Just("world").boxed(),
            Just("hello world").boxed(),
            Just("Ġhello").boxed(), 
            Just("Ġworld").boxed(),
            // Limited character set to increase chances of being in vocab
            prop::collection::vec("[helowrdg# ]{1,10}", 1..5).prop_map(|parts| {
                let mut s = parts.join(" ");
                // Ensure some non-whitespace content if parts were all spaces
                if s.trim().is_empty() && !s.is_empty() { "hello".to_string() } else { s.trim().to_string() }
            }),
        ].no_shrink()
    }

    proptest! {
        #[test]
        fn pt_encode_decode_invariant(ref s in reversible_string_strategy()) {
            let (_temp_file, wrapper) = setup_temp_tokenizer_file_and_wrapper(DUMMY_TOKENIZER_JSON);

            let add_special_tokens = false;
            let options = None;

            prop_assume!(!s.is_empty() && !s.trim().is_empty(), "Skipping empty or whitespace-only string for this invariant test.");

            match wrapper.encode(s, add_special_tokens, options.clone()) {
                Ok(encoded_ids) => {
                    prop_assume!(!(encoded_ids.iter().all(|&id| id == 1) && s != "[UNK]"), "Input string '{}' encoded entirely to UNK tokens, skipping strict assertion.", s);
                    prop_assume!(!encoded_ids.is_empty(), "Input string '{}' encoded to empty IDs, skipping.",s);

                    match wrapper.decode(&encoded_ids, true) { // skip_special_tokens = true
                        Ok(decoded_text) => {
                            let mut normalized_s = s.to_lowercase();
                            normalized_s = normalized_s.split_whitespace().collect::<Vec<&str>>().join(" ");

                            let mut expected_decoded_text = normalized_s.clone();
                            if s.starts_with("Ġ") && s.len() > 1 && !s.chars().nth(1).unwrap().is_whitespace() {
                                 expected_decoded_text = s.chars().skip(1).collect::<String>().to_lowercase();
                                 expected_decoded_text = expected_decoded_text.split_whitespace().collect::<Vec<&str>>().join(" ");
                            }
                            
                            prop_assert_eq!(decoded_text, expected_decoded_text,
                                "Decoded text did not match expected. Input: '{}', Expected: '{}', Actual Decoded: '{}', Encoded: {:?}", 
                                s, expected_decoded_text, decoded_text, encoded_ids);
                        }
                        Err(e) => {
                            prop_assert!(false, "Decoding failed: {} for input '{}', encoded_ids {:?}", e, s, encoded_ids);
                        }
                    }
                }
                Err(e) => {
                    prop_assert!(false, "Encoding failed for supposedly reversible string: {} for input '{}'", e, s);
                }
            }
        }

        #[test]
        fn pt_encode_determinism(ref s in ".*\\PC*") { 
            let (_temp_file, wrapper) = setup_temp_tokenizer_file_and_wrapper(DUMMY_TOKENIZER_JSON);
            
            let options = None; 

            let res1 = wrapper.encode(s, false, options.clone());
            let res2 = wrapper.encode(s, false, options.clone());

            match (res1, res2) {
                (Ok(ids1), Ok(ids2)) => prop_assert_eq!(ids1, ids2, "Encoding was not deterministic for input '{}'", s),
                (Err(e1), Err(e2)) => {
                     prop_assert_eq!(e1.to_string(), e2.to_string(), "Error messages were not deterministic for input '{}'. e1: {}, e2: {}", s, e1, e2);
                },
                (Ok(ids), Err(e)) => prop_assert!(false, "Encoding mismatch: first Ok({:?}), second Err({}), input '{}'", ids, e, s),
                (Err(e), Ok(ids)) => prop_assert!(false, "Encoding mismatch: first Err({}), second Ok({:?}), input '{}'", e, ids, s),
            }
        }

        #[test]
        fn pt_encode_crash_test(ref s in ".*\\PC*") {
            let (_temp_file, wrapper) = setup_temp_tokenizer_file_and_wrapper(DUMMY_TOKENIZER_JSON);
            
            let _ = wrapper.encode(s, false, None); 
            let _ = wrapper.encode(s, true, None);  
            
            let trunc_params = TruncationParams { max_length: 10, ..Default::default() };
            let pad_params = PaddingParams { strategy: tokenizers::PaddingStrategy::Fixed(15), pad_id: 0, pad_token: "[PAD]".to_string(), ..Default::default() };
            let some_options = Some(EncodeOptions {
                truncation: Some(trunc_params),
                padding: Some(pad_params),
            });
            let _ = wrapper.encode(s, true, some_options); 
        }

        #[test]
        fn pt_decode_crash_test(ids in proptest::collection::vec(0u32..20, 0..100)) { // Vocab size is 12, test up to 19 for some invalid IDs.
            let (_temp_file, wrapper) = setup_temp_tokenizer_file_and_wrapper(DUMMY_TOKENIZER_JSON);

            let _ = wrapper.decode(&ids, false); 
            let _ = wrapper.decode(&ids, true);  
        }
    }
}


#[test]
fn test_encode_empty_string() {
    let (_temp_file, wrapper) = setup_temp_tokenizer_file_and_wrapper(DUMMY_TOKENIZER_JSON);

    let expected_ids: Vec<u32> = Vec::new();

    // Test with add_special_tokens: false
    let result_no_special = wrapper.encode("", false, None);
    assert!(result_no_special.is_ok(), "Encoding empty string (no special tokens) failed: {:?}", result_no_special.err());
    assert_eq!(result_no_special.unwrap(), expected_ids, "Encoding empty string (no special tokens) should produce empty vec.");

    // Test with add_special_tokens: true
    // BertProcessing post-processor might add [CLS] and [SEP] even for empty input.
    // Expected: [CLS, SEP] -> [2, 3]
    let result_special = wrapper.encode("", true, None);
    assert!(result_special.is_ok(), "Encoding empty string (with special tokens) failed: {:?}", result_special.err());
    assert_eq!(result_special.unwrap(), vec![2,3], "Encoding empty string (with special tokens) should produce [CLS, SEP].");
}

#[test]
fn test_encode_invalid_utf8() {
    let (_temp_file, wrapper) = setup_temp_tokenizer_file_and_wrapper(DUMMY_TOKENIZER_JSON);

    // Create a string containing the Unicode replacement character U+FFFD
    // This happens when from_utf8_lossy encounters invalid UTF-8 bytes.
    let invalid_bytes = &[0xC3, 0x28]; // Invalid: C3 without a following continuation byte for a 2-byte sequence
    let lossy_str = String::from_utf8_lossy(invalid_bytes); // Will contain �
    assert!(lossy_str.contains('\u{FFFD}'));

    // Expected behavior: The replacement character �, if not in vocab, maps to [UNK] (ID 1)
    // DUMMY_TOKENIZER_JSON maps "[UNK]" to 1.
    // The normalizer (BertNormalizer) might clean or handle this.
    // If '�' itself becomes a token, it might be UNK.
    let result = wrapper.encode(&lossy_str, false, None);
    assert!(result.is_ok(), "Encoding invalid UTF-8 (lossy) failed: {:?}", result.err());
    let encoded_ids = result.unwrap();
    // Depending on how BertNormalizer and BPE model handle '�':
    // 1. If '�' is treated as an unknown character, it should become [UNK] -> [1]
    // 2. If normalizer removes it, it could be empty.
    // Based on typical behavior, it should map to [UNK].
    assert_eq!(encoded_ids, vec![1], "Encoding lossy UTF-8 string should produce [UNK].");

    // Test with a more complex case
    let text_with_invalid = format!("hello {}", String::from_utf8_lossy(&[0xF0, 0x90, 0x80])); // Incomplete 4-byte sequence
    let result_complex = wrapper.encode(&text_with_invalid, false, None);
    assert!(result_complex.is_ok(), "Encoding complex invalid UTF-8 failed: {:?}", result_complex.err());
    // Expected: "hello �" -> "hello", "�" (pre-tokenized)
    // "hello" -> 5
    // "�" (as "Ġ�" because it's not first) -> UNK (1) if "Ġ�" not in vocab.
    // "Ġ" is 8. If � is UNK (1), then "Ġ�" could be [8,1] or just UNK [1].
    // It's more likely that "�" is tokenized as [UNK] (1).
    // So, "hello �" -> [5, 1]
    assert_eq!(result_complex.unwrap(), vec![5, 1], "Encoding 'hello �' should be [hello, UNK].");
}


#[test]
fn test_tokenizer_encode_truncation() {
    let (_temp_file, wrapper) = setup_temp_tokenizer_file_and_wrapper(DUMMY_TOKENIZER_JSON);

    let text = "hello world and some more words for testing truncation";
    // Based on DUMMY_TOKENIZER_JSON: "hello"->5, "Ġworld"->10, "Ġand"->UNK(1), "Ġsome"->UNK(1), "Ġmore"->UNK(1) ...
    // For simplicity, let's use a text that maps to known tokens mostly.
    // "hello world hello world" -> [5, 10, 9, 10] (Ġhello is 9)
    let text_long = "hello world hello world"; 
    let original_ids = wrapper.encode(text_long, false, None).unwrap();
    assert!(original_ids.len() > 3, "Original encoding is too short to test truncation meaningfully: {:?}", original_ids);

    let truncation_params = TruncationParams {
        max_length: 3,
        strategy: tokenizers::TruncationStrategy::LongestFirst, // Default, but explicit
        stride: 0, // Default
    };
    let options = Some(EncodeOptions {
        truncation: Some(truncation_params.clone()),
        padding: None,
    });
    let truncated_ids_res = wrapper.encode(text_long, false, options);
    assert!(truncated_ids_res.is_ok(), "Encoding with truncation failed: {:?}", truncated_ids_res.err());
    let truncated_ids = truncated_ids_res.unwrap();
    assert_eq!(truncated_ids.len(), 3, "Truncation did not limit to max_length.");
    assert_eq!(truncated_ids, vec![original_ids[0], original_ids[1], original_ids[2]], "Truncated IDs don't match start of original.");


    // Test with add_special_tokens = true.
    // "hello world" (text_for_special_tokens from previous test) -> [2, 5, 10, 3] (CLS, hello, Ġworld, SEP)
    // Truncate to max_length 3: Should be [2, 5, 3] (CLS, hello, SEP)
    let text_short_for_special = "hello"; // Encodes to [5] without special tokens. With special: [2,5,3]
    let options_special_trunc = Some(EncodeOptions {
        truncation: Some(truncation_params.clone()),
        padding: None,
    });
    let truncated_special_ids_res = wrapper.encode(text_short_for_special, true, options_special_trunc);
    assert!(truncated_special_ids_res.is_ok(), "Encoding with truncation and special tokens failed: {:?}", truncated_special_ids_res.err());
    let truncated_special_ids = truncated_special_ids_res.unwrap();
    assert_eq!(truncated_special_ids.len(), 3, "Truncation with special tokens did not limit to max_length correctly.");
    assert_eq!(truncated_special_ids, vec![2, 5, 3], "Truncated IDs with special tokens do not match expected ([CLS], hello, [SEP]).");
}

#[test]
fn test_tokenizer_encode_padding() {
    let (_temp_file, wrapper) = setup_temp_tokenizer_file_and_wrapper(DUMMY_TOKENIZER_JSON);
    // [PAD] token ID is 0 in DUMMY_TOKENIZER_JSON

    let text = "hello"; // Encodes to [5] without special tokens
    let original_ids = wrapper.encode(text, false, None).unwrap();
    assert_eq!(original_ids.len(), 1);

    let padding_params_fixed = PaddingParams {
        strategy: tokenizers::PaddingStrategy::Fixed(5), // Pad to length 5
        direction: tokenizers::PaddingDirection::Right,  // Default
        pad_to_multiple_of: None,
        pad_id: 0, // From DUMMY_TOKENIZER_JSON
        pad_type_id: 0, // Default
        pad_token: "[PAD]".to_string(), // From DUMMY_TOKENIZER_JSON
    };
    let options_padding = Some(EncodeOptions {
        truncation: None,
        padding: Some(padding_params_fixed.clone()),
    });

    let padded_ids_res = wrapper.encode(text, false, options_padding.clone());
    assert!(padded_ids_res.is_ok(), "Encoding with padding failed: {:?}", padded_ids_res.err());
    let padded_ids = padded_ids_res.unwrap();
    assert_eq!(padded_ids.len(), 5, "Padding did not extend to specified length.");
    assert_eq!(padded_ids, vec![5, 0, 0, 0, 0], "Padded IDs do not match expected content.");

    // Test padding with special tokens
    // "hello" (text) -> [5]. With special tokens: [CLS], hello, [SEP] -> [2, 5, 3]. Length 3.
    // Pad to 5: [2, 5, 3, 0, 0]
    let padded_special_ids_res = wrapper.encode(text, true, options_padding.clone());
    assert!(padded_special_ids_res.is_ok(), "Encoding with padding and special tokens failed: {:?}", padded_special_ids_res.err());
    let padded_special_ids = padded_special_ids_res.unwrap();
    assert_eq!(padded_special_ids.len(), 5, "Padding with special tokens did not extend to specified length.");
    assert_eq!(padded_special_ids, vec![2, 5, 3, 0, 0], "Padded IDs with special tokens do not match expected content.");

    // Test padding to a length shorter than the input (should not change input if tokenizer's padding strategy is Fixed)
    let short_padding_params = PaddingParams {
        strategy: tokenizers::PaddingStrategy::Fixed(1), // Try to pad to 1
        ..padding_params_fixed.clone() 
    };
    let options_short_padding = Some(EncodeOptions {
        truncation: None,
        padding: Some(short_padding_params),
    });
    let no_padding_needed_ids = wrapper.encode(text, false, options_short_padding).unwrap();
    assert_eq!(no_padding_needed_ids, vec![5], "Padding shorter than input changed the input, or padding strategy does not prevent it.");
}

#[test]
fn test_encode_extreme_truncation() {
    let (_temp_file, wrapper) = setup_temp_tokenizer_file_and_wrapper(DUMMY_TOKENIZER_JSON);
    let text = "hello"; // Encodes to [5] without special tokens, [2,5,3] with.

    // Max_length: 0
    let trunc_to_0 = TruncationParams { max_length: 0, strategy: tokenizers::TruncationStrategy::LongestFirst, stride: 0 };
    let opts_trunc_0 = Some(EncodeOptions { truncation: Some(trunc_to_0.clone()), padding: None });
    
    let res_0_no_special = wrapper.encode(text, false, opts_trunc_0.clone()).unwrap();
    assert_eq!(res_0_no_special, Vec::<u32>::new(), "Truncation to 0 (no special) should be empty.");

    // With add_special_tokens=true, max_length=0.
    // The BertProcessing post-processor adds [CLS] and [SEP]. Truncation to 0 might still allow these.
    // Typically, tokenizers will output at least special tokens if added, even if max_length is very small.
    // For max_length=0, it usually means only special tokens if they are added, resulting in e.g. [CLS, SEP].
    let res_0_special = wrapper.encode(text, true, opts_trunc_0.clone()).unwrap();
    assert_eq!(res_0_special, vec![2,3], "Truncation to 0 (with special) should be [CLS, SEP].");

    // Max_length: 1
    let trunc_to_1 = TruncationParams { max_length: 1, strategy: tokenizers::TruncationStrategy::LongestFirst, stride: 0 };
    let opts_trunc_1 = Some(EncodeOptions { truncation: Some(trunc_to_1.clone()), padding: None });

    let res_1_no_special = wrapper.encode(text, false, opts_trunc_1.clone()).unwrap();
    assert_eq!(res_1_no_special, vec![5], "Truncation to 1 (no special) for 'hello' should be [5].");
    
    // With add_special_tokens=true, max_length=1.
    // Sequence "hello" -> [5]. With special tokens -> [CLS, hello, SEP] -> [2,5,3].
    // Truncated to 1, it should take the first token, which is [CLS] (ID 2).
    let res_1_special = wrapper.encode(text, true, opts_trunc_1.clone()).unwrap();
    assert_eq!(res_1_special, vec![2], "Truncation to 1 (with special) for 'hello' should be [CLS].");
}

#[test]
fn test_encode_extreme_padding() {
    let (_temp_file, wrapper) = setup_temp_tokenizer_file_and_wrapper(DUMMY_TOKENIZER_JSON);
    let text = "hello"; // Encodes to [5] (length 1)

    let padding_params = |len: usize| PaddingParams {
        strategy: tokenizers::PaddingStrategy::Fixed(len),
        direction: tokenizers::PaddingDirection::Right,
        pad_id: 0, // [PAD]
        pad_token: "[PAD]".to_string(),
        pad_to_multiple_of: None,
        pad_type_id: 0,
    };

    // Pad to length 0
    let opts_pad_0 = Some(EncodeOptions { padding: Some(padding_params(0)), truncation: None });
    let res_pad_0 = wrapper.encode(text, false, opts_pad_0).unwrap();
    assert_eq!(res_pad_0, vec![5], "Padding to 0 should not change 'hello'.");

    // Pad to length 1 (same as original length)
    let opts_pad_1 = Some(EncodeOptions { padding: Some(padding_params(1)), truncation: None });
    let res_pad_1 = wrapper.encode(text, false, opts_pad_1).unwrap();
    assert_eq!(res_pad_1, vec![5], "Padding to 1 should not change 'hello'.");

    // Pad to length 1 (with special tokens)
    // "hello" with special tokens is [2,5,3] (length 3)
    // Padding to 1 should not change it.
    let opts_pad_1_special = Some(EncodeOptions { padding: Some(padding_params(1)), truncation: None });
    let res_pad_1_special = wrapper.encode(text, true, opts_pad_1_special).unwrap();
    assert_eq!(res_pad_1_special, vec![2,5,3], "Padding to 1 (special) should not change '[CLS] hello [SEP]'.");
}

#[test]
fn test_decode_multiple_invalid_ids() {
    let (_temp_file, wrapper) = setup_temp_tokenizer_file_and_wrapper(DUMMY_TOKENIZER_JSON);
    
    let invalid_ids = vec![5, u32::MAX, 6, u32::MAX -1]; // hello, UNK_CONCEPTUAL, world, UNK_CONCEPTUAL
    let result = wrapper.decode(&invalid_ids, false);
    assert!(result.is_err(), "Decoding sequence with multiple invalid IDs should fail.");
    
    match result {
        Err(TokenizerError::DecodingFailed{ ids, source_message }) => {
            assert_eq!(ids, invalid_ids, "Error should contain the original invalid IDs.");
            assert!(source_message.contains("out of vocabulary bounds") || source_message.contains("invalid id"),
                    "Error message should indicate an out-of-vocabulary or invalid ID issue.");
        }
        other_err => panic!("Expected DecodingFailed error variant, got {:?}", other_err),
    }
}

#[test]
fn test_encode_only_special_tokens() {
    let (_temp_file, wrapper) = setup_temp_tokenizer_file_and_wrapper(DUMMY_TOKENIZER_JSON);

    // Test case 1: "[CLS] [SEP]"
}


#[test]
fn test_add_new_tokens() {
    let (_temp_file, mut wrapper) = setup_temp_tokenizer_file_and_wrapper(DUMMY_TOKENIZER_JSON);
    let initial_vocab_size = wrapper.get_vocab_size(); // is 12 for DUMMY_TOKENIZER_JSON

    let new_tokens = [
        AddedToken::from("[NEW_TOKEN_1]", true), // true for single_word
        AddedToken::from("customword", true).single_word(true), // another way for single_word
        AddedToken::from("anothercustom", false), // not single_word, might be split by BPE
    ];

    let num_added = wrapper.add_new_tokens(&new_tokens).expect("Failed to add new tokens");
    assert_eq!(num_added, 3, "Expected 3 tokens to be added."); // Assuming all are new and successfully added

    let new_vocab_size = wrapper.get_vocab_size();
    assert_eq!(new_vocab_size, initial_vocab_size + num_added as u32, "Vocabulary size did not update correctly.");

    // Test encoding with the new tokens
    let text1 = "[NEW_TOKEN_1]";
    let encoded1 = wrapper.encode(text1, false, None).unwrap();
    let new_token_1_id = initial_vocab_size; // First new ID
    assert_eq!(encoded1, vec![new_token_1_id], "Encoding '[NEW_TOKEN_1]' failed.");
    let decoded1 = wrapper.decode(&encoded1, true).unwrap();
    assert_eq!(decoded1, "[NEW_TOKEN_1]".to_lowercase(), "Decoding '[NEW_TOKEN_1]' failed."); // Normalizer lowercases

    let text2 = "customword";
    let encoded2 = wrapper.encode(text2, false, None).unwrap();
    let customword_id = initial_vocab_size + 1; // Second new ID
    assert_eq!(encoded2, vec![customword_id], "Encoding 'customword' as single token failed.");
    let decoded2 = wrapper.decode(&encoded2, true).unwrap();
    assert_eq!(decoded2, "customword", "Decoding 'customword' failed.");

    // Test 'anothercustom' - this one was added with single_word=false (default)
    // It might be tokenized into subwords if its parts exist in the vocab or BPE can form it.
    // Or it might become UNK if it's entirely unknown and not a special format.
    // Given "anothercustom", and vocab like "hello", "world", "##d", "Ġ",
    // it is likely to be tokenized into multiple UNK tokens or subwords if any part matches.
    // For simplicity, let's assume it becomes a single new token if the tokenizer simply adds it
    // without further BPE processing for non-special new tokens.
    // However, the `tokenizers` library, when adding tokens that are not special and single_word=false,
    // might still break them down if they are not found in the vocab.
    // If it's added to vocab, it should get an ID.
    // Let's verify if it gets its ID.
    let text3 = "anothercustom";
    let encoded3 = wrapper.encode(text3, false, None).unwrap();
    let anothercustom_id = initial_vocab_size + 2; // Third new ID
    // This assertion depends on how the BPE model handles new non-single-word tokens.
    // If it's added to vocab and directly matched, it will be its ID.
    // If it's broken down (e.g. "another" "custom") and those parts map to UNK, it'll be multiple UNKs.
    // The `add_tokens` method usually makes the exact string learnable.
    assert_eq!(encoded3, vec![anothercustom_id], "Encoding 'anothercustom' failed. It might have been tokenized differently.");
    let decoded3 = wrapper.decode(&encoded3, true).unwrap();
    assert_eq!(decoded3, "anothercustom", "Decoding 'anothercustom' failed.");

    // Test encoding text that mixes old and new tokens
    let text_mixed = "hello [NEW_TOKEN_1] world customword";
    let encoded_mixed = wrapper.encode(text_mixed, false, None).unwrap();
    // Expected: "hello" -> 5, "[NEW_TOKEN_1]" -> new_token_1_id, "Ġworld" -> 10, "customword" -> customword_id
    // Note: "Ġworld" because "world" is preceded by a space after "[NEW_TOKEN_1]".
    // However, if "[NEW_TOKEN_1]" is treated like a word, then " world" might be "Ġworld".
    // The pretokenizer (BertPretokenizer) will split by space.
    // "[NEW_TOKEN_1]" is a token. "world" is a token. "customword" is a token.
    // "hello", "[NEW_TOKEN_1]", "world", "customword"
    // -> 5, new_token_1_id, 10 (Ġworld), customword_id
    // It depends on whether "[NEW_TOKEN_1]" is seen as having space-like boundaries by pre_tokenizer.
    // If "[NEW_TOKEN_1]" is treated as a single unit and pre_tokenizer splits "hello [NEW_TOKEN_1] world..."
    // into "hello", "[NEW_TOKEN_1]", "world", "customword".
    // Then IDs: 5 (hello), new_token_1_id, 10 (Ġworld), customword_id
    // Let's assume the more direct mapping for now.
    // The BertPreTokenizer behavior with added tokens can be complex.
    // For this dummy tokenizer, "[NEW_TOKEN_1]" is a word. "customword" is a word.
    // So, "hello", "[NEW_TOKEN_1]", "world", "customword"
    // "hello" -> 5
    // "[NEW_TOKEN_1]" -> new_token_1_id
    // "world" -> (will be prefixed by "Ġ" by BPE if not first word) -> 10
    // "customword" -> customword_id
    // Expected: [5, new_token_1_id, 10, customword_id]
    let expected_mixed_ids = vec![5, new_token_1_id, 10, customword_id];
    assert_eq!(encoded_mixed, expected_mixed_ids, "Encoding mixed text with new tokens failed.");
}
    let text1 = "[CLS] [SEP]";
    // add_special_tokens: false -> BertPreTokenizer gives "[CLS]", "[SEP]". These are in vocab.
    // Expected: [2, 3]
    let res1_no_special = wrapper.encode(text1, false, None).unwrap();
    assert_eq!(res1_no_special, vec![2, 3], "Encoding '[CLS] [SEP]' (no special) failed.");

    // add_special_tokens: true -> Input is tokenized to [2,3]. Post-processor adds CLS/SEP around this.
    // Expected: [2, 2, 3, 3] ([CLS] [CLS] [SEP] [SEP])
    let res1_special = wrapper.encode(text1, true, None).unwrap();
    assert_eq!(res1_special, vec![2, 2, 3, 3], "Encoding '[CLS] [SEP]' (with special) failed.");

    // Test case 2: "[UNK] [MASK]"
    let text2 = "[UNK] [MASK]";
    // add_special_tokens: false -> BertPreTokenizer gives "[UNK]", "[MASK]". These are in vocab.
    // Expected: [1, 4]
    let res2_no_special = wrapper.encode(text2, false, None).unwrap();
    assert_eq!(res2_no_special, vec![1, 4], "Encoding '[UNK] [MASK]' (no special) failed.");

    // add_special_tokens: true -> Input is tokenized to [1,4]. Post-processor adds CLS/SEP around this.
    // Expected: [2, 1, 4, 3] ([CLS] [UNK] [MASK] [SEP])
    let res2_special = wrapper.encode(text2, true, None).unwrap();
    assert_eq!(res2_special, vec![2, 1, 4, 3], "Encoding '[UNK] [MASK]' (with special) failed.");

    // Test case 3: "hello [SEP]"
    let text3 = "hello [SEP]";
    // add_special_tokens: false -> "hello", "[SEP]" -> [5, 3]
    let res3_no_special = wrapper.encode(text3, false, None).unwrap();
    assert_eq!(res3_no_special, vec![5, 3], "Encoding 'hello [SEP]' (no special) failed.");

    // add_special_tokens: true -> Input [5,3]. Post-processor: [2, 5, 3, 3]
    let res3_special = wrapper.encode(text3, true, None).unwrap();
    assert_eq!(res3_special, vec![2, 5, 3, 3], "Encoding 'hello [SEP]' (with special) failed.");
}
