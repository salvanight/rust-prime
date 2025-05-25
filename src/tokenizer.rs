use tokenizers::Tokenizer; // From the 'tokenizers' crate
use std::path::Path;

#[derive(Debug)] // Added Debug derive for convenience
pub struct TokenizerWrapper {
    tokenizer: Tokenizer,
}

impl TokenizerWrapper {
    pub fn new(tokenizer_path: &Path) -> Result<Self, String> {
        let tokenizer_instance = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| format!("Failed to load tokenizer from {:?}: {}", tokenizer_path, e))?;
        Ok(Self { tokenizer: tokenizer_instance })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, String> {
        let encoding = self.tokenizer.encode(text, add_special_tokens)
            .map_err(|e| format!("Encoding failed for text '{}': {}", text, e))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, String> {
        self.tokenizer.decode(ids, skip_special_tokens)
            .map_err(|e| format!("Decoding failed for IDs {:?}: {}", ids, e))
    }

    pub fn get_vocab_size(&self) -> u32 {
        self.tokenizer.get_vocab_size(true) as u32 // Cast usize to u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::fs; // Required for cleaning up test file if necessary

    // Helper to get the path to the dummy tokenizer created in the previous step.
    // Assumes it's in the crate root.
    fn get_dummy_tokenizer_path() -> PathBuf {
        PathBuf::from("test_tokenizer.json")
    }
    
    // It's good practice to clean up created test files, though NamedTempFile is better for this.
    // For now, if test_tokenizer.json is part of the repo for testing, manual cleanup is fine.
    // If it's generated per test run, it should be cleaned up.
    // Given it was created in a previous step, we'll assume it exists for these tests.

    #[test]
    fn test_tokenizer_new_load_fails_for_nonexistent_file() {
        let non_existent_path = Path::new("non_existent_tokenizer.json");
        let result = TokenizerWrapper::new(non_existent_path);
        assert!(result.is_err());
        if let Err(e) = result {
            println!("Load non-existent file error: {}", e); // Print error for debugging
            assert!(e.contains("Failed to load tokenizer"));
        }
    }

    #[test]
    fn test_tokenizer_new_load_succeeds_and_get_vocab_size() {
        let dummy_path = get_dummy_tokenizer_path();
        // Ensure the dummy file exists before running this test.
        // If it doesn't, this test will fail, which is expected.
        // The previous step should have created it.
        if !dummy_path.exists() {
            panic!("Dummy tokenizer file 'test_tokenizer.json' not found. Create it before running tests.");
        }

        let wrapper_result = TokenizerWrapper::new(&dummy_path);
        assert!(wrapper_result.is_ok(), "Failed to load dummy tokenizer: {:?}", wrapper_result.err());
        let wrapper = wrapper_result.unwrap();
        
        // The dummy tokenizer has "hello", "world", "##d" + 5 special tokens ([PAD], [UNK], [CLS], [SEP], [MASK])
        // So, vocab size should be 3 (regular words) + 5 (added special tokens) = 8.
        // Let's verify the vocab in test_tokenizer.json:
        // "vocab": { "hello": 5, "world": 6, "##d": 7, "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4 }
        // The keys are the tokens. The values are their IDs.
        // The `get_vocab_size(true)` method returns the total size including added tokens.
        // The highest ID is 7 ("##d"), so IDs are 0-7, meaning size is 8.
        assert_eq!(wrapper.get_vocab_size(), 8, "Vocab size mismatch for dummy tokenizer.");
    }

    #[test]
    fn test_tokenizer_encode_decode() {
        let dummy_path = get_dummy_tokenizer_path();
        if !dummy_path.exists() {
            panic!("Dummy tokenizer file 'test_tokenizer.json' not found.");
        }
        let wrapper = TokenizerWrapper::new(&dummy_path).expect("Failed to load dummy tokenizer for encode/decode test.");

        let text_to_encode = "hello world";
        let encode_result = wrapper.encode(text_to_encode, false); // add_special_tokens = false
        assert!(encode_result.is_ok(), "Encoding failed: {:?}", encode_result.err());
        let encoded_ids = encode_result.unwrap();
        
        // Based on dummy_tokenizer.json: "hello" -> 5, "world" -> 6
        assert_eq!(encoded_ids, vec![5, 6], "Encoded IDs do not match expected for 'hello world'.");

        let decode_result = wrapper.decode(&encoded_ids, true); // skip_special_tokens = true
        assert!(decode_result.is_ok(), "Decoding failed: {:?}", decode_result.err());
        let decoded_text = decode_result.unwrap();

        // The HuggingFace tokenizers library often lowercases by default with BertNormalizer,
        // and might not perfectly reconstruct the original casing if not configured.
        // The dummy tokenizer uses BertNormalizer with lowercase: true.
        assert_eq!(decoded_text, "hello world", "Decoded text does not match original (post-normalization).");

        // Test with special tokens
        let text_with_special = "[CLS] hello world [SEP]";
        let encoded_special_result = wrapper.encode(text_with_special, true); // add_special_tokens = true (though already in text)
        assert!(encoded_special_result.is_ok());
        let encoded_special_ids = encoded_special_result.unwrap();
        // Expected: [CLS] -> 2, hello -> 5, world -> 6, [SEP] -> 3
        assert_eq!(encoded_special_ids, vec![2, 5, 6, 3]);

        // Decode skipping special tokens
        let decoded_skip_special = wrapper.decode(&encoded_special_ids, true).unwrap();
        assert_eq!(decoded_skip_special, "hello world");
        
        // Decode without skipping special tokens
        let decoded_no_skip_special = wrapper.decode(&encoded_special_ids, false).unwrap();
        assert_eq!(decoded_no_skip_special, "[CLS] hello world [SEP]");
    }
    
    // Optional: Clean up the dummy tokenizer file after all tests in this module if it's not version controlled.
    // This would require a custom test runner or a teardown mechanism, which is complex.
    // For now, manual cleanup or versioning the test_tokenizer.json is assumed.

    #[test]
    fn test_encode_empty_string() {
        let dummy_path = get_dummy_tokenizer_path();
        if !dummy_path.exists() {
            panic!("Dummy tokenizer file 'test_tokenizer.json' not found.");
        }
        let wrapper = TokenizerWrapper::new(&dummy_path).expect("Failed to load dummy tokenizer");

        let encode_result = wrapper.encode("", false);
        assert!(encode_result.is_ok(), "Encoding empty string failed: {:?}", encode_result.err());
        assert!(encode_result.unwrap().is_empty(), "Encoding an empty string should result in an empty Vec<u32>");
    }

    #[test]
    fn test_decode_empty_ids() {
        let dummy_path = get_dummy_tokenizer_path();
        if !dummy_path.exists() {
            panic!("Dummy tokenizer file 'test_tokenizer.json' not found.");
        }
        let wrapper = TokenizerWrapper::new(&dummy_path).expect("Failed to load dummy tokenizer");

        let decode_result = wrapper.decode(&[], true);
        assert!(decode_result.is_ok(), "Decoding empty ID list failed: {:?}", decode_result.err());
        assert!(decode_result.unwrap().is_empty(), "Decoding an empty Vec<u32> should result in an empty String");
    }

    #[test]
    fn test_encode_oov_string() {
        let dummy_path = get_dummy_tokenizer_path();
        if !dummy_path.exists() {
            panic!("Dummy tokenizer file 'test_tokenizer.json' not found.");
        }
        let wrapper = TokenizerWrapper::new(&dummy_path).expect("Failed to load dummy tokenizer");

        let text_oov = "xyz"; // These characters are not in test_tokenizer.json
        let encode_result = wrapper.encode(text_oov, false);
        assert!(encode_result.is_ok(), "Encoding OOV string failed: {:?}", encode_result.err());
        let encoded_ids = encode_result.unwrap();
        // Expect OOV characters to be mapped to the [UNK] token (ID 1 in test_tokenizer.json)
        // The WordPiece model, if it can't break down "xyz" into known subwords,
        // will map the entire "xyz" to a single [UNK].
        assert_eq!(encoded_ids, vec![1], "Encoding OOV string 'xyz' should result in a single UNK token.");

        // Test decoding this UNK token
        let decode_result = wrapper.decode(&encoded_ids, true); // skip_special_tokens = true
        assert!(decode_result.is_ok(), "Decoding UNK token failed: {:?}", decode_result.err());
        // Decoding [UNK] (ID 1) with skip_special_tokens = true should result in an empty string
        // because [UNK] is marked as "special": true in test_tokenizer.json.
        assert_eq!(decode_result.unwrap(), "", "Decoding UNK token with skip_special_tokens=true did not produce expected empty string.");
    }

    #[test]
    fn test_encode_only_special_tokens_string() {
        let dummy_path = get_dummy_tokenizer_path();
        if !dummy_path.exists() {
            panic!("Dummy tokenizer file 'test_tokenizer.json' not found.");
        }
        let wrapper = TokenizerWrapper::new(&dummy_path).expect("Failed to load dummy tokenizer");
        
        let text_special = "[CLS] [SEP]";
        let encode_result = wrapper.encode(text_special, true); // add_special_tokens = true
        assert!(encode_result.is_ok(), "Encoding string of only special tokens failed: {:?}", encode_result.err());
        // Expected: [CLS] -> 2, [SEP] -> 3
        assert_eq!(encode_result.unwrap(), vec![2, 3], "Encoding string of special tokens produced incorrect IDs.");
    }

    #[test]
    fn test_decode_only_special_tokens_ids() {
        let dummy_path = get_dummy_tokenizer_path();
        if !dummy_path.exists() {
            panic!("Dummy tokenizer file 'test_tokenizer.json' not found.");
        }
        let wrapper = TokenizerWrapper::new(&dummy_path).expect("Failed to load dummy tokenizer");

        let special_ids = vec![0, 1, 4]; // [PAD], [UNK], [MASK]
        
        // Decode skipping special tokens
        let decode_skip_result = wrapper.decode(&special_ids, true);
        assert!(decode_skip_result.is_ok(), "Decoding special IDs (skip=true) failed: {:?}", decode_skip_result.err());
        // "[UNK]" is often not skipped by `skip_special_tokens=true` if it's considered content.
        // However, [PAD] and [MASK] should be skipped.
        // The `tokenizers` library behavior for `skip_special_tokens` is that it skips tokens marked `special: true`
        // in `added_tokens`. In `test_tokenizer.json`, [PAD], [UNK], [CLS], [SEP], [MASK] are all `special: true`.
        assert_eq!(decode_skip_result.unwrap(), "", "Decoding only special IDs (skip=true) should result in empty or specific UNK string if not skipped.");

        // Decode without skipping special tokens
        let decode_no_skip_result = wrapper.decode(&special_ids, false);
        assert!(decode_no_skip_result.is_ok(), "Decoding special IDs (skip=false) failed: {:?}", decode_no_skip_result.err());
        assert_eq!(decode_no_skip_result.unwrap(), "[PAD] [UNK] [MASK]", "Decoding only special IDs (skip=false) did not produce expected string.");
    }
}
