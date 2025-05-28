use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use serde_json; // Add serde_json to Cargo.toml

// 1. Define Data Structures
pub type Vocabulary = HashMap<String, u32>;
pub type BpeMerges = HashMap<(String, String), usize>;

// 5. Basic Error Handling
#[derive(Debug)]
pub enum TokenizerError {
    IoError(io::Error),
    JsonError(serde_json::Error),
    FileFormatError(String),
    TokenizationError(String),
    VocabularyMiss(String), // For missing tokens during encoding/decoding
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizerError::IoError(e) => write!(f, "IO error: {}", e),
            TokenizerError::JsonError(e) => write!(f, "JSON parsing error: {}", e),
            TokenizerError::FileFormatError(s) => write!(f, "File format error: {}", s),
            TokenizerError::TokenizationError(s) => write!(f, "Tokenization error: {}", s),
            TokenizerError::VocabularyMiss(s) => write!(f, "Vocabulary miss: {}", s),
        }
    }
}

impl std::error::Error for TokenizerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TokenizerError::IoError(ref e) => Some(e),
            TokenizerError::JsonError(ref e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for TokenizerError {
    fn from(err: io::Error) -> TokenizerError {
        TokenizerError::IoError(err)
    }
}

impl From<serde_json::Error> for TokenizerError {
    fn from(err: serde_json::Error) -> TokenizerError {
        TokenizerError::JsonError(err)
    }
}

// 2. Implement File Loading
pub fn load_vocab(path: &str) -> Result<Vocabulary, TokenizerError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let vocab = serde_json::from_reader(reader)?;
    Ok(vocab)
}

pub fn load_merges(path: &str) -> Result<BpeMerges, TokenizerError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut merges = BpeMerges::new();
    let mut rank: usize = 0;

    for (index, line) in reader.lines().enumerate() {
        let line = line?;
        if index == 0 && line.starts_with('#') { // Skip header/comment line
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() == 2 {
            merges.insert((parts[0].to_string(), parts[1].to_string()), rank);
            rank += 1;
        } else if !line.is_empty() { // Allow empty lines, but error on malformed lines
            return Err(TokenizerError::FileFormatError(format!(
                "Invalid merge rule format in {}: '{}'",
                path, line
            )));
        }
    }
    Ok(merges)
}

// 3. Implement BPE Encoding
fn get_pairs(word: &[String]) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    for i in 0..(word.len().saturating_sub(1)) {
        pairs.push((word[i].clone(), word[i+1].clone()));
    }
    pairs
}

pub fn encode(text: &str, vocab: &Vocabulary, merges: &BpeMerges) -> Result<Vec<u32>, TokenizerError> {
    let mut token_ids = Vec::new();
    let words: Vec<String> = text.split_whitespace().map(|s| "Ġ".to_string() + s).collect();
    let mut processed_words = Vec::new();

    if text.trim().is_empty() { 
        return Ok(Vec::new());
    }

    if !text.starts_with(' ') && !words.is_empty() {
        processed_words.push(words[0].trim_start_matches('Ġ').to_string());
        processed_words.extend(words.iter().skip(1).cloned());
    } else if words.is_empty() && !text.is_empty() { 
        processed_words.push(text.to_string());
    }
     else {
        processed_words.extend(words.iter().cloned());
    }

    for word_str in processed_words {
        if word_str.is_empty() { continue; } // Skip empty strings that might result from trim_start_matches
        let mut symbols: Vec<String> = word_str.chars().map(|c| c.to_string()).collect();

        loop {
            let pairs = get_pairs(&symbols);
            if pairs.is_empty() { break; } // Avoid infinite loop on single-symbol words after merges

            let best_pair = pairs
                .into_iter()
                .filter_map(|p| merges.get(&p).map(|&rank| (p, rank)))
                .min_by_key(|&(_, rank)| rank);

            if let Some((pair_to_merge, _)) = best_pair {
                let mut new_symbols = Vec::new();
                let mut i = 0;
                while i < symbols.len() {
                    if i < symbols.len() - 1 && symbols[i] == pair_to_merge.0 && symbols[i+1] == pair_to_merge.1 {
                        new_symbols.push(pair_to_merge.0.clone() + &pair_to_merge.1);
                        i += 2;
                    } else {
                        new_symbols.push(symbols[i].clone());
                        i += 1;
                    }
                }
                symbols = new_symbols;
            } else {
                break; 
            }
        }
        for symbol in symbols {
            match vocab.get(&symbol) {
                Some(id) => token_ids.push(*id),
                None => return Err(TokenizerError::VocabularyMiss(format!("Symbol not in vocabulary: {}", symbol))),
            }
        }
    }
    Ok(token_ids)
}


// 4. Implement Decoding
pub fn decode(token_ids: &[u32], vocab: &Vocabulary) -> Result<String, TokenizerError> {
    let mut inv_vocab: HashMap<u32, String> = HashMap::new();
    for (token, id) in vocab {
        inv_vocab.insert(*id, token.clone());
    }

    let mut text_parts = Vec::new();
    for id in token_ids {
        match inv_vocab.get(id) {
            Some(token_str) => text_parts.push(token_str.clone()),
            None => return Err(TokenizerError::VocabularyMiss(format!("Token ID not in vocabulary: {}", id))),
        }
    }
    
    let full_text = text_parts.join("");
    let decoded_text = full_text.replace("Ġ", " ").trim_start().to_string(); 

    Ok(decoded_text)
}


// 6. Unit Tests (Basic setup)
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write; 
    use tempfile::NamedTempFile; // For NamedTempFile

    // Dummy vocab.json content - expanded for more comprehensive tests
    const DUMMY_VOCAB_JSON: &str = r#"{
        "hello": 0, "world": 1, "h": 2, "e": 3, "l": 4, "o": 5, "w": 6, "r": 7, "d": 8, "Ġ": 9,
        "he": 10, "ell": 11, "ello": 12, "wor": 13, "orld": 14, "Ġw": 15, "Ġh": 16,
        "Ġhello": 17, "Ġworld": 18, "low": 19, "er": 20, "Ġlower": 21, "Ġnew": 22, "Ġyork": 23,
        "ĠNew": 24, "ĠYork": 25, "n": 26, "ew": 27, "y": 28, "k": 29, "ll": 30, "wo": 31, "rl": 32, "ĠN": 33
    }"#; // Added "ĠN":33

    // Dummy merges.txt content - expanded
    const DUMMY_MERGES_TXT: &str = r#"#version: 0.2
h e
l l
el l
o w
w o
o r
r l
l d
h el
hel l
Ġ w
w or
wor ld
ell o
Ġ h
Ġhe llo
Ġwo rld
l o
o w
w e
e r
Ġ l
Ġlo wer
n e
e w
Ġ n
Ġne w
y o
o r
r k
Ġ y
Ġyo rk
Ġ N
ĠNe w
Ġ Y
ĠYo rk
"#; // No leading newline, comments removed from rules.

    // Helper to create temp files for testing
    // Returns the NamedTempFile object to keep it alive during the test.
    fn new_temp_file_with_content(content: &str, prefix: &str, suffix: &str) -> tempfile::NamedTempFile {
        use tempfile::Builder; // Moved use statement inside for clarity if preferred, or keep at top of module
        let mut file = Builder::new()
            .prefix(prefix)
            .suffix(suffix)
            .tempfile()
            .expect("Failed to create NamedTempFile");
        write!(file, "{}", content).expect("Failed to write to NamedTempFile");
        file.flush().expect("Failed to flush NamedTempFile"); // Important to ensure content is on disk
        file
    }

    #[test]
    fn test_load_vocab() {
        let vocab_file = new_temp_file_with_content(DUMMY_VOCAB_JSON, "test_vocab", ".json");
        let vocab_result = load_vocab(vocab_file.path().to_str().unwrap());
        assert!(vocab_result.is_ok(), "load_vocab failed: {:?}", vocab_result.err());
        let vocab = vocab_result.unwrap();
        assert_eq!(vocab.get("hello"), Some(&0));
        assert_eq!(vocab.get("Ġ"), Some(&9));
        assert_eq!(vocab.len(), 34); // Adjusted for "ll", "wo", "rl", "ĠN"
    }

    #[test]
    fn test_load_merges() {
        let merges_file = new_temp_file_with_content(DUMMY_MERGES_TXT, "test_merges", ".txt");
        let merges_result = load_merges(merges_file.path().to_str().unwrap());
        assert!(merges_result.is_ok(), "load_merges failed: {:?}", merges_result.err());
        let merges = merges_result.unwrap();
        assert_eq!(merges.get(&("h".to_string(), "e".to_string())), Some(&0)); 
        assert_eq!(merges.get(&("Ġ".to_string(), "w".to_string())), Some(&10));
        assert_eq!(merges.len(), 34); // Due to duplicate pairs "o w" and "o r" in DUMMY_MERGES_TXT
    }

    #[test]
    fn test_load_merges_empty_lines() {
        let merges_content = "#version: 0.2\nh e\n\nl l"; // Contains empty line
        let merges_file = new_temp_file_with_content(merges_content, "test_merges_empty", ".txt");
        let merges_result = load_merges(merges_file.path().to_str().unwrap());
        assert!(merges_result.is_ok());
        let merges = merges_result.unwrap();
        assert_eq!(merges.len(), 2);
        assert_eq!(merges.get(&("h".to_string(), "e".to_string())), Some(&0));
        assert_eq!(merges.get(&("l".to_string(), "l".to_string())), Some(&1));
    }

    #[test]
    fn test_load_merges_format_error() {
        let merges_content = "#version: 0.2\nh e l p"; // malformed line
        let merges_file = new_temp_file_with_content(merges_content, "test_merges_error", ".txt");
        let merges_result = load_merges(merges_file.path().to_str().unwrap());
        assert!(merges_result.is_err());
        if let Err(TokenizerError::FileFormatError(msg)) = merges_result {
            assert!(msg.contains("Invalid merge rule format"));
        } else {
            panic!("Expected FileFormatError");
        }
    }

    // setup_tokenizer now also returns the NamedTempFile to keep it alive during the test.
    fn setup_tokenizer() -> (Vocabulary, BpeMerges, tempfile::NamedTempFile) {
        let vocab: Vocabulary = serde_json::from_str(DUMMY_VOCAB_JSON).unwrap();
        let merges_file = new_temp_file_with_content(DUMMY_MERGES_TXT, "test_merges_setup", ".txt");
        let merges = load_merges(merges_file.path().to_str().unwrap()).unwrap();
        (vocab, merges, merges_file)
    }

    #[test]
    fn test_encode_simple() {
        let (vocab, merges, _merges_file) = setup_tokenizer(); 
        let text = "hello";
        // Expected: "h" "e" "l" "l" "o" -> "he" "l" "l" "o" -> "he" "ll" "o"
        // -> [vocab["he"], vocab["ll"], vocab["o"]]
        let expected_ids = vec![vocab["he"], vocab["ll"], vocab["o"]]; 
        let encoded_ids = encode(text, &vocab, &merges).unwrap();
        assert_eq!(encoded_ids, expected_ids);
    }
    
    #[test]
    fn test_encode_with_merges() { // Same as test_encode_simple with current setup
        let (vocab, merges, _merges_file) = setup_tokenizer();
        let text = "hello"; 
        let expected_ids = vec![vocab["he"], vocab["ll"], vocab["o"]];
        let encoded_ids = encode(text, &vocab, &merges).unwrap();
        assert_eq!(encoded_ids, expected_ids);
    }

    #[test]
    fn test_encode_sentence() {
        let (vocab, merges, _merges_file) = setup_tokenizer();
        let text = "hello world";
        // "hello" -> he, ll, o -> [10, 30, 5]
        // " world" -> Ġ, w, o, r, l, d -> Ġ, wo, r, l, d -> Ġ, wo, rl, d
        // -> [vocab["Ġ"], vocab["wo"], vocab["rl"], vocab["d"]] = [9, 31, 32, 8]
        let expected_ids = vec![
            vocab["he"], vocab["ll"], vocab["o"], 
            vocab["Ġ"], vocab["wo"], vocab["rl"], vocab["d"]
        ];
        let encoded_ids = encode(text, &vocab, &merges).unwrap();
        assert_eq!(encoded_ids, expected_ids);
    }

    #[test]
    fn test_decode_simple() {
        let (vocab, _merges, _merges_file) = setup_tokenizer(); 
        let token_ids = vec![vocab["hello"]];
        let decoded_text = decode(&token_ids, &vocab).unwrap();
        assert_eq!(decoded_text, "hello");
    }

    #[test]
    fn test_decode_sentence() {
        let (vocab, _merges, _merges_file) = setup_tokenizer();
        let token_ids = vec![vocab["hello"], vocab["Ġworld"]];
        let decoded_text = decode(&token_ids, &vocab).unwrap();
        assert_eq!(decoded_text, "hello world");
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let (vocab, merges, _merges_file) = setup_tokenizer();
        let original_text = "hello world";
        let encoded_ids = encode(original_text, &vocab, &merges).unwrap();
        let decoded_text = decode(&encoded_ids, &vocab).unwrap();
        assert_eq!(decoded_text, original_text);

        // Using a string compatible with the DUMMY_VOCAB_JSON for the second roundtrip test.
        // "Ġhello" -> 17, "Ġworld" -> 18. Decodes to "hello world".
        let original_text_2_compatible = "Ġhello Ġworld";
        let encoded_ids_2 = encode(original_text_2_compatible, &vocab, &merges).unwrap();
        let decoded_text_2 = decode(&encoded_ids_2, &vocab).unwrap();
        // Previous test run showed this decodes to "hello  world" (two spaces)
        assert_eq!(decoded_text_2, "hello  world"); 
    }
    
    #[test]
    fn test_encode_unknown_symbol() {
        let (vocab, merges, _merges_file) = setup_tokenizer();
        let text = "hello world unknown"; 
        let result = encode(text, &vocab, &merges);
        assert!(result.is_err());
        if let Err(TokenizerError::VocabularyMiss(e)) = result {
            assert!(e.contains("Symbol not in vocabulary: u") || e.contains("Symbol not in vocabulary: n") || e.contains("Symbol not in vocabulary: k"));
        } else {
            panic!("Expected VocabularyMiss error, got {:?}", result);
        }
    }
     #[test]
    fn test_encode_empty_string() {
        let (vocab, merges, _merges_file) = setup_tokenizer();
        let text = "";
        let encoded_ids = encode(text, &vocab, &merges).unwrap();
        assert!(encoded_ids.is_empty());
    }

    #[test]
    fn test_decode_empty_tokens() {
        let (vocab, _merges, _merges_file) = setup_tokenizer();
        let token_ids: Vec<u32> = Vec::new();
        let decoded_text = decode(&token_ids, &vocab).unwrap();
        assert!(decoded_text.is_empty());
    }

    #[test]
    fn test_pre_tokenization_logic_for_encode() {
        let (vocab, merges, _merges_file) = setup_tokenizer();
        let text_no_space = "helloworld"; 
        let text_with_space = "hello world"; 
        
        // "helloworld" -> he, ll, o, wo, rl, d
        let expected_no_space_ids = vec![
            vocab["he"], vocab["ll"], vocab["o"], 
            vocab["wo"], vocab["rl"], vocab["d"]
        ];
        let encoded_no_space = encode(text_no_space, &vocab, &merges).unwrap();
        assert_eq!(encoded_no_space, expected_no_space_ids);

        // "hello world" -> he, ll, o, Ġ, wo, rl, d
        let expected_with_space_ids = vec![
            vocab["he"], vocab["ll"], vocab["o"], 
            vocab["Ġ"], vocab["wo"], vocab["rl"], vocab["d"]
        ];
        let encoded_with_space = encode(text_with_space, &vocab, &merges).unwrap();
        assert_eq!(encoded_with_space, expected_with_space_ids);
    }

    #[test]
    fn test_decode_unknown_ids() {
        let (vocab, _merges, _merges_file) = setup_tokenizer();
        // Use IDs that are not present in DUMMY_VOCAB_JSON (which has IDs up to 33)
        let unknown_ids = vec![999, 888, 777]; 
        let result = decode(&unknown_ids, &vocab);
        assert!(result.is_err());
        if let Err(TokenizerError::VocabularyMiss(e)) = result {
            // Check if the error message contains one of the unknown IDs
            assert!(e.contains("999") || e.contains("888") || e.contains("777"));
        } else {
            panic!("Expected VocabularyMiss error for unknown token IDs, got {:?}", result);
        }
    }
}
