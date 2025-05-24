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
        if index == 0 && line.starts_with("#") { // Skip header/comment line
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
    // Simple pre-tokenization: split by space and then process each word.
    // GPT-2 uses a more complex regex for pre-tokenization.
    // For now, we simulate word splitting and then character splitting for each word.
    // The 'Ġ' character is used by GPT-2 to indicate a space prefix.
    
    let mut token_ids = Vec::new();
    let words: Vec<String> = text.split_whitespace().map(|s| "Ġ".to_string() + s).collect();
    let mut processed_words = Vec::new();

    if text.trim().is_empty() { // Handle empty or whitespace-only input
        return Ok(Vec::new());
    }

    // If the original text doesn't start with a space, the first "word" shouldn't have 'Ġ'
    // This is a simplification; GPT-2's pretokenizer is more sophisticated.
    if !text.starts_with(' ') && !words.is_empty() {
        processed_words.push(words[0].trim_start_matches('Ġ').to_string());
        processed_words.extend(words.iter().skip(1).cloned());
    } else if words.is_empty() && !text.is_empty() { // Case where input is e.g. "abc" (no spaces)
        processed_words.push(text.to_string());
    }
     else {
        processed_words.extend(words.iter().cloned());
    }


    for word_str in processed_words {
        let mut symbols: Vec<String> = word_str.chars().map(|c| c.to_string()).collect();

        loop {
            let pairs = get_pairs(&symbols);
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
                break; // No more merges possible
            }
        }
        // Convert symbols to token IDs
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
    
    // Concatenate parts and handle GPT-2 specific space prefix 'Ġ'
    let full_text = text_parts.join("");
    let decoded_text = full_text.replace("Ġ", " ").trim_start().to_string(); // trim_start for cases where first token was like "Ġhello" but original was "hello"

    Ok(decoded_text)
}


// 6. Unit Tests (Basic setup)
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write; // Ensure Write is in scope for create_temp_file

    // Dummy vocab.json content - expanded for more comprehensive tests
    const DUMMY_VOCAB_JSON: &str = r#"
    {
        "hello": 0, "world": 1, "h": 2, "e": 3, "l": 4, "o": 5, "w": 6, "r": 7, "d": 8, "Ġ": 9,
        "he": 10, "ell": 11, "ello": 12, "wor": 13, "orld": 14, "Ġw": 15, "Ġh": 16,
        "Ġhello": 17, "Ġworld": 18, "low": 19, "er": 20, "Ġlower": 21, "Ġnew": 22, "Ġyork": 23,
        "ĠNew": 24, "ĠYork": 25, "n": 26, "ew": 27, "y": 28, "k": 29
    }
    "#;

    // Dummy merges.txt content - expanded
    const DUMMY_MERGES_TXT: &str = r#"
#version: 0.2
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
Ġ w # Ġ + w
w or
wor ld
ell o
Ġ h # Ġ + h
Ġhe llo # Ġhello
Ġwo rld # Ġworld
l o # for lower
o w # for lower (repeated but ok, lowest rank wins)
w e # for lower
e r # for lower
Ġ l # for Ġlower
Ġlo wer # for Ġlower
n e # for new
e w # for new
Ġ n # for Ġnew
Ġne w # for Ġnew
y o # for york
o r # for york (repeated)
r k # for york
Ġ y # for Ġyork
Ġyo rk # for Ġyork
Ġ N # for ĠNew
ĠNe w # for ĠNew
Ġ Y # for ĠYork
ĠYo rk # for ĠYork (Note: This assumes 'Y' and 'y' are distinct in vocab if needed)
"#;
    // Helper to create temp files for testing
    fn create_temp_file(content: &str, name: &str) -> String {
        let dir = std::env::temp_dir();
        // Potentially problematic if many tests run in parallel with same name.
        // For this exercise, it's fine. A unique name per test would be better.
        let path = dir.join(format!("test_{}", name)); 
        let mut file = File::create(&path).unwrap_or_else(|e| panic!("Failed to create temp file {:?}: {}", path, e));
        write!(file, "{}", content).unwrap();
        path.to_str().unwrap().to_string()
    }

    #[test]
    fn test_load_vocab() {
        let vocab_path = create_temp_file(DUMMY_VOCAB_JSON, "vocab.json");
        let vocab_result = load_vocab(&vocab_path);
        assert!(vocab_result.is_ok(), "load_vocab failed: {:?}", vocab_result.err());
        let vocab = vocab_result.unwrap();
        assert_eq!(vocab.get("hello"), Some(&0));
        assert_eq!(vocab.get("Ġ"), Some(&9));
        assert_eq!(vocab.len(), 29); 
        std::fs::remove_file(vocab_path).unwrap();
    }

    #[test]
    fn test_load_merges() {
        let merges_path = create_temp_file(DUMMY_MERGES_TXT, "merges.txt");
        let merges_result = load_merges(&merges_path);
        assert!(merges_result.is_ok(), "load_merges failed: {:?}", merges_result.err());
        let merges = merges_result.unwrap();
        assert_eq!(merges.get(&("h".to_string(), "e".to_string())), Some(&0)); 
        assert_eq!(merges.get(&("Ġ".to_string(), "w".to_string())), Some(&10));
        assert_eq!(merges.len(), 32); // Number of actual merge rules
        std::fs::remove_file(merges_path).unwrap();
    }

    #[test]
    fn test_load_merges_empty_lines() {
        let merges_content = "#version: 0.2\nh e\n\nl l";
        let merges_path = create_temp_file(merges_content, "merges_empty.txt");
        let merges_result = load_merges(&merges_path);
        assert!(merges_result.is_ok());
        let merges = merges_result.unwrap();
        assert_eq!(merges.len(), 2);
        assert_eq!(merges.get(&("h".to_string(), "e".to_string())), Some(&0));
        assert_eq!(merges.get(&("l".to_string(), "l".to_string())), Some(&1));
        std::fs::remove_file(merges_path).unwrap();
    }

    #[test]
    fn test_load_merges_format_error() {
        let merges_content = "#version: 0.2\nh e l p"; // malformed line
        let merges_path = create_temp_file(merges_content, "merges_error.txt");
        let merges_result = load_merges(&merges_path);
        assert!(merges_result.is_err());
        if let Err(TokenizerError::FileFormatError(msg)) = merges_result {
            assert!(msg.contains("Invalid merge rule format"));
        } else {
            panic!("Expected FileFormatError");
        }
        std::fs::remove_file(merges_path).unwrap();
    }

    // Tests for encode and decode
    fn setup_tokenizer() -> (Vocabulary, BpeMerges) {
        // In a real scenario, you'd load from files. For tests, we can create them in memory.
        // Or, use the create_temp_file helpers if preferred, but direct creation is faster.
        let vocab: Vocabulary = serde_json::from_str(DUMMY_VOCAB_JSON).unwrap();
        
        // Manually construct merges for more precise control in some tests if needed,
        // or use load_merges with DUMMY_MERGES_TXT for broader testing.
        // For simplicity, re-using load_merges here.
        let merges_path = create_temp_file(DUMMY_MERGES_TXT, "merges_for_encode_decode.txt");
        let merges = load_merges(&merges_path).unwrap();
        std::fs::remove_file(merges_path).unwrap();
        (vocab, merges)
    }

    #[test]
    fn test_encode_simple() {
        let (vocab, merges) = setup_tokenizer();
        let text = "hello";
        let expected_ids = vec![vocab["hello"]]; // Assuming "hello" is a single token after BPE
        let encoded_ids = encode(text, &vocab, &merges).unwrap();
        assert_eq!(encoded_ids, expected_ids);
    }
    
    #[test]
    fn test_encode_with_merges() {
        let (vocab, merges) = setup_tokenizer();
        // "hello" -> "h", "e", "l", "l", "o" -> "he", "l", "l", "o" -> "hel", "l", "o" -> "hell", "o" -> "hello"
        // vocab must contain "hello" directly for this test to be simple, or intermediate forms
        let text = "hello"; // This should become a single token if "hello" is in vocab and merges support it
        let encoded_ids = encode(text, &vocab, &merges).unwrap();
         // The DUMMY_VOCAB_JSON has "hello": 0.
        assert_eq!(encoded_ids, vec![0]);
    }

    #[test]
    fn test_encode_sentence() {
        let (vocab, merges) = setup_tokenizer();
        let text = "hello world";
        // Expected based on DUMMY_VOCAB_JSON and DUMMY_MERGES_TXT:
        // "hello" -> 0
        // " world" -> "Ġworld" -> 18
        let expected_ids = vec![vocab["hello"], vocab["Ġworld"]];
        let encoded_ids = encode(text, &vocab, &merges).unwrap();
        assert_eq!(encoded_ids, expected_ids);
    }

    #[test]
    fn test_decode_simple() {
        let (vocab, _) = setup_tokenizer();
        let token_ids = vec![vocab["hello"]];
        let decoded_text = decode(&token_ids, &vocab).unwrap();
        assert_eq!(decoded_text, "hello");
    }

    #[test]
    fn test_decode_sentence() {
        let (vocab, _) = setup_tokenizer();
        let token_ids = vec![vocab["hello"], vocab["Ġworld"]];
        let decoded_text = decode(&token_ids, &vocab).unwrap();
        assert_eq!(decoded_text, "hello world");
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let (vocab, merges) = setup_tokenizer();
        let original_text = "hello world";
        let encoded_ids = encode(original_text, &vocab, &merges).unwrap();
        let decoded_text = decode(&encoded_ids, &vocab).unwrap();
        assert_eq!(decoded_text, original_text);

        let original_text_2 = "ĠNew York"; // Test with leading space marker
        let encoded_ids_2 = encode(original_text_2, &vocab, &merges).unwrap();
        let decoded_text_2 = decode(&encoded_ids_2, &vocab).unwrap();
        assert_eq!(decoded_text_2, "New York"); // Decode should handle the Ġ
    }
    
    #[test]
    fn test_encode_unknown_symbol() {
        let (vocab, merges) = setup_tokenizer();
        let text = "hello world unknown"; // "unknown" and its chars are not in vocab
        let result = encode(text, &vocab, &merges);
        assert!(result.is_err());
        if let Err(TokenizerError::VocabularyMiss(e)) = result {
            // Depending on how it breaks "unknown" down, the missing char might vary.
            // Example: 'u' if 'u' is not in vocab.
            assert!(e.contains("Symbol not in vocabulary: u") || e.contains("Symbol not in vocabulary: n") || e.contains("Symbol not in vocabulary: k"));
        } else {
            panic!("Expected VocabularyMiss error");
        }
    }
     #[test]
    fn test_encode_empty_string() {
        let (vocab, merges) = setup_tokenizer();
        let text = "";
        let encoded_ids = encode(text, &vocab, &merges).unwrap();
        assert!(encoded_ids.is_empty());
    }

    #[test]
    fn test_decode_empty_tokens() {
        let (vocab, _) = setup_tokenizer();
        let token_ids: Vec<u32> = Vec::new();
        let decoded_text = decode(&token_ids, &vocab).unwrap();
        assert!(decoded_text.is_empty());
    }

    #[test]
    fn test_pre_tokenization_logic_for_encode() {
        let (vocab, merges) = setup_tokenizer();
        // Test case: "helloworld" (no space) vs "hello world"
        let text_no_space = "helloworld"; // Should be treated as one word, potentially one token
        let text_with_space = "hello world"; // Two words, "hello" and "Ġworld"

        // Assuming "helloworld" is not in vocab, it will break down.
        // "h", "e", "l", "l", "o", "w", "o", "r", "l", "d"
        // Then BPE applied. Let's assume it becomes "hello" and "world" if merges allow
        // This requires "helloworld" to be composed of characters in vocab, and merges to form them.
        // For DUMMY_VOCAB, "hello":0, "world":1.
        // If "helloworld" itself is not a token and no merges combine "o" and "w" across the boundary.
        // "hello" -> 0
        // "world" -> 1
        // Expected: [0, 1]
        let encoded_no_space = encode(text_no_space, &vocab, &merges).unwrap();

        // "hello world" -> "hello", "Ġworld" -> [vocab["hello"], vocab["Ġworld"]]
        let encoded_with_space = encode(text_with_space, &vocab, &merges).unwrap();
        
        // This test depends heavily on the exact vocab and merges.
        // For the current DUMMY_VOCAB, "hello" is 0, "world" is 1.
        // "helloworld" as one word would be tokenized based on available merges.
        // "h" "e" "l" "l" "o" "w" "o" "r" "l" "d"
        // -> "he" "ll" "o" "w" "o" "r" "l" "d"
        // -> "hell" "o" "w" "o" "r" "l" "d"
        // -> "hello" "w" "o" "r" "l" "d" (token 0)
        // -> "hello" "wo" "r" "l" "d"
        // -> "hello" "wor" "l" "d" (token 13 for "wor")
        // -> "hello" "wor" "ld" (token 14 for "orld" - no, "ld" is not a merge from "r" and "l")
        // It's more likely "hello" -> 0, then "w", "o", "r", "l", "d" are processed.
        // "w","o","r","l","d" -> "wo","r","l","d" -> "wor","l","d" -> "wor","ld"
        // So, if "wor" and "ld" are tokens: [0, vocab["wor"], vocab["ld"]] (assuming "ld" is a token)
        // Our vocab has "wor":13, "orld":14. "d":8, "l":4
        // Let's trace "world": "w" "o" "r" "l" "d"
        // -> "wo" "r" "l" "d" (rank for "w o" is 4)
        // -> "wor" "l" "d" (rank for "wo r" is 11)
        // -> "wor" "ld" (rank for "r l" is 6, rank for "l d" is 7)
        //    Pairs: (wor, l), (l,d). Merge (l,d) first as rank 7 < rank for (wor,l) if it exists.
        //    Let's assume "ld" is a symbol. vocab["ld"] if it exists.
        //    If "ld" is not in vocab, this path fails.
        //    If "orld" is in vocab (it is: 14), and "wor" + "ld" -> "world" (it is: 1)
        //    "world" -> "w" "o" "r" "l" "d"
        //    Pairs: (w,o) (o,r) (r,l) (l,d)
        //    Ranks: (w,o):4, (o,r):5, (r,l):6, (l,d):7
        //    Merge (w,o) -> "wo" "r" "l" "d"
        //    Pairs: (wo,r) (r,l) (l,d)
        //    Ranks: (wo,r):11 (if "w o r" is "wor"), (r,l):6, (l,d):7
        //    Merge (r,l) -> "wo" "rl" "d" (assuming "rl" is a symbol)
        //    This shows the complexity. GPT-2 pre-tokenization splits "helloworld" into "hello" and "world".
        // My current naive char split for words will make "helloworld" one unit.
        // Let's simplify the assertion for `text_no_space` based on it being a single unit.
        // If "helloworld" is not in vocab, and cannot be fully merged into a single token from vocab,
        // it should become multiple tokens. E.g. [vocab["hello"], vocab["world"]] if merges make it so.
        // Based on current DUMMY_VOCAB and MERGES:
        // "hello": 0. "world": 1.
        // "helloworld" -> "h" "e" "l" "l" "o" "w" "o" "r" "l" "d"
        // -> "he" "l" "l" "o" "w" "o" "r" "l" "d" (merge h e)
        // -> "hel" "l" "o" "w" "o" "r" "l" "d" (merge he l)
        // -> "hell" "o" "w" "o" "r" "l" "d" (merge hel l)
        // -> "hello" "w" "o" "r" "l" "d" (merge hell o) -> token 0
        // Now for "world": "w" "o" "r" "l" "d"
        // -> "wo" "r" "l" "d" (merge w o)
        // -> "wor" "l" "d" (merge wo r)
        // -> "wor" "ld" (merge l d) -- this is "l" "d" -> "ld"
        // -> "world" (merge wor ld) -> token 1
        // So, "helloworld" should indeed become [0, 1] with the current setup.
        assert_eq!(encoded_no_space, vec![vocab["hello"], vocab["world"]]);

        // "hello world" -> "hello" (processed as "hello") and "Ġworld" (processed as "Ġworld")
        // "hello" -> 0
        // "Ġworld" -> 18
        assert_eq!(encoded_with_space, vec![vocab["hello"], vocab["Ġworld"]]);
    }
}
