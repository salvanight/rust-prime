// src/text_generator.rs

use crate::resonance_feedback::ResonanceFeedbackStore; // Added
use crate::tensor_engine::{Tensor, TensorError};
use crate::transformer_core::{GPT2Model, TransformerError as ModelError};
use rand::distributions::WeightedIndex;
use rand::prelude::*;

// 1. TextGeneratorError Enum
#[derive(Debug)]
pub enum TextGeneratorError {
    ModelError(ModelError),
    TensorError(TensorError),
    InvalidInput(String),
    GenerationLimitReached(String), // If max_length is hit in a way that's an error condition elsewhere
}

impl std::fmt::Display for TextGeneratorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TextGeneratorError::ModelError(e) => write!(f, "Model error: {:?}", e), // Use {:?} if Display is not detailed
            TextGeneratorError::TensorError(e) => write!(f, "Tensor error: {:?}", e), // Same here
            TextGeneratorError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
            TextGeneratorError::GenerationLimitReached(s) => {
                write!(f, "Generation limit reached: {}", s)
            }
        }
    }
}

impl std::error::Error for TextGeneratorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TextGeneratorError::ModelError(ref e) => Some(e), // Assuming ModelError (TransformerError) impls Error
            TextGeneratorError::TensorError(ref e) => Some(e), // Assuming TensorError impls Error
            _ => None,
        }
    }
}

impl From<ModelError> for TextGeneratorError {
    fn from(err: ModelError) -> TextGeneratorError {
        TextGeneratorError::ModelError(err)
    }
}

impl From<TensorError> for TextGeneratorError {
    fn from(err: TensorError) -> TextGeneratorError {
        TextGeneratorError::TensorError(err)
    }
}

// --- Sampling Traits and Types ---

pub trait TopKDecoder {
    fn sample_top_k(
        logits: &[f32],
        k: usize,
        rng: &mut impl Rng,
    ) -> Result<u32, TextGeneratorError>;
}

pub trait TopPDecoder {
    fn sample_top_p(logits: &[f32], p: f32, rng: &mut impl Rng) -> Result<u32, TextGeneratorError>;
}

#[derive(Clone, Copy, Debug)]
pub enum SamplingMethod {
    Greedy,
    TopK { k: usize },
    TopP { p: f32 },
}

pub struct DefaultSampler;

impl TopKDecoder for DefaultSampler {
    fn sample_top_k(
        logits: &[f32],
        k: usize,
        rng: &mut impl Rng,
    ) -> Result<u32, TextGeneratorError> {
        if k == 0 {
            return Err(TextGeneratorError::InvalidInput(
                "k must be greater than 0".into(),
            ));
        }
        if logits.is_empty() {
            return Err(TextGeneratorError::InvalidInput(
                "logits slice is empty".into(),
            ));
        }
        let mut pairs: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let slice = &pairs[..k.min(pairs.len())];
        let weights: Vec<f32> = slice.iter().map(|&(_, v)| v.exp()).collect();
        let dist = WeightedIndex::new(&weights)
            .map_err(|e| TextGeneratorError::InvalidInput(format!("Invalid weights: {}", e)))?;
        let idx = dist.sample(rng);
        Ok(slice[idx].0 as u32)
    }
}

impl TopPDecoder for DefaultSampler {
    fn sample_top_p(logits: &[f32], p: f32, rng: &mut impl Rng) -> Result<u32, TextGeneratorError> {
        if !(0.0..=1.0).contains(&p) {
            return Err(TextGeneratorError::InvalidInput(
                "p must be within [0,1]".into(),
            ));
        }
        if logits.is_empty() {
            return Err(TextGeneratorError::InvalidInput(
                "logits slice is empty".into(),
            ));
        }
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let mut probs: Vec<(usize, f32)> = exp
            .into_iter()
            .enumerate()
            .map(|(i, v)| (i, v / sum))
            .collect();
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut cumulative = 0.0;
        let mut cutoff = 0;
        for (i, &(_, prob)) in probs.iter().enumerate() {
            cumulative += prob;
            if cumulative >= p {
                cutoff = i + 1;
                break;
            }
        }
        let slice = &probs[..cutoff.max(1)];
        let weights: Vec<f32> = slice.iter().map(|&(_, pr)| pr).collect();
        let dist = WeightedIndex::new(&weights)
            .map_err(|e| TextGeneratorError::InvalidInput(format!("Invalid weights: {}", e)))?;
        let idx = dist.sample(rng);
        Ok(slice[idx].0 as u32)
    }
}

// 2. Local argmax helper function
fn argmax(data: &[f32]) -> Result<usize, TextGeneratorError> {
    if data.is_empty() {
        return Err(TextGeneratorError::InvalidInput(
            "Cannot perform argmax on empty data slice".to_string(),
        ));
    }
    let mut max_val = data[0];
    let mut max_idx = 0;
    for (i, &val) in data.iter().enumerate().skip(1) {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    Ok(max_idx)
}

// 3. `generate` Function
pub fn generate(
    model: &GPT2Model,
    input_ids: Vec<u32>,
    max_length: usize,
    eos_token_id: u32,
    sampling_method: SamplingMethod,
    feedback_store: Option<&ResonanceFeedbackStore>,
) -> Result<Vec<u32>, TextGeneratorError> {
    // Placeholder for feedback usage
    if let Some(store) = feedback_store {
        // TODO: Future enhancement:
        // Query the feedback_store to potentially adjust generation strategy.
        // For example, retrieve recent positive/negative experiences
        // to influence logits or sampling parameters.
        let recent_feedback = store.get_recent_experiences(5); // Example query
                                                               // Using a print statement for now to confirm it's accessible
                                                               // In a real scenario, this might be logged or used more subtly.
        if !recent_feedback.is_empty() {
            // Only print if there's actually feedback
            println!(
                "[TextGenerator] Feedback store available. Found {} recent entries (example).",
                recent_feedback.len()
            );
        }
    }

    if input_ids.is_empty() {
        return Err(TextGeneratorError::InvalidInput(
            "Input IDs cannot be empty.".to_string(),
        ));
    }
    if max_length == 0 {
        return Err(TextGeneratorError::InvalidInput(
            "Max length must be greater than 0.".to_string(),
        ));
    }
    if input_ids.len() >= max_length {
        // Return a copy of input_ids if it's already at or beyond max_length
        return Ok(input_ids[..max_length.min(input_ids.len())].to_vec());
    }

    let mut generated_ids = input_ids.clone();
    let mut rng = thread_rng();

    for _ in 0..(max_length - input_ids.len()) {
        let current_sequence_length = generated_ids.len();

        // a. Prepare input tensor
        let input_tensor = Tensor::new(generated_ids.clone(), vec![1, current_sequence_length])?;

        // b. Call model.forward()
        // Logits tensor shape: `[1, current_sequence_length, vocab_size]`.
        let logits_tensor = model.forward(&input_tensor, None, None, None)?;

        // c. Extract logits for the *last token*
        // Logits for the last token are at the end of the `data` Vec.
        // If shape is [1, S, V], data is flat.
        // Logits for token s (0-indexed) start at index s * V.
        // We want logits for the last token, so s = current_sequence_length - 1.
        let vocab_size = model.config.vocab_size;
        let last_token_logits_start_idx = (current_sequence_length - 1) * vocab_size;
        let last_token_logits_end_idx = last_token_logits_start_idx + vocab_size;

        if logits_tensor.data.len() < last_token_logits_end_idx {
            return Err(TextGeneratorError::TensorError(TensorError::OutOfBounds(
                "Logits tensor data length is too small for extracting last token logits."
                    .to_string(),
            )));
        }
        let last_token_logits_slice =
            &logits_tensor.data[last_token_logits_start_idx..last_token_logits_end_idx];

        // d. Select the next token according to the sampling method
        let next_token_id = match sampling_method {
            SamplingMethod::Greedy => argmax(last_token_logits_slice)? as u32,
            SamplingMethod::TopK { k } => {
                DefaultSampler::sample_top_k(last_token_logits_slice, k, &mut rng)?
            }
            SamplingMethod::TopP { p } => {
                DefaultSampler::sample_top_p(last_token_logits_slice, p, &mut rng)?
            }
        };

        // g. Append `next_token_id` to `generated_ids`.
        generated_ids.push(next_token_id);

        // f. If `next_token_id == eos_token_id`, break the loop.
        // Now checking after push, so EOS token is included in the output.
        if next_token_id == eos_token_id {
            break;
        }

        // h. If `generated_ids.len() >= max_length`, break the loop (already handled by loop condition).
        // This check is also implicitly handled by the main loop condition, but double check is fine.
        if generated_ids.len() >= max_length {
            break;
        }
    }

    Ok(generated_ids)
}

// 4. Unit Tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer_core::{Config, GPT2Model}; // For creating a dummy model
    use rand::{rngs::StdRng, SeedableRng};
    use std::collections::HashMap; // For dummy weights

    fn create_dummy_model_for_generation(
        vocab_size: usize,
        block_size: usize,
        n_embd: usize,
    ) -> GPT2Model {
        let config = Config {
            n_layer: 1,
            n_head: 1,
            n_embd,
            vocab_size,
            block_size,
            bias: true,
        };

        // Create minimal dummy weights
        let mut weights = HashMap::new();
        weights.insert(
            "wte.weight".to_string(),
            Tensor::zeros(vec![vocab_size, n_embd]),
        );
        weights.insert(
            "wpe.weight".to_string(),
            Tensor::zeros(vec![block_size, n_embd]),
        );

        weights.insert(
            "h.0.attn.c_attn.weight".to_string(),
            Tensor::zeros(vec![n_embd, 3 * n_embd]),
        );
        weights.insert(
            "h.0.attn.c_attn.bias".to_string(),
            Tensor::zeros(vec![3 * n_embd]),
        );
        weights.insert(
            "h.0.attn.c_proj.weight".to_string(),
            Tensor::zeros(vec![n_embd, n_embd]),
        );
        weights.insert(
            "h.0.attn.c_proj.bias".to_string(),
            Tensor::zeros(vec![n_embd]),
        );

        weights.insert(
            "h.0.mlp.c_fc.weight".to_string(),
            Tensor::zeros(vec![n_embd, 4 * n_embd]),
        );
        weights.insert(
            "h.0.mlp.c_fc.bias".to_string(),
            Tensor::zeros(vec![4 * n_embd]),
        );
        weights.insert(
            "h.0.mlp.c_proj.weight".to_string(),
            Tensor::zeros(vec![4 * n_embd, n_embd]),
        );
        weights.insert(
            "h.0.mlp.c_proj.bias".to_string(),
            Tensor::zeros(vec![n_embd]),
        );

        weights.insert("h.0.ln_1.weight".to_string(), Tensor::zeros(vec![n_embd]));
        weights.insert("h.0.ln_1.bias".to_string(), Tensor::zeros(vec![n_embd]));
        weights.insert("h.0.ln_2.weight".to_string(), Tensor::zeros(vec![n_embd]));
        weights.insert("h.0.ln_2.bias".to_string(), Tensor::zeros(vec![n_embd]));

        weights.insert("ln_f.weight".to_string(), Tensor::zeros(vec![n_embd]));
        weights.insert("ln_f.bias".to_string(), Tensor::zeros(vec![n_embd]));

        GPT2Model::new(config, weights).expect("Failed to create dummy GPT2Model for testing")
    }

    #[test]
    fn test_argmax_simple() {
        let data = vec![1.0, 0.0, 3.0, 2.0];
        assert_eq!(argmax(&data).unwrap(), 2);
        let data_neg = vec![-1.0, -5.0, -0.5];
        assert_eq!(argmax(&data_neg).unwrap(), 2);
    }

    #[test]
    fn test_argmax_empty() {
        let data: Vec<f32> = vec![];
        assert!(argmax(&data).is_err());
    }

    #[test]
    fn test_top_k_sampler_respects_k() {
        let logits = vec![0.0, 10.0, 5.0, 1.0];
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let token = DefaultSampler::sample_top_k(&logits, 2, &mut rng).unwrap();
        assert!(token == 1 || token == 2);
    }

    #[test]
    fn test_top_p_sampler_basic() {
        let logits = vec![0.0, 10.0, 5.0, 1.0];
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let token = DefaultSampler::sample_top_p(&logits, 0.7, &mut rng).unwrap();
        assert!(token == 1 || token == 2);
    }

    #[test]
    fn test_generate_reaches_max_length() {
        let vocab_size = 10;
        let block_size = 5;
        let n_embd = 4;
        let model = create_dummy_model_for_generation(vocab_size, block_size, n_embd);

        let input_ids = vec![1u32];
        let max_len = 3;
        let eos_id = 99;

        let result = generate(
            &model,
            input_ids.clone(),
            max_len,
            eos_id,
            SamplingMethod::Greedy,
            None,
        );

        assert!(result.is_ok(), "Generation failed: {:?}", result.err());
        let generated_sequence = result.unwrap();
        assert_eq!(generated_sequence.len(), max_len);
        assert_eq!(generated_sequence[0], input_ids[0]);
        assert_eq!(generated_sequence, vec![1, 0, 0]);
    }

    #[test]
    fn test_generate_stops_at_eos() {
        let vocab_size = 10;
        let block_size = 5;
        let n_embd = 4;
        let model = create_dummy_model_for_generation(vocab_size, block_size, n_embd);
        let input_ids = vec![1u32];
        let max_len = 5;
        let eos_id = 0;

        let result = generate(
            &model,
            input_ids.clone(),
            max_len,
            eos_id,
            SamplingMethod::Greedy,
            None,
        );
        assert!(result.is_ok(), "Generation failed: {:?}", result.err());
        let generated_sequence = result.unwrap();

        assert_eq!(generated_sequence, vec![1, 0]);
        assert_eq!(generated_sequence.last().unwrap(), &eos_id);
        assert!(generated_sequence.len() < max_len);
    }

    #[test]
    fn test_generate_input_already_max_length() {
        let vocab_size = 10;
        let block_size = 5;
        let n_embd = 4;
        let model = create_dummy_model_for_generation(vocab_size, block_size, n_embd);

        let input_ids = vec![1u32, 2, 3];
        let max_len = 3;
        let eos_id = 99;

        let result = generate(
            &model,
            input_ids.clone(),
            max_len,
            eos_id,
            SamplingMethod::Greedy,
            None,
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), input_ids);
    }

    #[test]
    fn test_generate_input_longer_than_max_length() {
        let vocab_size = 10;
        let block_size = 5;
        let n_embd = 4;
        let model = create_dummy_model_for_generation(vocab_size, block_size, n_embd);

        let input_ids = vec![1u32, 2, 3, 4];
        let max_len = 3;
        let eos_id = 99;

        let result = generate(
            &model,
            input_ids.clone(),
            max_len,
            eos_id,
            SamplingMethod::Greedy,
            None,
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn test_generate_empty_input_error() {
        let vocab_size = 10;
        let block_size = 5;
        let n_embd = 4;
        let model = create_dummy_model_for_generation(vocab_size, block_size, n_embd);
        let input_ids: Vec<u32> = vec![];
        let result = generate(&model, input_ids, 5, 0, SamplingMethod::Greedy, None);
        assert!(matches!(result, Err(TextGeneratorError::InvalidInput(_))));
    }

    #[test]
    fn test_generate_max_length_zero() {
        let vocab_size = 10;
        let block_size = 5;
        let n_embd = 4;
        let model = create_dummy_model_for_generation(vocab_size, block_size, n_embd);
        let input_ids = vec![1u32];
        let max_len = 0;
        let eos_id = 99;

        let result = generate(
            &model,
            input_ids,
            max_len,
            eos_id,
            SamplingMethod::Greedy,
            None,
        );
        assert!(
            matches!(result, Err(TextGeneratorError::InvalidInput(msg)) if msg.contains("Max length must be greater than 0"))
        );
    }

    #[test]
    fn test_generate_max_length_one_from_prompt_of_one() {
        let vocab_size = 10;
        let block_size = 5;
        let n_embd = 4;
        let model = create_dummy_model_for_generation(vocab_size, block_size, n_embd);

        let input_ids = vec![1u32];
        let max_len = 1;
        let eos_id = 99;

        let result = generate(
            &model,
            input_ids.clone(),
            max_len,
            eos_id,
            SamplingMethod::Greedy,
            None,
        );
        assert!(result.is_ok(), "Generation failed: {:?}", result.err());
        let generated_sequence = result.unwrap();
        assert_eq!(
            generated_sequence, input_ids,
            "Should return prompt if max_length is same as prompt length"
        );
        assert_eq!(generated_sequence.len(), 1);
    }

    #[test]
    fn test_generate_max_length_one_from_empty_prompt_is_error() {
        let vocab_size = 10;
        let block_size = 5;
        let n_embd = 4;
        let model = create_dummy_model_for_generation(vocab_size, block_size, n_embd);
        let input_ids: Vec<u32> = vec![];
        let max_len = 1;
        let eos_id = 99;
        let result = generate(
            &model,
            input_ids,
            max_len,
            eos_id,
            SamplingMethod::Greedy,
            None,
        );
        assert!(matches!(result, Err(TextGeneratorError::InvalidInput(_))));
    }
}
