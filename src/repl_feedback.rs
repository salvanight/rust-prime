use serde::{Serialize, Deserialize};
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, ErrorKind};
use std::path::{Path, PathBuf};
// ExperienceEntry is defined in this file, so direct usage is fine.

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ExperienceEntry {
    pub prompt_tokens: Vec<u32>,      // Tokens that formed the context for generation
    pub generated_token_id: u32,      // The token that was generated
    pub validation_status: bool,      // true for 's' (coherent), false for 'n' (not coherent)
    pub theta_hat_at_generation: f32, // Value of theta_hat when this token was generated
}

#[derive(Debug)]
pub struct ResonanceFeedbackStore {
    pub experiences: Vec<ExperienceEntry>,
    filepath: PathBuf,
}

impl ResonanceFeedbackStore {
    pub fn new(filepath: PathBuf) -> Self {
        match Self::load(&filepath) {
            Ok(experiences) => {
                println!("Successfully loaded {} experiences from {:?}", experiences.len(), filepath);
                Self { experiences, filepath }
            }
            Err(e) if e.kind() == ErrorKind::NotFound => {
                println!("No existing feedback store found at {:?}. Starting fresh.", filepath);
                Self { experiences: Vec::new(), filepath }
            }
            Err(e) => {
                eprintln!("Warning: Could not load experiences from {:?}: {}. Starting with an empty store.", filepath, e);
                Self { experiences: Vec::new(), filepath }
            }
        }
    }

    pub fn add_experience(&mut self, entry: ExperienceEntry) {
        self.experiences.push(entry);
    }

    pub fn save(&self) -> Result<(), io::Error> {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true) // Overwrite the file if it exists
            .open(&self.filepath)?;
        
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &self.experiences)?;
        Ok(())
    }

    fn load(filepath: &Path) -> Result<Vec<ExperienceEntry>, io::Error> {
        let file = File::open(filepath)?;
        if file.metadata()?.len() == 0 {
            // File is empty, return empty vec, not an error
            return Ok(Vec::new());
        }
        // Reset reader to beginning if metadata was read
        // This is important because metadata() might advance the conceptual cursor for some OS/FS,
        // or simply because from_reader expects to start at the beginning.
        let file_for_reading = File::open(filepath)?; 
        let reader = BufReader::new(file_for_reading);
        let experiences = serde_json::from_reader(reader)?;
        Ok(experiences)
    }

    pub fn predict_initial_theta(&self, current_prompt_tokens: &[u32]) -> Option<f32> {
        let mut similar_thetas = Vec::new();

        for experience in &self.experiences {
            if !experience.validation_status {
                continue; // Only consider validated experiences
            }

            // Revised Similarity Logic:
            // An experience is considered similar if its `prompt_tokens` are a prefix of `current_prompt_tokens`.
            // This implies the experience is a more general context from which the current prompt extends.
            // We only consider experiences that are not empty, and current prompt is not empty, for prefix matching to be meaningful.
            // Or if both are empty.
            let mut is_similar = false;
            if experience.prompt_tokens.is_empty() && current_prompt_tokens.is_empty() {
                // If both are empty, consider them similar for theta prediction.
                is_similar = true;
            } else if !experience.prompt_tokens.is_empty() && 
                      current_prompt_tokens.starts_with(&experience.prompt_tokens) {
                // Experience's prompt is a prefix of the current prompt.
                is_similar = true;
            }
            // Note: The case where current_prompt_tokens is empty but experience.prompt_tokens is not,
            // is not considered similar by this logic (experience.prompt_tokens cannot be a prefix of an empty current_prompt).
            // This seems reasonable: an empty current prompt has no specific history to match against non-empty experiences.

            if is_similar {
                similar_thetas.push(experience.theta_hat_at_generation);
            }
        }

        if similar_thetas.is_empty() {
            None
        } else {
            let sum: f32 = similar_thetas.iter().sum();
            Some(sum / similar_thetas.len() as f32)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::NamedTempFile; // Using tempfile for robust test file handling

    // Helper to create a named temp file for tests
    fn temp_filepath() -> PathBuf {
        NamedTempFile::new().unwrap().into_temp_path().to_path_buf()
    }

    #[test]
    fn test_new_store_no_existing_file() {
        let filepath = temp_filepath();
        // Ensure file does not exist before test
        fs::remove_file(&filepath).ok(); 

        let store = ResonanceFeedbackStore::new(filepath.clone());
        assert!(store.experiences.is_empty());
        assert_eq!(store.filepath, filepath);

        // Clean up by attempting to remove the file if it was created (it shouldn't be by `new` if not found)
        fs::remove_file(filepath).ok();
    }

    #[test]
    fn test_add_and_save_experience() {
        let filepath = temp_filepath();
        fs::remove_file(&filepath).ok(); 

        let mut store = ResonanceFeedbackStore::new(filepath.clone());
        let experience1 = ExperienceEntry {
            prompt_tokens: vec![1, 2],
            generated_token_id: 3,
            validation_status: true,
            theta_hat_at_generation: 0.5,
        };
        let experience2 = ExperienceEntry {
            prompt_tokens: vec![4, 5],
            generated_token_id: 6,
            validation_status: false,
            theta_hat_at_generation: 0.6,
        };

        store.add_experience(experience1.clone());
        store.add_experience(experience2.clone());
        assert_eq!(store.experiences.len(), 2);

        let save_result = store.save();
        assert!(save_result.is_ok(), "Save failed: {:?}", save_result.err());
        
        // Verify content by loading it back
        let loaded_experiences = ResonanceFeedbackStore::load(&filepath).unwrap();
        assert_eq!(loaded_experiences.len(), 2);
        assert_eq!(loaded_experiences[0].generated_token_id, experience1.generated_token_id);
        assert_eq!(loaded_experiences[1].generated_token_id, experience2.generated_token_id);

        fs::remove_file(filepath).ok();
    }

    #[test]
    fn test_load_existing_file() {
        let filepath = temp_filepath();
        fs::remove_file(&filepath).ok(); 

        let experiences_to_save = vec![
            ExperienceEntry {
                prompt_tokens: vec![10, 20],
                generated_token_id: 30,
                validation_status: true,
                theta_hat_at_generation: 0.7,
            }
        ];
        
        let file = OpenOptions::new().write(true).create(true).open(&filepath).unwrap();
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &experiences_to_save).unwrap();

        // Now use ResonanceFeedbackStore::new, which should load this file
        let store = ResonanceFeedbackStore::new(filepath.clone());
        assert_eq!(store.experiences.len(), 1);
        assert_eq!(store.experiences[0].generated_token_id, 30);

        fs::remove_file(filepath).ok();
    }

    #[test]
    fn test_load_empty_file() {
        let filepath = temp_filepath();
        // Create an empty file
        File::create(&filepath).unwrap(); 

        let experiences = ResonanceFeedbackStore::load(&filepath).unwrap();
        assert!(experiences.is_empty(), "Loading an empty file should result in an empty Vec.");

        // Also test with ResonanceFeedbackStore::new
        let store = ResonanceFeedbackStore::new(filepath.clone());
        assert!(store.experiences.is_empty(), "`new` should also handle empty file correctly.");
        
        fs::remove_file(filepath).ok();
    }
    
    #[test]
    fn test_save_overwrite_existing() {
        let filepath = temp_filepath();
        fs::remove_file(&filepath).ok();

        // First save
        let mut store1 = ResonanceFeedbackStore::new(filepath.clone());
        store1.add_experience(ExperienceEntry {
            prompt_tokens: vec![1], generated_token_id: 2, validation_status: true, theta_hat_at_generation: 0.1
        });
        store1.save().unwrap();

        // Second save (should overwrite)
        let mut store2 = ResonanceFeedbackStore::new(filepath.clone()); // This will load store1's data
        assert_eq!(store2.experiences.len(), 1); // Verify it loaded
        store2.add_experience(ExperienceEntry { // Add a new one, total 2 in memory
            prompt_tokens: vec![3], generated_token_id: 4, validation_status: false, theta_hat_at_generation: 0.2
        });
        // Modify store2's experiences directly for a clean overwrite check
        store2.experiences = vec![ExperienceEntry { // This is what store2 will save
            prompt_tokens: vec![100], generated_token_id: 200, validation_status: true, theta_hat_at_generation: 0.9
        }];
        store2.save().unwrap(); // This save should TRUNCATE and write only this new single entry

        // Load and verify
        let loaded_store = ResonanceFeedbackStore::new(filepath.clone());
        assert_eq!(loaded_store.experiences.len(), 1, "Store should only contain the overwritten data.");
        assert_eq!(loaded_store.experiences[0].generated_token_id, 200);

        fs::remove_file(filepath).ok();
    }

    #[test]
    fn test_predict_initial_theta() {
        let store_path = temp_filepath(); // Helper to get a temp file path
        let mut store = ResonanceFeedbackStore::new(store_path.clone());

        // Test Case 1: No experiences
        assert_eq!(store.predict_initial_theta(&[1, 2, 3]), None);

        // Add some experiences
        store.add_experience(ExperienceEntry {
            prompt_tokens: vec![1, 2], generated_token_id: 3, validation_status: true, theta_hat_at_generation: 0.5
        });
        store.add_experience(ExperienceEntry {
            prompt_tokens: vec![1, 2], generated_token_id: 4, validation_status: true, theta_hat_at_generation: 0.6
        });
        store.add_experience(ExperienceEntry { // Different prefix
            prompt_tokens: vec![1, 3], generated_token_id: 5, validation_status: true, theta_hat_at_generation: 0.7
        });
        store.add_experience(ExperienceEntry { // Same prefix as first two, but not validated
            prompt_tokens: vec![1, 2], generated_token_id: 6, validation_status: false, theta_hat_at_generation: 0.8
        });
        store.add_experience(ExperienceEntry { // Longer prefix, current is prefix of this
            prompt_tokens: vec![1, 2, 3, 4], generated_token_id: 7, validation_status: true, theta_hat_at_generation: 0.9
        });
        store.add_experience(ExperienceEntry { // Empty prompt in experience, validated
            prompt_tokens: vec![], generated_token_id: 8, validation_status: true, theta_hat_at_generation: 0.4
        });


        // Test Case 2: No similar experiences (non-matching prefixes)
        assert_eq!(store.predict_initial_theta(&[4, 5, 6]), None);
        
        // Test Case 3: One similar validated experience (due to current prompt being prefix of experience's prompt)
        // Current logic: experience.prompt_tokens must be a prefix of current_prompt_tokens
        // So, for [1,2,3,4,5] current prompt, experience [1,2,3,4] is similar.
        // The experience { prompt_tokens: vec![1, 2, 3, 4], ..., theta_hat_at_generation: 0.9 }
        // is NOT similar to current_prompt_tokens: &[1, 2, 3] because its prompt is not a prefix of [1,2,3].
        // Let's test with current_prompt_tokens: &[1, 2, 3, 4, 5]
        // Experience [1,2] -> theta 0.5, 0.6. Avg = 0.55
        // Experience [1,2,3,4] -> theta 0.9
        // All are prefixes of [1,2,3,4,5]
        let current_prompt_long = [1, 2, 3, 4, 5];
        let expected_thetas_long_prompt = vec![0.5, 0.6, 0.9, 0.4]; // 0.4 from empty prompt exp
        let expected_avg_long_prompt: f32 = expected_thetas_long_prompt.iter().sum::<f32>() / expected_thetas_long_prompt.len() as f32;
        assert_eq!(store.predict_initial_theta(&current_prompt_long), Some(expected_avg_long_prompt));


        // Test Case 4: Multiple similar validated experiences (current prompt: [1, 2, 3])
        // Similar experiences:
        // - prompt_tokens: vec![1, 2] (theta 0.5)
        // - prompt_tokens: vec![1, 2] (theta 0.6)
        // - prompt_tokens: vec![] (theta 0.4)
        // Not similar:
        // - prompt_tokens: vec![1, 3] (not a prefix)
        // - prompt_tokens: vec![1, 2, 3, 4] (not a prefix of [1,2,3])
        let current_prompt123 = [1, 2, 3];
        let relevant_thetas1 = vec![0.5, 0.6, 0.4]; // from exp [1,2], [1,2], and []
        let expected_avg1: f32 = relevant_thetas1.iter().sum::<f32>() / relevant_thetas1.len() as f32;
        assert_eq!(store.predict_initial_theta(&current_prompt123), Some(expected_avg1));

        // Test Case 5 (covered by above): Mix of similar validated and non-validated experiences
        // The non-validated { prompt_tokens: vec![1, 2], ..., theta_hat_at_generation: 0.8 } was ignored.

        // Test Case 6 (covered by above): Mix of similar and non-similar experiences
        // The { prompt_tokens: vec![1, 3], ... } was ignored for current_prompt123.

        // Test Case 7: Empty current prompt and empty experience prompt
        // Experience: { prompt_tokens: vec![], ..., theta_hat_at_generation: 0.4 }
        assert_eq!(store.predict_initial_theta(&[]), Some(0.4));
        
        // Test Case 8: Empty current prompt, non-empty experience prompt (not similar by current logic)
        // Already covered by predict_initial_theta(&[]) only matching empty experience prompts.
        // If we add another non-empty experience:
        let mut store_for_empty_curr = ResonanceFeedbackStore::new(temp_filepath()); // fresh store
        store_for_empty_curr.add_experience(ExperienceEntry{prompt_tokens:vec![1], validation_status:true, theta_hat_at_generation: 0.8, generated_token_id:0});
        store_for_empty_curr.add_experience(ExperienceEntry{prompt_tokens:vec![], validation_status:true, theta_hat_at_generation: 0.4, generated_token_id:0});
        assert_eq!(store_for_empty_curr.predict_initial_theta(&[]), Some(0.4));


        // Test Case 9: Non-empty current prompt, experience with empty prompt is similar
        // Current prompt: [7, 8]. Experience: { prompt_tokens: vec![], ..., theta_hat_at_generation: 0.4 }
        // This is considered similar by the logic `current_prompt_tokens.starts_with(&experience.prompt_tokens)`
        // because any non-empty slice starts with an empty slice.
        assert_eq!(store.predict_initial_theta(&[7, 8]), Some(0.4)); // Only the empty prompt experience matches

        // Test with a current prompt that is shorter than some experiences, but experience is not prefix
        // current_prompt: [1]. Experiences: [1,2] (0.5, 0.6), [1,2,3,4] (0.9), [] (0.4)
        // Only [] is a prefix of [1]. So, only 0.4.
        assert_eq!(store.predict_initial_theta(&[1]), Some(0.4));
        
        // Test with current_prompt that is an exact match to an experience's prompt
        // current_prompt: [1,2]. Experiences: [1,2] (0.5, 0.6), [] (0.4)
        // Both [1,2] are prefixes of [1,2]. Empty [] is also a prefix of [1,2].
        // So, 0.5, 0.6, 0.4. Avg = (0.5+0.6+0.4)/3 = 1.5/3 = 0.5
        let current_prompt12 = [1,2];
        let relevant_thetas2 = vec![0.5, 0.6, 0.4];
        let expected_avg2: f32 = relevant_thetas2.iter().sum::<f32>() / relevant_thetas2.len() as f32;
        assert_eq!(store.predict_initial_theta(&current_prompt12), Some(expected_avg2));

        // Clean up the temp file for the main store
        fs::remove_file(store_path).ok();
        // temp_filepath() for store_for_empty_curr cleans itself up when it goes out of scope
    }
}
