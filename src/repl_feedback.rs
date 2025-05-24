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
}
