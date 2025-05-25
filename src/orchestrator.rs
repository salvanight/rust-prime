// Ensure these 'use' statements are correct based on your project structure.
// If moe.rs, gating.rs, system_resources.rs are in src/, then crate::module_name is typical.
use crate::moe::Expert; // Assuming moe.rs is in src/ and contains pub trait Expert
use crate::gating::GatingLayer; // Assuming gating.rs is in src/
use crate::system_resources::SystemResources; // Assuming system_resources.rs is in src/

// For Array types if they are used in fields or method signatures directly in this file
use ndarray::{ArrayD, Array1};

#[derive(Debug)] // Added Debug derive
pub struct MoEOrchestrator {
    pub experts: Vec<Box<dyn Expert>>,
    pub system_status: SystemResources, // Will be refreshed
    pub gating_layer: GatingLayer,
    
    // Configuration for orchestration logic
    pub max_concurrent_experts_override: Option<usize>, // If Some, overrides dynamic calculation
    pub min_ram_gb_per_expert: f32,        // e.g., 0.5 GB per expert
    pub high_cpu_load_threshold: f32,      // e.g., 0.80 (80% load average for 1 min)
    pub num_experts_in_high_load: usize,   // Number of experts to use if CPU is high (e.g., 1)
    pub default_top_k_experts: usize,      // Default number of experts to try activating if no constraints hit
}

impl MoEOrchestrator {
    pub fn new(
        experts: Vec<Box<dyn Expert>>,
        gating_layer: GatingLayer,
        max_concurrent_override: Option<usize>,
        ram_per_expert_gb: f32,
        cpu_high_threshold: f32,
        experts_on_high_cpu: usize,
        default_k: usize,
    ) -> Self {
        let system_status = SystemResources::new(); // Initialize by fetching current system status
        
        Self {
            experts,
            system_status,
            gating_layer,
            max_concurrent_experts_override: max_concurrent_override,
            min_ram_gb_per_expert: ram_per_expert_gb,
            high_cpu_load_threshold: cpu_high_threshold,
            num_experts_in_high_load: experts_on_high_cpu,
            default_top_k_experts: default_k,
        }
    }

    // Other methods (determine_active_experts, forward) will be added later.

    // Determines which experts to activate and their re-normalized scores.
    // Returns a Vec of (expert_index, normalized_score).
    pub fn determine_active_experts(
        &mut self, // Takes &mut self to refresh system_status
        input_1d_features: &Array1<f32>,
        _theta_hat: f32, // theta_hat currently not used in this basic selection logic, but available for future use
    ) -> Result<Vec<(usize, f32)>, String> {
        if self.experts.is_empty() {
            return Ok(Vec::new()); // No experts to choose from
        }

        // 1. Refresh system status
        self.system_status.refresh();
        // println!("[Orchestrator] System status refreshed: CPU Load {:.2}%, RAM Avail {:.2}GB", 
        //          self.system_status.cpu_load_avg_one * 100.0, 
        //          self.system_status.ram_available_gb);


        // 2. Determine max number of experts to activate based on resources
        let mut num_to_activate = self.default_top_k_experts;

        // Override by explicit setting if present
        if let Some(max_override) = self.max_concurrent_experts_override {
            num_to_activate = max_override;
        } else {
            // Dynamic calculation based on RAM
            if self.min_ram_gb_per_expert > 0.0 { // Avoid division by zero or negative
                let max_by_ram = (self.system_status.ram_available_gb / self.min_ram_gb_per_expert).floor() as usize;
                num_to_activate = num_to_activate.min(max_by_ram);
                // println!("[Orchestrator] Max experts by RAM: {}, num_to_activate now: {}", max_by_ram, num_to_activate);
            }

            // Dynamic adjustment based on CPU load
            if self.system_status.cpu_load_avg_one > self.high_cpu_load_threshold {
                num_to_activate = num_to_activate.min(self.num_experts_in_high_load);
                // println!("[Orchestrator] High CPU load detected. num_to_activate reduced to: {}", num_to_activate);
            }
        }
        
        // Ensure at least 0 and at most total number of experts
        num_to_activate = num_to_activate.min(self.experts.len());
        if num_to_activate == 0 && !self.experts.is_empty() {
            // If constraints force 0 experts but we have experts, maybe activate at least one if possible?
            // Or is it valid to activate 0 experts if system is too constrained?
            // For now, if it becomes 0, it means no experts will be run by the current logic.
            // The MoE::forward or Orchestrator::forward should handle this (e.g. by returning input or error).
            println!("[Orchestrator] Resource constraints resulted in 0 experts to activate.");
            return Ok(Vec::new());
        }
        if num_to_activate == 0 && self.experts.is_empty() { // Should be caught by initial check
            return Ok(Vec::new());
        }


        // 3. Get gating scores
        let gating_scores = self.gating_layer.forward(input_1d_features)?;
        if gating_scores.len() != self.experts.len() {
            return Err(format!(
                "Gating scores length ({}) mismatch with number of experts ({}).",
                gating_scores.len(), self.experts.len()
            ));
        }

        // 4. Select top 'num_to_activate' experts based on gating scores
        let mut indexed_scores: Vec<(usize, f32)> = gating_scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
        indexed_scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Less));
        
        // Slice to get the actual top experts to be activated
        let selected_experts_indices_and_scores = &indexed_scores[..num_to_activate.min(indexed_scores.len())]; // Ensure slice is not out of bounds

        if selected_experts_indices_and_scores.is_empty() && num_to_activate > 0 {
            // This could happen if num_to_activate > 0 but indexed_scores is empty (e.g. all gating scores were NaN)
            return Err("No experts could be selected based on gating scores (e.g., all scores were NaN).".to_string());
        }

        // 5. Re-normalize scores of the selected experts so they sum to 1
        let total_score_of_selected: f32 = selected_experts_indices_and_scores.iter().map(|(_, score)| score).sum();

        let final_selected_experts: Vec<(usize, f32)> = if total_score_of_selected == 0.0 {
            // If sum of scores is 0 (e.g., all selected scores are 0, or only one expert with score 0)
            // Distribute weights equally among selected, or handle as error/special case.
            // For now, distribute equally if any were selected.
            if selected_experts_indices_and_scores.is_empty() {
                Vec::new()
            } else {
                let equal_weight = 1.0 / selected_experts_indices_and_scores.len() as f32;
                selected_experts_indices_and_scores.iter().map(|(idx, _)| (*idx, equal_weight)).collect()
            }
        } else {
            selected_experts_indices_and_scores.iter().map(|(idx, score)| (*idx, score / total_score_of_selected)).collect()
        };
        
        // println!("[Orchestrator] Determined active experts (index, re-normalized_score): {:?}", final_selected_experts);
        Ok(final_selected_experts)
    }

    pub fn forward(
        &mut self, // Takes &mut self to allow determine_active_experts to refresh system_status
        input_1d_features: &Array1<f32>,
        full_input_tensor: &ArrayD<f32>,
        theta_hat: f32,
    ) -> Result<ArrayD<f32>, String> {
        let active_experts_with_scores = self.determine_active_experts(input_1d_features, theta_hat)?;

        if active_experts_with_scores.is_empty() {
            // If no experts are activated (e.g., due to resource constraints or no experts defined)
            // Fallback strategy: Could return the input tensor unmodified, or a zero tensor, or an error.
            // Returning an error might be safest to indicate MoE did not run.
            // Or, could be a specific behavior like "passthrough if no experts active".
            // For now, let's return an error or a clear signal.
            // The user prompt implies fallback to "lightweight Expert::CacheResonator" if cpu load high.
            // This suggests that if determine_active_experts returns empty (e.g. due to high cpu),
            // a specific fallback expert could be chosen here. This is advanced logic.
            // For this step, if empty, return error.
            return Err("No experts were activated by the orchestrator.".to_string());
        }

        let mut final_output_accumulator: Option<ArrayD<f32>> = None;
        // This will sum to 1.0 if active_experts_with_scores is not empty, due to re-normalization.
        // let total_renormalized_score: f32 = active_experts_with_scores.iter().map(|(_, score)| score).sum();

        println!("[Orchestrator] Activating selected experts (count: {}):", active_experts_with_scores.len());

        for (expert_index, normalized_score) in &active_experts_with_scores {
            let expert = &self.experts[*expert_index]; // Get ref to expert using index
            println!(
                "  - Forwarding to Expert: '{}' with re-normalized score: {:.4}",
                expert.name(), normalized_score
            );

            match expert.forward(full_input_tensor, theta_hat) {
                Ok(expert_output) => {
                    if final_output_accumulator.is_none() {
                        final_output_accumulator = Some(ArrayD::zeros(expert_output.shape()));
                    }
                    // Ensure shapes match
                    if expert_output.shape() != final_output_accumulator.as_ref().unwrap().shape() {
                        return Err(format!(
                            "Expert '{}' output shape {:?} does not match expected shape {:?}.",
                            expert.name(), expert_output.shape(), final_output_accumulator.as_ref().unwrap().shape()
                        ));
                    }
                    // Add weighted expert output to accumulator: acc = acc + expert_output * weight
                    final_output_accumulator.as_mut().unwrap().scaled_add(*normalized_score, &expert_output);
                }
                Err(e) => return Err(format!("Expert '{}' failed during forward pass: {}", expert.name(), e)),
            }
        }
        
        // final_output_accumulator should be Some if active_experts_with_scores was not empty.
        final_output_accumulator.ok_or_else(|| "No expert outputs processed, final_output_accumulator is None.".to_string())
    }
}
