#![cfg(feature = "ndarray_backend")]
// Ensure these 'use' statements are correct based on your project structure.
// If moe.rs, gating.rs, system_resources.rs are in src/, then crate::module_name is typical.
use crate::moe::Expert; // Assuming moe.rs is in src/ and contains pub trait Expert
use crate::gating::GatingLayer; // Assuming gating.rs is in src/
use crate::system_resources::SystemResources; // Assuming system_resources.rs is in src/
use crate::cache_tier::{CacheTier, filter_experts_by_tier}; // Added for cache tier filtering

// For Array types if they are used in fields or method signatures directly in this file
use ndarray::{ArrayD, Array1};

/// Manages a collection of experts and orchestrates their execution based on gating logic,
/// system resources, and cache tier policies.
///
/// The orchestrator is responsible for selecting which experts to activate for a given input,
/// potentially adjusting its selection strategy based on real-time conditions such as
/// CPU load, RAM availability, and symbolic cache tiering rules.
///
/// In the REPL context, the `current_allowed_tiers` field is dynamically updated based
/// on system RAM and the intentionality score (θ̂) to guide expert selection.
#[derive(Debug)] // Added Debug derive
pub struct MoEOrchestrator {
    /// The collection of experts managed by this orchestrator.
    pub experts: Vec<Box<dyn Expert>>,
    /// Holds information about current system resources (CPU, RAM).
    /// This is refreshed internally during operation (e.g., in `determine_active_experts`).
    pub system_status: SystemResources, // Will be refreshed
    /// The gating layer used to score experts for selection.
    pub gating_layer: GatingLayer,
    
    // Configuration for orchestration logic
    /// If Some, overrides dynamic calculation of how many experts to run concurrently.
    pub max_concurrent_experts_override: Option<usize>, // If Some, overrides dynamic calculation
    /// Minimum RAM (in GB) deemed necessary per expert for dynamic calculation.
    pub min_ram_gb_per_expert: f32,        // e.g., 0.5 GB per expert
    /// CPU load average (e.g., 1-minute load) threshold to trigger constrained expert selection.
    pub high_cpu_load_threshold: f32,      // e.g., 0.80 (80% load average for 1 min)
    /// Number of experts to use if CPU load is high and no override is set.
    pub num_experts_in_high_load: usize,   // Number of experts to use if CPU is high (e.g., 1)
    /// Default number of top-K experts to try activating if no other constraints are hit.
    pub default_top_k_experts: usize,      // Default number of experts to try activating if no constraints hit
    /// Current list of `CacheTier`s that are allowed for expert activation.
    /// This policy can be dynamically updated (e.g., by the REPL based on system state).
    pub current_allowed_tiers: Vec<CacheTier>, 
}

impl MoEOrchestrator {
    /// Creates a new `MoEOrchestrator`.
    ///
    /// Initializes the orchestrator with a set of experts, a gating layer, and various
    /// configuration parameters that define its behavior under different system conditions.
    /// The `current_allowed_tiers` policy is initialized to allow all cache tiers by default;
    /// this can be changed dynamically during runtime (e.g., by the REPL).
    ///
    /// # Arguments
    /// * `experts`: A vector of boxed `Expert` trait objects.
    /// * `gating_layer`: The `GatingLayer` to be used for scoring experts.
    /// * `max_concurrent_override`: Optional override for the number of concurrent experts.
    /// * `ram_per_expert_gb`: Estimated RAM needed per expert, for resource calculations.
    /// * `cpu_high_threshold`: CPU load threshold for constrained mode.
    /// * `experts_on_high_cpu`: Number of experts to use in high CPU load mode.
    /// * `default_k`: Default number of top-K experts to activate.
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
            current_allowed_tiers: vec![CacheTier::L1, CacheTier::L2, CacheTier::L3, CacheTier::RAM],
        }
    }

    // Other methods (determine_active_experts, forward) will be added later.

    /// Determines which experts to activate and their re-normalized scores.
    ///
    /// This method performs several steps:
    /// 1. Refreshes the `system_status` (RAM, CPU load).
    /// 2. Calculates the maximum number of experts to activate (`num_to_activate`) based on
    ///    `default_top_k_experts`, resource constraints (RAM, CPU), and any overrides.
    /// 3. Retrieves gating scores for all experts from the `gating_layer`.
    /// 4. Filters these scored experts based on the `current_allowed_tiers` policy.
    /// 5. Selects the top `num_to_activate` experts from the tier-filtered list.
    /// 6. Re-normalizes the scores of these selected experts so they sum to 1.0.
    ///
    /// # Arguments
    /// * `self`: Takes `&mut self` to allow refreshing `system_status`.
    /// * `input_1d_features`: A 1D array of features for the gating layer (e.g., embedding of the current token).
    /// * `_theta_hat`: The current intentionality score (currently unused in selection logic but available).
    ///
    /// # Returns
    /// A `Result` containing a `Vec` of `(expert_index, normalized_score)` tuples for the
    /// selected experts, or an error string if issues occur (e.g., gating mismatch, no experts selectable).
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

        // Filter by cache tier BEFORE sorting and truncating by num_to_activate
        let tier_filtered_indexed_scores = filter_experts_by_tier(
            &indexed_scores,
            &self.experts, // self.experts is Vec<Box<dyn Expert>> which now requires ExpertTagged
            &self.current_allowed_tiers
        );

        // Sort the tier-filtered scores
        let mut sorted_tier_filtered_scores = tier_filtered_indexed_scores;
        sorted_tier_filtered_scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Less));
        
        // Slice to get the actual top experts to be activated from the sorted and filtered list
        let selected_experts_indices_and_scores = &sorted_tier_filtered_scores[..num_to_activate.min(sorted_tier_filtered_scores.len())]; // Ensure slice is not out of bounds

        if selected_experts_indices_and_scores.is_empty() && num_to_activate > 0 {
            // This could happen if num_to_activate > 0 but the filtered list is empty or all gating scores were NaN
            // Consider if a different message is needed if tier_filtered_indexed_scores was non-empty but became empty after sorting (e.g. NaNs)
            // For now, this message is general enough.
            return Err("No experts could be selected based on gating scores and tier filtering (e.g., all scores were NaN or no experts in allowed tiers).".to_string());
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

    /// Processes an input through the selected mixture of experts.
    ///
    /// This method orchestrates the end-to-end forward pass:
    /// 1. Calls `determine_active_experts` to get a list of experts to activate and their scores.
    /// 2. For each selected expert, calls its `forward` method with `full_input_tensor` and `theta_hat`.
    /// 3. Combines the outputs of the activated experts, weighted by their re-normalized scores.
    ///    The combined output is assumed to be token logits.
    /// 4. Collects details (name and cache tier) of the experts that were activated.
    ///
    /// # Arguments
    /// * `self`: Takes `&mut self` as `determine_active_experts` needs it.
    /// * `input_1d_features`: A 1D array of features, typically for the gating layer to decide
    ///   which experts to activate (e.g., current token's embedding).
    /// * `full_input_tensor`: The primary input tensor for the experts themselves
    ///   (e.g., sequence of embeddings).
    /// * `theta_hat`: The current intentionality score, passed to each activated expert.
    ///
    /// # Returns
    /// A `Result` which, on success, contains a tuple:
    ///   - `ArrayD<f32>`: The combined output tensor from the experts (assumed to be token logits).
    ///   - `Vec<(String, CacheTier)>`: A list of tuples, where each contains the name
    ///     and `CacheTier` of an expert that was activated during this forward pass.
    /// On failure, returns an error string.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moe::{MLPExpert, SymbolicExpert, Expert}; // Added Expert here
    use crate::gating::GatingLayer;
    use crate::cache_tier::CacheTier;
    use ndarray::Array1;

    // Helper to create a default orchestrator for tests
    fn setup_orchestrator(experts: Vec<Box<dyn Expert>>, num_features: usize, num_experts_in_gating: usize) -> MoEOrchestrator {
        let gating_layer = GatingLayer::new(num_features, num_experts_in_gating);
        MoEOrchestrator::new(
            experts,
            gating_layer,
            None,    // max_concurrent_experts_override
            0.5,     // min_ram_gb_per_expert
            0.8,     // high_cpu_load_threshold
            1,       // num_experts_in_high_load
            3        // default_top_k_experts
        )
    }

    #[test]
    fn test_orchestrator_tier_l1_only() {
        let experts_vec: Vec<Box<dyn Expert>> = vec![
            Box::new(SymbolicExpert::new("L1_Expert", 0.5)), // Index 0, L1
            Box::new(MLPExpert::new("L2_Expert")),          // Index 1, L2
        ];
        let num_experts = experts_vec.len();
        let mut orchestrator = setup_orchestrator(experts_vec, 3, num_experts);
        orchestrator.current_allowed_tiers = vec![CacheTier::L1];
        orchestrator.default_top_k_experts = 2; 

        let input_features = Array1::zeros(3); 
        let result = orchestrator.determine_active_experts(&input_features, 0.0);
        assert!(result.is_ok(), "Result should be Ok");
        let active_experts = result.unwrap();

        assert_eq!(active_experts.len(), 1, "Should only select one L1 expert");
        assert_eq!(active_experts[0].0, 0, "Selected expert should be L1_Expert (index 0)");
    }

    #[test]
    fn test_orchestrator_tier_l2_only() {
        let experts_vec: Vec<Box<dyn Expert>> = vec![
            Box::new(SymbolicExpert::new("L1_Expert", 0.5)), // Index 0, L1
            Box::new(MLPExpert::new("L2_Expert")),          // Index 1, L2
        ];
        let num_experts = experts_vec.len();
        let mut orchestrator = setup_orchestrator(experts_vec, 3, num_experts);
        orchestrator.current_allowed_tiers = vec![CacheTier::L2];
        orchestrator.default_top_k_experts = 2;

        let input_features = Array1::zeros(3);
        let result = orchestrator.determine_active_experts(&input_features, 0.0);
        assert!(result.is_ok(), "Result should be Ok");
        let active_experts = result.unwrap();
        
        assert_eq!(active_experts.len(), 1, "Should only select one L2 expert");
        assert_eq!(active_experts[0].0, 1, "Selected expert should be L2_Expert (index 1)");
    }

    #[test]
    fn test_orchestrator_tier_l1_and_l2_allowed() {
        let experts_vec: Vec<Box<dyn Expert>> = vec![
            Box::new(SymbolicExpert::new("L1_Expert", 0.5)), // Index 0, L1
            Box::new(MLPExpert::new("L2_Expert")),          // Index 1, L2
            Box::new(MLPExpert::new("L2_Expert_2")),        // Index 2, L2
        ];
        let num_experts = experts_vec.len();
        let mut orchestrator = setup_orchestrator(experts_vec, 3, num_experts);
        orchestrator.current_allowed_tiers = vec![CacheTier::L1, CacheTier::L2];
        orchestrator.default_top_k_experts = 3;

        let input_features = Array1::zeros(3);
        let result = orchestrator.determine_active_experts(&input_features, 0.0);
        assert!(result.is_ok(), "Result should be Ok");
        let active_experts = result.unwrap();
        
        assert_eq!(active_experts.len(), 3, "Should select all three experts");
        assert!(active_experts.iter().any(|&(idx, _)| idx == 0));
        assert!(active_experts.iter().any(|&(idx, _)| idx == 1));
        assert!(active_experts.iter().any(|&(idx, _)| idx == 2));
    }

    #[test]
    fn test_orchestrator_no_tiers_match() {
        let experts_vec: Vec<Box<dyn Expert>> = vec![
            Box::new(SymbolicExpert::new("L1_Expert", 0.5)),
            Box::new(MLPExpert::new("L2_Expert")),
        ];
        let num_experts = experts_vec.len();
        let mut orchestrator = setup_orchestrator(experts_vec, 3, num_experts);
        orchestrator.current_allowed_tiers = vec![CacheTier::L3];
        orchestrator.default_top_k_experts = 2;

        let input_features = Array1::zeros(3);
        let result = orchestrator.determine_active_experts(&input_features, 0.0);
        assert!(result.is_ok(), "Result should be Ok");
        let active_experts = result.unwrap();
        assert!(active_experts.is_empty(), "Should select no experts if no tiers match");
    }

    #[test]
    fn test_orchestrator_tier_filter_then_top_k_limit() {
        let experts_vec: Vec<Box<dyn Expert>> = vec![
            Box::new(SymbolicExpert::new("L1_Expert_1", 0.5)), // Index 0, L1
            Box::new(MLPExpert::new("L2_Expert_1")),          // Index 1, L2
            Box::new(SymbolicExpert::new("L1_Expert_2", 0.5)), // Index 2, L1
            Box::new(MLPExpert::new("L2_Expert_2")),          // Index 3, L2
        ];
        let num_experts = experts_vec.len();
        let mut orchestrator = setup_orchestrator(experts_vec, 3, num_experts);
        orchestrator.current_allowed_tiers = vec![CacheTier::L1]; 
        orchestrator.default_top_k_experts = 1; 

        let input_features = Array1::zeros(3);
        let result = orchestrator.determine_active_experts(&input_features, 0.0);
        assert!(result.is_ok(), "Result should be Ok");
        let active_experts = result.unwrap();
        
        assert_eq!(active_experts.len(), 1, "Should select only 1 expert due to top_k limit");
        let selected_expert_index = active_experts[0].0;
        assert!(selected_expert_index == 0 || selected_expert_index == 2, "Selected expert must be one of the L1 experts (index 0 or 2), got {}", selected_expert_index);
    }
}
