use crate::tensor_engine::{Tensor, TensorError};
use crate::transformer_core::{GPT2Model, TransformerError};

/// Simple stochastic gradient descent optimizer
pub struct SGDOptimizer {
    pub learning_rate: f32,
}

impl SGDOptimizer {
    pub fn apply(&self, param: &mut [f32], grad: &[f32]) {
        for (p, g) in param.iter_mut().zip(grad.iter()) {
            *p -= self.learning_rate * g;
        }
    }
}

/// Compute cross entropy loss between `logits` and `targets`.
/// `logits` shape: [B, S, V], `targets` shape: [B, S]
pub fn cross_entropy_loss(logits: &Tensor<f32>, targets: &Tensor<u32>) -> Result<f32, TensorError> {
    let probs = logits.softmax(logits.rank() - 1)?;
    let batch = targets.shape[0];
    let seq = targets.shape[1];
    let vocab = logits.shape[2];
    let mut loss = 0.0f32;
    for b in 0..batch {
        for s in 0..seq {
            let t = targets.data[b * seq + s] as usize;
            let idx = b * seq * vocab + s * vocab + t;
            let p = probs.data[idx].max(1e-12); // avoid log(0)
            loss -= p.ln();
        }
    }
    loss /= (batch * seq) as f32;
    Ok(loss)
}

/// Compute gradient of cross entropy with respect to logits
fn cross_entropy_grad(logits: &Tensor<f32>, targets: &Tensor<u32>) -> Result<Tensor<f32>, TensorError> {
    let probs = logits.softmax(logits.rank() - 1)?;
    let mut grad = probs.data.clone();
    let batch = targets.shape[0];
    let seq = targets.shape[1];
    let vocab = logits.shape[2];
    for b in 0..batch {
        for s in 0..seq {
            let t = targets.data[b * seq + s] as usize;
            let idx = b * seq * vocab + s * vocab + t;
            grad[idx] -= 1.0;
        }
    }
    Tensor::new(grad, logits.shape.clone())
}

/// Perform a single training step updating the model's embedding matrix
pub fn train_step(
    model: &mut GPT2Model,
    input_ids: &Tensor<u32>,
    target_ids: &Tensor<u32>,
    optimizer: &SGDOptimizer,
) -> Result<f32, TransformerError> {
    // forward pass and capture hidden states
    let (logits, hidden) = model.forward_with_hidden(input_ids, None, None, None)?;
    let loss = cross_entropy_loss(&logits, target_ids)?;
    let grad_logits = cross_entropy_grad(&logits, target_ids)?;

    // compute gradient for wte
    let batch = input_ids.shape[0];
    let seq = input_ids.shape[1];
    let emb = hidden.shape[2];
    let vocab = logits.shape[2];
    let mut grad_wte = vec![0f32; vocab * emb];
    for b in 0..batch {
        for s in 0..seq {
            for v in 0..vocab {
                let gl = grad_logits.data[b * seq * vocab + s * vocab + v];
                if gl == 0.0 { continue; }
                for e in 0..emb {
                    let h = hidden.data[b * seq * emb + s * emb + e];
                    grad_wte[v * emb + e] += h * gl;
                }
            }
        }
    }

    model.apply_wte_gradient(&grad_wte, optimizer.learning_rate);
    Ok(loss)
}
