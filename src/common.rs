use crate::accelerator::{CpuTensor, Device, Module, Tensor};
use std::error::Error;

#[derive(Debug, Clone)]
pub struct LayerNorm {
    gamma: CpuTensor<f32>,
    beta: CpuTensor<f32>,
    epsilon: f32,
    device: Device,
}

impl LayerNorm {
    pub fn new(
        gamma_data: &[f32],
        gamma_shape: &[usize],
        beta_data: &[f32],
        beta_shape: &[usize],
        epsilon: f32,
    ) -> Result<Self, Box<dyn Error>> {
        let gamma = CpuTensor::from_data_and_shape(gamma_data, gamma_shape, Device::CPU)?;
        let beta = CpuTensor::from_data_and_shape(beta_data, beta_shape, Device::CPU)?;
        Ok(Self {
            gamma,
            beta,
            epsilon,
            device: Device::CPU,
        })
    }
}

impl Module for LayerNorm {
    type Input = CpuTensor<f32>;
    type Output = CpuTensor<f32>;

    fn forward(&self, input: Self::Input) -> Result<Self::Output, Box<dyn Error>> {
        if input.device() != self.device {
            return Err(format!(
                "LayerNorm is on {:?} but input is on {:?}",
                self.device,
                input.device()
            )
            .into());
        }
        if self.device != Device::CPU {
            // This check is somewhat redundant for LayerNorm if gamma/beta are CpuTensor,
            // but good practice if it could hold other device-specific resources.
            return Err(format!(
                "LayerNorm on {:?} is not supported for forward pass, only CPU.",
                self.device
            )
            .into());
        }

        let x_slice = input.as_slice()?;
        let gamma_slice = self.gamma.as_slice()?;
        let beta_slice = self.beta.as_slice()?;

        let input_shape = input.shape();
        let last_dim_size = *input_shape
            .last()
            .ok_or("Input tensor has no dimensions")?;

        if gamma_slice.len() != last_dim_size || beta_slice.len() != last_dim_size {
            return Err(format!(
                "Gamma/Beta dimensions ({}, {}) do not match input's last dimension ({})",
                gamma_slice.len(),
                beta_slice.len(),
                last_dim_size
            )
            .into());
        }

        let mut output_data = Vec::with_capacity(x_slice.len());

        for chunk in x_slice.chunks_exact(last_dim_size) {
            let sum: f32 = chunk.iter().sum();
            let mean = sum / (last_dim_size as f32);

            let variance_sum: f32 = chunk.iter().map(|val| (val - mean).powi(2)).sum();
            let variance = variance_sum / (last_dim_size as f32);
            let inv_std_dev = 1.0 / (variance + self.epsilon).sqrt();

            for (i, val) in chunk.iter().enumerate() {
                let normalized_val = (val - mean) * inv_std_dev;
                output_data.push(gamma_slice[i] * normalized_val + beta_slice[i]);
            }
        }

        CpuTensor::from_data_and_shape(&output_data, input_shape, self.device.clone())
    }

    fn to_device(&mut self, device: Device) -> Result<(), Box<dyn Error>> {
        if self.device == device {
            return Ok(());
        }
        // Since gamma and beta are CpuTensors, they can only be on CPU.
        // Thus, LayerNorm can only be on CPU.
        if device != Device::CPU {
            return Err(format!(
                "LayerNorm with CpuTensor weights cannot be moved to {:?}. Only CPU is supported.",
                device
            )
            .into());
        }
        
        // Conceptually "move" by ensuring they are on the target device.
        // For CpuTensor, this only works if the target is CPU.
        self.gamma = self.gamma.to_device(device.clone())?;
        self.beta = self.beta.to_device(device.clone())?;
        self.device = device;
        Ok(())
    }

    fn current_device(&self) -> Device {
        self.device.clone()
    }
}

pub type ModelKVCache = Vec<Vec<f32>>;
