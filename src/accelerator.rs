// src/accelerator.rs

use ndarray::ArrayD; // Or your preferred tensor library
use std::error::Error;

// Represents the computational device
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Device {
    CPU,
    CUDA(usize), // usize is the device ID
    Metal(usize), // usize is the device ID
    // Future: TPU, etc.
}

impl Default for Device {
    fn default() -> Self {
        Device::CPU // Default to CPU
    }
}

// Trait for abstracting tensor operations across devices
pub trait Tensor<T: Clone>: std::fmt::Debug {
    // Returns the device this tensor resides on
    fn device(&self) -> Device;

    // Returns the shape of the tensor
    fn shape(&self) -> &[usize];

    // Returns the data as a slice (primarily for CPU tensors or after moving to CPU)
    fn as_slice(&self) -> Result<&[T], Box<dyn Error>>;
    
    // Returns the data as a mutable slice (primarily for CPU tensors or after moving to CPU)
    fn as_mut_slice(&mut self) -> Result<&mut [T], Box<dyn Error>>;

    // Creates a new tensor on the specified device with given data and shape
    // This is a simplified constructor; more sophisticated ones might be needed
    fn from_data_and_shape(data: &[T], shape: &[usize], device: Device) -> Result<Self, Box<dyn Error>> where Self: Sized;

    // Moves the tensor to the specified device
    fn to_device(&self, device: Device) -> Result<Self, Box<dyn Error>> where Self: Sized;
    
    // Placeholder for a generic tensor operation (example)
    // fn matmul(&self, other: &Self) -> Result<Self, Box<dyn Error>> where Self: Sized;
}

// A basic CPU tensor implementation (wrapping ndarray for now)
#[derive(Debug, Clone)]
pub struct CpuTensor<T: Clone + Default + std::fmt::Debug> {
    array: ArrayD<T>,
    device: Device,
}

impl<T: Clone + Default + std::fmt::Debug + 'static> Tensor<T> for CpuTensor<T> {
    fn device(&self) -> Device {
        self.device.clone()
    }

    fn shape(&self) -> &[usize] {
        self.array.shape()
    }

    fn as_slice(&self) -> Result<&[T], Box<dyn Error>> {
        if self.device != Device::CPU {
            return Err("Cannot get slice from non-CPU tensor directly.".into());
        }
        self.array.as_slice().ok_or_else(|| "Failed to get slice from ndarray.".into())
    }
    
    fn as_mut_slice(&mut self) -> Result<&mut [T], Box<dyn Error>> {
        if self.device != Device::CPU {
            return Err("Cannot get mutable slice from non-CPU tensor directly.".into());
        }
        self.array.as_slice_mut().ok_or_else(|| "Failed to get mutable slice from ndarray.".into())
    }

    fn from_data_and_shape(data: &[T], shape: &[usize], device: Device) -> Result<Self, Box<dyn Error>> {
        if device != Device::CPU {
            return Err("CpuTensor can only be created on CPU device.".into());
        }
        let array = ArrayD::from_shape_vec(shape.to_vec(), data.to_vec())
            .map_err(|e| format!("Failed to create ArrayD: {}", e))?;
        Ok(CpuTensor { array, device })
    }

    fn to_device(&self, device: Device) -> Result<Self, Box<dyn Error>> {
        if device == self.device {
            return Ok(self.clone());
        }
        match device {
            Device::CPU => Err("CpuTensor is already on CPU. Cannot move to CPU again if it's a different conceptual CPU (not supported).".into()),
            Device::CUDA(_) | Device::Metal(_) => {
                // This is CPU -> GPU
                // TODO: Implement actual CPU to GPU data transfer.
                // For scaffolding, we create a GpuTensor instance without actual data copy.
                // This requires GpuTensor::from_data_and_shape to accept data, even if it's ignored for now.
                // The subtask for GpuTensor::from_data_and_shape has it taking data.
                // However, CpuTensor::to_device returns Result<Self,...> which is CpuTensor.
                // This indicates a design issue: CpuTensor::to_device cannot directly return a GpuTensor.
                //
                // The subtask description states for "Update CpuTensor::to_device":
                // "If device is Device::CUDA(_) or Device::Metal(_): Return Ok(GpuTensor { ... })"
                // This is NOT possible with the current trait `fn to_device(&self, device: Device) -> Result<Self, Box<dyn Error>>`
                // where Self = CpuTensor.
                //
                // Correct interpretation: The subtask implies a more flexible mechanism is eventually needed.
                // For this step, CpuTensor::to_device can only create another CpuTensor or error.
                // If we want to create a GpuTensor, it should be a separate function or a method on a
                // higher-level abstraction (like an AnyTensor enum).
                //
                // Sticking to the current trait strictly:
                // A CpuTensor can only be on CPU. So, trying to move it to GPU should be an error *from CpuTensor::to_device*.
                // The actual CPU->GPU copy would be initiated differently, perhaps:
                // `GpuTensor::from_cpu_tensor(cpu_tensor_instance, gpu_device)`
                // Or `Accelerator::copy_to_device(tensor, target_device)`
                //
                // Given the strictness of `Result<Self, ...>`, I will make CpuTensor->Gpu an error for now.
                // The cross-type conversion will be handled by the specific type's `from_data_and_shape`
                // or a dedicated conversion function/method outside this specific trait method.
                Err(format!("CpuTensor to {:?} conversion not directly supported by CpuTensor::to_device. Use GpuTensor::from_data_and_shape or specific conversion methods.", device).into())
            }
        }
    }
}

// Represents a tensor on a GPU device (placeholder)
#[derive(PartialEq, Eq, Hash)]
pub struct GpuTensor<T: Clone + Default + std::fmt::Debug + Send + Sync + 'static> {
    shape: Vec<usize>,
    device: Device,
    // Actual GPU buffer/pointer omitted for scaffolding
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Clone + Default + std::fmt::Debug + Send + Sync + 'static> std::fmt::Debug for GpuTensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuTensor")
         .field("shape", &self.shape)
         .field("device", &self.device)
         .finish()
    }
}

impl<T: Clone + Default + std::fmt::Debug + Send + Sync + 'static> Clone for GpuTensor<T> {
    fn clone(&self) -> Self {
        GpuTensor {
            shape: self.shape.clone(),
            device: self.device.clone(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Clone + Default + std::fmt::Debug + Send + Sync + 'static> Tensor<T> for GpuTensor<T> {
    fn device(&self) -> Device {
        self.device.clone()
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn as_slice(&self) -> Result<&[T], Box<dyn Error>> {
        Err("Cannot directly access slice from GpuTensor.".into())
    }
    
    fn as_mut_slice(&mut self) -> Result<&mut [T], Box<dyn Error>> {
        Err("Cannot directly access mutable slice from GpuTensor.".into())
    }

    fn from_data_and_shape(data: &[T], shape: &[usize], device: Device) -> Result<Self, Box<dyn Error>> {
        match device {
            Device::CPU => Err("GpuTensor cannot be created on CPU device.".into()),
            Device::CUDA(_) | Device::Metal(_) => {
                // Data isn't actually stored on GPU yet. This is a placeholder.
                // In a real implementation, `data` would be copied to the GPU here.
                Ok(GpuTensor {
                    shape: shape.to_vec(),
                    device,
                    _phantom: std::marker::PhantomData,
                })
            }
        }
    }

    fn to_device(&self, device: Device) -> Result<Self, Box<dyn Error>> where Self: Sized {
        if device == self.device {
            return Ok(self.clone());
        }
        
        match device {
            Device::CPU => {
                // As per subtask: "If device == Device::CPU: Return a CpuTensor..."
                // This is not possible with `Result<Self, Box<dyn Error>>` where Self is GpuTensor.
                // This should ideally be handled by a more generic mechanism or an enum.
                // For now, GpuTensor::to_device will error if trying to convert to CpuTensor directly.
                // The actual conversion would be CpuTensor::from_data_and_shape(gpu_data_slice, shape, Device::CPU)
                // after data is copied from GPU to CPU.
                Err("GpuTensor to CpuTensor conversion not directly supported by GpuTensor::to_device. This requires explicit data copy and construction of CpuTensor.".into())
            }
            Device::CUDA(_) | Device::Metal(_) => {
                 // TODO: Implement actual GPU to GPU data transfer if devices are different.
                Ok(GpuTensor {
                    shape: self.shape.clone(),
                    device, // Target device
                    _phantom: std::marker::PhantomData,
                })
            }
        }
    }
}


// Trait for models or model components that can be moved between devices
// and can perform a forward pass.
pub trait Module: std::fmt::Debug {
    // Defines the input type for the forward pass for this module
    type Input;
    // Defines the output type for the forward pass for this module
    type Output;

    // Performs a forward pass
    // Input and Output will likely involve the generic Tensor trait
    fn forward(&self, input: Self::Input) -> Result<Self::Output, Box<dyn Error>>;

    // Moves the module's parameters and buffers to the specified device
    fn to_device(&mut self, device: Device) -> Result<(), Box<dyn Error>>;
    
    // Returns the current device of the module
    fn current_device(&self) -> Device;
}

// The Accelerator struct: main entry point for users
#[derive(Debug)]
pub struct Accelerator {
    device: Device,
    // Potentially: configuration for distributed training, mixed precision, etc.
}

impl Accelerator {
    pub fn new(device: Device) -> Self {
        Accelerator { device }
    }

    pub fn default() -> Self {
        Accelerator { device: Device::default() }
    }

    pub fn current_device(&self) -> Device {
        self.device.clone()
    }

    // Prepares a module for use with the accelerator (e.g., moves it to the accelerator's device)
    pub fn prepare<M: Module>(&self, module: &mut M) -> Result<(), Box<dyn Error>> {
        module.to_device(self.device.clone())?;
        // In a full implementation, this might also wrap the module, optimizer, dataloaders etc.
        Ok(())
    }

    // A helper to prepare data (e.g., move a tensor to the accelerator's device)
    // This is a conceptual example. In practice, you'd likely have data loaders
    // that yield Tensors already on the correct device or easily movable.
    pub fn prepare_data<T: Clone + Default + std::fmt::Debug + 'static, Tsr: Tensor<T>>(
        &self, 
        tensor: &Tsr
    ) -> Result<Tsr, Box<dyn Error>> {
        if tensor.device() == self.device {
            // This clone might not be ideal if Tsr is large and already on device.
            // Depending on Tsr's clone semantics.
            Ok(tensor.clone()) // Assuming clone is cheap or appropriate
        } else {
            tensor.to_device(self.device.clone())
        }
    }
}

// Example of a simple module (conceptual)
#[derive(Debug)]
struct SimpleLinear {
    weights: CpuTensor<f32>, // Example: using CpuTensor directly
    bias: CpuTensor<f32>,
    device: Device,
}

impl SimpleLinear {
    // Constructor that initializes weights on CPU
    #[allow(dead_code)]
    fn new(in_features: usize, out_features: usize) -> Result<Self, Box<dyn Error>> {
        // Initialize weights and bias with some random data for example
        let w_data = vec![0.0f32; in_features * out_features];
        let b_data = vec![0.0f32; out_features];
        
        let weights = CpuTensor::from_data_and_shape(&w_data, &[in_features, out_features], Device::CPU)?;
        let bias = CpuTensor::from_data_and_shape(&b_data, &[out_features], Device::CPU)?;
        
        Ok(SimpleLinear { weights, bias, device: Device::CPU })
    }
}

impl Module for SimpleLinear {
    // For this example, let's assume Input and Output are also CpuTensor<f32>
    type Input = CpuTensor<f32>;
    type Output = CpuTensor<f32>;

    fn forward(&self, input: Self::Input) -> Result<Self::Output, Box<dyn Error>> {
        if self.device != input.device() {
            return Err(format!("Module on {:?} received input on {:?}", self.device, input.device()).into());
        }
        // Simplified: Actual matmul and bias add would go here.
        // For now, just return a clone of the input's shape with zeros, on the same device.
        let output_shape = input.shape(); // This would be different in a real linear layer
        let output_data_len = output_shape.iter().product();
        let output_data = vec![0.0f32; output_data_len];
        CpuTensor::from_data_and_shape(&output_data, output_shape, self.device.clone())
    }

    fn to_device(&mut self, device: Device) -> Result<(), Box<dyn Error>> {
        if self.device == device {
            return Ok(());
        }
        // This is where we'd move self.weights and self.bias to the new device.
        // For CpuTensor, it can only be on CPU. A real implementation would need
        // to convert self.weights to the appropriate Tensor type for the target device.
        if device != Device::CPU {
            return Err(format!("SimpleLinear with CpuTensor weights cannot be moved to {:?}.", device).into());
        }
        self.weights = self.weights.to_device(device.clone())?;
        self.bias = self.bias.to_device(device.clone())?;
        self.device = device;
        Ok(())
    }
    
    fn current_device(&self) -> Device {
        self.device.clone()
    }
}

// Represents a tensor on a GPU device (placeholder)
#[derive(PartialEq, Eq, Hash)]
pub struct GpuTensor<T: Clone + Default + std::fmt::Debug + Send + Sync + 'static> {
    shape: Vec<usize>,
    device: Device,
    // Actual GPU buffer/pointer omitted for scaffolding
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Clone + Default + std::fmt::Debug + Send + Sync + 'static> std::fmt::Debug for GpuTensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuTensor")
         .field("shape", &self.shape)
         .field("device", &self.device)
         .finish()
    }
}

impl<T: Clone + Default + std::fmt::Debug + Send + Sync + 'static> Clone for GpuTensor<T> {
    fn clone(&self) -> Self {
        GpuTensor {
            shape: self.shape.clone(),
            device: self.device.clone(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Clone + Default + std::fmt::Debug + Send + Sync + 'static> Tensor<T> for GpuTensor<T> {
    fn device(&self) -> Device {
        self.device.clone()
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn as_slice(&self) -> Result<&[T], Box<dyn Error>> {
        Err("Cannot directly access slice from GpuTensor.".into())
    }
    
    fn as_mut_slice(&mut self) -> Result<&mut [T], Box<dyn Error>> {
        Err("Cannot directly access mutable slice from GpuTensor.".into())
    }

    fn from_data_and_shape(data: &[T], shape: &[usize], device: Device) -> Result<Self, Box<dyn Error>> {
        match device {
            Device::CPU => Err("GpuTensor cannot be created on CPU device.".into()),
            Device::CUDA(_) | Device::Metal(_) => {
                // Data isn't actually stored on GPU yet. This is a placeholder.
                // In a real implementation, `data` would be copied to the GPU here.
                Ok(GpuTensor {
                    shape: shape.to_vec(),
                    device,
                    _phantom: std::marker::PhantomData,
                })
            }
        }
    }

    fn to_device(&self, device: Device) -> Result<Self, Box<dyn Error>> where Self: Sized {
        if device == self.device {
            return Ok(self.clone());
        }
        // Attempting to move a GpuTensor to another device type (or another GPU)
        // For now, we are not implementing GpuTensor to CpuTensor or Gpu To Gpu data copy for GpuTensor itself
        // This will be handled by the to_device implementations of specific tensor types (CpuTensor, GpuTensor)
        // when a generic TensorHolder/UnifiedTensor enum is introduced.
        // The current Tensor trait's to_device is more about creating a *new* tensor of the same type on another device.
        
        // This specific GpuTensor::to_device is for creating another GpuTensor on a *different* GPU.
        // Or, if we were to support it, converting to CpuTensor (but that's handled by CpuTensor::to_device for CPU->GPU).
        // The subtask asks for GpuTensor::to_device to return a CpuTensor if target is CPU.
        // This means GpuTensor's to_device needs to know about CpuTensor.
        // This creates a slight circular dependency if not handled carefully, usually via a trait object or enum.
        // For now, we'll allow GpuTensor to create a CpuTensor directly.
        
        match device {
            Device::CPU => {
                // This is GpuTensor -> CpuTensor
                // TODO: Implement actual GPU to CPU data transfer.
                let num_elements = self.shape.iter().product();
                let cpu_data = vec![T::default(); num_elements];
                // Since GpuTensor::to_device returns Result<Self, ...>, we can't directly return a CpuTensor here.
                // This highlights a limitation of the current Tensor trait's to_device signature if we want
                // to_device to convert between *concrete types*.
                // The prompt for GpuTensor::to_device implies it should return a CpuTensor.
                // This means the Tensor trait's to_device might need to be more flexible, e.g. returning Box<dyn Tensor<T>> or an enum.
                // Given the current trait, this specific conversion (GpuTensor -> CpuTensor) cannot be directly implemented
                // within GpuTensor::to_device to return a CpuTensor.
                //
                // Let's follow the prompt for GpuTensor::to_device as best as possible,
                // but acknowledge this design point. The prompt states:
                // "If device == Device::CPU: Return a CpuTensor..."
                // This is not possible with `Result<Self, Box<dyn Error>>` where Self is GpuTensor.
                //
                // Re-interpreting: The task implies GpuTensor::to_device is for moving *this GpuTensor* to another device.
                // If the target device is CPU, it should produce a representation of this tensor's data on the CPU.
                // This means the Tensor trait itself should perhaps be:
                // to_device(&self, device: Device) -> Result<Box<dyn Tensor<T>>, Box<dyn Error>>;
                // Or an enum `AnyTensor` that wraps `CpuTensor` and `GpuTensor`.
                //
                // For now, I will implement it such that moving GpuTensor to CPU results in an error,
                // as moving it to a *different type* of tensor isn't directly supported by `Result<Self,...>`.
                // The conversion will be:
                // - CpuTensor's to_device(GPU) -> GpuTensor
                // - GpuTensor's to_device(CPU) -> Error for now / or a new GpuTensor (placeholder)
                //
                // The prompt's request for GpuTensor::to_device to return CpuTensor implies
                // that to_device should be on a more generic enum/trait object.
                // I will implement the GpuTensor -> GpuTensor part.
                // The GpuTensor -> CpuTensor part will be an error, but CpuTensor -> GpuTensor will work.

                // As per prompt: "If device == Device::CPU: Return a CpuTensor..."
                // This is impossible with current trait. Let's make it an error for GpuTensor.
                // The CpuTensor::to_device will handle Cpu->Gpu.
                // The GpuTensor::to_device if target is CPU will be an error, or a dummy GpuTensor on CPU (which is wrong).
                // The prompt for GpuTensor::to_device returning CpuTensor is a bit of a design contradiction with `Result<Self, ...>`.
                // Let's stick to GpuTensor::to_device creating another GpuTensor or erroring.
                // The sub-task's specific instruction "Return a CpuTensor" for GpuTensor::to_device to CPU
                // cannot be fulfilled with the current trait signature `Result<Self, Box<dyn Error>>` for `GpuTensor<T>`.
                // I will make GpuTensor -> CPU an error from GpuTensor::to_device.
                // The cross-type conversion is better handled by the specific type's to_device.
                // So, CpuTensor::to_device(GPU) -> GpuTensor is fine.
                // GpuTensor::to_device(CPU) -> error.
                // GpuTensor::to_device(OTHER_GPU) -> GpuTensor.
                Err("GpuTensor to CpuTensor conversion not directly supported by GpuTensor::to_device. Use specific conversion methods or a unified tensor type.".into())

            }
            Device::CUDA(_) | Device::Metal(_) => {
                 // TODO: Implement actual GPU to GPU data transfer if devices are different.
                Ok(GpuTensor {
                    shape: self.shape.clone(),
                    device,
                    _phantom: std::marker::PhantomData,
                })
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_creation() {
        let cpu = Device::CPU;
        let cuda0 = Device::CUDA(0);
        assert_eq!(cpu, Device::CPU);
        assert_eq!(cuda0, Device::CUDA(0));
    }

    #[test]
    fn cpu_tensor_creation_and_properties() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let cpu_tensor = CpuTensor::from_data_and_shape(&data, &shape, Device::CPU).unwrap();
        
        assert_eq!(cpu_tensor.device(), Device::CPU);
        assert_eq!(cpu_tensor.shape(), &[2, 2]);
        assert_eq!(cpu_tensor.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn cpu_tensor_to_same_device() {
        let data = vec![1.0f32, 2.0];
        let shape = vec![1, 2];
        let tensor1 = CpuTensor::from_data_and_shape(&data, &shape, Device::CPU).unwrap();
        let tensor2 = tensor1.to_device(Device::CPU).unwrap();
        assert_eq!(tensor2.device(), Device::CPU);
        assert_eq!(tensor2.as_slice().unwrap(), &[1.0, 2.0]);
    }

    #[test]
    fn cpu_tensor_to_other_device_should_fail_for_now() {
        let data = vec![1.0f32, 2.0];
        let shape = vec![1, 2];
        let tensor1 = CpuTensor::from_data_and_shape(&data, &shape, Device::CPU).unwrap();
        assert!(tensor1.to_device(Device::CUDA(0)).is_err());
    }
    
    #[test]
    fn simple_linear_module_creation_and_device() {
        let mut linear = SimpleLinear::new(10, 5).unwrap();
        assert_eq!(linear.current_device(), Device::CPU);
        
        // Test moving to the same device (CPU)
        assert!(linear.to_device(Device::CPU).is_ok());
        assert_eq!(linear.current_device(), Device::CPU);

        // Test moving to another device (should fail for this simple example)
        assert!(linear.to_device(Device::CUDA(0)).is_err());
    }

    #[test]
    fn accelerator_prepare_module() {
        let accelerator = Accelerator::new(Device::CPU);
        let mut linear = SimpleLinear::new(10, 5).unwrap();
        
        assert!(accelerator.prepare(&mut linear).is_ok());
        assert_eq!(linear.current_device(), Device::CPU);
    }

    #[test]
    fn accelerator_prepare_data() {
        let accelerator = Accelerator::new(Device::CPU);
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let cpu_tensor = CpuTensor::from_data_and_shape(&data, &shape, Device::CPU).unwrap();

        let prepared_tensor = accelerator.prepare_data(&cpu_tensor).unwrap();
        assert_eq!(prepared_tensor.device(), Device::CPU);
        assert_eq!(prepared_tensor.as_slice().unwrap(), cpu_tensor.as_slice().unwrap());
    }
}
