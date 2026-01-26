use std::sync::Arc;
use wgpu::{self, Device};
use crate::utils::create_pipeline;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, Copy)]
pub enum ActivationType {
    ReLU,
    Sigmoid,
    Tanh,
    LeakyReLU(f32),
}


impl Hash for ActivationType {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash discriminant
        std::mem::discriminant(self).hash(state);
        
        // For variants with parameters, hash the bits of the float
        match self {
            Self::LeakyReLU(alpha) => {
                // Convert f32 to bits and hash those
                alpha.to_bits().hash(state);
            }
            _ => {}
        }
    }
}

impl PartialEq for ActivationType {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::ReLU, Self::ReLU) => true,
            (Self::Sigmoid, Self::Sigmoid) => true,
            (Self::Tanh, Self::Tanh) => true,
            (Self::LeakyReLU(a), Self::LeakyReLU(b)) => {
                // Use epsilon comparison for floats
                (a - b).abs() < f32::EPSILON * 4.0
            }
            _ => false,
        }
    }
}

impl Eq for ActivationType {}

impl ActivationType {
    pub fn forward_shader(&self) -> &'static str {
        match self {
            Self::ReLU => include_str!("shaders/activation_and_bias/relu.wgsl"),
            Self::Sigmoid => include_str!("shaders/activation_and_bias/sigmoid.wgsl"),
            Self::Tanh => include_str!("shaders/activation_and_bias/tanh.wgsl"),
            Self::LeakyReLU(_) => include_str!("shaders/activation_and_bias/leaky_relu.wgsl"),
        }
    }
    
    pub fn backward_shader(&self) -> &'static str {
        match self {
            Self::ReLU => include_str!("shaders/backp_activation/relu_backp.wgsl"),
            Self::Sigmoid => include_str!("shaders/backp_activation/sigmoid_backp.wgsl"),
            Self::Tanh => include_str!("shaders/backp_activation/tanh_backp.wgsl"),
            Self::LeakyReLU(_) => include_str!("shaders/backp_activation/leaky_relu_backp.wgsl"),
        }
    }
    
    pub fn extra_params(&self) -> Vec<f32> {
        match self {
            Self::LeakyReLU(alpha) => vec![*alpha],
            _ => vec![],
        }
    }

    pub fn bind_group_layout(&self, device: &wgpu::Device) -> wgpu::BindGroupLayout {
        let non_write_storage_binding_type = wgpu::BindingType::Buffer {
                                                            ty: wgpu::BufferBindingType::Storage { read_only: (true) },
                                                            has_dynamic_offset: false,
                                                            min_binding_size: None,
                                                        };

        let write_storage_binding_type = wgpu::BindingType::Buffer {
                                                            ty: wgpu::BufferBindingType::Storage { read_only: (false) },
                                                            has_dynamic_offset: false,
                                                            min_binding_size: None,
                                                        };

        match self {
            Self::ReLU => { device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                                    entries: &[ wgpu::BindGroupLayoutEntry { binding: 0, 
                                                                            visibility: wgpu::ShaderStages::COMPUTE,
                                                                            ty: non_write_storage_binding_type,
                                                                            count: None,},
                                                wgpu::BindGroupLayoutEntry { binding: 1, 
                                                                            visibility: wgpu::ShaderStages::COMPUTE,
                                                                            ty: non_write_storage_binding_type,
                                                                            count: None,},
                                                wgpu::BindGroupLayoutEntry { binding: 2, 
                                                                            visibility: wgpu::ShaderStages::COMPUTE,
                                                                            ty: write_storage_binding_type,
                                                                            count: None,},
                                                wgpu::BindGroupLayoutEntry { binding: 3, 
                                                                            visibility: wgpu::ShaderStages::COMPUTE,
                                                                            ty: wgpu::BindingType::Buffer {
                                                                                ty: wgpu::BufferBindingType::Uniform,
                                                                                has_dynamic_offset: false,
                                                                                min_binding_size: None,
                                                                            },
                                                                            count: None,}],
                                    label: None,}) },
            Self::Sigmoid => { device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                                    entries: &[ wgpu::BindGroupLayoutEntry { binding: 0, 
                                                                            visibility: wgpu::ShaderStages::COMPUTE,
                                                                            ty: non_write_storage_binding_type,
                                                                            count: None,},
                                                wgpu::BindGroupLayoutEntry { binding: 1, 
                                                                            visibility: wgpu::ShaderStages::COMPUTE,
                                                                            ty: non_write_storage_binding_type,
                                                                            count: None,},
                                                wgpu::BindGroupLayoutEntry { binding: 2, 
                                                                            visibility: wgpu::ShaderStages::COMPUTE,
                                                                            ty: write_storage_binding_type,
                                                                            count: None,},
                                                wgpu::BindGroupLayoutEntry { binding: 3, 
                                                                            visibility: wgpu::ShaderStages::COMPUTE,
                                                                            ty: wgpu::BindingType::Buffer {
                                                                                ty: wgpu::BufferBindingType::Uniform,
                                                                                has_dynamic_offset: false,
                                                                                min_binding_size: None,
                                                                            },
                                                                            count: None,}],
                                    label: None,}) },
            Self::Tanh => { device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                                    entries: &[ wgpu::BindGroupLayoutEntry { binding: 0, 
                                                                            visibility: wgpu::ShaderStages::COMPUTE,
                                                                            ty: non_write_storage_binding_type,
                                                                            count: None,},
                                                wgpu::BindGroupLayoutEntry { binding: 1, 
                                                                            visibility: wgpu::ShaderStages::COMPUTE,
                                                                            ty: non_write_storage_binding_type,
                                                                            count: None,},
                                                wgpu::BindGroupLayoutEntry { binding: 2, 
                                                                            visibility: wgpu::ShaderStages::COMPUTE,
                                                                            ty: write_storage_binding_type,
                                                                            count: None,},
                                                wgpu::BindGroupLayoutEntry { binding: 3, 
                                                                            visibility: wgpu::ShaderStages::COMPUTE,
                                                                            ty: wgpu::BindingType::Buffer {
                                                                                ty: wgpu::BufferBindingType::Uniform,
                                                                                has_dynamic_offset: false,
                                                                                min_binding_size: None,
                                                                            },
                                                                            count: None,}],
                                    label: None,}) },
            Self::LeakyReLU(_) => { device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                                    entries: &[ wgpu::BindGroupLayoutEntry { binding: 0, 
                                                                            visibility: wgpu::ShaderStages::COMPUTE,
                                                                            ty: non_write_storage_binding_type,
                                                                            count: None,},
                                                wgpu::BindGroupLayoutEntry { binding: 1, 
                                                                            visibility: wgpu::ShaderStages::COMPUTE,
                                                                            ty: non_write_storage_binding_type,
                                                                            count: None,},
                                                wgpu::BindGroupLayoutEntry { binding: 2, 
                                                                            visibility: wgpu::ShaderStages::COMPUTE,
                                                                            ty: write_storage_binding_type,
                                                                            count: None,},
                                                wgpu::BindGroupLayoutEntry { binding: 3, 
                                                                            visibility: wgpu::ShaderStages::COMPUTE,
                                                                            ty: wgpu::BindingType::Buffer {
                                                                                ty: wgpu::BufferBindingType::Uniform,
                                                                                has_dynamic_offset: false,
                                                                                min_binding_size: None,
                                                                            },
                                                                            count: None,}],
                                    label: None,}) },
        }
    }

    pub fn backp_bind_group_layout(&self, device: &wgpu::Device) -> wgpu::BindGroupLayout  {
        let non_write_storage_binding_type = wgpu::BindingType::Buffer {
                                                            ty: wgpu::BufferBindingType::Storage { read_only: (true) },
                                                            has_dynamic_offset: false,
                                                            min_binding_size: None,
                                                        };

        let write_storage_binding_type = wgpu::BindingType::Buffer {
                                                            ty: wgpu::BufferBindingType::Storage { read_only: (false) },
                                                            has_dynamic_offset: false,
                                                            min_binding_size: None,
                                                        };

        match self {
            Self::ReLU => {device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                                entries: &[ wgpu::BindGroupLayoutEntry { binding: 0, 
                                                                        visibility: wgpu::ShaderStages::COMPUTE,
                                                                        ty: non_write_storage_binding_type,
                                                                        count: None,},
                                            wgpu::BindGroupLayoutEntry { binding: 1, 
                                                                        visibility: wgpu::ShaderStages::COMPUTE,
                                                                        ty: non_write_storage_binding_type,
                                                                        count: None,},
                                            wgpu::BindGroupLayoutEntry { binding: 2, 
                                                                        visibility: wgpu::ShaderStages::COMPUTE,
                                                                        ty: write_storage_binding_type,
                                                                        count: None,},
                                            wgpu::BindGroupLayoutEntry { binding: 3, 
                                                                        visibility: wgpu::ShaderStages::COMPUTE,
                                                                        ty: wgpu::BindingType::Buffer {
                                                                            ty: wgpu::BufferBindingType::Uniform,
                                                                            has_dynamic_offset: false,
                                                                            min_binding_size: None,
                                                                        },
                                                                        count: None,}],
                                label: None,})},
            Self::Sigmoid => {device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                                entries: &[ wgpu::BindGroupLayoutEntry { binding: 0, 
                                                                        visibility: wgpu::ShaderStages::COMPUTE,
                                                                        ty: non_write_storage_binding_type,
                                                                        count: None,},
                                            wgpu::BindGroupLayoutEntry { binding: 1, 
                                                                        visibility: wgpu::ShaderStages::COMPUTE,
                                                                        ty: non_write_storage_binding_type,
                                                                        count: None,},
                                            wgpu::BindGroupLayoutEntry { binding: 2, 
                                                                        visibility: wgpu::ShaderStages::COMPUTE,
                                                                        ty: write_storage_binding_type,
                                                                        count: None,},
                                            wgpu::BindGroupLayoutEntry { binding: 3, 
                                                                        visibility: wgpu::ShaderStages::COMPUTE,
                                                                        ty: wgpu::BindingType::Buffer {
                                                                            ty: wgpu::BufferBindingType::Uniform,
                                                                            has_dynamic_offset: false,
                                                                            min_binding_size: None,
                                                                        },
                                                                        count: None,}],
                                label: None,})},
            Self::Tanh => {device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                                entries: &[ wgpu::BindGroupLayoutEntry { binding: 0, 
                                                                        visibility: wgpu::ShaderStages::COMPUTE,
                                                                        ty: non_write_storage_binding_type,
                                                                        count: None,},
                                            wgpu::BindGroupLayoutEntry { binding: 1, 
                                                                        visibility: wgpu::ShaderStages::COMPUTE,
                                                                        ty: non_write_storage_binding_type,
                                                                        count: None,},
                                            wgpu::BindGroupLayoutEntry { binding: 2, 
                                                                        visibility: wgpu::ShaderStages::COMPUTE,
                                                                        ty: write_storage_binding_type,
                                                                        count: None,},
                                            wgpu::BindGroupLayoutEntry { binding: 3, 
                                                                        visibility: wgpu::ShaderStages::COMPUTE,
                                                                        ty: wgpu::BindingType::Buffer {
                                                                            ty: wgpu::BufferBindingType::Uniform,
                                                                            has_dynamic_offset: false,
                                                                            min_binding_size: None,
                                                                        },
                                                                        count: None,}],
                                label: None,})},
            Self::LeakyReLU(_) => {device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                                entries: &[ wgpu::BindGroupLayoutEntry { binding: 0, 
                                                                        visibility: wgpu::ShaderStages::COMPUTE,
                                                                        ty: non_write_storage_binding_type,
                                                                        count: None,},
                                            wgpu::BindGroupLayoutEntry { binding: 1, 
                                                                        visibility: wgpu::ShaderStages::COMPUTE,
                                                                        ty: non_write_storage_binding_type,
                                                                        count: None,},
                                            wgpu::BindGroupLayoutEntry { binding: 2, 
                                                                        visibility: wgpu::ShaderStages::COMPUTE,
                                                                        ty: write_storage_binding_type,
                                                                        count: None,},
                                            wgpu::BindGroupLayoutEntry { binding: 3, 
                                                                        visibility: wgpu::ShaderStages::COMPUTE,
                                                                        ty: wgpu::BindingType::Buffer {
                                                                            ty: wgpu::BufferBindingType::Uniform,
                                                                            has_dynamic_offset: false,
                                                                            min_binding_size: None,
                                                                        },
                                                                        count: None,}],
                                label: None,})},
        }
    }
}

// Store activation-specific parameters
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ActivationParams {
    pub alpha: f32,     // For LeakyReLU
    pub beta: f32,      // For other parameterized activations
    _pad: [f32; 2],     // Padding to 16 bytes
}

impl ActivationParams {
    pub fn new(activation_type: &ActivationType) -> Self {
        match activation_type {
            ActivationType::LeakyReLU(alpha) => Self {
                alpha: *alpha,
                beta: 0.0,
                _pad: [0.0; 2],
            },
            _ => Self {
                alpha: 0.0,
                beta: 0.0,
                _pad: [0.0; 2],
            },
        }
    }
}