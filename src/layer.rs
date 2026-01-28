// file: layer.rs
use crate::activation::{ActivationType, ActivationParams};
use crate::pipeline::PipelineCache;
use crate::tensor::GpuTensor;
use crate::utils;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MatVecParams {
    batch_size: u32,
    layer_size: u32,
    prev_layer_size: u32,
    _pad: u32, // padding to 16 bytes
}

pub enum NNLayer {
    FullyConnectedLayer {
        size: u32,
        prev_size: u32,
        batch_size: u32,
        activation_type: ActivationType,
        params_buffer: wgpu::Buffer,
        activation_params_buffer: wgpu::Buffer, // For activation-specific params
        
        // Tensors
        z_values: GpuTensor,
        del_z: GpuTensor,
        activation_values: GpuTensor,
        weights: GpuTensor,
        biases: GpuTensor,
        delta_weights: GpuTensor,
        delta_biases: GpuTensor,
        
        // Bind group layouts
        multiplication_bind_group_layout: wgpu::BindGroupLayout,
        activation_bind_group_layout: wgpu::BindGroupLayout,
        backp_activation_bind_group_layout: wgpu::BindGroupLayout,
    }
}

impl NNLayer {
    pub fn new(
        device: &wgpu::Device,
        size: u32,
        prev_size: u32,
        batch_size: u32,
        activation_type: ActivationType, // Each layer can have different activation!
    ) -> Self {
        // Create bind group layouts
        let multiplication_bind_group_layout = utils::create_multiplication_bind_group_layout(device);
        let activation_bind_group_layout = utils::create_activation_bind_group_layout(device);
        let backp_activation_bind_group_layout = activation_type.backp_bind_group_layout(device); 
        
        // Create uniform buffers
        let params = MatVecParams {batch_size, layer_size: size, prev_layer_size: prev_size, _pad: 0};
        let params_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("MatVec Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        
        // Create activation-specific params buffer
        let activation_params = ActivationParams::new(&activation_type);
        let activation_params_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Activation Params"),
                contents: bytemuck::bytes_of(&activation_params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        
        // Initialize tensors...
        let z_values = GpuTensor::zeros(size * batch_size, device);
        let del_z = GpuTensor::zeros(size * batch_size, device);
        let activation_values = GpuTensor::zeros(size * batch_size, device);
        
        let biases_data: Vec<f32> = (0..size).map(|_| rand::random::<f32>()/100. - 0.05).collect();
        let biases = GpuTensor::from_data(&biases_data, device);
        
        let weights_data: Vec<f32> = (0..size * prev_size).map(|_| rand::random::<f32>()/100. - 0.05).collect();
        let weights = GpuTensor::from_data(&weights_data, device);
        
        let delta_weights = GpuTensor::zeros(size * prev_size, device);
        let delta_biases = GpuTensor::zeros(size, device);
        
        NNLayer::FullyConnectedLayer {
            size,
            prev_size,
            batch_size,
            activation_type,
            params_buffer,
            activation_params_buffer,
            z_values,
            del_z,
            activation_values,
            weights,
            biases,
            delta_weights,
            delta_biases,
            multiplication_bind_group_layout,
            activation_bind_group_layout,
            backp_activation_bind_group_layout,
        }
    }
    
    pub fn activation_type(&self) -> ActivationType {
        match self {
            NNLayer::FullyConnectedLayer { activation_type, .. } => *activation_type,
        }
    }

    pub fn activation_values(&self) -> &GpuTensor {
        match self {
            NNLayer::FullyConnectedLayer {activation_values, ..} 
            => {
                &activation_values
            }
        }
    }
    
    pub fn forward_pass(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &GpuTensor,
        pipeline_cache: &PipelineCache,
    ) -> &GpuTensor {
        match self {
            NNLayer::FullyConnectedLayer {
                size,
                prev_size,
                batch_size,
                activation_type,
                params_buffer,
                activation_params_buffer,
                multiplication_bind_group_layout,
                activation_bind_group_layout,
                z_values,
                activation_values,
                weights,
                biases,
                ..
            } => {
                // 1. Matrix multiplication (shared pipeline)
                let multiplication_pipeline = pipeline_cache.multiplication_pipeline();
                let multiplication_bind_group = device.create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        layout: &multiplication_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: weights.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: input.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: z_values.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: params_buffer.as_entire_binding(),
                            }
                        ],
                        label: Some("FullyConnectedLayer multiplication pass"),
                    },
                );

                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    pass.set_pipeline(&multiplication_pipeline);
                    pass.set_bind_group(0, &multiplication_bind_group, &[]);
                    pass.dispatch_workgroups(
                        ((z_values.len + 255) / 256) as u32,
                        1,
                        1,
                    );
                }

                queue.submit(Some(encoder.finish()));
                
                // 2. Activation (activation-specific pipeline)

                let activation_pipeline = pipeline_cache.get_activation_forward_pipeline(
                    *activation_type,
                );
                
                let activation_bind_group = device.create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        layout: activation_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: z_values.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: biases.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: activation_values.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: params_buffer.as_entire_binding(),
                            },
                        ],
                        label: Some("Activation pass"),
                    },
                );
                
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    pass.set_pipeline(&activation_pipeline);
                    pass.set_bind_group(0, &activation_bind_group, &[]);
                    pass.dispatch_workgroups(
                        ((z_values.len + 255) / 256) as u32,
                        1,
                        1,
                    );
                }
                queue.submit(Some(encoder.finish()));
                

                activation_values
            }
        }
    }
    
    pub fn backpropagate(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        delta_activation: &GpuTensor,
        prev_input: &GpuTensor,
        pipeline_cache: &PipelineCache,
    ) -> GpuTensor {
        match self {
            NNLayer::FullyConnectedLayer {
                activation_type,
                prev_size,
                batch_size,
                weights,
                backp_activation_bind_group_layout,
                activation_params_buffer,
                del_z,
                z_values,
                params_buffer,
                delta_weights,
                delta_biases,
                ..
            } => {

                // Activation backward pass (activation-specific)
                let backp_activation_pipeline = pipeline_cache.get_activation_backward_pipeline(*activation_type);
                
                let backp_bind_group = device.create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        layout: backp_activation_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: delta_activation.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: z_values.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: del_z.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: params_buffer.as_entire_binding(),
                            },
                        ],
                        label: Some("FullyConnectedLayer backpropagation activation pass"),
                    },
                );
                
                
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

                {
                    let mut pass: wgpu::ComputePass<'_> = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    pass.set_pipeline(&backp_activation_pipeline);
                    pass.set_bind_group(0, &backp_bind_group, &[]);
                    pass.dispatch_workgroups(
                        ((del_z.len + 255) / 256) as u32,
                        1,
                        1,
                    );
                }
                
                queue.submit(Some(encoder.finish()));


                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

                let back_p_weight_and_bias_pipeline = pipeline_cache.backp_weight_bias_pipeline();
                let bind_group_layout = back_p_weight_and_bias_pipeline.get_bind_group_layout(0);

                let bind_group = device.create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        layout: &bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: prev_input.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: del_z.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: delta_weights.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: delta_biases.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: params_buffer.as_entire_binding(),
                            },
                        ],
                        label: Some("FullyConnectedLayer backpropagation weights and biases pass"),
                    },
                );

                // compute sizes (z_values.len = layer_size * batch_size)
                let layer_size_u32 = (z_values.len as u32) / batch_size;
                let prev_size_u32 = *prev_size; // prev_size is already a u32 from match binding
                
                let wg_x = (layer_size_u32 + 15) / 16;
                let wg_y = (prev_size_u32 + 15) / 16;

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    pass.set_pipeline(&back_p_weight_and_bias_pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups(wg_x, wg_y, 1);
                }

                queue.submit(Some(encoder.finish()));

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

                let prev_delta_activations = GpuTensor::zeros(prev_size * batch_size, device);

                let backp_multiplication_pipeline = pipeline_cache.backp_multiplication_pipeline();
                let bind_group_layout = backp_multiplication_pipeline.get_bind_group_layout(0);

                let bind_group = device.create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        layout: &bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: weights.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: del_z.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: prev_delta_activations.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: params_buffer.as_entire_binding(),
                            },
                        ],
                        label: Some("FullyConnectedLayer backpropagation weights and biases pass"),
                    },
                );

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    pass.set_pipeline(&backp_multiplication_pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups(
                        ((prev_input.len + 255) / 256) as u32,
                        1,
                        1,
                    );
                }

                queue.submit(Some(encoder.finish()));

                prev_delta_activations
            }
        }
    }
    
    pub fn apply_gradient( &self, device: &wgpu::Device, queue: &wgpu::Queue, pipeline_cache: &PipelineCache) {

        match self {
            NNLayer::FullyConnectedLayer {
                weights,
                delta_weights,
                delta_biases,
                biases,
                ..
            } => {

                //println!("Bias Gradient {:?}", delta_biases.read_to_cpu(device, queue));

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());


                let apply_gradient_pipeline = pipeline_cache.apply_gradient_pipeline();
                let bind_group_layout = apply_gradient_pipeline.get_bind_group_layout(0);

                let bind_group = device.create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        layout: &bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: weights.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: delta_weights.buffer.as_entire_binding(),
                            },
                        ],
                        label: Some("FullyConnectedLayer weights gradient adjustment"),
                    },
                );

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    pass.set_pipeline(&apply_gradient_pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups(
                        ((weights.len + 255) / 256) as u32,
                        1,
                        1,
                    );
                }

                queue.submit(Some(encoder.finish()));
                
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

                let bind_group = device.create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        layout: &bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: biases.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: delta_biases.buffer.as_entire_binding(),
                            },
                        ],
                        label: Some("FullyConnectedLayer bias gradient adjustment"),
                    },
                );

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    pass.set_pipeline(&apply_gradient_pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups(
                        ((biases.len + 255) / 256) as u32,
                        1,
                        1,
                    );
                }

                queue.submit(Some(encoder.finish()));
                
            }
        }
        
    }
}