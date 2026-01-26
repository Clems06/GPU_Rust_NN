use rand;
use wgpu::util::DeviceExt;
use crate::tensor::GpuTensor;
use crate::utils::create_pipeline;

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
        params_buffer: wgpu::Buffer,
        activation_pipeline: wgpu::ComputePipeline,
        multiplication_pipeline: wgpu::ComputePipeline,
        z_values: GpuTensor,
        del_z: GpuTensor,
        activation_values: GpuTensor,
        weights: GpuTensor,
        biases: GpuTensor,
        delta_weights: GpuTensor,
        delta_biases: GpuTensor,
        backp_activation_pipeline: wgpu::ComputePipeline,
        backp_multiplication_pipeline: wgpu::ComputePipeline,
        back_p_weight_and_bias_pipeline: wgpu::ComputePipeline,
        apply_gradient_pipeline: wgpu::ComputePipeline,
    }
}


impl NNLayer {
    pub fn new(device: &wgpu::Device, size: u32, prev_size: u32, batch_size: u32) -> Self {
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

        let multiplication_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                label: None,});
        
        let multiplication_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                                                                    label: None,
                                                                    bind_group_layouts: &[&multiplication_bind_group_layout],
                                                                    immediate_size: 0,});
        
        let activation_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                                                                    label: None,
                                                                    bind_group_layouts: &[&multiplication_bind_group_layout],
                                                                    immediate_size: 0,
                                                                });

        let back_p_weight_and_bias_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                                                        ty: write_storage_binding_type,
                                                        count: None,},
                            wgpu::BindGroupLayoutEntry { binding: 4, 
                                                        visibility: wgpu::ShaderStages::COMPUTE,
                                                        ty: wgpu::BindingType::Buffer {
                                                            ty: wgpu::BufferBindingType::Uniform,
                                                            has_dynamic_offset: false,
                                                            min_binding_size: None,
                                                        },
                                                        count: None,}],
                label: None,});
        
        let back_p_weight_and_bias_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                                                                    label: None,
                                                                    bind_group_layouts: &[&back_p_weight_and_bias_bind_group_layout],
                                                                    immediate_size: 0,});
        
        let apply_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[ wgpu::BindGroupLayoutEntry { binding: 0, 
                                                        visibility: wgpu::ShaderStages::COMPUTE,
                                                        ty: write_storage_binding_type,
                                                        count: None,},
                            wgpu::BindGroupLayoutEntry { binding: 1, 
                                                        visibility: wgpu::ShaderStages::COMPUTE,
                                                        ty: write_storage_binding_type,
                                                        count: None,},],
                label: None,});
        
        let apply_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                                                                    label: None,
                                                                    bind_group_layouts: &[&apply_bind_group_layout],
                                                                    immediate_size: 0,});
                            


        let params = MatVecParams {batch_size, layer_size: size, prev_layer_size: prev_size, _pad: 0, };

        let params_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("MatVec Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        let multiplication_pipeline = create_pipeline(device, include_str!("shaders/matvec.wgsl"), &multiplication_pipeline_layout);
        let activation_pipeline = create_pipeline(device, include_str!("shaders/relu_and_bias.wgsl"), &activation_pipeline_layout);
        let backp_activation_pipeline = create_pipeline(device, include_str!("shaders/relu_backprop.wgsl"), &activation_pipeline_layout);
        let backp_multiplication_pipeline = create_pipeline(device, include_str!("shaders/matvec_transposed.wgsl"), &activation_pipeline_layout);
        let back_p_weight_and_bias_pipeline = create_pipeline(device, include_str!("shaders/weight_bias_grad.wgsl"), &back_p_weight_and_bias_pipeline_layout);
        let apply_gradient_pipeline = create_pipeline(device, include_str!("shaders/apply_gradient.wgsl"), &apply_pipeline_layout);
        let z_values = GpuTensor::zeros(size*batch_size, device);
        let del_z: GpuTensor = GpuTensor::zeros(size*batch_size, device);
        let activation_values = GpuTensor::zeros(size*batch_size, device);
        let biases_data: Vec<f32> =(0..size).map(|_| rand::random::<f32>()/100. - 0.05).collect();
        let biases = GpuTensor::from_data(&biases_data, device);
        let weights_data: Vec<f32> =(0..size * prev_size).map(|_| rand::random::<f32>()/100. - 0.05).collect();
        let weights = GpuTensor::from_data(&weights_data, device);
        let delta_weights = GpuTensor::zeros(size * prev_size, device);
        let delta_biases = GpuTensor::zeros(size, device);
        NNLayer::FullyConnectedLayer {size, prev_size, batch_size, params_buffer, activation_pipeline, multiplication_pipeline, z_values, del_z, activation_values, weights, biases, delta_weights, delta_biases, backp_activation_pipeline, backp_multiplication_pipeline, back_p_weight_and_bias_pipeline, apply_gradient_pipeline}
    }

    pub fn activation_values(&self) -> &GpuTensor {
        match self {
            NNLayer::FullyConnectedLayer {activation_values, ..} 
            => {
                &activation_values
            }
        }
    }

    pub fn forward_pass(&self, device: &wgpu::Device, queue: &wgpu::Queue, input: &GpuTensor) -> &GpuTensor {

        match self {
            NNLayer::FullyConnectedLayer {size, params_buffer, activation_pipeline, multiplication_pipeline, z_values, activation_values, weights, biases, ..} 
            => {
                let bind_group_layout = multiplication_pipeline.get_bind_group_layout(0);

                

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
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups(
                        ((z_values.len + 255) / 256) as u32,
                        1,
                        1,
                    );
                }

                queue.submit(Some(encoder.finish()));

                let bind_group_layout = activation_pipeline.get_bind_group_layout(0);

                let bind_group = device.create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        layout: &bind_group_layout,
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
                        label: Some("FullyConnectedLayer activation pass"),
                    },
                );

                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    pass.set_pipeline(&activation_pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups(
                        ((z_values.len + 255) / 256) as u32,
                        1,
                        1,
                    );
                }

                queue.submit(Some(encoder.finish()));

                &activation_values
            }
        }

        
    }

    pub fn backpropagate(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        delta_activation: &GpuTensor,
        prev_input: &GpuTensor,
    ) -> GpuTensor {
        match self {
            NNLayer::FullyConnectedLayer {
                prev_size,
                batch_size,
                params_buffer,
                z_values,
                weights,
                delta_weights,
                delta_biases,
                del_z,
                backp_activation_pipeline,
                backp_multiplication_pipeline,
                back_p_weight_and_bias_pipeline,
                ..
            } => {

                let bind_group_layout = backp_activation_pipeline.get_bind_group_layout(0);

                let bind_group = device.create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        layout: &bind_group_layout,
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
                            }
                        ],
                        label: Some("FullyConnectedLayer backpropagation activation pass"),
                    },
                );

                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

                {
                    let mut pass: wgpu::ComputePass<'_> = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    pass.set_pipeline(&backp_activation_pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups(
                        ((del_z.len + 255) / 256) as u32,
                        1,
                        1,
                    );
                }
                
                queue.submit(Some(encoder.finish()));

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

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

    pub fn apply_gradient( &self, device: &wgpu::Device, queue: &wgpu::Queue) {

        match self {
            NNLayer::FullyConnectedLayer {
                weights,
                delta_weights,
                delta_biases,
                biases,
                apply_gradient_pipeline,
                ..
            } => {

                //println!("Bias Gradient {:?}", delta_biases.read_to_cpu(device, queue));

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

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