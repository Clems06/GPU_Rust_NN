use crate::tensor::{GpuTensor};
use rand;
use wgpu::ComputePipelineDescriptor;
use wgpu::util::DeviceExt;

fn create_pipeline(
    device: &wgpu::Device,
    shader_src: &str,
    layout: &wgpu::PipelineLayout,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(
        wgpu::ShaderModuleDescriptor {
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
            label: None,
        },
    );

    device.create_compute_pipeline(
        &wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(layout),
            module: &shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default()
        },
    )
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MatVecParams {
    batch_size: u32,
    layer_size: u32,
    prev_layer_size: u32,
    _pad: u32, // padding to 16 bytes
}

enum NNLayer {
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
    fn new(device: &wgpu::Device, size: u32, prev_size: u32, batch_size: u32) -> Self {
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

    fn activation_values(&self) -> &GpuTensor {
        match self {
            NNLayer::FullyConnectedLayer {activation_values, ..} 
            => {
                &activation_values
            }
        }
    }

    fn forward_pass(&self, device: &wgpu::Device, queue: &wgpu::Queue, input: &GpuTensor) -> &GpuTensor {

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

    fn backpropagate(
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

                //println!("Bias Gradient {:?}", delta_biases.read_to_cpu(device, queue));
                //println!("weight Gradient {:?}", delta_biases.read_to_cpu(device, queue));
                //println!("Del Z {:?}", del_z.read_to_cpu(device, queue));
                //println!("Weights {:?}", weights.read_to_cpu(device, queue));

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

    fn apply_gradient( &self, device: &wgpu::Device, queue: &wgpu::Queue) {

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

pub struct Network {
    layers: Vec<NNLayer>,
    batch_size: u32,
    pub device: wgpu::Device,
    queue: wgpu::Queue,
    loss_pipeline: wgpu::ComputePipeline,
}

impl Network {
    pub fn new(batch_size: u32, topology: &[u32]) -> anyhow::Result<Self>{
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }))?;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            }))?;


        let mut layers = Vec::new();
        for i in 1..topology.len() {
            layers.push(NNLayer::new(&device, topology[i as usize], topology[(i-1) as usize], batch_size));
        }

        let loss_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[ wgpu::BindGroupLayoutEntry { binding: 0, 
                                                        visibility: wgpu::ShaderStages::COMPUTE,
                                                        ty: wgpu::BindingType::Buffer {
                                                            ty: wgpu::BufferBindingType::Storage { read_only: (false) },
                                                            has_dynamic_offset: false,
                                                            min_binding_size: None,
                                                        },
                                                        count: None,},
                            wgpu::BindGroupLayoutEntry { binding: 1, 
                                                        visibility: wgpu::ShaderStages::COMPUTE,
                                                        ty: wgpu::BindingType::Buffer {
                                                            ty: wgpu::BufferBindingType::Storage { read_only: (true) },
                                                            has_dynamic_offset: false,
                                                            min_binding_size: None,
                                                        },
                                                        count: None,}],
                label: None,});
        
        let loss_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                                                                    label: None,
                                                                    bind_group_layouts: &[&loss_bind_group_layout],
                                                                    immediate_size: 0,});

        
        let loss_pipeline = create_pipeline(&device, include_str!("shaders/loss.wgsl"), &loss_layout);
        
        Ok(Self{layers, batch_size, device, queue, loss_pipeline})
    }

    pub fn forward_pass(&self, input: &GpuTensor) -> GpuTensor {
        let mut value = input;
        for layer in &self.layers {
            value = layer.forward_pass(&self.device, &self.queue, value);
        }

        value.clone()

    }

    pub fn backpropagation(&self, gradient: GpuTensor, input: &GpuTensor) {
        let mut value = gradient;
        for i in (1..self.layers.len()).rev() {
            value = self.layers[i].backpropagate(&self.device, &self.queue, &value, self.layers[i-1].activation_values());
            self.layers[i].apply_gradient(&self.device, &self.queue);
        }

        self.layers[0].backpropagate(&self.device, &self.queue, &value, input);
        self.layers[0].apply_gradient(&self.device, &self.queue);
    }


    pub fn train(&self, input: &GpuTensor, outputs: &GpuTensor) {
        let result = self.forward_pass(input);

        //println!("Result {:?}", result.read_to_cpu(&self.device, &self.queue));

        let bind_group_layout = self.loss_pipeline.get_bind_group_layout(0);

        let bind_group = self.device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: result.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: outputs.buffer.as_entire_binding(),
                    }
                ],
                label: Some("FullyConnectedLayer backpropagation activation pass"),
            },
        );

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        {
            let mut pass: wgpu::ComputePass<'_> = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.loss_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                ((result.len + 255) / 256) as u32,
                1,
                1,
            );
        }
        
        self.queue.submit(Some(encoder.finish()));

        //println!("MSE {:?}", result.read_to_cpu(&self.device, &self.queue));

        self.backpropagation(result, input);

    }

    pub fn test(&self, input: &GpuTensor) -> Vec<f32> {
        let output = self.forward_pass(input);
        output.read_to_cpu(&self.device, &self.queue)
    }

    pub fn layer_value(&self, index: usize) -> (Vec<f32>, Vec<f32>) {
        match &self.layers[index] {
            NNLayer::FullyConnectedLayer {
                weights,
                biases,
                ..
            } => {
                (weights.read_to_cpu(&self.device, &self.queue), biases.read_to_cpu(&self.device, &self.queue))
            }
        }
    }

}