use crate::tensor::{GpuTensor};
use crate::layer::NNLayer;
use crate::utils::create_pipeline;
use crate::activation::ActivationType;
use std::sync::Arc;
use crate::pipeline::PipelineCache;

pub struct Network {
    layers: Vec<NNLayer>,
    batch_size: u32,
    pub device: Arc<wgpu::Device>,
    queue: wgpu::Queue,
    loss_pipeline: wgpu::ComputePipeline,
    pipeline_cache: PipelineCache,
}

impl Network {
    pub fn new(batch_size: u32, topology: &[u32], activations: &[ActivationType]) -> anyhow::Result<Self>{
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
            let activation = activations.get(i - 1).copied().unwrap_or(ActivationType::ReLU);
            layers.push(NNLayer::new(&device, topology[i as usize], topology[(i-1) as usize], batch_size, activation,));
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
        
        let loss_pipeline = create_pipeline(&device, include_str!("shaders/loss.wgsl"), &loss_bind_group_layout);
        
        
        let device_arc = Arc::new(device);
        let pipeline_cache = PipelineCache::new(device_arc.clone());

        Ok(Self{layers, batch_size, device: device_arc, queue, loss_pipeline, pipeline_cache})
    }

    pub fn forward_pass(&self, input: &GpuTensor) -> GpuTensor {
        let mut value = input;
        for layer in &self.layers {
            value = layer.forward_pass(&self.device, &self.queue, value, &self.pipeline_cache);
        }

        value.clone()

    }

    pub fn backpropagation(&self, gradient: GpuTensor, input: &GpuTensor) {
        let mut value = gradient;
        for i in (1..self.layers.len()).rev() {
            value = self.layers[i].backpropagate(&self.device, &self.queue, &value, self.layers[i-1].activation_values(), &self.pipeline_cache);
            self.layers[i].apply_gradient(&self.device, &self.queue, &self.pipeline_cache);
        }

        self.layers[0].backpropagate(&self.device, &self.queue, &value, input, &self.pipeline_cache);
        self.layers[0].apply_gradient(&self.device, &self.queue, &self.pipeline_cache);
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