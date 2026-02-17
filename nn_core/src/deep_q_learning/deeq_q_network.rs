use rand::{Rng, random};
use rand::{seq::IteratorRandom};
use wgpu::ComputePipeline;
use wgpu::util::DeviceExt;
use std::sync::Arc;

use crate::general::{network::Network, tensor::GpuTensor, activation::ActivationType, loss::LossType::MSE, utils::create_pipeline};
use std::collections::VecDeque;

pub fn argmax<T: PartialOrd + Copy>(v: &[T], up_to: usize) -> Option<usize> {
    if v.is_empty() {
        return None;
    }

    let mut max_idx = 0;
    let mut max_val = v[0];

    for (i, &val) in v.iter().enumerate().skip(1) {
        if i > up_to {
            break;
        }
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }

    Some(max_idx)
}

fn pack_bits_chunks(bits: &[bool]) -> Vec<u32> {
    bits.chunks(32).map(|chunk| {
        let mut word = 0u32;
        for (j, &b) in chunk.iter().enumerate() {
            if b {
                word |= 1u32 << (j as u32);
            }
        }
        word
    }).collect()
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    batch_size: u32,
    layer_size: u32,
    _pad0: u32,
    _pad1: u32,
}


pub struct DeepQNetwork {
    target_network: Network,
    network: Network,
    alpha: f32,
    gamma: f32,
    max_buffer: usize,  
    pub epsilon: f32,   
    batch_size: u32,
    pub replay_buffer: VecDeque<(Vec<f32>, u32, f32, bool, Vec<f32>)>,
    target_q_pipeline: ComputePipeline
}

impl DeepQNetwork {
    pub fn new(batch_size: u32, topology: &[u32], activations: &[ActivationType]) -> Self {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })).expect("Failed to request adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        })).expect("Failed to request device");
        let device = Arc::new(device);


        let network = Network::new( device.clone(), queue.clone(), batch_size, topology, activations, MSE).expect("Cannot build network");
        let target_network = Network::new(device, queue, batch_size, topology, activations, MSE).expect("Cannot build target network");
        network.copy_weights_to(&target_network, &network.device, &network.queue);

        let bind_group_layout = network.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                                                                                ty: wgpu::BufferBindingType::Storage { read_only: (false) },
                                                                                has_dynamic_offset: false,
                                                                                min_binding_size: None,
                                                                            },
                                                                        count: None,},
                                            wgpu::BindGroupLayoutEntry { binding: 2, 
                                                                        visibility: wgpu::ShaderStages::COMPUTE,
                                                                        ty: wgpu::BindingType::Buffer {
                                                                                ty: wgpu::BufferBindingType::Storage { read_only: (true) },
                                                                                has_dynamic_offset: false,
                                                                                min_binding_size: None,
                                                                            },
                                                                        count: None,},
                                            wgpu::BindGroupLayoutEntry { binding: 3, 
                                                                        visibility: wgpu::ShaderStages::COMPUTE,
                                                                        ty: wgpu::BindingType::Buffer {
                                                                                ty: wgpu::BufferBindingType::Storage { read_only: (true) },
                                                                                has_dynamic_offset: false,
                                                                                min_binding_size: None,
                                                                            },
                                                                        count: None,},
                                            wgpu::BindGroupLayoutEntry { binding: 4, 
                                                                        visibility: wgpu::ShaderStages::COMPUTE,
                                                                        ty: wgpu::BindingType::Buffer {
                                                                                ty: wgpu::BufferBindingType::Storage { read_only: (true) },
                                                                                has_dynamic_offset: false,
                                                                                min_binding_size: None,
                                                                            },
                                                                        count: None,},
                                            wgpu::BindGroupLayoutEntry { binding: 5, 
                                                                        visibility: wgpu::ShaderStages::COMPUTE,
                                                                        ty: wgpu::BindingType::Buffer {
                                                                            ty: wgpu::BufferBindingType::Uniform,
                                                                            has_dynamic_offset: false,
                                                                            min_binding_size: None,
                                                                        },
                                                                        count: None,}],
                                label: None,});

        let target_q_pipeline = create_pipeline(&network.device, include_str!("target_q_pipeline.wgsl"), &bind_group_layout);
        let max_buffer = 1000;
        DeepQNetwork { target_network, network, alpha: 0.1, gamma: 0.99, epsilon: 0.1, max_buffer, batch_size, replay_buffer: VecDeque::with_capacity(max_buffer), target_q_pipeline }
    }

    pub fn choose_action(&self, inputs: Vec<f32>) -> u32 {
        let last_layer_size = *self.network.topology.last().unwrap();
        if random::<f32>() < self.epsilon {
            rand::rng().random_range(0..last_layer_size)
        } else {
            let inputs_tensor = GpuTensor::from_data(&inputs, self.network.device.as_ref());
            argmax(&self.network.get_output(&inputs_tensor), (last_layer_size-1) as usize).unwrap() as u32
        }

    }

    pub fn choose_best_action(&self, inputs: Vec<f32>) -> u32 {
        let last_layer_size = *self.network.topology.last().unwrap();

        let inputs_tensor = GpuTensor::from_data(&inputs, self.network.device.as_ref());
        let predictions = self.network.get_output(&inputs_tensor);
        let mut max_val = predictions[0];
        let mut predicted_class = 0;
        
        for j in 1..last_layer_size {
            let val = predictions[j as usize];
            if val > max_val {
                max_val = val;
                predicted_class = j;
            }
        }

        predicted_class
    }

    pub fn train(&mut self) {
        if (self.replay_buffer.len() as u32) < self.batch_size {
             return;
        }

        let num_iter = 10;
        for i in 1..num_iter{
            let sample: Vec<_> = self.replay_buffer.iter()
        .choose_multiple(&mut rand::rng(), self.batch_size as usize);
            
            let mut total_input_vector: Vec<f32> = Vec::new();
            let mut total_action_vector = Vec::new();
            let mut total_finished_vector = Vec::new();
            let mut total_observation_vector: Vec<f32> = Vec::new();
            let mut total_reward_vector = Vec::new();
            for (pre_state, action, reward, finished, observation) in sample {
                total_input_vector.extend(pre_state);
                total_observation_vector.extend(observation);
                total_action_vector.push(*action);
                total_finished_vector.push(*finished);
                total_reward_vector.push(*reward);
            }
            let input_tensor = GpuTensor::from_data(&total_input_vector, &self.network.device);
            let observation_tensor = GpuTensor::from_data(&total_observation_vector, &self.network.device);
            let action_tensor = GpuTensor::from_data(&total_action_vector, &self.network.device);
            let reward_tensor = GpuTensor::from_data(&total_reward_vector, &self.network.device);
            let finished_tensor = GpuTensor::from_data(&pack_bits_chunks(&total_finished_vector), &self.network.device);

            let current_q_values = self.network.forward_pass(&input_tensor);
            let target_q_values = self.target_network.forward_pass(&observation_tensor);
            
            let params_buffer = self.network.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("MatVec Params"),
                contents: bytemuck::bytes_of(&Params{batch_size: self.batch_size, layer_size: *self.network.topology.last().unwrap(), _pad0: 0, _pad1: 0}),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                },
            );

            let bind_group_layout= self.target_q_pipeline.get_bind_group_layout(0);
            let bind_group = self.network.device.create_bind_group(
                &wgpu::BindGroupDescriptor {
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: current_q_values.buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: target_q_values.buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: finished_tensor.buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: reward_tensor.buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: action_tensor.buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: params_buffer.as_entire_binding(),
                        }
                    ],
                    label: Some("DeeQLearning target_q_values_pass"),
                },
            );

            let mut encoder =
                self.network.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&self.target_q_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(
                    ((self.batch_size + 255) / 256) as u32,
                    1,
                    1,
                );
            }

            self.network.queue.submit(Some(encoder.finish())); 
            //let _ = self.network.device.poll(wgpu::PollType::wait_indefinitely());

            self.network.backpropagation(current_q_values, &input_tensor);



        }
        

        
    }

        pub fn push_transition(
        &mut self,
        state: Vec<f32>,
        action: u32,
        reward: f32,
        done: bool,
        next_state: Vec<f32>,
    ) {
        if self.replay_buffer.len() == self.max_buffer {
            self.replay_buffer.pop_front();
        }
        self.replay_buffer.push_back((state, action, reward, done, next_state));
    }

    pub fn buffer_len(&self) -> usize {
        self.replay_buffer.len()
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size as usize
    }

    pub fn update_target(&mut self) {
        self.network.copy_weights_to(&self.target_network, &self.network.device, &self.network.queue);
    }

}