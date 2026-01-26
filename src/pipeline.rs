// file: pipelines.rs
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use wgpu;
use crate::activation::{ActivationType, ActivationParams};
use crate::utils;

pub struct PipelineCache {
    device: Arc<wgpu::Device>,
    activation_forward_pipelines: RwLock<HashMap<ActivationType, Arc<wgpu::ComputePipeline>>>,
    activation_backward_pipelines: RwLock<HashMap<ActivationType, Arc<wgpu::ComputePipeline>>>,
    multiplication_pipeline: Arc<wgpu::ComputePipeline>,
    backp_multiplication_pipeline: Arc<wgpu::ComputePipeline>,
    backp_weight_bias_pipeline: Arc<wgpu::ComputePipeline>,
    apply_gradient_pipeline: Arc<wgpu::ComputePipeline>,
}

impl PipelineCache {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        let multiplication_pipeline = Arc::new(utils::create_multiplication_pipeline(&device));
        let backp_multiplication_pipeline = Arc::new(utils::create_backp_multiplication_pipeline(&device));
        let backp_weight_bias_pipeline = Arc::new(utils::create_backp_weight_bias_pipeline(&device));
        let apply_gradient_pipeline = Arc::new(utils::create_apply_gradient_pipeline(&device));
        
        Self {
            device,
            activation_forward_pipelines: RwLock::new(HashMap::new()),
            activation_backward_pipelines: RwLock::new(HashMap::new()),
            multiplication_pipeline,
            backp_multiplication_pipeline,
            backp_weight_bias_pipeline,
            apply_gradient_pipeline,
        }
    }
    
    pub fn get_activation_forward_pipeline(
        &self, 
        activation_type: ActivationType,
    ) -> Arc<wgpu::ComputePipeline> {
        self.get_or_create_pipeline(
            activation_type,
            &activation_type.bind_group_layout(&self.device),
            &self.activation_forward_pipelines,
            |act| act.forward_shader()
        )
    }
    
    pub fn get_activation_backward_pipeline(
        &self,
        activation_type: ActivationType,
    ) -> Arc<wgpu::ComputePipeline> {
        self.get_or_create_pipeline(
            activation_type,
            &activation_type.backp_bind_group_layout(&self.device),
            &self.activation_backward_pipelines,
            |act| act.backward_shader()
        )
    }
    
    fn get_or_create_pipeline(
        &self,
        activation_type: ActivationType,
        bind_group_layout: &wgpu::BindGroupLayout,
        cache: &RwLock<HashMap<ActivationType, Arc<wgpu::ComputePipeline>>>,
        shader_fn: impl Fn(&ActivationType) -> &'static str,
    ) -> Arc<wgpu::ComputePipeline> {
        // Check cache first
        {
            let cache_read = cache.read().unwrap();
            if let Some(pipeline) = cache_read.get(&activation_type) {
                return pipeline.clone();
            }
        }
        
        // Create new pipeline
        let shader_source = shader_fn(&activation_type);
        let pipeline = Arc::new(create_pipeline(
            &self.device,
            shader_source,
            bind_group_layout,
        ));
        
        // Insert into cache
        {
            let mut cache_write = cache.write().unwrap();
            cache_write.insert(activation_type, pipeline.clone());
        }
        
        pipeline
    }
    
    // Getters for other pipelines
    pub fn multiplication_pipeline(&self) -> &Arc<wgpu::ComputePipeline> {
        &self.multiplication_pipeline
    }
    
    pub fn backp_multiplication_pipeline(&self) -> &Arc<wgpu::ComputePipeline> {
        &self.backp_multiplication_pipeline
    }
    
    pub fn backp_weight_bias_pipeline(&self) -> &Arc<wgpu::ComputePipeline> {
        &self.backp_weight_bias_pipeline
    }
    
    pub fn apply_gradient_pipeline(&self) -> &Arc<wgpu::ComputePipeline> {
        &self.apply_gradient_pipeline
    }
}

// Helper to create pipeline with layout
fn create_pipeline(
    device: &wgpu::Device,
    shader_source: &str,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Activation Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[bind_group_layout],
        immediate_size: 0,
    });
    
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("main"),
        cache: None,
        compilation_options: Default::default()
    })
}

