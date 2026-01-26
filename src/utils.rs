use wgpu::BindGroupLayout;

pub fn create_pipeline(
    device: &wgpu::Device,
    shader_source: &str,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
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

pub fn create_multiplication_bind_group_layout(device: &wgpu::Device) -> BindGroupLayout{
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

    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                label: None,})
}

pub fn create_multiplication_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
    create_pipeline(device, include_str!("shaders/matvec.wgsl"), &create_multiplication_bind_group_layout(device))
}

pub fn create_activation_bind_group_layout(device: &wgpu::Device) -> BindGroupLayout{
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

    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                label: None,})
}

pub fn create_backp_multiplication_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
    create_pipeline(device, include_str!("shaders/matvec_transposed.wgsl"), &create_activation_bind_group_layout(device))
}

pub fn create_back_weight_bias_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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

    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                label: None,})
}

pub fn create_backp_weight_bias_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
    create_pipeline(device, include_str!("shaders/weight_bias_grad.wgsl"), &create_back_weight_bias_bind_group_layout(device))
}

pub fn create_apply_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let write_storage_binding_type = wgpu::BindingType::Buffer {
                                                        ty: wgpu::BufferBindingType::Storage { read_only: (false) },
                                                        has_dynamic_offset: false,
                                                        min_binding_size: None,
                                                    };

    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[ wgpu::BindGroupLayoutEntry { binding: 0, 
                                                        visibility: wgpu::ShaderStages::COMPUTE,
                                                        ty: write_storage_binding_type,
                                                        count: None,},
                            wgpu::BindGroupLayoutEntry { binding: 1, 
                                                        visibility: wgpu::ShaderStages::COMPUTE,
                                                        ty: write_storage_binding_type,
                                                        count: None,},],
                label: None,})

}

pub fn create_apply_gradient_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
    create_pipeline(device, include_str!("shaders/apply_gradient.wgsl"), &create_apply_bind_group_layout(device))

}