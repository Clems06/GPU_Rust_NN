use super::utils::create_pipeline;

#[derive(PartialEq, Clone)]
pub enum LossType {
    MSE,
    CrossEntropy,
}

impl LossType {
    pub fn get_pipeline(&self, device: &wgpu::Device) -> wgpu::ComputePipeline{
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

        match self {
            Self::MSE => {
                create_pipeline(device, include_str!("shaders/loss/mse.wgsl"), &loss_bind_group_layout)

            }
            Self::CrossEntropy => {
                create_pipeline(device, include_str!("shaders/loss/cross_entropy_softmax.wgsl"), &loss_bind_group_layout)
            }
        }
    }
}