use wgpu::util::DeviceExt;

#[derive(Clone)]
pub struct GpuTensor {
    pub buffer: wgpu::Buffer,
    pub len: usize,
}

impl GpuTensor {
    pub fn from_data(data: &[f32], device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE  
                     | wgpu::BufferUsages::COPY_DST
                     | wgpu::BufferUsages::COPY_SRC,
            },
        );

        GpuTensor {
            buffer,
            len: data.len(),
        }
    }

    pub fn zeros(amount: u32, device: &wgpu::Device) -> Self {
        let zeros = vec![0.0f32; amount as usize];
        GpuTensor::from_data(&zeros, device)
    }

    pub fn read_to_cpu(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<f32> {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read staging buffer"),
            size: (self.len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy GPU buffer → staging
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(
            &self.buffer,
            0,
            &staging,
            0,
            (self.len * std::mem::size_of::<f32>()) as u64,
        );
        queue.submit(Some(encoder.finish()));

        // Map & read
        let buffer_slice = staging.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        let data = buffer_slice.get_mapped_range();
        let vec: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        vec
    }
}
