use std::fs::File;
use std::io::{self, Read, Cursor};
use byteorder::{BigEndian, ReadBytesExt};
use crate::tensor::GpuTensor;

const IMAGE_MAGIC: u32 = 2051;
const LABEL_MAGIC: u32 = 2049;

pub struct MnistData {
    pub train_images: Vec<f32>,
    pub train_labels: Vec<f32>,  // f32 for network
    pub test_images: Vec<f32>,
    pub test_labels: Vec<f32>,   // f32 for network
    pub image_dim: (usize, usize),
}

impl MnistData {
    pub fn load() -> io::Result<Self> {
        let train_images = Self::load_images("data/train-images-idx3-ubyte")?;
        let train_labels = Self::load_labels("data/train-labels-idx1-ubyte")?;
        let test_images = Self::load_images("data/t10k-images-idx3-ubyte")?;
        let test_labels = Self::load_labels("data/t10k-labels-idx1-ubyte")?;

        // Normalize images to [0, 1]
        let normalized_train_images: Vec<f32> = train_images.iter().map(|&x| x as f32 / 255.0).collect();
        let normalized_test_images: Vec<f32> = test_images.iter().map(|&x| x as f32 / 255.0).collect();

        // One-hot encode labels
        let train_labels_f32 = Self::labels_to_one_hot(&train_labels, 10);
        let test_labels_f32 = Self::labels_to_one_hot(&test_labels, 10);

        Ok(Self {
            train_images: normalized_train_images,
            train_labels: train_labels_f32,
            test_images: normalized_test_images,
            test_labels: test_labels_f32,
            image_dim: (28, 28),
        })
    }

    fn labels_to_one_hot(labels: &[u8], num_classes: usize) -> Vec<f32> {
        let mut one_hot = Vec::with_capacity(labels.len() * num_classes);
        for &label in labels {
            for i in 0..num_classes {
                one_hot.push(if i == label as usize { 1.0 } else { 0.0 });
            }
        }
        one_hot
    }

    fn load_images(path: &str) -> io::Result<Vec<u8>> {
        let mut file = File::open(path)?;
        let magic = file.read_u32::<BigEndian>()?;
        if magic != IMAGE_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid image file magic"));
        }

        let num_images = file.read_u32::<BigEndian>()? as usize;
        let rows = file.read_u32::<BigEndian>()? as usize;
        let cols = file.read_u32::<BigEndian>()? as usize;

        let mut buf = vec![0u8; num_images * rows * cols];
        file.read_exact(&mut buf)?;
        Ok(buf)
    }

    fn load_labels(path: &str) -> io::Result<Vec<u8>> {
        let mut file = File::open(path)?;
        let magic = file.read_u32::<BigEndian>()?;
        if magic != LABEL_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid label file magic"));
        }

        let num_labels = file.read_u32::<BigEndian>()? as usize;
        let mut buf = vec![0u8; num_labels];
        file.read_exact(&mut buf)?;
        Ok(buf)
    }

    pub fn get_batch(&self, batch_size: usize, batch_index: usize, train: bool) -> (Vec<f32>, Vec<f32>) {
        let (images, labels) = if train {
            (&self.train_images, &self.train_labels)
        } else {
            (&self.test_images, &self.test_labels)
        };

        let total_samples = labels.len() / 10;
        let start = (batch_index * batch_size) % total_samples;
        let end = std::cmp::min(start + batch_size, total_samples);

        let mut batch_images = Vec::with_capacity(batch_size * 784);
        let mut batch_labels = Vec::with_capacity(batch_size * 10);

        for i in start..end {
            let img_start = i * 784;
            batch_images.extend_from_slice(&images[img_start..img_start + 784]);

            let label_start = i * 10;
            batch_labels.extend_from_slice(&labels[label_start..label_start + 10]);
        }

        // Repeat last sample if batch is incomplete
        while batch_images.len() < batch_size * 784 {
            let i = end - 1;
            let img_start = i * 784;
            batch_images.extend_from_slice(&images[img_start..img_start + 784]);

            let label_start = i * 10;
            batch_labels.extend_from_slice(&labels[label_start..label_start + 10]);
        }

        (batch_images, batch_labels)
    }

    pub fn create_gpu_batch(
        &self,
        batch_size: usize,
        batch_index: usize,
        train: bool,
        device: &wgpu::Device
    ) -> (GpuTensor, GpuTensor) {
        let (images, labels) = self.get_batch(batch_size, batch_index, train);
        (
            GpuTensor::from_data(&images, device),
            GpuTensor::from_data(&labels, device)
        )
    }

    pub fn get_label_indices(&self, batch_size: usize, batch_index: usize, train: bool) -> Vec<u8> {
        let labels = if train { &self.train_labels } else { &self.test_labels };
        let total_samples = labels.len() / 10;
        let start = (batch_index * batch_size) % total_samples;
        let end = std::cmp::min(start + batch_size, total_samples);

        let mut indices = Vec::new();
        for i in start..end {
            let label_start = i * 10;
            let label_slice = &labels[label_start..label_start + 10];
            let max_index = label_slice.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            indices.push(max_index as u8);
        }

        indices
    }
}
