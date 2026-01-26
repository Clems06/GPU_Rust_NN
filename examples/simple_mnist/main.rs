#![allow(dead_code, unused)]

use NeuralNetwork::{activation::{self, ActivationType}, network, tensor};
mod mnist;
mod download_mnist;
use std::time::{Instant, Duration};


#[tokio::main]
async fn main() {
    // Download MNIST if needed
    println!("Downloading MNIST data...");
    if let Err(e) = download_mnist::download_mnist().await {
        eprintln!("Failed to download MNIST: {}", e);
        return;
    }
    
    // Load MNIST data
    println!("Loading MNIST data...");
    let mnist_data = match mnist::MnistData::load() {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to load MNIST: {}", e);
            return;
        }
    };
    
    println!("MNIST loaded: {} training samples, {} test samples", 
             mnist_data.train_labels.len() / 10, 
             mnist_data.test_labels.len() / 10);
    
    // Create network for MNIST (784 inputs, 10 outputs)
    let batch_size = 32; 
    let topology = [784, 128, 64, 10];
    let activations = [ActivationType::LeakyReLU(0.1), ActivationType::LeakyReLU(0.1), ActivationType::LeakyReLU(0.1), ActivationType::LeakyReLU(0.1)];
    
    println!("Creating network with topology: {:?}", topology);
    let network = match network::Network::new(batch_size, &topology, &activations) {
        Ok(net) => net,
        Err(e) => {
            eprintln!("Failed to create network: {}", e);
            return;
        }
    };
    
    // Training parameters
    let epochs = 200;
    let batches_per_epoch = (mnist_data.train_labels.len() / 10) / batch_size as usize;
    
    println!("Starting training for {} epochs ({} batches per epoch)...", 
             epochs, batches_per_epoch);
    
    let mut best_accuracy = 0.0;
    
    for epoch in 0..epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0;
        
        // Training loop
        for batch_idx in 0..batches_per_epoch {
            let (inputs_gpu, labels_gpu) = mnist_data.create_gpu_batch(
                batch_size as usize,
                batch_idx,
                true,
                &network.device
            );
            
            network.train(&inputs_gpu, &labels_gpu);
            
            if batch_idx % 100 == 0 {
                println!("  Epoch {}, Batch {}/{}", 
                         epoch + 1, batch_idx, batches_per_epoch);

                
                /*let output = network.test(&inputs_gpu);
                println!("{:?}", output);*/
            }
        }
        
        // Test accuracy after epoch
        println!("  Testing accuracy...");
        let test_batches = 50; // Test on a subset for speed
        let mut correct = 0;
        let mut total = 0;
        
        for test_batch in 0..test_batches {
            let (test_inputs_gpu, test_labels_gpu) = mnist_data.create_gpu_batch(
                batch_size as usize,
                test_batch,
                false,
                &network.device
            );
            
            let predictions = network.test(&test_inputs_gpu);
            
            // Get actual labels for this batch
            let label_indices = mnist_data.get_label_indices(
                batch_size as usize,
                test_batch,
                false
            );
            
            // Check predictions
            for i in 0..batch_size as usize {
                let mut max_val = -1.0;
                let mut predicted_class = 0;
                
                for j in 0..10 {
                    let val = predictions[i * 10 + j];
                    if val > max_val {
                        max_val = val;
                        predicted_class = j;
                    }
                }
                
                if predicted_class == label_indices[i] as usize {
                    correct += 1;
                }
                total += 1;
            }
        }
        
        let accuracy = correct as f32 / total as f32;
        if accuracy > best_accuracy {
            best_accuracy = accuracy;
        }
        
        let epoch_duration = epoch_start.elapsed();
        println!("Epoch {} completed in {:?}, Test Accuracy: {:.2}% (Best: {:.2}%)",
                 epoch + 1, epoch_duration, accuracy * 100.0, best_accuracy * 100.0);
        
        // Early stopping if accuracy is good
        if accuracy > 0.95 && epoch >= 3 {
            println!("Good accuracy achieved, stopping early.");
            break;
        }
    }
    
    // Final evaluation
    println!("\nFinal evaluation on 1000 test samples:");
    let mut correct = 0;
    let test_samples = 1000;
    
    for i in 0..(test_samples / batch_size as usize) {
        let (test_inputs_gpu, _) = mnist_data.create_gpu_batch(
            batch_size as usize,
            i,
            false,
            &network.device
        );
        
        let predictions = network.test(&test_inputs_gpu);
        let label_indices = mnist_data.get_label_indices(
            batch_size as usize,
            i,
            false
        );
        
        for j in 0..batch_size as usize {
            if j >= label_indices.len() {
                break;
            }
            
            let mut max_val = -1.0;
            let mut predicted_class = 0;
            
            for k in 0..10 {
                let val = predictions[j * 10 + k];
                if val > max_val {
                    max_val = val;
                    predicted_class = k;
                }
            }
            
            if predicted_class == label_indices[j] as usize {
                correct += 1;
            }
        }
    }
    
    println!("Final Accuracy: {}/{} = {:.2}%", 
             correct, test_samples, (correct as f32 / test_samples as f32) * 100.0);
    
    // Show some predictions
    println!("\nSample predictions:");
    for i in 0..5 {
        let (input_gpu, _) = mnist_data.create_gpu_batch(1, i, false, &network.device);
        let output = network.test(&input_gpu);
        
        let mut max_prob = -1.0;
        let mut predicted_class = 0;
        for (j, &prob) in output.iter().enumerate() {
            if prob > max_prob {
                max_prob = prob;
                predicted_class = j;
            }
        }
        
        let actual_label = mnist_data.get_label_indices(1, i, false)[0];
        println!("Sample {}: Predicted {}, Actual {} - Output value: {:.3}",
                 i, predicted_class, actual_label, max_prob);
    }
}