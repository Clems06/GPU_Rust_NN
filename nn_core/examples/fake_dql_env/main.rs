use rand::Rng;

// adjust this import to your actual crate layout
use nn_core::{deep_q_learning::deeq_q_network::DeepQNetwork, general::activation::ActivationType::LeakyReLU}; 

const STATE_DIM: u32 = 4;
const ACTION_DIM: u32 = 2;

fn main() {
    println!("Starting pure Rust training test...");

    let mut net = DeepQNetwork::new(
        64,      
        &[STATE_DIM, 128, 64, ACTION_DIM],
        &[LeakyReLU(0.1), LeakyReLU(0.1), LeakyReLU(0.1), LeakyReLU(0.1)],
    );

    let mut rng = rand::rng();

    for episode in 1..1000 {
        println!("Episode: {}", episode);

        let mut episode_reward = 0.0;

        for _step in 0..200 {
            // fake state
            let state: Vec<f32> = (0..STATE_DIM)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();

            // random action
            let action: u32 = rng.random_range(0..ACTION_DIM as u32);

            // fake next state
            let next_state: Vec<f32> = (0..STATE_DIM)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();

            let reward: f32 = rng.random_range(-1.0..1.0);
            let done = rng.random_bool(0.05);

            episode_reward += reward;

            net.push_transition(
                state,
                action,
                reward,
                done,
                next_state,
            );

            if done {
                break;
            }
        }

        println!("Reward: {}", episode_reward);

        if net.buffer_len() >= net.batch_size() {
            net.train();
        }
    }

    println!("Done.");
}
