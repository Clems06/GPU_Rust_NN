import random
from PyRustNN import PyDeepQNetwork
import matplotlib.pyplot as plt

def plot_metrics(episode_rewards):
        """Plot training metrics for debugging"""
        plt.figure(figsize=(15, 10))

        # Plot episode rewards
        plt.subplot(1, 1, 1)
        plt.plot(episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')

        plt.tight_layout()
        plt.savefig(f"dqn_metrics_{len(episode_rewards)}.png")
        plt.close()

def main_loop(env, T=1000,
              episodes=100000,
              batch_size=32,
              topology=[8,64,4],
              activations=["leaky_relu","leaky_relu","leaky_relu"]):

    net = PyDeepQNetwork(topology, activations, batch_size)

    epsilon = 1.0
    min_epsilon = 0.05
    epsilon_decay_rate = 0.995
    keeping_chance = 1
    episode_rewards = []
    episode_lengths = []

    for episode in range(1, episodes + 1):
        print("Epoch:", episode)
        if episode % 500 == 499:
            try:
                net.save_data(f"./dqn_model_ep{episode}.json")
            except Exception as e:
                print("save_data not implemented in Rust:", e)

        epsilon = max(min_epsilon, epsilon * epsilon_decay_rate)
        net.set_epsilon(epsilon) 

        observation, info = env.reset()
        episode_reward = 0
        episode_steps = 0

        for t in range(T):
            pre_state = observation[:]   # list of floats
            action = net.choose_action(pre_state)  # Rust returns index (usize)
            observation, reward, terminated, truncated, info = env.step(action)

            episode_steps += 1
            episode_reward += reward

            if random.random() < keeping_chance:
                net.add_experience(pre_state, action, float(reward),terminated or truncated, observation)

            if t%100==0:
                net.train()

            if terminated or truncated:
                print("Reward:", episode_reward)
                break

        episode_rewards.append(episode_reward)
        #episode_lengths.append(episode_steps)


        if episode%50 == 10:
            net.update_target()
        if episode % 100 == 99:
            plot_metrics(episode_rewards)
            try:
                net.save_data(f"./dqn_model_ep{episode}.json")
            except Exception as e:
                print("save_data not implemented:", e)
            # plot metrics or save them


import gymnasium as gym
env = gym.make('CartPole-v1')

main_loop(env, topology=[4, 128, 64, 2])
env.close()