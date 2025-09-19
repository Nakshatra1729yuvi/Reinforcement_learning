import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import deque
import random


class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, h2_nodes, out_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.fc2 = nn.Linear(h1_nodes, h2_nodes)
        self.fc3 = nn.Linear(h2_nodes, out_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayMemory:
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


class LunarLanderDQL:
    # Hyperparameters
    learning_rate_a = 0.001
    discount_factor_g = 0.99
    network_sync_rate = 100
    replay_memory_size = 10000
    mini_batch_size = 32

    # Neural Network
    loss_fn = nn.MSELoss()
    optimizer = None

    def train(self, episodes,continue_training=False):
        env = gym.make("LunarLander-v3", render_mode=None)
        num_states = env.observation_space.shape[0]  # 8 for LunarLander-v3
        num_actions = env.action_space.n  # 4 for LunarLander-v3
        
        epsilon = 1.0
        memory = ReplayMemory(self.replay_memory_size)

        policy_dqn = DQN(in_states=num_states, h1_nodes=64, h2_nodes=64, out_actions=num_actions)

        if continue_training:
            policy_dqn.load_state_dict(torch.load("lunar_lander_dql.pt"))

        target_dqn = DQN(in_states=num_states, h1_nodes=64, h2_nodes=64, out_actions=num_actions)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        rewards_per_episode = []
        epsilon_history = []
        step_count = 0

        for i in tqdm(range(episodes)):
            state = env.reset()[0]
            total_reward = 0
            terminated = False
            truncated = False

            

            while not terminated and not truncated:
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy_dqn(torch.FloatTensor(state)).argmax().item()

                new_state, reward, terminated, truncated, _ = env.step(action)
                memory.append((state, action, new_state, reward, terminated))
                state = new_state
                total_reward += reward
                step_count += 1

                if len(memory) >= self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                if step_count >= self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

            rewards_per_episode.append(total_reward)
            epsilon = max(epsilon - 1 / episodes, 0.1)  # Minimum epsilon of 0.1
            epsilon_history.append(epsilon)

            # Print progress every 100 episodes
            if (i + 1) % 100 == 0:
                avg_reward = np.mean(rewards_per_episode[-100:])
                print(f"Episode {i+1}, Avg Reward (last 100): {avg_reward:.2f}, Epsilon: {epsilon:.2f}")

        env.close()
        

        torch.save(policy_dqn.state_dict(), "lunar_lander_dql.pt")
        return rewards_per_episode, epsilon_history

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:
            if terminated:
                target = torch.FloatTensor([reward])
            else:
                with torch.no_grad():
                    target = reward + self.discount_factor_g * target_dqn(torch.FloatTensor(new_state)).max()

            current_q = policy_dqn(torch.FloatTensor(state))
            current_q_list.append(current_q)

            target_q = policy_dqn(torch.FloatTensor(state)).clone().detach()
            target_q[action] = target
            target_q_list.append(target_q)

        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def state_to_dqn_input(self, state, num_states):
        return torch.FloatTensor(state)

    def test(self, episodes):
        env = gym.make("LunarLander-v3", render_mode="human")
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(in_states=num_states, h1_nodes=64, h2_nodes=64, out_actions=num_actions)
        policy_dqn.load_state_dict(torch.load("lunar_lander_dql.pt"))
        policy_dqn.eval()

        total_rewards = []
        for i in range(episodes):
            state = env.reset()[0]
            total_reward = 0
            terminated = False
            truncated = False

            while not terminated and not truncated:
                with torch.no_grad():
                    action = policy_dqn(torch.FloatTensor(state)).argmax().item()
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward

            total_rewards.append(total_reward)
            print(f"Test Episode {i+1}, Total Reward: {total_reward:.2f}")

        env.close()
        return total_rewards


if __name__ == "__main__":
    lunar_lander = LunarLanderDQL()
    rewards, epsilon_history = lunar_lander.train(100,True)  # Continue training from saved model
    lunar_lander.test(10)

   