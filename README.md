# Reinforcement Learning

A collection of reinforcement learning algorithms and implementations using Deep Q-Learning (DQN).

## Description

This repository contains implementations of reinforcement learning algorithms, specifically Deep Q-Networks (DQN), applied to classic environments like Frozen Lake and Lunar Lander. The projects demonstrate the application of deep reinforcement learning techniques to solve control problems.

## Environments

### Frozen Lake
- Environment: OpenAI Gym FrozenLake-v1
- Algorithm: Deep Q-Network (DQN)
- Goal: Navigate from start to goal while avoiding holes in the ice

### Lunar Lander
- Environment: OpenAI Gym LunarLander-v2
- Algorithm: Deep Q-Network (DQN)
- Goal: Land the spacecraft safely on the landing pad

## Files

- `frozen_lake_dqn.py` - DQN implementation for Frozen Lake environment
- `frozen_lake_dql.pt` - Trained model weights for Frozen Lake
- `lunar_lander_dqn.py` - DQN implementation for Lunar Lander environment
- `lunar_lander_dql.pt` - Trained model weights for Lunar Lander

## Features

- Deep Q-Network implementation
- Experience replay buffer
- Target network for stable training
- Epsilon-greedy exploration strategy
- Model saving and loading capabilities

## Requirements

- Python 3.x
- PyTorch
- OpenAI Gym
- NumPy
- Matplotlib (for visualization)

## Installation

1. Clone this repository
2. Install required dependencies:
   ```bash
   pip install torch gym numpy matplotlib
   ```
3. Run the training scripts:
   ```bash
   python frozen_lake_dqn.py
   python lunar_lander_dqn.py
   ```

## Usage

Each script can be run independently to train or test the DQN agent in the respective environment. The trained models are saved as `.pt` files and can be loaded for evaluation.

## Algorithm Details

The implementation uses:
- Deep Q-Network with fully connected layers
- Experience replay for stable learning
- Target network updated periodically
- Epsilon decay for exploration vs exploitation balance

## License

This project is licensed under the MIT License - see the LICENSE file for details.
