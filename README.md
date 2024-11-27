# Deep Q-Learning Network Implementation

A PyTorch implementation of Deep Q-Learning Networks (DQN) for training AI agents to play Atari 2600 games using only screen pixels as input.

## Overview

This project implements a DQN that can learn to play various Atari games without prior knowledge or game-specific rules. The agent learns purely from visual input and reward signals, making it highly extensible across different games.

### Key Features

- Screen-only input processing
- Frame stacking for motion detection
- Efficient memory management using HDF5
- Extensible architecture supporting multiple Atari games
- Implementation of key DQN concepts from the original paper

## Architecture

### Neural Network Structure
- Input: 84x84x4 (4 stacked grayscale frames)
- Conv1: 8x8 kernel, stride 4
- Conv2: 4x4 kernel, stride 2
- Flatten layer
- Dense: 2592 units
- Dense: 256 units
- Output: n_actions units

### Key Components

1. **Frame Processing**
- Grayscale conversion
- Downsampling (210x160 → 110x84)
- Cropping to 84x84
- Enhanced contrast
- 4-frame stacking

2. **Memory Management**
- Replay memory size: 1,000,000 transitions
- HDF5-based storage for memory efficiency
- Batch sampling for training

3. **Training System**
- Dual network architecture (policy and target networks)
- ε-greedy exploration strategy
- Target network updates every 10,000 frames
- Action selection every 4 frames

### Training Parameter
- Episodes: 5,000,000 frames
- Batch size: 32
- Learning rate: 1e-4 (initial), 5e-5 (after 5M frames)
- Gamma (discount factor): 0.99
- Epsilon decay: Linear from 1.0 to 0.1 over 1M frames

## Program Setup & Run
1. Create and activate a virtual environment:
    ```
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # Linux/Mac
    python -m venv venv
    source venv/bin/activate
    ```

2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

3. Run the Programs
    ```
    # Start new training
    python -m atari_dqn.main

    # Start CartPole training
    python -m cart_pole.main
    ```

## Reference
- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)

- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
