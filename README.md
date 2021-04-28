# Workshop: Snake - Reinforcement Learning

## Tasks

1. Look at the rewards

1. Look at the state

1. Introduce a target network

> One of the interesting things about Deep Q-Learning is that the learning process uses 2 neural networks. These networks have the same architecture but different weights. Every N steps, the weights from the main network are copied to the target network. Using both of these networks leads to more stability in the learning process and helps the algorithm to learn more effectively. In our implementation, the main network weights replace the target network weights every 100 steps.

## Problems

### "macOS 11 or later required !"

Solution: Download and install python from python.org as this is probebly an issue with the homebrew-version.

## README form source project

Snake

This project contains the following files:

1. snake_env.py : run this and you can play the game Snake by yourself
2. agent_1.py : run this and a Deep Reinforcement Learning Agent will learn to play snake
3. plot_script.py : plotting the results of the agent
4. requirements.txt : you will need some Python packages, like turtle, TensorFlow and Keras before you can run the scripts, install these first
