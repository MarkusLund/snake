# Workshop: Snake - Reinforcement Learning

## Deep Q-learning algorithm

1. Initialize replay memory capacity.
1. Initialize the network with random weights.
1. For each episode:
   1. Initialize the starting state.
   1. For each time step:
      1. Select an action.
         - Via exploration or exploitation
      1. Execute selected action in an emulator.
      1. Observe reward and next state.
      1. Store experience in replay memory.
      1. Sample random batch from replay memory.
      1. Preprocess states from batch.
      1. Pass batch of preprocessed states to policy network.
      1. Calculate loss between output Q-values and target Q-values.
         - Requires a second pass to the network for the next state
      1. Gradient descent updates weights in the policy network to minimize loss.

### The Bellman equation

![The bellman equation](imgs/be.png)

## Tasks

Find all the TODOs 🕵️‍♂️

1. Enviroment:

   1. Set rewards (`snake_env.py/calculate_reward()`)

   1. Define the state space. What is the agent allowed to observe? (`snake_env.py/get_state()`)

1. Agent:

   1. Build the neural network model which will estimate the Q-value. (`agent.py/build_model()`)
   1. Implement `agent.py/get_action()` to fetch which action to perfom given a state. Remember to consider exploration vs. explotation.
   1. Implement the Bellman Equation in `agent.py/train_with_experience_replay()` to actually train your model from previous (state, action)-pairs.
   1. OPTIONAL - Gradually change exploration vs. explotation by changing (`agent.py/update_exploration_strategy()`)

### Bonus task

1. Introduce a target network

> One of the interesting things about Deep Q-Learning is that the learning process uses 2 neural networks. These networks have the same architecture but different weights. Every N steps, the weights from the main network are copied to the target network. Using both of these networks leads to more stability in the learning process and helps the algorithm to learn more effectively. In our implementation, the main network weights replace the target network weights every 100 steps. [source](https://towardsdatascience.com/deep-q-learning-tutorial-mindqn-2a4c855abffc)

## Requirements

Recommended Python 3.8 downloaded from python.org on Mac OS.

You may need to install a tensorflow version manually, e.g. `pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-2.4.0-cp38-cp38-macosx_10_14_x86_64.whl` (for Python 3.8)

## Problems

### "macOS 11 or later required !"

Solution: Download and install python from python.org as this is probebly an issue with the homebrew-version.

## Files

1. agent.py : Edit this to implement the meat of the DQN algorithm
1. snake_env.py : Here you can edit the state and rewards given.
1. train.py : Train your model
1. play_snake.py : Play snake and check if your requirements are in place.
1. test.py : Test your saved models. Eg. python test.py 1043(id) 650(total_reward) (models stored in models/{timestamp}/{total_reward})
