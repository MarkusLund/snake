# Workshop: Snake - Reinforcement Learning

! WORK IN PROGRESS !

## Requirements

Recommended Python 3.8 downloaded from python.org on Mac OS.

You may need to install a tensorflow version manually, e.g. `pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-2.4.0-cp38-cp38-macosx_10_14_x86_64.whl` (for Python 3.8)

## Tips

Keep the neural network simple. Not much is needed for a simple state action space.

## Tasks

1. Look at the rewards

1. Look at the state

1. Introduce a target network

> One of the interesting things about Deep Q-Learning is that the learning process uses 2 neural networks. These networks have the same architecture but different weights. Every N steps, the weights from the main network are copied to the target network. Using both of these networks leads to more stability in the learning process and helps the algorithm to learn more effectively. In our implementation, the main network weights replace the target network weights every 100 steps. [source](https://towardsdatascience.com/deep-q-learning-tutorial-mindqn-2a4c855abffc)

## Problems

### "macOS 11 or later required !"

Solution: Download and install python from python.org as this is probebly an issue with the homebrew-version.

## Files

1. agent.py : Edit this to implement the meat of the DQN algorithm
1. snake_env.py : Here you can edit the state and rewards given.
1. train.py : Train your model
1. play_snake.py : Play snake and check if your requirements are in place.
1. test.py : Test your saved models. Eg. python test.py 1043(id) 650(total_reward) (models stored in models/{timestamp}/{total_reward})
