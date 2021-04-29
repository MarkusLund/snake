
from snake_env import Snake
import numpy as np
from tensorflow.keras import *

if __name__ == '__main__':
    human = True
    env = Snake(human=human)
    print("Use the arrow keys on keyboard to start and control the snake.")
    if human:
        while True:
            env.calculate_reward()
