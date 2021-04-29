from environment import Snake
from agent import DQN
import time
import numpy as np
from plot import plot_result
import sys


def test_dqn(env):
    agent = DQN(env, params)

    agent.load_model(sys.argv[1], sys.argv[2])

    state = env.reset()  # Reset enviroment before each episode to start fresh
    state = np.reshape(state, (1, env.state_space))
    max_steps = 10000
    total_reward = 0

    for step in range(max_steps):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)

        state = np.reshape(next_state, (1, env.state_space))
        total_reward += reward
        time.sleep(0.1)
        if done:
            print(f'Score: {total_reward}, steps: {step}')
            break
    return


if __name__ == '__main__':
    params = dict()
    params['gamma'] = 1
    params['batch_size'] = 1
    params['epsilon'] = 0
    params['epsilon_min'] = 1
    params['epsilon_max'] = 1
    params['epsilon_decay'] = 1
    params['learning_rate'] = 1

    results = dict()

    env = Snake()
    sum_of_rewards = test_dqn(env)
