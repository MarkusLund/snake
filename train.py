from snake_env import Snake
from agent import DQN

import numpy as np
from plot import plot_result
import datetime


def save_model(id, agent, best_score, total_reward):
    if (total_reward > best_score):
        agent.save_model(id, total_reward)
        best_score = total_reward


def train_dqn(episodes, env):
    now = datetime.datetime.now()
    id = f'{now.hour}{now.minute}'
    episode_rewards = []
    agent = DQN(env, params)
    best_score = -999
    for episode in range(episodes):
        state = env.reset()  # Reset enviroment before each episode to start fresh
        state = np.reshape(state, (1, env.state_space))
        total_reward = 0
        max_steps = 10000
        for step in range(max_steps):
            # 1. Find next action using the Epsilon-Greedy exploration Strategy
            action = agent.get_action(state)

            # 2. perform action in enviroment
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            next_state = np.reshape(next_state, (1, env.state_space))

            # 3. Update the Q-function (train model) or do 3b. TIPS: Wont the agent learn, check out experience replay!
            agent.learn(state, action, reward, next_state, done)

            agent.update_exploration_strategy(episode)
            state = next_state

            if done:
                print(f'episode: {episode+1}/{episodes}, score: {total_reward}, steps: {step}, epsilon: {agent.epsilon}')
                save_model(id, agent, best_score, total_reward)
                break

        save_model(id, agent, best_score, total_reward)
        episode_rewards.append(total_reward)
    return episode_rewards


if __name__ == '__main__':

    params = dict()
    params['name'] = None
    params['gamma'] = 0.95
    params['batch_size'] = 500
    params['epsilon'] = 1
    params['epsilon_min'] = 0.01
    params['epsilon_max'] = 1
    params['epsilon_decay'] = 0.02
    params['learning_rate'] = 0.7

    results = dict()
    episodes = 500

    env = Snake()
    sum_of_rewards = train_dqn(episodes, env)
    results[params['name']] = sum_of_rewards

    plot_result(results, direct=True, k=20)
