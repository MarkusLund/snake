from snake_env import Snake

import random
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
from plot import plot_result


class DQN:

    """ Deep Q Network """

    def __init__(self, env, params):

        self.action_space = env.action_space
        self.state_space = env.state_space
        self.gamma = params['gamma']
        self.batch_size = params['batch_size']
        self.epsilon = params['epsilon']
        self.epsilon_min = params['epsilon_min']
        self.epsilon_max = params['epsilon_max']
        self.epsilon_decay = params['epsilon_decay']
        self.learning_rate = params['learning_rate']
        self.memory = deque(maxlen=2500)
        self.model = self.build_model()

    def build_model(self):
        # TODO: Create model, remember to compile
        model = Sequential()

        # Input = state, output = action
        model.add(Dense(128, input_shape=(
            self.state_space,), activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.00025))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        # With episilon probablity choose random action vs most
        # TODO: Implement method
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values)

    def experience_replay(self):

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        current_qs = self.model.predict_on_batch(states)
        future_qs = self.model.predict_on_batch(next_states)  # Introduce target model??

        max_future_q = rewards + self.gamma * np.amax(future_qs, axis=1) * (1 - dones)

        ind = np.array([i for i in range(self.batch_size)])
        current_qs[[ind], [actions]] = (1 - self.learning_rate) * current_qs[[ind], [actions]] + self.learning_rate * max_future_q

        self.model.fit(states, current_qs, epochs=1, verbose=0)

    def update_q(self, state, action, reward, next_state, done):

        if len(self.memory) < self.batch_size:
            return

        current_qs = self.model.predict(state)
        future_qs = self.model.predict(next_state)

        max_future_q = reward + self.gamma * np.amax(future_qs, axis=1) * (1 - done)

        # current_qs[action] = (1 - self.learning_rate) * current_qs[action] + self.learning_rate * max_future_q
        current_qs[action] = max_future_q

        self.model.fit(state, current_qs, epochs=1, verbose=0)

    def update_exploration_strategy(self, episode):
        # Reduce epsilon
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-self.epsilon_decay * episode)


def train_dqn(episodes, env):

    episode_rewards = []
    agent = DQN(env, params)
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

            # 3a. Update the Q-function (train model) or do 3b.
            agent.update_q(state, action, reward, next_state, done)

            # # 3b. Use experience replay
            # agent.remember(state, action, reward, next_state, done)
            # agent.experience_replay()

            agent.update_exploration_strategy(episode)
            state = next_state
            ep = agent.epsilon
            if done:
                print(f'episode: {episode+1}/{episodes}, score: {total_reward}, steps: {step}, epsilon: {ep}')
                break
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
    params['epsilon_decay'] = 0.01
    params['learning_rate'] = 0.7

    results = dict()
    episodes = 50

    env = Snake()
    sum_of_rewards = train_dqn(episodes, env)
    results[params['name']] = sum_of_rewards

    plot_result(results, direct=True, k=20)
