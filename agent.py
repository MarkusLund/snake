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
        self.epsilon = params['epsilon']
        self.gamma = params['gamma']
        self.batch_size = params['batch_size']
        self.epsilon_min = params['epsilon_min']
        self.epsilon_decay = params['epsilon_decay']
        self.learning_rate = params['learning_rate']
        self.memory = deque(maxlen=2500)
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()

        # Input = state, output = action
        model.add(Dense(128, input_shape=(
            self.state_space,), activation='relu'))
        model.add(Dense(128, activation='relu'))

        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        print(act_values)
        return np.argmax(act_values[0])

    def replay(self):

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
        future_qs = self.model.predict_on_batch(next_states)

        targets = rewards + self.gamma * (np.amax(future_qs, axis=1)) * (1 - dones)

        ind = np.array([i for i in range(self.batch_size)])
        current_qs[[ind], [actions]] = targets

        self.model.fit(states, current_qs, epochs=1, verbose=0)

        # Reduce epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_dqn(episode, env):

    sum_of_rewards = []
    agent = DQN(env, params)
    for episode in range(episode):
        state = env.reset()  # Reset enviroment before each episode to start fresh
        state = np.reshape(state, (1, env.state_space))
        score = 0
        max_steps = 10000
        for i in range(max_steps):
            action = agent.act(state)
            prev_state = state
            next_state, reward, done, _ = env.step(action)
            score += reward
            next_state = np.reshape(next_state, (1, env.state_space))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if params['batch_size'] > 1:
                agent.replay()
            if done:
                print(f'final state before dying: {str(prev_state)}')
                print(f'episode: {episode+1}/{episode}, score: {score}')
                break
        sum_of_rewards.append(score)
    return sum_of_rewards


if __name__ == '__main__':

    params = dict()
    params['name'] = None
    params['epsilon'] = 1
    params['gamma'] = .95
    params['batch_size'] = 500
    params['epsilon_min'] = .01
    params['epsilon_decay'] = .995
    params['learning_rate'] = 0.00025

    results = dict()
    episodes = 50

    env = Snake()
    sum_of_rewards = train_dqn(episodes, env)
    results[params['name']] = sum_of_rewards

    plot_result(results, direct=True, k=20)
