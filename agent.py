import random
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
from tensorflow.python.keras.saving.save import load_model


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

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

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

    def save_model(self, id, name):
        self.model.save(f'./models/{id}/{name}')

    def load_model(self, id, name):
        self.model = load_model(f'./models/{id}/{name}')

    def build_model(self):
        # TODO: Create model, remember to compile
        model = Sequential()

        # Input = state, output = action
        model.add(Dense(128, input_shape=(self.state_space,), activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.00025))
        return model

    def get_action(self, state):
        # With episilon probablity choose random action vs most
        # TODO: Implement method
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values)

    def learn(self, state, action, reward, next_state, done):
        current_qs = self.model.predict(state)
        future_qs = self.model.predict(next_state)

        max_future_q = reward + self.gamma * np.amax(future_qs, axis=1) * (1 - done)

        current_qs[action] = (1 - self.learning_rate) * current_qs[action] + self.learning_rate * max_future_q

        self.model.fit(state, current_qs, epochs=1, verbose=0)

    def update_exploration_strategy(self, episode):
        # Reduce epsilon
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-self.epsilon_decay * episode)
