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

    def save_model(self, id, name):
        self.model.save(f'./models/{id}/{name}')

    def load_model(self, id, name):
        self.model = load_model(f'./models/{id}/{name}')

    def build_model(self):
        # TODO: Create model, remember to compile
        model = Sequential()
        return model

    def get_action(self, state):
        # With episilon probablity choose random action vs best action
        return 0

    def learn(self, state, action, reward, next_state, done):
        self.model.fit(None, None, epochs=1, verbose=0)

    def update_exploration_strategy(self, episode):
        # Reduce epsilon
        self.epsilon = 0.5
