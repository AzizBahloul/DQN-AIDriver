import numpy as np
import random
from collections import deque
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size, model):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = model

        # New attributes for tracking maximum score
        self.max_score = -float('inf')  # Initialize max score to negative infinity
        self.best_state = None  # To store the state with the maximum score

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def train(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
        target_f = self.model.predict(state, verbose=0)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)
        
        # Update max score and best state if current score is higher
        if reward > self.max_score:
            self.max_score = reward
            self.best_state = state  # Store the state that gave the max score

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

            # Update max score and best state if current score is higher
            if reward > self.max_score:
                self.max_score = reward
                self.best_state = state  # Store the state that gave the max score

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_network_weights(self):
        return [layer.get_weights() for layer in self.model.layers]

    def save_best_state(self, filename='best_state.npy'):
        if self.best_state is not None:
            np.save(filename, self.best_state)
            print(f"Best state saved to {filename} with score: {self.max_score}")
