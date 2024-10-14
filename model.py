from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_model(state_size, action_size):
    model = Sequential()
    model.add(Dense(16, input_dim=state_size, activation='relu'))  # Reduced number of neurons
    model.add(Dense(16, activation='relu'))  # Reduced number of neurons
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    return model
