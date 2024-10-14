import numpy as np

def normalize_state(state, max_x, max_y):
    """Normalize the state values to be between 0 and 1"""
    return np.array([state[0] / max_x, state[1] / max_y])

def calculate_distance(pos1, pos2):
    """Calculate Euclidean distance between two points"""
    return np.linalg.norm(pos1 - pos2)
