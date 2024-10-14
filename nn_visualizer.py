import numpy as np
import matplotlib.pyplot as plt

class NeuralNetworkVisualizer:
    def __init__(self, width, height):
        self.fig, self.ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        self.ax.axis('off')

    def draw_neural_network(self, layer_sizes):
        plt.cla()
        layer_colors = ['#FFCCCB', '#FF9999', '#FF6666']
        v_spacing = 1.0 / (max(layer_sizes) + 1)

        for n, layer_size in enumerate(layer_sizes):
            layer_y = (layer_size + 1) * v_spacing
            for m in range(layer_size):
                circle = plt.Circle((n * 0.5, layer_y - (m + 1) * v_spacing), 0.04, color='white', ec='black')
                self.ax.add_artist(circle)

                if n > 0:
                    for prev_m in range(layer_sizes[n - 1]):
                        plt.plot([n - 1 * 0.5, n * 0.5], [layer_sizes[n - 1] * v_spacing - (prev_m + 1) * v_spacing,
                                                          layer_y - (m + 1) * v_spacing], color='black')

    def update(self, weights):
        layer_sizes = [layer.shape[1] for layer in weights]
        self.draw_neural_network(layer_sizes)
        plt.pause(0.01)
