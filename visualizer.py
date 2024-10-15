import pygame
import numpy as np
import tensorflow as tf

class NeuralNetworkVisualizer:
    def __init__(self, model, width, height):
        self.model = model
        self.width = width
        self.height = height
        self.neuron_positions = self._calculate_neuron_positions()
        self.activations = []

    def _calculate_neuron_positions(self):
        # Get sizes for only the input and output layers
        input_size = self.model.input_shape[-1] if self.model.input_shape else 4  # Default input size
        output_size = self.model.output_shape[-1] if self.model.output_shape else 1  # Default output size

        # Calculate positions for input neurons
        input_positions = [(self.width / 4, (self.height / (input_size + 1)) * (i + 1)) for i in range(input_size)]
        # Calculate positions for output neurons
        output_positions = [(self.width * 3 / 4, (self.height / (output_size + 1)) * (i + 1)) for i in range(output_size)]
        
        return [input_positions, output_positions]  # Return only input and output positions

    def update_visualization(self, input_state):
        self.activations = self._get_activations(input_state)
        self.activations = [np.clip(layer_output.flatten(), 0, 1) for layer_output in self.activations]

    def _get_activations(self, input_state):
        current_output = input_state
        activations = [input_state]

        for layer in self.model.layers:
            current_output = layer(current_output)
            activations.append(current_output.numpy())

        return activations

    def render(self, screen):
        screen.fill((255, 255, 255))  # White background

        # Draw the connections from input layer to output layer
        for start_pos in self.neuron_positions[0]:  # Input neurons
            for end_pos in self.neuron_positions[1]:  # Output neurons
                pygame.draw.line(screen, (200, 200, 200), start_pos, end_pos, 1)  # Gray lines for connections

        # Draw the input neurons
        for neuron_idx, neuron_pos in enumerate(self.neuron_positions[0]):
            brightness = int(self.activations[0][neuron_idx] * 255) if self.activations else 100
            color = (brightness, brightness, 0)  # Yellow for active input neurons
            pygame.draw.circle(screen, color, (int(neuron_pos[0]), int(neuron_pos[1])), 10)  # Larger neurons for visibility

        # Draw the output neurons
        for neuron_idx, neuron_pos in enumerate(self.neuron_positions[1]):
            brightness = int(self.activations[-1][neuron_idx] * 255) if self.activations else 100
            color = (0, brightness, 0)  # Green for active output neurons
            pygame.draw.circle(screen, color, (int(neuron_pos[0]), int(neuron_pos[1])), 10)  # Larger neurons for visibility

        pygame.display.flip()
