import pygame
import numpy as np
from environment import RaceTrackEnv
from agent import DQNAgent
from model import create_model
from visualizer import NeuralNetworkVisualizer

# Initialize Pygame
pygame.init()

# Set up the display
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 600  # Increased width to accommodate visualizer
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("AI Race Track Game with Neural Network Visualizer")

# Create the environment
env = RaceTrackEnv(800, 600)  # Keep original size for race track

# Create the DQN agent
state_size = 4  # x, y, angle, speed
action_size = 5  # straight, left, right, accelerate, brake
model = create_model(state_size, action_size)

# Pass the model to the DQNAgent constructor
agent = DQNAgent(state_size, action_size, model)

# Create the Neural Network Visualizer
VISUALIZER_WIDTH, VISUALIZER_HEIGHT = 400, 300
visualizer = NeuralNetworkVisualizer(model, VISUALIZER_WIDTH, VISUALIZER_HEIGHT)

# Training parameters
n_episodes = 1000
batch_size = 32

# Font for displaying information
font = pygame.font.Font(None, 36)

for episode in range(n_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    score = 0

    while not done:
        # Clear the screen
        screen.fill((0, 0, 0))

        # Render the environment
        env.render(screen)

        # Update and render the neural network visualizer
        visualizer.update_visualization(state)
        visualizer_surface = pygame.Surface((VISUALIZER_WIDTH, VISUALIZER_HEIGHT))
        visualizer.render(visualizer_surface)
        screen.blit(visualizer_surface, (800, 0))  # Position next to the race track

        # Display episode and score information
        episode_text = font.render(f"Episode: {episode + 1}/{n_episodes}", True, (255, 255, 255))
        score_text = font.render(f"Score: {score}", True, (255, 255, 255))
        screen.blit(episode_text, (10, 10))
        screen.blit(score_text, (10, 50))

        pygame.display.flip()

        # Get action from the agent
        action = agent.get_action(state)

        # Take action and observe the result
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        score += reward

        # Store experience in memory
        agent.remember(state, action, reward, next_state, done)

        # Train the agent
        agent.train(state, action, reward, next_state, done)

        state = next_state

        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

    print(f"Episode: {episode + 1}/{n_episodes}, Score: {score}")

    # Train the model with experience replay
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

pygame.quit()   