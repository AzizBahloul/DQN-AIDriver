import pygame
import numpy as np
import gym
from gym import spaces
from scipy.interpolate import splprep, splev

class RaceTrackEnv(gym.Env):
    def __init__(self, width, height):
        super(RaceTrackEnv, self).__init__()

        self.width = width
        self.height = height

        # Define action and observation space
        self.action_space = spaces.Discrete(5)  # straight, left, right, accelerate, brake
        self.observation_space = spaces.Box(low=0, high=np.array([width, height, 360, 10]), dtype=np.float32)

        # Load car image
        self.car_img = pygame.image.load('assets/car.jpg')
        self.car_img = pygame.transform.scale(self.car_img, (30, 15))

        # Track properties
        self.track_width = 60
        self.track = self.create_race_track()

        # Car properties
        self.car_pos = np.array([0.0, 0.0])
        self.car_angle = 0
        self.car_speed = 0
        self.max_speed = 10  # Increased max speed

        self.reset()

    def create_race_track(self):
        # Create a more complex, realistic race track using splines
        points = np.array([
            [100, 400], [150, 300], [250, 200], [400, 150],
            [550, 200], [650, 300], [700, 400], [650, 500],
            [500, 550], [300, 500], [100, 400]
        ])

        tck, u = splprep(points.T, u=None, s=0.0, per=1)
        u_new = np.linspace(u.min(), u.max(), 1000)
        x_new, y_new = splev(u_new, tck, der=0)

        track = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.lines(track, (100, 100, 100), True, list(zip(x_new, y_new)), self.track_width)
        pygame.draw.lines(track, (255, 255, 255), True, list(zip(x_new, y_new)), 2)

        self.start_pos = np.array([x_new[0], y_new[0]])
        self.end_pos = np.array([x_new[-1], y_new[-1]])
        self.checkpoints = np.array(list(zip(x_new[::100], y_new[::100])))  # Ensure checkpoints are NumPy arrays

        return track

    def step(self, action):
        # Update car position and angle based on action
        if action == 0:  # straight
            pass
        elif action == 1:  # left
            self.car_angle -= 5
        elif action == 2:  # right
            self.car_angle += 5
        elif action == 3:  # accelerate
            self.car_speed = min(self.car_speed + 2, self.max_speed)  # Faster acceleration
        elif action == 4:  # brake
            self.car_speed = max(self.car_speed - 0.4, 0)  # Faster braking

        # Move car
        self.car_pos[0] += self.car_speed * np.cos(np.radians(self.car_angle))
        self.car_pos[1] += self.car_speed * np.sin(np.radians(self.car_angle))

        # Check if car is on track
        if self.track.get_at((int(self.car_pos[0]), int(self.car_pos[1]))).a == 0:
            reward = -10
            done = True
            self.car_pos = self.start_pos.copy()  # Return to start position
        else:
            reward = self.car_speed  # Reward based on speed
            done = False

        # Check if car has reached next checkpoint
        if len(self.checkpoints) > 0 and np.linalg.norm(self.car_pos - self.checkpoints[0]) < 30:
            reward += 20
            self.checkpoints = np.delete(self.checkpoints, 0, axis=0)  # Remove reached checkpoint
            if len(self.checkpoints) == 0:
                reward += 100
                done = True

        return np.array([self.car_pos[0], self.car_pos[1], self.car_angle, self.car_speed]), reward, done, {}

    def reset(self):
        self.car_pos = self.start_pos.copy()
        self.car_angle = 0
        self.car_speed = 0
        self.checkpoints = np.array(list(zip(self.checkpoints[:, 0], self.checkpoints[:, 1])))  # Reset checkpoints
        return np.array([self.car_pos[0], self.car_pos[1], self.car_angle, self.car_speed])

    def render(self, screen):
        screen.fill((0, 100, 0))  # Green background
        screen.blit(self.track, (0, 0))

        # Draw start and end lines
        pygame.draw.line(screen, (255, 0, 0), self.start_pos.astype(int), self.start_pos.astype(int) + np.array([30, 0]), 3)
        pygame.draw.line(screen, (0, 0, 255), self.end_pos.astype(int), self.end_pos.astype(int) + np.array([30, 0]), 3)

        # Draw checkpoints
        for checkpoint in self.checkpoints:
            pygame.draw.circle(screen, (255, 0, 0), checkpoint.astype(int), 5)

        # Draw car
        rotated_car = pygame.transform.rotate(self.car_img, -self.car_angle)
        car_rect = rotated_car.get_rect(center=self.car_pos.astype(int))
        screen.blit(rotated_car, car_rect.topleft)
