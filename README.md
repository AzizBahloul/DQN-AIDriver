# Project Title: AI Race Track Game



## Installation

1. Clone the repository:

   git clone https://github.com/yourusername/AI-Race-Track-Game.git

2. Navigate to the project directory:

   cd AI-Race-Track-Game

3. Install the required dependencies:

   pip install -r requirements.txt

## Project Structure

- **agent.py**: Contains the DQNAgent class for the reinforcement learning agent.
- **environment.py**: Defines the RaceTrackEnv class for the racing environment.
- **main.py**: The main entry point to run the game and train the agent.
- **model.py**: Contains the function to create the neural network model for the agent.
- **utils.py**: Provides utility functions for state normalization and distance calculations.
- **assets/**: Directory for images and other assets used in the game.

## Usage

1. Run the main script:

   python main.py

2. The game will start, and the AI agent will begin training on the racetrack. You can monitor the training progress through the displayed score and episode information.

## Training Parameters

- Number of episodes: 1000
- Batch size: 32
- Learning rate: 0.001
- Epsilon decay for exploration: 0.995

## Customization

Feel free to modify the following to customize the game:

- Change the racetrack design in `environment.py`.
- Adjust the neural network architecture in `model.py`.
- Tweak hyperparameters in `agent.py` for different training behaviors.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
