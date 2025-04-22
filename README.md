# Prerequisites

This project depends on numpy. In a Homebrew-managed Python environment, to install numpy without system conflicts, either:

Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate
pip install numpy

Install as user:

python3 -m pip install --user numpy

# Data Generation and Simulation

For our project, we are not using existing historical data. Instead, we generate all our data using our custom simulation. In our simulation, passenger requests are created randomly at each floor using a probability model (like a Poisson process). This way, the data reflects how passengers might call the elevator at unpredictable times in a real building.

Once the simulation runs, we gather data that includes the elevator’s current floor, the number of waiting passengers on each floor, and the elevator’s movement status (idle, going up, or going down). We then process this raw data into a format that our reinforcement learning agent can use. For example, we represent the state as a tuple that shows the current floor along with the waiting passenger counts for all floors.

In addition, we calculate a reward for each simulation step. The reward is positive when the elevator picks up waiting passengers and negative when the elevator is idle or moves unnecessarily. This reward information, along with the state and action taken, is stored and later analyzed to see how well the RL agent is learning and to compare its performance with a traditional fixed schedule.

# Summary

In summary, all data comes from our simulation, is processed into clear state-action-reward pairs, and then used for training and evaluating the RL agent.

