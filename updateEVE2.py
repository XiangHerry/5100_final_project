import time
import random
import logging
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(message)s')

# -------------------------
# Building Class
# -------------------------
class Building:
    def __init__(self, num_floors: int):
        if num_floors < 2:
            raise ValueError("A building must have at least 2 floors.")
        self.floors = list(range(1, num_floors + 1))

    def __repr__(self):
        return f"Building with floors: {self.floors}"

# -------------------------
# Elevator Class
# -------------------------
class Elevator:
    def __init__(self, building: Building, start_floor: int = 1, delay: float = 0):
        self.building = building
        self.current_floor = start_floor
        self.status = "idle"
        self.delay = delay

    def move_up(self):
        if self.current_floor < self.building.floors[-1]:
            self.current_floor += 1
            self.status = "up"
            logging.info("Moving up...")
        else:
            logging.warning("Already at the highest floor; cannot move up.")

    def move_down(self):
        if self.current_floor > self.building.floors[0]:
            self.current_floor -= 1
            self.status = "down"
            logging.info("Moving down...")
        else:
            logging.warning("Already at the lowest floor; cannot move down.")

    def idle(self):
        self.status = 'idle'
        logging.info("Staying idle.")

    def step(self, direction: str):
        if direction == 'up':
            self.move_up()
        elif direction == 'down':
            self.move_down()
        elif direction == 'idle':
            self.idle()
        else:
            logging.error("Invalid direction. Choose 'up', 'down', or 'idle'.")
            return
        if self.delay:
            time.sleep(self.delay)
        logging.info(f"Current floor: {self.current_floor}, Status: {self.status}")

# -------------------------
# Simulation Environment
# -------------------------
class ElevatorSimulation:
    def __init__(self, num_floors: int, delay: float = 0, poisson_lambda: float = 1.0):
        self.building = Building(num_floors)
        self.elevator = Elevator(self.building, start_floor=1, delay=delay)
        self.poisson_lambda = poisson_lambda
        self.passenger_requests = {floor: 0 for floor in self.building.floors}
        self.total_steps = 0
        self.max_steps = 100
        self.idle_moves = 0
        self.wait_time_sum = 0

    def generate_passenger_requests(self):
        for floor in self.building.floors:
            arrivals = np.random.poisson(self.poisson_lambda)
            if arrivals > 0:
                self.passenger_requests[floor] += arrivals
                logging.info(f"{arrivals} new request(s) at floor {floor}.")

    def get_state(self):
        return (self.elevator.current_floor,
                tuple(self.passenger_requests[f] for f in self.building.floors))

    def compute_reward(self):
        cf = self.elevator.current_floor
        if self.passenger_requests[cf] > 0:
            served = self.passenger_requests[cf]
            reward = 10 * served
            logging.info(f"Serviced {served} passenger(s) at floor {cf}.")
            self.passenger_requests[cf] = 0
        else:
            reward = -1
        # track metrics
        self.wait_time_sum += sum(self.passenger_requests.values())
        if self.elevator.status in ('up', 'down') and reward < 0:
            self.idle_moves += 1
        return reward

    def simulate_step(self, action: str):
        self.elevator.step(action)
        self.generate_passenger_requests()
        reward = self.compute_reward()
        self.total_steps += 1
        done = self.total_steps >= self.max_steps
        state = self.get_state()
        info = {'step': self.total_steps}
        return state, reward, done, info

    def reset(self):
        self.elevator.current_floor = 1
        self.elevator.status = 'idle'
        self.passenger_requests = {f: 0 for f in self.building.floors}
        self.total_steps = 0
        self.idle_moves = 0
        self.wait_time_sum = 0
        logging.info("Environment reset.")
        return self.get_state()
    

# -------------------------
# Baseline Scheduler
# -------------------------
class BaselineScheduler:
    def __init__(self, env: ElevatorSimulation):
        # Store the simulation environment instance
        self.env = env

    def run_episode(self):
        # Reset the environment to its initial state before running
        state = self.env.reset()
        done = False
        total_reward = 0

        # Continue until the simulation signals completion
        while not done:
            # Unpack the current floor from the state tuple (ignore passenger counts)
            current_floor, _ = state

            # Simple cyclic policy:
            # - If not at the top floor, move up
            # - Otherwise, move down
            if current_floor < self.env.building.floors[-1]:
                action = 'up'
            else:
                action = 'down'

            # Execute the chosen action:
            #   - Elevator moves
            #   - New passenger requests are generated
            #   - Reward is computed
            #   - State is updated
            state, reward, done, _ = self.env.simulate_step(action)

            # Accumulate the total reward over the episode
            total_reward += reward

        # After the episode ends, return:
        # 1) total_reward: sum of all rewards received
        # 2) wait_time_sum: cumulative waiting time of all passengers
        # 3) idle_moves: number of moves made without servicing any passenger
        return total_reward, self.env.wait_time_sum, self.env.idle_moves


# -------------------------
# Q-Learning Agent
# -------------------------
class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.99,
                 epsilon=1.0, eps_decay=0.995, eps_min=0.01):
        # Initialize Q-table as a dict of {state: {action: value}}
        # New states default to all-zero action values
        self.q_table = defaultdict(lambda: {a: 0.0 for a in actions})
        # Learning rate: how much new information overrides old
        self.alpha = alpha
        # Discount factor: how much to value future rewards
        self.gamma = gamma
        # Exploration rate: probability of choosing a random action
        self.epsilon = epsilon
        # Rate at which epsilon decays after each episode
        self.eps_decay = eps_decay
        # Minimum value for epsilon (always allow some exploration)
        self.eps_min = eps_min
        # List of possible actions (e.g., ['up', 'down', 'idle'])
        self.actions = actions

    def choose_action(self, state):
        # ε-greedy action selection:
        # With probability ε, pick a random action (explore)
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        # Otherwise, pick the action with highest Q-value (exploit)
        q_vals = self.q_table[state]
        return max(q_vals, key=q_vals.get)

    def learn(self, s, a, r, s_next):
        # Q-Learning update:
        # q_predict = current estimate Q(s, a)
        q_predict = self.q_table[s][a]
        # q_target = reward + γ * max_a' Q(s_next, a')
        q_target = r + self.gamma * max(self.q_table[s_next].values())
        # Update Q(s, a) towards the target by learning rate α
        self.q_table[s][a] += self.alpha * (q_target - q_predict)

    def decay_epsilon(self):
        # Reduce exploration rate, but keep at least eps_min
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)


# Training & Evaluation
def train_and_evaluate(episodes=200):
    env = ElevatorSimulation(num_floors=10, delay=0, poisson_lambda=1.0)
    agent = QLearningAgent(actions=['up', 'down', 'idle'])
    baseline = BaselineScheduler(env)

    # baseline performance
    b_r, b_wait, b_idle = baseline.run_episode()
    logging.info(f"Baseline (initial): Reward={b_r}, WaitTime={b_wait}, IdleMoves={b_idle}")

    # train agent
    metrics = []
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.simulate_step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        agent.decay_epsilon()
        metrics.append((ep, total_reward, env.wait_time_sum, env.idle_moves))
        logging.info(f"Episode {ep}: Reward={total_reward}, WaitTime={env.wait_time_sum}, Idle={env.idle_moves}")

    logging.info("Training completed.")
    return (b_r, b_wait, b_idle), agent, metrics


def plot_metrics(baseline_results, metrics):
    # Unpack baseline metrics: (total_reward, cumulative_wait_time, idle_move_count)
    b_r, b_wait, b_idle = baseline_results

    # Extract per-episode data from metrics list
    episodes   = [m[0] for m in metrics]  # episode indices
    rewards    = [m[1] for m in metrics]  # total reward per episode
    wait_times = [m[2] for m in metrics]  # cumulative wait time per episode
    idle_moves = [m[3] for m in metrics]  # idle move count per episode

    # 1. Plot total reward over episodes
    plt.figure()
    # Use a solid line with circle markers for the RL agent’s rewards
    plt.plot(episodes, rewards,
             linestyle='-', marker='o',
             label='RL Agent Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward vs. Episode')
    plt.grid(True)
    plt.legend()

    # 2. Compare average waiting time: Baseline vs. RL agent
    avg_wait_rl = sum(wait_times) / len(wait_times)
    plt.figure()
    # Bar chart: Baseline in gray, RL agent in skyblue
    plt.bar(['Baseline', 'RL Agent'],
            [b_wait, avg_wait_rl],
            color=['gray', 'skyblue'])
    plt.ylabel('Cumulative Wait Time')
    plt.title('Baseline vs. RL: Average Wait Time')

    # 3. Compare idle moves over episodes
    plt.figure()
    # RL agent: solid blue line with circle markers
    plt.plot(episodes, idle_moves,
             linestyle='-', marker='o',
             label='RL Agent Idle Moves')
    # Baseline: single horizontal dashed line in black
    plt.hlines(b_idle,
               xmin=episodes[0], xmax=episodes[-1],
               colors='black', linestyles='--',
               label='Baseline Idle Moves')
    plt.xlabel('Episode')
    plt.ylabel('Idle Moves')
    plt.title('Idle Moves: RL Agent vs. Baseline')
    plt.grid(True)
    plt.legend()

    # Display all figures
    plt.show()


if __name__ == "__main__":
    baseline_results, agent, metrics = train_and_evaluate(episodes=100)
    # Re-print baseline at end to ensure visibility
    b_r, b_wait, b_idle = baseline_results
    logging.info(f"Baseline (final): Reward={b_r}, WaitTime={b_wait}, IdleMoves={b_idle}")
    # metrics can be saved or plotted for result analysis
    plot_metrics(baseline_results, metrics)
