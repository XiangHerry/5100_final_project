import time
import random
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(message)s')

# -------------------------
# Building Class
# -------------------------
class Building:
    def __init__(self, num_floors: int):
        # Validate input: building must have at least 2 floors.
        if num_floors < 2:
            raise ValueError("A building must have at least 2 floors.")
        # Create a list of floors numbered 1 through num_floors.
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
        self.status = "idle"  # status can be "idle", "up", or "down"
        self.delay = delay    # configurable delay (seconds) for each move
    
    def move_up(self):
        if self.current_floor < self.building.floors[-1]:
            self.current_floor += 1
            self.status = "up"
            logging.info("Moving up...")
        else:
            # Refined boundary condition: if at top, stay and log a warning.
            logging.warning("Already at the highest floor; cannot move up.")
    
    def move_down(self):
        if self.current_floor > self.building.floors[0]:
            self.current_floor -= 1
            self.status = "down"
            logging.info("Moving down...")
        else:
            # Refined boundary condition: if at bottom, stay and log a warning.
            logging.warning("Already at the lowest floor; cannot move down.")
    
    def idle(self):
        self.status = 'idle'
        logging.info("Staying idle.")
    
    def step(self, direction: str):
        """
        Executes a single move in the specified direction.
        direction: "up", "down", or "idle"
        Returns: None (state updates and logging only)
        """
        if direction == 'up':
            self.move_up()
        elif direction == 'down':
            self.move_down()
        elif direction == 'idle':
            self.idle()
        else:
            logging.error("Invalid direction. Choose 'up', 'down', or 'idle'.")
            return
        
        # Configurable delay to simulate movement time
        if self.delay:
            time.sleep(self.delay)
        
        logging.info(f"Current floor: {self.current_floor}, Status: {self.status}")

# -------------------------
# Elevator Simulation Environment
# -------------------------
class ElevatorSimulation:
    def __init__(self, num_floors: int, delay: float = 0):
        # Initialize the building and elevator
        self.building = Building(num_floors)
        self.elevator = Elevator(self.building, start_floor=1, delay=delay)
        
        # Initialize passenger requests for each floor (0 means no waiting passengers)
        # We'll simulate passenger arrivals randomly.
        self.passenger_requests = {floor: 0 for floor in self.building.floors}
        self.total_steps = 0
        self.max_steps = 100  # Example episode length
    
    def generate_passenger_requests(self, arrival_prob: float = 0.3):
        """
        For each floor, with probability arrival_prob, add a passenger request.
        """
        for floor in self.building.floors:
            if random.random() < arrival_prob:
                self.passenger_requests[floor] += 1
                logging.info(f"New passenger request at floor {floor}.")
    
    def get_state(self):
        """
        Returns the current state.
        For simplicity, we define state as a tuple:
          (current_floor, tuple of passenger_requests per floor)
        """
        state = (self.elevator.current_floor, tuple(self.passenger_requests[floor] for floor in self.building.floors))
        return state
    
    def compute_reward(self):
        """
        Reward function:
          - If the elevator is at a floor with waiting passengers, give a positive reward.
          - Else, give a negative reward for waiting.
        After serving, reset the request count at that floor.
        """
        current_floor = self.elevator.current_floor
        if self.passenger_requests[current_floor] > 0:
            # Reward for servicing waiting passengers (reward proportional to number served)
            reward = 10 * self.passenger_requests[current_floor]
            logging.info(f"Serviced {self.passenger_requests[current_floor]} request(s) at floor {current_floor}.")
            self.passenger_requests[current_floor] = 0  # Reset requests after service
        else:
            reward = -1  # Penalty for time passing without service
        return reward
    
    def simulate_step(self, action: str):
        """
        Performs a simulation step:
          1. Process the elevator action.
          2. Generate new passenger requests.
          3. Compute the reward.
          4. Return the new state, reward, done flag, and info.
        """
        # Execute elevator movement based on the action.
        self.elevator.step(action)
        
        # Simulate passenger arrival on each floor.
        self.generate_passenger_requests(arrival_prob=0.3)
        
        # Compute reward for the current step.
        reward = self.compute_reward()
        
        # Update step count and check if simulation is done.
        self.total_steps += 1
        done = self.total_steps >= self.max_steps
        
        # Construct the state tuple.
        state = self.get_state()
        info = {"step": self.total_steps}
        
        return state, reward, done, info
    
    def reset(self):
        """
        Resets the simulation environment.
        """
        self.elevator.current_floor = 1
        self.elevator.status = "idle"
        self.passenger_requests = {floor: 0 for floor in self.building.floors}
        self.total_steps = 0
        logging.info("Environment reset.")
        return self.get_state()

# -------------------------
# Main Execution Example
# -------------------------
if __name__ == "__main__":
    # Create a simulation environment for a 10-floor building
    sim = ElevatorSimulation(num_floors=10, delay=1)  # set delay=1 sec for demonstration
    
    # Reset simulation
    state = sim.reset()
    logging.info(f"Initial state: {state}")
    
    # Demonstrate a sequence of actions
    actions = ['up', 'up', 'down', 'idle', 'up', 'idle']
    for action in actions:
        state, reward, done, info = sim.simulate_step(action)
        logging.info(f"Action: {action} | New state: {state} | Reward: {reward} | Info: {info}")
        if done:
            logging.info("Episode finished.")
            break