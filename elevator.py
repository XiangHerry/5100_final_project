import time

class Building: 
    def __init__(self, floors = 0):
        self.floors = list(range(1, floors + 1)) 
    def __repr__(self):
        return f"building with floors: {self.floors}"
    

class Elevator: 
    def __init__(self, building, start_floor=1):
        self.building = building
        self.current_floor = start_floor
        self.status = "idle"
    
    def move_up(self):
        if self.current_floor < self.building.floors[-1]:
            self.current_floor += 1
            self.status = "up"
        else: 
            print("already in the highest level, cannot go up more")
        
    def move_down(self):
        if self.current_floor > self.building.floors[0]:
            self.current_floor -= 1
            self.status = 'down'
        else:
            print("already in the lowest level, cannot do down")
    
    def idle(self):
        self.status = 'idle'
    
    def step(self, direction):
        """
        mock single move, direction can be "up", "down" or "idle"
        """
        if direction == 'up':
            self.move_up()
        elif direction == 'down':
            self.move_down()
        elif direction == 'idle':
            self.idle()
        else:
            print("invalid. Please choose 'up'、'down' or 'idle'")
            return
        
        time.sleep(5)
        print(f"current floor：{self.current_floor}，status：{self.status}")

if __name__ == "__main__":
    building = Building()
    print(building)
    
    # assume start from first
    elevator = Elevator(building)
    
    elevator.step('up')
    elevator.step('up')
    elevator.step('down')
    elevator.step('idle')

