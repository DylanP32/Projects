
class Direction():
    UP = 1
    DOWN = -1
    IDLE = 0

# An elevator has a current_floor, a direction,
# a destination, a destination queue, and stops
class Elevator():
    
    # upon instantiation, an elevator starts at the first floor, is "IDLE",
    # has no destination, and empty destination queue and no stops
    def __init__(self, elevator_num):
        self.elevator_ID = elevator_num
        self.current_floor: int = 1
        self.direction: Direction = Direction.IDLE
        self.destination: list[int | Direction] = None # current destination
        self.stop: list[int | Direction] = None # current stop
        
        
        # destinations will be added to the queue once the elevator reaches the stop floor location
        # destination queue holds the direction its heading as well as the destination floor
        # destination will only be added as a ordered list where self.destination 
        self.destinations_queue: list[list[Direction | int]] = []
        # an elevator will only make a stop if it is between the current_floor
        # and the destination.
        # stops queue holds the floor to stop at, the direction its heading, and the destination floor
        self.stops_queue: list[int | Direction] = []
        
        self.stops_achieved = 0
        self.destinations_achieved = 0
    
    
    # elevator_ID getter
    def elevator_ID(self):
        return self._elevator_ID
    
    # current_floor getter and setter
    @property
    def current_floor(self): 
        return self._current_floor
    
    @current_floor.setter
    def current_floor(self, value):
        self._current_floor = value
    
    # direction getter and setter
    @property
    def direction(self):
        return self._direction
    
    @direction.setter
    def direction(self, value):
        self._direction = value
    
    @property
    def destination(self) -> list[ int | Direction]:
        return self._destination
    
    @destination.setter
    def destination(self, value):
        self._destination = value
    
    # stop getter and setter
    @property
    def stop(self) -> list[int | Direction]:
        return self._stop
    
    @stop.setter
    def stop(self, value):
        self._stop = value
    
    
    # calculates the distance from the elevator to the stop location with respect to its current path
    def check_distance(self, stop_location, destination_direction):
        
        temp_distance: int = 0 # total distance to the stop location
        
        # handle idle state: simple distance between floors
        if self.direction is Direction.IDLE:
            return abs(self.current_floor - stop_location)
        
        # add a penalty for each stop and each destination already in the queue 
        # (prioritizes elevators with less places it needs to be)
        if len(self.stops_queue) > 0:
            temp_distance += (2 * len(self.stops_queue))
        if len(self.destinations_queue) > 0:
            temp_distance += len(self.destinations_queue)
        
        # handle direction being up
        if self.direction is Direction.UP:
            
            # check if the stop is above the current position
            if stop_location >= self.current_floor:
                temp_distance += abs(self.current_floor - stop_location) # stop is on the way
            else:
                return None
        
        # handle direction being down
        elif self.direction is Direction.DOWN:
            
            # check if the stop is below the current position
            if stop_location <= self.current_floor:
                temp_distance += abs(self.current_floor - stop_location) # stop is on the way
            else:
                return None

        # check if the destination is in the same direction as current direction
        if self.direction is destination_direction:
            return temp_distance
        
        return None
    
    # updates the direction of the elevator
    def update_direction(self):
        if self.destination:
            self.direction = self.destination[1]
        elif self.stop:
            if (self.stop[0] > self.current_floor):
                self.direction = Direction.UP
            else:
                self.direction = Direction.DOWN
        else:
            self.direction = Direction.IDLE
            return False
        return True
    
    # updates stop location as well as stops_queue
    def update_stop(self, instruction=None):
        
        if instruction:
            if self.stop is None:
                self.stop = instruction
            else:
                self.stop, self.stops_queue = self.insert(self.stop, self.stops_queue, instruction)
            return
        
        if self.stops_queue:
            self.stop = self.stops_queue.pop(0)
    
    # updates destination as well as destinations_queue
    def update_destination(self, instruction=None):
        
        if instruction:
            if self.destination is None:
                self.destination = instruction
            else:
                self.destination, self.destinations_queue = self.insert(self.destination, self.destinations_queue, instruction)
            return
        
        if self.destinations_queue:
            self.destination = self.destinations_queue.pop(0)
    
    def insert(self, current, queue, instruction):
        
        if self.direction is Direction.UP:
            if instruction[0] < current[0]:
                temp = current
                current = instruction
                return self.insert(current, queue, temp)

            else:
                if queue:
                    for i, element in enumerate(queue):
                        if instruction[0] < element[0]:
                            queue.insert(i, instruction)
                            break
                else:
                    queue.append(instruction)
        
        elif self.direction is Direction.DOWN:
            if instruction[0] > current[0]:
                temp = current
                current = instruction
                return self.insert(current, queue, temp)
            
            else:
                if queue:
                    for i, element in enumerate(queue):
                        if instruction[0] > element[0]:
                            queue.insert(i, instruction)
                            break
                else:
                    queue.append(instruction)
        
        return current, queue
    
    # maintains movement of the elevator
    def move(self):
        while self.destination and (self.current_floor == self.destination[0]):
            self.destinations_achieved += 1
            self.destination = None
            self.update_destination()
        
        while self.stop and (self.current_floor == self.stop[0]):
            self.stops_achieved += 1
            
            # retrieve the destination floor and the direction then
            # set it as the new destination
            next_destination = (self.stop.pop(2), self.stop.pop(1))
            self.update_destination(next_destination)
            
            self.stop = None
            self.update_stop()
        
        moving = self.update_direction()
        
        # returns false if there is nowhere else for the elevator to go 
        if not moving:
            return False
        
        self.step()
        
        # true if still moving
        return True
    
    # allows the elevators to move a floor up or down
    def step(self):
        if (self.destination is not None) or (self.stop is not None):
            if self.direction is Direction.UP:
                self.current_floor += 1
            elif self.direction is Direction.DOWN:
                self.current_floor -= 1
    
    def __str__(self):
        
        return (
            f"Elevator {self.elevator_ID}\n"
            f"  current_floor = {self.current_floor}\n"
            f"  direction = '{self.direction}'\n"
            f"  stop = {self.stop},\n"
            f"  stops_queue = {self.stops_queue}\n"
            f"  destination = {self.destination}\n"
            f"  destinations_queue = {self.destinations_queue}\n"
            f"\n  stops achieved = {self.stops_achieved}\n"
            f"  destinations achieved = {self.destinations_achieved}\n"
        )