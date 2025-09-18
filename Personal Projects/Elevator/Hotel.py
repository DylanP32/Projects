import random
from Elevator import Direction, Elevator

# A hotel has n floors and k elevators
class Hotel():
    
    def __init__(self, floors, total_elevators):
        self.total_floors: int = floors
        self.elevators: list = []
        
        for i in range(total_elevators):
            self.elevators.append(Elevator(i + 1))
        
        self.stops_queue: list[list[int | Direction]] = []
        
        # for i in self.elevators:
        #     print(i)
        #     print(f"current floor: {i.current_floor}  direction: {i.direction}  destination: {i.destination}")
        #     print()
    
    # gives directions to the elevator until there are none left
    def call_elevators(self):
        running = True
        run = 0
        while running:
            run += 1
            print(f"\nRun {run}")
            
            self.move_elevators()
            
            
            num = random.randint(1, 60)
            
            if self.stops_queue:
                if num % 3 == 0:
                    least_floors_away = None
                    selected_elevator = None
                    
                    stop_location = self.stops_queue[0][0]
                    destination_direction = self.stops_queue[0][1]
                    
                    print(f"new instruction: {self.stops_queue[0]}\n")
                    
                    for elevator in self.elevators:
                        
                        temp_distance = elevator.check_distance(stop_location, destination_direction)
                        print(f"Distance from Elevator {elevator.elevator_ID}: {temp_distance}")
                        
                        if temp_distance is not None:
                            if selected_elevator is None:
                                least_floors_away = temp_distance
                                selected_elevator = elevator
                            elif temp_distance < least_floors_away:
                                least_floors_away = temp_distance
                                selected_elevator = elevator
                    
                    if selected_elevator:
                        
                        print()
                        print(f"Least floors away: {least_floors_away}")
                        print(f"Selected elevator: Elevator {selected_elevator.elevator_ID} and current floor: {selected_elevator.current_floor}")
                        print()
                        
                        selected_elevator.update_stop(self.stops_queue[0])
                        self.stops_queue.pop(0)
                
                # for elevator in self.elevators:
                #     print(f"\nElevator {elevator.elevator_ID}'s curr floor: {elevator.current_floor}")
                #     print(f"Elevator {elevator.elevator_ID}'s curr stop: {elevator.stop}")
                #     print(f"Elevator {elevator.elevator_ID}'s stops_queue: {elevator.stops_queue}")
                #     print(f"Elevator {elevator.elevator_ID}'s curr destination: {elevator.destination}")
                #     print(f"Elevator {elevator.elevator_ID}'s destinations_queue: {elevator.destinations_queue}\n")
                #     print(f"  stops achieved: {elevator.stops_achieved}\n")
                #     print(f"  destinations achieved: {elevator.destinations_achieved}\n")
                # print("--------------------------------------------------------------------------------------------")
            
            else:
                idle_elevators = 0
                for elevator in self.elevators:
                    if elevator.direction == Direction.IDLE:
                        idle_elevators += 1
                if(idle_elevators == len(self.elevators)):
                    running = False
            
        
        self.print()
    
    
    def move_elevators(self):
        
        for elevator in self.elevators:
            elevator.move()
        print("\nMoving elevators:")
        for elevator in self.elevators:
            if elevator.direction != Direction.IDLE:
                print(f"Elevator {elevator.elevator_ID},")
        print("\n")
    
    def print(self):

        print("\nThe Hotel\n")
        for elevator in self.elevators:
            print(elevator)