# name: Dylan Pellegrin
# date: 3/10/25
# description: The goal of this program is to practice python by 
# emulating the way an elevator algorithmically operates

from Hotel import Hotel
from Elevator import Direction
import random

# get user input and then initialize hotel
floors: int = int(input("How many floors does the hotel have? "))
elevators: int = int(input("How many elevators does the hotel have? "))
The_Hotel: object = Hotel(floors, elevators)

# instructions have (in order) an int: stop floor, a string: direction of destination (UP or DOWN),
# and an int: destination floor
# randomize instructions
for i in range(random.randint(20, 80)):
    floor1 = random.randint(1, floors)
    floor2 = random.randint(1,floors)
    while floor2 == floor1: # make sure the two floors are different
        floor2 = random.randint(1, floors)
    if floor2 < floor1:
        direction = Direction.DOWN
    elif floor2 > floor1:
        direction = Direction.UP
    
    # add instructions to the hotel's queue
    The_Hotel.stops_queue.append([floor1, direction, floor2])


# call the main loop
The_Hotel.call_elevators()
