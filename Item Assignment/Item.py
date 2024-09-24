#####################################################################
# author: Dylan Pellegrin
# date: 3/16/24
# description: Person class. A person has a place on an x y plane
# and can move around
#####################################################################
import math

# global Constants to restrict the maximum x and y values that a person object
# can have.
MAX_X = 1000
MAX_Y = 800

# Person class.
# has a name, x value, y value, and a size.
# has getters and setters with range checking
# can go Left, Right, Up, Down
# can get the distance between 2 players
# has a string magic function to format it's variables to be displayed
class Item:
    
    def __init__(self, name=None, x_val=None, y_val=None):

        self.name = "player 1"
        self.x_val = 0
        self.y_val = 0
        self.size = 50
    
    # name getter and setter
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, value):
        self._name = value
    
    # x value getter and setter
    @property
    def x_val(self):
        return self._x_val
    
    @x_val.setter
    def x_val(self, value):
        self._x_val = value
    
    # y value getter and setter
    @property
    def y_val(self):
        return self._y_val
    
    @y_val.setter
    def y_val(self, value):
        self._y_val = value
    
    # size getter and setter
    @property
    def size(self):
        return self._size
    
    @size.setter
    def size(self, value):
        self._size = value
    
    # directional functions. Allows for movement
    def goRight(self):
        self.x_val += 1
    
    def goLeft(self):
        self.x_val -= 1
    
    def goUp(self):
        self.y_val -= 1
    
    def goDown(self):
        self.y_val += 1
    
    def getDistance(self, other):
        return math.sqrt(math.pow((other.x_val - self.x_val), 2)+math.pow((other.y_val - self.y_val), 2))
    
    # formats instance variables to be printed
    def __str__(self):
        s = f"{self.name}\tsize = {self.size},   x = {self.x_val},   y = {self.y_val}"
        return s

