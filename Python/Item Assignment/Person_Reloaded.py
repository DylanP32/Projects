#####################################################################
# author: Dylan Pellegrin
# date:4/1/24
# description: Person class that creates a screen and a pygame surface
#####################################################################
import pygame
from random import randint, choice
from Item import *
from Constants import *

class Person(pygame.sprite.Sprite, Item):
    def __init__(self):
        
        # initializes the sprite. Sets color and surf size
        Item.__init__(self)
        self.color = COLORS[randint(0,4)]
        self.surf = pygame.Surface((self.size, self.size))
        self.rect = self.surf.get_rect()
        self.surf.fill(self.color)
        
        # sets x and y values
        self.setRandomPosition()
        self.x_val = self.rect.x
        self.y_val = self.rect.y
    
    # color randomizer
    def setColor(self):
        
        self.color = COLORS[randint(0,4)]
        self.surf.fill(self.color)
    
    # size randomizer``
    def setSize(self):
        
        self.size = randint(10,100)
        self.surf = pygame.Surface((self.size, self.size))
    
    # detects key presses and moves surf accoridngly
    def update(self, pressed_keys):
        
        if pressed_keys[K_UP]:
            self.rect.move_ip(0, -1)
            self.goUp()
        if pressed_keys[K_DOWN]:
            self.rect.move_ip(0, 1)
            self.goDown()
        if pressed_keys[K_LEFT]:
            self.rect.move_ip(-1, 0)
            self.goLeft()
        if pressed_keys[K_RIGHT]:
            self.rect.move_ip(1, 0)
            self.goRight()
        if pressed_keys[K_SPACE]:
            self.setSize()
            self.setColor()

        # keeps player on the screen
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > WIDTH:
            self.rect.right = WIDTH
        if self.rect.top <= 0:
            self.rect.top = 0
        if self.rect.bottom >= HEIGHT:
            self.rect.bottom = HEIGHT
    
    # sets a random position
    def setRandomPosition(self):
        
        self.x_val = randint(0, 1000)
        self.rect.x = self.x_val
        self.y_val = randint(0, 800)
        self.rect.y = self.y_val
    
    # calculates the distance from the middle of the
    def getPosition(self):
        
        x_val = self.rect.x - (self.size/2)
        y_val = self.rect.y - (self.size/2)
        position = (x_val, y_val )
        return position
    
    def __str__(self):
        
        s = f"{self.name}\tsize = {self.size},   x = {self.x_val},   y = {self.y_val},   color: {self.color}"
        return s

########################### main game ################################

# Initialize pygame library and display
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Create a person object
p = Person()
RUNNING = True # A variable to determine whether to get out of the

# infinite game loop
while (RUNNING):
    # Look through all the events that happened in the last frame to see
    # if the user tried to exit.
    for event in pygame.event.get():
        if (event.type == KEYDOWN and event.key == K_ESCAPE):
            RUNNING = False
        elif (event.type == QUIT):
            RUNNING = False
        elif (event.type == KEYDOWN and event.key == K_SPACE):
            print(p)
    
    # Otherwise, collect the list/dictionary of all the keys that were
    # pressed
    pressedKeys = pygame.key.get_pressed()
    # and then send that dictionary to the Person object for them to
    # update themselves accordingly.
    p.update(pressedKeys)
    # fill the screen with a color
    screen.fill(WHITE)
    # then transfer the person to the screen
    screen.blit(p.surf, p.getPosition())
    pygame.display.flip()
