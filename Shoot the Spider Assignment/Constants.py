#####################################################################
# name: Dylan Pellegrin
# description: All constants
#####################################################################

# Import libraries needed for all files.
import pygame
import random
from random import randint
from itertools import cycle
import os

# Constants for screen size.
WIDTH = 1000
HEIGHT = 800
# Constants for colors.
RED = [0xe3, 0x1b, 0x23]
BLUE = [0x00,0x2F,0x8B]
GREY = [0xA2, 0xAA, 0xAD]
WHITE = [0xFF, 0xFF, 0xFF]
BLACK = [0x00, 0x00, 0x00]
LIGHT_BLUE = [0x87, 0xCE, 0xFA]

COLORS = [BLUE, RED, GREY, WHITE, BLACK, LIGHT_BLUE]
# Keys from pygame.
from pygame.locals import (
    RLEACCEL,
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
    K_SPACE,
)

# Initialize pygame.
pygame.init()
# Create custom events for adding a new enemy, a cloud, change level, and to end game after win or lose.
ADDSPIDER = pygame.USEREVENT + 1
pygame.time.set_timer(ADDSPIDER, 1000)
ADDCLOUD = pygame.USEREVENT + 2
pygame.time.set_timer(ADDCLOUD, 1000)
NEXTLEVEL = pygame.USEREVENT + 3
pygame.time.set_timer(NEXTLEVEL, 10000)
ENDGAME = pygame.USEREVENT + 4
# Sets game event timer to end game after 20 seconds when Win/Lose are initiated.
def endGameTimer():
    pygame.time.set_timer(ENDGAME, 20000)

# Create groups to hold enemy sprites, cloud sprites, and all sprites.
# - enemies is used for collision detection and position updates.
# - missilies is used for collision detection and position updates.
# - clouds is used for position updates.
# - explosions is used for animation rendering.
# - all_sprites is used for rendering.
spiders = pygame.sprite.Group()
missiles = pygame.sprite.Group()
clouds = pygame.sprite.Group()
explosions = pygame.sprite.Group()
allSprites = pygame.sprite.Group()

# Set lives and scores to starting values.
lives = 5
score = 0
level = 1

# Creates text box to display Lives, Score, and level
livesScoreLevel = pygame.font.Font(os.path.join("imgs", "Pixel.ttf"), 30)
text = livesScoreLevel.render(f"Lives: {lives} Score: {score} Level {level}", True, BLACK)
textRect = text.get_rect(center = (235,HEIGHT-45))

winLose = pygame.font.Font(os.path.join("imgs", "Pixel.ttf"), 70)
finalScore = pygame.font.Font(os.path.join("imgs", "Pixel.ttf"), 50)