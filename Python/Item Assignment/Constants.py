import pygame
from random import randint, choice
from Item import *
# constants for screen size
WIDTH = 1000
HEIGHT = 800
# constants for colors
RED = [0xe3, 0x1b, 0x23]
BLUE = [0x00,0x2F,0x8B]
GREY = [0xA2, 0xAA, 0xAD]
GREEN = [0x00, 0xFF, 0x00]
BLACK = [0x00, 0x00, 0x00]
WHITE = [0xFF, 0xFF, 0xFF]
COLORS = [BLUE, RED, GREY, GREEN, BLACK]
# keys from pygame
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
