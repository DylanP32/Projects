#####################################################################
# name: Dylan Pellegrin
# description: All Sprites used
#####################################################################

# Import constants.
from Constants import *

# Jet class. Jet displays an image on the screen
# and can be moved around using arrow keys.
# Jet is bound within the limits of WIDTH and HEIGHT
# Jet's starting location is in the middle left of the screen
# Jet can get its X and Y coordinates (used for Missile spawning)
class Jet(pygame.sprite.Sprite):
    
    def __init__(self):
        super(Jet, self).__init__()
        self.surf = pygame.image.load(os.path.join("imgs", "jet.png")).convert()
        self.surf = pygame.transform.scale(self.surf, (100, 60))
        self.surf.set_colorkey((255, 255, 255), RLEACCEL)
        self.rect = self.surf.get_rect(center=(100,HEIGHT/2,))
    
    # Move the sprite based on user keypresses of arrow keys.
    def update(self, pressed_keys):
        if pressed_keys[K_UP]:
            self.rect.move_ip(0, -10)
        if pressed_keys[K_DOWN]:
            self.rect.move_ip(0, 10)
        if pressed_keys[K_LEFT]:
            self.rect.move_ip(-10, 0)
        if pressed_keys[K_RIGHT]:
            self.rect.move_ip(10, 0)

        # Keep player on the screen.
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > WIDTH:
            self.rect.right = WIDTH
        if self.rect.top <= 0:
            self.rect.top = 0
        if self.rect.bottom >= HEIGHT:
            self.rect.bottom = HEIGHT
    
    # Calculates the X and Y of the middle of the sprite and 
    # Returns X and Y appropiately.
    def getPosition(self, cord:str) -> int:
        if cord == "X":
            return self.rect.x + (self.surf.get_width()/2)
        if cord == "Y":
            return self.rect.y + (self.surf.get_height()/2)

# Spider class. Spider is a pygame sprite which spawns
# in a random location and moves left at a constant speed.
# Spider is bound within the limit of HEIGHT. Spider is removed 
# from the screen once going past x=0
class Spider(pygame.sprite.Sprite):
    
    def __init__(self):
        super(Spider, self).__init__()
        self.surf = pygame.image.load(os.path.join("imgs", "spider.png")).convert_alpha()
        self.surf = pygame.transform.scale(self.surf, (75, 75))
        self.surf.set_colorkey((255, 255, 255), RLEACCEL)
        self.rect = self.surf.get_rect(center=(random.randint(WIDTH + 20, WIDTH + 100),
                                               random.randint(0, HEIGHT),))
        self.speed = 1
    # Move the sprite based on speed.
    # Remove the sprite when it passes the left edge of the screen.
    def update(self):
        self.rect.move_ip(-self.speed, randint(-10, 10))
        if (self.rect.right < 0):
            self.kill()
        # Keep spider on the screen.
        if self.rect.top <= 0:
            self.rect.top = 0
        if self.rect.bottom >= HEIGHT:
            self.rect.bottom = HEIGHT

# Define the Missile object by extending pygame.sprite.Sprite
class Missile(pygame.sprite.Sprite):
    
    def __init__(self, x:int, y:int):
        super(Missile, self).__init__()
        self.surf = pygame.image.load(os.path.join("imgs", "missile.png")).convert()
        self.surf = pygame.transform.scale(self.surf, (55, 20))
        self.surf.set_colorkey((255, 255, 255), RLEACCEL)
        # The starting position is determined by the coordinates of the Player.
        self.rect = self.surf.get_rect(center=(x, y+10))
        self.speed = 10
    
    # Move the sprite based on speed.
    # Remove the sprite when it passes the left edge of the screen.
    def update(self):
        self.rect.move_ip(self.speed, 0)
        if self.rect.left > WIDTH:
            self.kill()
    
    # Calculates the X and Y of the middle of the sprite and 
    # returns X and Y appropiately.
    def getPosition(self, cord:str) -> int:
        if cord == "X":
            return self.rect.x + (self.surf.get_width()/2)
        if cord == "Y":
            return self.rect.y + (self.surf.get_height()/2)

# Define the cloud object by extending pygame.sprite.Sprite
class Cloud(pygame.sprite.Sprite):
    
    def __init__(self):
        super(Cloud, self).__init__()
        self.surf = pygame.image.load(os.path.join("imgs", "cloud.png")).convert()
        self.surf.set_colorkey((0, 0, 0), RLEACCEL)
        # The starting position is randomly generated.
        self.rect = self.surf.get_rect(
            center=(
                random.randint(WIDTH + 20, WIDTH + 100),
                random.randint(0, HEIGHT),
            )
        )
    
    # Move the cloud based on a constant speed.
    # Remove the cloud when it passes the left edge of the screen.
    def update(self):
        self.rect.move_ip(-5, 0)
        if self.rect.right < 0:
            self.kill()

# Define the Explosion object by extending pygame.sprite.Sprite
class Explosion(pygame.sprite.Sprite):
    
    def __init__(self, x:int, y:int):
        super(Explosion, self).__init__()
        # Creates a list of explosion images/frames and then creates
        # surfs out of each image.
        self.images = []
        for num in range(1, 5):
            self.surf = pygame.image.load(os.path.join("imgs", f"exp{num}.png"))
            self.surf = pygame.transform.scale(self.surf, (100, 100))
            self.images.append(self.surf)
        self.index = 0
        self.image = self.images[self.index]
        # Sets explosion coordinates to the coordinates of each missile
        # that makes contact with a spider.
        self.rect = self.image.get_rect(center = [x, y])
        # Counter for the animation.
        self.counter = 0
    
    def update(self):
        # Update explosion animation.
        explosionSpeed = 4
        self.counter += 1
        
        # Animation. Iterates through each image frame every update.
        if self.counter >= explosionSpeed and self.index < len(self.images) - 1:
            self.counter = 0
            self.index += 1
            self.image = self.images[self.index]
        
        # If the animation is complete, reset animation index.
        if self.index >= len(self.images) - 1 and self.counter >= explosionSpeed:
            self.kill()