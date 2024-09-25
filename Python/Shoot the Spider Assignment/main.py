###################################################
# name: Dylan Pellegrin
# date: 4/23/24
# description: Shoot the spider game
###################################################

# RUN THIS FILE

# Import other files.
from Constants import *
from Sprites import *
 
# Create the screen object.
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Shoot The Spider")
# Setup framerate.
clock = pygame.time.Clock()
# Instantiate player and add to sprite group.
player = Jet()
allSprites.add(player)

endGameCounter = 0 # Makes sure that the end game timer is only set once.

# Main loop.
running = True
while running:
    
    # Look at every event in the queue.
    for event in pygame.event.get():
        
        # Did the user hit a key?
        if event.type == KEYDOWN:
            
            # Was it the Escape key? If so, stop the loop.
            if event.key == K_ESCAPE:
                running = False
            # Was it the Space key? If so,instantiate a missile.
            if event.key == K_SPACE:
                # Create a new missile and add it to sprite groups only if game is still going.
                if (level < 15) and (lives > 0):
                    newMissile = Missile(player.getPosition("X"), player.getPosition("Y"))
                    missiles.add(newMissile)
                    allSprites.add(newMissile)

                    # When level 4 comes, the player can shoot 3 missiles total.
                    if level >= 4:
                        newMissile = Missile(player.getPosition("X"), player.getPosition("Y")+30)
                        missiles.add(newMissile)
                        allSprites.add(newMissile)
                        
                        newMissile = Missile(player.getPosition("X"), player.getPosition("Y")-30)
                        missiles.add(newMissile)
                        allSprites.add(newMissile)

                    # At level 7, the player can shoot 5 missiles total.
                    if level >= 7:
                        newMissile = Missile(player.getPosition("X"), player.getPosition("Y")+50)
                        missiles.add(newMissile)
                        allSprites.add(newMissile)
                        
                        newMissile = Missile(player.getPosition("X"), player.getPosition("Y")-50)
                        missiles.add(newMissile)
                        allSprites.add(newMissile)
        
        # Did the user click the window close button? If so, stop the loop.
        elif event.type == QUIT:
            running = False
        
        # Increases spawn rate of spiders at certain levels
        elif event.type == NEXTLEVEL:
            level += 1
            if level == 2:
                pygame.time.set_timer(ADDSPIDER,750)
            if level == 3:
                pygame.time.set_timer(ADDSPIDER, 500)
            if level == 4:
                pygame.time.set_timer(ADDSPIDER, 250)
            if level == 5:
                pygame.time.set_timer(ADDSPIDER, 100)
            if level >= 10:
                pygame.time.set_timer(ADDSPIDER, 50)
        
        # Add a new spider?
        elif event.type == ADDSPIDER:
            # Create the new spider and add it to sprite groups.
            newSpider = Spider()
            spiders.add(newSpider)
            allSprites.add(newSpider)
        
        # Add a new cloud?
        elif event.type == ADDCLOUD:
            # Create the new cloud and add it to sprite groups.
            newCloud = Cloud()
            clouds.add(newCloud)
            allSprites.add(newCloud)
        
        # Ends game after 30 seconds of the Win/Lose screen.
        elif event.type == ENDGAME:
            player.kill()
            running = False
    
    # Get the set of keys pressed and check for user input.
    pressed_keys = pygame.key.get_pressed()
    player.update(pressed_keys)
    
    # Update spider, missile, and cloud positions.
    # Also updates explosion animation frames
    spiders.update()
    missiles.update()
    clouds.update()
    explosions.update()
    # Fill the screen with Light Blue.
    screen.fill((LIGHT_BLUE))
    
    # Draw all sprites.
    for entity in allSprites:
        screen.blit(entity.surf, entity.rect)
    screen.blit(text, textRect)
    # Draws explosions.
    explosions.draw(screen)

    # Check if any entities have collided with each other.
    for spider in spiders:
        
        if pygame.sprite.collide_rect(player, spider):
            # If so, then remove a player life.
            spider.kill()
            lives -= 1
        
        for missile in missiles:
            
            if pygame.sprite.collide_rect(spider, missile):
                
                # Creates an explosion where the missile touches the spider.
                explosion = Explosion(missile.getPosition("X"),missile.getPosition("Y"))
                explosions.add(explosion)
                
                # Removes the spider and missile that touched.
                spider.kill()
                missile.kill()
                
                # Adds score for every spider killed.
                score += 100
    
    # If the player reaches level 15, initiate Win. If the player has no lives, initiate lose
    if level == 15 or lives <= 0:
        
        # Sets off game timer once.
        while endGameCounter < 1:
            endGameTimer()
            endGameCounter += 1
        
        # Stop spawning spiders and changing the level as well as remove all spiders from the screen.
        ADDSPIDER = 0
        NEXTLEVEL = 0
        for spider in spiders:
            spider.kill()
        
        # Update livesScoreLevel to be blank.
        text = livesScoreLevel.render(f"", True, BLACK)
        
        # Display win or lose text (if lose kill player) and update rect position.
        if level == 15:
            winLoseText = winLose.render(f"You Win!", True, BLACK)
        if lives == 0:
            winLoseText = winLose.render(f"GAME OVER", True, BLACK)
            player.kill()
        winLoseTextRect = winLoseText.get_rect(center = (WIDTH/2,HEIGHT/2))
        
        # Display final score.
        finalScoreText = finalScore.render(f"Score: {score}", True, BLACK)
        finalScoreTextRect = finalScoreText.get_rect(center = (WIDTH/2,(HEIGHT/2) +70))
        
        # Screen blit both texts.
        screen.blit(winLoseText, winLoseTextRect)
        screen.blit(finalScoreText, finalScoreTextRect)
          
    # Otherwise continue displaying other info.
    else:
        text = livesScoreLevel.render(f"Lives: {lives} Score: {score} Level {level}", True, BLACK)
    
    # Update the display 
    pygame.display.flip()
    clock.tick(30)