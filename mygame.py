# Heavily inspired by https://github.com/Sanil2108/Medium/blob/master/nn_playing_game/part3/game.py

import pygame

clock_tick = 250

def run():
    global clock_tick
    
    pygame.init()
    pygame.font.init()
    caption = "My nice game window"
    gameDisplay = pygame.display.set_mode((800, 600))
    pygame.display.set_caption(caption)
    clock = pygame.time.Clock()
    crashed = False

    gameDisplay.fill("#ffffff")
    score = 0

    crashed = False
    gameEnded = False

    # main game loop
    while not crashed and not gameEnded:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crash = True

        #update(gameDisplay, player)

        pygame.display.update()
        clock.tick(clock_tick)

    pygame.quit()
    quit()

def controlled_run(wrapper, counter):
    gameEnded = False
    pygame.init()
    pygame.font.init()
    caption = "My nice game window"
    gameDisplay = pygame.display.set_mode((800, 600))
    pygame.display.set_caption(caption)
    clock = pygame.time.Clock()
    crashed = False

    # player = Player.Player()

if __name__ == '__main__':
	run()
