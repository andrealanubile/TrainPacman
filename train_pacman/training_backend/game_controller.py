import pygame
import time
from pygame.locals import *
from .constants import *
from .pacman import Pacman
from .nodes import Node, NodeGroup
from .pellets import PelletGroup
from .ghosts import GhostGroup, GhostGroup1
from .fruit import Fruit
from .pauser import Pause
from .text import TextGroup
from .sprites import LifeSprites
from .sprites import MazeSprites
from .mazedata import MazeData
import numpy as np
from pathlib import Path

class GameController(object):
    def __init__(self, debug, level, reward_type='pretrain', render=True):
        pygame.init()
        self.script_dir = Path(__file__).parent
        self.background = None
        self.background_norm = None
        self.background_flash = None
        self.clock = pygame.time.Clock()
        self.fruit = None
        self.pause = Pause(False)
        self.level = level
        self.initial_level = level
        self.lives = 5
        self.score = 0
        self.textgroup = TextGroup()
        self.flashBG = False
        self.flashTime = 0.2
        self.flashTimer = 0
        self.fruitCaptured = []
        self.fruitNode = None
        self.mazedata = MazeData()
        self.initial_pellet_positions = set()
        self.debug = debug
        self.reward_type = reward_type
        self.do_render = render
        self.time_since_pellet = 0
        if self.do_render:
            self.screen = pygame.display.set_mode(SCREENSIZE, 0, 32)
            self.lifesprites = LifeSprites(self.lives)


    def setBackground(self):
        self.background_norm = pygame.surface.Surface(SCREENSIZE).convert()
        self.background_norm.fill(BLACK)
        self.background_flash = pygame.surface.Surface(SCREENSIZE).convert()
        self.background_flash.fill(BLACK)
        self.background_norm = self.mazesprites.constructBackground(self.background_norm, self.level%5)
        self.background_flash = self.mazesprites.constructBackground(self.background_flash, 5)
        self.flashBG = False
        self.background = self.background_norm

    def startGame(self):      
        self.mazedata.loadMaze(self.level)
        if self.do_render:
            self.mazesprites = MazeSprites(self.mazedata.obj.name+".txt", self.mazedata.obj.name+"_rotation.txt")
            self.setBackground()
        self.nodes = NodeGroup(self.mazedata.obj.name+".txt")
        self.mazedata.obj.setPortalPairs(self.nodes)
        self.mazedata.obj.connectHomeNodes(self.nodes)
        # self.pacman = Pacman(self.nodes.getNodeFromTiles(*self.mazedata.obj.pacmanStart))
        self.pacman = Pacman(self.nodes.getNodeFromTiles(14, 26), self.do_render)
        self.pellets = PelletGroup(self.mazedata.obj.name+".txt")
        self.ghosts = GhostGroup1(self.nodes.getStartTempNode(), self.pacman, self.do_render)

        # self.ghosts.pinky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 3)))
        # self.ghosts.inky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(0, 3)))
        # self.ghosts.clyde.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(4, 3)))
        self.ghosts.setSpawnNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 3)))
        self.ghosts.blinky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 0)))

        self.nodes.denyHomeAccess(self.pacman)
        self.nodes.denyHomeAccessList(self.ghosts)
        # self.ghosts.inky.startNode.denyAccess(RIGHT, self.ghosts.inky)
        # self.ghosts.clyde.startNode.denyAccess(LEFT, self.ghosts.clyde)
        # self.mazedata.obj.denyGhostsAccess(self.ghosts, self.nodes)

        self.load_and_initialize_grid(self.mazedata.obj.name + ".txt")

        # Store initial pellet positions
        for pellet in self.pellets.pelletList:
            plx, ply = int(pellet.position.x // TILEWIDTH), int(pellet.position.y // TILEHEIGHT)
            self.initial_pellet_positions.add((plx, ply))

        

    def update(self, action, dt=1):
        # dt = self.clock.tick(30) / 1000.0
        actions_mapping = {
            0: 'UP',
            1: 'DOWN',
            2: 'LEFT',
            3: 'RIGHT'
        }
        if self.debug:
            print(f'action: {actions_mapping[action]}')
        self.textgroup.update(dt)
        self.pellets.update(dt)
        pellet_eaten = False
        if not self.pause.paused:
            self.ghosts.update(dt)
            self.checkGhostEvents()     
            # if self.fruit is not None:
            #     self.fruit.update(dt)
            if self.pacman.alive:
                self.pacman.update(dt, actions_mapping[action])
                pellet_eaten = self.checkPelletEvents()
                self.checkGhostEvents()
                # self.checkFruitEvents()
            else:
                pass
            
        if pellet_eaten:
            self.time_since_pellet = 0
        else:
            self.time_since_pellet += 1

        if self.flashBG:
            self.flashTimer += dt
            if self.flashTimer >= self.flashTime:
                self.flashTimer = 0
                if self.background == self.background_norm:
                    self.background = self.background_flash
                else:
                    self.background = self.background_norm

        afterPauseMethod = self.pause.update(dt)
        if afterPauseMethod is not None:
            afterPauseMethod()
        self.checkEvents()

        if self.do_render:
            self.render()

        reward = self.get_reward(pellet_eaten)
        state = self.get_state()
        
        if self.debug:
            print(f'reward: {reward}')
            print(f'pacman loc: {np.argwhere(state[2])}')
            print(f'num pellets: {np.sum(state[1])}')

        done = False
        if (self.lives <= 0) or (self.pellets.isEmpty()):
            done = True

        if not self.pacman.alive and not done:
            self.resetLevel()
            self.checkPelletEvents()
        return reward, state, done

    def get_reward(self, pellet_eaten):
        """Calculate the reward based on the game state."""

        # previous rewards: eat pellet: +10, do nothing: -1, dying: -100, completing stage: +1000
        
        reward = 0.0

        if self.reward_type == 'pretrain':
            if pellet_eaten:
                reward += 1.0
            else:
                # reward -= 0.03*self.time_since_pellet
                pass
        elif self.reward_type == 'hf':
            pass
        else:
            print('Warning: invalid reward type')
            
        if not self.pacman.alive:
            # reward -= 50.0 # Negative reward for dying
            reward -= 1.0
        elif self.pellets.isEmpty():
            # reward += 100.0  # Positive reward for collecting all pellets
            pass
        return reward  # Return the score difference as the reward

    def checkEvents(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                exit() 
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    if self.pacman.alive:
                        self.pause.setPause(playerPaused=True)
                        if not self.pause.paused:
                            self.textgroup.hideText()
                            self.showEntities()
                        else:
                            self.textgroup.showText(PAUSETXT)
                            #self.hideEntities()

    def checkPelletEvents(self):
        pellet = self.pacman.eatPellets(self.pellets.pelletList)
        if pellet:
            self.pellets.numEaten += 1
            self.updateScore(pellet.points)
            # if self.pellets.numEaten == 30:
            #     self.ghosts.inky.startNode.allowAccess(RIGHT, self.ghosts.inky)
            # if self.pellets.numEaten == 70:
            #     self.ghosts.clyde.startNode.allowAccess(LEFT, self.ghosts.clyde)
            self.pellets.pelletList.remove(pellet)
            if pellet.name == POWERPELLET:
                self.ghosts.startFreight()
            if self.pellets.isEmpty():
                self.flashBG = True
                self.hideEntities()
                # self.pause.setPause(pauseTime=3, func=self.nextLevel)
            return True
        else:
            return False

    def checkGhostEvents(self):
        for ghost in self.ghosts:
            if self.pacman.collideGhost(ghost):
                if ghost.mode.current is FREIGHT:
                    self.pacman.visible = False
                    ghost.visible = False
                    self.updateScore(ghost.points)                  
                    self.textgroup.addText(str(ghost.points), WHITE, ghost.position.x, ghost.position.y, 8, time=1)
                    self.ghosts.updatePoints()
                    self.pause.setPause(pauseTime=0, func=self.showEntities)
                    ghost.startSpawn()
                    self.nodes.allowHomeAccess(ghost)
                elif ghost.mode.current is not SPAWN:
                    if self.pacman.alive:
                        self.lives -=  1
                        if self.do_render:
                            self.lifesprites.removeImage()
                        self.pacman.die()               
                        self.ghosts.hide()
                        if self.lives <= 0:
                            pass
                            # self.textgroup.showText(GAMEOVERTXT)
                            # self.pause.setPause(pauseTime=3, func=self.restartGame)
                            # self.pause.setPause(pauseTime=0)
                        else:
                            # self.pause.setPause(pauseTime=0, func=self.resetLevel)
                            pass
    
    def checkFruitEvents(self):
        if self.pellets.numEaten == 50 or self.pellets.numEaten == 140:
            if self.fruit is None:
                self.fruit = Fruit(self.nodes.getNodeFromTiles(9, 20), self.level)
                # print(self.fruit)
        if self.fruit is not None:
            if self.pacman.collideCheck(self.fruit):
                self.updateScore(self.fruit.points)
                self.textgroup.addText(str(self.fruit.points), WHITE, self.fruit.position.x, self.fruit.position.y, 8, time=1)
                fruitCaptured = False
                for fruit in self.fruitCaptured:
                    if fruit.get_offset() == self.fruit.image.get_offset():
                        fruitCaptured = True
                        break
                if not fruitCaptured:
                    self.fruitCaptured.append(self.fruit.image)
                self.fruit = None
            elif self.fruit.destroy:
                self.fruit = None

    def showEntities(self):
        self.pacman.visible = True
        self.ghosts.show()

    def hideEntities(self):
        self.pacman.visible = False
        self.ghosts.hide()

    def nextLevel(self):
        self.showEntities()
        self.level += 1
        self.pause.paused = False
        self.startGame()
        self.textgroup.updateLevel(self.level)

    def restartGame(self):
        self.lives = 5
        self.level = self.initial_level
        self.pause.paused = False
        self.fruit = None
        self.startGame()
        self.score = 0
        self.textgroup.updateScore(self.score)
        self.textgroup.updateLevel(self.level)
        self.textgroup.showText(READYTXT)
        if self.do_render:
            self.lifesprites.resetLives(self.lives)
        self.fruitCaptured = []
        self.time_since_pellet = 0

    def resetLevel(self):
        self.pause.paused = False
        self.pacman.reset()
        self.ghosts.reset()
        self.fruit = None
        self.textgroup.showText(READYTXT)
        self.time_since_pellet = 0

    def updateScore(self, points):
        self.score += points
        self.textgroup.updateScore(self.score)

    def render(self):
        self.screen.blit(self.background, (0, 0))
        #self.nodes.render(self.screen)
        self.pellets.render(self.screen)
        if self.fruit is not None:
            self.fruit.render(self.screen)
        self.pacman.render(self.screen)
        self.ghosts.render(self.screen)
        self.textgroup.render(self.screen)

        for i in range(len(self.lifesprites.images)):
            x = self.lifesprites.images[i].get_width() * i
            y = SCREENHEIGHT - self.lifesprites.images[i].get_height()
            self.screen.blit(self.lifesprites.images[i], (x, y))

        for i in range(len(self.fruitCaptured)):
            x = SCREENWIDTH - self.fruitCaptured[i].get_width() * (i+1)
            y = SCREENHEIGHT - self.fruitCaptured[i].get_height()
            self.screen.blit(self.fruitCaptured[i], (x, y))

        pygame.display.update()


    def load_and_initialize_grid(self, filename):
        """Load the maze from a text file and initialize the grid."""
        with open(self.script_dir / 'assets' / filename, 'r') as file:
            content = file.read()
        
        # translation_table = str.maketrans({
        #     'X': 'x', '0': 'x', '1': 'x', '2': 'x', '3': 'x',
        #     '4': 'x', '5': 'x', '6': 'x', '7': 'x', '8': 'x',
        #     '9': 'x', '=': 'x', 'n': '/', '-': '/', 'l': '/',
        #     '.': '.', '+': '.', 'p': 'o','|':'/'
        # })
        
        # self.transformed_text = content.translate(translation_table)
        self.transformed_text = content
        
        # Create the grid, keeping the spaces intact
        self.grid = [list(line.replace(' ', '')) for line in self.transformed_text.split('\n') if line.strip()]
        self.initial_grid = np.array([row[:] for row in self.grid])  # Save initial grid state
        self.rows_use = np.where(np.logical_not(np.all(self.initial_grid == 'X', axis=1)))[0]
        self.cols_use = np.where(np.logical_not(np.all(self.initial_grid == 'X', axis=0)))[0]

        return True


    def get_state(self):
        """Update the grid with the current state of the maze."""
        # Reset the grid to its initial state
        # Determine the dimensions of the initial grid
        num_rows = len(self.initial_grid)
        num_cols = len(self.initial_grid[0]) if num_rows > 0 else 0

        # Create a matrix of all zeros with the same dimensions as the initial grid
        # self.grid_bin_pellet = [[0 for _ in range(num_cols)] for _ in range(num_rows)]
        # self.grid_bin_fruit = [[0 for _ in range(num_cols)] for _ in range(num_rows)]
        # self.grid_bin_ghost = [[0 for _ in range(num_cols)] for _ in range(num_rows)]
        # self.grid_bin_pacman = [[0 for _ in range(num_cols)] for _ in range(num_rows)]

        state = np.zeros((4, num_rows, num_cols))

        # Wall state
        state[0][np.char.isnumeric(self.initial_grid)] = 1


        # Update pellet positions
        # current_pellet_positions = {(int(pellet.position.x // TILEWIDTH), int(pellet.position.y // TILEHEIGHT)) for pellet in self.pellets.pelletList}

        for pellet in self.pellets.pelletList:
            plx, ply = int(pellet.position.x // TILEWIDTH), int(pellet.position.y // TILEHEIGHT)
            if pellet.name == 'POWERPELLET':
                state[1, ply, plx] = 1 
            else:
                state[1, ply, plx] = 1

        # Update fruit position
        # if self.fruit is not None:
        #     fx, fy = int(self.fruit.position.x // TILEWIDTH), int(self.fruit.position.y // TILEHEIGHT)
        #     self.grid_bin_fruit[fy][fx] = 1

        # Update ghost positions
        for ghost in self.ghosts:
            gx, gy = int(ghost.position.x // TILEWIDTH), int(ghost.position.y // TILEHEIGHT)
            if ghost.mode.current == FREIGHT:
                state[3,gy,gx] = 1  # Ghosts in frightened mode
            elif ghost.mode.current == SPAWN:
                state[3,gy,gx] = 1  # Ghosts returning to the spawn point
            else:
                state[3,gy,gx] = 1  # Normal mode

        # Update Pac-Man position
        px, py = int(self.pacman.position.x // TILEWIDTH), int(self.pacman.position.y // TILEHEIGHT)
        state[2, py, px] = 1

        state = state[:, np.min(self.rows_use):np.max(self.rows_use)+1, np.min(self.cols_use):np.max(self.cols_use)+1]

        return state


# if __name__ == "__main__":
#     game = GameController()
#     game.startGame()
#     while True:
#         game.update()