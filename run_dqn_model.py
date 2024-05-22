import sys
import pygame
from pygame.locals import *
import torch
import numpy as np
from constants import *
from pacman import Pacman
from nodes import NodeGroup
from pellets import PelletGroup
from ghosts import GhostGroup
from fruit import Fruit
from pauser import Pause
from text import TextGroup
from sprites import LifeSprites
from sprites import MazeSprites
from mazedata import MazeData
from dqn_agent import DQNAgent

class GameController(object):
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(SCREENSIZE, 0, 32)
        self.background = None
        self.background_norm = None
        self.background_flash = None
        self.clock = pygame.time.Clock()
        self.fruit = None
        self.pause = Pause(False)  # Start the game unpaused
        self.level = 0
        self.lives = 5
        self.score = 0
        self.pre_score = 0
        self.textgroup = TextGroup()
        self.lifesprites = LifeSprites(self.lives)
        self.flashBG = False
        self.flashTime = 0.2
        self.flashTimer = 0
        self.fruitCaptured = []
        self.fruitNode = None
        self.mazedata = MazeData()
        self.state_space = 28 * 36  # assuming grid size
        self.action_space = 4  # UP, DOWN, LEFT, RIGHT
        self.agent = DQNAgent(self.state_space, self.action_space)
        self.agent.load_model('dqn_model.pth')  # Load the trained model
        self.grid = None
        self.initial_pellet_positions = set()

    def startGame(self):
        print('Starting game')
        self.mazedata.loadMaze(self.level)
        self.mazesprites = MazeSprites(self.mazedata.obj.name + ".txt", self.mazedata.obj.name + "_rotation.txt")
        self.setBackground()
        self.nodes = NodeGroup(self.mazedata.obj.name + ".txt")
        self.mazedata.obj.setPortalPairs(self.nodes)
        self.mazedata.obj.connectHomeNodes(self.nodes)
        self.pacman = Pacman(self.nodes.getNodeFromTiles(*self.mazedata.obj.pacmanStart))
        self.pellets = PelletGroup(self.mazedata.obj.name + ".txt")
        self.ghosts = GhostGroup(self.nodes.getStartTempNode(), self.pacman)

        self.ghosts.pinky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 3)))
        self.ghosts.inky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(0, 3)))
        self.ghosts.clyde.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(4, 3)))
        self.ghosts.setSpawnNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 3)))
        self.ghosts.blinky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 0)))

        self.nodes.denyHomeAccess(self.pacman)
        self.nodes.denyHomeAccessList(self.ghosts)
        self.ghosts.inky.startNode.denyAccess(RIGHT, self.ghosts.inky)
        self.ghosts.clyde.startNode.denyAccess(LEFT, self.ghosts.clyde)
        self.mazedata.obj.denyGhostsAccess(self.ghosts, self.nodes)

        # Load and initialize grid from the maze file
        if not self.load_and_initialize_grid(self.mazedata.obj.name + ".txt"):
            print("Failed to initialize grid.")
            return

        # Store initial pellet positions
        for pellet in self.pellets.pelletList:
            plx, ply = int(pellet.position.x // TILEWIDTH), int(pellet.position.y // TILEHEIGHT)
            self.initial_pellet_positions.add((plx, ply))

        self.update_grid()

    def setBackground(self):
        self.background_norm = pygame.surface.Surface(SCREENSIZE).convert()
        self.background_norm.fill(BLACK)
        self.background_flash = pygame.surface.Surface(SCREENSIZE).convert()
        self.background_flash.fill(BLACK)
        self.background_norm = self.mazesprites.constructBackground(self.background_norm, self.level % 5)
        self.background_flash = self.mazesprites.constructBackground(self.background_flash, 5)
        self.flashBG = False
        self.background = self.background_norm

    def update(self):
        self.update_grid()
        dt = self.clock.tick(60) / 1000.0  # Update frame rate for faster updates
        self.textgroup.update(dt)
        self.pellets.update(dt)
        if not self.pause.paused:
            self.ghosts.update(dt)
            if self.fruit is not None:
                self.fruit.update(dt)
            self.checkPelletEvents()
            self.checkGhostEvents()
            self.checkFruitEvents()

        if self.pacman.alive:
            if not self.pause.paused:
                # Get current state
                state = tuple(self.grid_to_state())
                # Agent chooses action
                action = self.agent.choose_action(state)
                self.pacman.update(dt, action)
                # Update game state
                self.update_grid()
        else:
            self.pacman.update(dt, self.agent.last_action)
            if self.lives <= 0:
                self.resetGame()
            else:
                self.lives -= 1
                self.pacman.reset()

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
        self.pre_score = self.score  # Save previous score
        self.checkEvents()
        self.render()

    def grid_to_state(self):
        """Convert the grid to a state representation for DQN."""
        return tuple([cell for row in self.grid for cell in row])

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
            if self.pellets.numEaten == 30:
                self.ghosts.inky.startNode.allowAccess(RIGHT, self.ghosts.inky)
            if self.pellets.numEaten == 70:
                self.ghosts.clyde.startNode.allowAccess(LEFT, self.ghosts.clyde)
            self.pellets.pelletList.remove(pellet)
            if pellet.name == POWERPELLET:
                self.ghosts.startFreight()
            if self.pellets.isEmpty():
                self.flashBG = True
                self.hideEntities()
                self.pause.setPause(pauseTime=3, func=self.nextLevel)

    def checkGhostEvents(self):
        for ghost in self.ghosts:
            if self.pacman.collideGhost(ghost):
                if ghost.mode.current is FREIGHT:
                    self.pacman.visible = False
                    ghost.visible = False
                    self.updateScore(ghost.points)                  
                    self.textgroup.addText(str(ghost.points), WHITE, ghost.position.x, ghost.position.y, 8, time=1)
                    self.ghosts.updatePoints()
                    self.pause.setPause(pauseTime=1, func=self.showEntities)
                    ghost.startSpawn()
                    self.nodes.allowHomeAccess(ghost)
                elif ghost.mode.current is not SPAWN:
                    if self.pacman.alive:
                        self.lives -=  1
                        self.lifesprites.removeImage()
                        self.pacman.die()               
                        self.ghosts.hide()
                        if self.lives <= 0:
                            print(self.lives)
                            self.textgroup.showText(GAMEOVERTXT)
                            self.pause.setPause(pauseTime=3, func=self.restartGame)
                        else:
                            self.pause.setPause(pauseTime=3, func=self.resetLevel)
    
    def checkFruitEvents(self):
        if self.pellets.numEaten == 50 or self.pellets.numEaten == 140:
            if self.fruit is None:
                self.fruit = Fruit(self.nodes.getNodeFromTiles(9, 20), self.level)
                print(self.fruit)
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
        self.pause.paused = True
        self.startGame()
        self.textgroup.updateLevel(self.level)

    def restartGame(self):
        self.level = 0
        self.lives = 5
        self.score = 0
        self.pre_score = 0  # Reset previous score
        self.textgroup.updateScore(self.score)
        self.textgroup.updateLevel(self.level)
        self.lifesprites.resetLives(self.lives)
        self.fruitCaptured = []
        self.startGame()
        self.showEntities()

    def resetLevel(self):
        self.pause.paused = False
        self.pacman.reset()
        self.ghosts.reset()
        self.fruit = None
        self.textgroup.showText(READYTXT)

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
        with open(filename, 'r') as file:
            content = file.read()
        
        translation_table = str.maketrans({
            'X': 'x', '0': 'x', '1': 'x', '2': 'x', '3': 'x',
            '4': 'x', '5': 'x', '6': 'x', '7': 'x', '8': 'x',
            '9': 'x', '=': 'x', 'n': '/', '-': '/', 'l': '/',
            '.': '.', '+': '.', 'p': 'o','|':'/'
        })
        
        self.transformed_text = content.translate(translation_table)
        
        # Create the grid, keeping the spaces intact
        self.grid = [list(line.replace(' ', '')) for line in self.transformed_text.split('\n')]
        self.initial_grid = [row[:] for row in self.grid]  # Save initial grid state
        return True

    def print_grid(self):
        """Print the current state of the grid."""
        for row in self.grid:
            print("".join(row))

    def update_grid(self):
        """Update the grid with the current state of the maze."""
        # Reset the grid to its initial state
        self.grid = [row[:] for row in self.initial_grid]  # Reset grid to initial state

        # Update pellet positions
        current_pellet_positions = {(int(pellet.position.x // TILEWIDTH), int(pellet.position.y // TILEHEIGHT)) for pellet in self.pellets.pelletList}

        eaten_pellets = self.initial_pellet_positions - current_pellet_positions
        for (plx, ply) in eaten_pellets:
            self.grid[ply][plx] = "/"

        for pellet in self.pellets.pelletList:
            plx, ply = int(pellet.position.x // TILEWIDTH), int(pellet.position.y // TILEHEIGHT)
            if pellet.visible:
                self.grid[ply][plx] = "o" if pellet.name == "POWERPELLET" else "."
            else:
                self.grid[ply][plx] = "/"

        # Update fruit position
        if self.fruit is not None:
            fx, fy = int(self.fruit.position.x // TILEWIDTH), int(self.fruit.position.y // TILEHEIGHT)
            self.grid[fy][fx] = "F"

        # Update ghost positions
        for ghost in self.ghosts:
            gx, gy = int(ghost.position.x // TILEWIDTH), int(ghost.position.y // TILEHEIGHT)
            if ghost.mode.current == FREIGHT:
                self.grid[gy][gx] = "f"  # Ghosts in frightened mode
            elif ghost.mode.current == SPAWN:
                self.grid[gy][gx] = "s"  # Ghosts returning to the spawn point
            else:
                self.grid[gy][gx] = "G"  # Normal mode

        # Update Pac-Man position
        px, py = int(self.pacman.position.x // TILEWIDTH), int(self.pacman.position.y // TILEHEIGHT)
        self.grid[py][px] = "P"

if __name__ == "__main__":
    game = GameController()
    game.startGame()
    while True:
        game.update()
