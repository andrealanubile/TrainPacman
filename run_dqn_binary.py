import sys
import copy
import csv
import torch
import pygame
from pygame.locals import *
import numpy as np
from collections import defaultdict, deque
from constants import *
from pacman import Pacman
from nodes import NodeGroup
from pellets import PelletGroup
from ghosts import GhostGroup1
from fruit import Fruit
from pauser import Pause
from text import TextGroup
from sprites import LifeSprites
from sprites import MazeSprites
from mazedata import MazeData
from dqn_agent import DQNAgent
from print_progressbar import print_progress_bar

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
        self.state_space = (4, 36, 28)  # assuming grid size (channels, height, width)
        self.action_space = 4  # UP, DOWN, LEFT, RIGHT
        self.agent = DQNAgent(self.state_space, self.action_space)
        self.simulation_count = 0
        self.max_simulations = 10000
        self.last_action = None
        self.state = None
        self.last_state = None
        self.last_done = None
        self.last_action = None
        self.last_reward = None

        self.grid = None
        self.initial_pellet_positions = set()
        self.counter_target = 0
        self.last_time = 0
        self.final_scores = []
        self.action_number_target = 0
        self.action_number_exp = 0

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
        self.ghosts = GhostGroup1(self.nodes.getStartTempNode(), self.pacman)

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

        self.state = self.update_grid_binary()
        self.last_state = self.update_grid_binary()
        self.last_time = pygame.time.get_ticks()
        print(self.agent.exploration_rate)

    def setBackground(self):
        self.background_norm = pygame.surface.Surface(SCREENSIZE).convert()
        self.background_norm.fill(BLACK)
        self.background_flash = pygame.surface.Surface(SCREENSIZE).convert()
        self.background_flash.fill(BLACK)
        if self.level > 0:
            self.background_norm = self.mazesprites.constructBackground(self.background_norm, self.level-1 % 5)
        else:
            self.background_norm = self.mazesprites.constructBackground(self.background_norm, self.level % 5)
        self.background_flash = self.mazesprites.constructBackground(self.background_flash, 5)
        self.flashBG = False
        self.background = self.background_norm

    def resetGame(self):
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


    def update(self):
        
        # for row in self.grid_bin_pacman:
        #     print(row)
        #     # print('/n')
        
        if self.action_number_target == 6000:
            self.agent.update_target_model()
            self.action_number_target = 0

        if self.action_number_exp == 3000:
            self.agent.exploration_rate = max(self.agent.exploration_rate * self.agent.exploration_decay, self.agent.exploration_min)  # Decrease exploration rate after each game
            print(self.agent.exploration_rate)
            self.action_number_exp = 0

        self.action_number_target = self.action_number_target + 1
        self.action_number_exp = self.action_number_exp + 1
 
        if self.simulation_count >= self.max_simulations:
            self.agent.save_model('dqn_model.pth')
            print('Policy Saved')
            self.save_scores_to_csv()
            pygame.quit()
            sys.exit()
            return

        dt = 1
        self.textgroup.update(dt)
        self.pellets.update(dt)

        current_time = pygame.time.get_ticks()
        duration = (current_time - self.last_time) / 1000.0  # Convert duration to seconds
        self.duration = duration

        if self.pacman.alive:
            if not self.pause.paused:
                self.ghosts.update(dt)
                if self.fruit is not None:
                    self.fruit.update(dt)
                self.checkPelletEvents()
                self.checkGhostEvents()
                self.checkFruitEvents()
                
                if self.pacman.alive:
                    if self.last_reward is not None:
                        self.agent.remember(self.last_state, self.last_action, self.last_reward, self.state, self.last_done)

                    state = self.update_grid_binary()
                    # Agent chooses action
                    action = self.agent.choose_action_binary(state)
                    self.last_action = action
                    self.pacman.update(dt, action)
                    
                    # Get new state and reward
                    new_state = self.update_grid_binary()
                    # print(np.array_equal(new_state_stack, self.state_stack))
                    self.checkGhostEvents()
                    self.last_reward = self.get_reward()
                    # print(reward)
                    self.last_done = self.pacman.alive
                    # Store the experience in replay memory
                    self.last_state = copy.deepcopy(self.state)
                    self.state = copy.deepcopy(new_state)
                else:
                    reward = self.get_reward()
                    done = self.pacman.alive
                    self.agent.remember(self.last_state, self.last_action, reward, self.state, done)



                # Train the agent with the experience
                self.agent.experience_replay_binary(64)

        else:
            done = not self.pacman.alive
            # self.state_stack = copy.deepcopy(new_state_stack)
            if self.lives <= 0:
                self.simulation_count += 1
                self.final_scores.append((self.score, self.duration))
                print_progress_bar(self.simulation_count, self.max_simulations, prefix='Progress:', suffix='Complete', length=50)  # Update the progress bar
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
        """Convert the grid to a state representation for Q-learning."""
        mapping = {'x': 1.0/8, '.': 2.0/8, 'o': 3.0/8,'P': 4.0/8, 'G': 5.0/8, '/': 0.0, 's': 6.0/8, 'f' : 7.0/8, 'F' : 8.0/8}
        state = [[mapping[cell] for cell in row] for row in self.grid]
        return np.array(state, dtype=np.float32).reshape(1, 36, 28)
    
    def grid_to_state_walls(self):
        """Convert the grid to a state representation for Q-learning."""
        mapping = {'x': 1.0, '.': 0.0, 'o': 0.0,'P': 0.0, 'G': 0.0, '/': 0.0, 's': 0.0, 'f' : 0.0, 'F' : 0.0}
        state = [[mapping[cell] for cell in row] for row in self.grid]
        return np.array(state, dtype=np.float32).reshape(1, 36, 28)
    
    def grid_to_state_pellet(self):
        """Convert the grid to a state representation for Q-learning."""
        mapping = {'x': 0.0, '.': 1.0, 'o': 1.0,'P': 0.0, 'G': 0.0, '/': 0.0, 's': 0.0, 'f' : 0.0, 'F' : 0.0}
        state = [[mapping[cell] for cell in row] for row in self.grid]
        return np.array(state, dtype=np.float32).reshape(1, 36, 28)
    
    def grid_to_state_Pacman(self):
        """Convert the grid to a state representation for Q-learning."""
        mapping = {'x': 0.0, '.': 0.0, 'o': 0.0,'P': 1.0, 'G': 0.0, '/': 0.0, 's': 0.0, 'f' : 0.0, 'F' : 0.0}
        state = [[mapping[cell] for cell in row] for row in self.grid]
        return np.array(state, dtype=np.float32).reshape(1, 36, 28)
    
    def grid_to_state_Ghost(self):
        """Convert the grid to a state representation for Q-learning."""
        mapping = {'x': 0.0, '.': 0.0, 'o': 0.0,'P': 0.0, 'G': 1.0, '/': 0.0, 's': 0.0, 'f' : 0.0, 'F' : 0.0}
        state = [[mapping[cell] for cell in row] for row in self.grid]
        return np.array(state, dtype=np.float32).reshape(1, 36, 28)
    
    def grid_to_state_empty(self):
        """Convert the grid to a state representation for Q-learning."""
        mapping = {'x': 0.0, '.': 0.0, 'o': 0.0,'P': 0.0, 'G': 0.0, '/': 1.0, 's': 0.0, 'f' : 0.0, 'F' : 0.0}
        state = [[mapping[cell] for cell in row] for row in self.grid]
        return np.array(state, dtype=np.float32).reshape(1, 36, 28)
    
    def grid_to_state_GhostScared(self):
        """Convert the grid to a state representation for Q-learning."""
        mapping = {'x': 0.0, '.': 0.0, 'o': 0.0,'P': 0.0, 'G': 0.0, '/': 0.0, 's': 1.0, 'f' : 0.0, 'F' : 0.0}
        state = [[mapping[cell] for cell in row] for row in self.grid]
        return np.array(state, dtype=np.float32).reshape(1, 36, 28)
    
    def grid_to_state_GhostEaten(self):
        """Convert the grid to a state representation for Q-learning."""
        mapping = {'x': 0.0, '.': 0.0, 'o': 0.0,'P': 0.0, 'G': 0.0, '/': 0.0, 's': 0.0, 'f' : 1.0, 'F' : 0.0}
        state = [[mapping[cell] for cell in row] for row in self.grid]
        return np.array(state, dtype=np.float32).reshape(1, 36, 28)
    
    def grid_to_state_Fruit(self):
        """Convert the grid to a state representation for Q-learning."""
        mapping = {'x': 0.0, '.': 0.0, 'o': 0.0,'P': 0.0, 'G': 0.0, '/': 0.0, 's': 0.0, 'f' : 0.0, 'F' : 1.0}
        state = [[mapping[cell] for cell in row] for row in self.grid]
        return np.array(state, dtype=np.float32).reshape(1, 36, 28)


    def get_reward(self):
        """Calculate the reward based on the game state."""
        diff_score = self.score - self.pre_score  # Calculate score difference

        if diff_score == 0:
            reward = -0.1

        reward = diff_score
        if not self.pacman.alive:
            reward = -10000000000000  # Negative reward for dying
        elif self.pellets.isEmpty():
            reward = 100000000000000  # Positive reward for collecting all pellets
        # print(reward)
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
        self.save_scores_to_csv()
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
        self.grid = [list(line.replace(' ', '')) for line in self.transformed_text.split('\n') if line.strip()]
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
            if pellet.name == 'POWERPELLET':
                self.grid[ply][plx] = "o" 
            else:
                self.grid[ply][plx] = "."

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

    def update_grid_binary(self):
        """Update the grid with the current state of the maze."""
        # Reset the grid to its initial state
        # Determine the dimensions of the initial grid
        num_rows = len(self.initial_grid)
        num_cols = len(self.initial_grid[0]) if num_rows > 0 else 0

        # Create a matrix of all zeros with the same dimensions as the initial grid
        self.grid_bin_pellet = [[0 for _ in range(num_cols)] for _ in range(num_rows)]
        self.grid_bin_fruit = [[0 for _ in range(num_cols)] for _ in range(num_rows)]
        self.grid_bin_ghost = [[0 for _ in range(num_cols)] for _ in range(num_rows)]
        self.grid_bin_pacman = [[0 for _ in range(num_cols)] for _ in range(num_rows)]


        # Update pellet positions
        current_pellet_positions = {(int(pellet.position.x // TILEWIDTH), int(pellet.position.y // TILEHEIGHT)) for pellet in self.pellets.pelletList}

        for pellet in self.pellets.pelletList:
            plx, ply = int(pellet.position.x // TILEWIDTH), int(pellet.position.y // TILEHEIGHT)
            if pellet.name == 'POWERPELLET':
                self.grid_bin_pellet[ply][plx] = 1 
            else:
                self.grid_bin_pellet[ply][plx] = 1

        # Update fruit position
        if self.fruit is not None:
            fx, fy = int(self.fruit.position.x // TILEWIDTH), int(self.fruit.position.y // TILEHEIGHT)
            self.grid_bin_fruit[fy][fx] = 1

        # Update ghost positions
        for ghost in self.ghosts:
            gx, gy = int(ghost.position.x // TILEWIDTH), int(ghost.position.y // TILEHEIGHT)
            if ghost.mode.current == FREIGHT:
                self.grid_bin_ghost[gy][gx] = 1  # Ghosts in frightened mode
            elif ghost.mode.current == SPAWN:
                self.grid_bin_ghost[gy][gx] = 1  # Ghosts returning to the spawn point
            else:
                self.grid_bin_ghost[gy][gx] = 1  # Normal mode

        # Update Pac-Man position
        px, py = int(self.pacman.position.x // TILEWIDTH), int(self.pacman.position.y // TILEHEIGHT)
        self.grid_bin_pacman[py][px] = 1

        state = np.stack([self.grid_bin_fruit, self.grid_bin_ghost, self.grid_bin_pacman, self.grid_bin_pellet])

        return state

    def save_scores_to_csv(self):
        """Save the final scores to a CSV file."""
        with open('final_scores.csv', 'w', newline='') as csvfile:
            score_writer = csv.writer(csvfile)
            score_writer.writerow(['Episode', 'Score','Duration'])
            for i, (score,duration) in enumerate(self.final_scores):
                score_writer.writerow([i+1, score, duration])

    def print_state(self):
        print('Last state:')
        for row in self.last_state:
            str = np.array2string(row, threshold = np.inf)
            print(str)

    def plot_scores_and_durations(scores, durations):
        episodes = list(range(1, len(scores) + 1))

        plt.figure(figsize=(12, 6))

        # Plot scores
        plt.subplot(1, 2, 1)
        plt.plot(episodes, scores, label='Score')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title('Scores over Episodes')
        plt.legend()

        # Plot durations
        plt.subplot(1, 2, 2)
        plt.plot(episodes, durations, label='Duration', color='orange')
        plt.xlabel('Episode')
        plt.ylabel('Duration (s)')
        plt.title('Durations over Episodes')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def compare_state_dicts(self,state_dict1, state_dict2):
        if state_dict1.keys() != state_dict2.keys():
            print("The keys in the state dictionaries do not match.")
            return False

        for key in state_dict1.keys():
            if not torch.equal(state_dict1[key], state_dict2[key]):
                print(f"Parameter {key} differs.")
                return False

        print("All parameters match.")
        return True


if __name__ == "__main__":
    game = GameController()
    game.startGame()
    while True:
        game.update()
