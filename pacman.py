import pygame
import random
from pygame.locals import *
from vector import Vector2
from constants import *
from entity import Entity
from sprites import PacmanSprites

class Pacman(Entity):
    def __init__(self, node):
        Entity.__init__(self, node )
        self.name = PACMAN    
        self.color = YELLOW
        self.direction = LEFT
        self.setBetweenNodes(LEFT)
        self.alive = True
        self.sprites = PacmanSprites(self)

    def reset(self):
        Entity.reset(self)
        self.direction = LEFT
        self.setBetweenNodes(LEFT)
        self.alive = True
        self.image = self.sprites.getStartImage()
        self.sprites.reset()

    def die(self):
        self.alive = False
        self.direction = STOP

    def update(self, dt, action):	
        self.sprites.update(dt)
        self.direction = self.getValidKey(action)
        print(action)
        self.target = self.getNewTarget(self.direction)
        self.position += self.directions[self.direction]*self.speed*dt
        
        if self.overshotTarget():
            self.node = self.target
            if self.node.neighbors[PORTAL] is not None:
                self.node = self.node.neighbors[PORTAL]
            
            # if self.target is not self.node:
            #     self.direction = direction
            # else:
            # if self.target is self.node:
            #     self.target = self.getNewTarget(self.direction)
            self.direction = STOP

            self.setPosition()
        else: 
            if self.oppositeDirection(self.direction):
                self.reverseDirection()

    def getValidKey(self,action):
    #    key_pressed = pygame.key.get_pressed()
        if action == 'UP':
            return UP
        if action == 'DOWN':
            return DOWN
        if action == 'LEFT':
            return LEFT
        if action == 'RIGHT':
            return RIGHT
        return STOP  
    
    # def get_random_direction(self):
    #     """Return a random direction for Pacman."""
    #     directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    #     return random.choice(directions)


    def eatPellets(self, pelletList):
        for pellet in pelletList:
            if self.collideCheck(pellet):
                return pellet
        return None    
    
    def collideGhost(self, ghost):
        return self.collideCheck(ghost)

    def collideCheck(self, other):
        d = self.position - other.position
        dSquared = d.magnitudeSquared()
        rSquared = (self.collideRadius + other.collideRadius)**2
        if dSquared <= rSquared:
            return True
        return False
