#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 15:27:24 2018

@author: thinkpad
"""

import numpy as np
import matplotlib.pyplot as plt

from base.spaces import Discrete, Continuous

        
        
class SimpleGRID(object):

    name = "GRID"

    def __init__(self, grid_size=16, max_time=5000,square_size=2):
        
        self.square = square_size
        self.grid_size = grid_size
        self.max_time = max_time
        self.board = np.zeros((self.grid_size, self.grid_size))
        self.wall = np.zeros((self.grid_size, self.grid_size))
        self.state = np.zeros((self.grid_size*self.square,self.grid_size*self.square,3),dtype=np.int32)
        self.action_space = Discrete(4)
        self.observation_space = Continuous((self.grid_size*self.square,self.grid_size*self.square,3))
        
        
        self.wall[0,:] = self.wall[:,-1] = 1
        step = self.grid_size//4
        self.wall[2*step,:]=1
        self.wall[2*step,step:step+2]=0
        self.wall[2*step,3*step:3*step+2]=0
        self.wall = np.maximum(self.wall,self.wall.T)
        

    def get_screen(self):

        self.state = self.state*0
        self.state[::self.square][:,::self.square][self.board>0,0] = 255
        self.state[::self.square][:,::self.square][self.board<0,2] = 255
        self.state[::self.square][:,::self.square][self.x, self.y] = 255

        for i in range(self.square-1):
            self.state[i+1::self.square] = self.state[::self.square]
            self.state[:,i+1::self.square] = self.state[:,::self.square]
        return self.state
    def get_state(self):
        return self.get_screen()
    
    def step(self, action):

        reward = 0
        oldx,oldy = self.x,self.y
        if action == 0:
                self.x = self.x + 1
        elif action == 1:
                self.x = self.x - 1
        elif action == 2:
                self.y = self.y + 1
        elif action == 3:
                self.y = self.y - 1
        else:
            RuntimeError('Error: action not recognized')
        if self.board[self.x,self.y]<0:
            self.x,self.y = oldx,oldy
        reward += self.board[self.x,self.y]
        absorbed = (self.x == self.mouse_x and self.y == self.mouse_y)
        game_over = absorbed or self.t > self.max_time
        self.t = self.t + 1
 
        return self.get_state(), reward, game_over, False

    def reset(self):

        """This function resets the game and returns the initial state"""
        
        self.start = True
        self.t = 0
        self.board *= 0
        self.board[self.wall==1] = -1

        self.x = 3
        self.y = 3

        self.add_mouse()
        
        return self.get_state()
        
    def add_mouse(self):
        
        self.mouse_x,self.mouse_y = self.x,self.y
        
        self.mouse_x = self.grid_size-3
        self.mouse_y = self.grid_size-3

        self.board[self.mouse_x,self.mouse_y] = 1

    def get_mouse(self):
        return np.array([self.mouse_x,self.mouse_y])

    def get_cat(self):
        return np.array([self.x,self.y])
    
    def render(self):
        plt.imshow(self.get_screen())