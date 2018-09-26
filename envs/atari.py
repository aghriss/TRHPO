# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 12:59:20 2018

@author: gamer
"""
from ale_python_interface import ALEInterface
import utils.env as utils
import numpy as np
import collections

OPTIONS = {"IMAGES_SIZE":(80,80)}
CROP = {"breakout":(32,10,8,8)}

class ALE(ALEInterface):
    
    def __init__(self,game_name, render=True):
        
        super(ALE, self).__init__()    
        
        self.crop = CROP[game_name]
        self.num_frames = num_frames
        self.load_rom(game_name,render)
        self.load_params()
        self._actions_raw = self.getMinimalActionSet().tolist()  
        self.action_space = Discrete(len(self._actions_raw))
        self.observation_space = Box(0,255,OPTIONS["IMAGES_SIZE"]+(self.num_frames,))
  

    def load_params(self):
        self._start_lives = self.lives()
        self._current_state = np.zeros(self.observation_space.shape) 
 
    def load_rom(self,rom_file,render):

        self.setInt(str.encode('random_seed'), 123)
        self.setFloat(str.encode('repeat_action_probability'), 0.0)        
        self.setBool(str.encode('sound'), False)
        self.setBool(str.encode('display_screen'), render)
        self.loadROM(str.encode("./roms/"+utils.game_name(rom_file)))
        
    def get_current_state(self):
        up,down,left,right = self.crop
        return utils.process_frame(self.getScreenRGB()[up:-down,left:-right], OPTIONS["IMAGES_SIZE"]))
    
    def step(self,action):        
        
        reward = 0
        assert action in range(self.actions_n), "Action not available"
        
        reward = self.act(action)
            
        
        state = self.get_current_state()
        
        return state, reward, self.lives() != self._start_lives, None

    def reset(self):
        self.reset_game()
        self.load_params()
	return self.get_current_state()
    
    def clone(self):
        env = self.cloneSystemState()
        env.params()
        return env

