#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 12:27:48 2018

@author: thinkpad
"""

from base.logger import Logger
CHECK_PATH="./checks/"
import os
import core.console as C
if not os.path.exists(CHECK_PATH):
    os.makedirs(CHECK_PATH)
    
    

class BaseAgent(object):
    name ="BaseAgent"
    def __init__(self,log_file=None,name=""):
        self.name = self.name+name
        if log_file is None:
            log_file = self.name
        self.logger = Logger(log_file+".log")
        self.functions = []

    def act(self,state,train=False):
        
        raise NotImplementedError
    
    def train(self,episodes):
        
        raise NotImplementedError
        
    def save(self):
        for f in self.functions:
            f.save(CHECK_PATH+self.name)
        
    def load(self):
        try:
            for f in self.functions:
                f.load(CHECK_PATH+self.name)
        except:
            C.warning("No Checkpoint Found for:" + CHECK_PATH+self.name)
    def print(self):
        max_l = max(list(map(len,self.history.keys())))
        for k,v in self.history.items():
            print(k+(max_l-len(k))*" ",end="")
            if v is not None:
                print(" : %s"%str(v[-1]))
    def log(self,key,val):
        self.logger.log(key,val)
        
    def play(self,name='play'):        
        name = str(name)+self.env.name+self.name
        state = self.env.reset(record=True)
        done = False
        while not done:            
            action = self.act(state,train=False)
            state, _, done, info = self.env.step(action)
        self.env.save_episode(name)