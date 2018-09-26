#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 12:25:24 2018

@author: thinkpad
"""
import skvideo.io
import skimage.color
#import skimage.io
import matplotlib.pyplot as plt
import skimage.transform
from collections import deque
import numpy as np
import core.utils as U
from base.spaces import Continuous
PLAY_PATH="./plays/"

CROPS = {"Breakout-v0":(31,-5,8,-8),"Pong-v0":(33,-16,10,-5)}

class EnvWrapper2(object):
    def __init__(self, env, mode = "grey", frame_count = 1, frame_skip = 1, size = 64, crop = None, record_freq = 20, torch=True, net=None):

        self.name = "Wrapper"+env.name
        self.env = env
        self.size = size
        self.grey = mode=="grey"
        self.frame_count = frame_count
        self.frame_skip = frame_skip
        self.crop = {s:None for s in ["up","down","left","right"]}

        self.spin = net
        if crop:
            for s,val in zip(["up","down","left","right"],CROPS[crop]):
                self.crop[s]=val            
        if self.grey:
            self.channels = self.frame_count
        else:
            self.channels = 3 * self.frame_count

        self.record_freq = record_freq
        self.current_episode = 0
        self.memory = deque([],self.frame_count)
        self.recording = False
        self.episode = []
        self.axis =(2,0,1)
        
        self.action_space = env.action_space
        self.observation_space = Continuous(self.spin.network.output_shape)


    def reset(self):

        self.current_episode += 1

        s = self.transform(self.env.reset())
        for _ in range(self.frame_count):
            self.memory.append(s)

        if not self.current_episode%self.record_freq:
            self.save_episode(self.name+str(self.current_episode))
            self.current_episode = self.current_episode%(20*self.record_freq)

        self.episode = []

        return self.current_state()

    def step(self,a):
        s, reward, done, info = self._big_step(a)
        self.memory.append(s)
        
        for _ in range(self.frame_count-1):
            if not done:                    
                s, r, done, info = self._big_step(a)
                reward = reward + r
                self.memory.append(s)
        return self.current_state(), np.clip(reward,-1,1), done, info


    def current_state(self):
        return np.concatenate(self.memory, axis=0)


    def _step(self,a,clip=True):
        s,a,r,info=self.env.step(a)
        r = np.clip(r,-1,1)
        return s,a,r,info


    def transform(self, state0):
        state = state0[self.crop['up']:self.crop['down'],self.crop['left']:self.crop['right'],:].astype(int)
        state = skimage.transform.resize(state.astype(float),(self.size, self.size, 3),mode="constant")/255.0
        
        self.last_frame = state.copy()
        self.episode.append(self.last_frame)
        
        state= state.astype(float).transpose(self.axis)
        return U.get(self.spin(U.torchify(state).unsqueeze(0))).squeeze()

    def save_episode(self,fname):
        episode = (np.concatenate([self.episode])*255.0).astype(np.int32)
        try : 
            skvideo.io.vwrite(PLAY_PATH+fname+ '.mp4', episode, inputdict={'-r': '25'},outputdict={'-vcodec': 'libx264',
                                                                                              '-pix_fmt': 'yuv420p','-r': '25'})
        except:
            print("No video pluging, saving array")
            np.save(PLAY_PATH+fname+"_uncompiled.mp4",episode)
    def render(self):
        #skimage.io.imshow(self.last_frame)
        plt.imshow(self.last_frame)

    def _big_step(self,a):
        s, reward, done, _ = self.env.step(a)
        for _ in range(self.frame_skip-1):
            if not done:                    
                s, r, done, _ = self.env.step(a)
                reward += r
            else:
                break
        return self.transform(s), reward, done, None
