#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 13:00:17 2018

@author: thinkpad
"""
import sys
sys.path.append("../")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import base.basenetwork as B
from base.basefunction import Policy

class QFunction(B.BaseNetwork):
    name="QFunction"
    
    def __init__(self,env,**kwargs):
        super(QFunction, self).__init__(env.observation_space.shape,env.action_space.n,**kwargs)
        self.conv = [nn.Sequential(nn.Conv2d(self.input_shape[0], 8, kernel_size=6, stride=3, padding=2), nn.Tanh(),
                                    B.conv3_2(8, 16),
                                    B.conv3_2(16, 32))]
        x = B.output_shape(self.conv[0],self.input_shape)
        self.model = nn.Sequential(self.conv[0],
                                   B.Flatten(),nn.Tanh(),
                                   nn.Linear(np.prod(x), 512),
                                   nn.Linear(512,self.output_shape))

        self.compile()
        self.loss = torch.nn.SmoothL1Loss()
        self.optimizer = optim.RMSprop(self.parameters(), lr=0.00025,alpha=0.95,eps=0.01,momentum=0.95)
        

class TRPOPolicy(Policy):
    name="TRPOPolicy"
    def __init__(self,env,**kwargs):
        super(TRPOPolicy,self).__init__(env.observation_space.shape,env.action_space.n,**kwargs)
        self.conv = [nn.Sequential(nn.Conv2d(self.input_shape[0], 6, kernel_size=6, stride=3, padding=2), nn.ReLU(),
                                    B.conv3_2(6, 12),nn.ReLU())]
        x = B.output_shape(self.conv[0],self.input_shape)
        self.model = nn.Sequential(self.conv[0],
                                   B.Flatten(),
                                   nn.Linear(np.prod(x), 256),nn.Tanh(),
                                   nn.Linear(256,self.output_shape))

        self.compile()
        
        self.loss = torch.nn.MSELoss()
        self.optimizer = optim.RMSprop(self.parameters(),lr=1e-5,alpha=0.99,weight_decay=1e-4)
        
class GateTRPOPolicy(Policy):
    name="GateTRPOPolicy"
    def __init__(self,env,options_n,**kwargs):
        super(GateTRPOPolicy,self).__init__(env.observation_space.shape,options_n,**kwargs)
        self.conv = [nn.Sequential(nn.Conv2d(self.input_shape[0], 6, kernel_size=6, stride=3, padding=2), nn.ReLU(),
                                    B.conv3_2(6, 12),nn.ReLU())]
        x = B.output_shape(self.conv[0],self.input_shape)
        self.model = nn.Sequential(self.conv[0],
                                   B.Flatten(),
                                   nn.Linear(np.prod(x), 256),nn.Tanh(),
                                   nn.Linear(256,self.output_shape))

        self.compile()
        
        self.loss = torch.nn.MSELoss()
        self.optimizer = optim.RMSprop(self.parameters(),lr=1e-5,alpha=0.99,weight_decay=1e-4)

class VFunction(B.BaseNetwork):
    name="VFunction"
    def __init__(self,env,**kwargs):
        super(VFunction, self).__init__(env.observation_space.shape,1,**kwargs)
        
        self.conv = [nn.Sequential(nn.Conv2d(self.input_shape[0], 6, kernel_size=6, stride=3, padding=2), nn.ReLU(),
                                    B.conv3_2(6, 12),nn.ReLU())]
        x = B.output_shape(self.conv[0],self.input_shape)
        self.model = nn.Sequential(self.conv[0],
                                   B.Flatten(),
                                   nn.Linear(np.prod(x), 256),nn.Tanh(),
                                   nn.Linear(256,self.output_shape))

        self.compile()
        self.loss = torch.nn.SmoothL1Loss()
        self.optimizer = optim.RMSprop(self.parameters(), lr=1e-3,alpha=0.95,eps=0.01,momentum=0.95)


class QFunction_S(B.BaseNetwork):
    name="QFunction_S"
    
    def __init__(self, env):
        super(QFunction_S, self).__init__(env.observation_space.shape,env.action_space.n)
        self.conv = [nn.Sequential(nn.Conv2d(self.input_shape[0], 8, kernel_size=8, stride=4), nn.ReLU(),
                                    nn.Conv2d(8, 16, kernel_size=4, stride=2), nn.Tanh(),
                                    nn.Conv2d(16, 32, kernel_size=3, stride=2), nn.Tanh())]
        x = B.output_shape(self.conv,self.input_shape)
        self.model = nn.Sequential(self.conv[0],
                                   B.Flatten(),
                                   nn.Linear(np.prod(x), 512),nn.Tanh(),
                                   nn.Linear(512,self.output_shape))

        self.compile()
        self.loss = torch.nn.SmoothL1Loss()
        self.optimizer = optim.RMSprop(self.parameters(), lr=0.00025,alpha=0.95,eps=0.01,momentum=0.95)

class TRPOPolicyEmbed(B.BaseNetwork):
    name="TRPOPolicyEmbed"
    def __init__(self,env):
        super(TRPOPolicyEmbed, self).__init__(env.observation_space.shape,env.action_space.n)
        self.model = nn.Sequential(nn.BatchNorm1d(self.input_shape[0]),
                                   nn.Linear(self.input_shape[0],512),nn.Tanh(),
                                   nn.Linear(512, 256),
                                   nn.Linear(256, 512),nn.Tanh(),
                                   nn.Linear(512, 128),
                                   nn.Linear(128,self.output_shape))
        self.compile()
        self.loss = torch.nn.MSELoss()
        self.optimizer = optim.RMSprop(self.parameters(),lr=1e-5,alpha=0.99,weight_decay=1e-4)
    def act(self,state):
        return np.argmax(self.predict(state), axis=-1)
    def sample(self,states):
        logits = self.predict(states)
        soft = np.exp(logits-np.max(logits))
        p = (soft/np.sum(soft))
        #print(np.max(np.abs(logits)))
        #u = np.random.uniform(size=logits.shape)
        #return np.argmax(logits - np.log(-np.log(u)), axis=-1)
        #print(p)
        return np.random.choice(range(len(p)), p=p)
        
class VFunctionEmbed(B.BaseNetwork):
    name="VFunctionEmbed"
    def __init__(self,env):
        super(VFunctionEmbed, self).__init__(env.observation_space.shape,1)
        
        self.model = nn.Sequential(nn.BatchNorm1d(self.input_shape[0]),
                                   nn.Linear(self.input_shape[0],512),nn.Tanh(),
                                   nn.Linear(512, 256),
                                   nn.Linear(256, 512),nn.Tanh(),
                                   nn.Linear(512, 128),
                                   nn.Linear(128,self.output_shape))

        self.compile()
        self.loss = torch.nn.MSELoss()
        self.optimizer = optim.RMSprop(self.parameters(),lr=1e-5,alpha=0.99,weight_decay=1e-4)


"""
class TRPONetS(DQNetS):
    name="TRPONet_S"

class TRPONetSValue(TRPONetS):
    name="TRPONet_S_Value"
    def __init__(self, input_shape, output_shape):
        super(TRPONetSValue, self).__init__(input_shape, 1)
"""