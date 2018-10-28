#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 13:54:14 2018

@author: thinkpad
"""
import sys
sys.path.append("../")

import torch
import torch.autograd
import numpy as np

from base.basenetwork import BaseNetwork


class Policy(BaseNetwork):
    grad_enabled = True
    name = "Policy"
    softmax = torch.nn.LogSoftmax(-1)
    
#    def __init__(self,grad_enabled=Tru):
        
    def disable_grad(self):
        self.grad_enabled=False

    def forward(self,x):
        with torch.set_grad_enabled(self.grad_enabled):
            return super(Policy, self).forward(x)

    def sample(self,state):
        p = self.prob_predict(state)
        return np.random.choice(range(len(p)), p=p)
        
    def act(self,state):
        return argmax(self.predict(state), axis=-1)

    def logsoftmax(self,x):
        return self.softmax(self.forward(x))
          
    def entropy(self, states):
        logits = self.logsoftmax(states)
        return -(logits.exp()*logits).sum(dim=-1)
    
    def kl(self,pi,states):
        a1 = self.logsoftmax(states)
        a2 = self.pi.logsoftmax(states)
        z1 = a1.exp()
        return (z1*(a1-a2)).sum(dim=-1)
    
    def neglogp(self,states, actions):
        return torch.nn.CrossEntropyLoss(reduce=False)(self.forward(states), actions.squeeze())

    def logp(self,states, actions):
        return -self.neglogp(states, actions)

def argmax(vect):
    mx = max(vect)
    idx = np.where(vect==mx)[0]
    return np.random.choice(idx)
    