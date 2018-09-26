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

    name = "Policy"
    softmax = torch.nn.LogSoftmax(-1)
    def sample(self,states):
        logits = self.predict(states)
        #soft = np.exp(logits)
        #p = (soft/np.sum(soft))[0]
        
        u = np.random.uniform(size=logits.shape)
        return np.argmax(logits - np.log(-np.log(u)), axis=-1)
        #print(p)
        #return np.random.choice(range(len(p)), p=p)
        
    def act(self,state):
        return np.argmax(self.predict(state), axis=-1)
    def logsoftmax(self,x):
        return self.softmax(self.forward(x))
        
    def kl_logits(self, pi, states):
        logits1 = self.forward(states)
        logits2 = pi(states)
        a0 = logits1 - logits1.max(axis=-1, keepdim=True)
        a1 = logits2 - logits2.max(axis=-1, keepdim=True)
        ea0 = torch.exp(a0)
        ea1 = torch.exp(a1)
        z0 = ea0.sum(axis=-1, keepdim=True)
        z1 = ea1.sum(axis=-1, keepdims=True)
        p0 = ea0 / z0
        return (p0 * (a0 - torch.log(z0) - a1 + torch.log(z1))).sum(axis=-1)

