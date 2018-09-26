#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 15:26:44 2018

@author: thinkpad
"""
import numpy as np

class Discrete(object):
    def __init__(self,n):        
        self.n = n
        self.shape = (n,)
        self.dtype = np.int64
        self.type="Discrete"
    def sample(self):
        return np.random.randint(self.n)
    
    def __repr__(self):
        return "Discrete(%d)" % self.n

class Continuous(object):
    def __init__(self,shape,dtype=np.float32):        
        self.shape = tuple(shape)
        self.dtype=dtype
    def __repr__(self):
        return "Continuous" +str(self.shape)
    
    def __eq__(self,s):
        return self.shape == s
