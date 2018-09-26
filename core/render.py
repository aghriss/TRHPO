# -*- coding: utf-8 -*-
"""
Created on Wed May 16 09:32:56 2018

@author: gamer
"""

import pygame as pg
import numpy as np
import skimage.transform as transform

class Render(object):
    def __init__(self, window_size=(360,480)):
        
        pg.init()
        
        self.h,self.w = window_size
        self.display = pg.display.set_mode((self.w,self.h))
        
        pg.display.set_caption("My Game")
    def update(self,vect):
        
        arr = transform.resize(vect,(self.h,self.w),mode='edge',clip=True
                ).transpose((1,0,2))
        
        arr = (255*arr/np.max(arr)).astype('uint8')
        img =  pg.surfarray.make_surface(arr[:,:,:])
        self.display.blit(img, (0,0))
        pg.display.flip()
        
    def quit(self):
        
        pg.quit()