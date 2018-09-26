#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 16:44:56 2018

@author: thinkpad
"""

import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def entropy_logits(logits):
    a0 = logits - logits.max(dim=-1, keepdim=True)[0]
    ea0 = torch.exp(a0)
    z0 = ea0.sum(dim=-1, keepdim=True)
    p0 = ea0 / z0
    return (p0 * (torch.log(z0) - a0)).sum(dim=-1)

def kl_logits(logits1,logits2):
    a0 = logits1 - logits1.max(dim=-1, keepdim=True)[0]
    a1 = logits2 - logits2.max(dim=-1, keepdim=True)[0]
    ea0 = torch.exp(a0)
    ea1 = torch.exp(a1)
    z0 = ea0.sum(dim=-1, keepdim=True)
    z1 = ea1.sum(dim=-1, keepdim=True)
    p0 = ea0 / z0
    return (p0 * (a0 - torch.log(z0) - a1 + torch.log(z1))).sum(dim=-1)
    
def mode(pi):
    return pi.max(dim=-1)[1]
    
def neglogp(pi, actions):
    return torch.nn.CrossEntropyLoss(reduce=False)(pi, actions.squeeze())

def logp(pi, actions):
    return -neglogp(pi, actions)
def linesearch(f, x, fullstep, max_backtracks=20):
    """
    Backtracking linesearch, where expected_improve_rate is the slope dy/dx at the initial point
    """
    fval = f(x)
    print("fval before", fval)
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac*fullstep
        newfval = f(xnew)
        actual_improve = fval - newfval
        if actual_improve > 0:
            print("fval after:", newfval)
            return True, xnew
    return False, x

def conjugate_gradient(Avp, b, cg_iters=10, residual_tol=1e-10):
    x = torch.zeros(b.size()).to(device)
    r = b - Avp(x)
    p = r
    rdotr = torch.dot(r, r)

    for i in range(cg_iters):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x

def argmax(vect):
    mx = max(vect)
    idx = np.where(vect==mx)[0]
    return np.random.choice(idx)
