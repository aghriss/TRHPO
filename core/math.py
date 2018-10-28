#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 16:44:56 2018

@author: thinkpad
"""

import numpy as np
import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def entropy_logits(logits):
    logsoft = F.log_softmax(logits,-1)
    return -(logsoft.exp()*logsoft).sum(dim=-1)

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
