import torch
import numpy as np

"""def bidirect_linesearch(func, x0, grad, expected,
                        alpha=0.5, beta_down=.5, beta_up=1.5, max_iter=50):
    f0 = func(x0)
    f1 = func(x0+grad)
    if f0-f1 > alpha*expected:
        beta = beta_up
        for m in range(1,max_iter):
            f1 = func(x0+grad*beta**m)
            if f0-f1 < alpha*expected*beta**m:
                return True, beta**(m-1)
    else:
        beta = beta_down
        # 1st step is good, perhaps we can jump higher
        for m in range(1,max_iter):
            f1 = func(x0+grad*beta**m)
            if f0-f1 > alpha*expected*beta**m:
                return True, beta**m
        return False, 0
"""

def circlesearch(func, x0, grad1, grad2, expected, 
                 alpha = 0.5, beta=.5, max_iter=25):
    f0 = func(x0)
    t0 = torch.tensor(np.pi/2)
    for m in range(max_iter):
        f1 = func(x0+t0.cos()*grad1+t0.sin()*grad2)
        if f0-f1 > alpha*expected:
            return t0
        t0.mul_(beta)
    return torch.tensor(0.0)

def linesearch(func, x0, grad, expected, alpha = 0.5, beta=.5, max_iter=25):
    f0 = func(x0)
    coef = torch.tensor(1.0)
    for _ in range(max_iter):
        f1 = func(x0+grad*coef)
        if f0-f1 > alpha*expected*coef:# and f0-f1 < expected*coef:
            return coef
        coef.mul_(beta)
    return torch.tensor(0.0)

def double_circlesearch(func, x0, grad1, grad2, expected, 
                 alpha = 0.5, beta=.5, max_iter=25):
    f01,f02 = func(x0)
    t0 = torch.tensor(np.pi/2)
    for m in range(max_iter):
        f11,f12 = func(x0+t0.cos()*grad1+t0.sin()*grad2)
        if f01-f11 > alpha*expected and f02>f12:
            return t0
        t0.mul_(beta)
    return torch.tensor(0.0)

def double_linesearch(func, x0, grad, grad2, expected, 
                 alpha = 0.5, beta=.5, max_iter=25):
    f01,f02 = func(x0)
    t0 = torch.tensor(1.0)
    for m in range(max_iter):
        f11,f12 = func(x0+t0*grad+grad2)
        #print(f11,f12)
        if (f01-f11).abs() > alpha*expected and f02>f12:
            return t0
        t0.mul_(beta)
    return torch.tensor(0.0)


"""def euclidian_norm(mat):
    r = torch.mm(mat, mat.t())
    diag = r.diag().unsqueeze(0)
    diag = diag.expand_as(r)
    D = diag + diag.t() - 2*r
    su = (D<0).sum()
    if su:
        print(su,"Negative norms")
    return D.abs()

def grassmann_constraint(g_new,g_old):
    return (g_new-g_old)

def grassmann_metric(Y,A,B):
    return (Y.t().mm(Y).inverse().mm(A.t().mm(B))).trace()/B.shape[1]

def polynome_extremas(polynome):
    n = polynome.shape[0]
    Q = polynome[1:]*torch.arange(1,n)

    x,_ = polynome_solver(Q)
    
    eigs = torch.cat((x.clamp(min=0),torch.zeros(1,device=device))).unique(sorted=True)
    
    vals = torch.zeros_like(eigs)
    for i,e in enumerate(eigs):
        vals[i] = (polynome*e**torch.arange(n)).sum()
    return eigs, vals

def polynome_solver(P):
    Q = P.clone()
    Q = Q/(max(Q[Q!=0].abs().min(),EPS)**.5)
    n = Q.shape[0]
    C = torch.diagflat(torch.ones(n-1), offset=-1)
    C[:,-1] = -Q
    
    eigs = torch.eig(C)[0]
    eigs = eigs[:,0][eigs[:,1]==0].unique(sorted=True)
    
    vals = torch.zeros_like(eigs)
    for i,e in enumerate(eigs):
        vals[i] = (P*e**torch.arange(n)).sum()
    return eigs, vals"""
    