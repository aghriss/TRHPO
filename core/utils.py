import sys
sys.path.append("../")
import torch
from torch import autograd
from core.summary import summarize
from tabulate import tabulate
import numpy as np
import numpy.linalg as LA
import collections
from core.optim import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOUBLE = True
EPS = 1e-9

if DOUBLE:
    if device.type=='cuda':
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    else:
        torch.set_default_tensor_type(torch.DoubleTensor)
    print(torch.get_default_dtype())



def cholesky_inv(A):
    B = (1-EPS)*LA.cholesky(get(A)) + EPS*np.eye(A.shape[0])
    C = LA.inv(B)
    return torchify(C)

def get(x):
    y = x.detach().cpu().numpy()
    if y.ndim == 0:
        return y[()]
    return y

def get_grad(Func,Y):
    Z = autograd.Variable(Y,requires_grad=True)
    return autograd.grad(Func(Z),Z,retain_graph=False)[0]

def grassmann_distance(X):
    _,S,_ = X.svd()
    return ((S-1)**2).mean()
"""def grassmann_distance1(X):

    return torch.norm(X.t().mm(X)-torch.eye(X.size(1)))**2/X.size(1)"""

def laplacian(W):
    return torch.diag(W.sum(dim=1))-W
def laplacian_rw(W):
    return torch.eye(W.size(0))- torch.diag(1/W.sum(dim=1)).mm(W)

def norms_matrix(X):
    norms = np.repeat(np.diag(X.dot(X.T)).reshape(-1,1),X.shape[0],axis=1)
    return norms+norms.T-2*X.dot(X.T)
    
def pinv(X,tol=1e-15):
    U,S,V = X.svd()
    S_inv = torch.zeros(S.shape)
    S_inv[S.abs()>tol] = 1/S[S.abs()>tol]
    return V.mm(S_inv.diag()).mm(U.t())

def print_loss(*args):
    print(tabulate([i for i in args], headers=['Loss', 'Old', 'New'], tablefmt='orgtbl'))

def queue(n):
    return collections.deque([0]*n, n)

def rayleigh_np(X, Lap):
    return np.trace(LA.inv(X.T.dot(X)+EPS*np.eye(X.shape[1])).dot(
            X.T.dot(Lap.dot(X))))

def rayleigh(X, Lap):
    return (X.t().mm(X)+EPS*torch.eye(X.shape[1])).inverse().mm(
            X.t().mm(Lap.mm(X))).trace()
def rayleigh_s(X, Lap):
    return X.t().mm(Lap.mm(X)).trace()
def rayleigh_unbiased(X, Lap,Sig):
    return (Sig).inverse().mm(X.t().mm(Lap.mm(X))).trace()
    
def sequential_rayleigh(X, Lap):
    res = 0.0
    k = X.shape[1]
    Sig = X.t().mm(X)
    Sig[range(k),range(k)].add_(EPS)
    for i in range(k):
        #print(i)
        Xk = X[:,:i+1]
        Sigki = (Sig[:i+1][:,:i+1]).inverse()
        res += (Sigki.mm(Xk.t().mm(Lap.mm(Xk)))).trace()
    return res

def sequential_rayleigh_s(X, Lap):
    res = 0.0
    k = X.shape[1]
    for i in range(k):
        Xk = X[:,:i+1]
        res += Xk.t().mm(Lap.mm(Xk)).trace()
    return res

def sequential_rayleigh_unbiased(X, Lap,Sig):
    res = 0.0
    k = X.shape[1]
    for i in range(k):
        Xk = X[:,:i+1]
        Sigki = (Sig[:i+1][:,:i+1]).inverse()
        res += (Sigki.mm(Xk.t().mm(Lap.mm(Xk)))).trace()
    return res/k**.5


    #return rayleigh(X,Lap)

def sequential_rayleigh_np(X, Lap):
    res = 0.0
    k = X.shape[1]
    for i in range(k):
        res += rayleigh_np(X[:,:i+1],Lap)
    return res

def summary(model, input_size): summarize(model, input_size, device=device.type,double=DOUBLE)

def torchify(x, double=DOUBLE):
    if double:
        return torch.tensor(x,dtype=torch.float64)
    return torch.tensor(x).float()


def conjugate_gradient(Avp, b, cg_iters=10, residual_tol=1e-6):
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


