import core.utils as U
import torch
import numpy as np
from base.basenetwork import BaseNetwork
import torch.nn as nn
import core.console as C

class Cholesky(nn.Module):
    """
        Parameters:
            alpha :  learning rate
        Inputs:
            Batch : Y[nxk] to orthogonalize
            
        Operations:
            Updates Sigma <- Sigma + alpha*Y.T.dot(Y) 
            Runs the cholesky decomposition L of Sigma        
            then outputs : Y.dot(inv(L.T))
    """
    def __init__(self, k, alpha=1e-2):
        super(Cholesky, self).__init__()
        self.layer = nn.Linear(k,k,bias=False)
        self.k = k
        self.alpha = alpha
        self.layer.weight.requires_grad = False
        self.Sig = U.torchify(np.eye(k))
        self.min_eig = U.queue(50)
        self.update_weight()
    def forward(self, y):
        return self.layer(y)

    def update(self, y):
        self.Sig = (((1-self.alpha)*self.Sig*y.shape[0] + self.alpha*(y.t().mm(y)))/y.shape[0]).detach()
        self.min_eig.append(float(U.get(self.Sig.eig()[0][:,0].min())))
        if self.min_eig[-1]<0:
            print("None SPD matrix in Cholesky")
        self.update_weight()

    def update_weight(self):
        self.layer.weight.data= U.cholesky_inv(self.Sig)



class CholeskyBlock(BaseNetwork):
    name="CholeskyBlock"
    def __init__(self, output_shape, alpha=1e-2,n_blocks=2,**kwargs):
        super(CholeskyBlock, self).__init__(input_shape=output_shape,output_shape=output_shape,**kwargs)
        self.model = nn.Sequential(*[Cholesky(output_shape[0],alpha) for _ in range(n_blocks)])
        self.summary()

    def update(self,y):
        for l in self.model:
            l.update(y)
            y = l(y)

    def load(self,fname):
        try:
            print("Loading %s.%s"%(fname,self.name))
            dic = torch.load("%s.%s"%(fname,self.name))
            for i,l in enumerate(self.model):
                l.Sig = dic["Sig%i"%i]
                l.update_weight()
        except:
            C.warning("Couldn't load %s"%fname)

    def save(self,fname):
        print("Saving to %s.%s"%(fname,self.name))
        dic = {"Sig%i"%i:l.Sig for i,l in enumerate(self.model)}
        torch.save(dic, "%s.%s"%(fname,self.name))
