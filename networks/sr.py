from torch import nn
import torch
import networks.networks as NN
import base.basenetwork as BaseN
import numpy as np
import core.utils as U

class SRNetS(BaseN.BaseNetwork):
    name="SRSimple"
    def __init__(self,input_shape,owner_name=""):
        super(SRNetS,self).__init__(input_shape,1,owner_name)
        
        self.conv = [nn.Sequential(BaseN.conv3_2(input_shape[0], 8),nn.Softplus(),
                                    BaseN.conv3_2(8, 16))]
        x = BaseN.output_shape(input_shape, self.conv[0])
        self.model1 = nn.Sequential(self.conv[0],
                                   BaseN.Flatten(),
                                   nn.Linear(np.prod(x), 256),
                                   nn.Linear(256,64),nn.Tanh())
        
        self.model2 = nn.Sequential(BaseN.conv3_2(input_shape[0], 8),nn.Softplus(),
                                    BaseN.conv3_2(8, 16),
                                    BaseN.Flatten(),
                                    nn.Linear(np.prod(x), 256),
                                    nn.Linear(256,64),nn.Tanh())
        self.model3 = nn.Sequential(nn.Linear(128,64),
                                    nn.Linear(64,1))
        
        self.compile()
        
        self.loss = nn.MSELoss()
        self.optimizer = self.optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3)        

    def forward(self,states):
        
        states1 = states[:,0]
        states2 = states[:,1]
        s1 = self.model1(states1)
        s2 = self.model2(states2)
        return self.model3(torch.cat([s1,s2],-1))
         
    def summary(self):
        U.summary(self, (2,)+self.input_shape)

class Abs(nn.Module):
    def forward(self,x):
        return x.abs()
class SRNet(BaseN.BaseNetwork):
    name="SRSimple"
    def __init__(self,input_shape,owner_name=""):
        super(SRNet,self).__init__(input_shape,1,owner_name)
        
        self.conv = [nn.Sequential(BaseN.conv3_2(input_shape[0], 8),nn.Tanh(),
                                    BaseN.conv3_2(8, 16),nn.Tanh(),
                                    BaseN.conv3_2(16, 32))]
        x = BaseN.output_shape(input_shape, self.conv[0])
        self.model = nn.Sequential(self.conv[0],
                                   BaseN.Flatten(),
                                   nn.Linear(np.prod(x), 256),nn.Tanh(),
                                   nn.Linear(256,128),
                                   nn.Linear(128,1),Abs())
        
        self.compile()
        
        self.loss = nn.MSELoss()
        self.optimizer = self.optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-4)        
