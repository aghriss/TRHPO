from torch import nn
import numpy as np
import base.basenetwork as BaseN
from networks.cholesky import CholeskyBlock

class FCNet(BaseN.BaseNetwork):
    name ="FCNet"
    def __init__(self,input_shape,output_shape,owner_name=""):
        super(FCNet,self).__init__(input_shape,output_shape,owner_name)
        
        x = input_shape
        self.model = nn.Sequential(BaseN.Flatten(),
                                   nn.Linear(np.prod(x), 1024),nn.Softplus(),
                                   nn.Linear(1024,512),nn.Tanh(),
                                   nn.Linear(512,256),
                                   BaseN.EigenLayer(256,self.output_shape[0]))
        self.compile()

class FCSpectralNet(BaseN.BaseNetwork):
    name ="FCSpectralNet"
    def __init__(self,input_shape,output_shape,owner_name=""):
        super(FCSpectralNet,self).__init__(input_shape,output_shape,owner_name)
        
        x = input_shape
        self.model = nn.Sequential(BaseN.Flatten(),
                                   nn.Linear(np.prod(x), 1024),BaseN.AdaptiveTanh(),
                                   nn.Linear(1024,1024),BaseN.AdaptiveTanh(),
                                   nn.Linear(1024,512),BaseN.AdaptiveTanh(),
                                   BaseN.EigenLayer(512,self.output_shape[0]))
        self.compile()
class FCSpectralMNet(BaseN.BaseNetwork):
    name ="FCSpectralMNet"
    def __init__(self,input_shape,output_shape,owner_name=""):
        super(FCSpectralMNet,self).__init__(input_shape,output_shape,owner_name)
        
        x = input_shape
        self.model = nn.Sequential(BaseN.Flatten(),
                                   nn.Linear(np.prod(x), 1024),nn.ReLU(),
                                   nn.Linear(1024,1024),nn.ReLU(),
                                   nn.Linear(1024,512),nn.ReLU(),
                                   nn.Linear(512,self.output_shape[0]-1),nn.Tanh(),
                                   BaseN.EigenLayer())
        self.compile()
class FCNetQ(BaseN.BaseNetwork):
    name ="FCNetQ"
    def __init__(self,input_shape,output_shape,owner_name=""):
        super(FCNetQ,self).__init__(input_shape,output_shape,owner_name)
        
        x = int(np.prod(input_shape))
        self.model = nn.Sequential(BaseN.Flatten(),
                                   nn.Linear(x,x),nn.Tanh(),
                                   nn.Linear(x,self.output_shape[0]))
        self.compile()


class ConvNet(BaseN.BaseNetwork):
    name="ConvNet"
    def __init__(self,input_shape,output_shape,owner_name=""):
        super(ConvNet,self).__init__(input_shape,output_shape,owner_name)
        
        self.conv = [nn.Sequential(BaseN.conv3_2(input_shape[0], 8),nn.ReLU(),
                                                 BaseN.conv3_2(8, 16),nn.ReLU(),
                                                 BaseN.conv3_2(8, 8))]
        x = BaseN.output_shape(self.conv[0],input_shape)
        self.model = nn.Sequential(self.conv[0],
                                   BaseN.Flatten(),
                                   nn.Linear(np.prod(x), 512),BaseN.AdaptiveTanh(),
                                   nn.Linear(512,256),
                                   BaseN.EigenLayer(256,self.output_shape[0],bias=False))
        self.compile()
class ConvNetBias(BaseN.BaseNetwork):
    name="ConvNetBias"
    def __init__(self,input_shape,output_shape,owner_name=""):
        super(ConvNetBias,self).__init__(input_shape,output_shape,owner_name)
        
        self.conv = [nn.Sequential(BaseN.conv3_2(input_shape[0], 8),nn.Softplus(),
                                   BaseN.conv3_2(8, 12),BaseN.AdaptiveTanh(),
                                   BaseN.conv3_2(12, 16),
                                   BaseN.conv3_2(16, 20))]
        x = BaseN.output_shape(self.conv[0],input_shape)
        self.model = nn.Sequential(self.conv[0],
                                   BaseN.Flatten(),
                                   nn.Linear(np.prod(x), 512),BaseN.AdaptiveTanh(),
                                   nn.Linear(512,256),
                                   BaseN.EigenLayer(256,self.output_shape[0],bias=False))
        self.compile()

class FCConvNet(BaseN.BaseNetwork):
    name="FCConvNet"
    def __init__(self,input_shape,output_shape,owner_name=""):
        super(FCConvNet,self).__init__(input_shape,output_shape,owner_name)
        
        self.conv = [nn.Sequential(BaseN.conv3_2(input_shape[0], 4),nn.Softplus(),
                                   BaseN.conv3_2(4, 8),BaseN.AdaptiveTanh())]
        x = BaseN.output_shape(self.conv[0],input_shape)
        self.model = nn.Sequential(self.conv[0],
                                   BaseN.Flatten(),
                                   nn.Linear(np.prod(x), 512),
                                   nn.Linear(512,512),BaseN.AdaptiveTanh(),
                                   nn.Linear(512,256),
                                   BaseN.EigenLayer(256,self.output_shape[0],bias=False))
        self.compile()
        
class FCConvNetBias(BaseN.BaseNetwork):
    name="FCConvNetBias"
    def __init__(self,input_shape,output_shape,owner_name=""):
        super(FCConvNetBias,self).__init__(input_shape,output_shape,owner_name)
        
        self.conv = [nn.Sequential(BaseN.conv3_2(input_shape[0], 4),nn.ReLU(),
                                   BaseN.conv3_2(4, 4),BaseN.AdaptiveTanh())]
        x = BaseN.output_shape(self.conv[0],input_shape)
        self.model = nn.Sequential(self.conv[0],
                                   BaseN.Flatten(),
                                   nn.Linear(np.prod(x), 512),
                                   nn.Linear(512,1024),BaseN.AdaptiveTanh(),
                                   nn.Linear(1024,256),
                                   BaseN.EigenLayer(256,self.output_shape[0],bias=False))
        self.compile()
        
class ConvNet2(BaseN.BaseNetwork):
    name="ConvNet2"
    def __init__(self,input_shape,output_shape,owner_name=""):
        super(ConvNet2,self).__init__(input_shape,output_shape,owner_name)
        
        self.conv = [nn.Sequential(BaseN.conv3_2(input_shape[0], 3),nn.Softplus(),
                                                 BaseN.conv3_2(3, 6),BaseN.conv3_2(6, 12))]
        x = BaseN.output_shape(self.conv[0],input_shape)
        self.model = nn.Sequential(self.conv[0],
                                   BaseN.Flatten(),
                                   nn.Linear(np.prod(x), 512),
                                   nn.Linear(512,256),nn.Tanh(),
                                   nn.Linear(256,512),
                                   nn.Linear(512,1024),nn.Tanh(),
                                   nn.Linear(1024,512),
                                   nn.Linear(512,256),nn.Tanh(),
                                   nn.Linear(256,256),
                                   BaseN.EigenLayer(256,self.output_shape[0]))
        self.compile()
class ConvNetBig(BaseN.BaseNetwork):
    name="ConvNetBig"
    def __init__(self,input_shape,output_shape,owner_name=""):
        super(ConvNetBig,self).__init__(input_shape,output_shape,owner_name)
        
        self.conv = [nn.Sequential(BaseN.conv3_2(input_shape[0], 8),nn.Softplus(),
                                                 BaseN.conv3_2(8, 16),nn.Softplus(),
                                                 BaseN.conv3_2(16, 32))]
        x = BaseN.output_shape(self.conv[0],input_shape)
        self.model = nn.Sequential(self.conv[0],
                                   BaseN.Flatten(),
                                   nn.Linear(np.prod(x), 512),
                                   nn.Linear(512,256),nn.Tanh(),
                                   nn.Linear(256,512),
                                   BaseN.EigenLayer(512,self.output_shape[0]))
        self.compile()

class ConvNetBigBias(BaseN.BaseNetwork):
    name="ConvNetBigBias"
    def __init__(self,input_shape,output_shape,owner_name=""):
        super(ConvNetBigBias,self).__init__(input_shape,output_shape,owner_name)
        
        self.conv = [nn.Sequential(BaseN.conv3_2(input_shape[0], 4),nn.Softplus(),
                                                 BaseN.conv3_2(4, 4),BaseN.AdaptiveTanh())]
        x = BaseN.output_shape(self.conv[0],input_shape)
        self.model = nn.Sequential(self.conv[0],
                                   BaseN.Flatten(),
                                   nn.Linear(np.prod(x), 512),
                                   nn.Linear(512,256),nn.Tanh(),
                                   nn.Linear(256,512),
                                   BaseN.EigenLayer(512,self.output_shape[0],bias=False))
        self.compile()
class ConvNetBigAtari(BaseN.BaseNetwork):
    name="ConvNetBigAtari"
    def __init__(self,input_shape,output_shape,owner_name=""):
        super(ConvNetBigAtari,self).__init__(input_shape,output_shape,owner_name)
        
        self.conv = [nn.Sequential(BaseN.conv3_2(input_shape[0], 8),nn.Softplus(),
                                                 BaseN.conv3_2(8, 16),
                                                 BaseN.conv3_2(16, 32))]
        x = BaseN.output_shape(self.conv[0],input_shape)
        self.model = nn.Sequential(self.conv[0],
                                   BaseN.Flatten(),
                                   nn.Linear(np.prod(x), 512),
                                   nn.Linear(512,512),nn.Tanh(),
                                   nn.Linear(512,1024),
                                   BaseN.EigenLayer(1024,self.output_shape[0]))
        self.compile()

class ConvNetBigS(BaseN.BaseNetwork):
    name="ConvNetBigS"
    def __init__(self,input_shape,output_shape,owner_name=""):
        super(ConvNetBigS,self).__init__(input_shape,output_shape,owner_name)
        
        self.conv = [nn.Sequential(BaseN.conv3_2(input_shape[0], 8),nn.Softplus(),
                                                 BaseN.conv3_2(8, 16),
                                                 BaseN.conv3_2(16, 32))]
        x = BaseN.output_shape(self.conv[0],input_shape)
        self.model = nn.Sequential(self.conv[0],
                                   BaseN.Flatten(),
                                   nn.Linear(np.prod(x), 512),
                                   nn.Linear(512,256),nn.Tanh(),
                                   nn.Linear(256,512),
                                   nn.Linear(512,self.output_shape[0]))
        self.compile()
        
class ConvNetMNIST(BaseN.BaseNetwork):
    name = "ConvNetMNIST"
    def __init__(self,input_shape,output_shape,**kwargs):
        super(ConvNetMNIST,self).__init__(**kwargs)
        self.n = output_shape
        self.conv = [BaseN.ResNetBlock(1,32),
                         BaseN.conv3_2(32,64)]
        x = BaseN.output_shape(self.conv[0],input_shape)
        self.model = nn.Sequential(self.conv[0], nn.Softplus(),
                                    BaseN.Flatten(),
                                    nn.Linear(np.prod(x), 512),
                                    nn.Linear(512,256),nn.Tanh(),
                                    BaseN.EigenLayer(256,self.output_shape[0]))
        self.compile()
        
class ConvNetSimple(BaseN.BaseNetwork):
    name="ConvNetSimple"
    def __init__(self,input_shape,output_shape,owner_name=""):
        super(ConvNetSimple,self).__init__(input_shape,output_shape,owner_name)
        
        self.conv = [nn.Sequential(BaseN.conv3_2(input_shape[0], 4),nn.Softplus())]
        x = BaseN.output_shape(self.conv[0],input_shape)
        self.model = nn.Sequential(self.conv[0],
                                   BaseN.Flatten(),
                                   nn.Linear(np.prod(x), 512),
                                   nn.Linear(512,256),nn.Tanh(),
                                   nn.Linear(256,self.output_shape[0]))
        self.compile()

class FCNetSimple(BaseN.BaseNetwork):
    name ="FCNetSimple"
    def __init__(self,input_shape,output_shape,owner_name=""):
        super(FCNetSimple,self).__init__(input_shape,output_shape,owner_name)
        
        x = input_shape
        self.model = nn.Sequential(BaseN.Flatten(),
                                   nn.Linear(np.prod(x), 1024),nn.Softplus(),
                                   nn.Linear(1024,512),
                                   nn.Linear(512,256),nn.Tanh(),
                                   nn.Linear(256,self.output_shape[0]))
        self.compile()
