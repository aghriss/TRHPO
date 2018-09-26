import collections
import numpy as np
import json
import os
from tabulate import tabulate
import core.utils as U
import torch
import datetime
import pandas as pd
LOG_PATH="./logs/"

def empty_dict(keys):
    dic = collections.OrderedDict()
#    for k in keys:
#        dic[k]=0
    return dic

class Logger(object):
    
    def __init__(self,log_fname,history = 10, path=LOG_PATH):
        if not os.path.exists(path):
            os.makedirs(path)
        i = ""
        self.fname = path+log_fname+i+".json"
        if os.path.isfile(self.fname):
            i = str(sum([ log_fname in fname for fname in os.listdir(path)]))
        self.fname = path+log_fname+i+".json"
        self.file = open(os.path.join(self.fname), 'a')
        self.past = collections.deque([],history)
        self.start = False
        self.t = 0
        self.keys = set()
    def step(self):
        if self.start:
            self.save()
        self.past.append(empty_dict(self.keys))
        self.t = self.t + 1
        self.past[-1]["step"] = self.t
        self.past[-1]["time"] = str(datetime.datetime.utcnow())
        self.start = True
    def log(self, key, val):
        self.keys.add(key)
        if type(val) ==torch.Tensor:
            self.log(key,U.get(val))
        else:
            if "numpy" in str(type(val)):
                self.past[-1][key] = val.tolist()
            else:
                self.past[-1][key] = val
    def display(self):
        res = []
        for key,val in self.past[-1].items():
            if type(val)!=str:
                res.append([key, np.mean([p.get(key,0) for p in self.past])])
            else:
                res.append([key, val])
        print(tabulate(res, headers=['Var', 'Value'], tablefmt='orgtbl'))
    def save(self):
        self.file.write(json.dumps(self.past[-1], indent=4))
        self.file.flush()
        os.fsync(self.file.fileno())
    def load(self):
        file = open(os.path.join(self.fname), 'r')
        data = []
        block = file.__next__()
        for line in file:
            if line[0]=="}":
                block= block +"}"
                data.append(json.loads(block))
                block = "{"
            else:
                block = block + line
        data = pd.DataFrame(data)
        data["time"] = data["time"].apply(lambda x : datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S.%f").timestamp())
        file.close()
        return data
        

"""
0:No
1:jump
2:Nothing
3:right
4:left
5:down
6:right
7:left
8:right
9:left
10:jump
11:right jump
13:down
12:jump left
14:left
"""