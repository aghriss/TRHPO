import xxhash
import numpy as np
from base.grid import SimpleGRID
import scipy.sparse as SP

h = xxhash.xxh64()
 
s_to_i = lambda x,size : size*x[0]+x[1]
i_to_s = lambda x,size : (x%size,x//size)

def hash(x):
    h.reset()
    h.update(x)
    return h.digest()

class Indexer(object):
    
    def __init__(self):
        self.total = 0
        self.dict = {}
    
    def get(self,hs):
        val = self.dict.get(hs,-1)
        if val == -1:
            val = self.total
            self.dict[hs] = val
            self.total += 1
        return val
    
    def reset(self):
        self.__init__()
        
class HashIndexer(object):
    
    def __init__(self):
        self.total = 0
        self.dict = {}
    
    def get(self,state):
        hs=hash(state)
        val = self.dict.get(hs,-1)
        if val == -1:
            val = self.total
            self.dict[hs] = val
            self.total += 1
        return val
    def reset(self):
        self.__init__()

def get_graph(size):
    
    env = SimpleGRID(grid_size=size,max_time=5000)
    input_shape = env.observation_space.shape
    min_batch = size**2-size

    indexer = Indexer()
    W = np.zeros((min_batch,min_batch))
    states = np.zeros(min_batch).astype(int)
    data = np.zeros((min_batch,)+input_shape)

    while indexer.total<min_batch:
        done = False
        s = env.reset()
        #s = s.transpose(2,0,1)#np.expand_dims(s,axis=0)
        i = indexer.get(s_to_i(env.get_cat(),size))
        states[i] = s_to_i(env.get_cat(),size)
        data[states[i]] = s
        while not done:
            s,r,done = env.step(np.random.randint(4))
            #s = np.expand_dims(s,axis=0)
            #s = s.transpose(-1,0,1)
            j = indexer.get(s_to_i(env.get_cat(),size))
            states[j] = s_to_i(env.get_cat(),size)
            data[states[j]] = s
            W[states[i],states[j]] = W[states[j],states[i]] = 1
            if r==1:
                print(s_to_i(env.get_cat(),size),indexer.total)
            i = j
    return data, W

class GraphBuilder(object):
    def __init__(self, env, action_set, batch_size):
        self.env = env
        self.action_set = action_set
        self.h = xxhash.xxh64()
        self.max_size = batch_size 
        self.indices = set()
        self._total = 0
        self.dict = {}
        self.states = []
        self.prev = 0
        self.roll = self.roller()
    def submit(self,state, new=False):

        hs = self.hash(state)        
        val = self.dict.get(hs,-1)
        
        if val == -1:
            self.states.append(state)
            val = self._total
            self.dict[hs] = self._total
            self._total += 1

        if not new:
            self.indices.add((self.prev,val))
        self.prev = val

    def reset(self):
        self.indices = set()
        self._total = 0
        self.dict = {}
        self.states = []
        self.prev = 0
    def roller(self):
        done = True
        while True:
            self.reset()
            while not self.full:
                if done:
                    s = self.env.reset()
                    self.submit(s.copy(), new=done)
                    done = False
                while not done and not self.full:
                    s,_,done,_ = self.env.step(np.random.choice(self.action_set))
                    self.submit(s.copy())
            S,W =  self.get_graph()
            W = W.toarray()
            #W = (W+W.T)/2
            W = np.maximum(W,W.T)
            #np.fill_diagonal(W, 1)
            yield S, W
    def get(self):
        return self.roll.__next__()
    def hash(self,x):
        self.h.reset()
        self.h.update(x)
        return self.h.digest()

    def get_graph(self):
        if not self.full:
            raise "Graph not full Yet"
        indices = np.array(list(self.indices))
        rows = indices[:,0]
        cols = indices[:,1]
        data = np.ones(len(rows))
        return np.array(self.states),SP.coo_matrix((data, (rows, cols)),shape=(self.max_size, self.max_size))
    
    @property
    def size(self):
        return self._total
    @property
    def full(self):
        return self.size == self.max_size
    
    
    
