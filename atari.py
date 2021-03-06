#from envs.grid import GRID
from base.wrappers import EnvWrapper,AtariWrapper
from Gate import GateTRPO
from networks.nets import GateTRPOPolicy,TRPOPolicy, VFunction

import gc
import gym

gc.enable()
gc.collect()

env = gym.make("Breakout-v0")

size = 72
input_shape=(3,size,size)
env.name = "Breakout-v0_"+str(size)
env = AtariWrapper(env, size=size, frame_count = 3, crop= "Breakout-v0")


hrl = GateTRPO(env, GateTRPOPolicy,TRPOPolicy, VFunction, n_options=4, option_len=3,
        timesteps_per_batch=1024,
        gamma=0.98, lam=0.98,
        max_kl=1e-3,
        cg_iters=10,
        cg_damping=1e-3,
        vf_iters=2,
        max_train=1000,
        ls_step=0.5,
        checkpoint_freq=10)
hrl.load()
hrl.train()
