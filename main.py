from envs.grid import GRID
from base.wrappers import EnvWrapper
from Gate import  GateTRPO
from networks.nets import TRPOPolicy, VFunction

import gc

gc.enable()
gc.collect()

game = "gridHRL"
size = 72
input_shape=(3,size,size)

env = EnvWrapper(GRID(grid_size=36,max_time=2500,stochastic = True, square_size=2),
                 record_freq=10, size=size, mode="rgb", frame_count = 1)


hrl = GateTRPO(env, TRPOPolicy, VFunction, n_options=4, option_len=3,
        timesteps_per_batch=1000,
        gamma=0.99, lam=0.97,
        max_kl=1e-2,
        cg_iters=10,
        cg_damping=1e-2,
        vf_iters=2,
        max_train=1000,
        ls_step=0.5,
        checkpoint_freq=10)

hrl.train()
