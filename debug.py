
"""

policy_func=TRPOPolicy;
value_func= VFunction; n_options=4; option_len=3;
timesteps_per_batch=1000;
gamma=0.99; lam=0.97;
max_kl=1e-2;
cg_iters=10;
cg_damping=1e-2;
vf_iters=2;
max_train=1000;
ls_step=0.5;
checkpoint_freq=10
self=hrl
path = self.path_generator.__next__()
self.oldpolicy.copy(self.policy)
for p in self.options:
    p.oldpolicy.copy(p.policy)
import numpy as np
import torch
import collections
from base.baseagent import BaseAgent
from core.console import Progbar
import core.math as m_utils
import core.utils as U
from Option import OptionTRPO
import core.console as C


states = U.torchify(path["states"])
options = U.torchify(path["options"]).long()
actions = U.torchify(path["actions"]).long()
advantages = U.torchify(path["advantage"])
tdlamret = U.torchify(path["tdlamret"])
vpred = U.torchify(path["vf"]) # predicted value function before udpate
advantages = (advantages - advantages.mean()) / advantages.std() # standardized advantage function estimate        
losses = self.calculate_losses(states, options, actions, advantages)       
kl = losses["meankl"]
optimization_gain = losses["gain"]
loss_grad = self.policy.flaten.flatgrad(optimization_gain,retain=True)     
grad_kl = self.policy.flaten.flatgrad(kl,create=True,retain=True)
theta_before = self.policy.flaten.get()
self.log("Init param sum", theta_before.sum())
self.log("explained variance",(vpred-tdlamret).var()/tdlamret.var())
if np.allclose(loss_grad.detach().cpu().numpy(), 0,atol=1e-15):
    print("Got zero gradient. not updating")
else:
    with C.timeit("Conjugate Gradient"):
        stepdir = m_utils.conjugate_gradient(self.Fvp(grad_kl), loss_grad, cg_iters = self.cg_iters)
    
    self.log("Conjugate Gradient in s",C.elapsed)
    assert stepdir.sum()!=float("Inf")
    shs = .5*stepdir.dot(self.Fvp(grad_kl)(stepdir))
    lm = torch.sqrt(shs / self.max_kl)
    self.log("lagrange multiplier:", lm)
    self.log("gnorm:", np.linalg.norm(loss_grad.cpu().detach().numpy()))
    fullstep = stepdir / lm
    expected_improve = loss_grad.dot(fullstep)
    surrogate_before = losses["gain"].detach()
    
    
    
    with C.timeit("Line Search"):
        stepsize = 1.0
        for i in range(10):
            theta_new = theta_before + fullstep * stepsize
            self.policy.flaten.set(theta_new)
            surr = losses["surr_get"]() 
            improve = surr - surrogate_before
            kl = losses["meankl"]
            if surr == float("Inf") or kl ==float("Inf"):
                C.warning("Infinite value of losses")
            elif kl > 1.5*self.max_kl:
                C.warning("Violated KL")
            elif improve < 0:
                stepsize *= self.ls_step
            else:
                self.log("Line Search","OK")
                break
        else:
            improve = 0
            self.log("Line Search","NOPE")
            self.policy.flaten.set(theta_before)

print(improve)
for op in self.options:
    losses["gain"] = losses["surr_get"](grad=True)
    op.train(states, options, actions, advantages,tdlamret,losses)
"""