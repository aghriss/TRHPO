import numpy as np
import torch
from base.baseagent import BaseAgent
import core.math as m_utils
import core.utils as U
import core.console as C


class OptionTRPO(BaseAgent):

    name = "OptionTRPO"

    def __init__(self,name, option_n, env, policy_func, value_func, gamma, lam, option_len,
                                   max_kl, cg_iters, cg_damping, vf_iters,ls_step,
                                   logger, checkpoint_freq):
        
        super(OptionTRPO,self).__init__(name="Op%i_"%option_n+self.name+env.name)
        self.option_n = option_n
        self.gamma = gamma
        self.lam = lam
        self.option_len = option_len
        self.current_step = option_len
        self.max_kl = max_kl
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.vf_iters = vf_iters
        self.ls_step = ls_step
        
        self.policy = policy_func(env,verbose=0)
        self.oldpolicy = policy_func(env,verbose=0)
        self.value_function = value_func(env,verbose=0)
                
        self.functions = [self.policy, self.value_function]
        self.logger = logger

    def act(self,state,train=True):
        if train:
            x = self.policy.sample(state)
        else:
            x = self.policy.act(state)
        self.current_step +=1
        return x

    def calculate_kl(self, states, options, actions, advantages):

        old_pi = self.oldpolicy(states).detach()

        def kl_get(grad=False):
            if grad:
                return m_utils.kl_logits(old_pi,self.policy(states)).mean() 
            return m_utils.kl_logits(old_pi,self.policy(states).detach()).mean()            
        mean_kl = m_utils.kl_logits(old_pi,self.policy(states)).mean()

        return {"meankl":mean_kl, "kl_get":kl_get}           

    def train(self,states, options, actions, advantages, tdlamret, gate_losses):

        get_kl = self.calculate_kl(states, options, actions, advantages)
        loss_grad = self.policy.flaten.flatgrad(gate_losses["gain"], retain=True)
        grad_kl = self.policy.flaten.flatgrad(get_kl["meankl"], create=True, retain=True)

        theta_before = self.policy.flaten.get()
        self.log("Init param sum", theta_before.sum())        
        if np.allclose(U.get(loss_grad), 0, atol=1e-15):
            C.warning("Got zero gradient. not updating %i"%self.option_n)
        else:
            stepdir = m_utils.conjugate_gradient(self.Fvp(grad_kl), loss_grad, cg_iters=self.cg_iters)

            assert stepdir.sum()!=float("Inf")
            shs = .5*stepdir.dot(self.Fvp(grad_kl)(stepdir))
            lm = torch.sqrt(shs / self.max_kl)
            self.log("lagrange multiplier:", lm)
            self.log("gnorm:", np.linalg.norm(loss_grad.cpu().detach().numpy()))
            fullstep = stepdir / lm
            expected_improve = loss_grad.dot(fullstep)
            surrogate_before = gate_losses["surr_get"]()
            
            with C.timeit("Line Search"):
                stepsize = 1.0
                for i in range(10):
                    theta_new = theta_before + fullstep * stepsize
                    self.policy.flaten.set(theta_new)
                    #losses = self.calculate_losses(states,actions,advantages)
                    surr = gate_losses["surr_get"]()
                    improve = surr - surrogate_before
                    kl = get_kl["kl_get"]()
                    if surr == float("Inf") or kl ==float("Inf"):
                        C.warning("Infinite value of losses %i"%self.option_n)
                    elif kl > 1.5*self.max_kl:
                        C.warning("Violated KL %i"%self.option_n)
                    elif improve < 0:
                        stepsize *= self.ls_step
                    else:
                        self.log("Line Search","OK")
                        break
                else:
                    improve = 0
                    self.log("Line Search","NOPE")
                    self.policy.flaten.set(theta_before)

            self.log("Expected",expected_improve)
            self.log("Actual",improve)
            self.log("LS Steps",i)
            self.log("KL",kl)
        self.log("Selected",(options==self.option_n).sum())
        loss = self.value_function.fit(states[options==self.option_n], tdlamret[options==self.option_n], batch_size = 32, epochs = self.vf_iters,l1_decay=1e-4)
        self.log("Vfunction loss",loss)
        self.log("TDlamret mean",tdlamret[options==self.option_n].mean())

    def Fvp(self,grad_kl):
        def fisher_product(v):
            kl_v = (grad_kl * v).sum()
            grad_grad_kl = self.policy.flaten.flatgrad(kl_v, retain=True)
            return grad_grad_kl + v * self.cg_damping        
        return fisher_product

    def select(self):
        self.current_step = 0

    @property
    def finished(self):
        return self.current_step >= self.option_len
    
    def log(self,a,b):
        self.logger.log("Op%i"%self.option_n + a, b)