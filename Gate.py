import numpy as np
import torch
import collections
from base.baseagent import BaseAgent
from core.console import Progbar
import core.math as m_utils
import core.utils as U
from Option import OptionTRPO
import core.console as C
import gc

class GateTRPO(BaseAgent):

    name = "GateTRPO"

    def __init__(self,env, gatepolicy, policy_func, value_func, n_options,option_len=3,
        timesteps_per_batch=1000,
        gamma=0.99, lam=0.97, MI_lambda=1e-3,
        max_kl=1e-2,
        cg_iters=10,
        cg_damping=1e-2,
        vf_iters=2,
        max_train=1000,
        ls_step=0.5,
        checkpoint_freq=50):
        
        super(GateTRPO,self).__init__(name=env.name)

        self.n_options=n_options
        #self.name = self.name 
        self.env = env
        self.gamma = gamma
        self.lam = lam
        self.MI_lambda = MI_lambda
        self.current_option = 0
        self.timesteps_per_batch = timesteps_per_batch
        self.max_kl = max_kl
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.max_train = max_train
        self.ls_step = ls_step
        self.checkpoint_freq=checkpoint_freq
        
        self.policy = gatepolicy(env,self.n_options)
        self.oldpolicy = gatepolicy(env,self.n_options,verbose=0)
        self.progbar = Progbar(self.timesteps_per_batch)
        
        self.path_generator = self.roller()        
        self.episodes_reward=collections.deque([],50)
        self.episodes_len=collections.deque([],50)
        self.done = 0
        self.functions = [self.policy]
        
        self.options = [OptionTRPO(env.name, i,
                                   env, policy_func, value_func,
                                   gamma, lam, option_len,
                                   max_kl,cg_iters,cg_damping,vf_iters,ls_step,
                                   self.logger, checkpoint_freq) for i in range(n_options)]
        
    def act(self,state,train=True):
        if train:
            return self.policy.sample(state)
        return self.policy.act(state)
        
    def calculate_losses(self, states, options, actions, advantages):

        RIM = self.KLRIM(states, options, actions)
        old_pi = RIM["old_log_pi_oia_s"]
        pi = RIM["old_log_pi_oia_s"]

        ratio = torch.exp(m_utils.logp(pi,actions) - m_utils.logp(old_pi,actions)) # advantage * pnew / pold
        surrogate_gain = (ratio * advantages).mean()

        optimization_gain = surrogate_gain - self.MI_lambda*RIM["MI"]
        
        def surr_get(grad=False):
            Id,pid = RIM["MI_get"](grad)
            return (torch.exp(m_utils.logp(pid,actions) - m_utils.logp(old_pi,actions))*advantages).mean() - self.MI_lambda*Id
        
        RIM["gain"] = optimization_gain
        RIM["surr_get"] = surr_get
        return RIM

    def train(self):

        while self.done < self.max_train:
            print("="*40)
            print(" "*15, self.done,"\n")
            self.logger.step()
            path = self.path_generator.__next__()
            self.oldpolicy.copy(self.policy)
            for p in self.options:
                p.oldpolicy.copy(p.policy)
            self._train(path)
            self.logger.display()
            if not self.done%self.checkpoint_freq:
                self.save()    
                for p in self.options:
                    p.save()
            self.done = self.done+1
        self.done = 0 
        
    def _train(self,path):

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
                    elif kl > self.max_kl:
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
            

            for op in self.options:
                losses["gain"] = losses["surr_get"](grad=True)
                op.train(states, options, actions, advantages,tdlamret,losses)
            
            surr = losses["surr_get"]() 
            improve = surr- surrogate_before
            self.log("Expected",expected_improve)
            self.log("Actual",improve)
            self.log("Line Search in s",C.elapsed)
            self.log("LS Steps",i)
            self.log("KL",kl)
            self.log("MI",-losses["MI"])
            self.log("MI improve", -losses["MI_get"]()[0]+losses["MI"])
            self.log("Surrogate", surr)
            self.log("Gate KL",losses["KL_gate_get"]())
            self.log("HRL KL",losses["KL_get"]())
            self.log("TDlamret mean",tdlamret.mean())
            self.log("Last 50 rolls mean rew",np.mean(self.episodes_reward))
            self.log("Last 50 rolls mean len",np.mean(self.episodes_len))
            del(improve, surr, kl)
        del(losses, states, options, actions, advantages, tdlamret, vpred, optimization_gain, loss_grad, grad_kl)
        for _ in range(10):
            gc.collect()
    def roller(self):
        
        state = self.env.reset()
        path = {"states":np.array([state for _ in range(self.timesteps_per_batch)]),
                "options":np.zeros(self.timesteps_per_batch).astype(int),
                "actions":np.zeros(self.timesteps_per_batch).astype(int),
                "rewards":np.zeros(self.timesteps_per_batch),
                "terminated":np.zeros(self.timesteps_per_batch),
                "vf": np.zeros(self.timesteps_per_batch)}
        self.current_option = self.act(state)
        self.options[self.current_option].select()
        ep_rews = 0
        ep_len = 0
        t = 0
        done = True
        rew = 0.0
        self.progbar.__init__(self.timesteps_per_batch)
        while True:
            if self.options[self.current_option].finished:
                self.current_option = self.act(state)

            action = self.options[self.current_option].act(state)
            vf = self.options[self.current_option].value_function.predict(state)            
            if t > self.timesteps_per_batch-1:
                path["next_vf"] = vf*(1-done*1.0)
                self.add_vtarg_and_adv(path)
                yield path
                t = 0
                self.progbar.__init__(self.timesteps_per_batch)
            

            path["states"][t] = state
            state, rew, done,_ = self.env.step(action)
            path["options"][t] = self.options[self.current_option].option_n
            path["actions"][t] = action
            path["rewards"][t] = rew
            path["vf"][t] = vf
            path["terminated"][t] = done*1.0
            ep_rews += rew
            ep_len += 1
            t+= 1
            self.progbar.add(1)
                            
            if done:
                state = self.env.reset()
                self.episodes_reward.append(ep_rews)
                self.episodes_len.append(ep_len)
                ep_rews = 0
                ep_len = 0


    def add_vtarg_and_adv(self, path):
        # General Advantage Estimation
        terminal = np.append(path["terminated"],0)
        vpred = np.append(path["vf"], path["next_vf"])
        T = len(path["rewards"])
        path["advantage"] = np.empty(T, 'float32')
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1-terminal[t+1]
            delta = path["rewards"][t] + self.gamma * vpred[t+1] * nonterminal - vpred[t]
            path["advantage"][t] = lastgaelam = delta + self.gamma * self.lam * nonterminal * lastgaelam
        path["tdlamret"] = (path["advantage"] + path["vf"]).reshape(-1,1)

    def Fvp(self,grad_kl):
        def fisher_product(v):
            kl_v = (grad_kl * v).sum()
            grad_grad_kl = self.policy.flaten.flatgrad(kl_v, retain=True)
            return grad_grad_kl + v*self.cg_damping
        return fisher_product
    
    
    def KLRIM(self, states, options, actions):
        
        
        """
        pg : \pi_g
        pi_a_so : \pi(a|s,o)
        pi_oa_s : \pi(o,a|s)
        pi_o_as : \pi(o|a,s)
        pi_a_s  :  \pi(a|s)
        old : \tilde(\pi)
        """
        
        old_log_pi_a_so = torch.cat([p.oldpolicy.logsoftmax(states).unsqueeze(1).detach() for p in self.options],dim=1)
        old_log_pg_o_s = self.oldpolicy.logsoftmax(states).detach()
        old_log_pi_oa_s = old_log_pi_a_so+old_log_pg_o_s.unsqueeze(-1)
        old_log_pi_a_s = old_log_pi_oa_s.exp().sum(1).log()
        old_log_pi_oia_s = old_log_pi_oa_s[np.arange(states.shape[0]),options]
        
        def KL_get(grad=False):
            if grad:
                log_pi_a_so = torch.cat([p.policy.logsoftmax(states).unsqueeze(1) for p in self.options],dim=1)
                log_pg_o_s = self.policy.logsoftmax(states)
            else:
                log_pi_a_so = torch.cat([p.policy.logsoftmax(states).detach().unsqueeze(1) for p in self.options],dim=1)
                log_pg_o_s = self.policy.logsoftmax(states).detach()                
            log_pi_a_s = (log_pi_a_so+log_pg_o_s.unsqueeze(-1)).exp().sum(1).log()    
            mean_kl_new_old = m_utils.kl_logits(old_log_pi_a_s,log_pi_a_s).mean()    
            return mean_kl_new_old

        def KL_gate_get(grad=False):
            if grad:
                log_pg_o_s = self.policy.logsoftmax(states)
            else:                
                log_pg_o_s = self.policy.logsoftmax(states).detach()
            return m_utils.kl_logits(old_log_pg_o_s,log_pg_o_s).mean()

        def MI_get(grad=False):
            if grad:
                log_pi_a_so = torch.cat([p.policy.logsoftmax(states).unsqueeze(1) for p in self.options],dim=1)
                log_pg_o_s = self.policy.logsoftmax(states)
            else:
                log_pi_a_so = torch.cat([p.policy.logsoftmax(states).detach().unsqueeze(1) for p in self.options],dim=1)
                log_pg_o_s = self.policy.logsoftmax(states).detach()
            
            log_pi_oa_s = log_pi_a_so+log_pg_o_s.unsqueeze(-1)
            log_pi_a_s = log_pi_oa_s.exp().sum(1).log()
            log_pi_o_as = log_pi_oa_s - log_pi_a_s.unsqueeze(1)
            log_pi_o_ais = log_pi_o_as[np.arange(states.shape[0]),:,actions].exp().mean(0).log()        
            log_pi_oi_ais = log_pi_o_as[np.arange(states.shape[0]),options,actions]   
            log_pi_oia_s = log_pi_oa_s[np.arange(states.shape[0]),options]
            I = m_utils.entropy_logits(log_pi_o_ais) - m_utils.entropy_logits(log_pi_oi_ais)
            return I, log_pi_oia_s

        log_pi_a_so = torch.cat([p.policy.logsoftmax(states).unsqueeze(1) for p in self.options],dim=1)
        log_pg_o_s = self.policy.logsoftmax(states)
        log_pi_oa_s = log_pi_a_so+log_pg_o_s.unsqueeze(-1)
        log_pi_a_s = log_pi_oa_s.exp().sum(1).log()
        log_pi_o_as = log_pi_oa_s - log_pi_a_s.unsqueeze(1)
        log_pi_oia_s = log_pi_oa_s[np.arange(states.shape[0]),options]
        
        mean_kl_gate = m_utils.kl_logits(old_log_pg_o_s,log_pg_o_s).mean()
        
        log_pi_o_ais = log_pi_o_as[np.arange(states.shape[0]),:,actions].exp().mean(0).log()        
        log_pi_oi_ais = log_pi_o_as[np.arange(states.shape[0]),options,actions]
        

        I = m_utils.entropy_logits(log_pi_o_ais) - m_utils.entropy_logits(log_pi_oi_ais)

        return {"MI": I,
                "meankl": mean_kl_gate,
                "old_log_pi_oia_s": old_log_pi_oia_s,
                "log_pi_oia_s": log_pi_oia_s,
                "MI_get": MI_get,
                "KL_get": KL_get,
                "KL_gate_get": KL_gate_get}
  
    def load(self):
        super(GateTRPO,self).load()

        for p in self.options:
            p.load()
