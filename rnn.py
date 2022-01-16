import numpy as np
import torch
from torch import nn
from torchdiffeq import odeint
from copy import deepcopy

"""
Code author: Kabir Dabholkar

From the paper:
Charting and Navigating the Space of Solutions for Recurrent Neural Networks (NeurIPS 2021)
Elia Turner, Kabir Dabholkar, Omri Barak
"""


class smooth_rnn(nn.Module):
    """
    A class to implement a batch of 2-unit RNNs with continuous time dynamics.
    """
    def __init__(self,
                 W=None,
                 x0=[np.cos(np.pi/180*(0)),   # initial network state set to (1,0) 
                     np.sin(np.pi/180*(0))],
                 dt=0.1,
                 device='cpu',
                 dtype=torch.float64,
                 batches=1,
                 rtol=1e-9,
                 atol=1e-9,
                 phi = nn.Tanh(),
                 method='euler'):
        super().__init__()
        local_vars=locals().copy()
        [setattr(self,item,local_vars[item]) for item in local_vars.keys() if item != 'self']
        self.set_x0(x0)
        self.init_params(W=W)
    
    def init_params(self,W=None):
        """
        Initialise parameters from Unif(-1.5,1.5)
        """
        self.W = torch.rand((self.batches,2,2),device=self.device,dtype=self.dtype) * 3.0 - 1.5 if W is None else W
        self.W.requires_grad=True
        self.to_learn = [self.W]
    
    def set_x0(self,x0):
        self.x0 = torch.tensor(x0,dtype=self.dtype,device=self.device)[None,:,None].repeat(self.batches,1,1)
        
    def dxdt(self,t,x):
        """
        ODE for RNN.
        """
         #nn.Identity()# #nn.LeakyReLU() #nn.Tanh() #nn.Sigmoid()
        return -x + self.W @ self.phi(x)

    def sample(self,goodidx):
        subsampledself = deepcopy(self)
        subsampledself.W = self.W[goodidx].clone().detach()
        subsampledself.x0 = self.x0[goodidx].clone().detach()
        subsampledself.batches = len(goodidx)
        return subsampledself

    def integrate(self,T,x0=None):
        times = torch.tensor(np.arange(0,T+self.dt,self.dt),device=self.device,dtype=self.dtype)
        options=dict(step_size=self.dt) if self.method in ['euler','rk4'] else None
        return odeint(self.dxdt,self.x0 if x0 is None else x0,times,rtol=self.rtol,atol=self.atol,method=self.method,options=options)

