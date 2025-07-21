import torch
from einops import rearrange, einsum, reduce, repeat
import numpy as np
from collections.abc import Callable, Iterable
from typing import Optional
import math


#################### cross_entropy
def cross_entropy(logits : torch.Tensor, next_token: torch.Tensor):
    #print(logits.shape)
    #print(next_token.shape)
    def avg_loss(logits, next_token):
        max_input = torch.max(logits, -1, keepdim=True).values

        # print(type(max_input),type(x))
        logits_reduced = logits - max_input
        #print(torch.max(logits),torch.min(logits))
        dom = torch.squeeze(torch.log(torch.sum(torch.exp(logits_reduced), -1, keepdim=True)))
        # print(next_token.shape)


        nume = logits_reduced[torch.arange(logits_reduced.size(0)),next_token]
        #nume = logits_reduced[torch.arange(logits_reduced.size(0)), next_token_flattened]
        #print(dom.shape, nume.shape)
        # print(dom.shape)
        # print(nume.shape)
        #print(dom,nume)
        loss = dom - nume
        # print(loss)
        loss_avg = torch.mean(loss)
        # shape of x (logit) ((batch) seq_len num_vocabulary)
        # shape of next_token ((batch), seq_len)
        # print()
        # print(logits.shape)
        # print(next_token.shape)
        return loss_avg
    if len(logits.shape) == 3:
        # for batch_index in range(logits.shape[0]):
        #     logits_batch = logits[batch_index,:,:]
        #     next_token_batch = next_token[batch_index,:]
        #     if batch_index == 0:
        #         loss_avg = avg_loss(logits_batch, next_token_batch)
        #     else:
        #         loss_avg += avg_loss(logits_batch, next_token_batch)
        # loss_avg /= logits.shape[0]
        # return loss_avg
        logits = rearrange(logits, 'b s v -> (b s) v')
        next_token = rearrange(next_token, 'b s -> (b s)')
        return avg_loss(logits, next_token)
    else:
        return avg_loss(logits, next_token)
    

#################### SGD
class SGD(torch.optim.Optimizer):
    def __init__(self,params,lr = 1e-2):
        defaults = {"lr": lr}
        super().__init__(params,defaults)
    def step(self,closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            for p in group["params"]:
                if p.grad is None:
                    pass
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / np.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss

#################### test_SGD
def test_SGD():   
    def try_SGD(lr):
        
        weights = torch.nn.Parameter(5 * torch.randn((10,10)))
        
        opt = SGD([weights], lr = lr)
        for t in range(10):
            opt.zero_grad()
            loss = (weights**2).mean()
            print(loss.cpu().item())
            loss.backward()
            opt.step()
    lr_list = [1,10,100,1000]
    for lr in lr_list:
        print("Using learning rate",lr)
        try_SGD(lr)


#test_SGD()
class AdamW(torch.optim.Optimizer):
    def __init__(self,params,lr = 1e-2, betas=(0.9,0.999),weight_decay = 0.001,eps = 1e-8):
        defaults = {'lr':lr, "beta_1" : betas[0], "beta_2": betas[1], "weight_decay" : weight_decay, "eps": eps}
        super().__init__(params,defaults)
    def step(self,closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            beta_1 = group['beta_1']
            beta_2 = group['beta_2']
            weight_decay = group['weight_decay']
            eps = group['eps']
            for p in group["params"]:
                state = self.state[p]
                m = state.get('m',0)
                v = state.get('v',0)
                t = state.get('t',1)
                grad = p.grad.data
                m = beta_1 * m + (1-beta_1) * grad
                v = beta_2 * v + (1-beta_2) * grad**2
                alpha_t = lr * np.sqrt((1 - beta_2**t))/(1-beta_1**t)
                p.data -= alpha_t*m/(torch.sqrt(v)+eps)
                p.data -= lr * weight_decay * p.data
                state['t'] = t + 1
                state['v'] = v
                state['m'] = m


#################### learning_rate_schedule
def learning_rate_schedule(t,alphamax, alphamin, Tw,Tc):
    if t< Tw:
        return alphamax * t / Tw
    if  Tc >= t >= Tw:
        return alphamin + 1/2*(1+math.cos((t-Tw)/(Tc-Tw)*math.pi))*(alphamax-alphamin)
    if t> Tc:
        return alphamin
    

#################### gradient_clipping
def gradient_clipping(param,M):
    l2 = 0
    epsilon = 10**(-6)
    for p in param:
        if p.grad is not None:
        #print(p.grad)
            l2 += torch.sum(p.grad**2)
    l2 = torch.sqrt(l2)
    for p in param:
        if p.grad is not None:
        #print(p.grad)
            
            p.grad =  p.grad*M/(l2 + epsilon)


