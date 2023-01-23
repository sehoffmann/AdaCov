import math
import time
import torch
from torch import Tensor
from typing import cast, List, Optional, Dict, Tuple

def sherman_morrison_exp_decay(A: Tensor, v: Tensor, beta: Tensor):
    A_v = torch.matmul(A, v)
    v_A = torch.matmul(v.t(), A)
    divisor = 1 + torch.inner(v, A_v) * ((1-beta) / beta)
    factor = ((1-beta)/(beta**2)) / divisor
    update = torch.outer(A_v, v_A).mul_(factor)
    A.mul_(1/beta).sub_(update)


class AdaCov(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-7,
            weight_decay=0, block_size=32):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, block_size=block_size)
        super(AdaCov, self).__init__(params, defaults)

       
    def __setstate__(self, state):
        super().__setstate__(state)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))


    def _init_state(self, param, block_size):
        state = self.state[param]
        state['step'] = torch.tensor(0.)
        
        # Exponential moving average of gradient values
        state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
        
        # Exponential moving average of gradient covariances
        D = math.prod(param.shape)
        N = int(math.ceil(D/block_size))
        print(D, N, param.shape)
        #state['exp_cov'] = torch.zeros(N,block_size,block_size, device=param.device, dtype=param.dtype)
        state['exp_cov'] = torch.stack([1e8*torch.eye(block_size, device=param.device, dtype=param.dtype) for _ in range(N)])
        print(state['exp_cov'].shape)
    

    def step(self, closure=None, *, grad_scaler=None):
        t1 = time.time()
        with torch.no_grad():
            loss = None

            for group in self.param_groups:
                params_with_grad = []
                grads = []
                exp_avgs = []
                exp_covs = []
                state_steps = []
                beta1, beta2 = group['betas']
                block_size = group['block_size']

                for p in group['params']:
                    if p.grad is not None:
                        params_with_grad.append(p)
                        if p.grad.is_sparse:
                            raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                        grads.append(p.grad)

                        # Lazy state initialization
                        state = self.state[p]
                        if len(state) == 0:
                            self._init_state(p, block_size)

                        exp_avgs.append(state['exp_avg'])
                        exp_covs.append(state['exp_cov'])
                        state_steps.append(state['step'])

                _single_tensor_adam(params_with_grad,
                     grads,
                     exp_avgs,
                     exp_covs,
                     state_steps,
                     beta1=beta1,
                     beta2=beta2,
                     lr=group['lr'],
                     weight_decay=group['weight_decay'],
                     eps=group['eps'],
                     block_size=block_size)
            torch.cuda.synchronize()
            print(f'step took {time.time() - t1:.2f}s')
            return loss

    
def _single_tensor_adam(params: List[Tensor],
                        grads: List[Tensor],
                        exp_avgs: List[Tensor],
                        exp_covs: List[Tensor],
                        state_steps: List[Tensor],
                        *,
                        beta1: float,
                        beta2: float,
                        lr: float,
                        weight_decay: float,
                        eps: float,
                        block_size: int):

    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_cov = exp_covs[i]
        step_t = state_steps[i]

        # update step
        step_t += 1

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        
        if False:
            print(f't: {step_t}')
            print(f'cov: {exp_cov}')
            print(f'g: {grad}')
            print('-----------------------------')

        # Perform step
        step = step_t.item()

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        step_size = lr / bias_correction1

        # Split into chunks
        grad_blocks = torch.split(grad.flatten(), block_size)
        param_blocks = torch.split(param.flatten(), block_size)
        exp_avg_blocks = torch.split(exp_avg.flatten(), block_size)

        # Calculate second order moments and inverse for each chunk
        for j, (grad_block, param_block, exp_avg_block) in enumerate(zip(grad_blocks, param_blocks, exp_avg_blocks)):
            """
            cov = exp_cov[j, :grad_block.shape[0], :grad_block.shape[0]] # last block might be smaller
            cov.addr_(grad_block,grad_block,beta=beta2, alpha=1-beta2)

            # Perform Step
            L,Q = torch.linalg.eigh(cov / bias_correction2)
            L.clamp_(0) # numeric instability of EVD might produce (very small) negative EVs
            L_inv = torch.diag(1 / (L.sqrt()).add_(eps))
            L_inv.mul_((L*1e5).clamp(0,1))
            M = Q.matmul(L_inv).matmul(Q.t())
            param_block.addmv_(M, exp_avg_block, alpha=-step_size)
            """

            inv_cov = exp_cov[j, :grad_block.shape[0], :grad_block.shape[0]] # last block might be smaller
            sherman_morrison_exp_decay(inv_cov, grad_block, beta2)
            M,_ = torch.linalg.cholesky_ex(inv_cov * bias_correction2)
            param_block.addmv_(M.t(), exp_avg_block, alpha=-step_size)