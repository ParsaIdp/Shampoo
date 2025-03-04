import torch
from torch.optim.optimizer import Optimizer

def _matrix_power(matrix: torch.Tensor, power: float) -> torch.Tensor:
    
    device = matrix.device
    matrix = matrix.cpu()
    u, s, v = torch.svd(matrix)
    return (u @ s.pow_(power).diag() @ v.t()).to(device)


class Shampoo(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, weight_decay=0, epsilon=1e-4, update_freq=1):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, epsilon=epsilon, update_freq=update_freq)
        super(Shampoo, self).__init__(params, defaults)

    
    def step(self, closure = None):
        loss = None
        if closure is not None:
            loss = closure()

        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                grad_dim = grad.ndimension()
                orig_shape = grad.size()
                state = self.state[p]
                momentum = group["momentum"]
                lr = group["lr"]
                weight_decay = group["weight_decay"]

                if len(state) == 0:
                    state['step'] = 0
                    if momentum > 0:
                        state['momentum_buffer'] = grad.clone()
                    for dim_id, dim in enumerate(grad.size()):
                        state[f"precond_{dim_id}"] = group["epsilon"] * torch.eye(dim, out=grad.new(dim, dim))
                        state[f"inv_precond_{dim_id}"] = grad.new(dim, dim).zero_()
                    
                if momentum > 0:
                    grad.mul_(1 - momentum).add_(state["momentum_buffer"], alpha=momentum)
                
                if weight_decay > 0:
                    grad.add_(p.data, alpha=weight_decay)
                
                for dim_id, dim in enumerate(grad.size()):
                    precond = state[f"precond_{dim_id}"]
                    inv_precond = state[f"inv_precond_{dim_id}"]

                    grad = grad.transpose_(0, dim_id).contiguous()
                    transposed_shape = grad.size()
                    grad = grad.view(dim, -1)
                    gradt = grad.t()
                    precond.add_(grad @ gradt)

                    if state['step'] % group["update_freq"] == 0:
                        inv_precond.copy_(_matrix_power(precond, -1 / grad_dim))

                    if dim_id == grad_dim - 1:
                        grad = gradt @ inv_precond
                        grad = grad.view(orig_shape)
                    
                    else:
                        grad = inv_precond @ grad
                        grad = grad.view(transposed_shape)
                    
                state['step'] += 1
                state['momentum_buffer'] = grad
                p.data.add_(grad, alpha=-lr)
                
        return loss

                    

                    

                    

                        


        


