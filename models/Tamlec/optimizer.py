from collections import deque
import math
import torch
from torch.optim.optimizer import Optimizer


__all__ = ['DenseSparseAdam']


class GradientClipper(object):
    def __init__(self, clip_value=5.0, queue_length=10, increase_factor=2.0, start_value=1.0, weights = None):
        self.clip_value = clip_value
        self.queue_length = queue_length
        self.increase_factor = increase_factor
        self.start_value = start_value
        self.weights = weights

        self.queue = deque([start_value], maxlen=queue_length)

    def clip_gradient(self, model, task_id = 0):
        max_norm = max(self.queue)

        if self.weights is None : clip_value = self.clip_value
        else : clip_value = self.clip_value * self.weights[task_id]

        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm * clip_value)
        self.queue.append(min(total_norm, max_norm * self.increase_factor, self.start_value))
        has_clipped = (total_norm > max_norm*clip_value)

        return bool(has_clipped.cpu().detach()), total_norm


class DenseSparseAdam(Optimizer):
    """
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(DenseSparseAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.
        Parameters
        ----------
        closure : ``callable``, optional.
            A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if 'step' not in state:
                    state['step'] = 0
                if 'exp_avg' not in state:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                if 'exp_avg_sq' not in state:
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['step'] += 1

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                weight_decay = group['weight_decay']

                if grad.is_sparse:
                    grad = grad.coalesce()  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = grad.size()

                    def make_sparse(values):
                        constructor = grad.new
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor().resize_as_(grad)
                        return constructor(grad_indices, values, size)

                    # Decay the first and second moment running average coefficient
                    #      old <- b * old + (1 - b) * new
                    # <==> old += (1 - b) * (new - old)
                    old_exp_avg_values = exp_avg.sparse_mask(grad)._values()
                    exp_avg_update_values = grad_values.sub(old_exp_avg_values).mul_(1 - beta1)
                    exp_avg.add_(make_sparse(exp_avg_update_values))
                    old_exp_avg_sq_values = exp_avg_sq.sparse_mask(grad)._values()
                    exp_avg_sq_update_values = grad_values.pow(2).sub_(old_exp_avg_sq_values).mul_(1 - beta2)
                    exp_avg_sq.add_(make_sparse(exp_avg_sq_update_values))

                    # Dense addition again is intended, avoiding another sparse_mask
                    numer = exp_avg_update_values.add_(old_exp_avg_values)
                    exp_avg_sq_update_values.add_(old_exp_avg_sq_values)
                    denom = exp_avg_sq_update_values.sqrt_().add_(group['eps'])
                    del exp_avg_update_values, exp_avg_sq_update_values

                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                    p.data.add_(make_sparse(-step_size * numer.div_(denom)))
                    if weight_decay > 0.0:
                        p.data.add_(-group['lr'] * weight_decay, p.data.sparse_mask(grad))
                else:
                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                    p.data.addcdiv_(-step_size, exp_avg, denom)
                    if weight_decay > 0.0:
                        p.data.add_(-group['lr'] * weight_decay, p.data)

        return loss

        
def rate(step, model_size, factor, warmup):
    """
    - Increase the learning rate linearly for the first warmup_steps.
    - Decrease it thereafter proportionally to the inverse square root of the step number.
    """

    # we have to default the step to 1 for LambdaLR function to avoid zero raising to negative power.
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


def arate(step, factor, warmup=100):
    step = max(1, step - warmup)

    r = factor * (step ** (-0.5))
    return r
