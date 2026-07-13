import math
import torch
from torch.nn.parameter import Parameter
from einops import reduce, rearrange, einsum

from collections.abc import Callable, Iterable
from typing import Optional

from jaxtyping import Float, Int
from torch import Tensor

def cross_entropy_loss(
    inputs: Float[Tensor, "batch_size vocab_size"], 
    targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    '''
    Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs: inputs[i][j] is the unnormalized logit of jth class for the ith example.
        targets: Tensor of shape (batch_size,) with the index of the correct class. Each value must be between 0 and `num_classes - 1`.

    Returns:
        The average cross-entropy loss across examples.
    '''
    batch_size = inputs.shape[0]
    inputs_shifted = inputs - inputs.amax(dim=-1, keepdim=True)
    score_sum = torch.exp(inputs_shifted).sum(-1)
    loss = (
        torch.log(score_sum) 
        - inputs_shifted[torch.arange(batch_size, device=inputs.device), 
                         targets]
    )
    
    return loss.mean()


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not (
            (isinstance(betas[0], float) and isinstance(betas[1], float))
        ):
            raise ValueError("betas must be either both floats or both Tensors")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p
                t = state.get("t", 1)  # Get iteration number from the state
                m = state.get("m", 0)  # Get the first moment
                v = state.get("v", 0)  # Get the second moment

                p.data -= lr * weight_decay * p.data  # Apply weight decay
                
                grad = p.grad.data
                m = beta1 * m + (1- beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad * grad) 

                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss

