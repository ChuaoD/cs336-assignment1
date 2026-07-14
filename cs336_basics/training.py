import math
import torch
from torch.nn.parameter import Parameter

from collections.abc import Callable, Iterable
from typing import Optional, IO, BinaryIO
import numpy.typing as npt
import numpy as np
import os

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
    
def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """

    if (it < warmup_iters):
        return it / warmup_iters * max_learning_rate
    elif (it >= warmup_iters and it <= cosine_cycle_iters):
        return min_learning_rate + 0.5 * (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate
    
def gradient_clipping(
    parameters: Iterable[Parameter], 
    max_l2_norm: float,
    eps: float = 1e-6
) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    parameters = list(parameters)
    grad_squares = 0
    for p in parameters:
        grad = p.grad
        if grad is not None:
            grad_squares += (grad ** 2).sum()
    l2_norm = math.sqrt(grad_squares.item())
    if l2_norm > max_l2_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad *= max_l2_norm / (l2_norm + eps)

def load_data(
    dataset: npt.NDArray, 
    batch_size: int, 
    context_length: int, 
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item is the sampled input sequences, and the second tuple item is the corresponding language modeling labels.
    """
    # Sample batch_size contiguous sequences of length context_length
    starts = np.random.randint(0,  len(dataset) - context_length, 
                               size=batch_size)
    sample_indices = starts[:, None] + np.arange(context_length)[None, :]
    samples = torch.as_tensor(dataset[sample_indices], dtype=torch.long, 
                              device=device)
    label_indices = (starts + 1)[:, None] + np.arange(context_length)[None, :]
    labels = torch.as_tensor(dataset[label_indices], dtype=torch.long, 
                             device=device)
    return samples, labels

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    checkpoint = {
        "model_states": model.state_dict(),
        "optimizer_states": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_states"])
    optimizer.load_state_dict(checkpoint["optimizer_states"])
    return checkpoint["iteration"]