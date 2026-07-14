import math
import torch
from torch.nn.parameter import Parameter
from einops import einsum

from jaxtyping import Bool, Float, Int
from torch import Tensor

from collections.abc import Iterable


def silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying SiLU to each element.
    """
    return in_features * torch.sigmoid(in_features)

def softmax(x: torch.Tensor, dim: int):
    '''
    Apply a softmax operation on a tensor.

    Args:
        x (Tensor): input
        dim (int): Dimension along which softmax will be computed.
    '''
    scores = torch.exp(x - x.amax(dim=dim, keepdim=True))
    return scores / scores.sum(dim=dim, keepdim=True)

def attention(
    Q: Float[Tensor, " ... num_q d_k"],
    K: Float[Tensor, " ... num_k d_k"],
    V: Float[Tensor, " ... num_k d_v"],
    mask: Bool[Tensor, " ... num_q num_k"] | None = None,
) -> Float[Tensor, " ... num_q d_v"]:

    '''
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    '''
    d_k = K.shape[-1]
    score = einsum(Q, K, "... num_q d_k, ... num_k d_k -> ... num_q num_k") / math.sqrt(d_k)
    if mask is not None:
        score = score.masked_fill(~mask, value=float("-inf"))
    probabilities = softmax(score, dim=-1)
    output = einsum(probabilities, V, "... num_q num_k, ... num_k d_v -> ... num_q d_v")
    return output

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