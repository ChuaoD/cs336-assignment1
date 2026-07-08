import math
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter
from einops import reduce

from .linear import Linear


class RMSNorm(nn.Module):

    def __init__(self, 
                 d_model: int, 
                 eps: float = 1e-5, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None
    ):
        '''
        Construct the RMSNorm module.

        Args:
            d_model: Hidden dimension of the model
            eps: Epsilon value for numerical stability
            device: Device to store the parameters on
            dtype: Data type of the parameters
        '''
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.w1_weight = Parameter(torch.ones(d_model, device=device, dtype=dtype),
                               requires_grad=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        '''
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        rms = torch.sqrt(reduce(x * x, "b s d -> b s 1", "sum") / self.d_model + self.eps)
        result = x / rms * self.weight

        # Return the result in the original dtype
        return result.to(in_dtype)

    
class SwiGLU(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 d_ff: int | None = None, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None
    ):
        '''
        Construct the RMSNorm module.

        Args:
            d_model: Hidden dimension of the model
            d_ff: Dimension of the inner feed-forward layer
            device: Device to store the parameters on
            dtype: Data type of the parameters
        '''
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            d_ff = round(8.0 / 3.0 * d_model / 64) * 64
        self.d_ff = d_ff

        self.linear1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.linear2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.linear3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.linear1(x)
        out_silu = out1 * torch.sigmoid(out1)
        return self.linear2(out_silu * self.linear3(x))
      