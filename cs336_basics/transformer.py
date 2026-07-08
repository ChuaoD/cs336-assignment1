import math
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter
from einops import reduce, rearrange, einsum

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
        self.weight = Parameter(torch.ones(d_model, device=device, dtype=dtype), requires_grad=True)
        
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
    
class RotaryPositionalEmbedding(nn.Module):
    
    def __init__(self, 
                 theta: float, 
                 d_k: int, 
                 max_seq_len: int, 
                 device: torch.device | None = None):
        '''
        Construct the RoPE module and create buffers.

        Args:
            theta: Theta value for the RoPE
            d_k: Dimension of query and key vectors
            max_seq_len: Maximum sequence length that will be input
            device: Device to store the buffer on
        '''
        super().__init__()
        assert d_k % 2 == 0
        num_pairs = d_k // 2
        freq = theta ** (-2 * torch.arange(num_pairs, device=device) / d_k)
        freq = rearrange(freq, "d -> 1 d")
        position = torch.arange(max_seq_len, device=device)
        position = rearrange(position, "seq_len -> seq_len 1")
        self.register_buffer("sin", torch.sin(position * freq), 
                             persistent=False)  # (max_seq_len, num_pairs)
        self.register_buffer("cos", torch.cos(position * freq), 
                             persistent=False)  # (max_seq_len, num_pairs)

    def forward(self, 
                x: torch.Tensor, 
                token_positions: torch.Tensor
    ) -> torch.Tensor:
        '''
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. The token positions are a tensor of shape (..., seq_len) specifying the token positions of x.
        '''
        x_paired = rearrange(x, "... seq_len (num_pairs pair_dim) -> ... seq_len num_pairs pair_dim", pair_dim=2)
        result = torch.empty_like(x_paired)

        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        cos = rearrange(cos, "seq_len num_pairs -> seq_len num_pairs 1")
        sin = rearrange(sin, "seq_len num_pairs -> seq_len num_pairs 1")
        
        cos_res = cos * x_paired
        sin_res = sin * x_paired

        result[..., 0] = cos_res[..., 0] - sin_res[..., 1]
        result[..., 1] = sin_res[..., 0] + cos_res[..., 1]

        result = rearrange(result, "... seq_len num_pairs pair_dim -> ... seq_len (num_pairs pair_dim)", pair_dim=2)
  
        return result