import math
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter
from einops import einsum

class Linear(nn.Module):
    '''
    Applies an affine linear transformation to the incoming data:
    y = Wx.
    '''
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None
    ) -> None:
        '''
        Construct a linear transformation module. 
        Args:
            in_features: final dimension of the input
            out_features: final dimension of the output
            device: Device to store the parameters on
            dtype: Data type of the parameters
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_std = math.sqrt(2.0 / (in_features + out_features))
        self.weight = Parameter(
            init.trunc_normal_(
                torch.empty((out_features, in_features), 
                            device=device, dtype=dtype), 
                mean=0.0, std=init_std, a=-3*init_std, b=3*init_std),
            requires_grad=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Apply the linear transformation to the input.
        ''' 
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
        