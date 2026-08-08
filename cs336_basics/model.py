import math
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter
from einops import reduce, rearrange, einsum

from jaxtyping import Float, Int
from torch import Tensor

from .utils import silu, attention


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
    

class Embedding(nn.Module):
    def __init__(self, 
                 num_embeddings: int, 
                 embedding_dim: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None
    ) -> None:
        '''
        Construct an embedding module.

        Args:
            num_embeddings: Size of the vocabulary
            embedding_dim: Dimension of the embedding vectors
            device: Device to store the parameters on
            dtype: Data type of the parameters
        '''
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(
            init.trunc_normal_(
                torch.empty((num_embeddings, embedding_dim), 
                            device=device, dtype=dtype), 
                mean=0.0, std=1.0, a=-3.0, b=3.0),
            requires_grad=True)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        '''
        Lookup the embedding vectors for the given token IDs.
        '''
        return self.weight[token_ids]


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
        output = x / rms * self.weight

        # Return the output in the original dtype
        return output.to(in_dtype)
    

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

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))
    
class RotaryPositionalEmbedding(nn.Module):
    
    def __init__(self, 
                 theta: float, 
                 d_k: int, 
                 max_seq_len: int, 
                 device: torch.device | None = None
    ):
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
                token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        '''
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. The token positions are a tensor of shape (..., seq_len) specifying the token positions of x.
        '''
        x_paired = rearrange(x, "... seq_len (num_pairs pair_dim) -> ... seq_len num_pairs pair_dim", pair_dim=2)
        output = torch.empty_like(x_paired)

        if token_positions is None:
            token_positions = torch.arange(x.shape[-2])
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        cos = rearrange(cos, "... seq_len num_pairs -> ... seq_len num_pairs 1")
        sin = rearrange(sin, "... seq_len num_pairs -> ... seq_len num_pairs 1")

        cos_res = cos * x_paired
        sin_res = sin * x_paired

        output[..., 0] = cos_res[..., 0] - sin_res[..., 1]
        output[..., 1] = sin_res[..., 0] + cos_res[..., 1]

        output = rearrange(output, "... seq_len num_pairs pair_dim -> ... seq_len (num_pairs pair_dim)", pair_dim=2)
  
        return output


class MultiheadAttention(nn.Module):
    def __init__(self, 
                 d_model: int,
                 num_heads: int,
                 with_rope: bool = False,
                 theta: float = None,
                 max_seq_len: int = None,
                 device: torch.device | None = None
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.with_rope = with_rope
        d_k = d_model // num_heads
        d_v = d_k
        if self.with_rope:
            self.rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len, device)
        self.q_proj = Linear(d_model, num_heads * d_k, device=device)
        self.k_proj = Linear(d_model, num_heads * d_k, device=device)
        self.v_proj = Linear(d_model, num_heads * d_v, device=device)
        self.output_proj = Linear(num_heads * d_v, d_model, device=device)
        
        
    def forward(self, 
                x: Float[Tensor, " ... seq_len d_model"],
                token_positions: Int[Tensor, "... seq_len"] | None = None
    ):
        
        d_k = self.d_model // self.num_heads
        d_v = d_k
        Q = rearrange(self.q_proj(x), "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", d_k = d_k)
        K = rearrange(self.k_proj(x), "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", d_k = d_k)
        V = rearrange(self.v_proj(x), "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", d_v = d_v)

        # Apply RoPE
        if self.with_rope:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device))
        att = attention(Q, K, V, mask)
        att = rearrange(att, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)")

        output = self.output_proj(att)

        return output
    
class TransformerBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int, 
                 d_ff: int,
                 theta: float | None = None,
                 max_seq_len: int | None = None,
                 device: torch.device | None = None
    ) -> None:
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device)
        self.ln2 = RMSNorm(d_model, device=device)
        self.ffn = SwiGLU(d_model, d_ff, device)
        self.attn = MultiheadAttention(d_model, num_heads, with_rope=True,
                                       theta=theta, max_seq_len=max_seq_len, device=device)
    
    def forward(self, x: Tensor, token_positions: Tensor | None = None):
        out1 = x + self.attn(self.ln1(x), token_positions)
        return out1 + self.ffn(self.ln2(out1))
    
class TransformerLM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 num_layers: int,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 theta: float | None = None,
                 device: torch.device | None = None
    ) -> None:
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff, theta=theta, max_seq_len=context_length, device=device
            ) 
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device)
        self.lm_head = Linear(d_model, vocab_size, device=device)
        self.device = device

    def forward(self, x: Tensor, token_positions: Tensor | None = None):
        out = self.token_embeddings(x)
        for layer in self.layers:
            out = layer(out, token_positions)
        out= self.ln_final(out)
        return self.lm_head(out)
