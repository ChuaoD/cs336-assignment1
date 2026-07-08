import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter

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