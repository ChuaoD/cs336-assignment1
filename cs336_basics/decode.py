from __future__ import annotations

import torch
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.model import TransformerLM
from cs336_basics.utils import softmax

EOF_TOKEN = "<|endoftext|>"

def decode(input: str, 
           model: TransformerLM, 
           tokenizer: Tokenizer, 
           max_num_tokens: int = 100,
           temperature: float = 1, 
           top_p_level = 0.5
) -> str:
    '''
    Generate completions for a user-provided prompt.
    '''
    device = model.device
    token_ids = tokenizer.encode(input)
    with torch.no_grad():
        for _ in range(max_num_tokens):
            # list[int] -> (batch_size=1, sequence_length)
            input_ids = torch.tensor(
                token_ids,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)
            next_token_logits = model(input_ids)[0, -1, :]  # (vocab_size,)
            # Temperature scaling
            probs = softmax(next_token_logits, dim=-1, temperature=temperature)
            # Top-p sampling
            sorted_probs, sorted_ids = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            # Keep the token that crosses the top-p threshold.
            keep = cum_probs - sorted_probs < top_p_level
            top_p_probs = sorted_probs * keep
            sampled_index = torch.multinomial(
                top_p_probs,
                num_samples=1,
            )
            new_token_id = sorted_ids[sampled_index].item()
            token_ids.append(new_token_id)
            if tokenizer.is_eof(new_token_id):
                break
        return tokenizer.decode(token_ids)