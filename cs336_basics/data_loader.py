import torch
import numpy.typing as npt
import numpy as np

def load_data(
    dataset: npt.NDArray, 
    batch_size: int, 
    context_length: int, 
    device: str,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device to place the sampled input sequences and labels on.
        rng: Optional random generator. If omitted, use NumPy's global random state.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item is the sampled input sequences, and the second tuple item is the corresponding language modeling labels.
    """
    # Sample batch_size contiguous sequences of length context_length
    if rng is None:
        starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    else:
        starts = rng.integers(0, len(dataset) - context_length, size=batch_size)
    sample_indices = starts[:, None] + np.arange(context_length)[None, :]
    samples = torch.as_tensor(dataset[sample_indices], dtype=torch.long, 
                              device=device)
    label_indices = (starts + 1)[:, None] + np.arange(context_length)[None, :]
    labels = torch.as_tensor(dataset[label_indices], dtype=torch.long, 
                             device=device)
    return samples, labels
