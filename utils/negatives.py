import torch
import random

def get_hard_negative_indices(image_proj, threshold=0.5):
    """
    For each image in the batch, find the index of a candidate image (from the same batch)
    that has a cosine similarity greater than the threshold with the positive image feature.
    If no candidate meets the threshold, select a random negative (excluding self).

    Args:
        image_proj: Tensor of shape (N, d) representing positive image features (assumed normalized).
        threshold: Cosine similarity threshold for hard negatives.

    Returns:
        Tensor of shape (N,) containing indices of hard negative images.
    """
    batch_size = image_proj.size(0)
    sim_matrix = torch.mm(image_proj, image_proj.t())
    mask = torch.eye(batch_size, device=image_proj.device).bool()
    sim_matrix.masked_fill_(mask, -1.0)

    max_sim, max_indices = torch.max(sim_matrix, dim=1)

    hard_negative_indices = []
    for i in range(batch_size):
        if max_sim[i] > threshold:
            hard_negative_indices.append(max_indices[i])
        else:
            candidates = list(range(batch_size))
            candidates.remove(i)
            random_idx = random.choice(candidates)
            hard_negative_indices.append(torch.tensor(random_idx, device=image_proj.device))
    return torch.stack(hard_negative_indices)
