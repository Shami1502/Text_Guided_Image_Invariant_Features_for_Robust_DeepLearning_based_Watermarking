import torch
import torch.nn.functional as F

def decorrelation_loss(features):
    """
    Computes a decorrelation loss on a feature tensor to reduce redundancy.
    The loss is based on the off-diagonal elements of the covariance matrix.
    
    Args:
        features: Tensor of shape (N, d) where N is batch size and d is feature dimension.
    
    Returns:
        A scalar decorrelation loss.
    """
    features = features - features.mean(dim=0)
    cov = torch.mm(features.t(), features) / features.size(0)
    diag_mask = torch.eye(cov.size(0), device=features.device)
    off_diag_loss = ((cov * (1 - diag_mask)) ** 2).sum()
    return off_diag_loss

def contrastive_loss(image_proj, distorted_proj, text_proj,negative_image_proj, negative_distorted_proj, margin=0.2, lambda_decorrelation=0.1):
    """
    Contrastive loss with negative sampling and added decorrelation loss.
    
    Args:
        image_proj: Tensor of shape (N, d) for image features.
        distorted_proj: Tensor of shape (N, d) for distorted image features.
        text_proj: Tensor of shape (N, d) for text features.
        negative_image_proj: Tensor of shape (N, d) for negative image features.
        negative_distorted_proj: Tensor of shape (N, d) for negative distorted image features.
        margin: Margin used for negative sampling loss.
        lambda_decorrelation: Weighting factor for decorrelation loss.
    
    Returns:
        Scalar loss value.
    """
    # Cosine similarity for positive pairs
    sim_img_text = torch.cosine_similarity(image_proj, text_proj)
    sim_dist_text = torch.cosine_similarity(distorted_proj, text_proj)
    
    # Cosine similarity for negative pairs: pair the negative images with the correct text
    sim_neg_img_text = torch.cosine_similarity(negative_image_proj, text_proj)
    sim_neg_dist_text = torch.cosine_similarity(negative_distorted_proj, text_proj)
    
    # Positive loss: encourage positive pairs to be similar
    pos_loss_img = -torch.log(torch.exp(sim_img_text) /(torch.exp(sim_img_text) + torch.exp(sim_neg_img_text)))
    pos_loss_dist = -torch.log(torch.exp(sim_dist_text) /(torch.exp(sim_dist_text) + torch.exp(sim_neg_dist_text)))

    # Negative sampling loss: ensure negative pairs are less similar by at least a margin
    neg_loss_img = torch.clamp(sim_neg_img_text - sim_img_text + margin, min=0).mean()
    neg_loss_dist = torch.clamp(sim_neg_dist_text - sim_dist_text + margin, min=0).mean()

    # Compute decorrelation loss for each modality
    decorrelation_img = decorrelation_loss(image_proj)
    decorrelation_dist = decorrelation_loss(distorted_proj)
    decorrelation_text = decorrelation_loss(text_proj)

    total_loss = (pos_loss_img.mean() + pos_loss_dist.mean() + neg_loss_img + neg_loss_dist + lambda_decorrelation * (decorrelation_img + decorrelation_dist + decorrelation_text))

    return total_loss
