import torch.nn as nn
from .projection import ProjectionHead

class CLIPInvariantModel(nn.Module):
    def __init__(self, clip_model, projection_dim):
        super(CLIPInvariantModel, self).__init__()
        self.clip_model = clip_model

        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.projection_image = ProjectionHead(768, projection_dim)
        self.projection_text = ProjectionHead(768, projection_dim)

    def forward(self, image, distorted_image, text):
        image_features = self.clip_model.encode_image(image)
        distorted_image_features = self.clip_model.encode_image(distorted_image)
        text_features = self.clip_model.encode_text(text)

        image_proj = self.projection_image(image_features)
        distorted_proj = self.projection_image(distorted_image_features)
        text_proj = self.projection_text(text_features)

        return image_proj, distorted_proj, text_proj
