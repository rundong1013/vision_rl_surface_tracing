import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Cross-modal attention-like fusion of image features and positional target
class CrossAttentionFusion(nn.Module):
    def __init__(self, image_dim, pos_dim=3, out_dim=256):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, out_dim)
        self.pos_proj = nn.Linear(pos_dim, out_dim)
        self.fusion = nn.Sequential(
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, image_feat, target_point):
        img = self.image_proj(image_feat)
        pos = self.pos_proj(target_point)
        fused = img * pos  # element-wise attention-like interaction
        return self.fusion(fused)

# Full visual encoder with CNN + target fusion
class CustomCNNWithFusion(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # output: [B, 128, 1, 1]
            nn.Flatten()
        )

        dummy_input = torch.zeros(1, 3, 224, 224)
        n_flatten = self.cnn(dummy_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )
        # Fusion module that combines image feature + 3D position
        self.fusion = CrossAttentionFusion(features_dim, pos_dim=3, out_dim=features_dim)

    def forward(self, observations):
        x = observations["image"] / 255.0
        image_feat = self.linear(self.cnn(x))

        if "target_point" in observations:
            pos = observations["target_point"]
            if len(pos.shape) == 1:
                pos = pos.unsqueeze(0)
            return self.fusion(image_feat, pos)

        return image_feat
