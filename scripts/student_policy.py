"""Vision student policy: small CNN image encoder + MLP state encoder, fused, MLP action head."""

import torch
import torch.nn as nn


class SpatialSoftmax(nn.Module):
    """Turns each feature map into the expected (x, y) image position of its activation."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = features.shape
        weights = torch.softmax(features.flatten(2) / self.temperature, dim=-1)
        ys = torch.linspace(-1.0, 1.0, height, device=features.device)
        xs = torch.linspace(-1.0, 1.0, width, device=features.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        expected_x = (weights * grid_x.flatten()).sum(-1)
        expected_y = (weights * grid_y.flatten()).sum(-1)
        return torch.cat([expected_x, expected_y], dim=1)


class StudentPolicy(nn.Module):
    """Maps (camera image, robot state + goal) to the teacher's action.

    Inputs are raw: images as uint8 (B, H, W, 3), states and actions in environment units. The
    normalization statistics are stored in the module, so a checkpoint is self-contained.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3), nn.ReLU(),
            nn.Conv2d(64, 64, 3), nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            SpatialSoftmax(),
        )
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(2 * 64 + 64, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_std", torch.ones(action_dim))

    def set_normalization(self, states: torch.Tensor, actions: torch.Tensor):
        self.state_mean.copy_(states.mean(0))
        self.state_std.copy_(states.std(0).clamp_min(1e-3))
        self.action_mean.copy_(actions.mean(0))
        self.action_std.copy_(actions.std(0).clamp_min(1e-3))

    def normalize_action(self, actions: torch.Tensor) -> torch.Tensor:
        return (actions - self.action_mean) / self.action_std

    def forward_normalized(self, images: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Predicts the action in normalized units (what the training loss is computed on)."""
        pixels = images.permute(0, 3, 1, 2).float() / 255.0 - 0.5
        features = torch.cat([self.encoder(pixels), self.state_encoder((states - self.state_mean) / self.state_std)], dim=1)
        return self.head(features)

    def forward(self, images: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        return self.forward_normalized(images, states) * self.action_std + self.action_mean
