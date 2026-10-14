"""Vision student policy: image encoder (small CNN or ResNet-18) + MLP state encoder, fused, MLP action head."""

import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def resnet18_encoder(pretrained: bool) -> nn.Module:
    """ResNet-18 without its pooling and classifier, then 64 keypoints through a spatial softmax."""
    import torchvision

    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    trunk = torchvision.models.resnet18(weights=weights)
    return nn.Sequential(*list(trunk.children())[:-2], nn.Conv2d(512, 64, 1), SpatialSoftmax())


class StudentPolicy(nn.Module):
    """Maps (camera image, robot state + goal) to the teacher's action.

    Inputs are raw: images as uint8 (B, H, W, 3), states and actions in environment units. The
    normalization statistics are stored in the module, so a checkpoint is self-contained.

    encoder="cnn" is the original small CNN. encoder="resnet18" is an ImageNet ResNet-18 that is
    fine-tuned; it sees a crop of crop x crop pixels, random during training and centered otherwise.
    """

    def __init__(self, state_dim: int, action_dim: int, encoder: str = "cnn", crop: int = 0, pretrained: bool = True):
        super().__init__()
        self.encoder_name, self.crop = encoder, crop
        if encoder == "cnn":
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 5, stride=2), nn.ReLU(),
                nn.Conv2d(32, 64, 3), nn.ReLU(),
                nn.Conv2d(64, 64, 3), nn.ReLU(),
                nn.Conv2d(64, 64, 3),
                SpatialSoftmax(),
            )
        elif encoder == "resnet18":
            self.encoder = resnet18_encoder(pretrained)
        else:
            raise ValueError(f"unknown encoder {encoder}")
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
        self.register_buffer("pixel_mean", torch.tensor(IMAGENET_MEAN if encoder == "resnet18" else (0.5, 0.5, 0.5)).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(IMAGENET_STD if encoder == "resnet18" else (1.0, 1.0, 1.0)).view(1, 3, 1, 1), persistent=False)

    def set_normalization(self, states: torch.Tensor, actions: torch.Tensor):
        self.state_mean.copy_(states.mean(0))
        self.state_std.copy_(states.std(0).clamp_min(1e-3))
        self.action_mean.copy_(actions.mean(0))
        self.action_std.copy_(actions.std(0).clamp_min(1e-3))

    def normalize_action(self, actions: torch.Tensor) -> torch.Tensor:
        return (actions - self.action_mean) / self.action_std

    def crop_images(self, images: torch.Tensor) -> torch.Tensor:
        """Random crop per sample in training mode, center crop otherwise (no-op when crop is 0)."""
        batch, height, width, _ = images.shape
        if not self.crop or self.crop >= height:
            return images
        offsets = torch.arange(self.crop, device=images.device)
        if self.training:
            top = torch.randint(0, height - self.crop + 1, (batch, 1), device=images.device)
            left = torch.randint(0, width - self.crop + 1, (batch, 1), device=images.device)
            rows, cols = top + offsets, left + offsets
            return images[torch.arange(batch, device=images.device)[:, None, None], rows[:, :, None], cols[:, None, :]]
        top, left = (height - self.crop) // 2, (width - self.crop) // 2
        return images[:, top:top + self.crop, left:left + self.crop]

    def forward_normalized(self, images: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Predicts the action in normalized units (what the training loss is computed on)."""
        pixels = (self.crop_images(images).permute(0, 3, 1, 2).float() / 255.0 - self.pixel_mean) / self.pixel_std
        features = torch.cat([self.encoder(pixels), self.state_encoder((states - self.state_mean) / self.state_std)], dim=1)
        return self.head(features)

    def forward(self, images: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        return self.forward_normalized(images, states) * self.action_std + self.action_mean


def load_student(path: str, device) -> StudentPolicy:
    """Loads a checkpoint written by train_student.py in evaluation mode."""
    checkpoint = torch.load(path, map_location=device)
    student = StudentPolicy(checkpoint["state_dim"], checkpoint["action_dim"], encoder=checkpoint.get("encoder", "cnn"),
                            crop=checkpoint.get("crop", 0), pretrained=False).to(device)
    student.load_state_dict(checkpoint["model"])
    return student.eval()
