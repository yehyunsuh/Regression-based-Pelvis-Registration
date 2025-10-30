import cv2
import timm
import torch
import numpy as np
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torch.nn as nn

class RegressionModel(nn.Module):
    def __init__(self, model_name="resnet18", num_outputs=6, pretrained=True, pe_channels=2):
        super(RegressionModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        # Modify the first convolutional layer to accept (6 + pe_channels) channels
        old_conv = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            in_channels=6 + pe_channels,  
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )

        # Copy original conv weights for the first 6 channels
        with torch.no_grad():
            self.model.conv1.weight[:, :6] = old_conv.weight.repeat(1, 2, 1, 1)[:, :6]
            self.model.conv1.weight[:, 6:] = torch.zeros_like(self.model.conv1.weight[:, 6:])

        self.fc = nn.Linear(self.model.num_features, num_outputs)

    def forward(self, x):
        features = self.model(x)
        output = self.fc(features)
        return output


def load_model(weight_path, model_name="resnet18", num_outputs=3, DEVICE="cpu", pe_channels=10):
    """Load the trained model weights."""
    model = RegressionModel(model_name=model_name, num_outputs=6, pe_channels=pe_channels*2).to(DEVICE)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model


def generate_sinusoidal_embeddings(height, width, channels):
    """
    Generate a 2D sinusoidal positional embedding of shape (channels, height, width).
    """
    y_pos = torch.arange(height).unsqueeze(1).repeat(1, width)  # Shape: (H, W)
    x_pos = torch.arange(width).unsqueeze(0).repeat(height, 1)  # Shape: (H, W)

    div_term = torch.exp(torch.arange(0, channels, 2).float() * (-np.log(10000.0) / channels))
    
    pe = torch.zeros(channels, height, width)
    pe[0::2, :, :] = torch.sin(y_pos.unsqueeze(0) * div_term[:, None, None])  # Apply sin to even indices
    pe[1::2, :, :] = torch.cos(x_pos.unsqueeze(0) * div_term[:, None, None])  # Apply cos to odd indices

    return pe  # Shape: (C, H, W)


def test_model(args, images, images_baseline, DEVICE, experiment_type):
    """Run inference on randomly generated images and visualize predictions."""
    model_weights_path = args.weight_path
    pe_channels = 10
    model = load_model(model_weights_path, args.model_name, DEVICE=DEVICE, pe_channels=pe_channels)

    images = cv2.imread(f'tmp/{args.wandb_project}_target_{experiment_type}.png')
    images_baseline = cv2.imread(f'tmp/{args.wandb_project}_gt_{experiment_type}.png')

    transform = A.Compose([A.Resize(128, 128), A.Normalize(), ToTensorV2()])
    images = transform(image=images)['image']
    images_baseline = transform(image=images_baseline)['image']

    pos_embedding = generate_sinusoidal_embeddings(images.shape[1], images.shape[2], pe_channels)
    images = torch.cat([images, pos_embedding], dim=0)
    images_baseline = torch.cat([images_baseline, pos_embedding], dim=0)

    images = images.unsqueeze(0).to(DEVICE)
    images_baseline = images_baseline.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        input = torch.cat([images, images_baseline], dim=1)
        predictions = model(input)
        predictions = predictions.cpu().numpy()

    # print(f'Predictions: {predictions}')
    return predictions