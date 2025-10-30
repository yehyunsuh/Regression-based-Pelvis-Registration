import cv2
import torch
import timm
import numpy as np
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
import albumentations as A

class RegressionModel(nn.Module):
    def __init__(self, model_name="resnet18", num_outputs=6, pretrained=False):
        super(RegressionModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.fc = nn.Linear(self.model.num_features, num_outputs)

    def forward(self, x):
        features = self.model(x)
        output = self.fc(features)
        return output

def load_model(weight_path, model_name="resnet18", num_outputs=6, DEVICE="cpu"):
    """Load the trained model weights."""
    model = RegressionModel(model_name=model_name, num_outputs=num_outputs)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model

def generate_random_images(num_images=5, img_size=(128, 128, 3)):
    """Generate random images with pixel values in the range [0, 255]."""
    random_images = np.random.randint(0, 256, (num_images, *img_size), dtype=np.uint8)
    return random_images

def test_model(args, images, DEVICE, experiment_type, comparison_type = None):
    """Run inference on randomly generated images and visualize predictions."""
    model_weights_path = args.weight_path
    model = load_model(model_weights_path, model_name=args.model_name, DEVICE=DEVICE)
    images = cv2.imread(f'tmp/{args.wandb_project}_target_{experiment_type}.png')
    transform = A.Compose([A.Resize(128, 128), A.Normalize(), ToTensorV2()])
    images = transform(image=images)['image']
    images = images.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        predictions = model(images)
        predictions = predictions.cpu().numpy()

    # print(f'Predictions: {predictions}')
    return predictions