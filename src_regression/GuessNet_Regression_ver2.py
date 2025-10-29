import os
import re
import cv2
import timm
import torch
import wandb
import argparse
import numpy as np
import torch.nn as nn
import albumentations as A
import torch.optim as optim
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
from pathlib import Path
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, image_path_array, transform=None):
        self.image_paths = image_path_array
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        p = Path(image_path)
        parts = p.stem.split('_')
        parts[-6:] = ['0'] * 6
        new_name = '_'.join(parts) + p.suffix
        image_baseline_path = str(p.parent / new_name)
        image_baseline = cv2.imread(image_baseline_path)
        
        target_str_list = image_path.split('/')[-1].split('.')[0].split('_')[-6:]
        target = torch.tensor([float(val) for val in target_str_list], dtype=torch.float32)

        if self.transform:
            image = self.transform(image=image)["image"]
            image_baseline = self.transform(image=image_baseline)["image"]

        return image, image_baseline, target, image_path


class RegressionModel(nn.Module):
    def __init__(self, model_name="resnet18", num_outputs=6, pretrained=True):
        super(RegressionModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Modify the first conv layer to accept 6 channels instead of 3
        old_conv = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            in_channels=6,  # New input channels
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )

        # Initialize the new conv layer with pretrained weights (averaging across channels)
        with torch.no_grad():
            self.model.conv1.weight[:, :3] = old_conv.weight  # Copy weights for RGB channels
            self.model.conv1.weight[:, 3:] = old_conv.weight  # Duplicate for extra channels

        self.fc = nn.Linear(self.model.num_features, num_outputs)

    def forward(self, x):
        features = self.model(x)
        output = self.fc(features)
        return output
    

def train_model(args, model, train_loader, val_loader, criterion, optimizer, DEVICE, num_epochs=10, save_path="GuessNet_Regression.pth"):
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_abs_diffs = torch.zeros(6, device=DEVICE)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        for images, images_baseline, targets, _ in tqdm(train_loader):
            images, images_baseline, targets = images.to(DEVICE), images_baseline.to(DEVICE), targets.to(DEVICE)

            # concatenate images and images_baseline
            input = torch.cat([images, images_baseline], dim=1)

            optimizer.zero_grad()
            outputs = model(input)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_abs_diffs += torch.sum(torch.abs(outputs - targets), dim=0)

        avg_train_loss = running_loss / len(train_loader)
        avg_train_abs_diffs = (train_abs_diffs / len(train_loader.dataset)).detach().cpu().numpy()
        val_loss, val_abs_diffs = evaluate_model(model, val_loader, criterion, DEVICE)
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

        if args.wandb:
            wandb.log({
                "train_loss": avg_train_loss,
                "train rot1 diff": avg_train_abs_diffs[0],
                "train rot2 diff": avg_train_abs_diffs[1],
                "train rot3 diff": avg_train_abs_diffs[2],
                "train trans1 diff": avg_train_abs_diffs[3],
                "train trans2 diff": avg_train_abs_diffs[4],
                "train trans3 diff": avg_train_abs_diffs[5],
                "val_loss": val_loss,
                "val rot1 diff": val_abs_diffs[0],
                "val rot2 diff": val_abs_diffs[1],
                "val rot3 diff": val_abs_diffs[2],
                "val trans1 diff": val_abs_diffs[3],
                "val trans2 diff": val_abs_diffs[4],
                "val trans3 diff": val_abs_diffs[5]
            })

        # Save the model if the validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved at epoch {epoch + 1} with Validation Loss: {val_loss:.4f}")

        # Visualization every 5 epochs
        if (epoch + 1) % 100 == 0 or epoch == 0:
            os.makedirs(f"visualization/{args.vis_dir}/{epoch}", exist_ok=True)
            visualize_predictions(args, epoch, model, val_loader, DEVICE)

    print("Training complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")


def visualize_predictions(args, epoch, model, val_loader, DEVICE):
    model.eval()
    with torch.no_grad():
        images, images_baseline, targets, image_paths = next(iter(val_loader))
        images, images_baseline, targets = images.to(DEVICE), images_baseline.to(DEVICE), targets.to(DEVICE)

        input = torch.cat([images, images_baseline], dim=1)
        outputs = model(input)

        # Convert tensors to numpy for visualization
        images = images.cpu().numpy().transpose(0, 2, 3, 1)  # Convert to HxWxC for visualization
        targets = targets.cpu().numpy()
        outputs = outputs.cpu().numpy()

        for i in range(min(args.num_outputs, len(images))):
            img = (images[i] * 255).astype(np.uint8)  # Convert normalized image back to original scale
            
            # Prepare titles with formatted numbers
            ground_truth = ", ".join([f"{x:.1f}" for x in targets[i]])
            predicted = ", ".join([f"{x:.1f}" for x in outputs[i]])

            plt.figure(figsize=(4, 4))
            img_gray = np.mean(images[i], axis=-1)
            plt.imshow(img_gray, cmap='gray')
            plt.title(f"GT: {ground_truth}\nPred: {predicted}")
            plt.axis('off')
            plt.savefig(f'visualization/{args.vis_dir}/{epoch}/image_{i}.png')
            plt.close()


def evaluate_model(model, val_loader, criterion, DEVICE):
    model.eval()
    total_loss = 0.0
    abs_diffs = torch.zeros(6, device=DEVICE)

    with torch.no_grad():
        for images, images_baseline, targets, _ in val_loader:
            images, images_baseline, targets = images.to(DEVICE), images_baseline.to(DEVICE), targets.to(DEVICE)
            input = torch.cat([images, images_baseline], dim=1)
            outputs = model(input)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Calculate absolute differences for each parameter
            abs_diffs += torch.sum(torch.abs(outputs - targets), dim=0)

    avg_abs_diffs = (abs_diffs / len(val_loader.dataset)).cpu().numpy()
    return total_loss / len(val_loader), avg_abs_diffs



def main(args, DEVICE):
    cadaver_id_list = [Path(p).stem for p in glob(f'{args.CT_dir}/*')]

    train_image_path_list, test_image_path_list = [], []
    for cadaver_id in cadaver_id_list:
        image_path = glob(f'{args.img_dir}/{cadaver_id}/*.png')
        num_of_images = len(image_path)
        split_index = int(num_of_images * 0.8)
        train_image_path_list.extend(image_path[:split_index])
        test_image_path_list.extend(image_path[split_index:])
    print(f'Train images: {len(train_image_path_list)}, Test images: {len(test_image_path_list)}')

    if args.augmentation:
        # augmentation 1: random rotation from -15 to 15 for p = 0.3
        # augmentation 2: pixel dropout for 0.05 for p = 0.3
        # augmentation 3: grid distortion limit from -0.1 to 0.1 for p = 0.3
        transform = A.Compose([
            A.Resize(128, 128), 

            A.Rotate(limit=15, p=0.5),
            A.PixelDropout(dropout_prob=0.05, p=0.5),
            A.GridDistortion(distort_limit=(-0.1, 0.1), p=0.5),

            A.Normalize(), 
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(128, 128), 
            A.Normalize(), 
            ToTensorV2()
        ])
    train_dataset = CustomDataset(np.array(train_image_path_list), transform=transform)
    test_dataset = CustomDataset(np.array(test_image_path_list), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    model = RegressionModel(model_name=args.model, num_outputs=6).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_model(args, model, train_loader, test_loader, criterion, optimizer, DEVICE, num_epochs=args.num_epochs, save_path=f'{args.model_dir}/{args.model_name}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and Path
    parser.add_argument("--CT_dir", type=str, default="../data/CT", help="Path to the data directory")
    parser.add_argument("--img_dir", type=str, default="../data_projected_baseline", help="Path to the data directory")
    parser.add_argument("--vis_dir", type=str, default="GuessNet_Regression_baseline_ver2", help="Path to the visualization directory")
    parser.add_argument("--model_dir", type=str, default="../model_weights", help="Path to save the model directory")
    parser.add_argument("--model_name", type=str, default="GuessNet_Regression_baseline_ver2.pth", help="Path to save the model")

    # Model Hyperparameters
    parser.add_argument("--augmentation", action='store_true', help="Whether to use augmentation or not")
    parser.add_argument("--model", type=str, default="resnet18", help="Name of the model to use")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size")

    # Visualization and Evaluation
    parser.add_argument("--num_outputs", type=int, default=25, help="Number of output units")

    # Wandb
    parser.add_argument('--wandb', action='store_true', help='whether to use wandb or not')
    parser.add_argument('--wandb_project', type=str, default="GuessNet", help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default="", help='wandb entity name')
    parser.add_argument('--wandb_name', type=str, default="baseline", help='wandb name')
    parser.add_argument('--wandb_group', type=str, default="Regression", help='wandb group name')

    args = parser.parse_args()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if args.wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name, group=args.wandb_group)

    np.random.seed(2025)
    torch.manual_seed(2025)

    main(args, DEVICE)
