# Import necessary libraries
import os
import random
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp

from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, models, transforms

from AACN_Model import attention_augmented_resnet18, attention_augmented_inceptionv3,attention_augmented_vgg

# Set the environment variable for PyTorch memory management
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# Check if GPU is available and set the device accordingly
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Define data transformations for training and testing datasets
data_transforms = {
    'Train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'Test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Define preprocessing transformation
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Set parameters for data loading and processing
batch_size = 128
validation_split = .2
shuffle_dataset = True
random_seed = 43
main_root_directory = "/home/felhassan/Desktop/Project"
image_modal_directory = main_root_directory + "/500x500"
results_folder = "./results"
os.makedirs(results_folder, exist_ok=True)

# Function to collect files from directories
def collect_files_from_directories(main_root_directory):
    collected_files = []
    for root, dirs, files in os.walk(main_root_directory):
        for file in files:
            file_name = os.path.splitext(file)[0]
            file_path = os.path.join(root, file)
            collected_files.append((file_name, file_path))
    df = pd.DataFrame(collected_files, columns=['Filename', 'FilePath'])
    return df

# Function to display images with text
def display_images_with_text(images):
    plt.figure(figsize=(15, 15))
    plt.suptitle("Sample Images", fontsize=16)
    for i in range(min(9, len(images))):
        img_name = images["Filename"][i]
        img_path = images["FilePath"][i]
        img = Image.open(img_path)
        row = i // 3
        col = i % 3
        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.text(0, -10, f"{img_name}", color='white', fontsize=12, weight='bold', ha='left', va='bottom', bbox=dict(facecolor='black', alpha=0.7))
    plt.show()

# Collect and display images
images_df = collect_files_from_directories(image_modal_directory)
display_images_with_text(images_df)

# Define bounding box coordinates
box_coords = ((220, 90), (30, 30))

# Define custom dataset class with bounding box extraction
class ForecastingDataset(Dataset):
    def __init__(self, root_dir, box_coords, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.box_coords = box_coords
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if not f.startswith(".")]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        xmin, ymin = self.box_coords[0]
        width, height = self.box_coords[1]
        xmax, ymax = xmin + width, ymin + height
        img = img.crop((xmin, ymin, xmax, ymax))  # Crop the image to the bounding box
        if self.transform:
            img = self.transform(img)
        return img

# Function to create data loaders
def create_data_loaders(dataset, batch_size, train_val_test_split=(0.6, 0.2, 0.2)):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split1 = int(np.floor(train_val_test_split[0] * dataset_size))
    split2 = int(np.floor((train_val_test_split[0] + train_val_test_split[1]) * dataset_size))
    train_indices, val_indices, test_indices = indices[:split1], indices[split1:split2], indices[split2:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, val_loader, test_loader

# Initialize dataset
forecasting_dataset = ForecastingDataset(root_dir=image_modal_directory, box_coords=box_coords, transform=preprocess)

# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders(forecasting_dataset, batch_size, train_val_test_split=(0.6, 0.2, 0.2))

# Define the pretrained models
pretrained_models = {
    'InceptionV3': models.inception_v3(pretrained=True).to(device),
    'ResNet152': models.resnet152(pretrained=True).to(device),
    'VGG19': models.vgg19(pretrained=True).to(device),
    'ViT': ViT(
        image_size=384,
        patch_size=16,
        num_classes=1000,  # Dummy value, actual number of classes can be changed
        dim=2048,
        depth=12,
        heads=32,
        mlp_dim=4096,
        dropout=0.1,
        emb_dropout=0.1
    ).to(device),
    "AttentionAugmentedInceptionV3": attention_augmented_inceptionv3(attention=True).to(device),
    'AttentionAugmentedVGG19': attention_augmented_vgg('VGG19', num_classes=1000).to(device),
    "AttentionAugmentedResNet18": attention_augmented_resnet18(num_classes=1000, attention=[False, True, True, True], num_heads=8).to(device),
}

# Modify the models for feature extraction
def modify_model_for_feature_extraction(model, model_name):
    if 'resnet' in model_name.lower() or 'vgg' in model_name.lower():
        modules = list(model.children())[:-2]  # Remove the last convolutional block
    elif 'inception' in model_name.lower():
        modules = list(model.children())[:-1]  # Remove the last block
    elif 'vit' in model_name.lower():
        return model  # Use the whole model for ViT
    else:
        raise ValueError(f"Model {model_name} not supported for feature extraction")
    return nn.Sequential(*modules).to(device)

# Function to extract features from the dataset using the modified models
def extract_features(dataloader, model, device):
    model.eval()
    features = []
    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            features.append(outputs.cpu().numpy())
    features = np.concatenate(features)
    return features

# Extract and save features for each model
for model_name, model in pretrained_models.items():
    modified_model = modify_model_for_feature_extraction(model, model_name)
    print(f"Extracting features using {model_name}...")
    train_features = extract_features(train_loader, modified_model, device)
    val_features = extract_features(val_loader, modified_model, device)
    test_features = extract_features(test_loader, modified_model, device)
    
    np.save(os.path.join(results_folder, f'{model_name}_train_features.npy'), train_features)
    np.save(os.path.join(results_folder, f'{model_name}_val_features.npy'), val_features)
    np.save(os.path.join(results_folder, f'{model_name}_test_features.npy'), test_features)

print("Feature extraction completed for all models.")