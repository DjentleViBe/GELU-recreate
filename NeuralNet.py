import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

class ZCATestTransform:
    def __init__(self, mean, W_zca, epsilon=1e-8):
        self.mean = mean
        self.W_zca = W_zca
        self.epsilon = epsilon

    def __call__(self, img):
        # Convert to numpy and normalize
        x = np.array(img, dtype=np.float32).flatten() / 255.0
        
        # Apply ZCA whitening
        x_centered = x - self.mean
        x_zca = x_centered @ self.W_zca
        
        # Reshape and convert to tensor
        x_zca = x_zca.reshape(3, 32, 32)
        x_zca = (x_zca - x_zca.mean()) / (x_zca.std() + self.epsilon)
        
        return torch.tensor(x_zca, dtype=torch.float32)

class ZCADataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, add_noise_sigma=0.15, training=True):
        self.data = data
        self.labels = labels
        self.sigma = add_noise_sigma
        self.training = training  # Important: noise only during training

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        
        # Add Gaussian noise only during training
        if self.training and self.sigma > 0:
            x = x + torch.randn_like(x) * self.sigma
            
        return x, y
    
class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.15):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.sigma
            return x + noise
        return x
    
# Learning rate scheduler: linear decay after 100 epochs
def adjust_lr(optimizer, epoch, total_epochs=200):
    if epoch >= 100:
        lr = 1e-3 * (total_epochs - epoch) / 100
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
# -------------------------
# 9-layer CNN
# -------------------------
class CIFAR10CNN(nn.Module):
    def __init__(self, activation=F.gelu):
        super().__init__()
        self.activation = activation
        # Block 1
        self.conv1 = nn.Conv2d(3, 96, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.5)
        
        # Block 2
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(192)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(192)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(192)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.5)
        
        # Block 3
        self.conv7 = nn.Conv2d(192, 192, 3)  # 3x3 conv without padding, 8x8 -> 6x6
        self.bn7 = nn.BatchNorm2d(192)
        self.conv8 = nn.Conv2d(192, 192, 1)
        self.bn8 = nn.BatchNorm2d(192)
        self.conv9 = nn.Conv2d(192, 192, 1)
        self.bn9 = nn.BatchNorm2d(192)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(192, 10)
        
    def forward(self, x):
        # Block 1
        act = self.activation  # shortcut

        # Block 1
        x = act(self.bn1(self.conv1(x)))
        x = act(self.bn2(self.conv2(x)))
        x = act(self.bn3(self.conv3(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = act(self.bn4(self.conv4(x)))
        x = act(self.bn5(self.conv5(x)))
        x = act(self.bn6(self.conv6(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Block 3
        x = act(self.bn7(self.conv7(x)))
        x = act(self.bn8(self.conv8(x)))
        x = act(self.bn9(self.conv9(x)))

        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def compute_zca_statistics(data_root='./data'):
    """Compute ZCA statistics from CIFAR-10 training data"""
    print("Computing ZCA statistics...")
    
    # Load raw training data
    train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, 
                                   transform=transforms.ToTensor())
    
    # Convert to numpy array
    X_train = np.array([np.array(img) for img, _ in train_dataset], dtype=np.float32)
    labels = np.array([label for _, label in train_dataset])
    
    # Reshape and normalize
    X_train = X_train.reshape(len(X_train), -1) / 255.0  # (50000, 3072)
    
    # Compute mean and ZCA matrix
    X_mean = np.mean(X_train, axis=0)
    X_centered = X_train - X_mean
    
    # Covariance and SVD
    sigma = np.cov(X_centered, rowvar=False)  # (3072, 3072)
    U, S, _ = np.linalg.svd(sigma)
    epsilon = 1e-5
    W_zca = U @ np.diag(1.0 / np.sqrt(S + epsilon)) @ U.T
    
    # Apply ZCA to training data
    X_zca = X_centered @ W_zca
    X_zca = X_zca.reshape(-1, 3, 32, 32)  # (50000, 3, 32, 32)
    mean = X_zca_tensor.mean(dim=(0,2,3), keepdim=True)
    std = X_zca_tensor.std(dim=(0,2,3), keepdim=True)

    # Standardize
    X_zca_tensor = (X_zca_tensor - mean) / (std + 1e-5)
    X_zca_tensor = torch.tensor(X_zca, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return X_mean, W_zca, X_zca_tensor, labels_tensor

def prepare_datasets(X_mean, W_zca, X_zca_tensor, labels_tensor, val_ratio=0.1):
    """Prepare training, validation and test datasets"""
    
    # Create base dataset with all ZCA-transformed training data
    full_train_dataset = ZCADataset(X_zca_tensor, labels_tensor, add_noise_sigma=0.15, training=True)
    
    # Split into train/validation
    total_size = len(full_train_dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    # For validation, we need to disable noise but keep ZCA transformation
    # Extract the actual data from the subset
    val_indices = val_dataset.indices
    X_val_data = X_zca_tensor[val_indices]
    val_labels = labels_tensor[val_indices]
    
    # Create validation dataset without noise
    val_dataset = ZCADataset(X_val_data, val_labels, add_noise_sigma=0.0, training=False)
    
    # Test dataset with ZCA transform
    transform_test = ZCATestTransform(X_mean, W_zca)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, 
                                  transform=transform_test)
    
    return train_dataset, val_dataset, test_dataset