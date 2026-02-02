"""
CNN model for CIFAR-10 classification.
Small ResNet-like architecture with residual blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CIFAR10CNN(nn.Module):
    """
    Improved ResNet-like CNN for CIFAR-10 classification.
    
    Architecture:
    - Initial conv layer
    - 4 layers with more residual blocks
    - Dropout for regularization
    - Global average pooling
    - Fully connected layer for classification
    """
    
    def __init__(self, num_classes=10, dropout=0.3):
        super(CIFAR10CNN, self).__init__()
        
        # Initial convolutional layer with wider channels
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks - deeper architecture
        # Layer 1: 64 channels, 3 blocks
        self.layer1 = self._make_layer(64, 64, 1, num_blocks=3)
        # Layer 2: 64->128 channels, 3 blocks
        self.layer2 = self._make_layer(64, 128, 2, num_blocks=3)
        # Layer 3: 128->256 channels, 3 blocks
        self.layer3 = self._make_layer(128, 256, 2, num_blocks=3)
        # Layer 4: 256->512 channels, 2 blocks
        self.layer4 = self._make_layer(256, 512, 2, num_blocks=2)
        
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_layer(self, in_channels, out_channels, stride, num_blocks=2):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial conv
        out = F.relu(self.bn1(self.conv1(x)))
        
        # Residual blocks
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # Global average pooling
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        
        # Dropout
        out = self.dropout(out)
        
        # Classification
        out = self.fc(out)
        return out


def create_model(num_classes=10, device='cpu', dropout=0.3):
    """Create and initialize the CIFAR-10 CNN model."""
    model = CIFAR10CNN(num_classes=num_classes, dropout=dropout)
    model = model.to(device)
    return model
