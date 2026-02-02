"""
Training script for CIFAR-10 CNN model.
Includes data augmentation, optimizer, LR scheduler, and checkpointing.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn import create_model
from data.dataset import get_cifar10_dataloaders


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/(len(pbar)+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, test_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Validation')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/(len(pbar)+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description='Train CIFAR-10 CNN')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step'], help='LR scheduler type')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Data loaders
    print('Loading CIFAR-10 dataset...')
    train_loader, test_loader = get_cifar10_dataloaders(
        batch_size=args.batch_size,
        num_workers=2,
        data_dir=args.data_dir
    )
    
    # Model
    print('Creating model...')
    model = create_model(num_classes=10, device=device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0.0
    
    if args.resume:
        print(f'Loading checkpoint from {args.resume}...')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        print(f'Resumed from epoch {start_epoch}, best accuracy: {best_acc:.2f}%')
    
    # Training loop
    print(f'Starting training for {args.epochs} epochs...')
    for epoch in range(start_epoch, args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        print(f'Learning rate: {scheduler.get_last_lr()[0]:.6f}')
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_acc': train_acc,
            'val_acc': val_acc,
            'best_acc': best_acc
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'latest.pth'))
        
        # Save best checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint['best_acc'] = best_acc
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'best.pth'))
            print(f'New best accuracy: {best_acc:.2f}%')
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    print(f'\nTraining completed! Best accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()

