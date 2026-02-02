"""
Evaluation script for CIFAR-10 CNN model.
Computes test accuracy and generates confusion matrix.
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn import create_model
from data.dataset import get_cifar10_dataloaders, CIFAR10_CLASSES


def evaluate(model, test_loader, device):
    """Evaluate the model and return predictions and targets."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    return np.array(all_preds), np.array(all_targets)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f'Confusion matrix saved to {save_path}')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate CIFAR-10 CNN')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Data loader
    print('Loading CIFAR-10 test dataset...')
    _, test_loader = get_cifar10_dataloaders(
        batch_size=args.batch_size,
        num_workers=2,
        data_dir=args.data_dir
    )
    
    # Model
    print('Creating model...')
    model = create_model(num_classes=10, device=device)
    
    # Load checkpoint
    print(f'Loading checkpoint from {args.checkpoint}...')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if 'val_acc' in checkpoint:
        print(f'Checkpoint accuracy: {checkpoint["val_acc"]:.2f}%')
    
    # Evaluate
    print('Evaluating model...')
    predictions, targets = evaluate(model, test_loader, device)
    
    # Calculate accuracy
    accuracy = 100. * np.sum(predictions == targets) / len(targets)
    print(f'\nTest Accuracy: {accuracy:.2f}%')
    
    # Classification report
    print('\nClassification Report:')
    print(classification_report(
        targets,
        predictions,
        target_names=CIFAR10_CLASSES,
        digits=4
    ))
    
    # Confusion matrix
    print('\nGenerating confusion matrix...')
    plot_confusion_matrix(
        targets,
        predictions,
        CIFAR10_CLASSES,
        save_path=os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    
    print('\nEvaluation completed!')


if __name__ == '__main__':
    main()

