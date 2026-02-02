"""
Export trained model to TorchScript or ONNX format for efficient deployment.
"""

import os
import argparse
import torch

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn import create_model


def export_torchscript(model, checkpoint_path, output_path, device):
    """Export model to TorchScript format."""
    print(f'Loading checkpoint from {checkpoint_path}...')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    
    # Export to TorchScript
    print('Exporting to TorchScript...')
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(output_path)
    print(f'TorchScript model saved to {output_path}')
    
    # Verify the exported model
    print('Verifying exported model...')
    loaded_model = torch.jit.load(output_path)
    with torch.no_grad():
        output = loaded_model(dummy_input)
    print(f'Verification successful! Output shape: {output.shape}')


def export_onnx(model, checkpoint_path, output_path, device):
    """Export model to ONNX format."""
    print(f'Loading checkpoint from {checkpoint_path}...')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    
    # Export to ONNX
    print('Exporting to ONNX...')
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f'ONNX model saved to {output_path}')
    
    # Verify the exported model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print('ONNX model verification successful!')
    except ImportError:
        print('ONNX package not available for verification')


def main():
    parser = argparse.ArgumentParser(description='Export CIFAR-10 CNN model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--format', type=str, default='torchscript', choices=['torchscript', 'onnx', 'both'],
                       help='Export format')
    parser.add_argument('--output-dir', type=str, default='./models', help='Output directory')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Model
    print('Creating model...')
    model = create_model(num_classes=10, device=device)
    
    # Export based on format
    if args.format in ['torchscript', 'both']:
        torchscript_path = os.path.join(args.output_dir, 'cifar10_cnn.pt')
        export_torchscript(model, args.checkpoint, torchscript_path, device)
    
    if args.format in ['onnx', 'both']:
        onnx_path = os.path.join(args.output_dir, 'cifar10_cnn.onnx')
        export_onnx(model, args.checkpoint, onnx_path, device)
    
    print('\nModel export completed!')


if __name__ == '__main__':
    main()

