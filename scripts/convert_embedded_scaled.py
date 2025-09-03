#!/usr/bin/env python3
"""
Convert D-FINE model with embedded per-channel scaling for DeepStream
Fixes the channel normalization mismatch between PyTorch and DeepStream
"""

import os
import sys
import torch
import torch.nn as nn
import argparse
import subprocess
from pathlib import Path

# Add DEIM paths
current_dir = os.path.dirname(os.path.abspath(__file__))
deim_root = os.path.join(current_dir, '..', 'deim')
workspace_root = os.path.join(current_dir, '..')

sys.path.insert(0, deim_root)
sys.path.insert(0, workspace_root)


def load_deim_model(checkpoint_path: str, config_path: str):
    """Load DEIM model"""
    from engine.core import YAMLConfig
    
    cfg = YAMLConfig(config_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model', checkpoint)
    else:
        state_dict = checkpoint
        
    cfg.model.load_state_dict(state_dict)
    print("Model weights loaded successfully")
    return cfg


class SingleInputDFINEEmbeddedScaling(nn.Module):
    """
    D-FINE wrapper with proper per-channel normalization embedded in the model.
    DeepStream gives us BGR 0..255, we need to convert to PyTorch format internally.
    """
    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
        
        # Fixed target sizes for DeepStream
        self.register_buffer('orig_target_sizes', torch.tensor([[640, 640]], dtype=torch.int32))
        
        # PyTorch ImageNet normalization parameters
        # mean = [0.485, 0.456, 0.406] for RGB
        # std = [0.229, 0.224, 0.225] for RGB
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def forward(self, images):
        """
        Forward pass with embedded preprocessing.
        
        Args:
            images: [N, 3, 640, 640] - DeepStream input (BGR, 0..255)
        
        Returns:
            dict with 'labels', 'boxes', 'scores' (all float32)
        """
        # Convert to float32 if needed
        images = images.float()
        
        # BGR -> RGB channel swap
        images = images[:, [2, 1, 0], :, :]
        
        # Normalize to 0..1
        images = images / 255.0
        
        # Apply per-channel normalization (ImageNet statistics)
        images = (images - self.mean) / self.std
        
        # Forward through model
        batch_size = images.shape[0]
        target_sizes = self.orig_target_sizes.expand(batch_size, -1)
        
        outputs = self.model(images)
        labels, boxes, scores = self.postprocessor(outputs, target_sizes)
        
        # Ensure float32 for TensorRT compatibility
        labels = labels.to(torch.float32)
        boxes = boxes.to(torch.float32)
        scores = scores.to(torch.float32)
        
        return {
            "labels": labels,
            "boxes": boxes, 
            "scores": scores
        }


def export_to_onnx(model, output_path: str):
    """Export to ONNX with embedded scaling"""
    model.eval()
    
    # Test with dummy input (simulating DeepStream BGR 0..255)
    dummy_input = torch.randint(0, 256, (1, 3, 640, 640), dtype=torch.float32)
    
    # Verify model works
    with torch.no_grad():
        test_output = model(dummy_input)
    print("Model forward pass test: OK")
    print(f"Output shapes: labels={test_output['labels'].shape}, "
          f"boxes={test_output['boxes'].shape}, scores={test_output['scores'].shape}")
    
    # Export to ONNX
    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['images'],
        output_names=['labels', 'boxes', 'scores'],
        dynamic_axes={
            'images': {0: 'batch_size'},
            'labels': {0: 'batch_size'},
            'boxes': {0: 'batch_size'},
            'scores': {0: 'batch_size'}
        },
        opset_version=16,
        verbose=False,
        do_constant_folding=True,
    )
    
    # Validate ONNX
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX validation: PASSED")
    except ImportError:
        print("⚠ ONNX not available for validation")
    
    return output_path


def export_to_tensorrt(onnx_path: str, output_path: str, fp16: bool = True):
    """Export ONNX to TensorRT"""
    print(f"Converting to TensorRT: {onnx_path} -> {output_path}")
    
    cmd = [
        'trtexec',
        f'--onnx={onnx_path}',
        f'--saveEngine={output_path}',
        '--verbose',
        '--minShapes=images:1x3x640x640',
        '--optShapes=images:1x3x640x640',
        '--maxShapes=images:4x3x640x640',
        '--workspace=4096'
    ]
    
    if fp16:
        cmd.append('--fp16')
        
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ TensorRT conversion: SUCCESS")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"✗ TensorRT conversion failed:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Convert D-FINE model with embedded per-channel scaling")
    parser.add_argument("--pth_path", type=str, default="../models/best.pth",
                       help="Path to PyTorch checkpoint")
    parser.add_argument("--config_path", type=str, default="../models/model_config.yml", 
                       help="Path to model config")
    parser.add_argument("--output_dir", type=str, default="../models",
                       help="Output directory")
    parser.add_argument("--model_name", type=str, default="best_embedded_scaling",
                       help="Output model name (without extension)")
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Use FP16 for TensorRT")
    parser.add_argument("--convert_to", type=str, choices=['onnx', 'tensorrt', 'both'],
                       default='both', help="Conversion target")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.pth_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.pth_path}")
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config not found: {args.config_path}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== D-FINE Embedded Scaling Converter ===")
    print(f"Input checkpoint: {args.pth_path}")
    print(f"Input config: {args.config_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model name: {args.model_name}")
    
    
    # Load model
    print("\n1. Loading DEIM model...")
    cfg = load_deim_model(args.pth_path, args.config_path)
    
    # Create wrapper with embedded scaling
    print("\n2. Creating wrapper with embedded per-channel scaling...")
    model = SingleInputDFINEEmbeddedScaling(cfg)
    
    # Define output paths
    onnx_path = os.path.join(args.output_dir, f"{args.model_name}.onnx")
    engine_path = os.path.join(args.output_dir, f"{args.model_name}.engine")
    
    # Export to ONNX
    if args.convert_to in ['onnx', 'both']:
        print("\n3. Exporting to ONNX...")
        export_to_onnx(model, onnx_path)
        print(f"✓ ONNX saved: {onnx_path}")
    
    # Export to TensorRT
    if args.convert_to in ['tensorrt', 'both']:
        if args.convert_to == 'tensorrt' and not os.path.exists(onnx_path):
            print("\n3. Exporting to ONNX (required for TensorRT)...")
            export_to_onnx(model, onnx_path)
            
        print("\n4. Converting to TensorRT...")
        export_to_tensorrt(onnx_path, engine_path, args.fp16)
        print(f"✓ TensorRT saved: {engine_path}")
    
    print('Conversion completed successfully.')
    


if __name__ == "__main__":
    main()