#!/usr/bin/env python3
"""
Convert D-FINE model with two inputs to single input model for DeepStream compatibility
Based on deim/supervisely_integration/export.py and tools/deployment/export_onnx.py
"""

import os
import sys
import torch
import torch.nn as nn
import argparse
import subprocess
from pathlib import Path

# Add DEIM paths correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
deim_root = os.path.join(current_dir, '..', 'deim')
workspace_root = os.path.join(current_dir, '..')

sys.path.insert(0, deim_root)
sys.path.insert(0, workspace_root)

def load_deim_model(checkpoint_path: str, config_path: str):
    """Load DEIM model like in original code"""
    from engine.core import YAMLConfig
    
    # Load configuration
    cfg = YAMLConfig(config_path)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model weights
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
        
    # Load weights into model
    cfg.model.load_state_dict(state_dict)
    print("Model weights loaded successfully")
    
    return cfg

class SingleInputDFINE(nn.Module):
    """
    Wrapper for D-FINE model with single input
    Fixes orig_target_sizes = [640, 640] for DeepStream compatibility
    """
    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
        
        # Fixed sizes for DeepStream (640x640 input)
        self.register_buffer('orig_target_sizes', torch.tensor([[640, 640]], dtype=torch.int32))
        
    def forward(self, images):
        """
        Forward pass with single input
        Args:
            images: [N, 3, 640, 640] - input images
        Returns:
            outputs: dict with labels, boxes, scores
        """
        # Dynamically adapt batch size
        batch_size = images.shape[0]
        target_sizes = self.orig_target_sizes.expand(batch_size, -1)
        
        # Main inference
        outputs = self.model(images)
        
        # Post-processing with fixed sizes
        outputs = self.postprocessor(outputs, target_sizes)
        
        return outputs

def export_to_onnx(model, output_path: str):
    """Export to ONNX with single input"""
    model.eval()
    
    # Test data - only images
    dummy_input = torch.rand(1, 3, 640, 640)
    
    # Check that model works
    with torch.no_grad():
        _ = model(dummy_input)
    print("Model forward pass test: OK")
    
    # Dynamic axes only for batch
    dynamic_axes = {
        'images': {0: 'batch_size'},
        'labels': {0: 'batch_size'},
        'boxes': {0: 'batch_size'},  
        'scores': {0: 'batch_size'}
    }
    
    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,  # Single input only!
        output_path,
        input_names=['images'],  # Single input
        output_names=['labels', 'boxes', 'scores'],
        dynamic_axes=dynamic_axes,
        opset_version=16,
        verbose=True,
        do_constant_folding=True,
    )
    
    # Validate ONNX model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model validation: PASSED")
    except ImportError:
        print("⚠ ONNX not available for validation, but export completed")
    
    return output_path

def export_to_tensorrt(onnx_path: str, output_path: str, fp16: bool = True):
    """Export to TensorRT Engine"""
    print(f"Converting ONNX to TensorRT: {onnx_path} -> {output_path}")
    
    cmd = [
        'trtexec',
        f'--onnx={onnx_path}',
        f'--saveEngine={output_path}',
        '--verbose',
        '--minShapes=images:1x3x640x640',
        '--optShapes=images:1x3x640x640', 
        '--maxShapes=images:4x3x640x640',  # Support batches up to 4
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
    parser = argparse.ArgumentParser(description="Convert D-FINE model to single input for DeepStream")
    parser.add_argument("--pth_path", type=str, default="../models/best.pth", 
                       help="Path to PyTorch checkpoint")
    parser.add_argument("--config_path", type=str, default="../models/model_config.yml",
                       help="Path to model config")
    parser.add_argument("--output_dir", type=str, default="../models",
                       help="Output directory")
    parser.add_argument("--model_name", type=str, default="best_new_output",
                       help="Output model name (without extension)")
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Use FP16 for TensorRT")
    parser.add_argument("--convert_to", type=str, choices=['onnx', 'tensorrt', 'both'], 
                       default='both', help="Conversion target")
    
    args = parser.parse_args()
    
    # Check input files
    if not os.path.exists(args.pth_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.pth_path}")
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config not found: {args.config_path}")
        
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== DEIM D-FINE Single Input Converter ===")
    print(f"Input checkpoint: {args.pth_path}")
    print(f"Input config: {args.config_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model name: {args.model_name}")
    
    # Load original model
    print("\n1. Loading DEIM model...")
    cfg = load_deim_model(args.pth_path, args.config_path)
    
    # Create single-input model
    print("\n2. Creating single-input wrapper...")
    single_input_model = SingleInputDFINE(cfg)
    
    # Output paths
    onnx_path = os.path.join(args.output_dir, f"{args.model_name}.onnx")
    engine_path = os.path.join(args.output_dir, f"{args.model_name}.engine")
    
    # Convert to ONNX
    if args.convert_to in ['onnx', 'both']:
        print("\n3. Exporting to ONNX...")
        export_to_onnx(single_input_model, onnx_path)
        print(f"✓ ONNX saved: {onnx_path}")
    
    # Convert to TensorRT  
    if args.convert_to in ['tensorrt', 'both']:
        if args.convert_to == 'tensorrt' and not os.path.exists(onnx_path):
            print("\n3. Exporting to ONNX (required for TensorRT)...")
            export_to_onnx(single_input_model, onnx_path)
            
        print("\n4. Converting to TensorRT...")
        export_to_tensorrt(onnx_path, engine_path, args.fp16)
        print(f"✓ TensorRT saved: {engine_path}")
    
    print(f"\n=== SUCCESS ===")
    print(f"Single-input model ready for DeepStream!")
    print(f"Key changes:")
    print(f"  - Removed orig_target_sizes input")
    print(f"  - Fixed target size to 640x640")
    print(f"  - Compatible with DeepStream nvinfer")
    
    if args.convert_to in ['tensorrt', 'both']:
        print(f"\nTo use with DeepStream:")
        print(f"  1. Update config: model-engine-file=../../models/{args.model_name}.engine")
        print(f"  2. Set: infer-dims=3;640;640")
        print(f"  3. Input video should be resized to 640x640")

if __name__ == "__main__":
    main()