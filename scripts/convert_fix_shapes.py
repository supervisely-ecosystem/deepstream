#!/usr/bin/env python3
"""
Fixed D-FINE converter with static shapes for DeepStream
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

class SingleInputDFINE(nn.Module):
    """Wrapper for D-FINE with single input"""
    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
        
        # Fixed target sizes for DeepStream
        self.register_buffer('orig_target_sizes', torch.tensor([[640, 640]], dtype=torch.int32))
        
    def forward(self, images):
        """Forward with single input"""
        batch_size = images.shape[0]
        target_sizes = self.orig_target_sizes.expand(batch_size, -1)
        
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, target_sizes)
        
        return outputs

def export_to_onnx(model, output_path: str):
    """Export to ONNX with static shapes"""
    model.eval()
    dummy_input = torch.rand(1, 3, 640, 640)
    
    with torch.no_grad():
        test_output = model(dummy_input)
    print("Model forward pass test: OK")
    
    # NO dynamic axes - fixed shapes only!
    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['images'],
        output_names=['labels', 'boxes', 'scores'],
        opset_version=16,
        verbose=False,
        do_constant_folding=True,
    )
    
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model validation: PASSED")
    except ImportError:
        print("⚠ ONNX not available for validation")
    
    return output_path

def export_to_tensorrt(onnx_path: str, output_path: str, fp16: bool = True):
    """Export to TensorRT - простейший вариант"""
    print(f"Converting ONNX to TensorRT: {onnx_path} -> {output_path}")
    
    cmd = [
        'trtexec',
        f'--onnx={onnx_path}',
        f'--saveEngine={output_path}',
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth_path", type=str, default="../models/best.pth")
    parser.add_argument("--config_path", type=str, default="../models/model_config.yml")
    parser.add_argument("--output_dir", type=str, default="../models")
    parser.add_argument("--model_name", type=str, default="best_fixed_shapes")
    parser.add_argument("--fp16", action="store_true", default=True)
    
    args = parser.parse_args()
    
    # Check inputs
    if not os.path.exists(args.pth_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.pth_path}")
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config not found: {args.config_path}")
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== D-FINE Fixed Shapes Converter ===")
    print(f"Checkpoint: {args.pth_path}")
    print(f"Config: {args.config_path}")
    print(f"Output: {args.model_name}")
    
    # Load and convert
    print("\n1. Loading model...")
    cfg = load_deim_model(args.pth_path, args.config_path)
    
    print("\n2. Creating wrapper...")
    single_input_model = SingleInputDFINE(cfg)
    
    # Paths
    onnx_path = os.path.join(args.output_dir, f"{args.model_name}.onnx")
    engine_path = os.path.join(args.output_dir, f"{args.model_name}.engine")
    
    print("\n3. Exporting to ONNX...")
    export_to_onnx(single_input_model, onnx_path)
    
    print("\n4. Converting to TensorRT...")
    export_to_tensorrt(onnx_path, engine_path, args.fp16)
    
    print(f"\n=== SUCCESS ===")
    print(f"Fixed shapes model ready!")
    print(f"Engine: {engine_path}")

if __name__ == "__main__":
    main()