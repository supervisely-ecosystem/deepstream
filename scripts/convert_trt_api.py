#!/usr/bin/env python3
"""
Convert D-FINE model using Python TensorRT API instead of trtexec
"""

import os
import sys
import torch
import torch.nn as nn
import argparse
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
    """D-FINE wrapper with proper per-channel normalization embedded"""
    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
        
        # Fixed target sizes for DeepStream
        self.register_buffer('orig_target_sizes', torch.tensor([[640, 640]], dtype=torch.int32))
        
        # PyTorch ImageNet normalization parameters (RGB order)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def forward(self, images):
        """Forward pass with embedded preprocessing"""
        # Convert to float32
        images = images.float()
        
        # BGR -> RGB channel swap
        images = images[:, [2, 1, 0], :, :]
        
        # Normalize to 0..1
        images = images / 255.0
        
        # Apply per-channel normalization
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
        
        return labels, boxes, scores


def export_to_onnx(model, output_path: str):
    """Export to ONNX"""
    model.eval()
    
    # Test with dummy input (BGR 0..255)
    dummy_input = torch.randint(0, 256, (1, 3, 640, 640), dtype=torch.float32)
    
    # Verify model works
    with torch.no_grad():
        test_output = model(dummy_input)
    
    print("Model forward pass test: OK")
    print(f"Output shapes: labels={test_output[0].shape}, "
          f"boxes={test_output[1].shape}, scores={test_output[2].shape}")
    
    # Export to ONNX
    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['images'],
        output_names=['labels', 'boxes', 'scores'],
        opset_version=14,
        verbose=False,
        do_constant_folding=True,
        export_params=True,
    )
    
    # Validate ONNX
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX validation: PASSED")
    except ImportError:
        print("ONNX not available for validation")
    except Exception as e:
        print(f"ONNX validation issue: {e}")
    
    return output_path


def export_to_tensorrt_python(onnx_path: str, output_path: str, fp16: bool = True):
    """Convert ONNX to TensorRT using Python API"""
    try:
        import tensorrt as trt
        import numpy as np
    except ImportError:
        print("ERROR: TensorRT Python API not available")
        print("Try: pip install tensorrt")
        return None

    print(f"Converting ONNX to TensorRT: {onnx_path} -> {output_path}")
    
    # Create TensorRT logger
    logger = trt.Logger(trt.Logger.WARNING)
    
    # Create builder and network
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX model
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse ONNX model")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # Configure builder
    config = builder.create_builder_config()
    config.max_workspace_size = 2 * (1 << 30)  # 2GB
    
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    # Set input shape
    profile = builder.create_optimization_profile()
    profile.set_shape('images', (1, 3, 640, 640), (1, 3, 640, 640), (4, 3, 640, 640))
    config.add_optimization_profile(profile)
    
    # Build engine
    print("Building TensorRT engine...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("ERROR: Failed to build TensorRT engine")
        return None
    
    # Save engine
    with open(output_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"TensorRT engine saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert D-FINE model using Python TensorRT API")
    parser.add_argument("--pth_path", type=str, default="../models/best.pth",
                       help="Path to PyTorch checkpoint")
    parser.add_argument("--config_path", type=str, default="../models/model_config.yml", 
                       help="Path to model config")
    parser.add_argument("--output_dir", type=str, default="../models",
                       help="Output directory")
    parser.add_argument("--model_name", type=str, default="best_python_trt",
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
    
    print("=== D-FINE Python TensorRT Converter ===")
    print(f"Input checkpoint: {args.pth_path}")
    print(f"Input config: {args.config_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model name: {args.model_name}")
    
    # Load model
    print("\n1. Loading DEIM model...")
    cfg = load_deim_model(args.pth_path, args.config_path)
    
    # Create wrapper
    print("\n2. Creating wrapper with embedded scaling...")
    model = SingleInputDFINEEmbeddedScaling(cfg)
    
    # Define output paths
    onnx_path = os.path.join(args.output_dir, f"{args.model_name}.onnx")
    engine_path = os.path.join(args.output_dir, f"{args.model_name}.engine")
    
    # Export to ONNX
    if args.convert_to in ['onnx', 'both']:
        print("\n3. Exporting to ONNX...")
        export_to_onnx(model, onnx_path)
        print(f"ONNX saved: {onnx_path}")
    
    # Export to TensorRT
    if args.convert_to in ['tensorrt', 'both']:
        if args.convert_to == 'tensorrt' and not os.path.exists(onnx_path):
            print("\n3. Exporting to ONNX (required for TensorRT)...")
            export_to_onnx(model, onnx_path)
            
        print("\n4. Converting to TensorRT using Python API...")
        result = export_to_tensorrt_python(onnx_path, engine_path, args.fp16)
        
        if result:
            print(f"TensorRT engine saved: {engine_path}")
        else:
            print("TensorRT conversion failed")
            return
    
    print("\n=== CONVERSION COMPLETED ===")
    print("UPDATE DEEPSTREAM CONFIG:")
    print(f"model-engine-file=../../models/{args.model_name}.engine")
    print("net-scale-factor=1.0")
    print("offsets=0.0;0.0;0.0")


if __name__ == "__main__":
    main()