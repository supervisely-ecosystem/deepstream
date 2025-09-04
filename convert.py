#!/usr/bin/env python3
"""
Simple D-FINE converter with embedded scaling - DEFAULT pth -> onnx -> engine
"""

import os
import sys
import torch
import torch.nn as nn
import argparse
import subprocess

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
    """D-FINE wrapper with embedded per-channel normalization"""
    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
        
        # Fixed target sizes for DeepStream
        self.register_buffer('orig_target_sizes', torch.tensor([[640, 640]], dtype=torch.int32))
        
        # ImageNet normalization (RGB order)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def forward(self, images):
        """Forward with embedded preprocessing"""
        images = images.float()
        images = images[:, [2, 1, 0], :, :]  # BGR -> RGB
        images = images / 255.0
        images = (images - self.mean) / self.std
        
        batch_size = images.shape[0]
        target_sizes = self.orig_target_sizes.expand(batch_size, -1)
        
        outputs = self.model(images)
        labels, boxes, scores = self.postprocessor(outputs, target_sizes)
        
        return labels.float(), boxes.float(), scores.float()


def export_to_onnx(model, output_path: str):
    """Export to ONNX"""
    model.eval()
    dummy_input = torch.randint(0, 256, (1, 3, 640, 640), dtype=torch.float32)
    
    with torch.no_grad():
        test_output = model(dummy_input)
    print(f"Model test passed. Outputs: {len(test_output)}")
    
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
    
    print(f"ONNX exported: {output_path}")
    return output_path


def export_to_tensorrt(onnx_path: str, output_path: str, fp16: bool = False):
    """Convert to TensorRT using trtexec"""
    print(f"Converting to TensorRT: {onnx_path} -> {output_path}")
    
    cmd = [
        '/usr/bin/trtexec',
        f'--onnx={onnx_path}',
        f'--saveEngine={output_path}'
    ]
    
    if fp16:
        cmd.append('--fp16')
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
        print("TensorRT conversion: SUCCESS")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"TensorRT conversion failed: {e.returncode}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        if fp16:
            print("Retrying without FP16...")
            return export_to_tensorrt(onnx_path, output_path, fp16=False)
        raise


def main():
    parser = argparse.ArgumentParser(description="Simple D-FINE converter with embedded scaling")
    parser.add_argument("--pth_path", type=str, default="../models/best.pth")
    parser.add_argument("--config_path", type=str, default="../models/model_config.yml")
    parser.add_argument("--output_dir", type=str, default="../models")
    parser.add_argument("--model_name", type=str, default="best_simple_fixed")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 TensorRT export")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pth_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.pth_path}")
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config not found: {args.config_path}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== Simple D-FINE Converter ===")
    print(f"Model: {args.pth_path}")
    print(f"Config: {args.config_path}")
    print(f"Output: {args.model_name}")
    
    # Load model
    cfg = load_deim_model(args.pth_path, args.config_path)
    model = SingleInputDFINEEmbeddedScaling(cfg)
    
    onnx_path = os.path.join(args.output_dir, f"{args.model_name}.onnx")
    engine_path = os.path.join(args.output_dir, f"{args.model_name}.engine")
    
    # Export ONNX
    export_to_onnx(model, onnx_path)
    
    # Export TensorRT
    export_to_tensorrt(onnx_path, engine_path, fp16=args.fp16)
    
    print("\n=== CONVERSION COMPLETE ===")

if __name__ == "__main__":
    main()
