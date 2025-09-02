import sys
import argparse
sys.path.append('../')
sys.path.append('../deim')
from deim.supervisely_integration.export import export_onnx, export_tensorrt

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--convert_to", type=str, choices=['onnx', 'tensorrt', 'both'], default='both', help="Type of model conversion")
    parser.add_argument("--pth_path", type=str, default="../models/best.pth", help="Path to the .pth model file")
    parser.add_argument("--config_path", type=str, default="../models/model_config.yml", help="Path to the model config file")
    parser.add_argument("--output_dir", type=str, default="../models", help="Directory to save the exported models")
    parser.add_argument("--fp16", default=True, action="store_false", help="Export TensorRT model in FP16 precision") 
    args = parser.parse_args()
    
    if args.convert_to == 'onnx':
        onnx_path = export_onnx(args.pth_path, args.config_path, args.output_dir)
        print(f"ONNX model exported to: {onnx_path}")
    elif args.convert_to == 'tensorrt':
        onnx_path = export_onnx(args.pth_path, args.config_path, args.output_dir)
        trt_path = export_tensorrt(onnx_path, args.output_dir, args.fp16)
        print(f"TensorRT model exported to: {trt_path}")
    elif args.convert_to == 'both':
        onnx_path = export_onnx(args.pth_path, args.config_path, args.output_dir)
        print(f"ONNX model exported to: {onnx_path}")
        trt_path = export_tensorrt(onnx_path, args.output_dir, args.fp16)
        print(f"TensorRT model exported to: {trt_path}")    

if __name__ == "__main__":
    main()

