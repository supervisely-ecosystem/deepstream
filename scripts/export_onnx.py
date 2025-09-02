import sys
sys.path.append('../')
sys.path.append('../deim')
from deim.supervisely_integration.export import export_onnx, export_tensorrt

export_onnx(checkpoint_path='../models/best.pth', config_path='../models/model_config.yml', output_dir='../models')
