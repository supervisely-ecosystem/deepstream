import torch
try:
    model_data = torch.load('../models/best.pth', map_location='cpu')
    print('=== PYTORCH MODEL INFO ===')
    print('Keys in saved model:', list(model_data.keys()))
    
    if 'model_info' in model_data:
        print('Model info:', model_data['model_info'])
    
    if 'model' in model_data:
        model = model_data['model']
        print('Model type:', type(model))
        
        if hasattr(model, 'names'):
            print('Class names:', model.names)
        if hasattr(model, 'stride'):
            print('Model stride:', model.stride)
        if hasattr(model, 'yaml'):
            print('Model has YAML config')
            
except Exception as e:
    print('Error loading PyTorch model:', e)
    
import json
with open('../model_meta.json', 'r') as f:
    meta = json.load(f)

print('=== MODEL METADATA ===')
for key, value in meta.items():
    if key != 'classes':
        print(f'{key}: {value}')
        
print('\\n=== CLASSES ===')
for i, cls in enumerate(meta['classes']):
    print(f'{i}: {cls["title"]}')