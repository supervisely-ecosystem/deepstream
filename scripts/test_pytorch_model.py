#!/usr/bin/env python3
"""
Test original PyTorch D-FINE model
"""

import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path

# Add DEIM paths
current_dir = Path(__file__).parent
deim_root = current_dir / '..' / 'deim'
workspace_root = current_dir / '..'

sys.path.insert(0, str(deim_root))
sys.path.insert(0, str(workspace_root))

def load_model(checkpoint_path, config_path):
    """Load DEIM model"""
    from engine.core import YAMLConfig

    cfg = YAMLConfig(config_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model', checkpoint)
    else:
        state_dict = checkpoint

    model = cfg.model  # получаем модель
    model.load_state_dict(state_dict)
    model.eval()

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(f"Модель загружена: {checkpoint_path}")
    print(f"Устройство: {device}")
    return cfg, model, device


def preprocess_frame(frame, target_size=640):
    """Preprocess frame for model"""
    h, w = frame.shape[:2]
    
    # Resize keeping aspect ratio
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(frame, (new_w, new_h))
    
    # Pad to square
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    
    padded = cv2.copyMakeBorder(
        resized, pad_h, target_size - new_h - pad_h,
        pad_w, target_size - new_w - pad_w,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    
    # Convert for PyTorch: HWC -> CHW, BGR -> RGB, normalize
    tensor = torch.from_numpy(padded).float()
    tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
    tensor = tensor[[2, 1, 0], :, :]  # BGR -> RGB
    tensor = tensor / 255.0
    tensor = tensor.unsqueeze(0)  # Add batch dimension
    
    return tensor, scale, (pad_w, pad_h)

def draw_detections(frame, boxes, scores, labels, threshold=0.3):
    """Draw detections on frame"""
    result = frame.copy()
    
    valid_detections = 0
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        if score < threshold:
            continue
            
        valid_detections += 1
        x1, y1, x2, y2 = box.astype(int)
        
        # Draw bbox
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Label
        text = f"Class {int(label)}: {score:.2f}"
        cv2.putText(result, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (0, 255, 0), 1)
    
    return result, valid_detections

def main():
    # Parameters
    checkpoint_path = "../models/best.pth"
    config_path = "../models/model_config.yml"
    video_path = "../data/test_video_640.avi"
    output_path = "../data/pytorch_output.mp4"
    
    # Speed optimizations
    skip_frames = 10  # Process every 10th frame
    max_frames = 1000  # Max frames for quick test
    
    # Check files
    for path in [checkpoint_path, config_path, video_path]:
        if not os.path.exists(path):
            print(f"Файл не найден: {path}")
            return
    
    print("=== Тест PyTorch D-FINE модели ===")
    
    # Load model
    cfg, model, device = load_model(checkpoint_path, config_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Не удается открыть видео: {video_path}")
        return
    
    # Video parameters
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Видео: {width}x{height}, {fps} FPS, {total_frames} кадров")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    processed_count = 0
    total_detections = 0
    max_confidence = 0.0
    detection_stats = []
    
    print("\nОбработка видео...")
    
    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break
            
        frame_count += 1
        
        # Skip frames for speed
        if frame_count % skip_frames != 0:
            out.write(frame)  # Write original frame
            continue
            
        processed_count += 1
        
        # Preprocessing
        tensor, scale, padding = preprocess_frame(frame)
        tensor = tensor.to(device)  # Move to GPU
        orig_target_sizes = torch.tensor([[height, width]], dtype=torch.int32).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(tensor)
            outputs = cfg.postprocessor(outputs, orig_target_sizes)
        
        # Debug output structure on first frame
        if processed_count == 1:
            print(f"\nТип выходов: {type(outputs)}")
            if isinstance(outputs, (list, tuple)):
                print(f"Длина: {len(outputs)}")
                for i, item in enumerate(outputs):
                    print(f"  outputs[{i}]: {type(item)}")
                    if isinstance(item, dict):
                        print(f"    keys: {list(item.keys())}")
                    elif hasattr(item, 'shape'):
                        print(f"    shape: {item.shape}")
        
        # Parse outputs
        boxes = scores = labels = np.array([])
        
        try:
            # Case 1: list with dict inside
            if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                output_dict = outputs[0]
                if isinstance(output_dict, dict):
                    labels = output_dict.get('labels', torch.tensor([[]])).cpu().numpy().flatten()
                    boxes = output_dict.get('boxes', torch.tensor([[]])).cpu().numpy()
                    scores = output_dict.get('scores', torch.tensor([[]])).cpu().numpy().flatten()
                    
                    # Remove batch dimension if present
                    if len(boxes.shape) == 3:
                        boxes = boxes[0]
                    if len(labels.shape) == 2:
                        labels = labels[0] if labels.shape[0] > 0 else labels.flatten()
                    if len(scores.shape) == 2:
                        scores = scores[0] if scores.shape[0] > 0 else scores.flatten()
            
        except Exception as e:
            if processed_count == 1:
                print(f"Ошибка парсинга: {e}")
            boxes = scores = labels = np.array([])
        
        # Postprocess coordinates
        if len(boxes) > 0 and len(scores) > 0:
            # Remove padding and scale back to original size
            pad_w, pad_h = padding
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / scale
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / scale
            
            # Clip coordinates to image bounds
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
        
        # Statistics
        frame_detections = 0
        if len(scores) > 0:
            valid_mask = scores > 0.15
            valid_scores = scores[valid_mask]
            frame_detections = len(valid_scores)
            
            if frame_detections > 0:
                frame_max_conf = valid_scores.max()
                max_confidence = max(max_confidence, frame_max_conf)
                total_detections += frame_detections
                detection_stats.append((frame_count, frame_detections, frame_max_conf))
        
        # Draw detections (threshold 0.3 for visualization)
        if len(boxes) > 0 and len(scores) > 0 and len(labels) > 0:
            result_frame, visual_detections = draw_detections(
                frame, boxes, scores, labels, threshold=0.3
            )
        else:
            result_frame = frame
            visual_detections = 0
        
        # Write frame
        out.write(result_frame)
        
        # Debug every 10 processed frames
        if processed_count % 10 == 0:
            print(f"Обработано {processed_count} кадров из {frame_count} всего: "
                  f"детекций >0.15: {frame_detections}, "
                  f"макс. conf: {valid_scores.max():.3f}" if frame_detections > 0 
                  else f"детекций: 0")
    
    # Close files
    cap.release()
    out.release()
    
    # Final statistics
    print(f"\n=== РЕЗУЛЬТАТЫ ===")
    print(f"Обработано кадров: {processed_count} из {frame_count}")
    print(f"Всего детекций >0.15: {total_detections}")
    print(f"Максимальная уверенность: {max_confidence:.3f}")
    if processed_count > 0:
        print(f"Среднее детекций на кадр: {total_detections/processed_count:.2f}")
    
    # Top detections
    if detection_stats:
        detection_stats.sort(key=lambda x: x[2], reverse=True)
        print(f"\nТОП-5 лучших детекций:")
        for i, (frame_num, count, conf) in enumerate(detection_stats[:5]):
            print(f"  {i+1}. Кадр {frame_num}: {count} детекций, макс. conf: {conf:.3f}")
    
    print(f"\nРезультат сохранен: {output_path}")

if __name__ == "__main__":
    main()