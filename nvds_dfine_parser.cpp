#include <cstring>
#include <iostream>
#include <algorithm>
#include <cmath>
#include "nvdsinfer_custom_impl.h"

// Custom parser for D-FINE model outputs
extern "C" bool NvDsInferParseCustomDFINE(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    // Find output layers
    const NvDsInferLayerInfo* labelsLayer = nullptr;
    const NvDsInferLayerInfo* boxesLayer = nullptr;
    const NvDsInferLayerInfo* scoresLayer = nullptr;

    for (const auto& layer : outputLayersInfo) {
        if (strcmp(layer.layerName, "labels") == 0) {
            labelsLayer = &layer;
        } else if (strcmp(layer.layerName, "boxes") == 0) {
            boxesLayer = &layer;
        } else if (strcmp(layer.layerName, "scores") == 0) {
            scoresLayer = &layer;
        }
    }

    if (!labelsLayer || !boxesLayer || !scoresLayer) {
        std::cerr << "Could not find required output layers" << std::endl;
        return false;
    }

    // Get data pointers
    const float* labels = static_cast<const float*>(labelsLayer->buffer);
    const float* boxes = static_cast<const float*>(boxesLayer->buffer);
    const float* scores = static_cast<const float*>(scoresLayer->buffer);

    // Parse detections
    int numDetections = labelsLayer->inferDims.d[0];
    
    static int frame_count = 0;
    frame_count++;
    
    bool debug_this_frame = (frame_count % 30 == 0);
    
    if (debug_this_frame) {
        std::cout << "=== FRAME " << frame_count << " DEBUG ===" << std::endl;
        std::cout << "numDetections: " << numDetections << std::endl;
        std::cout << "threshold: " << detectionParams.perClassThreshold[0] << std::endl;
    }
    
    int validDetections = 0;
    int highConfDetections = 0;
    
    for (int i = 0; i < numDetections; i++) {
        float score = scores[i];
        
        // Skip NaN values
        if (std::isnan(score) || std::isnan(boxes[i*4]) || std::isnan(boxes[i*4+1]) || 
            std::isnan(boxes[i*4+2]) || std::isnan(boxes[i*4+3])) {
            continue;
        }
        
        // Увеличим threshold для качественной фильтрации
        if (score < 0.3f) {
            continue;
        }
        
        validDetections++;
        if (score > 0.5f) highConfDetections++;
        
        // Debug только первые 3 валидные детекции
        if (debug_this_frame && validDetections <= 3) {
            std::cout << "Detection " << validDetections << ": score=" << score << ", label=" << labels[i] 
                      << ", box=[" << boxes[i*4] << "," << boxes[i*4+1] << "," 
                      << boxes[i*4+2] << "," << boxes[i*4+3] << "]" << std::endl;
        }

        NvDsInferParseObjectInfo obj;
        obj.classId = static_cast<unsigned int>(std::lround(labels[i]));
        obj.detectionConfidence = scores[i];

        // Boxes уже в пикселях для 640x640
        float x1 = boxes[i * 4 + 0];
        float y1 = boxes[i * 4 + 1]; 
        float x2 = boxes[i * 4 + 2];
        float y2 = boxes[i * 4 + 3];

        obj.left = static_cast<unsigned int>(std::max(0.0f, x1));
        obj.top = static_cast<unsigned int>(std::max(0.0f, y1));
        obj.width = static_cast<unsigned int>(std::max(0.0f, x2 - x1));
        obj.height = static_cast<unsigned int>(std::max(0.0f, y2 - y1));

        objectList.push_back(obj);
    }

    if (debug_this_frame) {
        std::cout << "Valid detections: " << validDetections << "/" << numDetections;
        std::cout << " (High conf >0.5: " << highConfDetections << ")" << std::endl;
        std::cout << "=========================" << std::endl;
    }

    return true;
}