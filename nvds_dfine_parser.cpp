#include <cstring>
#include <iostream>
#include <algorithm>
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
    const int64_t* labels = static_cast<const int64_t*>(labelsLayer->buffer);
    const float* boxes = static_cast<const float*>(boxesLayer->buffer);
    const float* scores = static_cast<const float*>(scoresLayer->buffer);

    // Parse detections
    int numDetections = labelsLayer->inferDims.d[0]; // Should be 300
    
    std::cout << "=== DFINE PARSER DEBUG ===" << std::endl;
    std::cout << "numDetections: " << numDetections << std::endl;
    std::cout << "threshold: " << detectionParams.perClassThreshold[0] << std::endl;
    
    int validDetections = 0;
    for (int i = 0; i < numDetections; i++) {
        float score = scores[i];
        
        // Debug first 10 detections
        if (i < 10) {
            std::cout << "Detection " << i << ": score=" << score << ", label=" << labels[i] 
                      << ", box=[" << boxes[i*4] << "," << boxes[i*4+1] << "," 
                      << boxes[i*4+2] << "," << boxes[i*4+3] << "]" << std::endl;
        }
        
        // Filter by confidence threshold  
        if (score < detectionParams.perClassThreshold[0]) {
            continue;
        }
        
        validDetections++;

        NvDsInferParseObjectInfo obj;
        obj.classId = static_cast<unsigned int>(labels[i]);
        obj.detectionConfidence = score;

        // Boxes format: [x1, y1, x2, y2] normalized
        float x1 = boxes[i * 4 + 0] * networkInfo.width;
        float y1 = boxes[i * 4 + 1] * networkInfo.height;
        float x2 = boxes[i * 4 + 2] * networkInfo.width;
        float y2 = boxes[i * 4 + 3] * networkInfo.height;

        obj.left = static_cast<unsigned int>(std::max(0.0f, x1));
        obj.top = static_cast<unsigned int>(std::max(0.0f, y1));
        obj.width = static_cast<unsigned int>(std::max(0.0f, x2 - x1));
        obj.height = static_cast<unsigned int>(std::max(0.0f, y2 - y1));

        objectList.push_back(obj);
    }

    std::cout << "Total valid detections: " << validDetections << "/" << numDetections << std::endl;
    std::cout << "=========================" << std::endl;

    return true;
}