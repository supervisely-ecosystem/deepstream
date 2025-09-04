#include <cstring>
#include <iostream>
#include <algorithm>
#include <cmath>
#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferParseCustomDFINE(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
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

    const float* labels = static_cast<const float*>(labelsLayer->buffer);
    const float* boxes = static_cast<const float*>(boxesLayer->buffer);
    const float* scores = static_cast<const float*>(scoresLayer->buffer);

    int numDetections = labelsLayer->inferDims.d[0];
    
    for (int i = 0; i < numDetections; i++) {
        float score = scores[i];
        
        if (std::isnan(score) || std::isnan(boxes[i*4]) || std::isnan(boxes[i*4+1]) || 
            std::isnan(boxes[i*4+2]) || std::isnan(boxes[i*4+3])) {
            continue;
        }
        
        if (score < 0.3f) {
            continue;
        }

        NvDsInferParseObjectInfo obj;
        obj.classId = static_cast<unsigned int>(std::lround(labels[i]));
        obj.detectionConfidence = scores[i];

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

    return true;
}