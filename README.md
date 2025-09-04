# DeepStream Setup Guide

This guide explains how to run DeepStream with your custom model.

---

## 1. Pull NVIDIA DeepStream Image

Make sure **Docker** and **NVIDIA Container Toolkit** are installed.
Then pull the official DeepStream image:

```bash
docker pull nvcr.io/nvidia/deepstream:6.4-triton-multiarch
```

Run the container:

```bash
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    nvcr.io/nvidia/deepstream:6.4-triton-multiarch /bin/bash
```

---

## 2. Clone Repository and Setup Environment

Inside the container, clone the repository and install the environment:

```bash
git clone https://github.com/supervisely-research/deepstream
cd deepstream
pip install -e .
```

---

## 3. Prepare Model

Place your model files inside the `models/` directory of the repository:

```
deepstream/
 └── models/
      └── your_model.pth
      └── model_config.yml
      └── labels.txt
```

* **`your_model.pth`** — PyTorch checkpoint.
* **`model_config.yml`** — model configuration file for conversion.
* **`labels.txt`** — one class name per line.

Make sure all three files are present before proceeding.

---

## 4. Convert Model to TensorRT Engine

Run the provided conversion script to generate an `.engine` file:

```bash
python3 convert.py \
    --pth_path models/your_model.pth \
    --config_path models/model_config.yml \
    --model_name your_model
```

---

## 5. Compile Custom C++ Output Parser

A custom C++ parser (`nvds_dfine_parser.cpp`) is required for D-FINE model outputs.

### Variant A: Workstation (RTX 4090, x86\_64)

```bash
g++ -c -fPIC -std=c++11 nvds_dfine_parser.cpp \
    -I /opt/nvidia/deepstream/deepstream/sources/includes \
    -I /usr/local/cuda/include

g++ -shared nvds_dfine_parser.o -o libnvds_dfine_parser.so

# Copy to DeepStream system library folder
sudo cp libnvds_dfine_parser.so /opt/nvidia/deepstream/deepstream/lib/
```

### Variant B: Jetson (aarch64)

On Jetson devices, the compilation must be done directly on the device (different architecture):

```bash
g++ -c -fPIC -std=c++11 nvds_dfine_parser.cpp \
    -I /opt/nvidia/deepstream/deepstream/sources/includes \
    -I /usr/local/cuda/include

g++ -shared nvds_dfine_parser.o -o libnvds_dfine_parser.so

sudo cp libnvds_dfine_parser.so /opt/nvidia/deepstream/deepstream/lib/
```

⚠️ Note: the commands look the same, but the binary is architecture-specific. You cannot reuse the `.so` compiled on x86\_64 for Jetson.

---

## 6. Configure DeepStream

You need to adjust **two configuration files**:

1. `configs/deepstream-app/source4_1080p_dec_infer-resnet_tracker_sgie_tiled_display_int8.txt` (pipeline config).
2. `configs/deepstream-app/config_infer_dfine.txt` (inference config).

### A. Edit `source4_1080p_dec_infer-resnet_tracker_sgie_tiled_display_int8.txt`

Set input video and output file:

```ini
[source0]
uri=file://../../data/test_video.avi   # path to your input video

[sink0]
output-file=output.mp4                     # path to save result video

[primary-gie]
model-engine-file=../../models/your_model.engine  # path to your engine
config-file=config_infer_dfine.txt
```

### B. Edit `config_infer_dfine.txt`

Set engine file and labels file:

```ini
[property]
model-engine-file=../../models/your_model.engine
labelfile-path=../../models/labels.txt
num-detected-classes=5  # number of lines in labels.txt

parse-bbox-func-name=NvDsInferParseCustomDFINE
custom-lib-path=/opt/nvidia/deepstream/deepstream/lib/libnvds_dfine_parser.so
```

⚠️ Update `num-detected-classes` to match the exact number of classes in `labels.txt`.

---

## 7. Run DeepStream

Launch DeepStream with your configuration:

```bash
deepstream-app \
    -c configs/deepstream-app/source4_1080p_dec_infer-resnet_tracker_sgie_tiled_display_int8.txt
```

---

## 8. Save Predictions to JSON (Optional)

If you need to extract predictions (bounding boxes, confidence scores, track IDs) to a JSON file for further analysis, use the provided prediction extraction tool.

### A. Compile Prediction Extractor

Navigate to the configuration directory and compile the prediction extractor:

```bash
cd configs/deepstream-app/
make clean
make
```

This creates an executable called `deepstream_save_predictions`.

### B. Run with Prediction Saving

Execute the prediction extractor instead of the standard deepstream-app:

```bash
./deepstream_save_predictions ../../data/test_video.avi predictions.json
```

**Parameters:**
- `../../data/test_video.avi` — path to input video file
- `predictions.json` — output JSON file with predictions

### C. Output Format

The `predictions.json` file contains one JSON object per line (JSON Lines format):

```json
{"frame_id":0,"timestamp":1234567890,"objects":[{"bbox":{"left":100.5,"top":200.3,"width":50.2,"height":80.1},"confidence":0.85,"class_id":0,"track_id":1,"class_name":"vehicle"}]}
{"frame_id":1,"timestamp":1234567891,"objects":[{"bbox":{"left":102.1,"top":201.8,"width":49.8,"height":79.5},"confidence":0.83,"class_id":0,"track_id":1,"class_name":"vehicle"}]}
```

**Fields description:**
- `frame_id` — frame number (starts from 0)
- `timestamp` — frame timestamp
- `objects` — array of detected objects in the frame
- `bbox` — bounding box coordinates (left, top, width, height)  
- `confidence` — detection confidence score (0.0-1.0)
- `class_id` — object class ID
- `track_id` — unique tracking ID for the object
- `class_name` — class name from labels.txt (or "unknown" if not set)

## 9. DeepStream Notes and Workflow

### What is NVIDIA DeepStream?

NVIDIA DeepStream is an SDK for accelerated video analytics on GPUs. It allows building pipelines (detection, tracking, etc.) on top of GStreamer with minimal latency.

**Why did we choose it?**

* Optimized infrastructure for real-time processing
* High speed and low latency
* Built-in TensorRT support
* Ready-to-use GPU-accelerated trackers

**nvSORT** is NVIDIA’s GPU-optimized implementation of the SORT tracker. It delivers high FPS because it is lightweight, avoids heavy ReID computations, runs fully on GPU, and is tightly integrated into DeepStream.

---

### Overall Workflow

Connecting the D-FINE detector to DeepStream involves three main stages:

1. Building the container image and setting up the environment
2. Converting the model from PyTorch to TensorRT format
3. Configuring DeepStream files and running inference

Below we summarize key insights and issues encountered in each stage.

---

### 1. Building the Image and Setting Up the Environment

* Tests were performed on an RTX 4090 GPU with driver **575** and CUDA **12.9**.
* The image used was `nvcr.io/nvidia/deepstream:6.4-triton-multiarch`, which supports CUDA 12.6 and driver versions **560.35.03+**.
* On Jetson devices, it is recommended to use **JetPack** images, which bundle Ubuntu, CUDA, TensorRT, cuDNN, and DeepStream for ARM.

⚠️ Choosing the correct image is critical to avoid compatibility issues.

Additionally, the repository includes dependencies from the **DEIM project**, installed via `setup.py`. Ensure these dependencies are correctly installed to avoid conversion errors.

---

### 2. Model Conversion Insights

During conversion from PyTorch to TensorRT, several important issues were discovered:

1. **Multiple inputs issue**: DeepStream expects YOLO-like models with a single input. The D-FINE detector originally had two inputs: an image tensor and metadata with the original image size. Since we use a fixed input size, the metadata was treated as a constant, allowing us to hardcode it and remove the second input during conversion.

   * This resolved initial loading issues, and inference started successfully.

2. **Output format mismatch**: YOLO and D-FINE models produce different output structures. Instead of modifying the conversion script to mimic YOLO outputs, we implemented a **custom C++ parser** (`nvds_dfine_parser.cpp`) to correctly unpack D-FINE outputs in DeepStream.

3. **Normalization mismatch**: In PyTorch, channels are normalized separately (e.g., \[0.485, 0.456, 0.406]), while DeepStream applies a single global scale factor. To resolve this, the TensorRT conversion code was modified to apply **per-channel normalization**, ensuring correct detection results.

---

### 3. Configuring DeepStream

Configuration files are stored in `configs/deepstream-app/`. DeepStream is **very sensitive** to even small config errors, so edit carefully.

* The default reference file is `config_infer_primary.txt`.
* A custom config file `config_infer_dfine.txt` was created for the D-FINE detector.

#### Example changes in `config_infer_dfine.txt`:

```ini
[property]
model-engine-file=../../models/your_model.engine
labelfile-path=../../models/labels.txt
num-detected-classes=5  # match number of lines in labels.txt

parse-bbox-func-name=NvDsInferParseCustomDFINE
custom-lib-path=/opt/nvidia/deepstream/deepstream/lib/libnvds_dfine_parser.so

[class-attrs-0]
pre-cluster-threshold=0.3
# repeat [class-attrs-N] for each class with custom thresholds
```

#### Example changes in `source4_1080p_dec_infer-resnet_tracker_sgie_tiled_display_int8.txt`:

```ini
[source0]
uri=file://../../data/test_video.avi   # path to input video

[sink0]
output-file=output.mp4                 # path for output video

[primary-gie]
model-engine-file=../../models/your_model.engine
config-file=config_infer_dfine.txt
```

#### Example of `labels.txt`:

```
horse
horse head
number plate
rider
yellow stick
white stick
```

⚠️ Ensure all referenced files exist. Paths assume execution from `configs/deepstream-app/`. If running from project root, adjust paths accordingly.

---

### 4. Alternative Outputs

Instead of saving results only as `.mp4` videos, a **custom C module** (`deepstream_save_predictions.c`) was written to dump predictions into JSON format. This provides flexibility for downstream processing or custom integrations.

---

### 5. Performance

* Achieved **275 FPS** at **640×640 resolution**.
* The `nvSORT` tracker performed well, maintaining consistent object IDs visually.
