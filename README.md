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
