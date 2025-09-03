# run_trt_engine.py
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine("../models/fix_rgb.engine")
context = engine.create_execution_context()

# prepare input (read same frame)
frame = cv2.imread("../data/test_video_640_frame10.jpg")  # подготовь один png 640x640 (тот же кадр)
frame = cv2.resize(frame, (640,640))
# DeepStream дает BGR 0..255 NCHW, model expects that (we embedded preprocessing)
inp = frame.transpose(2,0,1).astype(np.float32)[None, ...]

# allocate buffers
inputs, outputs, bindings, stream = [], [], [], cuda.Stream()
for i in range(engine.num_bindings):
    name = engine.get_binding_name(i)
    shape = context.get_binding_shape(i)
    dtype = trt.nptype(engine.get_binding_dtype(i))
    size = int(np.prod(shape))
    host_mem = cuda.pagelocked_empty(size, dtype)
    dev_mem = cuda.mem_alloc(host_mem.nbytes)
    bindings.append(int(dev_mem))
    if engine.binding_is_input(i):
        inputs.append((name, host_mem, dev_mem, shape, dtype))
    else:
        outputs.append((name, host_mem, dev_mem, shape, dtype))

# copy input
np.copyto(inputs[0][1], inp.ravel())
cuda.memcpy_htod_async(inputs[0][2], inputs[0][1], stream)

# execute
context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
# copy outputs
for name, host_mem, dev_mem, shape, dtype in outputs:
    cuda.memcpy_dtoh_async(host_mem, dev_mem, stream)

stream.synchronize()

# reshape and print
for name, host_mem, dev_mem, shape, dtype in outputs:
    arr = np.array(host_mem).reshape(shape)
    print("OUTPUT", name, arr.dtype, arr.shape)
    print("first 20:", arr.ravel()[:20])
