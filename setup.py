from setuptools import setup, find_packages

setup(
    name="deim",
    version="0.1.0",
    description="DEIM: DETR with Improved Matching for Fast Convergence + converter to TensorRT",
    author="Supervisely Ecosystem",
    license="Apache-2.0",
    packages=find_packages(include=["deim", "deim.*"]),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "onnx>=1.13.0",
        "onnxruntime>=1.14.0",
        "numpy>=1.23.0",
        "tensorboard>=2.12.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0"
    ],
    extras_require={
        "dev": [
            "black",
            "flake8",
            "isort",
        ]
    },
    python_requires=">=3.8",
)
