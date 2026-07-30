from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent

setup(
    name="centroid-erasure",
    version="1.0.0",
    description=(
        "Centroid replacement, a training-free probe for text-vs-visual "
        "dependence in multimodal language models, and text centroid "
        "contrastive decoding (TCCD)."
    ),
    author="Akshay Paruchuri",
    license="Apache-2.0",
    url="https://github.com/yahskapar/centroid-erasure",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["centroid_erasure", "centroid_erasure.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch==2.6.0",
        "torchvision==0.21.0",
        "transformers==5.4.0",
        "accelerate==1.13.0",
        "numpy==2.2.6",
        "scikit-learn==1.7.2",
        "scipy==1.15.3",
        "datasets==4.8.4",
        # Hard import for the default model (Qwen2.5-VL) in prepare_inputs.
        "qwen-vl-utils==0.0.14",
        "Pillow==12.1.1",
        "tqdm==4.67.3",
        "huggingface-hub==1.8.0",
        "pandas==2.3.3",
    ],
    extras_require={
        # Needed only by quantised registry entries such as MedGemma.
        "quantization": ["bitsandbytes==0.49.2"],
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
    ],
)
