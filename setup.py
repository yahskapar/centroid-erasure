from setuptools import find_packages, setup

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
    packages=find_packages(include=["centroid_erasure", "centroid_erasure.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch==2.6.0",
        "transformers==5.4.0",
        "accelerate==1.13.0",
        "numpy==2.2.6",
        "scikit-learn==1.7.2",
        "scipy==1.15.3",
        "datasets>=4.0.0",
        # Hard import for the default model (Qwen2.5-VL) in prepare_inputs.
        "qwen-vl-utils>=0.0.8",
        "Pillow>=10.0.0",
        "tqdm>=4.66.0",
        # Needed by the quantised registry entries (e.g. medgemma).
        "bitsandbytes>=0.43.0",
    ],
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
    ],
)
