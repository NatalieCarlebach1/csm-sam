from setuptools import setup, find_packages

setup(
    name="csmsam",
    version="0.1.0",
    description="Cross-Session Memory SAM for adaptive radiotherapy tumor segmentation",
    author="",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.24.0",
        "SimpleITK>=2.3.0",
        "nibabel>=5.1.0",
        "monai>=1.3.0",
        "einops>=0.7.0",
        "omegaconf>=2.3.0",
        "tqdm>=4.66.0",
        "matplotlib>=3.8.0",
        "scipy>=1.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ]
    },
)
