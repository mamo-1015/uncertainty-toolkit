from setuptools import setup, find_packages

setup(
    name="uncertainty-toolkit",
    version="0.1.0",
    description="Plug-and-play uncertainty estimation for PyTorch classifiers",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
    ],
)
