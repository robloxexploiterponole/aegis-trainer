from setuptools import setup, find_packages

setup(
    name="aegis-trainer",
    version="0.1.0",
    author="Hardwick Software Services",
    author_email="jon@justcalljon.pro",
    description="AEGIS AI Trainer: Layer-by-layer model training and modification framework. Modify 80B+ parameter models on consumer GPUs using AirLLM layer streaming.",
    long_description=open("README.md", encoding="utf-8").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/jonhardwick-spec/aegis-trainer",
    packages=find_packages(),
    install_requires=[
        "airllm>=2.11.0",
        "torch>=2.4.0",
        "transformers>=4.51.0",
        "safetensors>=0.4.0",
        "accelerate>=0.34.0",
        "peft>=0.18.0",
        "textual>=0.40.0",
        "click>=8.0.0",
        "rich>=13.0.0",
        "psutil>=5.9.0",
    ],
    entry_points={
        "console_scripts": [
            "aegis-trainer=aegis_trainer.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    license="SSPL-1.0",
)
