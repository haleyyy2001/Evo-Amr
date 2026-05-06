#!/usr/bin/env python3
from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")


def get_version():
    """Read the package version from the repository package marker."""
    with open(this_directory / "__init__.py", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"


setup(
    name="evo-amr",
    version=get_version(),
    author="Haley Tran",
    author_email="haleyyy2001@users.noreply.github.com",
    description="Genomic sequence analysis using evolutionary language models for antimicrobial resistance prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/haleyyy2001/Evo-Amr",
    packages=(
        find_packages(where="src")
        + find_packages(exclude=["tests", "tests.*", "docs", "examples"])
    ),
    package_dir={"evo_amr": "src/evo_amr"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "biopython>=1.81",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "h5py>=3.9.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "gpu": [
            "flash-attn>=2.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "plotly>=5.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "evo-amr=evo_amr.cli:main",
            "evo-amr-setup=initial_setup:main",
            "evo-embed=embedding_pipeline.embedding_generator:main",
            "evo-minirocket=minirocket.minirocket_pipeline.amr_classification_pipeline:main",
        ],
    },
    include_package_data=True,
    package_data={
        "config": ["*.yaml"],
        "embedding_pipeline": ["*.yaml"],
        "models.evo_enhanced": ["*.yaml"],
        "minirocket.minirocket_pipeline.examples": ["*.yaml"],
    },
    zip_safe=False,
)
