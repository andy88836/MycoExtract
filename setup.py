"""
MycoExtract: Multi-Agent LLM Pipeline for Automated Construction of
Mycotoxin-Degrading Enzyme Kinetics Database

Setup script for installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="mycoextract",
    version="1.0.0",
    author="MycoExtract Contributors",
    author_email="your.email@institution.edu",
    description="Multi-Agent LLM Pipeline for Mycotoxin-Degrading Enzyme Data Extraction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/mycoextract",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "mycoextract=scripts.run_extraction:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.yaml", "*.yml", "*.json"],
    },
)
