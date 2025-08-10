"""
Setup script for InvisioVault - Advanced Steganography Suite

Author: Rolan (RNR)
Purpose: Educational project for advanced steganography and security
License: MIT Educational License
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""

# Read requirements - separate runtime from dev dependencies
runtime_requirements = [
    "PySide6>=6.5.0",
    "Pillow>=9.5.0",  
    "numpy>=1.24.0",
    "cryptography>=41.0.0"
]

build_requirements = [
    "pyinstaller>=5.13.0"
]

setup(
    name="invisiovault",
    version="1.0.0",
    author="Rolan (RNR)",
    author_email="rolanlobo901@gmail.com",
    description="Revolutionary steganography suite with 100x faster extraction and advanced security",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mrtracker-new/InvisioVault_R",
    packages=find_packages(exclude=["tests*", "scripts*", "build_scripts*"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Information Technology", 
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Security :: Cryptography",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Security",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ],
    python_requires=">=3.8",
    install_requires=runtime_requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "pre-commit>=3.0.0"
        ],
        "test": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-qt>=4.2.0",  # For PySide6 testing
            "pytest-xvfb>=3.0.0",  # For GUI testing on Linux
            "pytest-mock>=3.11.0"
        ],
        "build": build_requirements,
        "all": [
            "pytest>=7.4.0", "black>=23.7.0", "flake8>=6.0.0", "mypy>=1.5.0",
            "sphinx>=7.1.0", "sphinx-rtd-theme>=1.3.0", "pre-commit>=3.0.0",
            "pytest-cov>=4.1.0", "pytest-qt>=4.2.0", "pytest-xvfb>=3.0.0",
            "pytest-mock>=3.11.0", "pyinstaller>=5.13.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "invisiovault=main:main",
            "invisiovault-cli=main:main",
        ],
        "gui_scripts": [
            "invisiovault-gui=main:main",
        ]
    },
    package_data={
        "": [
            "assets/icons/*",
            "assets/images/*", 
            "assets/ui/*",
            "assets/demo/*",
            "docs/*.md",
            "README.md",
            "LICENSE"
        ]
    },
    include_package_data=True,
    zip_safe=False,
    keywords="steganography, encryption, security, privacy, aes-256, lsb, image-processing, educational, cybersecurity, data-hiding, cryptography, stealth",
    project_urls={
        "Homepage": "https://github.com/Mrtracker-new/InvisioVault_R",
        "Documentation": "https://github.com/Mrtracker-new/InvisioVault_R/tree/master/docs",
        "Bug Reports": "https://github.com/Mrtracker-new/InvisioVault_R/issues",
        "Source Code": "https://github.com/Mrtracker-new/InvisioVault_R",
        "Changelog": "https://github.com/Mrtracker-new/InvisioVault_R/blob/master/docs/changelog.md",
        "User Guide": "https://github.com/Mrtracker-new/InvisioVault_R/blob/master/docs/user_guide.md",
        "API Reference": "https://github.com/Mrtracker-new/InvisioVault_R/blob/master/docs/api_reference.md"
    }
)
