"""
Setup script for InvisioVault - Advanced Steganography Suite
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""

# Read requirements
requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    requirements = requirements_file.read_text(encoding='utf-8').strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="invisiovault",
    version="1.0.0",
    author="Rolan (RNR)",
    author_email="rolan.education@example.com",
    description="Professional-grade steganography application with advanced security features",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/invisiovault/invisiovault",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Security :: Cryptography",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Security"
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0"
        ],
        "test": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-xvfb>=3.0.0"  # For GUI testing
        ]
    },
    entry_points={
        "console_scripts": [
            "invisiovault=main:main",
        ],
        "gui_scripts": [
            "invisiovault-gui=main:main",
        ]
    },
    package_data={
        "invisiovault": [
            "assets/icons/*",
            "assets/themes/*",
            "assets/sounds/*",
            "docs/*.md",
            "tests/fixtures/*"
        ]
    },
    include_package_data=True,
    zip_safe=False,
    keywords="steganography, encryption, security, privacy, aes, lsb, image-processing",
    project_urls={
        "Documentation": "https://invisiovault.readthedocs.io/",
        "Bug Reports": "https://github.com/invisiovault/invisiovault/issues",
        "Source": "https://github.com/invisiovault/invisiovault",
        "Changelog": "https://github.com/invisiovault/invisiovault/blob/main/docs/changelog.md"
    }
)
