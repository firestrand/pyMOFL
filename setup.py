"""
Setup script for the pyMOFL package.
"""

from setuptools import setup, find_packages

setup(
    name="pyMOFL",
    version="0.1.0",
    description="Python Modular Optimization Function Library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="pyMOFL Team",
    author_email="example@example.com",
    url="https://github.com/yourusername/pyMOFL",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
) 