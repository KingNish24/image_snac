# image_snac/setup.py
import io
import os
from setuptools import find_packages, setup

# Read version from __init__.py (replace with actual version management if needed)
VERSION = "0.1.0" # Placeholder

def read(*paths, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *paths), encoding=kwargs.get("encoding", "utf8")) as open_file:
        content = open_file.read().strip()
    return content

def read_requirements(path):
    return [line.strip() for line in read(path).split("\n") if not line.startswith(('"', "#", "-", "git+"))]

setup(
    name="img_snac",
    version=VERSION,
    author="Your Name", # Replace
    author_email="your.email@example.com", # Replace
    description="Multi-Scale Neural Image Codec (Inspired by SNAC)",
    url="https://github.com/your_username/image_snac", # Replace
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["scripts", "data"]),
    install_requires=read_requirements("requirements.txt"),
    python_requires='>=3.8', # Specify Python version compatibility
)
