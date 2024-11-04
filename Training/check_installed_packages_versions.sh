#!/bin/bash

echo "Checking installed package versions..."

# Function to check package version
check_version() {
    package=$1
    if python -c "import $package; print(f'$package: {$package.__version__}')" 2>/dev/null; then
        return 0
    elif python -c "import $package" 2>/dev/null; then
        echo "$package: Version not available but package is installed"
    else
        echo "$package: Not installed"
    fi
}

# Check each package
check_version diffusers
check_version gradio
check_version skimage  # for scikit-image
check_version torch
check_version torchvision
check_version einops
check_version accelerate
check_version transformers
check_version colorlog
check_version cupy
check_version av
check_version gpustat
check_version trimesh
check_version facexlib
check_version omegaconf
check_version librosa
check_version mediapipe
check_version kornia
check_version yacs
check_version gfpgan
check_version numpy

echo "Package version check complete"
