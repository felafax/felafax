#!/bin/bash

# Default values
BUILD_JAX=0
BUILD_TORCH=0
SHOULD_PUSH=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --jax)
            BUILD_JAX=1
            shift
            ;;
        --torch)
            BUILD_TORCH=1
            shift
            ;;
        --push)
            SHOULD_PUSH=1
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# If no specific image is selected, build both
if [ $BUILD_JAX -eq 0 ] && [ $BUILD_TORCH -eq 0 ]; then
    BUILD_JAX=1
    BUILD_TORCH=1
fi

# Function to build and optionally push an image
build_and_push() {
    local dockerfile=$1
    local image_name=$2
    
    echo "Building $image_name..."
    docker build \
        --platform linux/amd64 \
        -f $dockerfile \
        -t $image_name \
        .

    if [ $? -ne 0 ]; then
        echo "Failed to build $image_name"
        exit 1
    fi

    if [ $SHOULD_PUSH -eq 1 ]; then
        echo "Pushing $image_name..."
        docker push $image_name
        if [ $? -ne 0 ]; then
            echo "Failed to push $image_name"
            exit 1
        fi
    fi
}

# Build JAX image if selected
if [ $BUILD_JAX -eq 1 ]; then
    build_and_push \
        "scripts/docker/Dockerfile.jax" \
        "gcr.io/felafax-training/roadrunner-jax:latest_v2"
fi

# Build Torch image if selected
if [ $BUILD_TORCH -eq 1 ]; then
    build_and_push \
        "scripts/docker/Dockerfile.torch" \
        "gcr.io/felafax-training/roadrunner-torchxla:latest_v2"
fi

echo "Done!"
