#!/bin/bash

build_and_push() {
  local dockerfile=$1
  local image_name=$2

  echo "Building Docker image: $image_name"
  docker build -t $image_name -f $dockerfile .

  if [ $? -eq 0 ]; then
    echo "Docker image built successfully"
  else
    echo "Docker image build failed"
    return 1
  fi

  echo "Pushing image to GCR: $image_name"
  docker push $image_name

  if [ $? -eq 0 ]; then
    echo "Image pushed to GCR successfully"
  else
    echo "Failed to push image to GCR"
    return 1
  fi

  return 0
}

# Check if an argument is provided
if [ $# -eq 0 ]; then
  echo "No argument provided. Building and pushing both images."

  build_and_push "utils/docker/Dockerfile.jax" "gcr.io/felafax-training/roadrunner-jax:latest_v2"
  jax_result=$?

  build_and_push "utils/docker/Dockerfile.torch" "gcr.io/felafax-training/roadrunner-torchxla:latest_v2"
  xla_result=$?

  if [ $jax_result -eq 0 ] && [ $xla_result -eq 0 ]; then
    echo "Both images built and pushed successfully"
  else
    echo "One or both image builds/pushes failed"
    exit 1
  fi
else
  case $1 in
  jax)
    build_and_push "utils/docker/Dockerfile.jax" "gcr.io/felafax-training/roadrunner-jax:latest_v2"
    ;;
  xla)
    build_and_push "utils/docker/Dockerfile.torch" "gcr.io/felafax-training/roadrunner-torchxla:latest_v2"
    ;;
  *)
    echo "Invalid argument. Please use 'jax' or 'xla', or no argument to build both."
    exit 1
    ;;
  esac

  if [ $? -ne 0 ]; then
    exit 1
  fi
fi

echo "Build and push process completed successfully"
