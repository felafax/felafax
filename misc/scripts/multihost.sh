#!/bin/bash

# Check if required arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <tpu-name> <zone>"
    echo "Example: $0 eager-hawk-13 europe-west4-b"
    exit 1
fi

TPU_NAME="$1"
ZONE="$2"
PROJECT="felafax-training"
CONTAINER_NAME="felafax-tunerx-container"
TARGET_DIR="/roadrunner-fork"

echo "Copying files from: ./llama3_jax/"

# Copy files to all TPU VM workers and then move them into the container
gcloud compute tpus tpu-vm scp --recurse ./llama3_jax/* ${TPU_NAME} \
--project=${PROJECT} \
--zone=${ZONE} \
--worker=all \
&& gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
--project=${PROJECT} \
--zone=${ZONE} \
--worker=all \
--command="sudo docker cp /home/\${USER}/llama3_jax ${CONTAINER_NAME}:${TARGET_DIR}/"
