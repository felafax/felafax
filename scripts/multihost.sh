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
TARGET_DIR="/home/felafax/"

# Delete existing directory if it exists on the TPU VM
echo "Deleting existing directory if it exists..."
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --worker=all \
  --command="rm -rf /home/${USER}/felafax"

# Copy src/ and trainers/ directories to all TPU VM workers
echo "Copying files from src/ and trainers/ to the TPU VM..."
gcloud compute tpus tpu-vm scp --recurse ./src ./trainers requirements.txt "${TPU_NAME}:/home/${USER}/felafax/" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --worker=all

# Copy the files into the Docker container
echo "Copying files into the Docker container..."
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --worker=all \
  --command="sudo docker cp /home/\${USER}/felafax ${CONTAINER_NAME}:${TARGET_DIR}"

# Install dependencies inside the Docker container
echo "Installing dependencies..."
PIP_INSTALL_CMD="cd ${TARGET_DIR} && pip install -r requirements.txt"
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --worker=all \
  --command="sudo docker exec ${CONTAINER_NAME} bash -c \"${PIP_INSTALL_CMD}\""

# Run the training script inside the Docker container
echo "Running training..."
TRAIN_CMD="cd ${TARGET_DIR} && python -m trainers.llama3_alpaca_finetune.pipeline"

gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --worker=all \
  --command="sudo docker exec ${CONTAINER_NAME} bash -c \"${TRAIN_CMD}\""