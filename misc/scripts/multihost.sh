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
TARGET_DIR="/home/llama3_jax/"

# Delete existing directory if it exists
echo "Deleting existing directory if it exists..."
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --worker=all \
  --command="rm -rf /home/${USER}/llama3_jax"

echo "Copying files from: ./llama3_jax/"
# Copy files to all TPU VM workers and then move them into the container
gcloud compute tpus tpu-vm scp --recurse ./llama3_jax/* "${TPU_NAME}:/home/${USER}/llama3_jax/" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --worker=all \
&& gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --worker=all \
  --command="sudo docker cp /home/\${USER}/llama3_jax ${CONTAINER_NAME}:${TARGET_DIR}/"

echo "Installing dependencies..."
PIP_INSTALL_CMD="cd ${TARGET_DIR} && pip install -r requirements.txt"
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --worker=all \
  --command="sudo docker exec ${CONTAINER_NAME} bash -c \"${PIP_INSTALL_CMD}\""

echo "Running training..."
TRAIN_CMD="cd ${TARGET_DIR} && python multihost_trainer.py --model_name 'colab-llama-3.1-8B-Instruct-JAX' --data_source 'yahma/alpaca-cleaned' --train_and_export"

gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --worker=all \
  --command="sudo docker exec ${CONTAINER_NAME} bash -c \"${TRAIN_CMD}\""
