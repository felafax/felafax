#!/bin/bash

# Configuration
PROJECT_NAME="felafax-tunerx"
PROJECT_ID="felafax-training"
ZONE="us-central1-a"
ACCELERATOR_TYPE="v3-8"
TPU_VERSION="tpu-vm-tf-2.16.1-pod-pjrt"
IMAGE_NAME="felafax-tunerx:latest"
CONTAINER_NAME="felafax-tunerx-container"
JUPYTER_PORT="8888"

# Generate TPU name with timestamp
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
# TPU_NAME="${PROJECT_NAME}-${TIMESTAMP}"
TPU_NAME="felafax-tunerx-20240715-111723"

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Creating TPU VM...${NC}"
gcloud compute tpus tpu-vm create "$TPU_NAME" \
  --zone="$ZONE" \
  --accelerator-type="$ACCELERATOR_TYPE" \
  --version="$TPU_VERSION"

echo -e "${GREEN}Copying Dockerfile and requirements.txt to TPU VM...${NC}"
gcloud compute tpus tpu-vm scp Dockerfile "$TPU_NAME":~ --zone="$ZONE"
gcloud compute tpus tpu-vm scp requirements.txt "$TPU_NAME":~ --zone="$ZONE"

echo -e "${GREEN}Connecting to TPU VM and building Docker image...${NC}"
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" <<EOF
  echo -e "${GREEN}Building Docker image...${NC}"
  sudo docker build -t "$IMAGE_NAME" .
  if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to build Docker image. Please check your Dockerfile and requirements.txt.${NC}"
    exit 1
  fi

  echo -e "${GREEN}Running Docker container with JupyterLab...${NC}"
  sudo docker run -d --rm --net=host --name "$CONTAINER_NAME" --privileged "$IMAGE_NAME"
  if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to start Docker container. Please check the previous error messages.${NC}"
    exit 1
  fi

  echo -e "${YELLOW}JupyterLab is starting. Please wait...${NC}"
  sleep 10 # Give JupyterLab some time to start

  echo -e "${GREEN}JupyterLab logs:${NC}"
  sudo docker logs "$CONTAINER_NAME"
  exit
EOF

if [ $? -ne 0 ]; then
  echo -e "${RED}An error occurred during TPU VM setup. Please check the error messages above.${NC}"
  exit 1
fi

echo -e "${GREEN}Setting up port forwarding for JupyterLab...${NC}"
echo -e "${YELLOW}Please keep this terminal open to maintain the connection.${NC}"
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" -- -L "$JUPYTER_PORT:localhost:$JUPYTER_PORT"

echo -e "${GREEN}Script completed. You can now access JupyterLab at http://localhost:$JUPYTER_PORT${NC}"
echo -e "${YELLOW}To reconnect later, use the following command:${NC}"
echo -e "gcloud compute tpus tpu-vm ssh --zone \"$ZONE\" \"$TPU_NAME\" --project \"$PROJECT_ID\" -- -L $JUPYTER_PORT:localhost:$JUPYTER_PORT"
