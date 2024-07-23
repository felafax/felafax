#!/bin/bash

# Configuration
PROJECT_NAME="LLaMa3-tunerX"
PROJECT_ID=$(gcloud config get-value project)
ZONE="europe-west4-b"
ACCELERATOR_TYPE="v5p-8"
TPU_VERSION="tpu-vm-tf-2.16.1-pod-pjrt"
IMAGE_NAME="gcr.io/felafax-training/tunerx-base-v5:latest"
CONTAINER_NAME="tunerx-base-container"
JUPYTER_PORT="8888"

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if TPU name is provided as an argument
if [ $# -eq 1 ]; then
  TPU_NAME=$1
  echo -e "${YELLOW}Using existing TPU: $TPU_NAME${NC}"
else
  # Generate TPU name with timestamp
  TIMESTAMP=$(date +%Y%m%d-%H%M%S)
  TPU_NAME="${PROJECT_NAME}-${TIMESTAMP}"

  echo -e "${GREEN}Creating TPU VM...${NC}"
  gcloud compute tpus tpu-vm create "$TPU_NAME" \
    --zone="$ZONE" \
    --accelerator-type="$ACCELERATOR_TYPE" \
    --version="$TPU_VERSION"
fi

echo -e "${GREEN}Connecting to TPU VM, cleaning up, and starting Docker container on all workers...${NC}"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
--zone=${ZONE} \
--project=${PROJECT_ID} \
--worker=all \
--command="
    echo -e '${YELLOW}Cleaning up existing Docker container...${NC}'
    sudo docker stop $CONTAINER_NAME 2>/dev/null
    sudo docker rm $CONTAINER_NAME 2>/dev/null

    echo -e '${GREEN}Pulling Docker image...${NC}'
    sudo docker pull $IMAGE_NAME
    if [ \$? -ne 0 ]; then
      echo -e '${RED}Failed to pull Docker image. Please check if the image exists and you have the necessary permissions.${NC}'
      exit 1
    fi

    echo -e '${GREEN}Running Docker container...${NC}'
    sudo docker run -d --rm --net=host --shm-size=16G --name $CONTAINER_NAME --privileged $IMAGE_NAME
    if [ \$? -ne 0 ]; then
      echo -e '${RED}Failed to start Docker container. Please check the previous error messages.${NC}'
      exit 1
    fi

    echo -e '${GREEN}Setting up environment and installing packages in the container...${NC}'
    sudo docker exec $CONTAINER_NAME /bin/bash -c '
      pip install torch~=2.3.0 torch_xla[tpu]~=2.3.0 torchvision -f https://storage.googleapis.com/libtpu-releases/index.html
      apt update && apt install -y vim
      echo \"export PJRT_DEVICE=TPU\" >> /root/.bashrc
    '

    echo -e '${YELLOW}JupyterLab is starting. Please wait...${NC}'
    sleep 10
    echo -e '${GREEN}JupyterLab logs:${NC}'
    sudo docker logs $CONTAINER_NAME
  "

if [ $? -ne 0 ]; then
  echo -e "${RED}An error occurred during TPU VM setup. Please check the error messages above.${NC}"
  exit 1
fi

echo -e "${GREEN}Setting up port forwarding for JupyterLab on the first worker...${NC}"
echo -e "${YELLOW}Please keep this terminal open to maintain the connection.${NC}"
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=0 -- -L "$JUPYTER_PORT:localhost:$JUPYTER_PORT" 

echo -e "${GREEN}Script completed. You can now access JupyterLab at http://localhost:$JUPYTER_PORT${NC}"
echo -e "${YELLOW}To reconnect later, use the following command:${NC}"
echo -e "nohup gcloud compute tpus tpu-vm ssh --zone \"$ZONE\" \"$TPU_NAME\" --project \"$PROJECT_ID\" --worker=0 -- -fNT -L $JUPYTER_PORT:localhost:$JUPYTER_PORT > /dev/null 2>&1 &"

