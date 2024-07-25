#!/bin/bash

# Configuration
PROJECT_NAME="llama3-tunerx"  # Changed to lowercase
PROJECT_ID=$(gcloud config get-value project)
ZONE="europe-west4-b"
ACCELERATOR_TYPE="v5p-8"
TPU_VERSION="tpu-vm-tf-2.16.1-pod-pjrt"
IMAGE_NAME="gcr.io/felafax-training/tunerx-base-v5:latest"
CONTAINER_NAME="tunerx-base-container"
JUPYTER_PORT="8888"
PERSISTENT_DISK_SIZE="2000GB"
PERSISTENT_DISK_TYPE="pd-balanced"

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to create a valid name
create_valid_name() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | sed -e 's/[^a-z0-9-]/-/g' -e 's/^[^a-z]*//' -e 's/-*$//' | cut -c1-63
}

# Check if TPU name is provided as an argument
if [ $# -eq 1 ]; then
  TPU_NAME=$(create_valid_name "$1")
  echo -e "${YELLOW}Using existing TPU: $TPU_NAME${NC}"
else
  # Generate TPU name with timestamp
  TIMESTAMP=$(date +%Y%m%d-%H%M%S)
  TPU_NAME=$(create_valid_name "${PROJECT_NAME}-${TIMESTAMP}")
  PERSISTENT_DISK_NAME="${TPU_NAME}-disk"

  echo -e "${GREEN}Creating Persistent Disk...${NC}"
  gcloud compute disks create $PERSISTENT_DISK_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --size=$PERSISTENT_DISK_SIZE \
    --type=$PERSISTENT_DISK_TYPE

  if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to create persistent disk. Exiting.${NC}"
    exit 1
  fi

  echo -e "${GREEN}Creating TPU VM...${NC}"
  gcloud compute tpus tpu-vm create "$TPU_NAME" \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --accelerator-type=$ACCELERATOR_TYPE \
    --version=$TPU_VERSION \
    --data-disk="source=projects/$PROJECT_ID/zones/$ZONE/disks/$PERSISTENT_DISK_NAME,mode=read-write"

  if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to create TPU VM. Exiting.${NC}"
    exit 1
  fi
fi

echo -e "${GREEN}Connecting to TPU VM, cleaning up, and starting Docker container on all workers...${NC}"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
--zone=${ZONE} \
--project=${PROJECT_ID} \
--worker=all \
--command="
    sudo docker stop $CONTAINER_NAME 2>/dev/null
    sudo docker rm $CONTAINER_NAME 2>/dev/null

    sudo docker pull $IMAGE_NAME
    if [ \$? -ne 0 ]; then
      echo 'Failed to pull Docker image.'
      exit 1
    fi

    DISK_PATH=$(readlink -f /dev/disk/by-id/google-persistent-disk-1)
    sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard $DISK_PATH
    sudo mkdir -p /mnt/persistent-disk
    sudo mount -o discard,defaults $DISK_PATH /mnt/persistent-disk
    sudo docker run -d --rm --net=host --shm-size=16G --name $CONTAINER_NAME --privileged -v /mnt/persistent-disk:/mnt/persistent-disk $IMAGE_NAME
    if [ \$? -ne 0 ]; then
      echo 'Failed to start Docker container.'
      exit 1
    fi
    if [ \$? -ne 0 ]; then
      echo 'Failed to start Docker container.'
      exit 1
    fi

    sudo docker exec $CONTAINER_NAME /bin/bash -c '
      pip install torch~=2.3.0 torch_xla[tpu]~=2.3.0 torchvision -f https://storage.googleapis.com/libtpu-releases/index.html
      apt update && apt install -y vim
      echo \"export PJRT_DEVICE=TPU\" >> /root/.bashrc
    '

    sleep 10
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