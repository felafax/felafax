#!/bin/bash

# Configuration
PROJECT_NAME="llama3-tunerx"
PROJECT_ID=$(gcloud config get-value project)
ZONE="europe-west4-b"  # "us-central1-a" # "europe-west4-b"
ACCELERATOR_TYPE="v5p-8"  # "us-central1-a" # "europe-west4-b"
TPU_VERSION="tpu-vm-tf-2.16.1-pod-pjrt"
IMAGE_NAME="gcr.io/felafax-training/tunerx-base-v5:latest"
CONTAINER_NAME="tunerx-base-container"
JUPYTER_PORT="8888"
PERSISTENT_DISK_SIZE="200GB"
PERSISTENT_DISK_TYPE="pd-balanced"
PERSISTENT_DISK_NAME=""

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to create a valid name
create_valid_name() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | sed -e 's/[^a-z0-9-]/-/g' -e 's/^[^a-z]*//' -e 's/-*$//' | cut -c1-63
}

# Function to echo with color
echo_color() {
    local color=$1
    shift
    echo -e "${color}$*${NC}"
}

# Function to run gcloud command and check for errors
run_gcloud_command() {
    if ! "$@"; then
        echo_color $RED "Failed to execute gcloud command. Exiting."
        exit 1
    fi
}

setup_new_tpu() {
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    TPU_NAME=$(create_valid_name "${PROJECT_NAME}-${TIMESTAMP}")
    PERSISTENT_DISK_NAME="${PERSISTENT_DISK_NAME:-${TPU_NAME}-disk}"

    echo_color $GREEN "Checking if Persistent Disk exists..."
    if ! gcloud compute disks describe $PERSISTENT_DISK_NAME --project=$PROJECT_ID --zone=$ZONE &>/dev/null; then
        echo_color $GREEN "Creating Persistent Disk..."
        run_gcloud_command gcloud compute disks create $PERSISTENT_DISK_NAME \
            --project=$PROJECT_ID \
            --zone=$ZONE \
            --size=$PERSISTENT_DISK_SIZE \
            --type=$PERSISTENT_DISK_TYPE
    else
        echo_color $YELLOW "Persistent Disk $PERSISTENT_DISK_NAME already exists. Skipping creation."
    fi

    echo_color $GREEN "Creating TPU VM..."
    run_gcloud_command gcloud compute tpus tpu-vm create "$TPU_NAME" \
        --project=$PROJECT_ID \
        --zone=$ZONE \
        --accelerator-type=$ACCELERATOR_TYPE \
        --version=$TPU_VERSION \
        --data-disk="source=projects/$PROJECT_ID/zones/$ZONE/disks/$PERSISTENT_DISK_NAME,mode=read-write"
}

start_docker_container() {
    echo_color $GREEN "Connecting to TPU VM, cleaning up, and starting Docker container on all workers..."
    run_gcloud_command gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
        --zone=${ZONE} \
        --project=${PROJECT_ID} \
        --worker=all \
        --command="
        set -e
        sudo docker stop $CONTAINER_NAME 2>/dev/null || true
        sudo docker rm $CONTAINER_NAME 2>/dev/null || true
        sudo docker pull $IMAGE_NAME

        DISK_PATH=\$(readlink -f /dev/disk/by-id/google-persistent-disk-1)
        echo \"Disk path: \$DISK_PATH\"
        
        sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard \$DISK_PATH
        sudo mkdir -p /mnt/persistent-disk
        sudo mount -o discard,defaults \$DISK_PATH /mnt/persistent-disk
        
        echo \"Disk mounted successfully\"
        ls -l /mnt/persistent-disk
        
        sudo docker run -d --rm --net=host --shm-size=16G --name $CONTAINER_NAME --privileged -v /mnt/persistent-disk:/mnt/persistent-disk $IMAGE_NAME
        
        echo \"Docker container started successfully\"
        sudo docker ps
        
        sudo docker exec $CONTAINER_NAME /bin/bash -c '
            pip install torch~=2.3.0 torch_xla[tpu]~=2.3.0 torchvision -f https://storage.googleapis.com/libtpu-releases/index.html
            apt update && apt install -y vim
            echo \"export PJRT_DEVICE=TPU\" >> /root/.bashrc
            rm -rf /home/tunerX/
            cd /home/
            git clone https://github.com/felafax/RoadrunnerX.git
        '
        
        sleep 10
        sudo docker logs $CONTAINER_NAME
        "
}

setup_port_forwarding() {
    echo_color $GREEN "Setting up port forwarding for JupyterLab on the first worker..."
    echo_color $YELLOW "Please keep this terminal open to maintain the connection."
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=0 -- -L "$JUPYTER_PORT:localhost:8888" > /dev/null 2>&1 &

    echo_color $YELLOW "To reconnect later, use the following command:"
    echo "gcloud compute tpus tpu-vm ssh --zone \"$ZONE\" \"$TPU_NAME\" --project \"$PROJECT_ID\" --worker=0 -- -fNT -L $JUPYTER_PORT:localhost:8888 > /dev/null 2>&1 &"
    echo_color $GREEN "Now, open JupyterLab at http://localhost:$JUPYTER_PORT"
}


# Main script logic
main() {
    if [ $# -eq 1 ]; then
        TPU_NAME=$(create_valid_name "$1")
        echo_color $YELLOW "Using existing TPU: $TPU_NAME"
    else
        setup_new_tpu
    fi

    start_docker_container
    setup_port_forwarding
}

main "$@"