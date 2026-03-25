#!/bin/bash
# =============================================================================
# GCP Setup for Parameter Golf (8xH100)
# =============================================================================
# Usage:
#   1. Set your GCP project: export GCP_PROJECT=your-project-id
#   2. Run: bash setup_gcp.sh
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - Sufficient quota for A3 instances in your region
#   - GPU quota for 8x H100 in the selected zone
# =============================================================================

set -euo pipefail

# Configuration
PROJECT="${GCP_PROJECT:?Set GCP_PROJECT env var}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE_NAME="${GCP_INSTANCE:-param-golf-8xh100}"
MACHINE_TYPE="a3-highgpu-8g"  # 8x H100 80GB SXM
IMAGE_FAMILY="pytorch-latest-gpu"
IMAGE_PROJECT="deeplearning-platform-release"
BOOT_DISK_SIZE="200GB"
BOOT_DISK_TYPE="pd-ssd"

echo "=== Parameter Golf GCP Setup ==="
echo "Project:  $PROJECT"
echo "Zone:     $ZONE"
echo "Instance: $INSTANCE_NAME"
echo "Machine:  $MACHINE_TYPE (8x H100 80GB)"
echo ""

# Check if instance already exists
if gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT" &>/dev/null; then
    echo "Instance '$INSTANCE_NAME' already exists."
    echo "To SSH: gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT"
    echo "To delete: gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --project=$PROJECT"
    exit 0
fi

echo "Creating instance (this takes 2-5 minutes)..."
gcloud compute instances create "$INSTANCE_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --image-family="$IMAGE_FAMILY" \
    --image-project="$IMAGE_PROJECT" \
    --boot-disk-size="$BOOT_DISK_SIZE" \
    --boot-disk-type="$BOOT_DISK_TYPE" \
    --accelerator="type=nvidia-h100-80gb,count=8" \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True" \
    --scopes="https://www.googleapis.com/auth/cloud-platform"

echo ""
echo "Instance created. Waiting for SSH to become available..."
sleep 30

# Install dependencies and clone repo
echo "Setting up environment..."
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT" --command='bash -s' << 'REMOTE_SETUP'
set -euo pipefail

echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install torch numpy sentencepiece huggingface-hub datasets tqdm zstandard

echo "=== Cloning parameter-golf ==="
cd ~
if [ ! -d "parameter-golf" ]; then
    git clone https://github.com/openai/parameter-golf.git
fi
cd parameter-golf

echo "=== Downloading dataset ==="
pip install -r requirements.txt
python data/cached_challenge_fineweb.py

echo "=== Verifying GPU setup ==="
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

echo "=== Setup complete! ==="
echo "To run training: torchrun --standalone --nproc_per_node=8 submission/train_gpt.py"
REMOTE_SETUP

echo ""
echo "=== DONE ==="
echo "SSH into instance:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT"
echo ""
echo "Run training:"
echo "  cd ~/parameter-golf && torchrun --standalone --nproc_per_node=8 submission/train_gpt.py"
echo ""
echo "COST WARNING: a3-highgpu-8g costs ~\$30-40/hr. Remember to stop/delete when done:"
echo "  gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE --project=$PROJECT"
