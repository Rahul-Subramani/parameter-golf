#!/bin/bash
# =============================================================================
# Complete GCP Setup for Parameter Golf (8xH100 SXM)
# =============================================================================
# COST: ~$30-40/hour for a3-highgpu-8g. Budget ~$20-40 total for 3 runs.
#
# Usage:
#   1. gcloud auth login
#   2. export GCP_PROJECT=your-project-id
#   3. bash submission/gcp_setup.sh
#
# This script:
#   - Creates an 8xH100 VM with PyTorch + CUDA pre-installed
#   - Installs all dependencies (FlashAttention 3, zstandard, etc.)
#   - Downloads the FineWeb dataset
#   - Uploads your submission code
#   - Ready to train in ~10-15 minutes
# =============================================================================

set -euo pipefail

# --- Configuration ---
PROJECT="${GCP_PROJECT:?ERROR: Set GCP_PROJECT first. Run: export GCP_PROJECT=your-project-id}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE="${GCP_INSTANCE:-param-golf}"
MACHINE="a3-highgpu-8g"

echo "============================================"
echo " Parameter Golf - GCP Setup"
echo "============================================"
echo " Project:  $PROJECT"
echo " Zone:     $ZONE"
echo " Instance: $INSTANCE"
echo " Machine:  $MACHINE (8x H100 80GB SXM)"
echo " Cost:     ~\$30-40/hour"
echo "============================================"
echo ""

# --- Step 1: Check quota ---
echo "[1/6] Checking GPU quota..."
QUOTA=$(gcloud compute regions describe "${ZONE%-*}" --project="$PROJECT" \
  --format="value(quotas[name=NVIDIA_H100_GPUS].limit)" 2>/dev/null || echo "0")
echo "  H100 quota in ${ZONE%-*}: ${QUOTA:-unknown}"
if [ "${QUOTA:-0}" = "0" ] || [ "${QUOTA:-0}" -lt 8 ]; then
  echo ""
  echo "  WARNING: You may not have enough H100 quota (need 8)."
  echo "  Request quota increase at:"
  echo "  https://console.cloud.google.com/iam-admin/quotas?project=$PROJECT"
  echo "  Filter for 'NVIDIA H100' in region ${ZONE%-*}"
  echo ""
  read -p "  Continue anyway? (y/n) " -n 1 -r
  echo
  [[ $REPLY =~ ^[Yy]$ ]] || exit 1
fi

# --- Step 2: Create VM ---
echo "[2/6] Creating VM (takes 2-5 minutes)..."
if gcloud compute instances describe "$INSTANCE" --zone="$ZONE" --project="$PROJECT" &>/dev/null; then
  echo "  Instance '$INSTANCE' already exists. Skipping creation."
else
  gcloud compute instances create "$INSTANCE" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --machine-type="$MACHINE" \
    --image-family="pytorch-latest-gpu" \
    --image-project="deeplearning-platform-release" \
    --boot-disk-size="200GB" \
    --boot-disk-type="pd-ssd" \
    --accelerator="type=nvidia-h100-80gb,count=8" \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True" \
    --scopes="https://www.googleapis.com/auth/cloud-platform"
  echo "  Waiting for VM to boot..."
  sleep 45
fi

# --- Step 3: Install dependencies ---
echo "[3/6] Installing dependencies on VM..."
gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT" --command='bash -s' << 'SETUP'
set -euo pipefail

echo "--- Installing Python packages ---"
pip install --upgrade pip
pip install torch numpy sentencepiece huggingface-hub datasets tqdm zstandard lzma

# FlashAttention 3 (critical for H100 performance)
echo "--- Installing FlashAttention ---"
pip install flash-attn --no-build-isolation 2>/dev/null || echo "flash-attn install failed, will use fallback"
pip install flash-attn-interface 2>/dev/null || echo "flash-attn-interface not available, will use F.sdpa fallback"

echo "--- Verifying GPU setup ---"
nvidia-smi
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
try:
    from flash_attn_interface import flash_attn_func
    print('FlashAttention 3: OK')
except ImportError:
    print('FlashAttention 3: NOT AVAILABLE (will use fallback)')
"
SETUP

# --- Step 4: Clone repo + upload submission ---
echo "[4/6] Setting up repository..."
gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT" --command='bash -s' << 'REPO'
cd ~
if [ ! -d "parameter-golf" ]; then
  git clone https://github.com/openai/parameter-golf.git
fi
mkdir -p ~/parameter-golf/submission
REPO

echo "  Uploading submission code..."
gcloud compute scp --zone="$ZONE" --project="$PROJECT" --recurse \
  submission/train_gpt.py \
  submission/run.sh \
  submission/run_safe.sh \
  submission/run_medium.sh \
  "$INSTANCE":~/parameter-golf/submission/

# --- Step 5: Download dataset ---
echo "[5/6] Downloading FineWeb dataset (this takes 5-10 minutes)..."
gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT" --command='bash -s' << 'DATA'
cd ~/parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024
echo "Dataset files:"
ls -lh data/datasets/fineweb10B_sp1024/ | head -5
echo "..."
ls data/datasets/fineweb10B_sp1024/ | wc -l
echo "total files"
DATA

# --- Step 6: Ready! ---
echo ""
echo "============================================"
echo " SETUP COMPLETE!"
echo "============================================"
echo ""
echo " SSH into your VM:"
echo "   gcloud compute ssh $INSTANCE --zone=$ZONE --project=$PROJECT"
echo ""
echo " Then run (from ~/parameter-golf/):"
echo ""
echo "   # SAFE version (minimal changes, lowest risk):"
echo "   bash submission/run_safe.sh 1337"
echo ""
echo "   # MEDIUM version (moderate improvements):"
echo "   bash submission/run_medium.sh 1337"
echo ""
echo "   # AGGRESSIVE version (all improvements):"
echo "   bash submission/run.sh 1337"
echo ""
echo " Each run takes ~20 minutes (10 train + 10 eval)."
echo " Run all 3 tiers with seed 1337, then pick the best."
echo ""
echo " COST WARNING: ~\$30-40/hour!"
echo " STOP when done:"
echo "   gcloud compute instances stop $INSTANCE --zone=$ZONE --project=$PROJECT"
echo " DELETE when fully done:"
echo "   gcloud compute instances delete $INSTANCE --zone=$ZONE --project=$PROJECT"
echo "============================================"
