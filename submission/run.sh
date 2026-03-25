#!/bin/bash
# Run training + LEGAL evaluation on 8xH100
# Target: < 1.115 BPB (beating SOTA 1.1194 by > 0.005 nats)
#
# Key innovations (all legal):
#   1. CROWN-Q quantization penalty during warmdown (PR #693: -0.003 pre-TTT)
#   2. Legal 4-epoch TTT with AdamW + per-layer LR (better than SOTA's 3ep SGD)
#   3. MTP auxiliary loss during training (better representations)
#   4. SWA/EMA 50/50 blend (PR #693: smoother weight distribution)
#   5. Mixed int5/int6 quantization + magnitude pruning + lzma-9
#   6. Cosine warmdown 4000 + BigramHash(2048) + VE(7,8,9,10)
#
# NOTE: Depth recurrence disabled by default (RECUR_LAYERS="") to keep
# eval budget under 600s. Enable with RECUR_LAYERS=4,5 if running
# without TTT or with fewer epochs.

set -euo pipefail

SEED="${1:-1337}"

echo "=== Parameter Golf Submission ==="
echo "Seed: $SEED"
echo ""

NUM_LAYERS=11 \
BIGRAM_VOCAB_SIZE=2048 \
XSA_LAST_N=4 \
SWA_ENABLED=1 \
SWA_EVERY=50 \
SWA_EMA_BLEND=0.5 \
ROPE_DIMS=16 \
LN_SCALE=1 \
LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 \
VE_DIM=128 \
VE_LAYERS=7,8,9,10 \
MTP_NUM_HEADS=1 \
MTP_LOSS_WEIGHT=0.1 \
LABEL_SMOOTHING=0.01 \
RECUR_LAYERS="" \
CROWNQ_LAMBDA=0.01 \
CROWNQ_WARMDOWN_ONLY=1 \
TTT_ENABLED=1 \
TTT_LR=0.001 \
TTT_EPOCHS=4 \
TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 \
TTT_BATCH_SEQS=32 \
TTT_GRAD_CLIP=1.0 \
TTT_WD=0.0 \
TTT_BETA1=0.9 \
TTT_BETA2=0.999 \
TTT_MLP_PROJ_LR_MUL=2.0 \
TTT_MLP_FC_LR_MUL=0.5 \
MUON_WD=0.04 \
ADAM_WD=0.04 \
MATRIX_LR=0.025 \
SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=4000 \
ITERATIONS=9000 \
MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=64 \
PRUNE_FRACTION=0.02 \
LZMA_PRESET=9 \
SEED=$SEED \
torchrun --standalone --nproc_per_node=8 submission/train_gpt.py

echo ""
echo "=== Done! Check logs/ for full output ==="
