#!/bin/bash
# SAFE VERSION: Minimal changes from proven SOTA (1.1194 BPB)
# Only changes with HIGH confidence of improvement:
#   1. AdamW for TTT instead of SGD (well-understood improvement)
#   2. 4 epochs instead of 3 (small, safe increase)
#   3. Per-layer TTT LR (proj 2x, fc 0.5x)
# Everything else IDENTICAL to SOTA PR #549 defaults.
# Expected: ~1.116-1.118 BPB (modest but safe improvement)

set -euo pipefail
SEED="${1:-1337}"
echo "=== SAFE Submission (minimal changes from SOTA) ==="
echo "Seed: $SEED"

NUM_LAYERS=11 \
BIGRAM_VOCAB_SIZE=1536 \
XSA_LAST_N=4 \
SWA_ENABLED=1 \
SWA_EVERY=50 \
SWA_EMA_BLEND=0.0 \
ROPE_DIMS=16 \
LN_SCALE=1 \
LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 \
VE_DIM=128 \
VE_LAYERS=9,10 \
MTP_NUM_HEADS=0 \
MTP_LOSS_WEIGHT=0.0 \
LABEL_SMOOTHING=0.0 \
RECUR_LAYERS="" \
CROWNQ_LAMBDA=0.0 \
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
WARMDOWN_ITERS=3500 \
ITERATIONS=9000 \
MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=64 \
PRUNE_FRACTION=0.0 \
LZMA_PRESET=6 \
SEED=$SEED \
torchrun --standalone --nproc_per_node=8 submission/train_gpt.py
