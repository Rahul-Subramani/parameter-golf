#!/bin/bash
# MEDIUM VERSION: Proven improvements + moderate new additions
# Changes from SOTA with MEDIUM confidence:
#   1. AdamW for TTT (4 epochs, per-layer LR)
#   2. CROWN-Q penalty during warmdown
#   3. SWA/EMA 50/50 blend
#   4. Cosine warmdown 4000
#   5. BigramHash(2048)
# Omits risky changes: MTP, label smoothing, extra VE layers, magnitude pruning
# Expected: ~1.114-1.117 BPB

set -euo pipefail
SEED="${1:-1337}"
echo "=== MEDIUM Submission ==="
echo "Seed: $SEED"

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
VE_LAYERS=9,10 \
MTP_NUM_HEADS=0 \
MTP_LOSS_WEIGHT=0.0 \
LABEL_SMOOTHING=0.0 \
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
PRUNE_FRACTION=0.0 \
LZMA_PRESET=6 \
SEED=$SEED \
torchrun --standalone --nproc_per_node=8 submission/train_gpt.py
