#!/usr/bin/env bash

# run_pretrain_mae.sh
# Usage: ./run_pretrain_mae.sh /path/to/maps_dir

MAPS_DIR=${1:-"/home/bcv_researcher/merged_disk2/amp/Matriz_Distancias/Distance_Maps"}   # pass maps_dir as first arg, or edit default
IMG_SIZE=224
PATCH_SIZE=16
D_MODEL=256
N_HEADS=8
ENC_LAYERS=6
DEC_LAYERS=4
MASK_RATIO=0.75
BATCH_SIZE=64
EPOCHS=50
LR=1e-4
DEVICE="cuda"
OUT_PATH="mae_struct_encoder.pth"

python mae.py \
  --maps_dir "$MAPS_DIR" \
  --img_size $IMG_SIZE \
  --patch_size $PATCH_SIZE \
  --d_model $D_MODEL \
  --n_heads $N_HEADS \
  --enc_layers $ENC_LAYERS \
  --dec_layers $DEC_LAYERS \
  --mask_ratio $MASK_RATIO \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --lr $LR \
  --device $DEVICE \
  --out_path $OUT_PATH