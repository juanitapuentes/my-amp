DEVICE=2
FOLD=1
MODE="joint_fusion"
RUN_NAME=${MODE}_fold${FOLD}_joint_fusion

CUDA_VISIBLE_DEVICES=$DEVICE python -u main.py \
    --fold ${FOLD} \
    --mode ${MODE} \
    --run_name ${RUN_NAME} \
    --batch_size 64 \
    --wandb True \