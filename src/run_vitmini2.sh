DEVICE=3
FOLD=2
MODE="cross_juanis"
RUN_NAME=${MODE}_fold${FOLD}_tinyVIT

CUDA_VISIBLE_DEVICES=$DEVICE python -u main.py \
    --fold ${FOLD} \
    --mode ${MODE} \
    --run_name ${RUN_NAME} \
    --batch_size 64 \
    --wandb True \
