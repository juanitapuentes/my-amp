DEVICE=0
FOLD=1
MODE="cross_juanis"
RUN_NAME=${MODE}_fold${FOLD}_best_newVocab2

CUDA_VISIBLE_DEVICES=$DEVICE python -u main.py \
    --fold ${FOLD} \
    --mode ${MODE} \
    --run_name ${RUN_NAME} \
    --batch_size 1024 \
    --wandb True \