DEVICE=3
FOLD=1
MODE="concat_juanis"
RUN_NAME=${MODE}_fold${FOLD}_allImage_head4

CUDA_VISIBLE_DEVICES=$DEVICE python -u main.py \
    --fold ${FOLD} \
    --mode ${MODE} \
    --run_name ${RUN_NAME} \
    --batch_size 64 \
    --wandb True \
    --seq_n_heads 4 \

DEVICE=3
FOLD=2
MODE="concat_juanis"
RUN_NAME=${MODE}_fold${FOLD}_allImage_head4

CUDA_VISIBLE_DEVICES=$DEVICE python -u main.py \
    --fold ${FOLD} \
    --mode ${MODE} \
    --run_name ${RUN_NAME} \
    --batch_size 64 \
    --wandb True \
    --seq_n_heads 4 \