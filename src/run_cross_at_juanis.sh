DEVICE=1
FOLD=1
MODE="cross_juanis"
RUN_NAME=${MODE}_fold${FOLD}_miniVit_gate_head4

CUDA_VISIBLE_DEVICES=$DEVICE python -u main.py \
    --fold ${FOLD} \
    --mode ${MODE} \
    --run_name ${RUN_NAME} \
    --batch_size 64 \
    --wandb True \

DEVICE=1
FOLD=2
MODE="cross_juanis"
RUN_NAME=${MODE}_fold${FOLD}_miniVit_gate_head4

CUDA_VISIBLE_DEVICES=$DEVICE python -u main.py \
    --fold ${FOLD} \
    --mode ${MODE} \
    --run_name ${RUN_NAME} \
    --batch_size 64 \
    --wandb True \