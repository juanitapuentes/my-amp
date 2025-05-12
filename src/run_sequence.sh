DEVICE=2
FOLD=1
MODE="sequence"
RUN_NAME=${MODE}_fold${FOLD}

CUDA_VISIBLE_DEVICES=$DEVICE python -u main.py \
    --fold ${FOLD} \
    --mode ${MODE} \
    --run_name ${RUN_NAME} \
    --batch_size 64 \

DEVICE=2
FOLD=2
MODE="sequence"
RUN_NAME=${MODE}_fold${FOLD}

CUDA_VISIBLE_DEVICES=$DEVICE python -u main.py \
    --fold ${FOLD} \
    --mode ${MODE} \
    --run_name ${RUN_NAME} \
    --batch_size 64 \