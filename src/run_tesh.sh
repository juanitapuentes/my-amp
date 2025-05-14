DEVICE=3
MODE="cross_juanis"
MODEL1="/home/bcv_researcher/merged_disk2/amp/AMP_Former/src/outputs/cross_juanis/1/cross_juanis_fold1_best_newVocab/model_cross_juanis_cross_juanis_fold1_best_newVocab_epoch70.pth"
MODEL2="/home/bcv_researcher/merged_disk2/amp/AMP_Former/src/outputs/cross_juanis/2/cross_juanis_fold2_best_newVocab/model_cross_juanis_cross_juanis_fold2_best_newVocab_epoch70.pth"

CUDA_VISIBLE_DEVICES=$DEVICE python -u test.py \
    --mode ${MODE} \
    --batch_size 64 \
    --model_fold1 ${MODEL1} \
    --model_fold2 ${MODEL2} \
