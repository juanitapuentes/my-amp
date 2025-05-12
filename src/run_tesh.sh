DEVICE=3
MODE="cross_mini_juanis"
MODEL1="/home/bcv_researcher/merged_disk2/amp/AMP_Former/src/outputs/cross_mini_juanis/1/model_cross_mini_juanis_Fold1_cross_mini_juanis_fold1_nopretrain_epoch130.pth"
MODEL2="/home/bcv_researcher/merged_disk2/amp/AMP_Former/src/outputs/cross_mini_juanis/2/model_cross_mini_juanis_Fold2_cross_mini_juanis_fold2_nopretrain_epoch130.pth"

CUDA_VISIBLE_DEVICES=$DEVICE python -u test.py \
    --mode ${MODE} \
    --batch_size 64 \
    --model_fold1 ${MODEL1} \
    --model_fold2 ${MODEL2} \
