DEVICE=3
MODE="cross_juanis"
MODEL1="/home/bcv_researcher/merged_disk2/amp/AMP_Former/src/outputs/cross_juanis/1/cross_juanis_fold1_miniVit_gate_head4/modelFINAL_cross_juanis_fold1_miniVit_gate_head4.pth"
MODEL2="/home/bcv_researcher/merged_disk2/amp/AMP_Former/src/outputs/cross_juanis/2/cross_juanis_fold2_miniVit_gate_head4/modelFINAL_cross_juanis_fold2_miniVit_gate_head4.pth"

CUDA_VISIBLE_DEVICES=$DEVICE python -u test.py \
    --mode ${MODE} \
    --batch_size 64 \
    --model_fold1 ${MODEL1} \
    --model_fold2 ${MODEL2} \
    --seq_n_heads 4 \
