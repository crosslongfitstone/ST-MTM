# export CUDA_VISIBLE_DEVICES=0

python -u run.py \
    --task_name pretrain \
    --root_path datasets/ \
    --data_path Electricity.csv \
    --model_id STMTM \
    --model STMTM \
    --data Electricity \
    --features M \
    --seq_len 336 \
    --e_layers 2 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --n_heads 8 \
    --d_model 16 \
    --d_ff 16 \
    --kernel_size 100 \
    --seg_len 25 \
    --p_tmask 0.2 \
    --topk 3 \
    --learning_rate 0.001 \
    --batch_size 8 \
    --train_epochs 10 \
