
for pred_len in 96 192 336 720; do
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path datasets/ \
        --data_path ETTm1.csv \
        --model_id STMTM \
        --model STMTM \
        --data ETTm1 \
        --features M \
        --seq_len 336 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2  \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --n_heads 16 \
        --d_model 32 \
        --d_ff 64 \
        --kernel_size 100 \
        --seg_len 25 \
        --p_tmask 0.2 \
        --topk 3 \
        --learning_rate 0.0001 \
        --dropout 0.2 \
        --batch_size 16 \
        
done

