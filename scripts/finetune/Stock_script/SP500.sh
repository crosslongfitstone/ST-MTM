

for pred_len in 96 192 336 720; do
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path datasets/ \
        --data_path SP500.csv \
        --model_id STMTM \
        --model STMTM \
        --data SP500 \
        --features M \
        --seq_len 192 \
        --e_layers 2 \
        --enc_in 1 \
        --dec_in 1 \
        --c_out 1 \
        --d_model 32 \
        --d_ff 32 \
        --n_heads 4 \
        --learning_rate 0.0001 \
        --batch_size 32 \
        --dropout 0.2 \
        --kernel_size 100 \
        --seg_len 10 \
        --p_tmask 0.2 \
        --topk 3 \

done


    