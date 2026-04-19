#!/bin/bash
# ============================================================
# iTransformer — BTC 5 分鐘 K 線價格方向預測訓練腳本
# 與 1h 版差異：--freq t, --seq_len 576 (2天), 較小 batch_size
# ============================================================
export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/stock/ \
  --data_path stock_features.csv \
  --model_id BTC_5min_v2 \
  --model $model_name \
  --data custom \
  --features MS \
  --target target \
  --freq t \
  --seq_len 576 \
  --label_len 48 \
  --pred_len 1 \
  --enc_in 35 \
  --dec_in 35 \
  --c_out 1 \
  --e_layers 2 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 8 \
  --dropout 0.2 \
  --batch_size 64 \
  --patience 15 \
  --train_epochs 100 \
  --learning_rate 0.0001 \
  --lradj type3 \
  --focal_alpha 0.93 \
  --des BTC_5min_v2 \
  --itr 1
