#!/bin/bash
. activate scientific-citation-detection

cd ..
run_name="transformer"
tag="transformer"
model_dir=models/transformer
python neural_baselines.py \
  --train_data data/train.jsonl \
  --validation_data data/dev.jsonl \
  --test_data data/test.jsonl \
  --run_name "${run_name}" \
  --tag "${tag}" \
  --model_name scratch_transformer \
  --model_dir ${model_dir} \
  --balance_class_weight \
  --n_gpu 1 \
  --batch_size 64 \
  --ff_dim 128 \
  --n_heads 3 \
  --n_layers 5 \
  --learning_rate 0.0001406 \
  --weight_decay 0.1 \
  --dropout_prob 0.4 \
  --n_epochs 33 \
  --seed 1000
