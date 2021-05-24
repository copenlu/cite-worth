#!/bin/bash
. activate scientific-citation-detection

cd ..
run_name="longformer-solo"
tag="citeworth"
model_dir=models/longformer_solo
python neural_baselines.py \
  --train_data data/train.jsonl \
  --validation_data data/dev.jsonl \
  --test_data data/test.jsonl \
  --run_name "${run_name}" \
  --tag "${tag}" \
  --model_name allenai/longformer-base-4096 \
  --model_dir ${model_dir} \
  --balance_class_weight \
  --n_gpu 1 \
  --batch_size 4 \
  --learning_rate 0.000001351 \
  --warmup_steps 300 \
  --weight_decay 0.1 \
  --n_epochs 3 \
  --seed 1000
