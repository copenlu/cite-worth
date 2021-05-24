#!/bin/bash
. activate scientific-citation-detection

cd ..
wandb login 753c1fc1ed65c854b08c2b76fe8002cf59caf983
run_name="scibert unbalanced"
tag="citeworth"
model_dir=models/scibert_unbalanced
python neural_baselines.py \
  --train_data data/train.jsonl \
  --validation_data data/dev.jsonl \
  --test_data data/test.jsonl \
  --run_name "${run_name}" \
  --tag "${tag}" \
  --model_name allenai/scibert_scivocab_uncased \
  --model_dir ${model_dir} \
  --n_gpu 1 \
  --batch_size 4 \
  --learning_rate 0.000001351 \
  --warmup_steps 300 \
  --weight_decay 0.1 \
  --n_epochs 3 \
  --seed 1000
