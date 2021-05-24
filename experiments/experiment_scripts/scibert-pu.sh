#!/bin/bash
. activate scientific-citation-detection

cd ..
run_name="scibert pu learning"
tag="citeworth"
model_dir=models/scibert_pu_learning
base_model_dir=models/scibert_unbalanced
mapfile -t base_run_dirs < <(ls ${base_model_dir} | sort)
j=`expr ${2} - 1`

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
  --seed 1000 \
  --pu_learning "pu_learning" \
  --pu_learning_model "${base_model_dir}/${base_run_dirs[${j}]}/model.pth"

