#!/bin/bash
. activate scientific-citation-detection

cd ..
python train_masked_language_modeling.py \
  --train_data data/train.jsonl \
  --validation_data data/dev.jsonl \
  --model_dir models/language_modeling/citebert \
  --model_name allenai/scibert_scivocab_uncased \
  --seed 1000 \
  --learning_rate 0.000001351 \
  --weight_decay 0.01 \
  --warmup_steps 300 \
  --n_epochs 10 \
  --use_cite_objective \
  --batch_size 32
