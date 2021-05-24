#!/bin/bash
. activate scientific-citation-detection

cd ..
python train_masked_language_modeling.py \
  --train_data data/train.jsonl \
  --validation_data data/dev.jsonl \
  --model_dir models/language_modeling/citebert_cite_only \
  --model_name allenai/scibert_scivocab_uncased \
  --seed 1000 \
  --learning_rate 0.000001351 \
  --weight_decay 0.01 \
  --warmup_steps 300 \
  --n_epochs 10 \
  --cite_only \
  --unmask_citation \
  --batch_size 32
