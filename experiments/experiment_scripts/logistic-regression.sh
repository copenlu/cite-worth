#!/bin/bash
. activate scientific-citation-detection

cd ..
run_name="logistic-regression"
tag="citeworth"
model_dir=models/logistic_regression_baseline
python logistic_regression_baseline.py \
  --train_data data/train.jsonl \
  --validation_data data/dev.jsonl \
  --test_data data/test.jsonl \
  --run_name "${run_name}" \
  --tag "${tag}" \
  --C 0.11513953993264457 \
  --seed 1000

