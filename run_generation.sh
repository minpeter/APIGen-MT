#!/bin/bash
cd $HOME/data/APIGen-MT
PYTHONPATH=src:. python src/generate_step_by_step.py \
  --num-datapoints 10 \
  --num-actions 10 \
  --output data/generated/stateful_10x10_datapoints.jsonl \
  > data/generated/stateful_10x10_log.txt 2>&1
