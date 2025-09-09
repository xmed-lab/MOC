#!/bin/bash
# result summary, not actually model inference

base="/home/txiang/pathology/MOC"
dataset_name="nsclc" # "rcc"
cd $base
result_dir="${base}/results/moc_train/${dataset_name}"

python main_moc.py --summary --summary_dir $result_dir