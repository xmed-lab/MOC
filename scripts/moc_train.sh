#!/bin/bash
# modify device, dataset_name, base_result_dir

base="/home/txiang/pathology/MOC"
cd $base
dataset_name="nsclc" # "rcc"
base_result_dir="${base}/results/moc_train/${dataset_name}"
pretrain="conch"

# device
cuda_shot_1=0
cuda_shot_2=1
cuda_shot_4=2
cuda_shot_8=3
# cuda_shot_16=4


shot=1
result_dir="${base_result_dir}/${shot}_shot"
mkdir -p $result_dir
cp $0 $result_dir

CUDA_VISIBLE_DEVICES=$cuda_shot_1 python main_moc.py --fold 0 --shot ${shot} --topj 400 --topk 10 --result_dir $result_dir --dataset ${dataset_name} --disable_tqdm --pretrain $pretrain >> ${result_dir}/fold_0_shot_${shot}_output.txt 2>&1 &

CUDA_VISIBLE_DEVICES=$cuda_shot_1 python main_moc.py --fold 1 --shot ${shot} --topj 400 --topk 10 --result_dir $result_dir --dataset ${dataset_name} --disable_tqdm --pretrain $pretrain >> ${result_dir}/fold_1_shot_${shot}_output.txt 2>&1 &

CUDA_VISIBLE_DEVICES=$cuda_shot_1 python main_moc.py --fold 2 --shot ${shot} --topj 400 --topk 10 --result_dir $result_dir --dataset ${dataset_name} --disable_tqdm --pretrain $pretrain >> ${result_dir}/fold_2_shot_${shot}_output.txt 2>&1 &

CUDA_VISIBLE_DEVICES=$cuda_shot_1 python main_moc.py --fold 3 --shot ${shot} --topj 400 --topk 10 --result_dir $result_dir --dataset ${dataset_name} --disable_tqdm --pretrain $pretrain >> ${result_dir}/fold_3_shot_${shot}_output.txt 2>&1 &

CUDA_VISIBLE_DEVICES=$cuda_shot_1 python main_moc.py --fold 4 --shot ${shot} --topj 400 --topk 10 --result_dir $result_dir --dataset ${dataset_name} --disable_tqdm --pretrain $pretrain >> ${result_dir}/fold_4_shot_${shot}_output.txt 2>&1 &


shot=2
result_dir="${base_result_dir}/${shot}_shot"
mkdir -p $result_dir
cp $0 $result_dir

CUDA_VISIBLE_DEVICES=$cuda_shot_2 python main_moc.py --fold 0 --shot ${shot} --topj 400 --topk 10 --result_dir $result_dir --dataset ${dataset_name} --disable_tqdm --pretrain $pretrain >> ${result_dir}/fold_0_shot_${shot}_output.txt 2>&1 &

CUDA_VISIBLE_DEVICES=$cuda_shot_2 python main_moc.py --fold 1 --shot ${shot} --topj 400 --topk 10 --result_dir $result_dir --dataset ${dataset_name} --disable_tqdm --pretrain $pretrain >> ${result_dir}/fold_1_shot_${shot}_output.txt 2>&1 &

CUDA_VISIBLE_DEVICES=$cuda_shot_2 python main_moc.py --fold 2 --shot ${shot} --topj 400 --topk 10 --result_dir $result_dir --dataset ${dataset_name} --disable_tqdm --pretrain $pretrain >> ${result_dir}/fold_2_shot_${shot}_output.txt 2>&1 &

CUDA_VISIBLE_DEVICES=$cuda_shot_2 python main_moc.py --fold 3 --shot ${shot} --topj 400 --topk 10 --result_dir $result_dir --dataset ${dataset_name} --disable_tqdm --pretrain $pretrain >> ${result_dir}/fold_3_shot_${shot}_output.txt 2>&1 &

CUDA_VISIBLE_DEVICES=$cuda_shot_2 python main_moc.py --fold 4 --shot ${shot} --topj 400 --topk 10 --result_dir $result_dir --dataset ${dataset_name} --disable_tqdm --pretrain $pretrain >> ${result_dir}/fold_4_shot_${shot}_output.txt 2>&1 &


shot=4
result_dir="${base_result_dir}/${shot}_shot"
mkdir -p $result_dir
cp $0 $result_dir

CUDA_VISIBLE_DEVICES=$cuda_shot_4 python main_moc.py --fold 0 --shot ${shot} --topj 400 --topk 10 --result_dir $result_dir --dataset ${dataset_name} --disable_tqdm --pretrain $pretrain >> ${result_dir}/fold_0_shot_${shot}_output.txt 2>&1 &

CUDA_VISIBLE_DEVICES=$cuda_shot_4 python main_moc.py --fold 1 --shot ${shot} --topj 400 --topk 10 --result_dir $result_dir --dataset ${dataset_name} --disable_tqdm --pretrain $pretrain >> ${result_dir}/fold_1_shot_${shot}_output.txt 2>&1 &

CUDA_VISIBLE_DEVICES=$cuda_shot_4 python main_moc.py --fold 2 --shot ${shot} --topj 400 --topk 10 --result_dir $result_dir --dataset ${dataset_name} --disable_tqdm --pretrain $pretrain >> ${result_dir}/fold_2_shot_${shot}_output.txt 2>&1 &

CUDA_VISIBLE_DEVICES=$cuda_shot_4 python main_moc.py --fold 3 --shot ${shot} --topj 400 --topk 10 --result_dir $result_dir --dataset ${dataset_name} --disable_tqdm --pretrain $pretrain >> ${result_dir}/fold_3_shot_${shot}_output.txt 2>&1 &

CUDA_VISIBLE_DEVICES=$cuda_shot_4 python main_moc.py --fold 4 --shot ${shot} --topj 400 --topk 10 --result_dir $result_dir --dataset ${dataset_name} --disable_tqdm --pretrain $pretrain >> ${result_dir}/fold_4_shot_${shot}_output.txt 2>&1 &


shot=8
result_dir="${base_result_dir}/${shot}_shot"
mkdir -p $result_dir
cp $0 $result_dir

CUDA_VISIBLE_DEVICES=$cuda_shot_8 python main_moc.py --fold 0 --shot ${shot} --topj 400 --topk 10 --result_dir $result_dir --dataset ${dataset_name} --disable_tqdm --pretrain $pretrain >> ${result_dir}/fold_0_shot_${shot}_output.txt 2>&1 &

CUDA_VISIBLE_DEVICES=$cuda_shot_8 python main_moc.py --fold 1 --shot ${shot} --topj 400 --topk 10 --result_dir $result_dir --dataset ${dataset_name} --disable_tqdm --pretrain $pretrain >> ${result_dir}/fold_1_shot_${shot}_output.txt 2>&1 &

CUDA_VISIBLE_DEVICES=$cuda_shot_8 python main_moc.py --fold 2 --shot ${shot} --topj 400 --topk 10 --result_dir $result_dir --dataset ${dataset_name} --disable_tqdm --pretrain $pretrain >> ${result_dir}/fold_2_shot_${shot}_output.txt 2>&1 &

CUDA_VISIBLE_DEVICES=$cuda_shot_8 python main_moc.py --fold 3 --shot ${shot} --topj 400 --topk 10 --result_dir $result_dir --dataset ${dataset_name} --disable_tqdm --pretrain $pretrain >> ${result_dir}/fold_3_shot_${shot}_output.txt 2>&1 &

CUDA_VISIBLE_DEVICES=$cuda_shot_8 python main_moc.py --fold 4 --shot ${shot} --topj 400 --topk 10 --result_dir $result_dir --dataset ${dataset_name} --disable_tqdm --pretrain $pretrain >> ${result_dir}/fold_4_shot_${shot}_output.txt 2>&1 &


# shot=16

# result_dir="${base_result_dir}/${shot}_shot"
# mkdir -p $result_dir
# cp $0 $result_dir

# CUDA_VISIBLE_DEVICES=$cuda_shot_16 python main_moc.py --fold 0 --shot ${shot} --topj 400 --topk 10 --result_dir $result_dir --dataset ${dataset_name} --disable_tqdm --pretrain $pretrain >> ${result_dir}/fold_0_shot_${shot}_output.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=$cuda_shot_16 python main_moc.py --fold 1 --shot ${shot} --topj 400 --topk 10 --result_dir $result_dir --dataset ${dataset_name} --disable_tqdm --pretrain $pretrain >> ${result_dir}/fold_1_shot_${shot}_output.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=$cuda_shot_16 python main_moc.py --fold 2 --shot ${shot} --topj 400 --topk 10 --result_dir $result_dir --dataset ${dataset_name} --disable_tqdm --pretrain $pretrain >> ${result_dir}/fold_2_shot_${shot}_output.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=$cuda_shot_16 python main_moc.py --fold 3 --shot ${shot} --topj 400 --topk 10 --result_dir $result_dir --dataset ${dataset_name} --disable_tqdm --pretrain $pretrain >> ${result_dir}/fold_3_shot_${shot}_output.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=$cuda_shot_16 python main_moc.py --fold 4 --shot ${shot} --topj 400 --topk 10 --result_dir $result_dir --dataset ${dataset_name} --disable_tqdm --pretrain $pretrain >> ${result_dir}/fold_4_shot_${shot}_output.txt 2>&1 &