#!/bin/sh

source ../../venv/cv/bin/activate
source set_pythonpath.sh

model_params="/Users/jongbeomkim/Documents/ddpm/kr-ml-test/ddpm_celeba_64×64.pth"
# model_params="/home/dmeta0304/Downloads/ddpm_celeba_64×64.pth"
img_size=64
dataset="celeba"
data_dir="/Users/jongbeomkim/Documents/datasets"
diffusion_step_idx=300
n_colors=6
batch_size=36

ref_indices=(31 32 33 34 35 36 37 38 39 40 41)
for ref_idx in "${ref_indices[@]}"
do
    python3 ../sample.py\
        --mode="from_stroke"\
        --model_params="$model_params"\
        --img_size=$img_size\
        --dataset="$dataset"\
        --data_dir="$data_dir"\
        --ref_idx=$ref_idx\
        --diffusion_step_idx=$diffusion_step_idx\
        --n_colors=$n_colors\
        --batch_size=$batch_size
done