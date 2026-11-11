#!/bin/bash
#SBATCH --partition=mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=9:00:0
#SBATCH --job-name="CS 601.471/671 homework6"
#SBATCH --output=slurm-%j.out
#SBATCH --mem=16G

source ~/.bashrc
module load anaconda
conda activate ssm_hw6

#Qwen
# python generate.py --prompt "What is the Capital of Argentina?" --checkpoint_path checkpoints/Qwen/Qwen2.5-7B-Instruct/model.pth --num_samples 1 --max_new_tokens 50 --device cuda

#Mistral
# python generate.py --prompt "What is the Capital of Argentina?" --checkpoint_path checkpoints/Mistral-7B/model.pth --num_samples 1 --max_new_tokens 50 --device cuda

#Long prompt - test
# python generate.py --prompt "Explain the theory of general relativity in detail, including its historical background, the mathematical formulation using tensors, and the modern experimental evidence supporting it. Discuss implications on time dilation, black holes, and gravitational waves." --checkpoint_path checkpoints/Mistral-7B/model.pth --num_samples 1 --max_new_tokens 512 --device cuda

python generate.py --checkpoint_path checkpoints/Mistral-7B/model.pth --prompt "Explain the theory of general relativity in detail, including its historical background, the mathematical formulation using tensors, and the modern experimental evidence supporting it. Discuss implications on time dilation, black holes, and gravitational waves." --max_new_tokens 1999 --device cuda