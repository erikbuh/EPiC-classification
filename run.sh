#!/bin/bash
 
#########################
## SLURM JOB COMMANDS ###
#########################
#SBATCH --partition=maxgpu            ## or allgpu / cms / cms-uhh / maxgpu
#SBATCH --time=1-00:00:00      # d-HH:MM:SS
#SBATCH --nodes=1
#SBATCH --job-name epicClass         # give job unique name
#SBATCH --output ./joblog/%j.out      # terminal output
#SBATCH --error ./joblog/%j.err
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user erik.buhmann@desy.de
##SBATCH --constraint=GPU

##SBATCH --nodelist=max-cmsg004         # you can select specific nodes, if necessary
##SBATCH --constraint=V100
#SBATCH --constraint="A100|V100|P100"
##SBATCH --constraint="GPUx1&A100"
 
#####################
### BASH COMMANDS ###
#####################
 
## examples:
 
# source and load modules (GPU drivers, anaconda, etc)
source ~/.bashrc
#source /etc/profile.d/modules.sh
#module load maxwell
#module load cuda
#module load cuda/10.1 maxwell
 
# activate your conda environment the job should use
conda activate torch16_EF

# specific command as there was a libary CXX error (points to libaries of used environment)
# LD_LIBRARY_PATH=/beegfs/desy/user/buhmae/conda/envs/torch16_EF_sink/lib/:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH
 
# go to your folder with your python scripts
cd /home/buhmae/7_EPiC-classification/
# source erik_env.sh
# run

PARAMS=(
    --epochs 500
    --early_stopping 100
    --log_comet True
    --reason "first test"

    --out_prefix top30_
    --dataset_train /beegfs/desy/user/buhmae/7_EPiC-classification/dataset/mixed/top_jetnet30_train.npz
    --dataset_val /beegfs/desy/user/buhmae/7_EPiC-classification/dataset/mixed/top_jetnet30_val.npz
    --dataset_test /beegfs/desy/user/buhmae/7_EPiC-classification/dataset/mixed/top_jetnet30_test.npz

)

srun -n 1 python train.py "${PARAMS[@]}"
