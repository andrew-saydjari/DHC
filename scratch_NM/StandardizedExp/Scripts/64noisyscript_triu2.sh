#!/bin/bash
#SBATCH -J 64_noisy
#SBATCH --account=finkbeiner_lab
#SBATCH -p shared
#SBATCH -n 4 # Number of cores/tasks
#SBATCH -N 1       # Ensures that all cores are on one Node
#SBATCH -t 0-00:20:00 # Runtime in D-HH:MM:SS
#SBATCH --mem=8000
#SBATCH -o ../Nx64/noisy_stdtrue/SFDTargSFDCov/log_apd_noiso/%a_fullcov_triu_%A.o
#SBATCH -e ../Nx64/noisy_stdtrue/SFDTargSFDCov/log_apd_noiso/%a_fullcov_triu_%A.e


#module load gcc/9.3.0-fasrc01
module load Julia/1.5.3-linux-x86_64
#module load julia/1.5.0-fasrc01
#module load intel/17.0.4-fasrc01

julia runexp_triu.jl "log" "apd" "noiso" "../Nx64/noisy_stdtrue/" "_fullcov_triu" "Full+Eps"
