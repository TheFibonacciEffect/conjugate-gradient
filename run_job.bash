#!/bin/bash
#SBATCH --job-name=dimensional-scaling         # Job name
#SBATCH --output=output_%j.txt     # Output file (%j expands to jobID)
#SBATCH --error=error_%j.txt       # Error file (%j expands to jobID)
#SBATCH --ntasks=1                 # Number of tasks (usually 1 for most jobs)
#SBATCH --cpus-per-task=1          # Number of CPU cores per task (adjust as needed)
#SBATCH --mem=8G                   # Memory per node (8GB)
#SBATCH --time=01:00:00            # Time limit (hh:mm:ss)
## #SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --partition=defq 

# Execute your program or script here
srun julia --project=. src/main.jl
