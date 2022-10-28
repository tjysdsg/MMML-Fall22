sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 $1
