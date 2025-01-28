#!/bin/bash
#SBATCH --account=pgi-8
#SBATCH --mail-user=m.abedi@fz-juelich.de
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --job-name=rlqc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=06:00:00


if [ $# -le 1 ]; then
    echo "Usage: $0 <configs_parent_folder> <episodes>"
    exit 1
fi

module load Stages/2024
module load Julia

n_t=0
for fldrs in $1/*; do
	srun -n1 -c4 --exclusive --cpu-bind=threads --threads-per-core=1 --output=outs/experiment_%j_${n_t}.out --error=errs/experiment_%j_${n_t}.err julia -t 4 --project=../../. run_experiment.jl "${fldrs}/config.toml" ${2} &
    ((n_t++))
done
echo $n_t

wait
