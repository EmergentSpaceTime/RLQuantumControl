#!/bin/bash
#SBATCH --account=pgi-8
#SBATCH --mail-user=m.abedi@fz-juelich.de
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --nodes=1
#SBATCH --time=06:00:00

module load Julia

if [ $# -le 0 ]; then
    echo "Usage: $0 <config_integer> <folder> <episodes> <cont_number> <data_folder>"
    exit 1
fi

for ((i=$1; i<=$(($1 + 31)); i++))
do
	srun -n1 -c4 --exclusive --cpu-bind=threads --threads-per-core=1 --output=outs/experiment_%j_${i}.out --error=errs/experiment_%j_${i}.err julia -t 4 --project=../../. continue_experiment.jl ${2}/config_${i}.toml ${3} ${4} ${5} &
done
wait
