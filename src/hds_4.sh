#!/bin/bash -l

# Set SCC project
#$ -P ds598

module load miniconda
module load academic-ml/spring-2024

conda activate spring-2024-pyt
# conda activate sub-seg

# Change this path to point to your project directory
export PYTHONPATH="/projectnb/ds598/students/jun/subtitle-segmentation:$PYTHONPATH"

python hds_4.py

### The command below is used to submit the job to the cluster
### qsub -pe omp 4 -P ds598 -l gpus=1 demo_train.sh
