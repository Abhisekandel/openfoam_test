#!/bin/bash 
# 
#SBATCH  --job-name  D026_3
#SBATCH  --nodes=1
#SBATCH  --ntasks=36
#SBATCH  --ntasks-per-node=36
#SBATCH  --partition prandtl 
#SBATCH  --time=0-24:00:00 


module purge
module load gnu8
module load openmpi3
source /opt/software/openfoam/openfoamv2106/OpenFOAM-v2106/etc/bashrc
. $WM_PROJECT_DIR/bin/tools/RunFunctions
#export WM_PRECISION_OPTION=SPDP 

decomposePar >log.decomposePar
#mpirun -n 36 pimpleFoam -parallel -fileHandler collated > log.pimpleFoam.$(date +%s) 2>&1 #2>&1 #Co40.$(date +%s) 2>&1
#reconstructPar >log.reconstructPar

