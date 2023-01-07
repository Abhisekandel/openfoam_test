#!/bin/bash 
# 
#SBATCH  --job-name  D029_snappy
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


#export WM_PRECISION_OPTION=SPDP;


#mpirun -n 2048 omplace -nt 1 -c 0-127:st=2 checkMesh -parallel -fileHandler collated > log.pimpleFoam.$(date +%s) 2>&1;
#mpirun -n 2048 omplace -nt 1 -c 0-127:st=2 transformPoints -scale 0.001 -parallel -fileHandler collated > log.transformPoints.$(date +%s) 2>&1;


#mpirun -n 256 omplace -nt 1 -c 0-127:st=2 overPimpleDyMFoam -parallel -fileHandler collated > log.pimpleFoam.$(date +%s) 2>&1;

#mpirun -n 256 omplace -nt 1 -c 0-127:st=2  foamFormatConvert -parallel -latestTime -fileHandler uncollated

./Allrun.pre

