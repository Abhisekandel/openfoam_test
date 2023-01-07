#!/bin/bash

##PBS -N 3Dblad15
##PBS -l select=32:node_type=rome:mpiprocs=64
##PBS -l walltime=24:00:00

#PBS -N E020
##PBS -q test
#PBS -l select=1:node_type=rome:mpiprocs=64
##PBS -l select=16:node_type=rome:mpiprocs=64
#PBS -l walltime=23:50:00
##PBS -l walltime=00:25:00


##export MPI_DSM_DISTRIBUTE=1
##export MPI_DSM_CPULIST=0-127/4:allhosts

# Change to the direcotry that the job was submitted from
dateName=$(date +%s)
cd $PBS_O_WORKDIR

module purge
module load gcc/9.2.0;
module load mpt/2.23;

module load openfoam/2012-int32
source /opt/hlrs/spack/current/openfoam/2012-gcc-9.2.0-xul5nngt/etc/bashrc


#export WM_PRECISION_OPTION=SPDP;


#mpirun -n 2048 omplace -nt 1 -c 0-127:st=2 checkMesh -parallel -fileHandler collated > log.pimpleFoam.$(date +%s) 2>&1;
#mpirun -n 2048 omplace -nt 1 -c 0-127:st=2 transformPoints -scale 0.001 -parallel -fileHandler collated > log.transformPoints.$(date +%s) 2>&1;
#mpirun -n 768 omplace -nt 1 -c 0-127:st=2 pimpleFoam -parallel -fileHandler collated >> log.pimpleFoam.$dateName 2>&1;

#decomposePar >log.decomposePar
#cat job.sh > log.pimpleFoam.$dateName
#printf '*%.0s' {1..80} >> log.pimpleFoam.$dateName;
#printf "\n" >> log.pimpleFoam.$dateName;
#mpirun -n 1024 omplace -nt 1 -c 0-127:st=2 pimpleFoam -parallel -fileHandler collated >> log.pimpleFoam.$dateName 2>&1;
reconstructPar >log.reconstructPar

