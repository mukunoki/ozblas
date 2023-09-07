#!/bin/sh
#------ pjsub option --------#
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM --omp thread=36
#PJM -L elapse=0:45:00
#PJM -g jh180023a
#PJM -j
#------- Program execution -------

module load intel/2022.1.2
module load cuda/11.4

export OMP_NUM_THREADS=36
export MKL_NUM_THREADS=36

CPU=a100
CPU=8360y

ARRAY=( gyro_k pdb1HYS nd24k )
#ARRAY=( gridgena cfd1 cbuckle BenElechi1 gyro_k pdb1HYS nd24k )
#ARRAY=( tmt_sym gridgena cfd1 cbuckle BenElechi1 gyro_k pdb1HYS nd24k )
SUF=

for((i = 0; i<${#ARRAY[*]};i++))
{
	matrix=${ARRAY[i]}
        matsrc=/work/jh180023a/k23027/matrix/${matrix}.mtx

	numactl --localalloc ./testing/testing_ozblas_dcg --tol=1e-16 --maxiter=80000 --verbose=80000 --repromode=0 --degree=1 --mtx=${matsrc} > _testing_ozblas_dcg_${matrix}_${CPU}_${SUF}.dat 
#	numactl --localalloc ./testing/testing_ozblas_dcg --tol=1e-16 --maxiter=80000 --verbose=80000 --repromode=0 --degree=1 --precx=1 --mtx=${matsrc} > _testing_ozblas_dxcg_${matrix}_${CPU}_${SUF}.dat 
#	numactl --localalloc ./testing/testing_ozblas_dcg --tol=1e-16 --maxiter=80000 --verbose=80000 --summode=1 --splitepsmode=1 --mtx=${matsrc} > _testing_ozblas_dcg-cr_${matrix}_${CPU}_eps1_${SUF}.dat 
	numactl --localalloc ./testing/testing_ozblas_dcg --tol=1e-16 --maxiter=80000 --verbose=80000 --summode=1 --splitepsmode=2 --mtx=${matsrc} > _testing_ozblas_dcg-cr_${matrix}_${CPU}_eps2_${SUF}.dat 
}
	

