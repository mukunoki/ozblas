#!/bin/sh
#------ pjsub option --------#
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM --omp thread=36
#PJM -L elapse=1:25:00
#PJM -g jh180023a
#PJM -j
#------- Program execution -------

module load intel/2022.1.2
module load cuda/11.4

export OMP_NUM_THREADS=36
export MKL_NUM_THREADS=36

CPU=a100
CPU=8360y

ARRAY=( tmt_sym gridgena cfd1 cbuckle BenElechi1 gyro_k pdb1HYS nd24k )

for((i = 0; i<${#ARRAY[*]};i++))
{
	matrix=${ARRAY[i]}
    matsrc=/work/jh180023a/k23027/matrix/${matrix}.mtx
#	numactl --localalloc ./testing/testing_ozblas_dcsrmv --nodisp=1 --summode=1 --mode=p --splitepsmode=1 --mtx=${matsrc} > _testing_ozblas_dcsrmv_${matrix}_${CPU}_myspmv_eps1.dat 
#	numactl --localalloc ./testing/testing_ozblas_dcsrmv --nodisp=1 --summode=1 --mode=p --splitepsmode=2 --mtx=${matsrc} > _testing_ozblas_dcsrmv_${matrix}_${CPU}_myspmv_eps2.dat 
#	numactl --localalloc ./testing/testing_ozblas_dcsrmv --nodisp=1 --summode=1 --mode=p --splitepsmode=1 --mtx=${matsrc} > _testing_ozblas_dcsrmv_${matrix}_${CPU}_mklspmv.dat 
#	numactl --localalloc ./testing/testing_cblas_dcsrmm --nodisp=1 --mode=p --range=1:6:1 --mtx=${matsrc} > _testing_cblas_dcsrmm_${matrix}_${CPU}_my.dat 
	numactl --localalloc ./testing/testing_cblas_dcsrmm --nodisp=1 --mode=p --range=1:6:1 --mtx=${matsrc} > _testing_cblas_dcsrmm_${matrix}_${CPU}_my_x2_avx256.dat 
}
	

