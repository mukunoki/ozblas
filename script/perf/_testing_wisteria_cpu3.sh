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
#export KMP_AFFINITY=scatter
#export KMP_AFFINITY=compact

CPU=a100
CPU=8360y

numactl --localalloc ./testing/testing_ozblas_ddot --mode=p --phi=1 --mnk=33554432 --repromode=0 --degree=1 > _testing_ozblas_ddot_phi1_${CPU}.dat 
numactl --localalloc ./testing/testing_ozblas_ddot --mode=p --phi=1 --mnk=33554432 --summode=1 --splitepsmode=1 > _testing_ozblas_ddot-cr_phi1_${CPU}_eps1.dat 
numactl --localalloc ./testing/testing_ozblas_ddot --mode=p --phi=1 --mnk=33554432 --summode=1 --splitepsmode=2 > _testing_ozblas_ddot-cr_phi1_${CPU}_eps2.dat 
numactl --localalloc ./testing/testing_ozblas_ddot --mode=p --phi=4 --mnk=33554432 --repromode=0 --degree=1 > _testing_ozblas_ddot_phi4_${CPU}.dat 
numactl --localalloc ./testing/testing_ozblas_ddot --mode=p --phi=4 --mnk=33554432 --summode=1 --splitepsmode=1 > _testing_ozblas_ddot-cr_phi4_${CPU}_eps1.dat 
numactl --localalloc ./testing/testing_ozblas_ddot --mode=p --phi=4 --mnk=33554432 --summode=1 --splitepsmode=2 > _testing_ozblas_ddot-cr_phi4_${CPU}_eps2.dat 
numactl --localalloc ./testing/testing_ozblas_ddot --mode=p --phi=8 --mnk=33554432 --repromode=0 --degree=1 > _testing_ozblas_ddot_phi8_${CPU}.dat 
numactl --localalloc ./testing/testing_ozblas_ddot --mode=p --phi=8 --mnk=33554432 --summode=1 --splitepsmode=1 > _testing_ozblas_ddot-cr_phi8_${CPU}_eps1.dat 
numactl --localalloc ./testing/testing_ozblas_ddot --mode=p --phi=8 --mnk=33554432 --summode=1 --splitepsmode=2 > _testing_ozblas_ddot-cr_phi8_${CPU}_eps2.dat 
numactl --localalloc ./testing/testing_ozblas_ddot --mode=p --phi=16 --mnk=33554432 --repromode=0 --degree=1 > _testing_ozblas_ddot_phi16_${CPU}.dat 
numactl --localalloc ./testing/testing_ozblas_ddot --mode=p --phi=16 --mnk=33554432 --summode=1 --splitepsmode=1 > _testing_ozblas_ddot-cr_phi16_${CPU}_eps1.dat 
numactl --localalloc ./testing/testing_ozblas_ddot --mode=p --phi=16 --mnk=33554432 --summode=1 --splitepsmode=2 > _testing_ozblas_ddot-cr_phi16_${CPU}_eps2.dat 

