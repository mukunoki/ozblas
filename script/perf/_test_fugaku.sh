#!/bin/bash
#PJM --rsc-list "node=1"
#PJM --rsc-list "rscgrp=small"
#PJM --rsc-list "elapse=20:00"
#PJM --mpi "proc=1"
#PJM -S

export OMP_NUM_THREADS=1
export PLE_MPI_STD_EMPTYFILE=off
export FLIB_FASTOMP=TRUE

for((i = 0; i<100;i++))
{
	dim_m=$(($RANDOM % 1000))
	dim_n=$(($RANDOM % 10))
	if((${dim_m} != 0 && ${dim_n} != 0)); then
		./testing/testing_ozblas_qdgemv --nodisp=1 --m=${dim_m} --n=${dim_n} --phi=1 --splitmode=3 --summode=3
	fi	
}

