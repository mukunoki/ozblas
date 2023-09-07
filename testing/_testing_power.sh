#!/bin/bash
#PJM --rsc-list "node=1"
#PJM --rsc-list "rscgrp=small"
#PJM --rsc-list "elapse=30:00"
#PJM --mpi "proc=1"
#PJM -S

export OMP_NUM_THREADS=12
export PLE_MPI_STD_EMPTYFILE=off
export FLIB_FASTOMP=TRUE

make -j9
#numactl -C12-23 -m4 ./testing_cblas_hgemm --mode=p --range=1024:10240:1024 > _testing_ssl2_hgemm_power.dat
#numactl -C12-23 -m4 ./testing_cblas_sgemm --mode=p --range=1024:10240:1024 > _testing_ssl2_sgemm_power.dat
#numactl -C12-23 -m4 ./testing_cblas_dgemm --mode=p --range=1024:10240:1024 > _testing_ssl2_dgemm_power.dat
numactl -C12-23 -m4 ./testing_cblas_hgemv --mode=p --range=1024:20480:1024 > _testing_ssl2_hgemv_power.dat
numactl -C12-23 -m4 ./testing_cblas_sgemv --mode=p --range=1024:20480:1024 > _testing_ssl2_sgemv_power.dat
numactl -C12-23 -m4 ./testing_cblas_dgemv --mode=p --range=1024:20480:1024 > _testing_ssl2_dgemv_power.dat
numactl -C12-23 -m4 ./testing_cblas_hdot --mode=p --range=1024:102400000:pow > _testing_ssl2_hdot_power.dat
numactl -C12-23 -m4 ./testing_cblas_sdot --mode=p --range=1024:102400000:pow > _testing_ssl2_sdot_power.dat
numactl -C12-23 -m4 ./testing_cblas_ddot --mode=p --range=1024:102400000:pow > _testing_ssl2_ddot_power.dat

