#!/bin/bash
#PJM --rsc-list "node=1"
#PJM --rsc-list "rscgrp=small"
#PJM --rsc-list "elapse=20:00"
#PJM --mpi "proc=1"
#PJM -S

export OMP_NUM_THREADS=48
export PLE_MPI_STD_EMPTYFILE=off
export FLIB_FASTOMP=TRUE

. /vol0004/apps/oss/spack/share/spack/setup-env.sh
spack load /yug2jlm
make clean
make -j16

