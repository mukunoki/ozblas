#!/bin/bash
#SBATCH --job-name=jobname
#SBATCH --partition=a100
#SBATCH -t 02:00:00

module load system/x86_64
export OMP_NUM_THREADS=256

numactl --localalloc ./testing_ozblas_sgemm --mode=p --summode=1 --range=128:8192:pow > _testing_ozblas_sgemm_nn_sum1.dat
numactl --localalloc ./testing_ozblas_dgemm --mode=p --summode=1 --range=128:8192:pow > _testing_ozblas_dgemm_nn_sum1.dat

numactl --localalloc ./testing_ozblas_sgemv --mode=p --summode=1 --range=128:8192:pow > _testing_ozblas_sgemv_n_sum1.dat
numactl --localalloc ./testing_ozblas_dgemv --mode=p --summode=1 --range=128:8192:pow > _testing_ozblas_dgemv_n_sum1.dat

numactl --localalloc ./testing_ozblas_sdot --mode=p --summode=1 --range=128:8192000:pow > _testing_ozblas_sdot_sum1.dat
numactl --localalloc ./testing_ozblas_ddot --mode=p --summode=1 --range=128:8192000:pow > _testing_ozblas_ddot_sum1.dat



numactl --localalloc ./testing_cuozblas_sgemm --mode=p --summode=1 --range=128:8192:pow > _testing_cuozblas_sgemm_nn_sum1.dat
numactl --localalloc ./testing_cuozblas_dgemm --mode=p --summode=1 --range=128:8192:pow > _testing_cuozblas_dgemm_nn_sum1.dat

numactl --localalloc ./testing_cuozblas_sgemv --mode=p --summode=1 --range=128:8192:pow > _testing_cuozblas_sgemv_n_sum1.dat
numactl --localalloc ./testing_cuozblas_dgemv --mode=p --summode=1 --range=128:8192:pow > _testing_cuozblas_dgemv_n_sum1.dat

numactl --localalloc ./testing_cuozblas_sdot --mode=p --summode=1 --range=128:8192000:pow > _testing_cuozblas_sdot_sum1.dat
numactl --localalloc ./testing_cuozblas_ddot --mode=p --summode=1 --range=128:8192000:pow > _testing_cuozblas_ddot_sum1.dat
