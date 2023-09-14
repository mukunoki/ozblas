#!/bin/bash
#SBATCH --job-name=jobname
#SBATCH --partition=a100
#SBATCH -t 02:00:00

module load system/x86_64
export OMP_NUM_THREADS=256

numactl --localalloc ./testing_cblas_sgemm --mode=p --range=128:8192:pow > _testing_cblas_sgemm_nn.dat
numactl --localalloc ./testing_cblas_dgemm --mode=p --range=128:8192:pow > _testing_cblas_dgemm_nn.dat
numactl --localalloc ./testing_ozblas_sgemm --mode=p --range=128:8192:pow > _testing_ozblas_sgemm_nn.dat
numactl --localalloc ./testing_ozblas_dgemm --mode=p --range=128:8192:pow > _testing_ozblas_dgemm_nn.dat
numactl --localalloc ./testing_ozblas_dsgemm --mode=p --range=128:8192:pow > _testing_ozblas_dsgemm_nn.dat
numactl --localalloc ./testing_ozblas_qsgemm --mode=p --range=128:8192:pow > _testing_ozblas_qsgemm_nn.dat
numactl --localalloc ./testing_ozblas_qdgemm --mode=p --range=128:8192:pow > _testing_ozblas_qdgemm_nn.dat

numactl --localalloc ./testing_cblas_sgemv --mode=p --range=128:8192:pow > _testing_cblas_sgemv_n.dat
numactl --localalloc ./testing_cblas_dgemv --mode=p --range=128:8192:pow > _testing_cblas_dgemv_n.dat
numactl --localalloc ./testing_ozblas_sgemv --mode=p --range=128:8192:pow > _testing_ozblas_sgemv_n.dat
numactl --localalloc ./testing_ozblas_dgemv --mode=p --range=128:8192:pow > _testing_ozblas_dgemv_n.dat
numactl --localalloc ./testing_ozblas_dsgemv --mode=p --range=128:8192:pow > _testing_ozblas_dsgemv_n.dat
numactl --localalloc ./testing_ozblas_qsgemv --mode=p --range=128:8192:pow > _testing_ozblas_qsgemv_n.dat
numactl --localalloc ./testing_ozblas_qdgemv --mode=p --range=128:8192:pow > _testing_ozblas_qdgemv_n.dat

numactl --localalloc ./testing_cblas_sdot --mode=p --range=128:8192000:pow > _testing_cblas_sdot.dat
numactl --localalloc ./testing_cblas_ddot --mode=p --range=128:8192000:pow > _testing_cblas_ddot.dat
numactl --localalloc ./testing_ozblas_sdot --mode=p --range=128:8192000:pow > _testing_ozblas_sdot.dat
numactl --localalloc ./testing_ozblas_ddot --mode=p --range=128:8192000:pow > _testing_ozblas_ddot.dat
numactl --localalloc ./testing_ozblas_dsdot --mode=p --range=128:8192000:pow > _testing_ozblas_dsdot.dat
numactl --localalloc ./testing_ozblas_qsdot --mode=p --range=128:8192000:pow > _testing_ozblas_qsdot.dat
numactl --localalloc ./testing_ozblas_qddot --mode=p --range=128:8192000:pow > _testing_ozblas_qddot.dat



numactl --localalloc ./testing_cublas_sgemm --mode=p --range=128:8192:pow > _testing_cublas_sgemm_nn.dat
numactl --localalloc ./testing_cublas_dgemm --mode=p --range=128:8192:pow > _testing_cublas_dgemm_nn.dat
numactl --localalloc ./testing_cuozblas_sgemm --mode=p --range=128:8192:pow > _testing_cuozblas_sgemm_nn.dat
numactl --localalloc ./testing_cuozblas_dgemm --mode=p --range=128:8192:pow > _testing_cuozblas_dgemm_nn.dat
numactl --localalloc ./testing_cuozblas_dsgemm --mode=p --range=128:8192:pow > _testing_cuozblas_dsgemm_nn.dat

numactl --localalloc ./testing_cublas_sgemv --mode=p --range=128:8192:pow > _testing_cublas_sgemv_n.dat
numactl --localalloc ./testing_cublas_dgemv --mode=p --range=128:8192:pow > _testing_cublas_dgemv_n.dat
numactl --localalloc ./testing_cuozblas_sgemv --mode=p --range=128:8192:pow > _testing_cuozblas_sgemv_n.dat
numactl --localalloc ./testing_cuozblas_dgemv --mode=p --range=128:8192:pow > _testing_cuozblas_dgemv_n.dat
numactl --localalloc ./testing_cuozblas_dsgemv --mode=p --range=128:8192:pow > _testing_cuozblas_dsgemv_n.dat

numactl --localalloc ./testing_cublas_sdot --mode=p --range=128:8192000:pow > _testing_cublas_sdot.dat
numactl --localalloc ./testing_cublas_ddot --mode=p --range=128:8192000:pow > _testing_cublas_ddot.dat
numactl --localalloc ./testing_cuozblas_sdot --mode=p --range=128:8192000:pow > _testing_cuozblas_sdot.dat
numactl --localalloc ./testing_cuozblas_ddot --mode=p --range=128:8192000:pow > _testing_cuozblas_ddot.dat
numactl --localalloc ./testing_cuozblas_dsdot --mode=p --range=128:8192000:pow > _testing_cuozblas_dsdot.dat
