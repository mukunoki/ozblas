#!/bin/bash
#SBATCH --job-name=jobname
#SBATCH --partition=a100
#SBATCH -t 02:00:00

module load system/x86_64
export OMP_NUM_THREADS=256

# sum1
../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=n --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 > _dgemv_n_sum1_fast0_bgemm0_sq_s_cuda.dat 
../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=t --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 > _dgemv_t_sum1_fast0_bgemm0_sq_s_cuda.dat 

../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=n --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 --m=77 > _dgemv_n_sum1_fast0_bgemm0_m77_cuda.dat 
../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=t --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 --m=77 > _dgemv_t_sum1_fast0_bgemm0_m77_cuda.dat 

../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=n --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 --n=77 > _dgemv_n_sum1_fast0_bgemm0_n77_cuda.dat 
../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=t --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 --n=77 > _dgemv_t_sum1_fast0_bgemm0_n77_cuda.dat 



../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=n --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 > _dgemv_n_sum1_fast0_bgemm1_sq_s_cuda.dat 
../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=t --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 > _dgemv_t_sum1_fast0_bgemm1_sq_s_cuda.dat 

../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=n --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 --m=77 > _dgemv_n_sum1_fast0_bgemm1_m77_cuda.dat 
../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=t --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 --m=77 > _dgemv_t_sum1_fast0_bgemm1_m77_cuda.dat 

../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=n --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 --n=77 > _dgemv_n_sum1_fast0_bgemm1_n77_cuda.dat 
../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=t --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 --n=77 > _dgemv_t_sum1_fast0_bgemm1_n77_cuda.dat 



# sum0
../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 > _dgemv_n_sum0_fast0_bgemm0_sq_s_cuda.dat 
../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 > _dgemv_t_sum0_fast0_bgemm0_sq_s_cuda.dat 

../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 --m=77 > _dgemv_n_sum0_fast0_bgemm0_m77_cuda.dat 
../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 --m=77 > _dgemv_t_sum0_fast0_bgemm0_m77_cuda.dat 

../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 --n=77 > _dgemv_n_sum0_fast0_bgemm0_n77_cuda.dat 
../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 --n=77 > _dgemv_t_sum0_fast0_bgemm0_n77_cuda.dat 



../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:13 > _dgemv_n_sum0_fast1_bgemm0_sq_s_cuda.dat 
../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:13 > _dgemv_t_sum0_fast1_bgemm0_sq_s_cuda.dat 

../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:13 --m=77 > _dgemv_n_sum0_fast1_bgemm0_m77_cuda.dat 
../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:13 --m=77 > _dgemv_t_sum0_fast1_bgemm0_m77_cuda.dat 

../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:13 --n=77 > _dgemv_n_sum0_fast1_bgemm0_n77_cuda.dat 
../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:13 --n=77 > _dgemv_t_sum0_fast1_bgemm0_n77_cuda.dat 



../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 > _dgemv_n_sum0_fast0_bgemm1_sq_s_cuda.dat 
../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 > _dgemv_t_sum0_fast0_bgemm1_sq_s_cuda.dat 

../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 --m=77 > _dgemv_n_sum0_fast0_bgemm1_m77_cuda.dat 
../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 --m=77 > _dgemv_t_sum0_fast0_bgemm1_m77_cuda.dat 

../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 --n=77 > _dgemv_n_sum0_fast0_bgemm1_n77_cuda.dat 
../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 --n=77 > _dgemv_t_sum0_fast0_bgemm1_n77_cuda.dat 



../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:13 > _dgemv_n_sum0_fast1_bgemm0_sq_s_cuda.dat 
../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:13 > _dgemv_t_sum0_fast1_bgemm0_sq_s_cuda.dat 

../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:13 --m=77 > _dgemv_n_sum0_fast1_bgemm0_m77_cuda.dat 
../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:13 --m=77 > _dgemv_t_sum0_fast1_bgemm0_m77_cuda.dat 

../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:13 --n=77 > _dgemv_n_sum0_fast1_bgemm0_n77_cuda.dat 
../../testing/testing_cuozblas_dgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:13 --n=77 > _dgemv_t_sum0_fast1_bgemm0_n77_cuda.dat 


