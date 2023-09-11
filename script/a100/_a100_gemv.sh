#!/bin/bash
#SBATCH --job-name=jobname
#SBATCH --partition=a100
#SBATCH -t 02:00:00

module load system/x86_64
export OMP_NUM_THREADS=256

#################
# dgemv
#################
# sum1
../../testing/testing_ozblas_dgemv --nodisp=1 --transa=n --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _dgemv_n_sum1_fast0_bgemm0_sq_s.dat 
../../testing/testing_ozblas_dgemv --nodisp=1 --transa=t --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _dgemv_t_sum1_fast0_bgemm0_sq_s.dat 

../../testing/testing_ozblas_dgemv --nodisp=1 --transa=n --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _dgemv_n_sum1_fast0_bgemm0_m77.dat 
../../testing/testing_ozblas_dgemv --nodisp=1 --transa=t --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _dgemv_t_sum1_fast0_bgemm0_m77.dat 

../../testing/testing_ozblas_dgemv --nodisp=1 --transa=n --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _dgemv_n_sum1_fast0_bgemm0_n77.dat 
../../testing/testing_ozblas_dgemv --nodisp=1 --transa=t --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _dgemv_t_sum1_fast0_bgemm0_n77.dat 



../../testing/testing_ozblas_dgemv --nodisp=1 --transa=n --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _dgemv_n_sum1_fast0_bgemm1_sq_s.dat 
../../testing/testing_ozblas_dgemv --nodisp=1 --transa=t --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _dgemv_t_sum1_fast0_bgemm1_sq_s.dat 

../../testing/testing_ozblas_dgemv --nodisp=1 --transa=n --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _dgemv_n_sum1_fast0_bgemm1_m77.dat 
../../testing/testing_ozblas_dgemv --nodisp=1 --transa=t --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _dgemv_t_sum1_fast0_bgemm1_m77.dat 

../../testing/testing_ozblas_dgemv --nodisp=1 --transa=n --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _dgemv_n_sum1_fast0_bgemm1_n77.dat 
../../testing/testing_ozblas_dgemv --nodisp=1 --transa=t --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _dgemv_t_sum1_fast0_bgemm1_n77.dat 



# sum0
../../testing/testing_ozblas_dgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _dgemv_n_sum0_fast0_bgemm0_sq_s.dat 
../../testing/testing_ozblas_dgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _dgemv_t_sum0_fast0_bgemm0_sq_s.dat 

../../testing/testing_ozblas_dgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _dgemv_n_sum0_fast0_bgemm0_m77.dat 
../../testing/testing_ozblas_dgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _dgemv_t_sum0_fast0_bgemm0_m77.dat 

../../testing/testing_ozblas_dgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _dgemv_n_sum0_fast0_bgemm0_n77.dat 
../../testing/testing_ozblas_dgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _dgemv_t_sum0_fast0_bgemm0_n77.dat 



../../testing/testing_ozblas_dgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 > _dgemv_n_sum0_fast1_bgemm0_sq_s.dat 
../../testing/testing_ozblas_dgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 > _dgemv_t_sum0_fast1_bgemm0_sq_s.dat 

../../testing/testing_ozblas_dgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --m=77 > _dgemv_n_sum0_fast1_bgemm0_m77.dat 
../../testing/testing_ozblas_dgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --m=77 > _dgemv_t_sum0_fast1_bgemm0_m77.dat 

../../testing/testing_ozblas_dgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --n=77 > _dgemv_n_sum0_fast1_bgemm0_n77.dat 
../../testing/testing_ozblas_dgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --n=77 > _dgemv_t_sum0_fast1_bgemm0_n77.dat 



../../testing/testing_ozblas_dgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _dgemv_n_sum0_fast0_bgemm1_sq_s.dat 
../../testing/testing_ozblas_dgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _dgemv_t_sum0_fast0_bgemm1_sq_s.dat 

../../testing/testing_ozblas_dgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _dgemv_n_sum0_fast0_bgemm1_m77.dat 
../../testing/testing_ozblas_dgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _dgemv_t_sum0_fast0_bgemm1_m77.dat 

../../testing/testing_ozblas_dgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _dgemv_n_sum0_fast0_bgemm1_n77.dat 
../../testing/testing_ozblas_dgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _dgemv_t_sum0_fast0_bgemm1_n77.dat 



../../testing/testing_ozblas_dgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 > _dgemv_n_sum0_fast1_bgemm1_sq_s.dat 
../../testing/testing_ozblas_dgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 > _dgemv_t_sum0_fast1_bgemm1_sq_s.dat 

../../testing/testing_ozblas_dgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --m=77 > _dgemv_n_sum0_fast1_bgemm1_m77.dat 
../../testing/testing_ozblas_dgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --m=77 > _dgemv_t_sum0_fast1_bgemm1_m77.dat 

../../testing/testing_ozblas_dgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --n=77 > _dgemv_n_sum0_fast1_bgemm1_n77.dat 
../../testing/testing_ozblas_dgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --n=77 > _dgemv_t_sum0_fast1_bgemm1_n77.dat 


#################
# sgemv
#################
# sum1
../../testing/testing_ozblas_sgemv --nodisp=1 --transa=n --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _sgemv_n_sum1_fast0_bgemm0_sq_s.dat 
../../testing/testing_ozblas_sgemv --nodisp=1 --transa=t --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _sgemv_t_sum1_fast0_bgemm0_sq_s.dat 

../../testing/testing_ozblas_sgemv --nodisp=1 --transa=n --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _sgemv_n_sum1_fast0_bgemm0_m77.dat 
../../testing/testing_ozblas_sgemv --nodisp=1 --transa=t --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _sgemv_t_sum1_fast0_bgemm0_m77.dat 

../../testing/testing_ozblas_sgemv --nodisp=1 --transa=n --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _sgemv_n_sum1_fast0_bgemm0_n77.dat 
../../testing/testing_ozblas_sgemv --nodisp=1 --transa=t --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _sgemv_t_sum1_fast0_bgemm0_n77.dat 



../../testing/testing_ozblas_sgemv --nodisp=1 --transa=n --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _sgemv_n_sum1_fast0_bgemm1_sq_s.dat 
../../testing/testing_ozblas_sgemv --nodisp=1 --transa=t --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _sgemv_t_sum1_fast0_bgemm1_sq_s.dat 

../../testing/testing_ozblas_sgemv --nodisp=1 --transa=n --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _sgemv_n_sum1_fast0_bgemm1_m77.dat 
../../testing/testing_ozblas_sgemv --nodisp=1 --transa=t --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _sgemv_t_sum1_fast0_bgemm1_m77.dat 

../../testing/testing_ozblas_sgemv --nodisp=1 --transa=n --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _sgemv_n_sum1_fast0_bgemm1_n77.dat 
../../testing/testing_ozblas_sgemv --nodisp=1 --transa=t --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _sgemv_t_sum1_fast0_bgemm1_n77.dat 



# sum0
../../testing/testing_ozblas_sgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _sgemv_n_sum0_fast0_bgemm0_sq_s.dat 
../../testing/testing_ozblas_sgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _sgemv_t_sum0_fast0_bgemm0_sq_s.dat 

../../testing/testing_ozblas_sgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _sgemv_n_sum0_fast0_bgemm0_m77.dat 
../../testing/testing_ozblas_sgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _sgemv_t_sum0_fast0_bgemm0_m77.dat 

../../testing/testing_ozblas_sgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _sgemv_n_sum0_fast0_bgemm0_n77.dat 
../../testing/testing_ozblas_sgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _sgemv_t_sum0_fast0_bgemm0_n77.dat 



../../testing/testing_ozblas_sgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 > _sgemv_n_sum0_fast1_bgemm0_sq_s.dat 
../../testing/testing_ozblas_sgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 > _sgemv_t_sum0_fast1_bgemm0_sq_s.dat 

../../testing/testing_ozblas_sgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --m=77 > _sgemv_n_sum0_fast1_bgemm0_m77.dat 
../../testing/testing_ozblas_sgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --m=77 > _sgemv_t_sum0_fast1_bgemm0_m77.dat 

../../testing/testing_ozblas_sgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --n=77 > _sgemv_n_sum0_fast1_bgemm0_n77.dat 
../../testing/testing_ozblas_sgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --n=77 > _sgemv_t_sum0_fast1_bgemm0_n77.dat 



../../testing/testing_ozblas_sgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _sgemv_n_sum0_fast0_bgemm1_sq_s.dat 
../../testing/testing_ozblas_sgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _sgemv_t_sum0_fast0_bgemm1_sq_s.dat 

../../testing/testing_ozblas_sgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _sgemv_n_sum0_fast0_bgemm1_m77.dat 
../../testing/testing_ozblas_sgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _sgemv_t_sum0_fast0_bgemm1_m77.dat 

../../testing/testing_ozblas_sgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _sgemv_n_sum0_fast0_bgemm1_n77.dat 
../../testing/testing_ozblas_sgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _sgemv_t_sum0_fast0_bgemm1_n77.dat 



../../testing/testing_ozblas_sgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 > _sgemv_n_sum0_fast1_bgemm1_sq_s.dat 
../../testing/testing_ozblas_sgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 > _sgemv_t_sum0_fast1_bgemm1_sq_s.dat 

../../testing/testing_ozblas_sgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --m=77 > _sgemv_n_sum0_fast1_bgemm1_m77.dat 
../../testing/testing_ozblas_sgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --m=77 > _sgemv_t_sum0_fast1_bgemm1_m77.dat 

../../testing/testing_ozblas_sgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --n=77 > _sgemv_n_sum0_fast1_bgemm1_n77.dat 
../../testing/testing_ozblas_sgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --n=77 > _sgemv_t_sum0_fast1_bgemm1_n77.dat 



#################
# dsgemv
#################
# sum0
../../testing/testing_ozblas_dsgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _dsgemv_n_sum0_fast0_bgemm0_sq_s.dat 
../../testing/testing_ozblas_dsgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _dsgemv_t_sum0_fast0_bgemm0_sq_s.dat 

../../testing/testing_ozblas_dsgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _dsgemv_n_sum0_fast0_bgemm0_m77.dat 
../../testing/testing_ozblas_dsgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _dsgemv_t_sum0_fast0_bgemm0_m77.dat 

../../testing/testing_ozblas_dsgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _dsgemv_n_sum0_fast0_bgemm0_n77.dat 
../../testing/testing_ozblas_dsgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _dsgemv_t_sum0_fast0_bgemm0_n77.dat 



../../testing/testing_ozblas_dsgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 > _dsgemv_n_sum0_fast1_bgemm0_sq_s.dat 
../../testing/testing_ozblas_dsgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 > _dsgemv_t_sum0_fast1_bgemm0_sq_s.dat 

../../testing/testing_ozblas_dsgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --m=77 > _dsgemv_n_sum0_fast1_bgemm0_m77.dat 
../../testing/testing_ozblas_dsgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --m=77 > _dsgemv_t_sum0_fast1_bgemm0_m77.dat 

../../testing/testing_ozblas_dsgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --n=77 > _dsgemv_n_sum0_fast1_bgemm0_n77.dat 
../../testing/testing_ozblas_dsgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --n=77 > _dsgemv_t_sum0_fast1_bgemm0_n77.dat 



../../testing/testing_ozblas_dsgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _dsgemv_n_sum0_fast0_bgemm1_sq_s.dat 
../../testing/testing_ozblas_dsgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _dsgemv_t_sum0_fast0_bgemm1_sq_s.dat 

../../testing/testing_ozblas_dsgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _dsgemv_n_sum0_fast0_bgemm1_m77.dat 
../../testing/testing_ozblas_dsgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _dsgemv_t_sum0_fast0_bgemm1_m77.dat 

../../testing/testing_ozblas_dsgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _dsgemv_n_sum0_fast0_bgemm1_n77.dat 
../../testing/testing_ozblas_dsgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _dsgemv_t_sum0_fast0_bgemm1_n77.dat 



../../testing/testing_ozblas_dsgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 > _dsgemv_n_sum0_fast1_bgemm1_sq_s.dat 
../../testing/testing_ozblas_dsgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 > _dsgemv_t_sum0_fast1_bgemm1_sq_s.dat 

../../testing/testing_ozblas_dsgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --m=77 > _dsgemv_n_sum0_fast1_bgemm1_m77.dat 
../../testing/testing_ozblas_dsgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --m=77 > _dsgemv_t_sum0_fast1_bgemm1_m77.dat 

../../testing/testing_ozblas_dsgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --n=77 > _dsgemv_n_sum0_fast1_bgemm1_n77.dat 
../../testing/testing_ozblas_dsgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --n=77 > _dsgemv_t_sum0_fast1_bgemm1_n77.dat 



#################
# qsgemv
#################
# sum0
../../testing/testing_ozblas_qsgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _qsgemv_n_sum0_fast0_bgemm0_sq_s.dat 
../../testing/testing_ozblas_qsgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _qsgemv_t_sum0_fast0_bgemm0_sq_s.dat 

../../testing/testing_ozblas_qsgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _qsgemv_n_sum0_fast0_bgemm0_m77.dat 
../../testing/testing_ozblas_qsgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _qsgemv_t_sum0_fast0_bgemm0_m77.dat 

../../testing/testing_ozblas_qsgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _qsgemv_n_sum0_fast0_bgemm0_n77.dat 
../../testing/testing_ozblas_qsgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _qsgemv_t_sum0_fast0_bgemm0_n77.dat 



../../testing/testing_ozblas_qsgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 > _qsgemv_n_sum0_fast1_bgemm0_sq_s.dat 
../../testing/testing_ozblas_qsgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 > _qsgemv_t_sum0_fast1_bgemm0_sq_s.dat 

../../testing/testing_ozblas_qsgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --m=77 > _qsgemv_n_sum0_fast1_bgemm0_m77.dat 
../../testing/testing_ozblas_qsgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --m=77 > _qsgemv_t_sum0_fast1_bgemm0_m77.dat 

../../testing/testing_ozblas_qsgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --n=77 > _qsgemv_n_sum0_fast1_bgemm0_n77.dat 
../../testing/testing_ozblas_qsgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --n=77 > _qsgemv_t_sum0_fast1_bgemm0_n77.dat 



../../testing/testing_ozblas_qsgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _qsgemv_n_sum0_fast0_bgemm1_sq_s.dat 
../../testing/testing_ozblas_qsgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _qsgemv_t_sum0_fast0_bgemm1_sq_s.dat 

../../testing/testing_ozblas_qsgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _qsgemv_n_sum0_fast0_bgemm1_m77.dat 
../../testing/testing_ozblas_qsgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _qsgemv_t_sum0_fast0_bgemm1_m77.dat 

../../testing/testing_ozblas_qsgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _qsgemv_n_sum0_fast0_bgemm1_n77.dat 
../../testing/testing_ozblas_qsgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _qsgemv_t_sum0_fast0_bgemm1_n77.dat 



../../testing/testing_ozblas_qsgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 > _qsgemv_n_sum0_fast1_bgemm1_sq_s.dat 
../../testing/testing_ozblas_qsgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 > _qsgemv_t_sum0_fast1_bgemm1_sq_s.dat 

../../testing/testing_ozblas_qsgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --m=77 > _qsgemv_n_sum0_fast1_bgemm1_m77.dat 
../../testing/testing_ozblas_qsgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --m=77 > _qsgemv_t_sum0_fast1_bgemm1_m77.dat 

../../testing/testing_ozblas_qsgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --n=77 > _qsgemv_n_sum0_fast1_bgemm1_n77.dat 
../../testing/testing_ozblas_qsgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --n=77 > _qsgemv_t_sum0_fast1_bgemm1_n77.dat 



#################
# qdgemv
#################
# sum0
../../testing/testing_ozblas_qdgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _qdgemv_n_sum0_fast0_bgemm0_sq_s.dat 
../../testing/testing_ozblas_qdgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _qdgemv_t_sum0_fast0_bgemm0_sq_s.dat 

../../testing/testing_ozblas_qdgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _qdgemv_n_sum0_fast0_bgemm0_m77.dat 
../../testing/testing_ozblas_qdgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _qdgemv_t_sum0_fast0_bgemm0_m77.dat 

../../testing/testing_ozblas_qdgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _qdgemv_n_sum0_fast0_bgemm0_n77.dat 
../../testing/testing_ozblas_qdgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _qdgemv_t_sum0_fast0_bgemm0_n77.dat 



../../testing/testing_ozblas_qdgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 > _qdgemv_n_sum0_fast1_bgemm0_sq_s.dat 
../../testing/testing_ozblas_qdgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 > _qdgemv_t_sum0_fast1_bgemm0_sq_s.dat 

../../testing/testing_ozblas_qdgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --m=77 > _qdgemv_n_sum0_fast1_bgemm0_m77.dat 
../../testing/testing_ozblas_qdgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --m=77 > _qdgemv_t_sum0_fast1_bgemm0_m77.dat 

../../testing/testing_ozblas_qdgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --n=77 > _qdgemv_n_sum0_fast1_bgemm0_n77.dat 
../../testing/testing_ozblas_qdgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --n=77 > _qdgemv_t_sum0_fast1_bgemm0_n77.dat 



../../testing/testing_ozblas_qdgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _qdgemv_n_sum0_fast0_bgemm1_sq_s.dat 
../../testing/testing_ozblas_qdgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _qdgemv_t_sum0_fast0_bgemm1_sq_s.dat 

../../testing/testing_ozblas_qdgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _qdgemv_n_sum0_fast0_bgemm1_m77.dat 
../../testing/testing_ozblas_qdgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _qdgemv_t_sum0_fast0_bgemm1_m77.dat 

../../testing/testing_ozblas_qdgemv --nodisp=1 --transa=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _qdgemv_n_sum0_fast0_bgemm1_n77.dat 
../../testing/testing_ozblas_qdgemv --nodisp=1 --transa=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _qdgemv_t_sum0_fast0_bgemm1_n77.dat 



../../testing/testing_ozblas_qdgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 > _qdgemv_n_sum0_fast1_bgemm1_sq_s.dat 
../../testing/testing_ozblas_qdgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 > _qdgemv_t_sum0_fast1_bgemm1_sq_s.dat 

../../testing/testing_ozblas_qdgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --m=77 > _qdgemv_n_sum0_fast1_bgemm1_m77.dat 
../../testing/testing_ozblas_qdgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --m=77 > _qdgemv_t_sum0_fast1_bgemm1_m77.dat 

../../testing/testing_ozblas_qdgemv --nodisp=1 --transa=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --n=77 > _qdgemv_n_sum0_fast1_bgemm1_n77.dat 
../../testing/testing_ozblas_qdgemv --nodisp=1 --transa=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --n=77 > _qdgemv_t_sum0_fast1_bgemm1_n77.dat 


