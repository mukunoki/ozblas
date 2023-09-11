#!/bin/bash
#SBATCH --job-name=jobname
#SBATCH --partition=a100
#SBATCH -t 02:00:00

module load system/x86_64
export OMP_NUM_THREADS=256

#################
# dgemm
#################
# sum1
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _dgemm_nn_sum1_fast0_bgemm0_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _dgemm_nt_sum1_fast0_bgemm0_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _dgemm_tn_sum1_fast0_bgemm0_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _dgemm_tt_sum1_fast0_bgemm0_sq_s.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _dgemm_nn_sum1_fast0_bgemm0_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _dgemm_nt_sum1_fast0_bgemm0_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _dgemm_tn_sum1_fast0_bgemm0_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _dgemm_tt_sum1_fast0_bgemm0_m77.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _dgemm_nn_sum1_fast0_bgemm0_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _dgemm_nt_sum1_fast0_bgemm0_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _dgemm_tn_sum1_fast0_bgemm0_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _dgemm_tt_sum1_fast0_bgemm0_n77.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --k=77 > _dgemm_nn_sum1_fast0_bgemm0_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --k=77 > _dgemm_nt_sum1_fast0_bgemm0_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --k=77 > _dgemm_tn_sum1_fast0_bgemm0_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --k=77 > _dgemm_tt_sum1_fast0_bgemm0_k77.dat 



../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _dgemm_nn_sum1_fast0_bgemm1_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _dgemm_nt_sum1_fast0_bgemm1_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _dgemm_tn_sum1_fast0_bgemm1_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _dgemm_tt_sum1_fast0_bgemm1_sq_s.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _dgemm_nn_sum1_fast0_bgemm1_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _dgemm_nt_sum1_fast0_bgemm1_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _dgemm_tn_sum1_fast0_bgemm1_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _dgemm_tt_sum1_fast0_bgemm1_m77.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _dgemm_nn_sum1_fast0_bgemm1_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _dgemm_nt_sum1_fast0_bgemm1_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _dgemm_tn_sum1_fast0_bgemm1_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _dgemm_tt_sum1_fast0_bgemm1_n77.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --k=77 > _dgemm_nn_sum1_fast0_bgemm1_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --k=77 > _dgemm_nt_sum1_fast0_bgemm1_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --k=77 > _dgemm_tn_sum1_fast0_bgemm1_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --k=77 > _dgemm_tt_sum1_fast0_bgemm1_k77.dat 



# sum0
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _dgemm_nn_sum0_fast0_bgemm0_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _dgemm_nt_sum0_fast0_bgemm0_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _dgemm_tn_sum0_fast0_bgemm0_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _dgemm_tt_sum0_fast0_bgemm0_sq_s.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _dgemm_nn_sum0_fast0_bgemm0_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _dgemm_nt_sum0_fast0_bgemm0_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _dgemm_tn_sum0_fast0_bgemm0_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _dgemm_tt_sum0_fast0_bgemm0_m77.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _dgemm_nn_sum0_fast0_bgemm0_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _dgemm_nt_sum0_fast0_bgemm0_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _dgemm_tn_sum0_fast0_bgemm0_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _dgemm_tt_sum0_fast0_bgemm0_n77.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --k=77 > _dgemm_nn_sum0_fast0_bgemm0_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --k=77 > _dgemm_nt_sum0_fast0_bgemm0_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --k=77 > _dgemm_tn_sum0_fast0_bgemm0_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --k=77 > _dgemm_tt_sum0_fast0_bgemm0_k77.dat 



../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 > _dgemm_nn_sum0_fast1_bgemm0_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 > _dgemm_nt_sum0_fast1_bgemm0_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 > _dgemm_tn_sum0_fast1_bgemm0_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 > _dgemm_tt_sum0_fast1_bgemm0_sq_s.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --m=77 > _dgemm_nn_sum0_fast1_bgemm0_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --m=77 > _dgemm_nt_sum0_fast1_bgemm0_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --m=77 > _dgemm_tn_sum0_fast1_bgemm0_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --m=77 > _dgemm_tt_sum0_fast1_bgemm0_m77.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --n=77 > _dgemm_nn_sum0_fast1_bgemm0_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --n=77 > _dgemm_nt_sum0_fast1_bgemm0_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --n=77 > _dgemm_tn_sum0_fast1_bgemm0_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --n=77 > _dgemm_tt_sum0_fast1_bgemm0_n77.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --k=77 > _dgemm_nn_sum0_fast1_bgemm0_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --k=77 > _dgemm_nt_sum0_fast1_bgemm0_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --k=77 > _dgemm_tn_sum0_fast1_bgemm0_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --k=77 > _dgemm_tt_sum0_fast1_bgemm0_k77.dat 



../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _dgemm_nn_sum0_fast0_bgemm1_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _dgemm_nt_sum0_fast0_bgemm1_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _dgemm_tn_sum0_fast0_bgemm1_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _dgemm_tt_sum0_fast0_bgemm1_sq_s.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _dgemm_nn_sum0_fast0_bgemm1_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _dgemm_nt_sum0_fast0_bgemm1_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _dgemm_tn_sum0_fast0_bgemm1_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _dgemm_tt_sum0_fast0_bgemm1_m77.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _dgemm_nn_sum0_fast0_bgemm1_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _dgemm_nt_sum0_fast0_bgemm1_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _dgemm_tn_sum0_fast0_bgemm1_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _dgemm_tt_sum0_fast0_bgemm1_n77.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --k=77 > _dgemm_nn_sum0_fast0_bgemm1_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --k=77 > _dgemm_nt_sum0_fast0_bgemm1_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --k=77 > _dgemm_tn_sum0_fast0_bgemm1_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --k=77 > _dgemm_tt_sum0_fast0_bgemm1_k77.dat 



../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 > _dgemm_nn_sum0_fast1_bgemm1_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 > _dgemm_nt_sum0_fast1_bgemm1_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 > _dgemm_tn_sum0_fast1_bgemm1_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 > _dgemm_tt_sum0_fast1_bgemm1_sq_s.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --m=77 > _dgemm_nn_sum0_fast1_bgemm1_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --m=77 > _dgemm_nt_sum0_fast1_bgemm1_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --m=77 > _dgemm_tn_sum0_fast1_bgemm1_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --m=77 > _dgemm_tt_sum0_fast1_bgemm1_m77.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --n=77 > _dgemm_nn_sum0_fast1_bgemm1_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --n=77 > _dgemm_nt_sum0_fast1_bgemm1_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --n=77 > _dgemm_tn_sum0_fast1_bgemm1_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --n=77 > _dgemm_tt_sum0_fast1_bgemm1_n77.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --k=77 > _dgemm_nn_sum0_fast1_bgemm1_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --k=77 > _dgemm_nt_sum0_fast1_bgemm1_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --k=77 > _dgemm_tn_sum0_fast1_bgemm1_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --k=77 > _dgemm_tt_sum0_fast1_bgemm1_k77.dat 


#################
# sgemm
#################
# sum1
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=n --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _sgemm_nn_sum1_fast0_bgemm0_sq_s.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=t --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _sgemm_nt_sum1_fast0_bgemm0_sq_s.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=n --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _sgemm_tn_sum1_fast0_bgemm0_sq_s.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=t --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _sgemm_tt_sum1_fast0_bgemm0_sq_s.dat 

../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=n --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _sgemm_nn_sum1_fast0_bgemm0_m77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=t --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _sgemm_nt_sum1_fast0_bgemm0_m77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=n --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _sgemm_tn_sum1_fast0_bgemm0_m77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=t --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _sgemm_tt_sum1_fast0_bgemm0_m77.dat 

../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=n --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _sgemm_nn_sum1_fast0_bgemm0_n77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=t --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _sgemm_nt_sum1_fast0_bgemm0_n77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=n --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _sgemm_tn_sum1_fast0_bgemm0_n77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=t --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _sgemm_tt_sum1_fast0_bgemm0_n77.dat 

../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=n --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --k=77 > _sgemm_nn_sum1_fast0_bgemm0_k77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=t --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --k=77 > _sgemm_nt_sum1_fast0_bgemm0_k77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=n --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --k=77 > _sgemm_tn_sum1_fast0_bgemm0_k77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=t --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --k=77 > _sgemm_tt_sum1_fast0_bgemm0_k77.dat 



../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=n --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _sgemm_nn_sum1_fast0_bgemm1_sq_s.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=t --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _sgemm_nt_sum1_fast0_bgemm1_sq_s.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=n --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _sgemm_tn_sum1_fast0_bgemm1_sq_s.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=t --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _sgemm_tt_sum1_fast0_bgemm1_sq_s.dat 

../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=n --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _sgemm_nn_sum1_fast0_bgemm1_m77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=t --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _sgemm_nt_sum1_fast0_bgemm1_m77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=n --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _sgemm_tn_sum1_fast0_bgemm1_m77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=t --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _sgemm_tt_sum1_fast0_bgemm1_m77.dat 

../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=n --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _sgemm_nn_sum1_fast0_bgemm1_n77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=t --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _sgemm_nt_sum1_fast0_bgemm1_n77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=n --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _sgemm_tn_sum1_fast0_bgemm1_n77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=t --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _sgemm_tt_sum1_fast0_bgemm1_n77.dat 

../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=n --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --k=77 > _sgemm_nn_sum1_fast0_bgemm1_k77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=t --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --k=77 > _sgemm_nt_sum1_fast0_bgemm1_k77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=n --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --k=77 > _sgemm_tn_sum1_fast0_bgemm1_k77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=t --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --k=77 > _sgemm_tt_sum1_fast0_bgemm1_k77.dat 



# sum0
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _sgemm_nn_sum0_fast0_bgemm0_sq_s.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _sgemm_nt_sum0_fast0_bgemm0_sq_s.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _sgemm_tn_sum0_fast0_bgemm0_sq_s.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _sgemm_tt_sum0_fast0_bgemm0_sq_s.dat 

../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _sgemm_nn_sum0_fast0_bgemm0_m77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _sgemm_nt_sum0_fast0_bgemm0_m77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _sgemm_tn_sum0_fast0_bgemm0_m77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _sgemm_tt_sum0_fast0_bgemm0_m77.dat 

../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _sgemm_nn_sum0_fast0_bgemm0_n77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _sgemm_nt_sum0_fast0_bgemm0_n77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _sgemm_tn_sum0_fast0_bgemm0_n77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _sgemm_tt_sum0_fast0_bgemm0_n77.dat 

../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --k=77 > _sgemm_nn_sum0_fast0_bgemm0_k77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --k=77 > _sgemm_nt_sum0_fast0_bgemm0_k77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --k=77 > _sgemm_tn_sum0_fast0_bgemm0_k77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --k=77 > _sgemm_tt_sum0_fast0_bgemm0_k77.dat 



../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 > _sgemm_nn_sum0_fast1_bgemm0_sq_s.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 > _sgemm_nt_sum0_fast1_bgemm0_sq_s.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 > _sgemm_tn_sum0_fast1_bgemm0_sq_s.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 > _sgemm_tt_sum0_fast1_bgemm0_sq_s.dat 

../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --m=77 > _sgemm_nn_sum0_fast1_bgemm0_m77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --m=77 > _sgemm_nt_sum0_fast1_bgemm0_m77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --m=77 > _sgemm_tn_sum0_fast1_bgemm0_m77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --m=77 > _sgemm_tt_sum0_fast1_bgemm0_m77.dat 

../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --n=77 > _sgemm_nn_sum0_fast1_bgemm0_n77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --n=77 > _sgemm_nt_sum0_fast1_bgemm0_n77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --n=77 > _sgemm_tn_sum0_fast1_bgemm0_n77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --n=77 > _sgemm_tt_sum0_fast1_bgemm0_n77.dat 

../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --k=77 > _sgemm_nn_sum0_fast1_bgemm0_k77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --k=77 > _sgemm_nt_sum0_fast1_bgemm0_k77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --k=77 > _sgemm_tn_sum0_fast1_bgemm0_k77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --k=77 > _sgemm_tt_sum0_fast1_bgemm0_k77.dat 



../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _sgemm_nn_sum0_fast0_bgemm1_sq_s.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _sgemm_nt_sum0_fast0_bgemm1_sq_s.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _sgemm_tn_sum0_fast0_bgemm1_sq_s.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _sgemm_tt_sum0_fast0_bgemm1_sq_s.dat 

../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _sgemm_nn_sum0_fast0_bgemm1_m77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _sgemm_nt_sum0_fast0_bgemm1_m77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _sgemm_tn_sum0_fast0_bgemm1_m77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _sgemm_tt_sum0_fast0_bgemm1_m77.dat 

../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _sgemm_nn_sum0_fast0_bgemm1_n77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _sgemm_nt_sum0_fast0_bgemm1_n77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _sgemm_tn_sum0_fast0_bgemm1_n77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _sgemm_tt_sum0_fast0_bgemm1_n77.dat 

../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --k=77 > _sgemm_nn_sum0_fast0_bgemm1_k77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --k=77 > _sgemm_nt_sum0_fast0_bgemm1_k77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --k=77 > _sgemm_tn_sum0_fast0_bgemm1_k77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --k=77 > _sgemm_tt_sum0_fast0_bgemm1_k77.dat 



../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 > _sgemm_nn_sum0_fast1_bgemm1_sq_s.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 > _sgemm_nt_sum0_fast1_bgemm1_sq_s.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 > _sgemm_tn_sum0_fast1_bgemm1_sq_s.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 > _sgemm_tt_sum0_fast1_bgemm1_sq_s.dat 

../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --m=77 > _sgemm_nn_sum0_fast1_bgemm1_m77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --m=77 > _sgemm_nt_sum0_fast1_bgemm1_m77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --m=77 > _sgemm_tn_sum0_fast1_bgemm1_m77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --m=77 > _sgemm_tt_sum0_fast1_bgemm1_m77.dat 

../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --n=77 > _sgemm_nn_sum0_fast1_bgemm1_n77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --n=77 > _sgemm_nt_sum0_fast1_bgemm1_n77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --n=77 > _sgemm_tn_sum0_fast1_bgemm1_n77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --n=77 > _sgemm_tt_sum0_fast1_bgemm1_n77.dat 

../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --k=77 > _sgemm_nn_sum0_fast1_bgemm1_k77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --k=77 > _sgemm_nt_sum0_fast1_bgemm1_k77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --k=77 > _sgemm_tn_sum0_fast1_bgemm1_k77.dat 
../../testing/testing_ozblas_sgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --k=77 > _sgemm_tt_sum0_fast1_bgemm1_k77.dat 


#################
# dsgemm
#################
# sum0
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _dsgemm_nn_sum0_fast0_bgemm0_sq_s.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _dsgemm_nt_sum0_fast0_bgemm0_sq_s.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _dsgemm_tn_sum0_fast0_bgemm0_sq_s.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _dsgemm_tt_sum0_fast0_bgemm0_sq_s.dat 

../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _dsgemm_nn_sum0_fast0_bgemm0_m77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _dsgemm_nt_sum0_fast0_bgemm0_m77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _dsgemm_tn_sum0_fast0_bgemm0_m77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _dsgemm_tt_sum0_fast0_bgemm0_m77.dat 

../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _dsgemm_nn_sum0_fast0_bgemm0_n77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _dsgemm_nt_sum0_fast0_bgemm0_n77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _dsgemm_tn_sum0_fast0_bgemm0_n77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _dsgemm_tt_sum0_fast0_bgemm0_n77.dat 

../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --k=77 > _dsgemm_nn_sum0_fast0_bgemm0_k77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --k=77 > _dsgemm_nt_sum0_fast0_bgemm0_k77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --k=77 > _dsgemm_tn_sum0_fast0_bgemm0_k77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --k=77 > _dsgemm_tt_sum0_fast0_bgemm0_k77.dat 



../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 > _dsgemm_nn_sum0_fast1_bgemm0_sq_s.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 > _dsgemm_nt_sum0_fast1_bgemm0_sq_s.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 > _dsgemm_tn_sum0_fast1_bgemm0_sq_s.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 > _dsgemm_tt_sum0_fast1_bgemm0_sq_s.dat 

../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --m=77 > _dsgemm_nn_sum0_fast1_bgemm0_m77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --m=77 > _dsgemm_nt_sum0_fast1_bgemm0_m77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --m=77 > _dsgemm_tn_sum0_fast1_bgemm0_m77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --m=77 > _dsgemm_tt_sum0_fast1_bgemm0_m77.dat 

../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --n=77 > _dsgemm_nn_sum0_fast1_bgemm0_n77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --n=77 > _dsgemm_nt_sum0_fast1_bgemm0_n77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --n=77 > _dsgemm_tn_sum0_fast1_bgemm0_n77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --n=77 > _dsgemm_tt_sum0_fast1_bgemm0_n77.dat 

../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --k=77 > _dsgemm_nn_sum0_fast1_bgemm0_k77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --k=77 > _dsgemm_nt_sum0_fast1_bgemm0_k77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --k=77 > _dsgemm_tn_sum0_fast1_bgemm0_k77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --k=77 > _dsgemm_tt_sum0_fast1_bgemm0_k77.dat 



../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _dsgemm_nn_sum0_fast0_bgemm1_sq_s.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _dsgemm_nt_sum0_fast0_bgemm1_sq_s.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _dsgemm_tn_sum0_fast0_bgemm1_sq_s.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _dsgemm_tt_sum0_fast0_bgemm1_sq_s.dat 

../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _dsgemm_nn_sum0_fast0_bgemm1_m77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _dsgemm_nt_sum0_fast0_bgemm1_m77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _dsgemm_tn_sum0_fast0_bgemm1_m77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _dsgemm_tt_sum0_fast0_bgemm1_m77.dat 

../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _dsgemm_nn_sum0_fast0_bgemm1_n77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _dsgemm_nt_sum0_fast0_bgemm1_n77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _dsgemm_tn_sum0_fast0_bgemm1_n77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _dsgemm_tt_sum0_fast0_bgemm1_n77.dat 

../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --k=77 > _dsgemm_nn_sum0_fast0_bgemm1_k77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --k=77 > _dsgemm_nt_sum0_fast0_bgemm1_k77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --k=77 > _dsgemm_tn_sum0_fast0_bgemm1_k77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --k=77 > _dsgemm_tt_sum0_fast0_bgemm1_k77.dat 



../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 > _dsgemm_nn_sum0_fast1_bgemm1_sq_s.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 > _dsgemm_nt_sum0_fast1_bgemm1_sq_s.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 > _dsgemm_tn_sum0_fast1_bgemm1_sq_s.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 > _dsgemm_tt_sum0_fast1_bgemm1_sq_s.dat 

../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --m=77 > _dsgemm_nn_sum0_fast1_bgemm1_m77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --m=77 > _dsgemm_nt_sum0_fast1_bgemm1_m77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --m=77 > _dsgemm_tn_sum0_fast1_bgemm1_m77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --m=77 > _dsgemm_tt_sum0_fast1_bgemm1_m77.dat 

../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --n=77 > _dsgemm_nn_sum0_fast1_bgemm1_n77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --n=77 > _dsgemm_nt_sum0_fast1_bgemm1_n77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --n=77 > _dsgemm_tn_sum0_fast1_bgemm1_n77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --n=77 > _dsgemm_tt_sum0_fast1_bgemm1_n77.dat 

../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --k=77 > _dsgemm_nn_sum0_fast1_bgemm1_k77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --k=77 > _dsgemm_nt_sum0_fast1_bgemm1_k77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --k=77 > _dsgemm_tn_sum0_fast1_bgemm1_k77.dat 
../../testing/testing_ozblas_dsgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --k=77 > _dsgemm_tt_sum0_fast1_bgemm1_k77.dat 


#################
# qsgemm
#################
# sum0
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _qsgemm_nn_sum0_fast0_bgemm0_sq_s.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _qsgemm_nt_sum0_fast0_bgemm0_sq_s.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _qsgemm_tn_sum0_fast0_bgemm0_sq_s.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _qsgemm_tt_sum0_fast0_bgemm0_sq_s.dat 

../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _qsgemm_nn_sum0_fast0_bgemm0_m77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _qsgemm_nt_sum0_fast0_bgemm0_m77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _qsgemm_tn_sum0_fast0_bgemm0_m77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _qsgemm_tt_sum0_fast0_bgemm0_m77.dat 

../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _qsgemm_nn_sum0_fast0_bgemm0_n77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _qsgemm_nt_sum0_fast0_bgemm0_n77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _qsgemm_tn_sum0_fast0_bgemm0_n77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _qsgemm_tt_sum0_fast0_bgemm0_n77.dat 

../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --k=77 > _qsgemm_nn_sum0_fast0_bgemm0_k77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --k=77 > _qsgemm_nt_sum0_fast0_bgemm0_k77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --k=77 > _qsgemm_tn_sum0_fast0_bgemm0_k77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --k=77 > _qsgemm_tt_sum0_fast0_bgemm0_k77.dat 



../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 > _qsgemm_nn_sum0_fast1_bgemm0_sq_s.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 > _qsgemm_nt_sum0_fast1_bgemm0_sq_s.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 > _qsgemm_tn_sum0_fast1_bgemm0_sq_s.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 > _qsgemm_tt_sum0_fast1_bgemm0_sq_s.dat 

../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --m=77 > _qsgemm_nn_sum0_fast1_bgemm0_m77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --m=77 > _qsgemm_nt_sum0_fast1_bgemm0_m77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --m=77 > _qsgemm_tn_sum0_fast1_bgemm0_m77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --m=77 > _qsgemm_tt_sum0_fast1_bgemm0_m77.dat 

../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --n=77 > _qsgemm_nn_sum0_fast1_bgemm0_n77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --n=77 > _qsgemm_nt_sum0_fast1_bgemm0_n77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --n=77 > _qsgemm_tn_sum0_fast1_bgemm0_n77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --n=77 > _qsgemm_tt_sum0_fast1_bgemm0_n77.dat 

../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --k=77 > _qsgemm_nn_sum0_fast1_bgemm0_k77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --k=77 > _qsgemm_nt_sum0_fast1_bgemm0_k77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --k=77 > _qsgemm_tn_sum0_fast1_bgemm0_k77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --k=77 > _qsgemm_tt_sum0_fast1_bgemm0_k77.dat 



../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _qsgemm_nn_sum0_fast0_bgemm1_sq_s.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _qsgemm_nt_sum0_fast0_bgemm1_sq_s.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _qsgemm_tn_sum0_fast0_bgemm1_sq_s.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _qsgemm_tt_sum0_fast0_bgemm1_sq_s.dat 

../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _qsgemm_nn_sum0_fast0_bgemm1_m77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _qsgemm_nt_sum0_fast0_bgemm1_m77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _qsgemm_tn_sum0_fast0_bgemm1_m77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _qsgemm_tt_sum0_fast0_bgemm1_m77.dat 

../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _qsgemm_nn_sum0_fast0_bgemm1_n77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _qsgemm_nt_sum0_fast0_bgemm1_n77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _qsgemm_tn_sum0_fast0_bgemm1_n77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _qsgemm_tt_sum0_fast0_bgemm1_n77.dat 

../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --k=77 > _qsgemm_nn_sum0_fast0_bgemm1_k77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --k=77 > _qsgemm_nt_sum0_fast0_bgemm1_k77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --k=77 > _qsgemm_tn_sum0_fast0_bgemm1_k77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --k=77 > _qsgemm_tt_sum0_fast0_bgemm1_k77.dat 



../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 > _qsgemm_nn_sum0_fast1_bgemm1_sq_s.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 > _qsgemm_nt_sum0_fast1_bgemm1_sq_s.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 > _qsgemm_tn_sum0_fast1_bgemm1_sq_s.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 > _qsgemm_tt_sum0_fast1_bgemm1_sq_s.dat 

../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --m=77 > _qsgemm_nn_sum0_fast1_bgemm1_m77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --m=77 > _qsgemm_nt_sum0_fast1_bgemm1_m77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --m=77 > _qsgemm_tn_sum0_fast1_bgemm1_m77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --m=77 > _qsgemm_tt_sum0_fast1_bgemm1_m77.dat 

../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --n=77 > _qsgemm_nn_sum0_fast1_bgemm1_n77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --n=77 > _qsgemm_nt_sum0_fast1_bgemm1_n77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --n=77 > _qsgemm_tn_sum0_fast1_bgemm1_n77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --n=77 > _qsgemm_tt_sum0_fast1_bgemm1_n77.dat 

../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --k=77 > _qsgemm_nn_sum0_fast1_bgemm1_k77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --k=77 > _qsgemm_nt_sum0_fast1_bgemm1_k77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --k=77 > _qsgemm_tn_sum0_fast1_bgemm1_k77.dat 
../../testing/testing_ozblas_qsgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --k=77 > _qsgemm_tt_sum0_fast1_bgemm1_k77.dat 



#################
# qdgemm
#################
# sum0
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _qdgemm_nn_sum0_fast0_bgemm0_sq_s.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _qdgemm_nt_sum0_fast0_bgemm0_sq_s.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _qdgemm_tn_sum0_fast0_bgemm0_sq_s.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 > _qdgemm_tt_sum0_fast0_bgemm0_sq_s.dat 

../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _qdgemm_nn_sum0_fast0_bgemm0_m77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _qdgemm_nt_sum0_fast0_bgemm0_m77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _qdgemm_tn_sum0_fast0_bgemm0_m77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --m=77 > _qdgemm_tt_sum0_fast0_bgemm0_m77.dat 

../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _qdgemm_nn_sum0_fast0_bgemm0_n77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _qdgemm_nt_sum0_fast0_bgemm0_n77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _qdgemm_tn_sum0_fast0_bgemm0_n77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --n=77 > _qdgemm_tt_sum0_fast0_bgemm0_n77.dat 

../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --k=77 > _qdgemm_nn_sum0_fast0_bgemm0_k77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --k=77 > _qdgemm_nt_sum0_fast0_bgemm0_k77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --k=77 > _qdgemm_tn_sum0_fast0_bgemm0_k77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:33 --k=77 > _qdgemm_tt_sum0_fast0_bgemm0_k77.dat 



../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 > _qdgemm_nn_sum0_fast1_bgemm0_sq_s.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 > _qdgemm_nt_sum0_fast1_bgemm0_sq_s.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 > _qdgemm_tn_sum0_fast1_bgemm0_sq_s.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 > _qdgemm_tt_sum0_fast1_bgemm0_sq_s.dat 

../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --m=77 > _qdgemm_nn_sum0_fast1_bgemm0_m77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --m=77 > _qdgemm_nt_sum0_fast1_bgemm0_m77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --m=77 > _qdgemm_tn_sum0_fast1_bgemm0_m77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --m=77 > _qdgemm_tt_sum0_fast1_bgemm0_m77.dat 

../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --n=77 > _qdgemm_nn_sum0_fast1_bgemm0_n77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --n=77 > _qdgemm_nt_sum0_fast1_bgemm0_n77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --n=77 > _qdgemm_tn_sum0_fast1_bgemm0_n77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --n=77 > _qdgemm_tt_sum0_fast1_bgemm0_n77.dat 

../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --k=77 > _qdgemm_nn_sum0_fast1_bgemm0_k77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --k=77 > _qdgemm_nt_sum0_fast1_bgemm0_k77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --k=77 > _qdgemm_tn_sum0_fast1_bgemm0_k77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:33 --k=77 > _qdgemm_tt_sum0_fast1_bgemm0_k77.dat 



../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _qdgemm_nn_sum0_fast0_bgemm1_sq_s.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _qdgemm_nt_sum0_fast0_bgemm1_sq_s.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _qdgemm_tn_sum0_fast0_bgemm1_sq_s.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 > _qdgemm_tt_sum0_fast0_bgemm1_sq_s.dat 

../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _qdgemm_nn_sum0_fast0_bgemm1_m77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _qdgemm_nt_sum0_fast0_bgemm1_m77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _qdgemm_tn_sum0_fast0_bgemm1_m77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --m=77 > _qdgemm_tt_sum0_fast0_bgemm1_m77.dat 

../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _qdgemm_nn_sum0_fast0_bgemm1_n77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _qdgemm_nt_sum0_fast0_bgemm1_n77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _qdgemm_tn_sum0_fast0_bgemm1_n77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --n=77 > _qdgemm_tt_sum0_fast0_bgemm1_n77.dat 

../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --k=77 > _qdgemm_nn_sum0_fast0_bgemm1_k77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --k=77 > _qdgemm_nt_sum0_fast0_bgemm1_k77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --k=77 > _qdgemm_tn_sum0_fast0_bgemm1_k77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:33 --k=77 > _qdgemm_tt_sum0_fast0_bgemm1_k77.dat 



../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 > _qdgemm_nn_sum0_fast1_bgemm1_sq_s.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 > _qdgemm_nt_sum0_fast1_bgemm1_sq_s.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 > _qdgemm_tn_sum0_fast1_bgemm1_sq_s.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 > _qdgemm_tt_sum0_fast1_bgemm1_sq_s.dat 

../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --m=77 > _qdgemm_nn_sum0_fast1_bgemm1_m77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --m=77 > _qdgemm_nt_sum0_fast1_bgemm1_m77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --m=77 > _qdgemm_tn_sum0_fast1_bgemm1_m77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --m=77 > _qdgemm_tt_sum0_fast1_bgemm1_m77.dat 

../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --n=77 > _qdgemm_nn_sum0_fast1_bgemm1_n77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --n=77 > _qdgemm_nt_sum0_fast1_bgemm1_n77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --n=77 > _qdgemm_tn_sum0_fast1_bgemm1_n77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --n=77 > _qdgemm_tt_sum0_fast1_bgemm1_n77.dat 

../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --k=77 > _qdgemm_nn_sum0_fast1_bgemm1_k77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --k=77 > _qdgemm_nt_sum0_fast1_bgemm1_k77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --k=77 > _qdgemm_tn_sum0_fast1_bgemm1_k77.dat 
../../testing/testing_ozblas_qdgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:33 --k=77 > _qdgemm_tt_sum0_fast1_bgemm1_k77.dat 


