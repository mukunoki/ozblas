#!/bin/sh
#------ pjsub option --------#
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM --omp thread=36
#PJM -L elapse=0:45:00
#PJM -g jh180023a
#PJM -j
#------- Program execution -------

module load intel/2023.1.0
module load cuda/12.1

export OMP_NUM_THREADS=36
export MKL_NUM_THREADS=36

# sum1
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 > _dgemm_nn_sum1_fast0_bgemm0_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 > _dgemm_nt_sum1_fast0_bgemm0_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 > _dgemm_tn_sum1_fast0_bgemm0_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 > _dgemm_tt_sum1_fast0_bgemm0_sq_s.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 --m=77 > _dgemm_nn_sum1_fast0_bgemm0_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 --m=77 > _dgemm_nt_sum1_fast0_bgemm0_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 --m=77 > _dgemm_tn_sum1_fast0_bgemm0_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 --m=77 > _dgemm_tt_sum1_fast0_bgemm0_m77.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 --n=77 > _dgemm_nn_sum1_fast0_bgemm0_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 --n=77 > _dgemm_nt_sum1_fast0_bgemm0_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 --n=77 > _dgemm_tn_sum1_fast0_bgemm0_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 --n=77 > _dgemm_tt_sum1_fast0_bgemm0_n77.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 --k=77 > _dgemm_nn_sum1_fast0_bgemm0_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 --k=77 > _dgemm_nt_sum1_fast0_bgemm0_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 --k=77 > _dgemm_tn_sum1_fast0_bgemm0_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=1 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 --k=77 > _dgemm_tt_sum1_fast0_bgemm0_k77.dat 



../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 > _dgemm_nn_sum1_fast0_bgemm1_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 > _dgemm_nt_sum1_fast0_bgemm1_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 > _dgemm_tn_sum1_fast0_bgemm1_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 > _dgemm_tt_sum1_fast0_bgemm1_sq_s.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 --m=77 > _dgemm_nn_sum1_fast0_bgemm1_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 --m=77 > _dgemm_nt_sum1_fast0_bgemm1_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 --m=77 > _dgemm_tn_sum1_fast0_bgemm1_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 --m=77 > _dgemm_tt_sum1_fast0_bgemm1_m77.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 --n=77 > _dgemm_nn_sum1_fast0_bgemm1_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 --n=77 > _dgemm_nt_sum1_fast0_bgemm1_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 --n=77 > _dgemm_tn_sum1_fast0_bgemm1_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 --n=77 > _dgemm_tt_sum1_fast0_bgemm1_n77.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 --k=77 > _dgemm_nn_sum1_fast0_bgemm1_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 --k=77 > _dgemm_nt_sum1_fast0_bgemm1_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 --k=77 > _dgemm_tn_sum1_fast0_bgemm1_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=1 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 --k=77 > _dgemm_tt_sum1_fast0_bgemm1_k77.dat 



# sum0
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 > _dgemm_nn_sum0_fast0_bgemm0_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 > _dgemm_nt_sum0_fast0_bgemm0_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 > _dgemm_tn_sum0_fast0_bgemm0_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 > _dgemm_tt_sum0_fast0_bgemm0_sq_s.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 --m=77 > _dgemm_nn_sum0_fast0_bgemm0_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 --m=77 > _dgemm_nt_sum0_fast0_bgemm0_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 --m=77 > _dgemm_tn_sum0_fast0_bgemm0_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 --m=77 > _dgemm_tt_sum0_fast0_bgemm0_m77.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 --n=77 > _dgemm_nn_sum0_fast0_bgemm0_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 --n=77 > _dgemm_nt_sum0_fast0_bgemm0_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 --n=77 > _dgemm_tn_sum0_fast0_bgemm0_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 --n=77 > _dgemm_tt_sum0_fast0_bgemm0_n77.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 --k=77 > _dgemm_nn_sum0_fast0_bgemm0_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 --k=77 > _dgemm_nt_sum0_fast0_bgemm0_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 --k=77 > _dgemm_tn_sum0_fast0_bgemm0_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=0 --range=1:256:13 --k=77 > _dgemm_tt_sum0_fast0_bgemm0_k77.dat 



../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:13 > _dgemm_nn_sum0_fast1_bgemm0_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:13 > _dgemm_nt_sum0_fast1_bgemm0_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:13 > _dgemm_tn_sum0_fast1_bgemm0_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:13 > _dgemm_tt_sum0_fast1_bgemm0_sq_s.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:13 --m=77 > _dgemm_nn_sum0_fast1_bgemm0_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:13 --m=77 > _dgemm_nt_sum0_fast1_bgemm0_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:13 --m=77 > _dgemm_tn_sum0_fast1_bgemm0_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:13 --m=77 > _dgemm_tt_sum0_fast1_bgemm0_m77.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:13 --n=77 > _dgemm_nn_sum0_fast1_bgemm0_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:13 --n=77 > _dgemm_nt_sum0_fast1_bgemm0_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:13 --n=77 > _dgemm_tn_sum0_fast1_bgemm0_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:13 --n=77 > _dgemm_tt_sum0_fast1_bgemm0_n77.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:13 --k=77 > _dgemm_nn_sum0_fast1_bgemm0_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:13 --k=77 > _dgemm_nt_sum0_fast1_bgemm0_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:13 --k=77 > _dgemm_tn_sum0_fast1_bgemm0_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=0 --range=1:256:13 --k=77 > _dgemm_tt_sum0_fast1_bgemm0_k77.dat 



../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 > _dgemm_nn_sum0_fast0_bgemm1_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 > _dgemm_nt_sum0_fast0_bgemm1_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 > _dgemm_tn_sum0_fast0_bgemm1_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 > _dgemm_tt_sum0_fast0_bgemm1_sq_s.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 --m=77 > _dgemm_nn_sum0_fast0_bgemm1_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 --m=77 > _dgemm_nt_sum0_fast0_bgemm1_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 --m=77 > _dgemm_tn_sum0_fast0_bgemm1_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 --m=77 > _dgemm_tt_sum0_fast0_bgemm1_m77.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 --n=77 > _dgemm_nn_sum0_fast0_bgemm1_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 --n=77 > _dgemm_nt_sum0_fast0_bgemm1_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 --n=77 > _dgemm_tn_sum0_fast0_bgemm1_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 --n=77 > _dgemm_tt_sum0_fast0_bgemm1_n77.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 --k=77 > _dgemm_nn_sum0_fast0_bgemm1_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 --k=77 > _dgemm_nt_sum0_fast0_bgemm1_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 --k=77 > _dgemm_tn_sum0_fast0_bgemm1_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=0 --usebatchedgemm=1 --range=1:256:13 --k=77 > _dgemm_tt_sum0_fast0_bgemm1_k77.dat 



../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:13 > _dgemm_nn_sum0_fast1_bgemm0_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:13 > _dgemm_nt_sum0_fast1_bgemm0_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:13 > _dgemm_tn_sum0_fast1_bgemm0_sq_s.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:13 > _dgemm_tt_sum0_fast1_bgemm0_sq_s.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:13 --m=77 > _dgemm_nn_sum0_fast1_bgemm0_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:13 --m=77 > _dgemm_nt_sum0_fast1_bgemm0_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:13 --m=77 > _dgemm_tn_sum0_fast1_bgemm0_m77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:13 --m=77 > _dgemm_tt_sum0_fast1_bgemm0_m77.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:13 --n=77 > _dgemm_nn_sum0_fast1_bgemm0_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:13 --n=77 > _dgemm_nt_sum0_fast1_bgemm0_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:13 --n=77 > _dgemm_tn_sum0_fast1_bgemm0_n77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:13 --n=77 > _dgemm_tt_sum0_fast1_bgemm0_n77.dat 

../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:13 --k=77 > _dgemm_nn_sum0_fast1_bgemm0_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=n --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:13 --k=77 > _dgemm_nt_sum0_fast1_bgemm0_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=n --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:13 --k=77 > _dgemm_tn_sum0_fast1_bgemm0_k77.dat 
../../testing/testing_ozblas_dgemm --nodisp=1 --transa=t --transb=t --summode=0 --fastmode=1 --usebatchedgemm=1 --range=1:256:13 --k=77 > _dgemm_tt_sum0_fast1_bgemm0_k77.dat 


