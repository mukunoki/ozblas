#!/bin/bash
#SBATCH --job-name=jobname
#SBATCH --partition=a100
#SBATCH -t 02:00:00

module load system/x86_64
export OMP_NUM_THREADS=256

# ddot
# sum1
../../testing/testing_cuozblas_ddot --nodisp=1 --summode=1 --range=1:256:13 > _ddot_sum1_s_cuda.dat 
../../testing/testing_cuozblas_ddot --nodisp=1 --summode=1 --range=1:256:13 > _ddot_sum1_s_cuda.dat 
../../testing/testing_cuozblas_ddot --nodisp=1 --summode=1 --range=1:2560000:pow > _ddot_sum1_r_cuda.dat 
../../testing/testing_cuozblas_ddot --nodisp=1 --summode=1 --range=1:2560000:pow > _ddot_sum1_r_cuda.dat 

# sum0
../../testing/testing_cuozblas_ddot --nodisp=1 --summode=0 --range=1:256:13 > _ddot_sum0_s_cuda.dat 
../../testing/testing_cuozblas_ddot --nodisp=1 --summode=0 --range=1:256:13 > _ddot_sum0_s_cuda.dat 
../../testing/testing_cuozblas_ddot --nodisp=1 --summode=0 --range=1:2560000:pow > _ddot_sum0_r_cuda.dat 
../../testing/testing_cuozblas_ddot --nodisp=1 --summode=0 --range=1:2560000:pow > _ddot_sum0_r_cuda.dat 



# sdot
# sum1
../../testing/testing_cuozblas_sdot --nodisp=1 --summode=1 --range=1:256:13 > _sdot_sum1_s_cuda.dat 
../../testing/testing_cuozblas_sdot --nodisp=1 --summode=1 --range=1:256:13 > _sdot_sum1_s_cuda.dat 
../../testing/testing_cuozblas_sdot --nodisp=1 --summode=1 --range=1:2560000:pow > _sdot_sum1_r_cuda.dat 
../../testing/testing_cuozblas_sdot --nodisp=1 --summode=1 --range=1:2560000:pow > _sdot_sum1_r_cuda.dat 

# sum0
../../testing/testing_cuozblas_sdot --nodisp=1 --summode=0 --range=1:256:13 > _sdot_sum0_s_cuda.dat 
../../testing/testing_cuozblas_sdot --nodisp=1 --summode=0 --range=1:256:13 > _sdot_sum0_s_cuda.dat 
../../testing/testing_cuozblas_sdot --nodisp=1 --summode=0 --range=1:2560000:pow > _sdot_sum0_r_cuda.dat 
../../testing/testing_cuozblas_sdot --nodisp=1 --summode=0 --range=1:2560000:pow > _sdot_sum0_r_cuda.dat 



# dsdot
# sum0
../../testing/testing_cuozblas_dsdot --nodisp=1 --summode=0 --range=1:256:13 > _dsdot_sum0_s_cuda.dat 
../../testing/testing_cuozblas_dsdot --nodisp=1 --summode=0 --range=1:256:13 > _dsdot_sum0_s_cuda.dat 
../../testing/testing_cuozblas_dsdot --nodisp=1 --summode=0 --range=1:2560000:pow > _dsdot_sum0_r_cuda.dat 
../../testing/testing_cuozblas_dsdot --nodisp=1 --summode=0 --range=1:2560000:pow > _dsdot_sum0_r_cuda.dat 



