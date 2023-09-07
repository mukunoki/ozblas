#!/bin/bash
#SBATCH --job-name=jobname
#SBATCH --partition=mi100
#SBATCH -t 02:00:00

module load system/x86_64
export OMP_NUM_THREADS=256

# ddot
# sum1
../testing/testing_ozblas_ddot --nodisp=1 --summode=1 --range=1:256:13 > _ddot_sum1_s.dat 
../testing/testing_ozblas_ddot --nodisp=1 --summode=1 --range=1:256:13 > _ddot_sum1_s.dat 
../testing/testing_ozblas_ddot --nodisp=1 --summode=1 --range=1:2560000:pow > _ddot_sum1_r.dat 
../testing/testing_ozblas_ddot --nodisp=1 --summode=1 --range=1:2560000:pow > _ddot_sum1_r.dat 

# sum0
../testing/testing_ozblas_ddot --nodisp=1 --summode=0 --range=1:256:13 > _ddot_sum0_s.dat 
../testing/testing_ozblas_ddot --nodisp=1 --summode=0 --range=1:256:13 > _ddot_sum0_s.dat 
../testing/testing_ozblas_ddot --nodisp=1 --summode=0 --range=1:2560000:pow > _ddot_sum0_r.dat 
../testing/testing_ozblas_ddot --nodisp=1 --summode=0 --range=1:2560000:pow > _ddot_sum0_r.dat 



# dsdot
# sum1
../testing/testing_ozblas_dsdot --nodisp=1 --summode=1 --range=1:256:13 > _dsdot_sum1_s.dat 
../testing/testing_ozblas_dsdot --nodisp=1 --summode=1 --range=1:256:13 > _dsdot_sum1_s.dat 
../testing/testing_ozblas_dsdot --nodisp=1 --summode=1 --range=1:2560000:pow > _dsdot_sum1_r.dat 
../testing/testing_ozblas_dsdot --nodisp=1 --summode=1 --range=1:2560000:pow > _dsdot_sum1_r.dat 

# sum0
../testing/testing_ozblas_dsdot --nodisp=1 --summode=0 --range=1:256:13 > _dsdot_sum0_s.dat 
../testing/testing_ozblas_dsdot --nodisp=1 --summode=0 --range=1:256:13 > _dsdot_sum0_s.dat 
../testing/testing_ozblas_dsdot --nodisp=1 --summode=0 --range=1:2560000:pow > _dsdot_sum0_r.dat 
../testing/testing_ozblas_dsdot --nodisp=1 --summode=0 --range=1:2560000:pow > _dsdot_sum0_r.dat 



# qddot
# sum1
../testing/testing_ozblas_qddot --nodisp=1 --summode=1 --range=1:256:13 > _qddot_sum1_s.dat 
../testing/testing_ozblas_qddot --nodisp=1 --summode=1 --range=1:256:13 > _qddot_sum1_s.dat 
../testing/testing_ozblas_qddot --nodisp=1 --summode=1 --range=1:2560000:pow > _qddot_sum1_r.dat 
../testing/testing_ozblas_qddot --nodisp=1 --summode=1 --range=1:2560000:pow > _qddot_sum1_r.dat 

# sum0
../testing/testing_ozblas_qddot --nodisp=1 --summode=0 --range=1:256:13 > _qddot_sum0_s.dat 
../testing/testing_ozblas_qddot --nodisp=1 --summode=0 --range=1:256:13 > _qddot_sum0_s.dat 
../testing/testing_ozblas_qddot --nodisp=1 --summode=0 --range=1:2560000:pow > _qddot_sum0_r.dat 
../testing/testing_ozblas_qddot --nodisp=1 --summode=0 --range=1:2560000:pow > _qddot_sum0_r.dat 



# qsdot
# sum1
../testing/testing_ozblas_qsdot --nodisp=1 --summode=1 --range=1:256:13 > _qsdot_sum1_s.dat 
../testing/testing_ozblas_qsdot --nodisp=1 --summode=1 --range=1:256:13 > _qsdot_sum1_s.dat 
../testing/testing_ozblas_qsdot --nodisp=1 --summode=1 --range=1:2560000:pow > _qsdot_sum1_r.dat 
../testing/testing_ozblas_qsdot --nodisp=1 --summode=1 --range=1:2560000:pow > _qsdot_sum1_r.dat 

# sum0
../testing/testing_ozblas_qsdot --nodisp=1 --summode=0 --range=1:256:13 > _qsdot_sum0_s.dat 
../testing/testing_ozblas_qsdot --nodisp=1 --summode=0 --range=1:256:13 > _qsdot_sum0_s.dat 
../testing/testing_ozblas_qsdot --nodisp=1 --summode=0 --range=1:2560000:pow > _qsdot_sum0_r.dat 
../testing/testing_ozblas_qsdot --nodisp=1 --summode=0 --range=1:2560000:pow > _qsdot_sum0_r.dat 

