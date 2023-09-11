#!/bin/bash
#SBATCH --job-name=jobname
#SBATCH --partition=a100
#SBATCH -t 02:00:00

module load system/x86_64
export OMP_NUM_THREADS=256


ARRAY=( 1138_bus gridgena power9 pdb1HYS nd24k )

for((i = 0; i<${#ARRAY[*]};i++))
{
	matrix=${ARRAY[i]}
	matsrc=/home/users/daichi.mukunoki/matrix/${matrix}.mtx

	../../testing/testing_cuozblas_dcsrmv --nodisp=1 --summode=1 --mtx=${matsrc} > _dcsrmv_n_sum1_${matrix}_cuda.dat 
	../../testing/testing_cuozblas_dcsrmv --nodisp=1 --summode=0 --mtx=${matsrc} > _dcsrmv_n_sum0_${matrix}_cuda.dat 
	../../testing/testing_cuozblas_scsrmv --nodisp=1 --summode=1 --mtx=${matsrc} > _scsrmv_n_sum1_${matrix}_cuda.dat 
	../../testing/testing_cuozblas_scsrmv --nodisp=1 --summode=0 --mtx=${matsrc} > _scsrmv_n_sum0_${matrix}_cuda.dat 
	../../testing/testing_cuozblas_dscsrmv --nodisp=1 --summode=1 --mtx=${matsrc} > _dscsrmv_n_sum1_${matrix}_cuda.dat 
	../../testing/testing_cuozblas_dscsrmv --nodisp=1 --summode=0 --mtx=${matsrc} > _dscsrmv_n_sum0_${matrix}_cuda.dat 
}

