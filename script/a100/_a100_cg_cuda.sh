#!/bin/bash
#SBATCH --job-name=jobname
#SBATCH --partition=a100
#SBATCH -t 02:00:00

module load system/x86_64
export OMP_NUM_THREADS=256


ARRAY=( bundle1 qa8fm gridgena cfd1 )

for((i = 0; i<${#ARRAY[*]};i++))
{
	matrix=${ARRAY[i]}
	matsrc=/home/users/daichi.mukunoki/matrix/${matrix}.mtx

	../../testing/testing_cuozblas_dcg --nodisp=1 --tol=1e-16 --maxiter=10000 --verbose=0 --summode=1 --mtx=${matsrc} > _dcg_sum1_${matrix}_cuda.dat 
	../../testing/testing_cuozblas_dcg --nodisp=1 --tol=1e-16 --maxiter=10000 --verbose=0 --summode=0 --mtx=${matsrc} > _dcg_sum0_${matrix}_cuda.dat 
	../../testing/testing_cuozblas_dcg --nodisp=1 --tol=1e-16 --maxiter=10000 --verbose=0 --summode=1 --splitepsmode=2 --mtx=${matsrc} > _dcg_sum1_eps2_${matrix}_cuda.dat 
}

