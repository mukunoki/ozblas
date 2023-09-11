#!/bin/bash
#SBATCH --job-name=jobname
#SBATCH --partition=a100
#SBATCH -t 02:00:00

module load system/x86_64
export OMP_NUM_THREADS=1


ARRAY=( bundle1 qa8fm gridgena cfd1 )

for((i = 0; i<${#ARRAY[*]};i++))
{
	matrix=${ARRAY[i]}
	matsrc=/home/users/daichi.mukunoki/matrix/${matrix}.mtx

	../../testing/testing_ozblas_dcg --nodisp=1 --tol=1e-12 --maxiter=1000 --verbose=0 --summode=1 --mtx=${matsrc} > _dcg_sum1_${matrix}.dat 
	../../testing/testing_ozblas_dcg --nodisp=1 --tol=1e-12 --maxiter=1000 --verbose=0 --summode=0 --mtx=${matsrc} > _dcg_sum0_${matrix}.dat 
	../../testing/testing_ozblas_dcg --nodisp=1 --tol=1e-12 --maxiter=1000 --verbose=0 --summode=1 --splitepsmode=2 --mtx=${matsrc} > _dcg_sum1_eps2_${matrix}.dat 

	../../testing/testing_ozblas_scg --nodisp=1 --tol=1e-5 --maxiter=1000 --verbose=0 --summode=1 --mtx=${matsrc} > _scg_sum1_${matrix}.dat 
	../../testing/testing_ozblas_scg --nodisp=1 --tol=1e-5 --maxiter=1000 --verbose=0 --summode=0 --mtx=${matsrc} > _scg_sum0_${matrix}.dat 

	../../testing/testing_ozblas_dscg --nodisp=1 --tol=1e-12 --maxiter=1000 --verbose=0 --summode=0 --mtx=${matsrc} > _dscg_sum0_${matrix}.dat 

	../../testing/testing_ozblas_qscg --nodisp=1 --tol=1e-25 --maxiter=1000 --verbose=0 --summode=0 --mtx=${matsrc} > _qscg_sum0_${matrix}.dat 

	../../testing/testing_ozblas_qdcg --nodisp=1 --tol=1e-25 --maxiter=1000 --verbose=0 --summode=0 --mtx=${matsrc} > _qdcg_sum0_${matrix}.dat 
}

