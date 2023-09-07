#!/bin/bash
#PBS -l nodes=1:rascal04:ppn=4
cd $PBS_O_WORKDIR

GPU=titanv
CPU=w2123

#ARRAY=( tmt_sym gridgena cfd1 cbuckle )
ARRAY=( BenElechi1 gyro_k pdb1HYS nd24k )
#ARRAY=( tmt_sym gridgena cfd1 cbuckle BenElechi1 gyro_k pdb1HYS nd24k )

for((i = 0; i<${#ARRAY[*]};i++))
{
	matrix=${ARRAY[i]}
	matsrc=/home/mukunoki/matrix/${matrix}.mtx

	./testing/testing_ozblas_dcg --tol=1e-16 --maxiter=80000 --verbose=80000 --repromode=0 --degree=1 --mtx=${matsrc} > _testing_ozblas_dcg_${matrix}_${CPU}.dat 
	./testing/testing_ozblas_dcg --tol=1e-16 --maxiter=80000 --verbose=80000 --usebatchedgemm=0 --summode=1 --splitepsmode=1 --mtx=${matsrc} > _testing_ozblas_dcg-cr_${matrix}_${CPU}_eps1.dat 
	./testing/testing_ozblas_dcg --tol=1e-16 --maxiter=80000 --verbose=80000 --usebatchedgemm=0 --summode=1 --splitepsmode=2 --mtx=${matsrc} > _testing_ozblas_dcg-cr_${matrix}_${CPU}_eps2.dat 
}
	

