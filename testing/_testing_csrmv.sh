#!/bin/sh
## #PBS -l nodes=1:rascal06:ppn=4
## cd $PBS_O_WORKDIR

ARRAY=( tmt_sym gridgena cfd1 cbuckle BenElechi1 gyro_k pdb1HYS nd24k )
GPU=titanv
CPU=w2123

echo "" > _testing.dat 
for((i = 0; i<${#ARRAY[*]};i++))
{
	matrix=${ARRAY[i]}
	matsrc=/home/mukunoki/matrix/${matrix}.mtx

	./testing_cuozblas_dcsrmv --mode=p --nodisp=1 --mtx=${matsrc} >> _testing.dat 
#	./testing_cuozblas_dcsrmv --mode=p --nodisp=1 --repromode=0 --degree=1 --mtx=${matsrc} >> _testing.dat 

}
