#!/bin/bash

# examples

echo "# accurate DGEMM (Ozaki scheme + pure floating-point sum)"
echo "./testing_ozblas_dgemm --range=1:1024:pow"
./testing_ozblas_dgemm --range=1:1024:pow 

echo "# accurate DGEMM (Ozaki scheme + pure floating-point sum + fastmode (less DGEMMs are used))"
echo "./testing_ozblas_dgemm --range=1:1024:pow --fastmode=1"
./testing_ozblas_dgemm --range=1:1024:pow --fastmode=1

echo "# infinite-precision correctly-rounded DGEMM (Ozaki scheme + NearSum)"
echo "./testing_ozblas_dgemm --range=1:1024:pow --summode=1"
./testing_ozblas_dgemm --range=1:1024:pow --summode=1

echo "# accurate DGEMM using SGEMM (Ozaki scheme + pure floating-point sum)"
echo "./testing_ozblas_dsgemm --range=1:1024:pow"
./testing_ozblas_dsgemm --range=1:1024:pow 

echo "# accurate QGEMM using DGEMM (Ozaki scheme + pure floating-point sum in binary128 emulation)"
echo "./testing_ozblas_qdgemm --range=1:1024:pow"
./testing_ozblas_qdgemm --range=1:1024:pow 

echo "# accurate QGEMM using DGEMM (Ozaki scheme + sum3 + split3, fast but only for binary64 exponent range, note: batchedgemm cannot be used)"
echo "./testing_ozblas_qdgemm --range=1:1024:pow --usebatchedgemm=0 --summode=3 --splitmode=3"
./testing_ozblas_qdgemm --range=1:1024:pow --usebatchedgemm=0 --summode=3 --splitmode=3

echo "# accurate QGEMM using SGEMM (Ozaki scheme + pure floating-point sum in binary128 emulation)"
echo "./testing_ozblas_qsgemm --range=1:1024:pow"
./testing_ozblas_qsgemm --range=1:1024:pow 

echo "# transposed operations"
echo "./testing_ozblas_dgemm --range=1:1024:pow --transa=t --transb=t"
./testing_ozblas_dgemm --range=1:1024:pow --transa=t --transb=t 

echo "# matrix size example 1"
echo "./testing_ozblas_dgemm --range=0:1024:128 --m=77"
./testing_ozblas_dgemm --range=0:1024:128 --m=77

echo "# matrix size example 2"
echo "./testing_ozblas_dgemm --mnk=128 --k=440"
./testing_ozblas_dgemm --mnk=128 --k=440

echo "# input data range (adjustable with phi, e.g., phi=5 is [1e0,1e5))"
echo "./testing_ozblas_dgemm --mnk=128 --k=440 --phi=15"
./testing_ozblas_dgemm --mnk=128 --k=440 --phi=15

echo "# Ozaki scheme with a specific number of split matrices (result can be inaccurate)"
echo "./testing_ozblas_dgemm --mnk=128 --k=440 --degree=2 --splitmode=0 --phi=15"
./testing_ozblas_dgemm --mnk=128 --k=440 --degree=2 --splitmode=0 --phi=15

echo "# performance test (averaged runtime of 10 executions)"
echo "./testing_ozblas_dgemm --range=1:1024:pow --mode=p"
./testing_ozblas_dgemm --range=1:1024:pow --mode=p

