#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define DefaultWorkSize 5e9 // 5GB
#define NumSplitDefaultMax 30

#define OUTPUT stdout // stderr

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <typeinfo>

#include <inttypes.h>
#include <sys/time.h>
#include <float.h>

#if defined (MKL)
#include <mkl_cblas.h>
#include <mkl_trans.h>
#include <mkl_spblas.h>
#else
#include <cblas.h>
#endif

#include <omp.h>

#if defined (__INTEL_LLVM_COMPILER)
Intel LLVM Compiler is not supported...
#endif

#if defined (ARM)
#define __float128 long double
#define FLT128_MAX LDBL_MAX
#define FLT128_MIN LDBL_MIN
#else
#include <quadmath.h>
#endif

#if defined (MPLAPACK)
#include <mplapack/mpblas__Float128.h>
#endif

#if defined (CUBLAS)
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#endif

#include "../../include/ozblas.h"
#include "ozblas_internal.h"

// -------------------------------------
// constexpr functions
// -------------------------------------

// getEmin
template <typename TYPE>
constexpr int32_t getEmin () {
	fprintf (OUTPUT, "OzBLAS error: TYPE is not specified in getEmin.\n");
	exit (1);
	return 0;
}
template <>
constexpr int32_t getEmin <float> () {
	return -126;
}
template <>
constexpr int32_t getEmin <double> () {
	return -1022;
}
template <>
constexpr int32_t getEmin <__float128> () {
	return -16382;
}

// getEpse
template <typename TYPE>
constexpr int32_t getEpse () {
	fprintf (OUTPUT, "OzBLAS error: TYPE is not specified in getEpse.\n");
	exit (1);
	return 0;
}
template <>
constexpr int32_t getEpse <float> () {
	return 24;
}
template <>
constexpr int32_t getEpse <double> () {
	return 53;
}
template <>
constexpr int32_t getEpse <__float128> () {
	return 113;
}
template <typename TYPE>
constexpr int32_t getEpse2 () {
	fprintf (OUTPUT, "OzBLAS error: TYPE is not specified in getEpse2.\n");
	exit (1);
	return 0;
}
template <>
constexpr int32_t getEpse2 <float> () {
	return 53;
}
template <>
constexpr int32_t getEpse2 <double> () {
	return 106;
}

// getTypeMax
template <typename TYPE>
constexpr TYPE getTypeMax () {
	fprintf (OUTPUT, "OzBLAS error: TYPE is not specified in getTypeMax.\n");
	exit (1);
	return 0;
}
template <>
constexpr float getTypeMax <float> () {
	return FLT_MAX;
}
template <>
constexpr double getTypeMax <double> () {
	return DBL_MAX;
}
template <>
constexpr __float128 getTypeMax <__float128> () {
	return FLT128_MAX;
}

// getTypeMin
template <typename TYPE>
constexpr TYPE getTypeMin () {
	fprintf (OUTPUT, "OzBLAS error: TYPE is not specified in getTypeMin.\n");
	exit (1);
	return 0;
}
template <>
constexpr float getTypeMin <float> () {
	return FLT_MIN;
}
template <>
constexpr double getTypeMin <double> () {
	return DBL_MIN;
}
template <>
constexpr __float128 getTypeMin <__float128> () {
	return FLT128_MIN;
}

