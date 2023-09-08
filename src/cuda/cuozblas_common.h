#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define DefaultWorkSize 1e9 // 1GB
#define NumSplitDefaultMax 20

#define OUTPUT stdout // stderr
/*
#define ADD(x,y) __dadd_rn(x,y)
#define SUB(x,y) __dsub_rn(x,y)
#define MUL(x,y) __dmul_rn(x,y)
#define FMA(x,y,z) __fma_rn(x,y,z)
*/

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cmath> 
#include <iostream>
#include <typeinfo>

#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <sys/time.h>
#include <float.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>

#include "../../include/cuozblas.h"
#include "cuozblas_internal.h"

// -------------------------------------
// constexpr functions
// -------------------------------------

// -------------------------------------
// getEmin
// -------------------------------------
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

// -------------------------------------
// getEpse
// -------------------------------------
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


// -------------------------------------
// getTypeMax
// -------------------------------------
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

// -------------------------------------
// getTypeMin
// -------------------------------------
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
