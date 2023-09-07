#include "cuozblas_common.h"

int32_t cucheckTrans (const char tran) {
	if (tran == 'N' || tran == 'n') 
		return 0;
	else
		return 1;
}

// check if matrix elements (on TYPE1) fit in the range of TYPE2
/*
template <typename TYPE1, typename TYPE2>
int32_t curangeCheck (
	const int32_t m,
	const int32_t n,
	const TYPE1 *mat,
	const int32_t ld
) {
	int32_t checkGlobal = 1; // OK
	constexpr TYPE2 type2Max = getTypeMax <TYPE2> ();
	constexpr TYPE2 type2Min = getTypeMin <TYPE2> ();
	TYPE1 type1MaxAbs = (TYPE1)fabs(type2Max);
	TYPE1 type1MinAbs = (TYPE1)fabs(type2Min);
	#pragma omp parallel for
	for (int32_t addry = 0; addry < n; addry++) {
		int32_t checkLocal = 1;
		for (int32_t addrx = 0; addrx < m; addrx++) {
			TYPE1 valAbs = fabs(mat[addry * ld + addrx]);
			if (valAbs > type1MaxAbs || valAbs < type1MinAbs) checkLocal = 0; // NG
		}
		if (checkLocal == 0) { // If NG
			#pragma omp atomic write
			checkGlobal = 0; // NG
		}
	}
	return checkGlobal; // 0:NG (out of range), 1:OK (within range)
}
template int32_t curangeCheck <double, float> (const int32_t m, const int32_t n, const double *mat, const int32_t ld);
template int32_t curangeCheck <double, double> (const int32_t m, const int32_t n, const double *mat, const int32_t ld);
template int32_t curangeCheck <float, double> (const int32_t m, const int32_t n, const float *mat, const int32_t ld);
template int32_t curangeCheck <float, float> (const int32_t m, const int32_t n, const float *mat, const int32_t ld);
*/

// =========================================
// Print floating-point value with bit representation
// =========================================
/*
typedef union{
	FP_TYPE3 d;
	int64_t i;
} d_and_i;

void printBits (FP_TYPE3 val) {
	d_and_i di;
	di.d = val;
	// sign
	printf ("%zu", (di.i >> 63) & 1);
	printf ("|");
	// exponent
	for (int i = 62; i >= 62-10; i--) 
		printf ("%zu", (di.i >> i) & 1);
	printf ("|");
	// fraction
	for (int i = 62-11; i >= 0; i--) 
		printf ("%zu", (di.i >> i) & 1);
	printf (" : ");
	printf ("%+1.18e", val);
}

// =========================================
// PrintMat (for debugging)
// =========================================
template <typename TYPE>
void cuozblasPrintMat (
	const int32_t m,
	const int32_t n,
	const TYPE *devC,
	const int32_t ldd
) {
	TYPE tmp;
	int32_t i, j, ldh;
	ldh = m;
	printf ("\n");
	for (j = 0; j < n; j++) {
		for (i = 0; i < m; i++) {
			tmp = devC[j*ldh+i];
			if (tmp != 0) {
				printf ("[%2d,%2d] ", j, i);
				printBits (tmp);
				printf ("\n");
			}
		}
	}
	printf ("\n");
}

template <typename TYPE>
void cuozblasPrintVec (
	const int32_t n,
	const TYPE *devC
) {
	for (int32_t i = 0; i < n; i++) {
		#if defined (PREC_Q_D) || defined (PREC_Q_S)
		char buf[128];
		quadmath_snprintf(buf, sizeof(buf), "%1.18Qe", devC[i]);
		puts(buf);
		printf ("\n");
		#else
		printf ("%1.18e\n", devC[i]);
		#endif
	}
}
*/

cublasOperation_t ToCublasOp (const char tran) {
	if (tran == 'N' || tran == 'n') return CUBLAS_OP_N;
	if (tran == 'T' || tran == 't') return CUBLAS_OP_T;
	if (tran == 'C' || tran == 'c') return CUBLAS_OP_C;
	return CUBLAS_OP_N; //default
}
cusparseOperation_t ToCusparseOp (const char tran) {
	if (tran == 'N' || tran == 'n') return CUSPARSE_OPERATION_NON_TRANSPOSE;
	if (tran == 'T' || tran == 't') return CUSPARSE_OPERATION_TRANSPOSE;
	return CUSPARSE_OPERATION_NON_TRANSPOSE; //default
}

void cuPrintMatInt (
	const int32_t m,
	const int32_t n,
	const int32_t *devC,
	const int32_t ldd
) {
	int32_t tmp;
	int32_t i, j, ldh;
	ldh = m;
	printf ("\n");
	for (j = 0; j < n; j++) {
		for (i = 0; i < m; i++) {
			tmp = devC[j*ldh+i];
		//	if (tmp != 0) {
				printf ("[%2d,%2d] %d", j, i, tmp);
				printf ("\n");
		//	}
		}
	}
	printf ("\n");
}

// for CG
template <typename TYPE>
void cuozblasCopyVec (
	const int32_t n,
	const TYPE *devIn,
	TYPE *devOut
) {
	cudaMemcpy (devOut, devIn, sizeof(TYPE) * n, cudaMemcpyDeviceToDevice); 
}
template void cuozblasCopyVec <double> (const int32_t, const double*, double*);
template void cuozblasCopyVec <float> (const int32_t, const float*, float*);

// =========================================
// Matrix Allocation
// =========================================

int32_t cumemCheck (cuozblasHandle_t *oh) {
	if (oh->memAddr > oh->workSizeBytes) return 1;
	return 0;
}

void cuozblasMatAddrAlloc (
	cuozblasHandle_t *oh,
	const int32_t m,
	const int32_t n,
	const int32_t size,
	void **dev,
	int32_t &ld
) {
	ld = cugetPitchSize (m);
	dev[0] = oh->devWork + oh->memAddr;
	oh->memAddr += (uint64_t)size * ld * n;
}

void cuozblasVecAddrAlloc (
	cuozblasHandle_t *oh,
	const int32_t n,
	const int32_t size,
	void **dev
) {
	int32_t ld = cugetPitchSize (n);
	dev[0] = oh->devWork + oh->memAddr;
	oh->memAddr += (uint64_t)size * ld;
}

double cutimer () {
	struct timeval tv;
	cudaDeviceSynchronize ();
	gettimeofday (&tv, NULL);
	return tv.tv_sec + (double) tv.tv_usec * 1.0e-6;
}

// note: this is temporal...
int32_t cugetPitchSize (int32_t n) {
	return ceil((float)n / 128) * 128;
}

