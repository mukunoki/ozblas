#include "cuozblas_common.h"

int32_t cucheckTrans (const char tran) {
	if (tran == 'N' || tran == 'n') 
		return 0;
	else
		return 1;
}

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
#define ALIGN 128
int32_t cugetPitchSize (int32_t n) {
	return ceil((float)n / ALIGN) * ALIGN;
}

