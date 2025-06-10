#include "cuozblas_common.h"

#define NTX_AXPY 512

template <typename TYPE>
__global__ void
AxpyKernel (
	const int32_t n,
	const TYPE alpha,
	const TYPE* __restrict__ devX,
	TYPE * devY
) {
	const int32_t rowid = blockIdx.x * blockDim.x + threadIdx.x;
	if (rowid < n) 
		devY[rowid] = fma (alpha, devX[rowid], devY[rowid]);
}

template <typename TYPE>
int32_t cuozblasRaxpy (
	cuozblasHandle_t *oh,
	const int32_t n,
	const TYPE alpha,
	const TYPE *devX,
	const int32_t incx,
	TYPE *devY,
	const int32_t incy
) {
	if (oh->reproMode == 0) {
		blasRaxpy (oh->ch, n, alpha, (TYPE*)devX, incx, (TYPE*)devY, incy);
		return 0;
	}
	if (incx != 1 || incy != 1) {
		fprintf (OUTPUT, "OzBLAS error: incx and incy are not supported.\n");
		exit (1);
	}

	dim3 threads = dim3 (NTX_AXPY);
	dim3 grid = dim3 (ceil ((float)n/NTX_AXPY));
	AxpyKernel <<< grid, threads >>> (n, alpha, devX, devY);
	return 0;
}
template int32_t cuozblasRaxpy (cuozblasHandle_t *oh, const int32_t n, const double alpha, const double *devX, const int32_t incx, double *devY, const int32_t incy);
template int32_t cuozblasRaxpy (cuozblasHandle_t *oh, const int32_t n, const float alpha, const float *devX, const int32_t incx, float *devY, const int32_t incy);

