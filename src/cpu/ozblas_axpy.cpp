#include "ozblas_common.h"

template <typename TYPE>
int32_t ozblasRaxpy (
	ozblasHandle_t *oh,
	const int32_t n,
	const TYPE alpha,
	const TYPE *devX,
	const int32_t incx,
	TYPE *devY,
	const int32_t incy
) {
	if (oh->reproModeFlag == 0) {
		blasRaxpy (n, alpha, (TYPE*)devX, incx, (TYPE*)devY, incy);
		return 0;
	}
	if (incx != 1 || incy != 1) {
		fprintf (OUTPUT, "OzBLAS error: incx and incy are not supported.\n");
		exit (1);
	}

	#pragma omp parallel for 
	for (int i = 0; i < n; i++) 
		devY[i] = fma1 (alpha, devX[i], devY[i]);

	return 0;
}
template int32_t ozblasRaxpy (ozblasHandle_t *oh, const int32_t n, const __float128 alpha, const __float128 *devX, const int32_t incx, __float128 *devY, const int32_t incy);
template int32_t ozblasRaxpy (ozblasHandle_t *oh, const int32_t n, const double alpha, const double *devX, const int32_t incx, double *devY, const int32_t incy);
template int32_t ozblasRaxpy (ozblasHandle_t *oh, const int32_t n, const float alpha, const float *devX, const int32_t incx, float *devY, const int32_t incy);

