#include "ozblas_common.h"

template <typename TYPE1, typename TYPE2>
int32_t ozblasRgemv (
	ozblasHandle_t *oh,
	const char transA, 
	const int32_t m, const int32_t n,
	const TYPE1 alpha,
	const TYPE1 *devA, const int32_t lda,
	const TYPE1 *devB, const int32_t incx,
	const TYPE1 beta,
	TYPE1 *devC, const int32_t incy
) {
	if (oh->reproModeFlag == 0 && oh->nSplitMax == 1) {
		blasRgemv (transA, m, n, alpha, (TYPE1*)devA, lda, (TYPE1*)devB, incx, beta, devC, incy);
		return 0;
	}
	if (incx != 1 || incy != 1 ) {
		fprintf (OUTPUT, "OzBLAS error: incx and incy are not supported.\n");
		exit (1);
	}

	if (checkTrans (transA) == 0) 
		ozblasRgemm <TYPE1, TYPE2> (oh, 'n', 'n', m, 1, n, alpha, devA, lda, devB, n, beta, devC, m);
	else
		ozblasRgemm <TYPE1, TYPE2> (oh, 't', 'n', n, 1, m, alpha, devA, lda, devB, m, beta, devC, n);

	return 0;
}
template int32_t ozblasRgemv <__float128, double> (ozblasHandle_t *oh, const char transA, const int32_t m, const int32_t n, const __float128 alpha, const __float128 *devA, const int32_t lda, const __float128 *devB, const int32_t incx, const __float128 beta, __float128 *devC, const int32_t incy);
template int32_t ozblasRgemv <__float128, float> (ozblasHandle_t *oh, const char transA, const int32_t m, const int32_t n, const __float128 alpha, const __float128 *devA, const int32_t lda, const __float128 *devB, const int32_t incx, const __float128 beta, __float128 *devC, const int32_t incy);
template int32_t ozblasRgemv <double, double> (ozblasHandle_t *oh, const char transA, const int32_t m, const int32_t n, const double alpha, const double *devA, const int32_t lda, const double *devB, const int32_t incx, const double beta, double *devC, const int32_t incy);
template int32_t ozblasRgemv <double, float> (ozblasHandle_t *oh, const char transA, const int32_t m, const int32_t n, const double alpha, const double *devA, const int32_t lda, const double *devB, const int32_t incx, const double beta, double *devC, const int32_t incy);
template int32_t ozblasRgemv <float, float> (ozblasHandle_t *oh, const char transA, const int32_t m, const int32_t n, const float alpha, const float *devA, const int32_t lda, const float *devB, const int32_t incx, const float beta, float *devC, const int32_t incy);
template int32_t ozblasRgemv <float, double> (ozblasHandle_t *oh, const char transA, const int32_t m, const int32_t n, const float alpha, const float *devA, const int32_t lda, const float *devB, const int32_t incx, const float beta, float *devC, const int32_t incy);

