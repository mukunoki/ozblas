#include "ozblas_common.h"

template <typename TYPE1, typename TYPE2, typename TYPE3>
TYPE1 ozblasRdot (
	ozblasHandle_t *oh,
	const int32_t n,
	const TYPE1 *devA, const int32_t incx,
	const TYPE1 *devB, const int32_t incy
) {
	if (oh->reproMode == 0 && oh->nSplitMax == 1) {
		double t0 = timer();
		TYPE1 r;
		if (oh->precxFlag == 1) 
			r = blasRdotX (n, (TYPE1*)devA, incx, (TYPE1*)devB, incy);
		else
			r = blasRdot (n, (TYPE1*)devA, incx, (TYPE1*)devB, incy);
		oh->t_DOT_NRM2_total += timer() - t0;
		return r;
	}
	if (incx != 1 || incy != 1 ) {
		fprintf (OUTPUT, "OzBLAS error: incx and incy are not supported.\n");
		exit (1);
	}

	TYPE1 fone = 1., fzero = 0., ret;
	ozblasRgemm <TYPE1, TYPE2, TYPE3> (oh, 't', 'n', 1, 1, n, fone, devA, n, devB, n, fzero, &ret, 1);

	// for CG, time
	// =================================
	oh->t_SplitMat_total += 0.;
	oh->t_SplitVec_total += oh->t_SplitA + oh->t_SplitB;
	oh->t_Sum_total += oh->t_sum;
	oh->t_AXPY_SCAL_total += 0.;
	oh->t_DOT_NRM2_total += oh->t_comp;
	oh->t_SpMV_SpMM_total += 0.;
	// =================================

	return ret;
}
template __float128 ozblasRdot <__float128, double, double> (ozblasHandle_t *oh, const int32_t n, const __float128 *devA, const int32_t incx, const __float128 *devB, const int32_t incy);
template __float128 ozblasRdot <__float128, float, float> (ozblasHandle_t *oh, const int32_t n, const __float128 *devA, const int32_t incx, const __float128 *devB, const int32_t incy);
template double ozblasRdot <double, double, double> (ozblasHandle_t *oh, const int32_t n, const double *devA, const int32_t incx, const double *devB, const int32_t incy);
template double ozblasRdot <double, float, float> (ozblasHandle_t *oh, const int32_t n, const double *devA, const int32_t incx, const double *devB, const int32_t incy);
template float ozblasRdot <float, float, float> (ozblasHandle_t *oh, const int32_t n, const float *devA, const int32_t incx, const float *devB, const int32_t incy);
template float ozblasRdot <float, double, double> (ozblasHandle_t *oh, const int32_t n, const float *devA, const int32_t incx, const float *devB, const int32_t incy);

