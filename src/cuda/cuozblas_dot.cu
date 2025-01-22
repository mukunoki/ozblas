#include "cuozblas_common.h"

template <typename TYPE1, typename TYPE2, typename TYPE3>
int32_t cuozblasRdot (
	cuozblasHandle_t *oh,
	const int32_t n,
	const TYPE1 *devA, const int32_t incx,
	const TYPE1 *devB, const int32_t incy,
	TYPE1 *ret
) {
	if (oh->reproModeFlag == 0 && oh->nSplitMax == 1) {
		double t0 = cutimer();
		if (oh->precxFlag == 1) 
			blasRdotX (oh->ch, n, devA, devB, (TYPE1*)oh->devWorkCommon, ret);
		else
			blasRdot (oh->ch, n, devA, incx, devB, incy, ret);
		oh->t_DOT_NRM2_total += cutimer() - t0;
		return 0;
	}
	if (incx != 1 || incy != 1 ) {
		fprintf (OUTPUT, "OzBLAS error: incx and incy are not supported.\n");
		exit (1);
	}

	TYPE1 fone = 1., fzero = 0.;
	cuozblasRgemm <TYPE1, TYPE2, TYPE3> (oh, 't', 'n', 1, 1, n, fone, devA, n, devB, n, fzero, (TYPE1*)oh->devWorkCommon, 1);
	cudaMemcpy (ret, oh->devWorkCommon, sizeof(TYPE1), cudaMemcpyDeviceToHost);

	// for CG, time
	// =================================
	oh->t_SplitMat_total += 0.;
	oh->t_SplitVec_total += oh->t_SplitA + oh->t_SplitB;
	oh->t_Sum_total += oh->t_sum;
	oh->t_AXPY_SCAL_total += 0.;
	oh->t_DOT_NRM2_total += oh->t_comp;
	oh->t_SpMV_SpMM_total += 0.;
	// =================================

	return 0;
}
template int32_t cuozblasRdot <double, double, double> (cuozblasHandle_t *oh, const int32_t n, const double *devA, const int32_t incx, const double *devB, const int32_t incy, double *ret);
template int32_t cuozblasRdot <double, float, float> (cuozblasHandle_t *oh, const int32_t n, const double *devA, const int32_t incx, const double *devB, const int32_t incy, double *ret);
template int32_t cuozblasRdot <double, half, float> (cuozblasHandle_t *oh, const int32_t n, const double *devA, const int32_t incx, const double *devB, const int32_t incy, double *ret);
template int32_t cuozblasRdot <float, float, float> (cuozblasHandle_t *oh, const int32_t n, const float *devA, const int32_t incx, const float *devB, const int32_t incy, float *ret);
template int32_t cuozblasRdot <float, half, float> (cuozblasHandle_t *oh, const int32_t n, const float *devA, const int32_t incx, const float *devB, const int32_t incy, float *ret);
template int32_t cuozblasRdot <float, double, double> (cuozblasHandle_t *oh, const int32_t n, const float *devA, const int32_t incx, const float *devB, const int32_t incy, float *ret);

