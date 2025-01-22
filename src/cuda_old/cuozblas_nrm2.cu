#include "cuozblas_common.h"

template <typename TYPE1, typename TYPE2>
int32_t cuozblasRnrm2 (
	cuozblasHandle_t *oh,
	const int32_t n,
	const TYPE1* devX,
	const int32_t incx,
	TYPE1* ret
) {
	if (oh->reproModeFlag == 0 && oh->nSplitMax == 1) {
		if (oh->precxFlag == 1) 
			blasRnrm2X (oh->ch, n, (TYPE1*)devX, (TYPE1*)oh->devWorkCommon, ret);
		else
			blasRnrm2 (oh->ch, n, (TYPE1*)devX, incx, ret);
		return 0;
	}
	if (incx != 1) {
		fprintf (OUTPUT, "OzBLAS error: incx is not supported.\n");
		exit (1);
	}
	cucounterInit (oh);
	
	cuozblasRdot <TYPE1, TYPE2> (oh, n, devX, incx, devX, incx, ret);

	// ------------------------------
	// computation of SQRT (ret) on host
	// Not accurate but reproducible
	ret[0] = sqrt (ret[0]);
	// ------------------------------

	return 0;
}
template int32_t cuozblasRnrm2 <double, double> (cuozblasHandle_t *oh, const int32_t n, const double* devX, const int32_t incx, double *ret);
template int32_t cuozblasRnrm2 <double, float> (cuozblasHandle_t *oh, const int32_t n, const double* devX, const int32_t incx, double *ret);
template int32_t cuozblasRnrm2 <float, float> (cuozblasHandle_t *oh, const int32_t n, const float* devX, const int32_t incx, float *ret);
template int32_t cuozblasRnrm2 <float, double> (cuozblasHandle_t *oh, const int32_t n, const float* devX, const int32_t incx, float *ret);

