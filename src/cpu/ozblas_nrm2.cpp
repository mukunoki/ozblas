#include "ozblas_common.h"

template <typename TYPE1, typename TYPE2>
TYPE1 ozblasRnrm2 (
	ozblasHandle_t *oh,
	const int n,
	const TYPE1* devX,
	const int incx
) {
	if (oh->reproModeFlag == 0 && oh->nSplitMax == 1) {
		if (oh->precxFlag == 1) 
			return blasRnrm2X (n, (TYPE1*)devX, incx);
		else
			return blasRnrm2 (n, (TYPE1*)devX, incx);
	}
	if (incx != 1) {
		fprintf (OUTPUT, "OzBLAS error: incx and incy are not supported.\n");
		exit (1);
	}
	counterInit (oh);
	
	TYPE1 ret = ozblasRdot <TYPE1, TYPE2> (oh, n, devX, incx, devX, incx);

	// ------------------------------
	// computation of SQRT (ret) on host
	// Not accurate but reproducible
	ret = sqrt (ret);
	// ------------------------------

	return ret;
}
#if defined (FLOAT128)
template __float128 ozblasRnrm2 <__float128, double> (ozblasHandle_t *oh, const int n, const __float128* devX, const int incx);
template __float128 ozblasRnrm2 <__float128, float> (ozblasHandle_t *oh, const int n, const __float128* devX, const int incx);
#endif
template double ozblasRnrm2 <double, double> (ozblasHandle_t *oh, const int n, const double* devX, const int incx);
template double ozblasRnrm2 <double, float> (ozblasHandle_t *oh, const int n, const double* devX, const int incx);
template float ozblasRnrm2 <float, float> (ozblasHandle_t *oh, const int n, const float* devX, const int incx);
template float ozblasRnrm2 <float, double> (ozblasHandle_t *oh, const int n, const float* devX, const int incx);

