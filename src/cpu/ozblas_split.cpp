#include "ozblas_common.h"

// =========================================
// Split
// Based on K.Ozaki et al., "Error-free transformations of matrix multiplication
// by using fast routines of matrix multiplication and its applications", 2012
// =========================================

#define CONST 0.75

template <typename TYPE>
TYPE NextPowTwo (const TYPE p) {
	constexpr int32_t epse_type = getEpse <TYPE> ();
	return scalbn1 (p, epse_type) - (TYPE)((scalbn1 (1., epse_type) - 1) * p);
}

template <typename TYPE1, typename TYPE2>
int32_t getRho (const int32_t dim, const int32_t splitEpsModeFlag) {
	constexpr int32_t epse_type1 = getEpse <TYPE1> ();
	constexpr int32_t epse_type2 = getEpse <TYPE2> ();
	constexpr int32_t epse_type22 = getEpse2 <TYPE2> ();
	switch (splitEpsModeFlag) {
		case 1: // 
			return ceil((epse_type1-(epse_type2-log2(1.*dim))/2)); // standard-ver
			break;
		case 2: // Dot2
			return ceil((epse_type1-(epse_type22-log2(1.*dim))/2)); // standard-ver with Dot2
			break;
		case 9: // overflow-mode
			return ceil((epse_type1-(epse_type2-log2(2.*sqrt(dim)))/2)); // overflow-ver
			break;
		default: // non-overflow-mode
			return ceil((epse_type1-(epse_type2-log2(1.*dim))/2)); // standard-ver
	}
}

// =========================================
// Error-free transformations
// =========================================
template <typename TYPE>
__inline__ void
QuickTwoSum (
	const TYPE a,
	const TYPE b,
	TYPE &s,
	TYPE &e
) {
	TYPE t;
	s = a + b;
	t = s - a;
	e = b - t;
}
// =========================================

template <typename TYPE>
void ozblasFindMaxRKernel (
	const int32_t m,
	const int32_t n,
	const TYPE *devInput,
	const int32_t ldi,
	TYPE *devMax
) {
	for (int32_t addrx = 0; addrx < m; addrx++) {
		TYPE max_ = 0.;
		int32_t addry;
		#pragma omp parallel for reduction(max:max_)
		for (addry = 0; addry < n; addry++) 
			max_ = MAX(max_, fabs1(devInput[addry * ldi + addrx]));
		devMax[addrx] = max_;
	}
}

template <typename TYPE>
void ozblasFindMaxCKernel (
	const int32_t m,
	const int32_t n,
	const TYPE *devInput,
	const int32_t ldi,
	TYPE *devMax
) {
	int32_t addry;
	#pragma omp parallel for
	for (addry = 0; addry < n; addry++) {
		TYPE max = 0.;
		for (int32_t addrx = 0; addrx < m; addrx++) 
			max = MAX(max, fabs1(devInput[addry * ldi + addrx]));
		devMax[addry] = max;
	} 
}

template <typename TYPE>
void ozblasFindMaxDevice (
	const char major,
	const int32_t m, 
	const int32_t n,
	const TYPE *devInput,
	const int32_t ldi, 
	TYPE *devMax
) {
	if (n == 1 && major == 'c') {
		const int32_t ptrMax = blasRiamax (m, devInput, 1);
		devMax[0] = fabs1(devInput[ptrMax]);
	} else {
		if (major == 'r') {
			ozblasFindMaxRKernel (m, n, devInput, ldi, devMax);
		} else {
			ozblasFindMaxCKernel (m, n, devInput, ldi, devMax);
		}
	}
}

template <typename TYPE>
void ozblasSplitVecKernel (
	const int32_t n,
	const int32_t rho,
	const TYPE *devInput,
	TYPE *devOutput,
	TYPE *devSplit,
	short *devSpExp,
	TYPE *devMax,
	const int32_t splitShift
) {
	const TYPE sigma = CONST * scalbn1 (1., rho) * NextPowTwo <TYPE> (devMax[0]) * splitShift;
	TYPE max_ = 0.;
	#pragma omp parallel for reduction(max:max_)
	for (int32_t i = 0; i < n; i++) {
		TYPE input = devInput[i];
		const TYPE split = (input + sigma) - sigma;
		input = input - split;
		devSplit[i] = split;
		devOutput[i] = input;
		max_ = MAX(max_, fabs1(input));
	}
	devMax[0] = max_;
}

template <typename TYPE1, typename TYPE2>
void ozblasSplitVecKernel (
	const int32_t n,
	const int32_t rho,
	const TYPE1 *devInput,
	TYPE1 *devOutput,
	TYPE2 *devSplit,
	short *devSpExp,
	TYPE1 *devMax,
	const int32_t splitShift
) {
	const short tau = devSpExp[0] = ceil(log2(devMax[0]));
	const TYPE1 sigma = CONST * scalbn1 (1., rho + tau) * splitShift;
	TYPE1 max_ = 0.;
	#pragma omp parallel for reduction(max:max_)
	for (int32_t i = 0; i < n; i++) {
		TYPE1 input = devInput[i];
		const TYPE1 split = (input + sigma) - sigma;
		input = input - split;
		devSplit[i] = scalbn1(split, -tau);
		devOutput[i] = input;
		max_ = MAX(max_, fabs1(input));
	}
	devMax[0] = max_;
}

// max for n
/*
template <typename TYPE1, typename TYPE2>
void ozblasSplitRKernel (
	const int32_t m,
	const int32_t n,
	const int32_t rho,
	const TYPE1 *devInput,
	const int32_t ldi,
	TYPE1 *devOutput,
	const int32_t ldo,
	TYPE2 *devSplit,
	const int32_t lds,
	short *devSpExp,
	TYPE1 *devMax,
	const int32_t splitShift
) {
	for (int32_t addrx = 0; addrx < m; addrx++) {
		const short tau = devSpExp[addrx] = ceil(log2(devMax[addrx]));
		const TYPE1 sigma = CONST * scalbn1 (1., rho + tau) * splitShift;
		int32_t addry = 0;
		TYPE1 max_ = 0.;
		#pragma omp parallel for reduction(max:max_)
		for (addry = 0; addry < n; addry++) {
			TYPE1 input = devInput[addry * ldi + addrx];
			const TYPE1 split = (input + sigma) - sigma;
			input = input - split;
			devSplit[addry * lds + addrx] = scalbn1(split, -tau);
			devOutput[addry * ldo + addrx] = input;
			max_ = MAX(max_, fabs1(input));
		}
		devMax[addrx] = max_;
	}
}
*/

// max for m
template <typename TYPE>
void ozblasSplitCKernel (
	const int32_t m,
	const int32_t n,
	const int32_t rho,
	const TYPE *devInput,
	const int32_t ldi,
	TYPE *devOutput,
	const int32_t ldo,
	TYPE *devSplit,
	const int32_t lds,
	short *devSpExp,
	TYPE *devMax,
	const int32_t splitShift
) {
	int32_t addry = 0;
	#pragma omp parallel for
	for (addry = 0; addry < n; addry++) {
		const TYPE sigma = CONST * scalbn1 (1., rho) * NextPowTwo <TYPE> (devMax[addry]) * splitShift;
		TYPE max = 0.;
		for (int32_t addrx = 0; addrx < m; addrx++) {
			TYPE input = devInput[addry * ldi + addrx];
			const TYPE split = (input + sigma) - sigma;
			input = input - split;
			devSplit[addry * lds + addrx] = split;
			devOutput[addry * ldo + addrx] = input;
			max = MAX(max, fabs1(input));
		}
		devMax[addry] = max;
	}
}

template <typename TYPE1, typename TYPE2>
void ozblasSplitCKernel (
	const int32_t m,
	const int32_t n,
	const int32_t rho,
	const TYPE1 *devInput,
	const int32_t ldi,
	TYPE1 *devOutput,
	const int32_t ldo,
	TYPE2 *devSplit,
	const int32_t lds,
	short *devSpExp,
	TYPE1 *devMax,
	const int32_t splitShift
) {
	int32_t addry = 0;
	#pragma omp parallel for
	for (addry = 0; addry < n; addry++) {
		const short tau = devSpExp[addry] = ceil(log2(devMax[addry]));
		const TYPE1 sigma = CONST * scalbn1 (1., rho + tau) * splitShift;
		TYPE1 max = 0.;
		for (int32_t addrx = 0; addrx < m; addrx++) {
			TYPE1 input = devInput[addry * ldi + addrx];
			const TYPE1 split = (input + sigma) - sigma;
			input = input - split;
			devSplit[addry * lds + addrx] = scalbn1(split, -tau);
			devOutput[addry * ldo + addrx] = input;
			max = MAX(max, fabs1(input));
		}
		devMax[addry] = max;
	}
}

template <typename TYPE1, typename TYPE2>
void ozblasSplitDevice (
	const char major,
	const int32_t m, 
	const int32_t n,
	const TYPE1 *devInput, // input matrix (devAwrk) 
	const int32_t ldi, // leading dimension of input matrix
	TYPE1 *devOutput, // output matrix (devAwrk)
	const int32_t ldo,
	TYPE2 *devSplit, // split matrices (output): this includes NumSplitMax matrices
	const int32_t lds, // leading dimension of split matrix (# of cols)
	short *devSpExp, // exponent of split matrix
	TYPE1 *devMax,
	int32_t splitEpsModeFlag,
	const int32_t splitShift
) {
	const int32_t dim = ((major == 'r') && n > 1) ? n : m;
	const int32_t rho = getRho <TYPE1, TYPE2> (dim, splitEpsModeFlag);

	if (major == 'r') {
		fprintf (OUTPUT, "OzBLAS error: ozblasSplitRKernel is not available.\n");
//		ozblasSplitRKernel (m, n, rho, devInput, ldi, devOutput, ldo, devSplit, lds, devSpExp, devMax, splitShift);
	} else {
		if (n == 1) 
			ozblasSplitVecKernel (m, rho, devInput, devOutput, devSplit, devSpExp, devMax, splitShift);
		else 
			ozblasSplitCKernel (m, n, rho, devInput, ldi, devOutput, ldo, devSplit, lds, devSpExp, devMax, splitShift);
	}
}

template <typename TYPE1, typename TYPE2>
int32_t ozblasSplit (
	ozblasHandle_t *oh,
	const char major,
	const int32_t m,
	const int32_t n,
	const TYPE1 *devInput,
	const int32_t ldi,
	TYPE1 *devOutput,
	const int32_t ldo,
	TYPE2 *devSplit,
	const int32_t lds,
	short *devSpExp, 
	const int32_t ldse,
	TYPE1 *devMax
) {
	// FindMax^(0)
	ozblasFindMaxDevice (major, m, n, devInput, ldi, devMax);
	// Split^(0) & FindMax^(1)
	ozblasSplitDevice (major, m, n, devInput, ldi, devOutput, ldo, devSplit, lds, devSpExp, devMax, oh->splitEpsModeFlag, oh->splitShift);
	const int32_t maxS = (oh->nSplitMax > 0) ? oh->nSplitMax : NumSplitDefaultMax;
	int32_t s;
	for (s = 1; s < maxS; s++) {
		const TYPE1 check = (n == 1 && major == 'c') ? devMax[0] : blasRasum ((major=='r')?m:n, devMax, 1);
		if (check == 0) return s;
		// Split^(i) & FindMax^(i+1)
		ozblasSplitDevice (major, m, n, devOutput, ldo, devOutput, ldo, &devSplit[lds*n*s], lds, &devSpExp[ldse*s], devMax, oh->splitEpsModeFlag, oh->splitShift);
	}
	if (oh->splitModeFlag > 0)
		fprintf (OUTPUT, "OzBLAS error: infSplit is failed.\n");
	return s;
}
#if defined (FLOAT128)
template int32_t ozblasSplit <__float128, double> (ozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const __float128 *devInput, const int32_t ldi, __float128 *devOutput, const int32_t ldo, double *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, __float128 *devMax);
template int32_t ozblasSplit <__float128, float> (ozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const __float128 *devInput, const int32_t ldi, __float128 *devOutput, const int32_t ldo, float *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, __float128 *devMax);
#endif
template int32_t ozblasSplit <double, double> (ozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const double *devInput, const int32_t ldi, double *devOutput, const int32_t ldo, double *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, double *devMax);
template int32_t ozblasSplit <double, float> (ozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const double *devInput, const int32_t ldi, double *devOutput, const int32_t ldo, float *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, double *devMax);
template int32_t ozblasSplit <float, float> (ozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const float *devInput, const int32_t ldi, float *devOutput, const int32_t ldo, float *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, float *devMax);
template int32_t ozblasSplit <float, double> (ozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const float *devInput, const int32_t ldi, float *devOutput, const int32_t ldo, double *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, float *devMax);

//=================================================
// for binary128
//=================================================

template <typename TYPE>
void ozblasSplitNKernel2 (
	const int32_t m,
	const int32_t n,
	const int32_t rho,
	const TYPE *devInput,
	const int32_t ldi,
	TYPE *devTmpD1,
	const int32_t ldt1,
	TYPE *devTmpD2,
	const int32_t ldt2,
	TYPE *devTmpD3,
	const int32_t ldt3,
	TYPE *devSplit,
	const int32_t lds,
	short *devSpExp,
	TYPE *devMax,
	const int32_t splitShift
) {
	int32_t addrx = 0;
	#pragma omp parallel for
	for (addrx = 0; addrx < m; addrx++) {
		const TYPE sigma = CONST * scalbn1 (1., rho) * NextPowTwo <TYPE> (devMax[addrx]) * splitShift;
		TYPE max = 0.;
		for (int32_t j = 0; j < n; j++) {
			TYPE input = devInput[j * ldi + addrx];
			const TYPE split = (input + sigma) - sigma;
			input = input - split;
			devSplit[j * lds + addrx] = split;
			TYPE tmpD1 = input;
			TYPE tmpD2 = devTmpD2[j * ldt2 + addrx];
			TYPE tmpD3 = devTmpD3[j * ldt3 + addrx];
			QuickTwoSum (tmpD1, tmpD2, tmpD1, tmpD2);
			QuickTwoSum (tmpD2, tmpD3, tmpD2, tmpD3);
			devTmpD1[j * ldt1 + addrx] = tmpD1;
			devTmpD2[j * ldt2 + addrx] = tmpD2;
			devTmpD3[j * ldt3 + addrx] = tmpD3;
			max = MAX(max, fabs1(tmpD1));
		}
		devMax[addrx] = max;
	}
}

template <typename TYPE>
void ozblasSplitTKernel2 (
	const int32_t m,
	const int32_t n,
	const int32_t rho,
	const TYPE *devInput,
	const int32_t ldi,
	TYPE *devTmpD1,
	const int32_t ldt1,
	TYPE *devTmpD2,
	const int32_t ldt2,
	TYPE *devTmpD3,
	const int32_t ldt3,
	TYPE *devSplit,
	const int32_t lds,
	short *devSpExp,
	TYPE *devMax,
	const int32_t splitShift
) {
	int32_t addry = 0;
	#pragma omp parallel for
	for (addry = 0; addry < n; addry++) {
		const TYPE sigma = CONST * scalbn1 (1., rho) * NextPowTwo <TYPE> (devMax[addry]) * splitShift;
		TYPE max = 0.;
		for (int32_t i = 0; i < m; i++) {
			TYPE input = devInput[addry * ldi + i];
			const TYPE split = (input + sigma) - sigma;
			input = input - split;
			devSplit[addry * lds + i] = split;
			TYPE tmpD1 = input;
			TYPE tmpD2 = devTmpD2[addry * ldt2 + i];
			TYPE tmpD3 = devTmpD3[addry * ldt3 + i];
			QuickTwoSum (tmpD1, tmpD2, tmpD1, tmpD2);
			QuickTwoSum (tmpD2, tmpD3, tmpD2, tmpD3);
			devTmpD1[addry * ldt1 + i] = tmpD1;
			devTmpD2[addry * ldt2 + i] = tmpD2;
			devTmpD3[addry * ldt3 + i] = tmpD3;
			max = MAX(max, fabs1(tmpD1));
		}
		devMax[addry] = max;
	}
}


template <typename TYPE>
void ozblasSplitDevice2 (
	const char major,
	const int32_t m, 
	const int32_t n,
	const TYPE *devInput,
	const int32_t ldi,
	TYPE *devTmpD1,
	const int32_t ldt1,
	TYPE *devTmpD2,
	const int32_t ldt2,
	TYPE *devTmpD3,
	const int32_t ldt3,
	TYPE *devSplit, 
	const int32_t lds, 
	short *devSpExp, 
	TYPE *devMax,
	int32_t splitEpsModeFlag,
	const int32_t splitShift
) {
	const int32_t dim = ((major == 'r') && n > 1) ? n : m;
	const int32_t rho = getRho <TYPE, TYPE> (dim, splitEpsModeFlag);

	if (major == 'r') {
		ozblasSplitNKernel2 (m, n, rho, devInput, ldi, devTmpD1, ldt1, devTmpD2, ldt2, devTmpD3, ldt3, devSplit, lds, devSpExp, devMax, splitShift);
	} else {
//		if (n == 1) 
//			ozblasSplitVecKernel2 (m, rho, devInput, devOutput, devSplit, devSpExp, devMax, splitShift);
//		else
			ozblasSplitTKernel2 (m, n, rho, devInput, ldi, devTmpD1, ldt1, devTmpD2, ldt2, devTmpD3, ldt3, devSplit, lds, devSpExp, devMax, splitShift);
	}
}

template <typename TYPE1, typename TYPE2>
int32_t ozblasSplit3 (
	ozblasHandle_t *oh,
	const char major,
	const int32_t m,
	const int32_t n,
	const TYPE1 *devInput,
	const int32_t ldi,
	TYPE2 *devSplit,
	const int32_t lds,
	short *devSpExp,
	const int32_t ldse,
	TYPE2 *devMax,
	TYPE2 *devTmpD1,
	const int32_t ldt1,
	TYPE2 *devTmpD2,
	const int32_t ldt2,
	TYPE2 *devTmpD3,
	const int32_t ldt3,
	TYPE2 *devTmp,
	const int32_t ldt
) {
	int32_t addry;
	#pragma omp parallel for
	for (addry = 0; addry < n; addry++) {
		for (int32_t addrx = 0; addrx < m; addrx++) {
			TYPE1 tmp1q = devInput[addry * ldi + addrx];
			TYPE2 tmp1d = tmp1q;
			TYPE1 tmpq = tmp1q - (TYPE1)tmp1d; 
			TYPE2 tmpd = tmpq;
			TYPE2 tmp3d = tmpq - (TYPE1)tmpd;
			devTmpD1[addry * ldt1 + addrx] = tmp1d;
			devTmpD2[addry * ldt2 + addrx] = tmpd;
			devTmpD3[addry * ldt3 + addrx] = tmp3d;
		}
	}
				
	// FindMax^(0)
	ozblasFindMaxDevice (major, m, n, devTmpD1, ldt1, devMax);
	// Split^(0) & FindMax^(1)
	ozblasSplitDevice2 (major, m, n, devTmpD1, ldt1, devTmp, ldt, devTmpD2, ldt2, devTmpD3, ldt3, &devSplit[0], lds, devSpExp, devMax, oh->splitEpsModeFlag, oh->splitShift);
	const int32_t maxS = (oh->nSplitMax > 0) ? oh->nSplitMax : NumSplitDefaultMax;
	int32_t s;
	for (s = 1; s < maxS; s++) {
		const TYPE2 check = (n == 1 && major == 'c') ? devMax[0] : blasRasum ((major=='r')?m:n, devMax, 1);
		if (check == 0) return s;
		// Split^(i) & FindMax^(i+1)
		ozblasSplitDevice2 (major, m, n, devTmp, ldt, devTmpD1, ldt1, devTmpD2, ldt2, devTmpD3, ldt3, &devSplit[lds*n*s], lds, &devSpExp[ldse*s], devMax, oh->splitEpsModeFlag, oh->splitShift);
	}
	if (oh->splitModeFlag > 0)
		fprintf (OUTPUT, "OzBLAS error: infSplit is failed.\n");
	return s;
}

#if defined (FLOAT128)
template int32_t ozblasSplit3 (ozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const __float128 *devInput, const int32_t ldi, double *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, double *devMax, double *devTmpD1, const int32_t ldt1, double *devTmpD2, const int32_t ldt2, double *devTmpD3, const int32_t ldt3, double *devTmp, const int32_t ldt);
template int32_t ozblasSplit3 (ozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const __float128 *devInput, const int32_t ldi, float *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, float *devMax, float *devTmpD1, const int32_t ldt1, float *devTmpD2, const int32_t ldt2, float *devTmpD3, const int32_t ldt3, float *devTmp, const int32_t ldt);
#endif
template int32_t ozblasSplit3 (ozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const double *devInput, const int32_t ldi, double *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, double *devMax, double *devTmpD1, const int32_t ldt1, double *devTmpD2, const int32_t ldt2, double *devTmpD3, const int32_t ldt3, double *devTmp, const int32_t ldt);
template int32_t ozblasSplit3 (ozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const float *devInput, const int32_t ldi, double *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, double *devMax, double *devTmpD1, const int32_t ldt1, double *devTmpD2, const int32_t ldt2, double *devTmpD3, const int32_t ldt3, double *devTmp, const int32_t ldt);
template int32_t ozblasSplit3 (ozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const double *devInput, const int32_t ldi, float *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, float *devMax, float *devTmpD1, const int32_t ldt1, float *devTmpD2, const int32_t ldt2, float *devTmpD3, const int32_t ldt3, float *devTmp, const int32_t ldt);
template int32_t ozblasSplit3 (ozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const float *devInput, const int32_t ldi, float *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, float *devMax, float *devTmpD1, const int32_t ldt1, float *devTmpD2, const int32_t ldt2, float *devTmpD3, const int32_t ldt3, float *devTmp, const int32_t ldt);

//=================================================
// determination of the num of split matrices
//=================================================

template <typename TYPE>
void ozblasGenEVec (
	const int32_t n,
	TYPE *devA
) {
	int32_t addrx;
	#pragma omp parallel for
	for (addrx = 0; addrx < n; addrx++) 
		devA[addrx] = 1.;
}

template <typename TYPE>
int32_t ozblasCompVecAll (
	const int n,
	const TYPE *devA,
	const TYPE *devB
) {
	int32_t addrx;
	int32_t val = 0;
	#pragma omp parallel for 
	for (addrx = 0; addrx < n; addrx++) {
		if (devA[addrx] > devB[addrx]) {
			#pragma omp critical (atomicadd)
			{
				val++;
			}
		}
	}
	return val;
}

template <typename TYPE>
void ozblasAbsMat (
	const int m,
	const int n,
	const TYPE *devA,
	const int lda,
	TYPE *devB,
	const int ldb
) {
	int addrx, addry;
	#pragma omp parallel for private (addrx)
	for (addry = 0; addry < n; addry++) {
		for (addrx = 0; addrx < m; addrx++) 
			devB[addry*ldb+addrx] = fabs1(devA[addry*lda+addrx]);
	}
}

template <typename TYPE1, typename TYPE2>
void ozblasSplitNKernelA (
	const int32_t m,
	const int32_t n,
	const int32_t rho,
	const TYPE1 *devInput,
	const int32_t ldi,
	TYPE1 *devOutput,
	const int32_t ldo,
	TYPE2 *devSplit,
	const int32_t lds,
	TYPE1 *devSplitD,
	const int32_t ldd,
	short *devSpExp,
	TYPE1 *devMax
) {
	int32_t addrx = 0;
	#pragma omp parallel for
	for (addrx = 0; addrx < m; addrx++) {
		volatile TYPE1 max = 0., sigma, input, split, tmp;
		short tau = ceil(log2(devMax[addrx]));
		sigma = CONST * scalbn1 (1., rho + tau);
		devSpExp[addrx] = tau;
		for (int32_t j = 0; j < n; j++) {
			input = devInput[j * ldi + addrx];
			tmp = input + sigma;
			split = tmp - sigma;
			input = input - split;
			devSplitD[j * ldd + addrx] = fabs1(split);
			devSplit[j * lds + addrx] = scalbn1(split, -tau);
			devOutput[j * ldo + addrx] = input;
			if (max < fabs1(input)) max = fabs1(input);
		}
		devMax[addrx] = max;
	}
}

template <typename TYPE>
void ozblasSplitNKernelA (
	const int32_t m,
	const int32_t n,
	const int32_t rho,
	const TYPE *devInput,
	const int32_t ldi,
	TYPE *devOutput,
	const int32_t ldo,
	TYPE *devSplit,
	const int32_t lds,
	TYPE *devSplitD,
	const int32_t ldd,
	short *devSpExp,
	TYPE *devMax
) {
	int32_t addrx = 0;
	#pragma omp parallel for
	for (addrx = 0; addrx < m; addrx++) {
		volatile TYPE max = 0., sigma, input, split, tmp;
		sigma = CONST * scalbn1 (1., rho) * NextPowTwo <TYPE> (devMax[addrx]);
		for (int32_t j = 0; j < n; j++) {
			input = devInput[j * ldi + addrx];
			tmp = input + sigma;
			split = tmp - sigma;
			input = input - split;
			devSplitD[j * ldd + addrx] = fabs1(split);
			devSplit[j * lds + addrx] = split;
			devOutput[j * ldo + addrx] = input;
			if (max < fabs1(input)) max = fabs1(input);
		}
		devMax[addrx] = max;
	}
}

template <typename TYPE1, typename TYPE2>
void ozblasSplitTKernelA (
	const int32_t m,
	const int32_t n,
	const int32_t rho,
	const TYPE1 *devInput,
	const int32_t ldi,
	TYPE1 *devOutput,
	const int32_t ldo,
	TYPE2 *devSplit,
	const int32_t lds,
	TYPE1 *devSplitD,
	const int32_t ldd,
	short *devSpExp,
	TYPE1 *devMax
) {
	int32_t addry = 0;
	#pragma omp parallel for
	for (addry = 0; addry < n; addry++) {
		volatile TYPE1 max = 0., sigma, input, split, tmp;
		short tau = ceil(log2(devMax[addry]));
		sigma = CONST * scalbn1 (1., rho + tau);
		devSpExp[addry] = tau;
		for (int32_t i = 0; i < m; i++) {
			input = devInput[addry * ldi + i];
			tmp = input + sigma;
			split = tmp - sigma;
			input = input - split;
			devSplitD[addry * ldd + i] = fabs1(split);
			devSplit[addry * lds + i] = scalbn1(split, -tau);
			devOutput[addry * ldo + i] = input;
			if (max < fabs1(input)) max = fabs1(input);
		}
		devMax[addry] = max;
	}
}

template <typename TYPE>
void ozblasSplitTKernelA (
	const int32_t m,
	const int32_t n,
	const int32_t rho,
	const TYPE *devInput,
	const int32_t ldi,
	TYPE *devOutput,
	const int32_t ldo,
	TYPE *devSplit,
	const int32_t lds,
	TYPE *devSplitD,
	const int32_t ldd,
	short *devSpExp,
	TYPE *devMax
) {
	int32_t addry = 0;
	#pragma omp parallel for
	for (addry = 0; addry < n; addry++) {
		volatile TYPE max = 0., sigma, input, split, tmp;
		sigma = CONST * scalbn1 (1., rho) * NextPowTwo <TYPE> (devMax[addry]);
		for (int32_t i = 0; i < m; i++) {
			input = devInput[addry * ldi + i];
			tmp = input + sigma;
			split = tmp - sigma;
			input = input - split;
			devSplitD[addry * ldd + i] = fabs1(split);
			devSplit[addry * lds + i] = split;
			devOutput[addry * ldo + i] = input;
			if (max < fabs1(input)) max = fabs1(input);
		}
		devMax[addry] = max;
	}
}

template <typename TYPE1, typename TYPE2>
void ozblasSplitDeviceA (
	const char major,
	const int32_t m, 
	const int32_t n,
	const TYPE1 *devInput, 
	const int32_t ldi, 
	TYPE1 *devOutput, 
	const int32_t ldo,
	TYPE2 *devSplit, 
	const int32_t lds, 
	TYPE1 *devSplitD, 
	const int32_t ldd,
	short *devSpExp, 
	TYPE1 *devMax,
	int32_t splitEpsModeFlag
) {
	const int32_t dim = ((major == 'r') && n > 1) ? n : m;
	int32_t rho = getRho <TYPE1, TYPE2> (dim, splitEpsModeFlag);
	if (major == 'r') 
		ozblasSplitNKernelA (m, n, rho, devInput, ldi, devOutput, ldo, devSplit, lds, devSplitD, ldd, devSpExp, devMax);
	else 
		ozblasSplitTKernelA (m, n, rho, devInput, ldi, devOutput, ldo, devSplit, lds, devSplitD, ldd, devSpExp, devMax);
}

template <typename TYPE1, typename TYPE2>
int32_t ozblasSplitA (
	ozblasHandle_t *oh,
	const char major,
	const int32_t ma,
	const int32_t ka,
	const TYPE1 *devAInput,
	const int32_t ldai,
	const int32_t kb,
	const int32_t nb,
	const TYPE1 *devBInput,
	const int32_t ldbi,
	TYPE1 *devAOutput,
	const int32_t ldao,
	TYPE2 *devASplit,
	const int32_t ldas,
	short *devASpExp,
	const int32_t ldase,
	TYPE1 *devAMax,
	TYPE1 *devAtmp, 
	const int32_t ldat,
	TYPE1 *devBtmp,
	const int32_t ldbt,
	TYPE1 *devE,
	TYPE1 *devBe, 
	TYPE1 *devB1,
	TYPE1 *devB2
) {
	const char Atran = (major == 'r') ? 'n':'t';
	const char Btran = 'n';
	TYPE1 done = 1., dzero = 0.;

	// For determining # of splits -------------------------------------------------------------
	constexpr int32_t epse = getEpse <TYPE1> ();
	TYPE1 c = 2. * scalbn1 (sqrt(kb), -epse);
	ozblasGenEVec (nb, devE);	 								// devE    = ones (n, 1)
	ozblasAbsMat (ma, ka, devAInput, ldai, devAtmp, ldat);		// devAtmp = abs (devA)
	ozblasAbsMat (kb, nb, devBInput, ldbi, devBtmp, ldbt);		// devBtmp = abs (devB)
	blasRgemv (Btran, kb, nb, done, devBtmp, ldbt, devE, 1, dzero, devBe, 1); 	// devBe = devBtmp * e
	blasRgemv (Atran, ma, ka, c, devAtmp, ldat, devBe, 1, dzero, devB1, 1); 	// devB1 = c * devAtmp * devBe

	// FindMax^(0) -----------------------------------------------------------------------------
	ozblasFindMaxDevice (major, ma, ka, devAInput, ldai, devAMax);
	// Split^(0) & FindMax^(1) -----------------------------------------------------------------
	ozblasSplitDevice (major, ma, ka, devAInput, ldai, devAOutput, ldao, &devASplit[0], ldas, devASpExp, devAMax, oh->splitEpsModeFlag, 1);
	// Split^(i) & FindMax^(i+1) ---------------------------------------------------------------
	int32_t s;
	int32_t maxS = (oh->nSplitMax > 0) ? oh->nSplitMax : NumSplitDefaultMax;
	for (s = 1; s < maxS; s++) {
		TYPE1 check = (ka == 1 && major == 'c') ? devAMax[0] : blasRasum ((major=='r')?ma:ka, devAMax, 1);
		if (check == 0) {
			oh->fastModeFlag = 0; // turn fastmode off
			fprintf (OUTPUT, "OzBLAS warning: fastMode was disabled. The result may not be accurate enough.\n");
			return s;
		}
		// Split^(i) & FindMax^(i+1)
		ozblasSplitDeviceA (major, ma, ka, devAOutput, ldao, devAOutput, ldao, &devASplit[ldas*ka*s], ldas, devAtmp, ldat, &devASpExp[ldase*s], devAMax, oh->splitEpsModeFlag);
		// Determining # of splits -------------------------------------------------------------
		TYPE1 di = (TYPE1)(s+1);
		blasRgemv (Atran, ma, ka, di, devAtmp, ldat, devBe, 1, dzero, devB2, 1);	// devB2 = i * devASplit * devBe
		int32_t icheck = ozblasCompVecAll (ma, devB1, devB2);
		if (icheck == ma) return s;
		// ------------------------------------------------------------------------------------------
	}
	if (oh->splitModeFlag > 0)
		fprintf (OUTPUT, "OzBLAS error: infSplit is failed.\n");
	return s;
}
#if defined (FLOAT128)
template int32_t ozblasSplitA (ozblasHandle_t *oh, const char major, const int32_t ma, const int32_t ka, const __float128 *devAInput, const int32_t ldai, const int32_t kb, const int32_t nb, const __float128 *devBInput, const int32_t ldbi, __float128 *devAOutput, const int32_t ldao, double *devASplit, const int32_t ldas, short *devASpExp, const int32_t ldase, __float128 *devAMax, __float128 *devAtmp, const int32_t ldat, __float128 *devBtmp, const int32_t ldbt, __float128 *devE, __float128 *devBe, __float128 *devB1, __float128 *devB2);
template int32_t ozblasSplitA (ozblasHandle_t *oh, const char major, const int32_t ma, const int32_t ka, const __float128 *devAInput, const int32_t ldai, const int32_t kb, const int32_t nb, const __float128 *devBInput, const int32_t ldbi, __float128 *devAOutput, const int32_t ldao, float *devASplit, const int32_t ldas, short *devASpExp, const int32_t ldase, __float128 *devAMax, __float128 *devAtmp, const int32_t ldat, __float128 *devBtmp, const int32_t ldbt, __float128 *devE, __float128 *devBe, __float128 *devB1, __float128 *devB2);
#endif
template int32_t ozblasSplitA (ozblasHandle_t *oh, const char major, const int32_t ma, const int32_t ka, const double *devAInput, const int32_t ldai, const int32_t kb, const int32_t nb, const double *devBInput, const int32_t ldbi, double *devAOutput, const int32_t ldao, double *devASplit, const int32_t ldas, short *devASpExp, const int32_t ldase, double *devAMax, double *devAtmp, const int32_t ldat, double *devBtmp, const int32_t ldbt, double *devE, double *devBe, double *devB1, double *devB2);
template int32_t ozblasSplitA (ozblasHandle_t *oh, const char major, const int32_t ma, const int32_t ka, const double *devAInput, const int32_t ldai, const int32_t kb, const int32_t nb, const double *devBInput, const int32_t ldbi, double *devAOutput, const int32_t ldao, float *devASplit, const int32_t ldas, short *devASpExp, const int32_t ldase, double *devAMax, double *devAtmp, const int32_t ldat, double *devBtmp, const int32_t ldbt, double *devE, double *devBe, double *devB1, double *devB2);
template int32_t ozblasSplitA (ozblasHandle_t *oh, const char major, const int32_t ma, const int32_t ka, const float *devAInput, const int32_t ldai, const int32_t kb, const int32_t nb, const float *devBInput, const int32_t ldbi, float *devAOutput, const int32_t ldao, float *devASplit, const int32_t ldas, short *devASpExp, const int32_t ldase, float *devAMax, float *devAtmp, const int32_t ldat, float *devBtmp, const int32_t ldbt, float *devE, float *devBe, float *devB1, float *devB2);

// =========================================
// Sparse
// =========================================

template <typename TYPE>
void ozblasFindMaxSparseNKernel (
	const int32_t m,
	const TYPE *devInput,
	const int32_t *devRowptr,
	TYPE *devMax
) {
	#pragma omp parallel for schedule (static) 
	for (int32_t addrx = 0; addrx < m; addrx++) {
		TYPE max = 0.;
		for (int32_t i = devRowptr[addrx]; i < devRowptr[addrx+1]; i++) 
			max = MAX(max, fabs1(devInput[i]));
		devMax[addrx] = max;
	}
}

template <typename TYPE>
void ozblasFindMaxSparseDevice (
	const char major,
	const int32_t m, 
	const TYPE *devInput,
	const int32_t *devRowptr,
	TYPE *devMax
) {
	if (major == 'r') {
		ozblasFindMaxSparseNKernel (m, devInput, devRowptr, devMax);
	} else {
		fprintf (OUTPUT, "OzBLAS error: Split-T is not implemented.\n");
		exit (1);
	}
}

template <typename TYPE>
void ozblasSplitSparseNKernel (
	const int32_t m,
	const TYPE *devInput,
	const int32_t *devRowptr,
	TYPE *devOutput,
	TYPE *devSplit,
	short *devSpExp,
	TYPE *devMax,
	const int32_t splitEpsModeFlag,
	const int32_t splitShift
) {
	int32_t addrx = 0;
	#pragma omp parallel for schedule (static) 
	for (addrx = 0; addrx < m; addrx++) {
		const int32_t dim = devRowptr[addrx+1] - devRowptr[addrx];
		const int32_t rho = getRho <TYPE, TYPE> (dim, splitEpsModeFlag);
		const TYPE sigma = CONST * scalbn1 (1., rho) * NextPowTwo <TYPE> (devMax[addrx]) / splitShift;
		TYPE max = 0.;
		for (int32_t i = devRowptr[addrx]; i < devRowptr[addrx+1]; i++) {
			TYPE input = devInput[i];
			const TYPE split = (input + sigma) - sigma;
			input = input - split;
			devSplit[i] = split;
			devOutput[i] = input;
			max = MAX(max, fabs1(input));
		}
		devMax[addrx] = max;
	}
}

template <typename TYPE1, typename TYPE2>
void ozblasSplitSparseNKernel (
	const int32_t m,
	const TYPE1 *devInput,
	const int32_t *devRowptr,
	TYPE1 *devOutput,
	TYPE2 *devSplit,
	short *devSpExp,
	TYPE1 *devMax,
	const int32_t splitEpsModeFlag,
	const int32_t splitShift
) {
	int32_t addrx = 0;
	#pragma omp parallel for schedule (static) 
	for (addrx = 0; addrx < m; addrx++) {
		const int32_t dim = devRowptr[addrx+1] - devRowptr[addrx];
		const int32_t rho = getRho <TYPE1, TYPE2> (dim, splitEpsModeFlag);
		const short tau = devSpExp[addrx] = ceil(log2(devMax[addrx]));
		const TYPE1 sigma = CONST * scalbn1 (1., rho+tau) / splitShift;
		TYPE1 max = 0.;
		for (int32_t i = devRowptr[addrx]; i < devRowptr[addrx+1]; i++) {
			TYPE1 input = devInput[i];
			const TYPE1 split = (input + sigma) - sigma;
			input = input - split;
			devSplit[i] = scalbn1(split, -tau);
			devOutput[i] = input;
			max = MAX(max, fabs1(input));
		}
		devMax[addrx] = max;
	}
}

template <typename TYPE1, typename TYPE2>
void ozblasSplitSparseDevice (
	const char major,
	const int32_t m, 
	const TYPE1 *devInput, 
	const int32_t *devRowptr,
	TYPE1 *devOutput,
	TYPE2 *devSplit, 
	const int32_t lds, 
	short *devSpExp, 
	TYPE1 *devMax,
	int32_t splitEpsModeFlag,
	const int32_t splitShift
) {
	if (major == 'r') {
		ozblasSplitSparseNKernel (m, devInput, devRowptr, devOutput, devSplit, devSpExp, devMax, splitEpsModeFlag, splitShift);
	} else {
		fprintf (OUTPUT, "OzBLAS error: SplitT is not implemented.\n");
		exit (1);
	}
}

template <typename TYPE1, typename TYPE2>
int32_t ozblasSplitSparse (
	ozblasHandle_t *oh,
	const char major,
	const int32_t m,
	const TYPE1 *devInput,
	const int32_t *devRowptr,
	TYPE1 *devOutput,
	TYPE2 *devSplit,
	const int32_t lds,
	short *devSpExp, 
	const int32_t ldse,
	TYPE1 *devMax
) {
	// ozblasFindMax^(0)
	ozblasFindMaxSparseDevice (major, m, devInput, devRowptr, devMax);
	// Split^(0) & ozblasFindMax^(1)
	ozblasSplitSparseDevice (major, m, devInput, devRowptr, devOutput, &devSplit[0], lds, devSpExp, devMax, oh->splitEpsModeFlag, oh->splitShift);
	const int32_t maxS = (oh->nSplitMax > 0) ? oh->nSplitMax : NumSplitDefaultMax;
	int32_t s;
	for (s = 1; s < maxS; s++) {
		const TYPE1 check = blasRasum (m, devMax, 1);
		if (check == 0) return s;
		// Split^(i) & ozblasFindMax^(i+1)
		ozblasSplitSparseDevice (major, m, devOutput, devRowptr, devOutput, &devSplit[lds*s], lds, &devSpExp[ldse*s], devMax, oh->splitEpsModeFlag, oh->splitShift);
	}
	if (oh->splitModeFlag > 0)
		fprintf (OUTPUT, "OzBLAS error: infSplit is failed.\n");
	return s;
}
#if defined (FLOAT128)
template int32_t ozblasSplitSparse <__float128, double> (ozblasHandle_t *oh, const char major, const int32_t m, const __float128 *devInput, const int32_t *devRowptr, __float128 *devOutput, double *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, __float128 *devMax);
template int32_t ozblasSplitSparse <__float128, float> (ozblasHandle_t *oh, const char major, const int32_t m, const __float128 *devInput, const int32_t *devRowptr, __float128 *devOutput, float *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, __float128 *devMax);
#endif
template int32_t ozblasSplitSparse <double, double> (ozblasHandle_t *oh, const char major, const int32_t m, const double *devInput, const int32_t *devRowptr, double *devOutput, double *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, double *devMax);
template int32_t ozblasSplitSparse <double, float> (ozblasHandle_t *oh, const char major, const int32_t m, const double *devInput, const int32_t *devRowptr, double *devOutput, float *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, double *devMax);
template int32_t ozblasSplitSparse <float, float> (ozblasHandle_t *oh, const char major, const int32_t m, const float *devInput, const int32_t *devRowptr, float *devOutput, float *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, float *devMax);
template int32_t ozblasSplitSparse <float, double> (ozblasHandle_t *oh, const char major, const int32_t m, const float *devInput, const int32_t *devRowptr, float *devOutput, double *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, float *devMax);
