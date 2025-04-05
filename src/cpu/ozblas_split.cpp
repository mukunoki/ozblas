#include "ozblas_common.h"

// =========================================
// Split
// Based on K.Ozaki et al., "Error-free transformations of matrix multiplication
// by using fast routines of matrix multiplication and its applications", 2012
// =========================================

#define CONST 0.75

/*
template <typename TYPE>
TYPE NextPowTwo (const TYPE p) {
	constexpr int32_t epse_type = getEpse <TYPE> ();
	return scalbn1 (p, epse_type) - (TYPE)((scalbn1 (1., epse_type) - 1) * p);
}
*/

template <typename TYPE1, typename TYPE2>
int32_t getRho (const int32_t dim, const int32_t splitEpsMode) {
	constexpr int32_t epse_type1 = getEpse <TYPE1> ();
	constexpr int32_t epse_type2 = getEpse <TYPE2> ();
	constexpr int32_t epse_type22 = getEpse2 <TYPE2> ();
	switch (splitEpsMode) {
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

template <typename TYPE>
void copyVec (
	const int32_t n,
	TYPE *devSplit,
	const TYPE *devTmp
) {
	#pragma omp parallel for
	for (int32_t addrx = 0; addrx < n; addrx++) 
		devSplit[addrx] = devTmp[addrx];
}
	

template <typename TYPE>
void copyMat (
	const int32_t m,
	const int32_t n,
	TYPE *devSplit,
	const int32_t lds,
	const TYPE *devTmp,
	const int32_t ldt
) {
	#pragma omp parallel for
	for (int32_t addry = 0; addry < n; addry++) {
		for (int32_t addrx = 0; addrx < m; addrx++) {
            devSplit[addry * lds + addrx] = devTmp[addry * ldt + addrx];
        }
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
	#pragma omp parallel for
	for (int32_t addry = 0; addry < n; addry++) {
		TYPE max = 0.;
		for (int32_t addrx = 0; addrx < m; addrx++) 
			max = std::max(max, fabs1(devInput[addry * ldi + addrx]));
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
			fprintf (OUTPUT, "OzBLAS error: ozblasFindMaxRKernel is not available.\n");
		} else {
			ozblasFindMaxCKernel (m, n, devInput, ldi, devMax);
		}
	}
}

template <typename TYPE>
TYPE ozblasSplitVecKernel (
	const int32_t n,
	const int32_t rho,
	const TYPE *devInput,
	TYPE *devOutput,
	TYPE *devSplit,
	short *devSpExp,
	TYPE *devMax,
	const int32_t splitShift,
	const int32_t reproflag
) {
	const short tau = ceil(log21(fabs1(devMax[0])));
	const TYPE one = (TYPE)1.;
	const TYPE sigma = CONST * scalbn1 (one, rho + tau) * splitShift;
	TYPE max_ = 0.;
	#pragma omp parallel for reduction(max:max_)
	for (int32_t i = 0; i < n; i++) {
		TYPE input = devInput[i];
		if (reproflag == 0) devSplit[n] = input;
		const TYPE split = (input + sigma) - sigma;
		input = input - split;
		devSplit[i] = split;
		devOutput[i] = input;
		max_ = std::max(max_, fabs1(input));
	}
	devMax[0] = max_;
	return max_;
}

template <typename TYPE1, typename TYPE2>
TYPE1 ozblasSplitVecKernel (
	const int32_t n,
	const int32_t rho,
	const TYPE1 *devInput,
	TYPE1 *devOutput,
	TYPE2 *devSplit,
	short *devSpExp,
	TYPE1 *devMax,
	const int32_t splitShift,
	const int32_t reproflag
) {
	const short tau = devSpExp[0] = ceil(log21(fabs1(devMax[0])));
	const TYPE1 one = (TYPE1)1.;
	const TYPE1 sigma = CONST * scalbn1 (one, rho + tau) * splitShift;
	TYPE1 max_ = 0.;
	#pragma omp parallel for reduction(max:max_)
	for (int32_t i = 0; i < n; i++) {
		TYPE1 input = devInput[i];
		const TYPE1 split = (input + sigma) - sigma;
		input = input - split;
		devSplit[i] = scalbn1(split, -tau);
		devOutput[i] = input;
		max_ = std::max(max_, fabs1(input));
	}
	devMax[0] = max_;
	return max_;
}

// max for m
template <typename TYPE>
TYPE ozblasSplitCKernel (
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
	const int32_t splitShift,
	const int32_t reproflag
) {
	TYPE check = 0.;
	const TYPE one = (TYPE)1.;
	#pragma omp parallel for
	for (int32_t addry = 0; addry < n; addry++) {
		const short tau = ceil(log21(fabs1(devMax[addry])));
		const TYPE sigma = CONST * scalbn1 (one, rho + tau) * splitShift;
		TYPE max = 0.;
		for (int32_t addrx = 0; addrx < m; addrx++) {
			TYPE input = devInput[addry * ldi + addrx];
			if (reproflag == 0) devSplit[addry * lds + addrx + lds * n] = input;
			const TYPE split = (input + sigma) - sigma;
			input = input - split;
			devSplit[addry * lds + addrx] = split;
			devOutput[addry * ldo + addrx] = input;
			max = std::max(max, fabs1(input));
		}
		devMax[addry] = max;
		#pragma omp critical (atomicadd)
		{
			check += max;
		}
	}
	return check;
}

template <typename TYPE1, typename TYPE2>
TYPE1 ozblasSplitCKernel (
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
	const int32_t splitShift,
	const int32_t reproflag
) {
	TYPE1 check = 0.;
	const TYPE1 one = (TYPE1)1.;
	#pragma omp parallel for
	for (int32_t addry = 0; addry < n; addry++) {
		const short tau = devSpExp[addry] = ceil(log21(fabs1(devMax[addry])));
		const TYPE1 sigma = CONST * scalbn1 (one, rho + tau) * splitShift;
		TYPE1 max = 0.;
		for (int32_t addrx = 0; addrx < m; addrx++) {
			TYPE1 input = devInput[addry * ldi + addrx];
			const TYPE1 split = (input + sigma) - sigma;
			input = input - split;
			devSplit[addry * lds + addrx] = scalbn1(split, -tau);
			devOutput[addry * ldo + addrx] = input;
			max = std::max(max, fabs1(input));
		}
		devMax[addry] = max;
		#pragma omp critical (atomicadd)
		{
			check += max;
		}
	}
	return check;
}

template <typename TYPE1, typename TYPE2>
TYPE1 ozblasSplitDevice (
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
	int32_t splitEpsMode,
	const int32_t splitShift,
	const int32_t reproflag
) {
	const int32_t dim = ((major == 'r') && n > 1) ? n : m;
	const int32_t rho = getRho <TYPE1, TYPE2> (dim, splitEpsMode);
	if (major == 'r') {
		fprintf (OUTPUT, "OzBLAS error: ozblasSplitRKernel is not available.\n");
	} else {
		if (n == 1) 
			return ozblasSplitVecKernel (m, rho, devInput, devOutput, devSplit, devSpExp, devMax, splitShift, reproflag);
		else 
			return ozblasSplitCKernel (m, n, rho, devInput, ldi, devOutput, ldo, devSplit, lds, devSpExp, devMax, splitShift, reproflag);
	}
	return 0;
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
	int32_t s;
	// FindMax^(0)
	ozblasFindMaxDevice (major, m, n, devInput, ldi, devMax);
	// Split^(0) & FindMax^(1)
	if (ozblasSplitDevice (major, m, n, devInput, ldi, devOutput, ldo, devSplit, lds, devSpExp, devMax, oh->splitEpsMode, oh->splitShift, oh->reproMode) == 0.) return s;
	int32_t maxS = (oh->nSplitMax > 0) ? oh->nSplitMax : NumSplitDefaultMax;
	// Split^(s) & FindMax^(s+1)
	for (s = 1; s < maxS; s++) {
		if (ozblasSplitDevice (major, m, n, devOutput, ldo, devOutput, ldo, &devSplit[lds*n*s], lds, &devSpExp[ldse*s], devMax, oh->splitEpsMode, oh->splitShift, oh->reproMode) == 0.) break;
	}
	if constexpr (std::is_same<TYPE1, TYPE2>::value) {
		if (oh->reproMode == 0) {
			if (s == maxS && oh->nSplitMax > 0) 
				copyMat (m, n, &devSplit[lds*n*(s-1)], lds, &devSplit[lds*n*(s)], lds);
			if (oh->nSplitMax == 0) 
				copyMat (m, n, &devSplit[lds*n*(s)], lds, &devSplit[lds*n*(s+1)], lds);
			return s;
		}
		if (s == maxS && oh->nSplitMax > 0) 
			return s;
	}
	s++;
	if (oh->splitMode == 1) fprintf (OUTPUT, "OzBLAS error: still splittable.\n");
	return s;
}
template int32_t ozblasSplit <__float128, double> (ozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const __float128 *devInput, const int32_t ldi, __float128 *devOutput, const int32_t ldo, double *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, __float128 *devMax);
template int32_t ozblasSplit <__float128, float> (ozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const __float128 *devInput, const int32_t ldi, __float128 *devOutput, const int32_t ldo, float *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, __float128 *devMax);
template int32_t ozblasSplit <double, double> (ozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const double *devInput, const int32_t ldi, double *devOutput, const int32_t ldo, double *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, double *devMax);
template int32_t ozblasSplit <double, float> (ozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const double *devInput, const int32_t ldi, double *devOutput, const int32_t ldo, float *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, double *devMax);
template int32_t ozblasSplit <float, float> (ozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const float *devInput, const int32_t ldi, float *devOutput, const int32_t ldo, float *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, float *devMax);
template int32_t ozblasSplit <float, double> (ozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const float *devInput, const int32_t ldi, float *devOutput, const int32_t ldo, double *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, float *devMax);

//=================================================
// for binary128
//=================================================

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

template <typename TYPE>
TYPE ozblasSplitCKernel3 (
	const int32_t m,
	const int32_t n,
	const int32_t rho,
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
	const int32_t splitShift,
	const int32_t reproflag
) {
	TYPE check = 0.;
	const TYPE one = (TYPE)1.;
	#pragma omp parallel for
	for (int32_t addry = 0; addry < n; addry++) {
		const short tau = ceil(log21(fabs1(devMax[addry])));
		const TYPE sigma = CONST * scalbn1 (one, rho + tau) * splitShift;
		TYPE max = 0.;
		for (int32_t addrx = 0; addrx < m; addrx++) {
			TYPE input = devTmpD1[addry * ldt1 + addrx];
			if (reproflag == 0) devSplit[addry * lds + addrx + lds * n] = input;
			const TYPE split = (input + sigma) - sigma;
			devSplit[addry * lds + addrx] = split;
			input = input - split;
			TYPE tmpD1 = input;
			TYPE tmpD2 = devTmpD2[addry * ldt2 + addrx];
			TYPE tmpD3 = devTmpD3[addry * ldt3 + addrx];
			QuickTwoSum (tmpD1, tmpD2, tmpD1, tmpD2);
			QuickTwoSum (tmpD2, tmpD3, tmpD2, tmpD3);
			devTmpD1[addry * ldt1 + addrx] = tmpD1;
			devTmpD2[addry * ldt2 + addrx] = tmpD2;
			devTmpD3[addry * ldt3 + addrx] = tmpD3;
			max = std::max(max, fabs1(tmpD1));
		}
		devMax[addry] = max;
		#pragma omp critical (atomicadd)
		{
			check += max;
		}
	}
	return check;
}

template <typename TYPE>
TYPE ozblasSplitDevice3 (
	const char major,
	const int32_t m, 
	const int32_t n,
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
	int32_t splitEpsMode,
	const int32_t splitShift,
	const int32_t reproflag
) {
	const int32_t dim = ((major == 'r') && n > 1) ? n : m;
	const int32_t rho = getRho <TYPE, TYPE> (dim, splitEpsMode);
	if (major == 'c') {
		return ozblasSplitCKernel3 (m, n, rho, devTmpD1, ldt1, devTmpD2, ldt2, devTmpD3, ldt3, devSplit, lds, devSpExp, devMax, splitShift, reproflag);
	} else {
		fprintf (OUTPUT, "OzBLAS error: R-major is not supported in Split.\n");
		exit (1);
	}
	return 0;
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
	const int32_t ldt3
) {
	#pragma omp parallel for
	for (int32_t addry = 0; addry < n; addry++) {
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
	int32_t s;
	// FindMax^(0)
	ozblasFindMaxDevice (major, m, n, devTmpD1, ldt1, devMax);
	// Split^(0) & FindMax^(1)
	if (ozblasSplitDevice3 (major, m, n, devTmpD1, ldt1, devTmpD2, ldt2, devTmpD3, ldt3, devSplit, lds, devSpExp, devMax, oh->splitEpsMode, oh->splitShift, oh->reproMode) == 0.) return s;
	int32_t maxS = (oh->nSplitMax > 0) ? oh->nSplitMax : NumSplitDefaultMax;
	// Split^(s) & FindMax^(s+1)
	for (s = 1; s < maxS; s++) {
		if (ozblasSplitDevice3 (major, m, n, devTmpD1, ldt1, devTmpD2, ldt2, devTmpD3, ldt3, &devSplit[lds*n*s], lds, &devSpExp[ldse*s], devMax, oh->splitEpsMode, oh->splitShift, oh->reproMode) == 0.) break;
	}
	if (oh->reproMode == 0) {
		if (s == maxS && oh->nSplitMax > 0) 
			copyMat (m, n, &devSplit[lds*n*(s-1)], lds, &devSplit[lds*n*(s)], lds);
		if (oh->nSplitMax == 0) 
			copyMat (m, n, &devSplit[lds*n*(s)], lds, &devSplit[lds*n*(s+1)], lds);
		return s;
	}
	if (s == maxS && oh->nSplitMax > 0) 
		return s;
	s++;
	if (oh->splitMode == 1 || oh->splitMode == 31) fprintf (OUTPUT, "OzBLAS error: still splittable.\n");
	return s;
}
template int32_t ozblasSplit3 (ozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const __float128 *devInput, const int32_t ldi, double *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, double *devMax, double *devTmpD1, const int32_t ldt1, double *devTmpD2, const int32_t ldt2, double *devTmpD3, const int32_t ldt3);
template int32_t ozblasSplit3 (ozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const __float128 *devInput, const int32_t ldi, float *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, float *devMax, float *devTmpD1, const int32_t ldt1, float *devTmpD2, const int32_t ldt2, float *devTmpD3, const int32_t ldt3);
template int32_t ozblasSplit3 (ozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const double *devInput, const int32_t ldi, double *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, double *devMax, double *devTmpD1, const int32_t ldt1, double *devTmpD2, const int32_t ldt2, double *devTmpD3, const int32_t ldt3);
template int32_t ozblasSplit3 (ozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const float *devInput, const int32_t ldi, double *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, double *devMax, double *devTmpD1, const int32_t ldt1, double *devTmpD2, const int32_t ldt2, double *devTmpD3, const int32_t ldt3);
template int32_t ozblasSplit3 (ozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const double *devInput, const int32_t ldi, float *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, float *devMax, float *devTmpD1, const int32_t ldt1, float *devTmpD2, const int32_t ldt2, float *devTmpD3, const int32_t ldt3);
template int32_t ozblasSplit3 (ozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const float *devInput, const int32_t ldi, float *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, float *devMax, float *devTmpD1, const int32_t ldt1, float *devTmpD2, const int32_t ldt2, float *devTmpD3, const int32_t ldt3);

//=================================================
// determination of the num of split matrices
//=================================================
/*
template <typename TYPE>
void ozblasGenEVec (
	const int32_t n,
	TYPE *devA
) {
	#pragma omp parallel for
	for (int32_t addrx = 0; addrx < n; addrx++) 
		devA[addrx] = 1.;
}

template <typename TYPE>
int32_t ozblasCompVecAll (
	const int n,
	const TYPE *devA,
	const TYPE *devB
) {
	int32_t val = 0;
	#pragma omp parallel for 
	for (int32_t addrx = 0; addrx < n; addrx++) {
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
	#pragma omp parallel for 
	for (int32_t addry = 0; addry < n; addry++) {
		for (int32_t addrx = 0; addrx < m; addrx++) 
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
    const TYPE1 one = (TYPE1)1.;
	#pragma omp parallel for
	for (int32_t addrx = 0; addrx < m; addrx++) {
		TYPE1 max = 0.;
		const short tau = devSpExp[addrx] = ceil(log21(devMax[addrx]));
		const TYPE1 sigma = CONST * scalbn1 (one, rho + tau);
		for (int32_t j = 0; j < n; j++) {
			TYPE1 input = devInput[j * ldi + addrx];
			const TYPE1 split = (input + sigma) - sigma;
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
    const TYPE one = (TYPE)1.;
	#pragma omp parallel for
	for (int32_t addrx = 0; addrx < m; addrx++) {
		TYPE max = 0.;
		const short tau = ceil(log21(fabs1(devMax[addrx])));
		const TYPE sigma = CONST * scalbn1 (one, rho + tau);
		for (int32_t j = 0; j < n; j++) {
			TYPE input = devInput[j * ldi + addrx];
			const TYPE split = (input + sigma) - sigma;
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
    const TYPE1 one = (TYPE1)1.;
	#pragma omp parallel for
	for (int32_t addry = 0; addry < n; addry++) {
		TYPE1 max = 0.;
		const short tau = devSpExp[addry] = ceil(log21(devMax[addry]));
		const TYPE1 sigma = CONST * scalbn1 (one, rho + tau);
		for (int32_t i = 0; i < m; i++) {
			TYPE1 input = devInput[addry * ldi + i];
			TYPE1 const split = (input + sigma) - sigma;
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
    const TYPE one = (TYPE)1.;
	#pragma omp parallel for
	for (int32_t addry = 0; addry < n; addry++) {
		TYPE max = 0.;
		const short tau = ceil(log21(fabs1(devMax[addry])));
		const TYPE sigma = CONST * scalbn1 (one, rho + tau);
		for (int32_t i = 0; i < m; i++) {
			TYPE input = devInput[addry * ldi + i];
			const TYPE split = (input + sigma) - sigma;
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
	int32_t splitEpsMode
) {
	const int32_t dim = ((major == 'r') && n > 1) ? n : m;
	int32_t rho = getRho <TYPE1, TYPE2> (dim, splitEpsMode);
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
	ozblasSplitDevice (major, ma, ka, devAInput, ldai, devAOutput, ldao, &devASplit[0], ldas, devASpExp, devAMax, oh->splitEpsMode, 1);
	// Split^(i) & FindMax^(i+1) ---------------------------------------------------------------
	int32_t s;
	int32_t maxS = (oh->nSplitMax > 0) ? oh->nSplitMax : NumSplitDefaultMax;
	for (s = 1; s < maxS; s++) {
		TYPE1 check = (ka == 1 && major == 'c') ? devAMax[0] : blasRasum ((major=='r')?ma:ka, devAMax, 1);
		if (check == 0) {
			oh->fastMode = 0; // turn fastmode off
			fprintf (OUTPUT, "OzBLAS warning: fastMode was disabled. The result may not be accurate enough.\n");
			return s;
		}
		// Split^(i) & FindMax^(i+1)
		ozblasSplitDeviceA (major, ma, ka, devAOutput, ldao, devAOutput, ldao, &devASplit[ldas*ka*s], ldas, devAtmp, ldat, &devASpExp[ldase*s], devAMax, oh->splitEpsMode);
		// Determining # of splits -------------------------------------------------------------
		TYPE1 di = (TYPE1)(s+1);
		blasRgemv (Atran, ma, ka, di, devAtmp, ldat, devBe, 1, dzero, devB2, 1);	// devB2 = i * devASplit * devBe
		int32_t icheck = ozblasCompVecAll (ma, devB1, devB2);
		if (icheck == ma) return s;
		// ------------------------------------------------------------------------------------------
	}
	if (oh->splitMode == 1)
		fprintf (OUTPUT, "OzBLAS error: still splittable.\n");
	return s;
}
template int32_t ozblasSplitA (ozblasHandle_t *oh, const char major, const int32_t ma, const int32_t ka, const __float128 *devAInput, const int32_t ldai, const int32_t kb, const int32_t nb, const __float128 *devBInput, const int32_t ldbi, __float128 *devAOutput, const int32_t ldao, double *devASplit, const int32_t ldas, short *devASpExp, const int32_t ldase, __float128 *devAMax, __float128 *devAtmp, const int32_t ldat, __float128 *devBtmp, const int32_t ldbt, __float128 *devE, __float128 *devBe, __float128 *devB1, __float128 *devB2);
template int32_t ozblasSplitA (ozblasHandle_t *oh, const char major, const int32_t ma, const int32_t ka, const __float128 *devAInput, const int32_t ldai, const int32_t kb, const int32_t nb, const __float128 *devBInput, const int32_t ldbi, __float128 *devAOutput, const int32_t ldao, float *devASplit, const int32_t ldas, short *devASpExp, const int32_t ldase, __float128 *devAMax, __float128 *devAtmp, const int32_t ldat, __float128 *devBtmp, const int32_t ldbt, __float128 *devE, __float128 *devBe, __float128 *devB1, __float128 *devB2);
template int32_t ozblasSplitA (ozblasHandle_t *oh, const char major, const int32_t ma, const int32_t ka, const double *devAInput, const int32_t ldai, const int32_t kb, const int32_t nb, const double *devBInput, const int32_t ldbi, double *devAOutput, const int32_t ldao, double *devASplit, const int32_t ldas, short *devASpExp, const int32_t ldase, double *devAMax, double *devAtmp, const int32_t ldat, double *devBtmp, const int32_t ldbt, double *devE, double *devBe, double *devB1, double *devB2);
template int32_t ozblasSplitA (ozblasHandle_t *oh, const char major, const int32_t ma, const int32_t ka, const double *devAInput, const int32_t ldai, const int32_t kb, const int32_t nb, const double *devBInput, const int32_t ldbi, double *devAOutput, const int32_t ldao, float *devASplit, const int32_t ldas, short *devASpExp, const int32_t ldase, double *devAMax, double *devAtmp, const int32_t ldat, double *devBtmp, const int32_t ldbt, double *devE, double *devBe, double *devB1, double *devB2);
template int32_t ozblasSplitA (ozblasHandle_t *oh, const char major, const int32_t ma, const int32_t ka, const float *devAInput, const int32_t ldai, const int32_t kb, const int32_t nb, const float *devBInput, const int32_t ldbi, float *devAOutput, const int32_t ldao, float *devASplit, const int32_t ldas, short *devASpExp, const int32_t ldase, float *devAMax, float *devAtmp, const int32_t ldat, float *devBtmp, const int32_t ldbt, float *devE, float *devBe, float *devB1, float *devB2);
*/
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
	#pragma omp parallel for 
	for (int32_t addrx = 0; addrx < m; addrx++) {
		TYPE max = 0.;
		for (int32_t i = devRowptr[addrx]; i < devRowptr[addrx+1]; i++) 
			max = std::max(max, fabs1(devInput[i]));
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
TYPE ozblasSplitSparseNKernel (
	const int32_t m,
	const TYPE *devInput,
	const int32_t *devRowptr,
	TYPE *devOutput,
	TYPE *devSplit,
	const int32_t lds, 
	short *devSpExp,
	TYPE *devMax,
	const int32_t splitEpsMode,
	const int32_t splitShift,
	const int32_t reproflag
) {
	TYPE check = 0.;
	const TYPE one = (TYPE)1.;
	#pragma omp parallel for 
	for (int32_t addrx = 0; addrx < m; addrx++) {
		const int32_t dim = devRowptr[addrx+1] - devRowptr[addrx];
		const int32_t rho = getRho <TYPE, TYPE> (dim, splitEpsMode);
		const short tau = ceil(log21(devMax[addrx]));
		const TYPE sigma = CONST * scalbn1 (one, rho+tau) / splitShift;
		TYPE max = 0.;
		for (int32_t i = devRowptr[addrx]; i < devRowptr[addrx+1]; i++) {
			TYPE input = devInput[i];
			if (reproflag == 0) devSplit[m] = input;
			const TYPE split = (input + sigma) - sigma;
			input = input - split;
			devSplit[i] = split;
			devOutput[i] = input;
			max = std::max(max, fabs1(input));
		}
		devMax[addrx] = max;
		#pragma omp critical (atomicadd)
		{
			check += max;
		}
	}
	return check;
}

template <typename TYPE1, typename TYPE2>
TYPE1 ozblasSplitSparseNKernel (
	const int32_t m,
	const TYPE1 *devInput,
	const int32_t *devRowptr,
	TYPE1 *devOutput,
	TYPE2 *devSplit,
	const int32_t lds, 
	short *devSpExp,
	TYPE1 *devMax,
	const int32_t splitEpsMode,
	const int32_t splitShift,
	const int32_t reproflag
) {
	TYPE1 check = 0.;
	const TYPE1 one = (TYPE1)1.;
	#pragma omp parallel for 
	for (int32_t addrx = 0; addrx < m; addrx++) {
		const int32_t dim = devRowptr[addrx+1] - devRowptr[addrx];
		const int32_t rho = getRho <TYPE1, TYPE2> (dim, splitEpsMode);
		const short tau = devSpExp[addrx] = ceil(log21(devMax[addrx]));
		const TYPE1 sigma = CONST * scalbn1 (one, rho+tau) / splitShift;
		TYPE1 max = 0.;
		for (int32_t i = devRowptr[addrx]; i < devRowptr[addrx+1]; i++) {
			TYPE1 input = devInput[i];
			const TYPE1 split = (input + sigma) - sigma;
			input = input - split;
			devSplit[i] = scalbn1(split, -tau);
			devOutput[i] = input;
			max = std::max(max, fabs1(input));
		}
		devMax[addrx] = max;
		#pragma omp critical (atomicadd)
		{
			check += max;
		}
	}
	return check;
}

template <typename TYPE1, typename TYPE2>
TYPE1 ozblasSplitSparseDevice (
	const char major,
	const int32_t m, 
	const TYPE1 *devInput, 
	const int32_t *devRowptr,
	TYPE1 *devOutput,
	TYPE2 *devSplit, 
	const int32_t lds, 
	short *devSpExp, 
	TYPE1 *devMax,
	int32_t splitEpsMode,
	const int32_t splitShift,
	const int32_t reproflag
) {
	if (major == 'r') {
		return ozblasSplitSparseNKernel (m, devInput, devRowptr, devOutput, devSplit, lds, devSpExp, devMax, splitEpsMode, splitShift, reproflag);
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
	int32_t s;
	// FindMax^(0)
	ozblasFindMaxSparseDevice (major, m, devInput, devRowptr, devMax);
	// Split^(0) & FindMax^(1)
	if (ozblasSplitSparseDevice (major, m, devInput, devRowptr, devOutput, &devSplit[0], lds, devSpExp, devMax, oh->splitEpsMode, oh->splitShift, oh->reproMode) == 0.) return s;
	int32_t maxS = (oh->nSplitMax > 0) ? oh->nSplitMax : NumSplitDefaultMax;
	// Split^(s) & FindMax^(s+1)
	for (s = 1; s < maxS; s++) {
		if (ozblasSplitSparseDevice (major, m, devOutput, devRowptr, devOutput, &devSplit[lds*s], lds, &devSpExp[ldse*s], devMax, oh->splitEpsMode, oh->splitShift, oh->reproMode) == 0.) break;
	}
	if constexpr (std::is_same<TYPE1, TYPE2>::value) {
		if (oh->reproMode == 0) {
			if (s == maxS && oh->nSplitMax > 0) 
				copyVec (lds, &devSplit[lds*(s-1)], &devSplit[lds*(s)]);
			if (oh->nSplitMax == 0) 
				copyVec (lds, &devSplit[lds*(s)], &devSplit[lds*(s+1)]);
			return s;
		}
		if (s == maxS && oh->nSplitMax > 0) 
			return s;
	}
	s++;
	if (oh->splitMode == 1) fprintf (OUTPUT, "OzBLAS error: still splittable.\n");
	return s;
}
template int32_t ozblasSplitSparse <__float128, double> (ozblasHandle_t *oh, const char major, const int32_t m, const __float128 *devInput, const int32_t *devRowptr, __float128 *devOutput, double *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, __float128 *devMax);
template int32_t ozblasSplitSparse <__float128, float> (ozblasHandle_t *oh, const char major, const int32_t m, const __float128 *devInput, const int32_t *devRowptr, __float128 *devOutput, float *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, __float128 *devMax);
template int32_t ozblasSplitSparse <double, double> (ozblasHandle_t *oh, const char major, const int32_t m, const double *devInput, const int32_t *devRowptr, double *devOutput, double *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, double *devMax);
template int32_t ozblasSplitSparse <double, float> (ozblasHandle_t *oh, const char major, const int32_t m, const double *devInput, const int32_t *devRowptr, double *devOutput, float *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, double *devMax);
template int32_t ozblasSplitSparse <float, float> (ozblasHandle_t *oh, const char major, const int32_t m, const float *devInput, const int32_t *devRowptr, float *devOutput, float *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, float *devMax);
template int32_t ozblasSplitSparse <float, double> (ozblasHandle_t *oh, const char major, const int32_t m, const float *devInput, const int32_t *devRowptr, float *devOutput, double *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, float *devMax);
