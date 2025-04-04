#include "ozblas_common.h"

template <typename TYPE1, typename TYPE2>
int32_t ozblasRcsrmv (
	ozblasHandle_t *oh,
	const char tranA, 
	const int32_t m,
	const int32_t n,
	const int32_t nnz,
	const TYPE1 alpha,
	const char *descrA, 
	const TYPE1 *devA,
	const int32_t *devAcolind,
	const int32_t *devArowptr,
	const TYPE1 *devB,
	const TYPE1 beta,
	TYPE1 *devC
) {
	if (oh->reproMode == 0 && oh->nSplitMax == 1) {
		double t0 = timer();
		if (oh->precxFlag == 1) 
			blasRcsrmvX (tranA, m, n, nnz, alpha, descrA, devA, devAcolind, devArowptr, devB, beta, devC);
		else
			blasRcsrmv (tranA, m, n, nnz, alpha, descrA, devA, devAcolind, devArowptr, devB, beta, devC);
		oh->t_SpMV_SpMM_total += timer() - t0;
		return 0;
	}
	if (tranA == 't' || tranA == 'T') {
		fprintf (OUTPUT, "error: transposed mode is not implemented.\n");
		exit (1);
	}
	counterInit (oh);
	double t1, t0 = timer();
	short *devASpExp, *devBSpExp;
	TYPE2 fone = 1., fzero = 0.;
	TYPE2 *devASplit, *devBSplit, *devCSplit;
	TYPE1 *devATmp, *devBTmp, *devCTmp;
	TYPE1 *devAmax_, *devBmax_;
	TYPE2 *devBmax, *devBTmpD1, *devBTmpD2, *devBTmpD3;
	// for quadruple, use double
	double *devCTmp1, *devCTmp2, *devCTmp3;
	int32_t sizeTypeT = sizeof (double);
	int32_t ldas, ldbs, ldcs, ldase;
	int32_t sizeType1 = sizeof (TYPE1);
	int32_t sizeType2 = sizeof (TYPE2);
	int32_t sizeTypeS = sizeof (short);
	int32_t nSplitMaxLoc = (oh->nSplitMax > 0) ? oh->nSplitMax : NumSplitDefaultMax;
    if (oh->reproMode == 1) nSplitMaxLoc++;

	if (oh->memMaskSplitA != 0) oh->memAddr = 0;
	// --- here is preserved ---
	ozblasMatAddrAlloc (oh, nnz, nSplitMaxLoc, sizeType2, (void**)&devASplit, ldas);
	ozblasVecAddrAlloc (oh, nnz, sizeType1, (void**)&devATmp);
	ozblasVecAddrAlloc (oh, m,   sizeType1, (void**)&devAmax_);
	ozblasMatAddrAlloc (oh, m, nSplitMaxLoc, sizeTypeS, (void**)&devASpExp, ldase);
	// --- here is preserved ---
	if (oh->memMaskSplitA != 0) oh->memMaskSplitA = oh->memAddr;

	ozblasMatAddrAlloc (oh, n, nSplitMaxLoc, sizeType2, (void**)&devBSplit, ldbs);
	ozblasMatAddrAlloc (oh, m, nSplitMaxLoc * nSplitMaxLoc * ((oh->splitEpsMode == 2)?2:1), sizeType2, (void**)&devCSplit, ldcs);
	ozblasVecAddrAlloc (oh, n, sizeType1, (void**)&devBTmp);
	ozblasVecAddrAlloc (oh, m, sizeType1, (void**)&devCTmp);
	if (oh->sumMode == 3) {
		ozblasVecAddrAlloc (oh, m, sizeTypeT, (void**)&devCTmp1);
		ozblasVecAddrAlloc (oh, m, sizeTypeT, (void**)&devCTmp2);
		ozblasVecAddrAlloc (oh, m, sizeTypeT, (void**)&devCTmp3);
	}
	// Exp
	ozblasVecAddrAlloc (oh, nSplitMaxLoc, sizeTypeS, (void**)&devBSpExp);
	// Splitting
	ozblasVecAddrAlloc (oh, 1, sizeType1, (void**)&devBmax_);
	// above must be allocated even if splitMode is 3 as they may be used if Split3 is not used
	if (oh->splitMode == 3) {
		// Currently, split3 is only for B
		ozblasVecAddrAlloc (oh, 1, sizeType2, (void**)&devBmax);
		ozblasVecAddrAlloc (oh, n, sizeType2, (void**)&devBTmpD1);
		ozblasVecAddrAlloc (oh, n, sizeType2, (void**)&devBTmpD2);
		ozblasVecAddrAlloc (oh, n, sizeType2, (void**)&devBTmpD3);
	}
	
	if (memCheck (oh)) {
		fprintf (OUTPUT, "OzBLAS error: memory shortage.\n");
		exit (1);
	}

	// Split of A -----------------------------------
	t1 = timer();
	if (oh->splitMode == 3) {
		fprintf (OUTPUT, "OzBLAS warning: split3 for sparse matrix is not implemented.\n");
		exit (1);
	}
	int32_t nSplitA;
	if (oh->memMaskSplitA == 0) {
		nSplitA = ozblasSplitSparse (oh, 'r', m, devA, devArowptr, devATmp, devASplit, ldas, devASpExp, ldase, devAmax_);
	} else {
		devASplit = (TYPE2*)devA;
		nSplitA = oh->nSplitA_;
	}
	oh->t_SplitA += timer() - t1;

	// Split of B -----------------------------------
	t1 = timer();
//	int32_t split3FlagB = (oh->splitMode == 3) ? rangeCheck <TYPE1, TYPE2> (n, 1, devB, n) : 0; // on (if 1)
	int32_t nSplitB;
//	if (split3FlagB) 
	if (oh->splitMode == 3) 
		nSplitB = ozblasSplit3 (oh, 'c', n, 1, devB, n, devBSplit, ldbs, devBSpExp, 1, devBmax,
								devBTmpD1, n, devBTmpD2, n, devBTmpD3, n);
	else
		nSplitB = ozblasSplit (oh, 'c', n, 1, devB, n, devBTmp, n, devBSplit, ldbs, devBSpExp, 1, devBmax_);
	oh->t_SplitB += timer() - t1;

	// Compute --------------------------------------
	t1 = timer();
	int32_t ia, ib, ic, ik;
    int32_t maxlevel = std::max ((nSplitA + nSplitB) - oh->fastMode, std::min (nSplitA, nSplitB));
	TYPE2 *ptrB = devBSplit;
	TYPE2 *ptrC = devCSplit;
	ic = 0;
	for (ia = 0; ia < std::min (maxlevel+1, nSplitA); ia++) {
		const int32_t numB = std::min (nSplitB, maxlevel+1 - ia);
		TYPE2 *ptrA = devASplit+ldas*ia;
		if (oh->splitEpsMode == 2) {
			blasRcsrmm_x2 (tranA, m, numB, n, nnz, fone, descrA, ptrA, devAcolind, devArowptr, ptrB, ldbs, fzero, ptrC, ldcs);
			ptrC += ldcs*numB*2;
		} else {
			blasRcsrmm (tranA, m, numB, n, nnz, fone, descrA, ptrA, devAcolind, devArowptr, ptrB, ldbs, fzero, ptrC, ldcs);
			ptrC += ldcs*numB;
		}
		ic += numB;
	}
	const int32_t nSplitC = ic;
	oh->nSplitA += nSplitA;
	oh->nSplitB += nSplitB;
	oh->nSplitC += nSplitC;
	oh->t_comp += timer() - t1;
	
	// Sum -----------------------------------------
	t1 = timer();
	ic = 0;
	if (oh->sumMode == 3) {
		for (ik = 0; ik <= maxlevel; ik++) {
			for (ia = 0; ia < nSplitA; ia++) {
				for (ib = 0; ib < nSplitB; ib++) {
					if (ik == ia + ib) {
						int32_t it = nSplitB * ia + ib; // unlike GEMV, here is transposed
						if (ozblasLocalFsum3 (m, 1, &devASpExp[ldase*ia], &devBSpExp[ib], devCSplit+ldcs*it, ldcs,
											devCTmp, m, devCTmp1, m, devCTmp2, m, devCTmp3, m, (ic==nSplitC-1)?-1:ic, 0, 0)) {
							fprintf (OUTPUT, "OzBLAS error: Sum3 is failed.\n");
							exit (1);
						}
						ic++;
					} // EndIf (ik)
				} // EndFor (ib)
			} // EndFor (ia)
		} // EndFor (ik)
		ozblasAxpby (m, 1, devCTmp, 1, devC, 1, alpha, beta);
	} else { // sumMode < 3
		if (oh->splitEpsMode == 2) maxlevel = nSplitA + nSplitB*2;
		if (ozblasGlobalSum (oh, m, 1, devASpExp, ldase, nSplitA,
							devBSpExp, 1, nSplitB*((oh->splitEpsMode == 2)?2:1), devCSplit, ldcs, ldcs, devC, 1, alpha, beta, maxlevel, 3, 0, 0)) {
			fprintf (OUTPUT, "OzBLAS error: sum is failed\n");
			exit (1);
		}
	}
	oh->t_sum += timer() - t1;
	oh->t_total = timer() - t0;

	// for CG, time
	// =================================
	oh->t_SplitMat_total += oh->t_SplitA;
	oh->t_SplitVec_total += oh->t_SplitB;
	oh->t_Sum_total += oh->t_sum;
	oh->t_AXPY_SCAL_total += 0.;
	oh->t_DOT_NRM2_total += 0.;
	oh->t_SpMV_SpMM_total += oh->t_comp;
	// =================================

	return 0;
}
template int32_t ozblasRcsrmv <__float128, double> (ozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const __float128 alpha, const char *descrA, const __float128 *devA, const int32_t *devAcolind, const int32_t *devArowptr, const __float128 *devB, const __float128 beta, __float128 *devC);
template int32_t ozblasRcsrmv <__float128, float> (ozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const __float128 alpha, const char *descrA, const __float128 *devA, const int32_t *devAcolind, const int32_t *devArowptr, const __float128 *devB, const __float128 beta, __float128 *devC);
template int32_t ozblasRcsrmv <double, double> (ozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const double alpha, const char *descrA, const double *devA, const int32_t *devAcolind, const int32_t *devArowptr, const double *devB, const double beta, double *devC);
template int32_t ozblasRcsrmv <double, float> (ozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const double alpha, const char *descrA, const double *devA, const int32_t *devAcolind, const int32_t *devArowptr, const double *devB, const double beta, double *devC);
template int32_t ozblasRcsrmv <float, float> (ozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const float alpha, const char *descrA, const float *devA, const int32_t *devAcolind, const int32_t *devArowptr, const float *devB, const float beta, float *devC);
template int32_t ozblasRcsrmv <float, double> (ozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const float alpha, const char *descrA, const float *devA, const int32_t *devAcolind, const int32_t *devArowptr, const float *devB, const float beta, float *devC);


// splitting only (for CG solvers)
template <typename TYPE1, typename TYPE2>
TYPE2 *ozblasRcsrmvSplitA (
	ozblasHandle_t *oh,
	const char tranA, 
	const int32_t m,
	const int32_t n,
	const int32_t nnz,
	const char *descrA, 
	const TYPE1 *devA,
	const int32_t *devArowptr
) {
	if (oh->reproMode == 0 && oh->nSplitMax == 1) {
		return (TYPE2*)devA;
	}
	if (tranA == 't' || tranA == 'T') {
		fprintf (OUTPUT, "OzBLAS error: transposed mode is not implemented.\n");
		exit (1);
	}
	counterInit (oh);
	short *devASpExp;
	TYPE1 *devAmax, *devATmp;
	TYPE2 *devASplit;
	int32_t sizeType1 = sizeof (TYPE1);
	int32_t sizeType2 = sizeof (TYPE2);
	int32_t ldas, ldase;
	int32_t nSplitMaxLoc = (oh->nSplitMax > 0) ? oh->nSplitMax : NumSplitDefaultMax;
    if (oh->reproMode == 1) nSplitMaxLoc++;
	// --- here is preserved ---
	ozblasMatAddrAlloc (oh, nnz, nSplitMaxLoc, sizeType2, (void**)&devASplit, ldas);
	ozblasVecAddrAlloc (oh, nnz, sizeType1, (void**)&devATmp);
	ozblasVecAddrAlloc (oh, m, sizeType1, (void**)&devAmax);
	ozblasMatAddrAlloc (oh, m, nSplitMaxLoc, sizeof(short), (void**)&devASpExp, ldase);
	// --- here is preserved ---
	if (memCheck (oh)) {
		fprintf (OUTPUT, "OzBLAS error: memory shortage.\n");
		exit (1);
	}
	oh->memMaskSplitA = oh->memAddr;

	double t1 = timer();
	// Split of A -----------------------------------
	if (oh->splitMode == 3) {
		fprintf (OUTPUT, "OzBLAS error: split3 for sparse matrix is not implemented.\n");
		exit (1);
	}
	int32_t nSplitAlocal = ozblasSplitSparse (oh, 'r', m, devA, devArowptr, devATmp, devASplit, ldas, devASpExp, ldase, devAmax);
	int32_t nSplitAlocalOld = nSplitAlocal;
	// shiftSize-tuning
	// ------------------------------------------------------------------------
	if (oh->nSplitMax == 0) { // tuning is possible only when full-splitting (d=0)
		//printf ("## CSRMV: << shift-size tuning >> num.split = %d\n", nSplitAlocal);
		int32_t nSplitAOld;
		oh->splitShift = 1; 
		do {
			nSplitAOld = nSplitAlocal;
			oh->splitShift *= 2;
			nSplitAlocal = ozblasSplitSparse (oh, 'r', m, devA, devArowptr, devATmp, devASplit, ldas, devASpExp, ldase, devAmax);
			//printf ("... try splitShift = %d -> nSplitA = %d\n", oh->splitShift, nSplitAlocal);
		} while (nSplitAOld == nSplitAlocal && oh->splitShift < 512); // 512 (9bit) is max
		if (nSplitAOld == nSplitAlocal) oh->splitShift = 1;
		// do again with the optimal shift-size
		nSplitAlocal = ozblasSplitSparse (oh, 'r', m, devA, devArowptr, devATmp, devASplit, ldas, devASpExp, ldase, devAmax);
        if (oh->verbose)
    		printf ("\n## splitShift = %d (%d-bit), nSplitA = %d -> %d\n", oh->splitShift, (int)log2((double)oh->splitShift), nSplitAlocalOld, nSplitAlocal);
	}
	// ------------------------------------------------------------------------
	oh->nSplitA_ = oh->nSplitA = nSplitAlocal;
	oh->t_SplitA += timer() - t1;

	// for CG, time
	// =================================
	oh->t_SplitMat_total += oh->t_SplitA;
	oh->t_SplitVec_total += 0.;
	oh->t_Sum_total += 0.;
	oh->t_AXPY_SCAL_total += 0.;
	oh->t_DOT_NRM2_total += 0.;
	oh->t_SpMV_SpMM_total += 0.;
	// =================================

	return devASplit;
}
template double *ozblasRcsrmvSplitA <__float128, double> (ozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const char *descrA, const __float128 *devA, const int32_t *devArowptr);
template float *ozblasRcsrmvSplitA <__float128, float> (ozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const char *descrA, const __float128 *devA, const int32_t *devArowptr);
template double *ozblasRcsrmvSplitA <double, double> (ozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const char *descrA, const double *devA, const int32_t *devArowptr);
template float *ozblasRcsrmvSplitA <double, float> (ozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const char *descrA, const double *devA, const int32_t *devArowptr);
template float *ozblasRcsrmvSplitA <float, float> (ozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const char *descrA, const float *devA, const int32_t *devArowptr);
template double *ozblasRcsrmvSplitA <float, double> (ozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const char *descrA, const float *devA, const int32_t *devArowptr);
