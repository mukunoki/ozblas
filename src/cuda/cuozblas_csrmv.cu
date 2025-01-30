#include "cuozblas_common.h"

template <typename TYPE1, typename TYPE2>
int32_t cuozblasRcsrmv (
	cuozblasHandle_t *oh,
	const char tranA, 
	const int32_t m,
	const int32_t n,
	const int32_t nnz,
	const TYPE1 alpha,
	const cusparseMatDescr_t descrA, 
	const TYPE1 *devA,
	const int32_t *devAcolind,
	const int32_t *devArowptr,
	const TYPE1 *devB,
	const TYPE1 beta,
	TYPE1 *devC
) {
	if (oh->reproModeFlag == 0 && oh->nSplitMax == 1) {
		printf ("here\n");
		double t0 = cutimer();
		if (oh->precxFlag == 1) 
			blasRcsrmvX (oh->csh, tranA, m, n, nnz, alpha, descrA, devA, devAcolind, devArowptr, devB, beta, devC, 0);
		else
			blasRcsrmv (oh->csh, tranA, m, n, nnz, alpha, descrA, devA, devAcolind, devArowptr, devB, beta, devC);
		oh->t_SpMV_SpMM_total += cutimer() - t0;
		return 0;
	}
	if (tranA == 't' || tranA == 'T') {
		fprintf (OUTPUT, "error: transposed mode is not implemented.\n");
		exit (1);
	}

	cucounterInit (oh);
	double t1, t0 = cutimer();
	short *devASpExp, *devBSpExp;
	TYPE2 fone = 1., fzero = 0.;
	TYPE2 *devASplit, *devBSplit, *devCSplit;
	TYPE1 *devATmp, *devBTmp;
	TYPE1 *devAmax, *devBmax;
	int32_t ldas, ldbs, ldcs, ldase;
	int32_t sizeType1 = sizeof (TYPE1);
	int32_t sizeType2 = sizeof (TYPE2);
	int32_t sizeTypeS = sizeof (short);
	int32_t nSplitMaxLoc = (oh->nSplitMax > 0) ? oh->nSplitMax : NumSplitDefaultMax;

	if (oh->memMaskSplitA != 0) oh->memAddr = 0;
	// --- here is preserved ---
	cuozblasMatAddrAlloc (oh, nnz, nSplitMaxLoc, sizeType2, (void**)&devASplit, ldas);
	cuozblasVecAddrAlloc (oh, nnz, sizeType1, (void**)&devATmp);
	cuozblasVecAddrAlloc (oh, m,   sizeType1, (void**)&devAmax);
	cuozblasMatAddrAlloc (oh, m, nSplitMaxLoc, sizeTypeS, (void**)&devASpExp, ldase);
	// --- here is preserved ---
	if (oh->memMaskSplitA != 0) oh->memMaskSplitA = oh->memAddr;

	cuozblasMatAddrAlloc (oh, n, nSplitMaxLoc, sizeType2, (void**)&devBSplit, ldbs);
	cuozblasMatAddrAlloc (oh, m, nSplitMaxLoc * nSplitMaxLoc * ((oh->splitEpsModeFlag == 2)?2:1), sizeType2, (void**)&devCSplit, ldcs);
	cuozblasVecAddrAlloc (oh, n, sizeType1, (void**)&devBTmp);
	// Exp
	cuozblasVecAddrAlloc (oh, nSplitMaxLoc, sizeTypeS, (void**)&devBSpExp);
	// Splitting
	cuozblasVecAddrAlloc (oh, 1, sizeType1, (void**)&devBmax);
	
	if (cumemCheck (oh)) {
		fprintf (OUTPUT, "OzBLAS error: memory shortage.\n");
		exit (1);
	}

	// Split of A -----------------------------------
	t1 = cutimer();
	int32_t nSplitA;
	if (oh->memMaskSplitA == 0) {
		nSplitA = cuozblasSplitSparse (oh, 'r', m, devA, devArowptr, devATmp, devASplit, ldas, devASpExp, ldase, devAmax);
	} else {
		devASplit = (TYPE2*)devA;
		nSplitA = oh->nSplitA_;
	}
	oh->t_SplitA += cutimer() - t1;

	// Split of B -----------------------------------
	t1 = cutimer();
	int32_t nSplitB = cuozblasSplit (oh, 'c', n, 1, devB, n, devBTmp, n, devBSplit, ldbs, devBSpExp, 1, devBmax);
	oh->t_SplitB += cutimer() - t1;

	// Compute --------------------------------------
	t1 = cutimer();
	int32_t ia, ic;
	int32_t maxlevel = (nSplitA-1) + (nSplitB-1);
	TYPE2 *ptrB = devBSplit;
	TYPE2 *ptrC = devCSplit;
	ic = 0;
	for (ia = 0; ia < MIN (maxlevel+1, nSplitA); ia++) {
		const int32_t numB = MIN (nSplitB, maxlevel+1 - ia);
		TYPE2 *ptrA = devASplit+ldas*ia;
		if (oh->splitEpsModeFlag == 2) {
			blasRcsrmm_x2 (oh->csh, tranA, m, numB, n, nnz, fone, descrA, ptrA, devAcolind, devArowptr, ptrB, ldbs, fzero, ptrC, ldcs);
			ptrC += ldcs*numB*2;
		} else {
			blasRcsrmm (oh->csh, tranA, m, numB, n, nnz, fone, descrA, ptrA, devAcolind, devArowptr, ptrB, ldbs, fzero, ptrC, ldcs);
			ptrC += ldcs*numB;
		}
		ic += numB;
	}
	const int32_t nSplitC = ic;
	oh->nSplitA += nSplitA;
	oh->nSplitB += nSplitB;
	oh->nSplitC += nSplitC;
	oh->t_comp += cutimer() - t1;
	
	// Sum -----------------------------------------
	t1 = cutimer();
	ic = 0;
	if (oh->splitEpsModeFlag == 2) maxlevel = (nSplitA-1) + (nSplitB*2-1);
	if (cuozblasGlobalSum (oh, m, 1, 1, devASpExp, ldase, nSplitA,
						devBSpExp, 1, nSplitB*((oh->splitEpsModeFlag == 2)?2:1), devCSplit, ldcs, ldcs, devC, 1, alpha, beta, maxlevel, 3)) {
		fprintf (OUTPUT, "OzBLAS error: sum is failed\n");
		exit (1);
	}
	oh->t_sum += cutimer() - t1;
	oh->t_total = cutimer() - t0;

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
template int32_t cuozblasRcsrmv <double, double> (cuozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const double alpha, const cusparseMatDescr_t descrA, const double *devA, const int32_t *devAcolind, const int32_t *devArowptr, const double *devB, const double beta, double *devC);
template int32_t cuozblasRcsrmv <double, float> (cuozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const double alpha, const cusparseMatDescr_t descrA, const double *devA, const int32_t *devAcolind, const int32_t *devArowptr, const double *devB, const double beta, double *devC);
template int32_t cuozblasRcsrmv <float, float> (cuozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const float alpha, const cusparseMatDescr_t descrA, const float *devA, const int32_t *devAcolind, const int32_t *devArowptr, const float *devB, const float beta, float *devC);
template int32_t cuozblasRcsrmv <float, double> (cuozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const float alpha, const cusparseMatDescr_t descrA, const float *devA, const int32_t *devAcolind, const int32_t *devArowptr, const float *devB, const float beta, float *devC);


// splitting only (for CG solvers)
template <typename TYPE1, typename TYPE2>
TYPE2 *cuozblasRcsrmvSplitA (
	cuozblasHandle_t *oh,
	const char tranA, 
	const int32_t m,
	const int32_t n,
	const int32_t nnz,
	const cusparseMatDescr_t descrA, 
	const TYPE1 *devA,
	const int32_t *devArowptr
) {
	if (oh->reproModeFlag == 0 && oh->nSplitMax == 1) {
		return (TYPE2*)devA;
	}
	if (tranA == 't' || tranA == 'T') {
		fprintf (OUTPUT, "OzBLAS error: transposed mode is not implemented.\n");
		exit (1);
	}

	cucounterInit (oh);
	short *devASpExp;
	TYPE1 *devAmax, *devATmp;
	TYPE2 *devASplit;
	int32_t sizeType1 = sizeof (TYPE1);
	int32_t sizeType2 = sizeof (TYPE2);
	int32_t ldas, ldase;
	int32_t nSplitMaxLoc = (oh->nSplitMax > 0) ? oh->nSplitMax : NumSplitDefaultMax;
	// --- here is preserved ---
	cuozblasMatAddrAlloc (oh, nnz, nSplitMaxLoc, sizeType2, (void**)&devASplit, ldas);
	cuozblasVecAddrAlloc (oh, nnz, sizeType1, (void**)&devATmp);
	cuozblasVecAddrAlloc (oh, m, sizeType1, (void**)&devAmax);
	cuozblasMatAddrAlloc (oh, m, nSplitMaxLoc, sizeof(short), (void**)&devASpExp, ldase);
	// --- here is preserved ---
	if (cumemCheck (oh)) {
		fprintf (OUTPUT, "OzBLAS error: memory shortage.\n");
		exit (1);
	}
	oh->memMaskSplitA = oh->memAddr;

	double t1 = cutimer();
	// Split of A -----------------------------------
	int32_t nSplitAlocal = cuozblasSplitSparse (oh, 'r', m, devA, devArowptr, devATmp, devASplit, ldas, devASpExp, ldase, devAmax);
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
			nSplitAlocal = cuozblasSplitSparse (oh, 'r', m, devA, devArowptr, devATmp, devASplit, ldas, devASpExp, ldase, devAmax);
			//printf ("... try splitShift = %d -> nSplitA = %d\n", oh->splitShift, nSplitAlocal);
		} while (nSplitAOld == nSplitAlocal && oh->splitShift < 512); // 512 (9bit) is max
		if (nSplitAOld == nSplitAlocal) oh->splitShift = 1;
		// do again with the optimal shift-size
		nSplitAlocal = cuozblasSplitSparse (oh, 'r', m, devA, devArowptr, devATmp, devASplit, ldas, devASpExp, ldase, devAmax);
        if (oh->verbose)
		    printf ("\n## splitShift = %d (%d-bit), nSplitA = %d -> %d\n", oh->splitShift, (int)log2((double)oh->splitShift), nSplitAlocalOld, nSplitAlocal);
	}
	// ------------------------------------------------------------------------
	oh->nSplitA_ = oh->nSplitA = nSplitAlocal;
	oh->t_SplitA += cutimer() - t1;

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
template double *cuozblasRcsrmvSplitA <double, double> (cuozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const cusparseMatDescr_t descrA, const double *devA, const int32_t *devArowptr);
template float *cuozblasRcsrmvSplitA <double, float> (cuozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const cusparseMatDescr_t descrA, const double *devA, const int32_t *devArowptr);
template float *cuozblasRcsrmvSplitA <float, float> (cuozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const cusparseMatDescr_t descrA, const float *devA, const int32_t *devArowptr);
template double *cuozblasRcsrmvSplitA <float, double> (cuozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const cusparseMatDescr_t descrA, const float *devA, const int32_t *devArowptr);
