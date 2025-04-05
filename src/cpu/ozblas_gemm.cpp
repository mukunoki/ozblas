#include "ozblas_common.h"

template <typename TYPE1, typename TYPE2, typename TYPE3>
int32_t ozblasRgemm (
	ozblasHandle_t *oh,	
	const char transA, const char transB,
	const int32_t m, const int32_t n, const int32_t k,
	const TYPE1 alpha,
	const TYPE1 *devA, const int32_t lda,
	const TYPE1 *devB, const int32_t ldb,
	const TYPE1 beta,
	TYPE1 *devC, const int32_t ldc
) {
	if (oh->reproMode == 0 && oh->nSplitMax == 1) {
		blasRgemm (transA, transB, m, n, k, alpha, devA, lda, devB, ldb, beta, devC, ldc);
		return 0;
	}
	counterInit (oh);
	double t1, t0 = timer();

	// DOT does not use batchedGEMM and fastmode as it is computed by a GEMM,
	// Therefore, batchedGEMM and fastmode are disabled here.
	int32_t _useBatchedGemmFlag = oh->useBatchedGemmFlag;
	int32_t _fastMode = oh->fastMode;
	if (m == 1 && n == 1) { // DOT
		oh->useBatchedGemmFlag = 0;
		oh->fastMode = 0;
	}
	
	TYPE1 *devTmp1, *devCTmp, *devMax1;
	TYPE2 *devMax2, *devTmp21, *devTmp22, *devTmp23;
	TYPE2 *devASplit, *devBSplit, *devCSplit;
	TYPE2 fone = 1., fzero = 0.;
	short *devASpExp, *devBSpExp;
	// for QSGEMM, double is used, not TYPE2
	double *devCTmp1, *devCTmp2, *devCTmp3;
	int32_t	sizeTypeT = sizeof(double);
	// --
	int32_t ldas, ldbs, ldcs, ldase, ldbse, ldt, ldct;
	int32_t ldct1, ldct2, ldct3;
	int32_t mbk = m;
	int32_t nbk = n;
	int32_t nSplitMaxLoc = ((oh->nSplitMax > 0) ? oh->nSplitMax + 1 : NumSplitDefaultMax);
	if (oh->reproMode == 0)
		nSplitMaxLoc = std::max(oh->nSplitMax + 1, NumSplitDefaultMax);
	int32_t sizeType1 = sizeof (TYPE1);
	int32_t sizeType2 = sizeof (TYPE2);
	int32_t sizeTypeS = sizeof (short);

    int32_t globalSum = (oh->useBatchedGemmFlag == 1 || oh->sumMode == 0 || oh->sumMode == 1 || oh->sumMode == 30) ? 1 : 0;

	// Memory allocation 
	TYPE2 **batchAptr, **batchBptr, **batchCptr;
	if (oh->useBatchedGemmFlag) {
		ozblasVecAddrAlloc (oh, nSplitMaxLoc * nSplitMaxLoc, sizeof(TYPE2*), (void**)&batchAptr);
		ozblasVecAddrAlloc (oh, nSplitMaxLoc * nSplitMaxLoc, sizeof(TYPE2*), (void**)&batchBptr);
		ozblasVecAddrAlloc (oh, nSplitMaxLoc * nSplitMaxLoc, sizeof(TYPE2*), (void**)&batchCptr);
	}
	int64_t memAddrTmp = oh->memAddr;
	while (mbk > 0 && nbk > 0) { // blocking
		int32_t sizeCn = globalSum ? (nbk * nSplitMaxLoc * nSplitMaxLoc) : nbk;
		if (oh->splitEpsMode == 2) sizeCn *= 2;
		ozblasMatAddrAlloc (oh, k, mbk * nSplitMaxLoc, sizeType2, (void**)&devASplit, ldas); // Note: A is transposed!! o ldas is k-based
		ozblasMatAddrAlloc (oh, k, nbk * nSplitMaxLoc, sizeType2, (void**)&devBSplit, ldbs);
		ozblasMatAddrAlloc (oh, mbk, sizeCn,           sizeType2, (void**)&devCSplit, ldcs);
		ozblasMatAddrAlloc (oh, k, std::max(mbk,nbk),  sizeType1, (void**)&devTmp1,   ldt); // TRANSPOSE
		ozblasMatAddrAlloc (oh, mbk, nbk,              sizeType1, (void**)&devCTmp,   ldct);
		if (oh->sumMode == 3 && oh->useBatchedGemmFlag == 0) {
			ozblasMatAddrAlloc (oh, mbk, nbk, sizeTypeT, (void**)&devCTmp1, ldct1);
			ozblasMatAddrAlloc (oh, mbk, nbk, sizeTypeT, (void**)&devCTmp2, ldct2);
			ozblasMatAddrAlloc (oh, mbk, nbk, sizeTypeT, (void**)&devCTmp3, ldct3);
		}
		// Exp
		ozblasMatAddrAlloc (oh, mbk, nSplitMaxLoc, sizeTypeS, (void**)&devASpExp, ldase);
		ozblasMatAddrAlloc (oh, nbk, nSplitMaxLoc, sizeTypeS, (void**)&devBSpExp, ldbse);
		// Splitting
		ozblasVecAddrAlloc (oh, std::max(k, std::max(mbk,nbk)), sizeType1, (void**)&devMax1);
		// above must be allocated even if splitMode is 3 as they may be used if Split3 is not used
		if (oh->splitMode == 3) {
			ozblasVecAddrAlloc (oh, std::max(k, std::max(mbk,nbk)), sizeType2, (void**)&devMax2);
			ozblasMatAddrAlloc (oh, k, std::max(mbk,nbk), sizeType2, (void**)&devTmp21, ldt); 
			ozblasMatAddrAlloc (oh, k, std::max(mbk,nbk), sizeType2, (void**)&devTmp22, ldt); 
			ozblasMatAddrAlloc (oh, k, std::max(mbk,nbk), sizeType2, (void**)&devTmp23, ldt); 
		} 
		ldt = getPitchSize (k);
		if (!memCheck (oh)) break; // check if work-memory is enough or not
		oh->memAddr = memAddrTmp;
		if (mbk == 1 && nbk == 1 && !memCheck (oh)) {
			fprintf (OUTPUT, "OzBLAS error: out of memory\n");
			exit (1);
		}
		if (mbk != 1 || nbk != 1) {
			if (mbk >= nbk) 
				mbk = ceil (mbk / 2.);
			else 
				nbk = ceil (nbk / 2.);
		}
	}
	oh->mbk = mbk;
	oh->nbk = nbk;

	// main part (Split, Comp, Sum)
	char transA_ = transA;
	char transB_ = transB;
	int32_t block_count = 0;
	for (int32_t im = 0; im < ceil((float)m/mbk); im++) {
		int32_t mbk_ = (m-mbk*im >= mbk) ? mbk : m-mbk*im;
		// SplitA -----------------------------------
		int32_t split3FlagA;
		int32_t nSplitA;
		t1 = timer();
		if (checkTrans (transA) == 0) {
			//split3FlagA = (oh->splitMode == 3) ? rangeCheck <TYPE1, TYPE2> (mbk_, k, devA+im*mbk, lda) : 0; // on (if 1)
			blasRomatcopy ('t', mbk_, k, devA+im*mbk, lda, devTmp1, ldt); // transpose matA for performance
			transA_ = 't';
			if (oh->splitMode == 3) 
			//if (split3FlagA == 1) 
				nSplitA = ozblasSplit3 (oh, 'c', k, mbk_, devTmp1, ldt, devASplit, ldas, devASpExp, ldase,
										devMax2, devTmp21, ldt, devTmp22, ldt, devTmp23, ldt);
			else 
				nSplitA = ozblasSplit (oh, 'c', k, mbk_, devTmp1, ldt, devTmp1, ldt, devASplit, ldas, devASpExp, ldase, devMax1);
		} else { // transposed 
			//split3FlagA = (oh->splitMode == 3) ? rangeCheck <TYPE1, TYPE2> (k, mbk_, devA+im*mbk, lda) : 0; // on (if 1)
			if (oh->splitMode == 3) 
			//if (split3FlagA == 1)
				nSplitA = ozblasSplit3 (oh, 'c', k, mbk_, devA+im*mbk*lda, lda, devASplit, ldas, devASpExp, ldase,
										devMax2, devTmp21, ldt, devTmp22, ldt, devTmp23, ldt);
			else 
				nSplitA = ozblasSplit (oh, 'c', k, mbk_, devA+im*mbk*lda, lda, devTmp1, ldt, devASplit, ldas, devASpExp, ldase, devMax1);
		}
		oh->t_SplitA += timer() - t1;

		for (int32_t in = 0; in < ceil((float)n/nbk); in++) {
			int32_t nbk_ = (n-nbk*in >= nbk) ? nbk : n-nbk*in;
			// SplitB -----------------------------------
			int32_t split3FlagB;
			int32_t nSplitB;
			t1 = timer();
			if (checkTrans (transB) == 0) {
				//split3FlagB = (oh->splitMode == 3) ? rangeCheck <TYPE1, TYPE2> (k, nbk_, devB+in*nbk*ldb, ldb) : 0; // on (if 1)
				if (oh->splitMode == 3) 
				//if (split3FlagB == 1) 
					nSplitB = ozblasSplit3 (oh, 'c', k, nbk_, devB+in*nbk*ldb, ldb, devBSplit, ldbs, devBSpExp, ldbse,
											devMax2, devTmp21, ldt, devTmp22, ldt, devTmp23, ldt);
				else 
					nSplitB = ozblasSplit (oh, 'c', k, nbk_, devB+in*nbk*ldb, ldb, devTmp1, ldt, devBSplit, ldbs, devBSpExp, ldbse, devMax1);
			} else { // transposed
				//split3FlagB = (oh->splitMode == 3) ? rangeCheck <TYPE1, TYPE2> (nbk_, k, devB+in*nbk, ldb) : 0; // on (if 1)
				blasRomatcopy ('t', nbk_, k, devB+in*nbk, ldb, devTmp1, ldt); // transpose matB for performance
				transB_ = 'n';
				if (oh->splitMode == 3) 
				//if (split3FlagB == 1) 
					nSplitB = ozblasSplit3 (oh, 'c', k, nbk_, devTmp1, ldt, devBSplit, ldbs, devBSpExp, ldbse,
											devMax2, devTmp21, ldt, devTmp22, ldt, devTmp23, ldt);
				else
					nSplitB = ozblasSplit (oh, 'c', k, nbk_, devTmp1, ldt, devTmp1, ldt, devBSplit, ldbs, devBSpExp, ldbse, devMax1);
			}
			oh->t_SplitB += timer() - t1;

			// Compute --------------------------------------
			t1 = timer();
			double t_sum_local = 0.;
			int32_t ic = 0;
			int32_t maxlevel = std::max (2, std::max ((nSplitA + nSplitB) - oh->fastMode - 1, std::min (nSplitA, nSplitB)));

			if (n == 1 && m == 1 && oh->splitEpsMode == 2 && oh->fastMode == 0 && (oh->sumMode < 2 || oh->sumMode == 30)) { // Dot2 (only on DOT)
				TYPE2 *ptrA, *ptrB, *ptrC;
				ptrA = devASplit;
				ptrB = devBSplit;
				ptrC = devCSplit;
				// Computation (GEMM) -----------------------------------
				blasRgemm_x2 (transA_, transB_, nSplitA, nSplitB, k, fone, ptrA, ldas, ptrB, ldbs, fzero, ptrC, nSplitA);
				ic++;
			} else {
				if (oh->useBatchedGemmFlag) { // with batched GEMM (for DOT, useBatchedGemmFlag is always 0)
					int32_t numB;
					if (n == 1 && oh->fastMode == 0) { // GEMV with fast=0
						for (int32_t ia = 0; ia < std::min (maxlevel, nSplitA); ia++) {
							numB = std::min (nSplitB, maxlevel - ia);
							batchAptr[ic] = devASplit+ldas*mbk_*ia;
							batchBptr[ic] = devBSplit;
							batchCptr[ic] = devCSplit+ldcs*numB*ic; // as nbk=1
							ic++;
						}
					} else {
						for (int32_t ik = 0; ik < maxlevel; ik++) {
							for (int32_t ia = 0; ia < nSplitA; ia++) {
								for (int32_t ib = 0; ib < nSplitB; ib++) {
									if (ik == ia + ib) {
										batchAptr[ic] = devASplit+ldas*mbk_*ia;
										batchBptr[ic] = devBSplit+ldbs*nbk_*ib;
										batchCptr[ic] = devCSplit+ldcs*nbk_*ic;
										ic++;
									}
								}
							}
						}
					}
					#if defined (MKL) 
					int32_t n_ = (n == 1 && oh->fastMode == 0) ? numB : nbk_;
					blasRgemmBatch (transA_, transB_, mbk_, n_, k, fone, (const TYPE2**)batchAptr, ldas,
									(const TYPE2**)batchBptr, ldbs, fzero, (TYPE2**)batchCptr, ldcs, 1, ic);
					#else
					fprintf (OUTPUT, "OzBLAS error: batched BLAS is not available.\n");
					exit(1);
					#endif
				} else { // without batchedGEMM (DOT always goes without batchedGEMM)
					TYPE2 *ptrA, *ptrB, *ptrC;
					if (n == 1 && m == 1 && oh->fastMode == 0 && (oh->sumMode < 2 || oh->sumMode == 30)) { // DOT with fast=0 with sumMode=0 or 1
						ptrA = devASplit;
						ptrB = devBSplit;
						ptrC = devCSplit;
						// Computation (GEMM) -----------------------------------
						blasRgemm (transA_, transB_, nSplitA, nSplitB, k, fone, ptrA, ldas, ptrB, ldbs, fzero, ptrC, nSplitA);
						ic++;
					} else if (n == 1 && oh->fastMode == 0 && (oh->sumMode < 2 || oh->sumMode == 30)) { // GEMV with fast=0 with sumMode=0 or 1
						for (int32_t ia = 0; ia < std::min (maxlevel, nSplitA); ia++) {
							int32_t numB = std::min (nSplitB, maxlevel - ia);
							ptrA = devASplit+ldas*mbk_*ia;
							ptrB = devBSplit;
							ptrC = devCSplit+ldcs*numB*ic;
							// Computation (GEMM) -----------------------------------
							blasRgemm (transA_, transB_, mbk_, numB, k, fone, ptrA, ldas, ptrB, ldbs, fzero, ptrC, ldcs);
							ic++;
						} // EndFor (ia)
					} else { // GEMM and the other cases with fast=1
						// Check num of GEMMs 
						ic = 0;
						for (int32_t ia = 0; ia < nSplitA; ia++) {
							for (int32_t ib = 0; ib < nSplitB; ib++) {
								if (ia + ib < maxlevel) 
									ic++;
							}
						}
						int32_t nSplitC = ic;
						ic = 0;
						for (int32_t ik = 0; ik < maxlevel; ik++) {
							for (int32_t ia = 0; ia < nSplitA; ia++) {
								for (int32_t ib = 0; ib < nSplitB; ib++) {
									if (ik == ia + ib) {
										ptrA = devASplit+ldas*mbk_*ia;
										ptrB = devBSplit+ldbs*nbk_*ib;
										ptrC = (oh->sumMode < 2 || oh->sumMode == 30) ? devCSplit+ldcs*nbk_*ic : devCSplit;
										// Computation (GEMM) -----------------------------------
										blasRgemm (transA_, transB_, mbk_, nbk_, k, fone, ptrA, ldas, ptrB, ldbs, fzero, ptrC, ldcs);
										// Summation ------------------------------------
										double t000 = timer();
										if (oh->sumMode == 3) {
											if (ozblasLocalFsum3 (mbk_, nbk_, &devASpExp[ldase*ia], &devBSpExp[ldbse*ib], (TYPE2*)ptrC, ldcs, devCTmp, ldct,
																	devCTmp1, ldct1, devCTmp2, ldct2, devCTmp3, ldct3, (ic==nSplitC-1)?-1:ic, split3FlagA, split3FlagB)) {
											//if (ozblasLocalFsum3simd (mbk_, nbk_, &devASpExp[ldase*ia], &devBSpExp[ldbse*ib], (TYPE2*)ptrC, ldcs, devCTmp, ldct,
											//						devCTmp1, ldct1, devCTmp2, ldct2, devCTmp3, ldct3, (ic==nSplitC-1)?-1:ic, split3FlagA, split3FlagB)) {
												fprintf (OUTPUT, "OzBLAS error: Sum3 is failed.\n");
												exit (1);
											}
										} else if (oh->sumMode == 2) {
											ozblasLocalFsum (mbk_, nbk_, &devASpExp[ldase*ia], &devBSpExp[ldbse*ib], ptrC, ldcs, devCTmp, ldct, ic, split3FlagA, split3FlagB);
										}
										t_sum_local += timer() - t000;
										ic++;
									} // EndIf (ik == ia+ib)
								} // EndFor (ib)
							} // EndFor (ia)
						} // EndFor (ik)
						if (oh->sumMode == 2 || oh->sumMode == 3) { // copy and compute with alpha and beta
							double t000 = timer();
							ozblasAxpby (mbk_, nbk_, devCTmp, ldct, &devC[ldc*(in*nbk)+im*mbk], ldc, alpha, beta);
							t_sum_local += timer() - t000;
						}
					} 
				} // EndIf (useBatchedGemmFlag)
			} // EndIf (Dot2)
			oh->t_comp += timer() - t1;
			oh->t_comp -= t_sum_local;
			oh->t_sum += t_sum_local;
	
			// Sum -----------------------------------------
			if (globalSum) {
				t1 = timer();
				int32_t sumorder = 1;
				if (m == 1 && n == 1) { 
					sumorder = (oh->fastMode == 0) ? 2 : 1; // DOT w/o fastmode -> 2
					if (oh->splitEpsMode == 2) maxlevel = nSplitA + nSplitB*2;
					if (ozblasGlobalSum (oh, 1, 1, devASpExp, ldase, nSplitA, devBSpExp, ldbse, nSplitB*((oh->splitEpsMode==2)?2:1),
										devCSplit, 1, 1, &devC[ldc*(in*nbk)+im*mbk], ldc, alpha, beta, maxlevel, sumorder, split3FlagA, split3FlagB)) {
						fprintf (OUTPUT, "OzBLAS error: sum is failed\n");
						exit (1);
					}
				} else {
					sumorder = (oh->fastMode == 0 && nbk_ == 1) ? 3 : 1; // GEMV w/o fastmode -> 3
					if (ozblasGlobalSum (oh, mbk_, nbk_, devASpExp, ldase, nSplitA, devBSpExp, ldbse, nSplitB,
					  					devCSplit, ldcs*nbk_, ldcs, &devC[ldc*(in*nbk)+im*mbk], ldc, alpha, beta, maxlevel, sumorder, split3FlagA, split3FlagB)) {
					  	fprintf (OUTPUT, "OzBLAS error: sum is failed\n");
					   	exit (1);
                    }
				}
				oh->t_sum += timer() - t1;
			}
			block_count++;
			oh->nSplitA += nSplitA;
			oh->nSplitB += nSplitB;
			oh->nSplitC += ic;
		} // EndFor (in)
	} // EndFor (im)

	oh->t_total = timer() - t0;
	oh->nSplitA /= (float)block_count;
	oh->nSplitB /= (float)block_count;
	oh->nSplitC /= (float)block_count;

	oh->useBatchedGemmFlag = _useBatchedGemmFlag;
	oh->fastMode = _fastMode;

	return 0;
}
template int32_t ozblasRgemm <__float128, double, double> (ozblasHandle_t *oh,	const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const __float128 alpha, const __float128 *devA, const int32_t lda, const __float128 *devB, const int32_t ldb, const __float128 beta, __float128 *devC, const int32_t ldc);
template int32_t ozblasRgemm <__float128, float, float> (ozblasHandle_t *oh,	const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const __float128 alpha, const __float128 *devA, const int32_t lda, const __float128 *devB, const int32_t ldb, const __float128 beta, __float128 *devC, const int32_t ldc);
template int32_t ozblasRgemm <double, double, double> (ozblasHandle_t *oh,	const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const double alpha, const double *devA, const int32_t lda, const double *devB, const int32_t ldb, const double beta, double *devC, const int32_t ldc);
template int32_t ozblasRgemm <double, float, float> (ozblasHandle_t *oh,	const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const double alpha, const double *devA, const int32_t lda, const double *devB, const int32_t ldb, const double beta, double *devC, const int32_t ldc);
template int32_t ozblasRgemm <float, float, float> (ozblasHandle_t *oh,	const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const float alpha, const float *devA, const int32_t lda, const float *devB, const int32_t ldb, const float beta, float *devC, const int32_t ldc); 
template int32_t ozblasRgemm <float, double, double> (ozblasHandle_t *oh,	const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const float alpha, const float *devA, const int32_t lda, const float *devB, const int32_t ldb, const float beta, float *devC, const int32_t ldc); 

