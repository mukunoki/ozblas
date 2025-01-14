#include "ozblas_common.h"

template <typename TYPE1, typename TYPE2>
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
	if (oh->reproModeFlag == 0 && oh->nSplitMax == 1) {
		blasRgemm (transA, transB, m, n, k, alpha, devA, lda, devB, ldb, beta, devC, ldc);
		return 0;
	}
	counterInit (oh);
	double t1, t0 = timer();

	// DOT does not use batchedGEMM and fastmode as it is computed by a GEMM,
	// Therefore, batchedGEMM and fastmode are disabled here.
	int32_t _useBatchedGemmFlag = oh->useBatchedGemmFlag;
	int32_t _fastModeFlag = oh->fastModeFlag;
	if (m == 1 && n == 1) { // DOT
		oh->useBatchedGemmFlag = 0;
		oh->fastModeFlag = 0;
	}
	
	TYPE1 *devATmp, *devBTmp, *devCTmp;
	TYPE1 *devAmax_, *devBmax_;
	TYPE2 *devAmax, *devATmpD1, *devATmpD2, *devATmpD3;
	TYPE2 *devBmax, *devBTmpD1, *devBTmpD2, *devBTmpD3;
	TYPE2 *devASplit, *devBSplit, *devCSplit;
	TYPE2 fone = 1., fzero = 0.;
	short *devASpExp, *devBSpExp;
	// for QSGEMM, double is used, not TYPE2
	double *devCTmp1, *devCTmp2, *devCTmp3;
	int32_t	sizeTypeT = sizeof(double);
	// --
	int32_t ldas, ldbs, ldcs, ldase, ldbse, ldat, ldbt, ldct;
	int32_t ldct1, ldct2, ldct3;
	int32_t mbk = m;
	int32_t nbk = n;
	int32_t nSplitMaxLoc = (oh->nSplitMax > 0) ? oh->nSplitMax : NumSplitDefaultMax;
	int32_t sizeType1 = sizeof (TYPE1);
	int32_t sizeType2 = sizeof (TYPE2);
	int32_t sizeTypeS = sizeof (short);

	// Memory allocation 
	TYPE2 **batchAptr, **batchBptr, **batchCptr;
	if (oh->useBatchedGemmFlag) {
		ozblasVecAddrAlloc (oh, nSplitMaxLoc * nSplitMaxLoc, sizeof(TYPE2*), (void**)&batchAptr);
		ozblasVecAddrAlloc (oh, nSplitMaxLoc * nSplitMaxLoc, sizeof(TYPE2*), (void**)&batchBptr);
		ozblasVecAddrAlloc (oh, nSplitMaxLoc * nSplitMaxLoc, sizeof(TYPE2*), (void**)&batchCptr);
	}
	int64_t memAddrTmp = oh->memAddr;
	while (mbk > 0 && nbk > 0) { // blocking
		int32_t sizeCn = (oh->useBatchedGemmFlag || oh->sumModeFlag < 2) ? (nbk * nSplitMaxLoc * nSplitMaxLoc) : nbk;
		if (oh->splitEpsModeFlag == 2) sizeCn *= 2;
		ozblasMatAddrAlloc (oh, k, mbk * nSplitMaxLoc, sizeType2, (void**)&devASplit, ldas); // Note: A is transposed!! so ldas is k-based
		ozblasMatAddrAlloc (oh, k, nbk * nSplitMaxLoc, sizeType2, (void**)&devBSplit, ldbs);
		ozblasMatAddrAlloc (oh, mbk, sizeCn,           sizeType2, (void**)&devCSplit, ldcs);
		ozblasMatAddrAlloc (oh, k, mbk,                sizeType1, (void**)&devATmp,   ldat); // TRANSPOSE
		ozblasMatAddrAlloc (oh, k, nbk,                sizeType1, (void**)&devBTmp,   ldbt); 
		ozblasMatAddrAlloc (oh, mbk, nbk,              sizeType1, (void**)&devCTmp,   ldct);
		if (oh->sumModeFlag >= 2 && oh->useBatchedGemmFlag == 0) {
			ozblasMatAddrAlloc (oh, mbk, nbk, sizeTypeT, (void**)&devCTmp1, ldct1);
			ozblasMatAddrAlloc (oh, mbk, nbk, sizeTypeT, (void**)&devCTmp2, ldct2);
			ozblasMatAddrAlloc (oh, mbk, nbk, sizeTypeT, (void**)&devCTmp3, ldct3);
		}
		// Exp
		ozblasMatAddrAlloc (oh, mbk, nSplitMaxLoc, sizeTypeS, (void**)&devASpExp, ldase);
		ozblasMatAddrAlloc (oh, nbk, nSplitMaxLoc, sizeTypeS, (void**)&devBSpExp, ldbse);
		// Splitting
		ozblasVecAddrAlloc (oh, mbk, sizeType1, (void**)&devAmax_);
		ozblasVecAddrAlloc (oh, nbk, sizeType1, (void**)&devBmax_);
		// above must be allocated even if splitModeFlag is 3 as they may be used if Split3 is not used
		if (oh->splitModeFlag == 3) {
			ozblasVecAddrAlloc (oh,    mbk, sizeType2, (void**)&devAmax);
			ozblasMatAddrAlloc (oh, k, mbk, sizeType2, (void**)&devATmpD1, ldat); 
			ozblasMatAddrAlloc (oh, k, mbk, sizeType2, (void**)&devATmpD2, ldat); 
			ozblasMatAddrAlloc (oh, k, mbk, sizeType2, (void**)&devATmpD3, ldat); 
			ozblasVecAddrAlloc (oh,    nbk, sizeType2, (void**)&devBmax);
			ozblasMatAddrAlloc (oh, k, nbk, sizeType2, (void**)&devBTmpD1, ldbt);
			ozblasMatAddrAlloc (oh, k, nbk, sizeType2, (void**)&devBTmpD2, ldbt);
			ozblasMatAddrAlloc (oh, k, nbk, sizeType2, (void**)&devBTmpD3, ldbt);
		} 
		if (!memCheck (oh)) break; // check if work-memory is enough or not
		oh->memAddr = memAddrTmp;
		mbk = ceil (mbk / 2.);
		nbk = ceil (nbk / 2.);
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
		t1 = timer();
		int32_t split3FlagA;
		int32_t nSplitA;
		if (checkTrans (transA) == 0) {
			split3FlagA = (oh->splitModeFlag == 3) ? rangeCheck <TYPE1, TYPE2> (mbk_, k, devA+im*mbk, lda) : 0; // on (if 1)
			blasRomatcopy ('t', mbk_, k, devA+im*mbk, lda, devATmp, ldat); // transpose matA for performance
			transA_ = 't';
			if (split3FlagA == 1) 
				nSplitA = ozblasSplit3 (oh, 'c', k, mbk_, devATmp, ldat, devASplit, ldas, devASpExp, ldase,
										devAmax, devATmpD1, ldat, devATmpD2, ldat, devATmpD3, ldat, devATmpD1, ldat);
			else 
				nSplitA = ozblasSplit (oh, 'c', k, mbk_, devATmp, ldat, devATmp, ldat, devASplit, ldas, devASpExp, ldase, devAmax_);
		} else { // transposed 
			split3FlagA = (oh->splitModeFlag == 3) ? rangeCheck <TYPE1, TYPE2> (k, mbk_, devA+im*mbk, lda) : 0; // on (if 1)
			if (split3FlagA == 1)
				nSplitA = ozblasSplit3 (oh, 'c', k, mbk_, devA+im*mbk, lda, devASplit, ldas, devASpExp, ldase,
										devAmax, devATmpD1, ldat, devATmpD2, ldat, devATmpD3, ldat, devATmpD1, ldat);
			else 
				nSplitA = ozblasSplit (oh, 'c', k, mbk_, devA+im*mbk*lda, lda, devATmp, ldat, devASplit, ldas, devASpExp, ldase, devAmax_);
		}
		oh->t_SplitA += timer() - t1;

		for (int32_t in = 0; in < ceil((float)n/nbk); in++) {
			int32_t nbk_ = (n-nbk*in >= nbk) ? nbk : n-nbk*in;
			// SplitB -----------------------------------
			t1 = timer();
			int32_t split3FlagB;
			int32_t nSplitB;
			if (checkTrans (transB) == 0) {
				split3FlagB = (oh->splitModeFlag == 3) ? rangeCheck <TYPE1, TYPE2> (k, nbk_, devB+in*nbk*ldb, ldb) : 0; // on (if 1)
				if (split3FlagB == 1) 
					nSplitB = ozblasSplit3 (oh, 'c', k, nbk_, devB+in*nbk*ldb, ldb, devBSplit, ldbs, devBSpExp, ldbse,
											devBmax, devBTmpD1, ldbt, devBTmpD2, ldbt, devBTmpD3, ldbt, devBTmpD1, ldbt);
				else
					nSplitB = ozblasSplit (oh, 'c', k, nbk_, devB+in*nbk*ldb, ldb, devBTmp, ldbt, devBSplit, ldbs, devBSpExp, ldbse, devBmax_);
			} else { // transposed
				split3FlagB = (oh->splitModeFlag == 3) ? rangeCheck <TYPE1, TYPE2> (nbk_, k, devB+in*nbk, ldb) : 0; // on (if 1)
				blasRomatcopy ('t', nbk_, k, devB+in*nbk, ldb, devBTmp, ldbt); // transpose matB for performance
				transB_ = 'n';
				if (split3FlagB == 1) 
					nSplitB = ozblasSplit3 (oh, 'c', k, nbk_, devBTmp, ldbt, devBSplit, ldbs, devBSpExp, ldbse,
											devBmax, devBTmpD1, ldbt, devBTmpD2, ldbt, devBTmpD3, ldbt, devBTmpD1, ldbt);
				else
					nSplitB = ozblasSplit (oh, 'c', k, nbk_, devBTmp, ldbt, devBTmp, ldbt, devBSplit, ldbs, devBSpExp, ldbse, devBmax_);
			}
			oh->t_SplitB += timer() - t1;

			// Compute --------------------------------------
			t1 = timer();
			double t_sum_local = 0.;
			int32_t ic = 0;
			int32_t maxlevel = (oh->fastModeFlag) ? MIN (nSplitA-1, nSplitB-1) : (nSplitA-1) + (nSplitB-1);
			if (n == 1 && m == 1 && oh->splitEpsModeFlag == 2 && oh->fastModeFlag == 0 && oh->sumModeFlag < 2) { // Dot2 (only on DOT)
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
					if (n == 1 && oh->fastModeFlag == 0) { // GEMV with fast=0
						for (int32_t ia = 0; ia < MIN (maxlevel+1, nSplitA); ia++) {
							numB = MIN (nSplitB, maxlevel+1 - ia);
							batchAptr[ic] = devASplit+ldas*mbk*ia;
							batchBptr[ic] = devBSplit;
							batchCptr[ic] = devCSplit+ldcs*numB*ic; // as nbk=1
							ic++;
						}
					} else {
						for (int32_t ik = 0; ik <= maxlevel; ik++) {
							for (int32_t ia = 0; ia < nSplitA; ia++) {
								for (int32_t ib = 0; ib < nSplitB; ib++) {
									if (ik == ia + ib) {
										batchAptr[ic] = devASplit+ldas*mbk*ia;
										batchBptr[ic] = devBSplit+ldbs*nbk*ib;
										batchCptr[ic] = devCSplit+ldcs*nbk*ic;
										ic++;
									}
								}
							}
						}
					}
					#if defined (MKL) 
					int32_t n_ = (n == 1 && oh->fastModeFlag == 0) ? numB : nbk_;
					blasRgemmBatch (transA_, transB_, mbk_, n_, k, fone, (const TYPE2**)batchAptr, ldas,
									(const TYPE2**)batchBptr, ldbs, fzero, (TYPE2**)batchCptr, ldcs, 1, ic);
					#else
					fprintf (OUTPUT, "OzBLAS error: batched BLAS is not available.\n");
					exit(1);
					#endif
				} else { // without batchedGEMM (DOT always goes without batchedGEMM)
					TYPE2 *ptrA, *ptrB, *ptrC;
					if (n == 1 && m == 1 && oh->fastModeFlag == 0 && oh->sumModeFlag < 2) { // DOT with fast=0 with sumMode=0 or 1
						ptrA = devASplit;
						ptrB = devBSplit;
						ptrC = devCSplit;
						// Computation (GEMM) -----------------------------------
						blasRgemm (transA_, transB_, nSplitA, nSplitB, k, fone, ptrA, ldas, ptrB, ldbs, fzero, ptrC, nSplitA);
						ic++;
					} else if (n == 1 && oh->fastModeFlag == 0 && oh->sumModeFlag < 2) { // GEMV with fast=0 with sumMode=0 or 1
						for (int32_t ia = 0; ia < MIN (maxlevel+1, nSplitA); ia++) {
							int32_t numB = MIN (nSplitB, maxlevel+1 - ia);
							ptrA = devASplit+ldas*mbk*ia;
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
								if (ia + ib <= maxlevel) 
									ic++;
							}
						}
						int32_t nSplitC = ic;
						ic = 0;
						for (int32_t ik = 0; ik <= maxlevel; ik++) {
							for (int32_t ia = 0; ia < nSplitA; ia++) {
								for (int32_t ib = 0; ib < nSplitB; ib++) {
									if (ik == ia + ib) {
										ptrA = devASplit+ldas*mbk_*ia;
										ptrB = devBSplit+ldbs*nbk_*ib;
										//ptrC = (oh->sumModeFlag < 2) ? devCSplit+ldcs*nbk_*ic : devCSplit;//NG
										ptrC = (oh->sumModeFlag < 2) ? devCSplit+ldcs*nbk*ic : devCSplit;
										// Computation (GEMM) -----------------------------------
										blasRgemm (transA_, transB_, mbk_, nbk_, k, fone, ptrA, ldas, ptrB, ldbs, fzero, ptrC, ldcs);
										// Summation ------------------------------------
										double t000 = timer();
										if (oh->sumModeFlag == 3) {
											if (ozblasLocalFsum3 (mbk_, nbk_, &devASpExp[ldase*ia], &devBSpExp[ldbse*ib], (TYPE2*)ptrC, ldcs, devCTmp, ldct,
																	devCTmp1, ldct1, devCTmp2, ldct2, devCTmp3, ldct3, (ic==nSplitC-1)?-1:ic)) {
												fprintf (OUTPUT, "OzBLAS error: Sum3 is failed.\n");
												exit (1);
											}
										} else if (oh->sumModeFlag == 2) {
											ozblasLocalFsum (mbk_, nbk_, &devASpExp[ldase*ia], &devBSpExp[ldbse*ib], ptrC, ldcs, devCTmp, ldct, ic);
										}
										t_sum_local += timer() - t000;
										ic++;
									} // EndIf (ik == ia+ib)
								} // EndFor (ib)
							} // EndFor (ia)
						} // EndFor (ik)
						if (oh->sumModeFlag >= 2) { // copy and compute with alpha and beta
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
			if (oh->useBatchedGemmFlag || oh->sumModeFlag < 2) {
				t1 = timer();
				int32_t sumorder = 1;
				if (m == 1 && n == 1) { // DOT
                printf ("sum dot mbk_=%d nbk_=%d\n", mbk_, nbk_);
					sumorder = (oh->fastModeFlag == 0) ? 2 : 1; // DOT w/o fastmode -> 2
					if (oh->splitEpsModeFlag == 2) maxlevel = (nSplitA-1) + (nSplitB*2-1);
					if (ozblasGlobalSum (oh, 1, 1, devASpExp, ldase, nSplitA, devBSpExp, ldbse, nSplitB*((oh->splitEpsModeFlag==2)?2:1),
										devCSplit, 1, 1, &devC[ldc*(in*nbk)+im*mbk], ldc, alpha, beta, maxlevel, sumorder)) {
						fprintf (OUTPUT, "OzBLAS error: sum is failed\n");
						exit (1);
					}
				} else if (n == 1) { // GEMV 
                printf ("sum gemv mbk_=%d nbk_=%d\n", mbk_, nbk_);
					sumorder = (oh->fastModeFlag == 0) ? 3 : 1; // GEMV w/o fastmode -> 3
					if (ozblasGlobalSum (oh, mbk_, nbk_, devASpExp, ldase, nSplitA, devBSpExp, ldbse, nSplitB,
										devCSplit, ldcs*nbk, ldcs, &devC[ldc*(in*nbk)+im*mbk], ldc, alpha, beta, maxlevel, sumorder)) {
						fprintf (OUTPUT, "OzBLAS error: sum is failed\n");
						exit (1);
					}
				} else { // GEMM
//                printf ("sum gemm mbk_=%d nbk_=%d\n", mbk_, nbk_);
					if (ozblasGlobalSum (oh, mbk_, nbk_, devASpExp, ldase, nSplitA, devBSpExp, ldbse, nSplitB,
										devCSplit, ldcs*nbk, ldcs, &devC[ldc*(in*nbk)+im*mbk], ldc, alpha, beta, maxlevel, sumorder)) {
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
	oh->fastModeFlag = _fastModeFlag;

	return 0;
}
template int32_t ozblasRgemm <__float128, double> (ozblasHandle_t *oh,	const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const __float128 alpha, const __float128 *devA, const int32_t lda, const __float128 *devB, const int32_t ldb, const __float128 beta, __float128 *devC, const int32_t ldc);
template int32_t ozblasRgemm <__float128, float> (ozblasHandle_t *oh,	const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const __float128 alpha, const __float128 *devA, const int32_t lda, const __float128 *devB, const int32_t ldb, const __float128 beta, __float128 *devC, const int32_t ldc);
template int32_t ozblasRgemm <double, double> (ozblasHandle_t *oh,	const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const double alpha, const double *devA, const int32_t lda, const double *devB, const int32_t ldb, const double beta, double *devC, const int32_t ldc);
template int32_t ozblasRgemm <double, float> (ozblasHandle_t *oh,	const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const double alpha, const double *devA, const int32_t lda, const double *devB, const int32_t ldb, const double beta, double *devC, const int32_t ldc);
template int32_t ozblasRgemm <float, float> (ozblasHandle_t *oh,	const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const float alpha, const float *devA, const int32_t lda, const float *devB, const int32_t ldb, const float beta, float *devC, const int32_t ldc); 
template int32_t ozblasRgemm <float, double> (ozblasHandle_t *oh,	const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const float alpha, const float *devA, const int32_t lda, const float *devB, const int32_t ldb, const float beta, float *devC, const int32_t ldc); 
