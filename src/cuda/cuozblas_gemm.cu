#include "cuozblas_common.h"
#include "blas/myblas_dot.cu"

template <typename TYPE1, typename TYPE2>
int32_t cuozblasRgemm (
	cuozblasHandle_t *oh,	
	const char transA, const char transB,
	const int32_t m, const int32_t n, const int32_t k,
	const TYPE1 alpha,
	const TYPE1 *devA, const int32_t lda,
	const TYPE1 *devB, const int32_t ldb,
	const TYPE1 beta,
	TYPE1 *devC, const int32_t ldc
) {
	if (oh->reproModeFlag == 0 && oh->nSplitMax == 1) {
		blasRgemm (oh->ch, transA, transB, m, n, k, alpha, (TYPE1*)devA, lda, (TYPE1*)devB, ldb, beta, devC, ldc);
		return 0;
	}
	cucounterInit (oh);
	double t1, t0 = cutimer();

	// DOT does not use batchedGEMM and fastmode as it is computed by a GEMM,
	// Therefore, batchedGEMM and fastmode are disabled here.
	int32_t _useBatchedGemmFlag = oh->useBatchedGemmFlag;
	int32_t _fastModeFlag = oh->fastModeFlag;
	if (m == 1 && n == 1) { // DOT
		oh->useBatchedGemmFlag = 0;
		oh->fastModeFlag = 0;
	}

	TYPE1 *devATmp, *devBTmp, *devCTmp;
	TYPE1 *devAmax, *devBmax;
	TYPE2 *devASplit, *devBSplit, *devCSplit;
	TYPE2 fone = 1., fzero = 0.;
	short *devASpExp, *devBSpExp;
	TYPE2 *devCTmp1, *devCTmp2, *devCTmp3;
	int32_t sizeTypeT = sizeof (TYPE2);
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
	TYPE2 **batchAptrHst, **batchBptrHst, **batchCptrHst;
	if (oh->useBatchedGemmFlag) {
		int32_t memorysize = 0;
		batchAptr    = (TYPE2**)(oh->devBatchAddr);
		batchAptrHst = (TYPE2**)(oh->hstBatchAddr);
		memorysize  += sizeof(TYPE2*) * nSplitMaxLoc * nSplitMaxLoc;
		batchBptr    = (TYPE2**)(oh->devBatchAddr + memorysize);
		batchBptrHst = (TYPE2**)(oh->hstBatchAddr + memorysize);
		memorysize  += sizeof(TYPE2*) * nSplitMaxLoc * nSplitMaxLoc;
		batchCptr    = (TYPE2**)(oh->devBatchAddr + memorysize);
		batchCptrHst = (TYPE2**)(oh->hstBatchAddr + memorysize);
	}

	int32_t memAddrTmp = oh->memAddr;
	while (mbk > 0 && nbk > 0) { // blocking
		int32_t sizeCn = (oh->useBatchedGemmFlag || oh->sumModeFlag < 2) ? (nbk * nSplitMaxLoc * nSplitMaxLoc) : nbk;
		if (oh->splitEpsModeFlag == 2) sizeCn *= 2;
		if (cucheckTrans (transA) == 0) {
			cuozblasMatAddrAlloc (oh, mbk, k * nSplitMaxLoc, sizeType2, (void**)&devASplit, ldas); 
			cuozblasMatAddrAlloc (oh, mbk, k,                sizeType1, (void**)&devATmp,   ldat); 
		} else {
			cuozblasMatAddrAlloc (oh, k, mbk * nSplitMaxLoc, sizeType2, (void**)&devASplit, ldas); 
			cuozblasMatAddrAlloc (oh, k, mbk,                sizeType1, (void**)&devATmp,   ldat); 
		}
		if (cucheckTrans (transB) == 0) {
			cuozblasMatAddrAlloc (oh, k, nbk * nSplitMaxLoc, sizeType2, (void**)&devBSplit, ldbs);
			cuozblasMatAddrAlloc (oh, k, nbk,                sizeType1, (void**)&devBTmp,   ldbt); 
		} else {
			cuozblasMatAddrAlloc (oh, nbk, k * nSplitMaxLoc, sizeType2, (void**)&devBSplit, ldbs);
			cuozblasMatAddrAlloc (oh, nbk, k,                sizeType1, (void**)&devBTmp,   ldbt);
		}
		cuozblasMatAddrAlloc (oh, mbk, sizeCn,           sizeType2, (void**)&devCSplit, ldcs);
		cuozblasMatAddrAlloc (oh, mbk, nbk,              sizeType1, (void**)&devCTmp,   ldct);
		if (oh->sumModeFlag >= 2 && oh->useBatchedGemmFlag == 0) {
			cuozblasMatAddrAlloc (oh, mbk, nbk, sizeTypeT, (void**)&devCTmp1, ldct1);
			cuozblasMatAddrAlloc (oh, mbk, nbk, sizeTypeT, (void**)&devCTmp2, ldct2);
			cuozblasMatAddrAlloc (oh, mbk, nbk, sizeTypeT, (void**)&devCTmp3, ldct3);
		}
		// Exp
		cuozblasMatAddrAlloc (oh, mbk, nSplitMaxLoc, sizeTypeS, (void**)&devASpExp, ldase);
		cuozblasMatAddrAlloc (oh, nbk, nSplitMaxLoc, sizeTypeS, (void**)&devBSpExp, ldbse);
		// Splitting
		cuozblasVecAddrAlloc (oh, mbk, sizeType1, (void**)&devAmax);
		cuozblasVecAddrAlloc (oh, nbk, sizeType1, (void**)&devBmax);
		if (!cumemCheck (oh)) break; // check if work-memory is enough or not
		oh->memAddr = memAddrTmp;
		mbk = ceil (mbk / 2.);
		nbk = ceil (nbk / 2.);
	}
	oh->mbk = mbk;
	oh->nbk = nbk;

	// main part (Split, Comp, Sum)
	int32_t block_count = 0;
	for (int32_t im = 0; im < ceil((float)m/mbk); im++) {
		int32_t mbk_ = (m-mbk*im >= mbk) ? mbk : m-mbk*im;
		// SplitA -----------------------------------
		t1 = cutimer();
		int32_t nSplitA = 0;
		if (cucheckTrans (transA) == 0) 
			nSplitA = cuozblasSplit (oh, 'r', mbk_, k, devA+im*mbk, lda, devATmp, ldat, devASplit, ldas, devASpExp, ldase, devAmax);
		else 
			nSplitA = cuozblasSplit (oh, 'c', k, mbk_, devA+im*mbk*lda, lda, devATmp, ldat, devASplit, ldas, devASpExp, ldase, devAmax);
		oh->t_SplitA += cutimer() - t1;

		for (int32_t in = 0; in < ceil((float)n/nbk); in++) {
			int32_t nbk_ = (n-nbk*in >= nbk) ? nbk : n-nbk*in;
			// SplitB -----------------------------------
			t1 = cutimer();
			int32_t nSplitB = 0;
			if (cucheckTrans (transB) == 0) 
				nSplitB = cuozblasSplit (oh, 'c', k, nbk_, devB+in*nbk*ldb, ldb, devBTmp, ldbt, devBSplit, ldbs, devBSpExp, ldbse, devBmax);
			else
				nSplitB = cuozblasSplit (oh, 'r', nbk_, k, devB+in*nbk, ldb, devBTmp, ldbt, devBSplit, ldbs, devBSpExp, ldbse, devBmax);
			oh->t_SplitB += cutimer() - t1;

			// Compute --------------------------------------
			t1 = cutimer();
			double t_sum_local = 0.;
			int32_t ia, ib, ic, ik;
			int32_t maxlevel = (oh->fastModeFlag) ? MIN (nSplitA-1, nSplitB-1) : (nSplitA-1) + (nSplitB-1);

			// Check num of GEMMs 
			ic = 0;
			for (ia = 0; ia < nSplitA; ia++) {
				for (ib = 0; ib < nSplitB; ib++) {
					if (ia + ib <= maxlevel) {
						ic++;
						oh->n_comp += 2. * mbk_ * nbk_ * k;
					}
				}
			}
			int32_t nSplitC = ic;
			int32_t numB;
			ic = 0;
			if (n == 1 && m == 1 && oh->splitEpsModeFlag == 2 && oh->fastModeFlag == 0 && oh->sumModeFlag < 2) { // Dot2 (only on DOT)
				TYPE2 *ptrA, *ptrB, *ptrC;
				ptrA = devASplit;
				ptrB = devBSplit;
				ptrC = devCSplit;
				// Computation (GEMM) -----------------------------------
				//	blasRgemm (oh->ch, transA, transB, nSplitA, nSplitB, k, fone, ptrA, ldas, ptrB, ldbs, fzero, ptrC, nSplitA);
				blasRgemmSkinny (oh, transA, transB, nSplitA, nSplitB, k, fone, ptrA, ldas, ptrB, ldbs, fzero, ptrC, nSplitA);
				ic++;
			} else {
				if (oh->useBatchedGemmFlag) { // with batched GEMM
					if (n == 1 && oh->fastModeFlag == 0) { // DOT & GEMV with fast=0
						for (ia = 0; ia < MIN (maxlevel+1, nSplitA); ia++) {
							numB = MIN (nSplitB, maxlevel+1 - ia);
							batchAptrHst[ic] = devASplit+ldas*((cucheckTrans(transA) == 0) ? k:mbk)*ia;
							batchBptrHst[ic] = devBSplit;
							batchCptrHst[ic] = devCSplit+ldcs*numB*ic; // as nbk=1
							ic++;
						}
						nSplitC = ic;
					} else { // GEMM, DOT & GEMV with fast=1
						for (ik = 0; ik <= maxlevel; ik++) {
							for (ia = 0; ia < nSplitA; ia++) {
								for (ib = 0; ib < nSplitB; ib++) {
									if (ik == ia + ib) {
										batchAptrHst[ic] = devASplit+ldas*((cucheckTrans(transA) == 0) ? k:mbk)*ia;
										batchBptrHst[ic] = devBSplit+ldbs*((cucheckTrans(transB) == 0) ? nbk:k)*ib;
										batchCptrHst[ic] = devCSplit+ldcs*nbk*ic;
										ic++;
									}
								}
							}
						}
						nSplitC = ic;
					}
					cudaMemcpy(batchAptr, batchAptrHst, sizeof(TYPE2*) * nSplitC, cudaMemcpyHostToDevice);
					cudaMemcpy(batchBptr, batchBptrHst, sizeof(TYPE2*) * nSplitC, cudaMemcpyHostToDevice);
					cudaMemcpy(batchCptr, batchCptrHst, sizeof(TYPE2*) * nSplitC, cudaMemcpyHostToDevice);
					int32_t n_ = (n == 1 && oh->fastModeFlag == 0) ? numB : nbk_;
					blasRgemmBatch (oh->ch, transA, transB, mbk_, n_, k, fone, (const TYPE2**)batchAptr, ldas,
									(const TYPE2**)batchBptr, ldbs, fzero, (TYPE2**)batchCptr, ldcs, 1, nSplitC);
				} else {
					// without batched GEMM
					TYPE2 *ptrA, *ptrB, *ptrC;
					if (n == 1 && m == 1 && oh->fastModeFlag == 0 && oh->sumModeFlag < 2) { // DOT with fast=0 with sumMode=0 or 1
						ptrA = devASplit;
						ptrB = devBSplit;
						ptrC = devCSplit;
						// Computation (GEMM) -----------------------------------
					//	blasRgemm (oh->ch, transA, transB, nSplitA, nSplitB, k, fone, ptrA, ldas, ptrB, ldbs, fzero, ptrC, nSplitA);
						blasRgemmSkinny (oh, transA, transB, nSplitA, nSplitB, k, fone, ptrA, ldas, ptrB, ldbs, fzero, ptrC, nSplitA);
						ic++;
					} else if (n == 1 && oh->fastModeFlag == 0 && oh->sumModeFlag < 2) { // GEMV with fast=0 with sumMode=0 or 1
						for (ia = 0; ia < MIN (maxlevel+1, nSplitA); ia++) {
							numB = MIN (nSplitB, maxlevel+1 - ia);
							ptrA = devASplit+ldas*((cucheckTrans(transA) == 0) ? k:mbk)*ia;
							ptrB = devBSplit;
							ptrC = devCSplit+ldcs*numB*ic;
							// Computation (GEMM) -----------------------------------
							blasRgemm (oh->ch, transA, transB, mbk_, numB, k, fone, ptrA, ldas, ptrB, ldbs, fzero, ptrC, ldcs);
							ic++;
						} // EndFor (ia)
					} else { // GEMM and the other cases with fast=1
						for (ik = 0; ik <= maxlevel; ik++) {
							for (ia = 0; ia < nSplitA; ia++) {
								for (ib = 0; ib < nSplitB; ib++) {
									if (ik == ia + ib) {
										ptrA = devASplit+ldas*((cucheckTrans(transA) == 0) ? k:mbk)*ia;
										ptrB = devBSplit+ldbs*((cucheckTrans(transB) == 0) ? nbk:k)*ib;
										ptrC = (oh->sumModeFlag < 2) ? devCSplit+ldcs*nbk*ic : devCSplit;
										// Computation (GEMM) -----------------------------------
										blasRgemm (oh->ch, transA, transB, mbk_, nbk_, k, fone, ptrA, ldas, ptrB, ldbs, fzero, ptrC, ldcs);
										// Summation ------------------------------------
										// SumMode = 0:FSum(global), 1:NearSum(global), 2:FSum(local), 3:Sum3(local)
										if (oh->sumModeFlag == 2) {
											double t000 = cutimer();
											cuozblasLocalFsum (mbk_, nbk_, &devASpExp[ldase*ia], &devBSpExp[ldbse*ib], ptrC, ldcs, devCTmp, ldct, ic);
											t_sum_local += cutimer() - t000;
										}
										ic++;
									} // EndIf (ik == ia+ib)
								} // EndFor (ib)
							} // EndFor (ia)
						} // EndFor (ik)
						if (oh->sumModeFlag >= 2) { // copy and compute with alpha and beta
							double t000 = cutimer();
							cuozblasAxpby (mbk_, nbk_, devCTmp, ldct, &devC[ldc*(in*nbk)+im*mbk], ldc, alpha, beta);
							t_sum_local += cutimer() - t000;
						}
					} // DOT,GEMV / GEMM
				} // EndIf (useBatchedGemmFlag)
			} // Dot2
			oh->t_comp += cutimer() - t1;
			oh->t_comp -= t_sum_local;
			oh->t_sum += t_sum_local;

			// Sum -----------------------------------------
			if (oh->useBatchedGemmFlag || oh->sumModeFlag < 2) {
				t1 = cutimer();
				int32_t sumorder = 1;
				if (m == 1 && n == 1) { // DOT
					sumorder = (oh->fastModeFlag == 0) ? 2 : 1; // DOT w/o fastmode -> 2
					if (oh->splitEpsModeFlag == 2) maxlevel = (nSplitA-1) + (nSplitB*2-1);
					if (cuozblasGlobalSum (oh, 1, 1, devASpExp, ldase, nSplitA, devBSpExp, ldbse, nSplitB*((oh->splitEpsModeFlag==2)?2:1),
										devCSplit, 1, 1, &devC[ldc*(in*nbk)+im*mbk], ldc, alpha, beta, maxlevel, sumorder)) {
						fprintf (OUTPUT, "OzBLAS error: sum is failed\n");
						exit (1);
					}
				} else {
					if (cuozblasGlobalSum (oh, mbk_, nbk_, devASpExp, ldase, nSplitA, devBSpExp, ldbse, nSplitB,
										devCSplit, ldcs*nbk, ldcs, &devC[ldc*(in*nbk)+im*mbk], ldc, alpha, beta, maxlevel, sumorder)) {
						fprintf (OUTPUT, "OzBLAS error: sum is failed\n");
						exit (1);
					}
				}
				oh->t_sum += cutimer() - t1;
			}
			block_count++;
			oh->nSplitA += nSplitA;
			oh->nSplitB += nSplitB;
			oh->nSplitC += nSplitC;
		} // EndFor (in)
	} // EndFor (im)

	oh->t_total = cutimer() - t0;
	oh->nSplitA /= (float)block_count;
	oh->nSplitB /= (float)block_count;
	oh->nSplitC /= (float)block_count;

	oh->useBatchedGemmFlag = _useBatchedGemmFlag;
	oh->fastModeFlag = _fastModeFlag;

	return 0;
}
template int32_t cuozblasRgemm <double, double> (cuozblasHandle_t *oh,	const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const double alpha, const double *devA, const int32_t lda, const double *devB, const int32_t ldb, const double beta, double *devC, const int32_t ldc);
template int32_t cuozblasRgemm <double, float> (cuozblasHandle_t *oh,	const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const double alpha, const double *devA, const int32_t lda, const double *devB, const int32_t ldb, const double beta, double *devC, const int32_t ldc);
template int32_t cuozblasRgemm <float, float> (cuozblasHandle_t *oh,	const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const float alpha, const float *devA, const int32_t lda, const float *devB, const int32_t ldb, const float beta, float *devC, const int32_t ldc); 
template int32_t cuozblasRgemm <float, double> (cuozblasHandle_t *oh,	const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const float alpha, const float *devA, const int32_t lda, const float *devB, const int32_t ldb, const float beta, float *devC, const int32_t ldc); 
