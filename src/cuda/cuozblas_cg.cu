#include "cuozblas_common.h"

template <typename TYPE>
TYPE getTrueResidual (
	cuozblasHandle_t *oh,
	const char tranA, 
	const int32_t dimN,
	const int32_t dimNNZ,
	const cusparseMatDescr_t descrA, 
	const TYPE *matA,
	const int32_t *matAcolind,
	const int32_t *matArowptr,
    const TYPE *vecB,
	const TYPE *vecX,
	TYPE *vecT
) {
	// residual: r_i = b-Ax_i
	cuozblasCopyVec (dimN, vecB, vecT);  // t = b
	blasRcsrmv (oh->csh, tranA, dimN, dimN, dimNNZ, -1., descrA, matA, matAcolind, matArowptr, vecX, 1., vecT); // t = t-Ax (b-Ax)
	TYPE ret;
	blasRnrm2 (oh->ch, dimN, vecT, 1, &ret);
	return ret;
}

template <typename TYPE1, typename TYPE2>
int32_t cuozblasRcg (
	cuozblasHandle_t *oh,
	const char tranA, 
	const int32_t dimN,
	const int32_t dimNNZ,
	const cusparseMatDescr_t descrA, 
	const TYPE1 *matA,
	const int32_t *matAcolind,
	const int32_t *matArowptr,
    const TYPE1 *vecB,
	TYPE1 *vecX,
	int32_t maxiter,
	TYPE1 tol
) {
    double t0, t1, t2;

	// =================================
	oh->t_SplitMat_total = 0.;
	oh->t_SplitVec_total = 0.;
	oh->t_Sum_total = 0.;
	oh->t_AXPY_SCAL_total = 0.;
	oh->t_DOT_NRM2_total = 0.;
	oh->t_SpMV_SpMM_total = 0.;
	TYPE1* ptr_cg_verbose1 = (TYPE1*)oh->cg_verbose1;
	TYPE1* ptr_cg_verbose2 = (TYPE1*)oh->cg_verbose2;
	double* ptr_cg_verbose2t = (double*)oh->cg_verbose2;
	// =================================

    TYPE1 alpha, beta, resi, nrmb, tmp, dnew, dold;
	TYPE1 *vecT, *vecR, *vecP, *vecQ;
	cudaMalloc ((void **) &vecT, sizeof(TYPE1) * dimN);
	cudaMalloc ((void **) &vecR, sizeof(TYPE1) * dimN);
	cudaMalloc ((void **) &vecP, sizeof(TYPE1) * dimN);
	cudaMalloc ((void **) &vecQ, sizeof(TYPE1) * dimN);

	t1 = cutimer();
    oh->cg_numiter = 0;

	TYPE2 *matASplit = cuozblasRcsrmvSplitA <TYPE1, TYPE2> (oh, tranA, dimN, dimN, dimNNZ, descrA, matA, matArowptr); // splitting only
	cuozblasRnrm2 <TYPE1, TYPE2, TYPE2> (oh, dimN, vecB, 1, &nrmb); // nrmb = |b|
	// residual: r_0 = b-Ax_0
	cuozblasCopyVec (dimN, vecB, vecR);  // r = b
	cuozblasRcsrmv <TYPE1, TYPE2> (oh, tranA, dimN, dimN, dimNNZ, -1., descrA, (TYPE1*)matASplit, matAcolind, matArowptr, vecX, 1., vecR); // r = r-Ax (b-Ax)
	cuozblasRdot <TYPE1, TYPE2, TYPE2> (oh, dimN, vecR, 1, vecR, 1, &dold); // dold = <r,r>
	resi = sqrt (dold); // resi = |r|
	if (oh->verbose > 0) {
		t2 = cutimer();
		ptr_cg_verbose1[0] = resi/nrmb;
		if (oh->trueresFlag) 
			ptr_cg_verbose2[0] = getTrueResidual (oh, tranA, dimN, dimNNZ, descrA, matA, matAcolind, matArowptr, vecB, vecX, vecT) / nrmb;
		else 
			ptr_cg_verbose2t[0] = t2-t1;
	}
	cuozblasCopyVec (dimN, vecR, vecP);  // p = r

	int32_t verbose_cnt = 1;
    while (oh->cg_numiter < maxiter) {
        oh->cg_numiter++;

		cuozblasRcsrmv <TYPE1, TYPE2> (oh, tranA, dimN, dimN, dimNNZ, 1., descrA, (TYPE1*)matASplit, matAcolind, matArowptr, vecP, 0., vecQ); // q = Ap
		cuozblasRdot <TYPE1, TYPE2, TYPE2> (oh, dimN, vecP, 1, vecQ, 1, &tmp); // tmp = <p,q>
        alpha = dold / tmp;

		t0 = cutimer();
		cuozblasRaxpy <TYPE1> (oh, dimN, alpha, vecP, 1, vecX, 1); // x = x+alpha*p
		cuozblasRaxpy <TYPE1> (oh, dimN, -alpha, vecQ, 1, vecR, 1); // r = r-alpha*q
		oh->t_AXPY_SCAL_total += cutimer() - t0;

		cuozblasRdot <TYPE1, TYPE2, TYPE2> (oh, dimN, vecR, 1, vecR, 1, &dnew); // dnew = <r,r>
		resi = sqrt (dnew);
        beta = dnew / dold; // beta = dnew/dold
		dold = dnew;

		t0 = cutimer();
        blasRscal (oh->ch, dimN, beta, vecP, 1); // p = beta*p
		cuozblasRaxpy <TYPE1> (oh, dimN, (TYPE1)1., vecR, 1, vecP, 1); // p = p+r
		oh->t_AXPY_SCAL_total += cutimer() - t0;

		if (oh->verbose > 0) {
    		if (resi/nrmb < tol || oh->cg_numiter%oh->verbose == 0) {
				ptr_cg_verbose1[verbose_cnt] = resi/nrmb;
				if (oh->trueresFlag) {
					ptr_cg_verbose2[verbose_cnt] = getTrueResidual (oh, tranA, dimN, dimNNZ, descrA, matA, matAcolind, matArowptr, vecB, vecX, vecT) / nrmb;
				} else {
					t2 = cutimer();
					ptr_cg_verbose2t[verbose_cnt] = t2-t1;
				}
				verbose_cnt++;
           	}
		}
		if (resi/nrmb < tol) break;
    }

	cudaFree (vecT);
	cudaFree (vecR);
	cudaFree (vecP);
	cudaFree (vecQ);

	// =================================
	// reset parameters
	oh->memMaskSplitA = 0; // disable pre-split of matA 
	oh->splitShift = 1; // default (no-Splitshift)
	// =================================

	return 0;
}  
template int32_t cuozblasRcg <double, double> ( cuozblasHandle_t *oh, const char tranA, const int32_t dimN, const int32_t dimNNZ, const cusparseMatDescr_t descrA, const double *matA, const int32_t *matAcolind, const int32_t *matArowptr, const double *vecB, double *vecX, int32_t maxiter, double tol);
template int32_t cuozblasRcg <double, float> ( cuozblasHandle_t *oh, const char tranA, const int32_t dimN, const int32_t dimNNZ, const cusparseMatDescr_t descrA, const double *matA, const int32_t *matAcolind, const int32_t *matArowptr, const double *vecB, double *vecX, int32_t maxiter, double tol);
template int32_t cuozblasRcg <float, float> ( cuozblasHandle_t *oh, const char tranA, const int32_t dimN, const int32_t dimNNZ, const cusparseMatDescr_t descrA, const float *matA, const int32_t *matAcolind, const int32_t *matArowptr, const float *vecB, float *vecX, int32_t maxiter, float tol);
template int32_t cuozblasRcg <float, double> ( cuozblasHandle_t *oh, const char tranA, const int32_t dimN, const int32_t dimNNZ, const cusparseMatDescr_t descrA, const float *matA, const int32_t *matAcolind, const int32_t *matArowptr, const float *vecB, float *vecX, int32_t maxiter, float tol);
