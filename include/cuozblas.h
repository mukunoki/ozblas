#ifndef CUOZBLAS_H
#define CUOZBLAS_H

#include <cublas_v2.h>
#include <cusparse_v2.h>

typedef struct {
	cublasHandle_t ch;
	cusparseHandle_t csh;

	// core 
	char* devWork;
	char* devWorkCommon;
	uint64_t workSizeBytes;
	uint64_t memAddr;
	int32_t nSplitMax;
	char* hstBatchAddr;
	char* devBatchAddr;

	int32_t initialized;

	// option flags
	int32_t splitMode;
	int32_t fastMode;
	int32_t reproMode;
	int32_t sumMode;
	int32_t useBatchedGemmFlag;
	int32_t splitEpsMode;
	int32_t precxFlag;

	// exec info
	float nSplitA;
	float nSplitB;
	float nSplitC;

	float t_SplitA;
	float t_SplitB;
	float t_comp;
	float t_sum;
	float t_total;
	float n_comp;
	int32_t mbk;
	int32_t nbk;

	// for SpMV in iterative solvers
	int32_t nSplitA_;
	int32_t splitShift;
	uint64_t memMaskSplitA;

	// for CG
	int32_t trueresFlag;
	int32_t verbose;
	int32_t cg_numiter;
	void* cg_verbose1;
	void* cg_verbose2;
	double t_SplitMat_total;
	double t_SplitVec_total;
	double t_Sum_total;
	double t_AXPY_SCAL_total;
	double t_DOT_NRM2_total;
	double t_SpMV_SpMM_total;

} cuozblasHandle_t;

// helper routines
extern void cuozblasCreate (cuozblasHandle_t*, uint64_t);
extern void cuozblasDestroy (cuozblasHandle_t*);

// ================================
// BLAS template
// ================================

template <typename TYPE1, typename TYPE2, typename TYPE3> int32_t cuozblasRnrm2 (cuozblasHandle_t *oh, const int32_t n, const TYPE1* devX, const int32_t incx, TYPE1* ret);
template <typename TYPE> int32_t cuozblasRaxpy (cuozblasHandle_t *oh, const int32_t n, const TYPE alpha, const TYPE *devX, const int32_t incx, TYPE *devY, const int32_t incy);
template <typename TYPE1, typename TYPE2, typename TYPE3> int32_t cuozblasRdot (cuozblasHandle_t *oh, const int32_t n, const TYPE1 *devA, const int32_t incx, const TYPE1 *devB, const int32_t incy, TYPE1 *ret);
template <typename TYPE1, typename TYPE2, typename TYPE3> int32_t cuozblasRgemv (cuozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const TYPE1 alpha, const TYPE1 *devA, const int32_t lda, const TYPE1 *devB, const int32_t incx, const TYPE1 beta, TYPE1 *devC, const int32_t incy);
template <typename TYPE1, typename TYPE2, typename TYPE3> int32_t cuozblasRgemm (cuozblasHandle_t *oh,	const char tranA, const char tranB, const int32_t m, const int32_t n, const int32_t k, const TYPE1 alpha, const TYPE1 *devA, const int32_t lda, const TYPE1 *devB, const int32_t ldb, const TYPE1 beta, TYPE1 *devC, const int32_t ldc);
template <typename TYPE1, typename TYPE2> int32_t cuozblasRcsrmv (cuozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const TYPE1 alpha, const cusparseMatDescr_t descrA, const TYPE1 *devA, const int32_t *devAcolind, const int32_t *devArowptr, const TYPE1 *devB, const TYPE1 beta, TYPE1 *devC);
template <typename TYPE1, typename TYPE2> TYPE2 * cuozblasRcsrmvSplitA (cuozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const cusparseMatDescr_t descrA, const TYPE1 *devA, const int32_t *devArowptr);
template <typename TYPE1, typename TYPE2> int32_t cuozblasRcg (cuozblasHandle_t *oh, const char tranA, const int32_t dimN, const int32_t dimNNZ, const cusparseMatDescr_t descrA, const TYPE1 *matA, const int32_t *matAcolind, const int32_t *matArowptr, const TYPE1 *vecB, TYPE1 *vecX, int32_t maxiter, TYPE1 tol);

int32_t cuozblasRcgfr (cuozblasHandle_t *oh, const char tranA, const int32_t dimN, const int32_t dimNNZ, const cusparseMatDescr_t descrA, const double *matA, const int32_t *matAcolind, const int32_t *matArowptr, const double *vecB, double *vecX, int32_t maxiter, double tol);
int32_t cuozblasRcgfr_x (cuozblasHandle_t *oh, const char tranA, const int32_t dimN, const int32_t dimNNZ, const cusparseMatDescr_t descrA, const double *matA, const int32_t *matAcolind, const int32_t *matArowptr, const double *vecB, double *vecX, int32_t maxiter, double tol);

#endif
