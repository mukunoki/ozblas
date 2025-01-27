#ifndef OZBLAS_H
#define OZBLAS_H

#if defined (CUBLAS)
#include <cublas_v2.h>
#endif

typedef struct {
	#if defined (CUBLAS)
	cublasHandle_t ch;
	#endif

	// core 
	char* devWork;
	uint64_t workSizeBytes;
	uint64_t memAddr;
	int32_t nSplitMax;

	int32_t initialized;

	// option flags
	int32_t splitModeFlag;
	int32_t fastModeFlag;
	int32_t reproModeFlag;
	int32_t sumModeFlag;
	int32_t useBatchedGemmFlag;
	int32_t splitEpsModeFlag;
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
} ozblasHandle_t;

// helper routines
extern void ozblasCreate (ozblasHandle_t*, uint64_t);
extern void ozblasDestroy (ozblasHandle_t*);

// ================================
// BLAS template
// ================================

template <typename TYPE1, typename TYPE2, typename TYPE3> TYPE1 ozblasRnrm2 (ozblasHandle_t *oh, const int n, const TYPE1* devX, const int incx);
template <typename TYPE> int32_t ozblasRaxpy (ozblasHandle_t *oh, const int32_t n, const TYPE alpha, const TYPE *devX, const int32_t incx, TYPE *devY, const int32_t incy);
template <typename TYPE1, typename TYPE2, typename TYPE3> TYPE1 ozblasRdot (ozblasHandle_t *oh, const int32_t n, const TYPE1 *devA, const int32_t incx, const TYPE1 *devB, const int32_t incy);
template <typename TYPE1, typename TYPE2, typename TYPE3> int32_t ozblasRgemv (ozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const TYPE1 alpha, const TYPE1 *devA, const int32_t lda, const TYPE1 *devB, const int32_t incx, const TYPE1 beta, TYPE1 *devC, const int32_t incy);
template <typename TYPE1, typename TYPE2, typename TYPE3> int32_t ozblasRgemm (ozblasHandle_t *oh,	const char tranA, const char tranB, const int32_t m, const int32_t n, const int32_t k, const TYPE1 alpha, const TYPE1 *devA, const int32_t lda, const TYPE1 *devB, const int32_t ldb, const TYPE1 beta, TYPE1 *devC, const int32_t ldc);
template <typename TYPE1, typename TYPE2> int32_t ozblasRcsrmv (ozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const TYPE1 alpha, const char *descrA, const TYPE1 *devA, const int32_t *devAcolind, const int32_t *devArowptr, const TYPE1 *devB, const TYPE1 beta, TYPE1 *devC);
template <typename TYPE1, typename TYPE2> TYPE2 * ozblasRcsrmvSplitA (ozblasHandle_t *oh, const char tranA, const int32_t m, const int32_t n, const int32_t nnz, const char *descrA, const TYPE1 *devA, const int32_t *devArowptr);
template <typename TYPE1, typename TYPE2> int32_t ozblasRcg (ozblasHandle_t *oh, const char tranA, const int32_t dimN, const int32_t dimNNZ, const char *descrA, const TYPE1 *matA, const int32_t *matAcolind, const int32_t *matArowptr, const TYPE1 *vecB, TYPE1 *vecX, int32_t maxiter, TYPE1 tol);

// ================================
// BLAS wrapper (for testing)
// ================================
void blasRcsrmv (const char trans, const int32_t m, const int32_t n, const int32_t nnz, const float alpha, const char *descrA, const float *A, const int32_t *devAcolind, const int32_t *devArowptr, const float *X, const float beta, float *Y);
void blasRcsrmv (const char trans, const int32_t m, const int32_t n, const int32_t nnz, const double alpha, const char *descrA, const double *A, const int32_t *devAcolind, const int32_t *devArowptr, const double *X, const double beta, double *Y);
void blasRcsrmm (const char trans, const int32_t m, const int32_t n, const int32_t k, const int32_t nnz, const double alpha, const char *descrA, const double *A, const int32_t *devAcolind, const int32_t *devArowptr, const double *B, const int32_t ldb, const double beta, double *C, const int32_t ldc);
void blasRcsrmm_x2 (const char trans, const int32_t m, const int32_t n, const int32_t k, const int32_t nnz, const double alpha, const char *descrA, const double *A, const int32_t *devAcolind, const int32_t *devArowptr, const double *B, const int32_t ldb, const double beta, double *C, const int32_t ldc);

#endif
