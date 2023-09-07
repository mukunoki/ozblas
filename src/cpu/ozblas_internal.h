#ifndef OZBLAS_INTERNAL_H
#define OZBLAS_INTERNAL_H

//=============================================
// ozblas_aux.cpp
//=============================================
int32_t checkTrans (const char tran);
template <typename TYPE> constexpr int32_t getEmin ();
template <typename TYPE> TYPE fabs1 (const TYPE v);
template <typename TYPE> TYPE scalbn1 (const TYPE v, const int n);
template <typename TYPE1, typename TYPE2> int32_t rangeCheck (const int32_t m, const int32_t n, const TYPE1 *mat, const int32_t ld);
double timer ();
int32_t memCheck (ozblasHandle_t *oh);
void ozblasMatAddrAlloc (ozblasHandle_t *oh, const int32_t m, const int32_t n, const int32_t size, void **dev, int32_t &lds);
void ozblasVecAddrAlloc (ozblasHandle_t *oh, const int32_t m, const int32_t size, void **dev);
template <typename TYPE> void ozblasCopyVec (const int32_t n, const TYPE *devIn, TYPE *devOut);
void PrintMat (const int32_t m, const int32_t n, const double *devC, const int32_t ldd);
void PrintMatInt (const int32_t m, const int32_t n, const int32_t *devC, const int32_t ldd);
CBLAS_TRANSPOSE ToCblasOp (const char tran);
#if defined (CUBLAS)
cublasOperation_t ToCublasOp (const char tran);
#endif
char FromCublasOp (CBLAS_TRANSPOSE tran);

//=============================================
// ozblas_XXX
//=============================================

int32_t getPitchSize (int32_t n);
void counterInit (ozblasHandle_t *oh);

template <typename TYPE1, typename TYPE2> int32_t ozblasSplit (ozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const TYPE1 *devInput, const int32_t ldi, TYPE1 *devOutput, const int32_t ldo, TYPE2 *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, TYPE1 *devMax);
template <typename TYPE1, typename TYPE2> int32_t ozblasSplit3 (ozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const TYPE1 *devInput, const int32_t ldi, TYPE2 *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, TYPE2 *devMax, TYPE2 *devTmpD1, const int32_t ldt1, TYPE2 *devTmpD2, const int32_t ldt2, TYPE2 *devTmpD3, const int32_t ldt3, TYPE2 *devTmp, const int32_t ldt);
template <typename TYPE1, typename TYPE2> int32_t ozblasSplitA (ozblasHandle_t *oh, const char major, const int32_t ma, const int32_t ka, const TYPE1 *devAInput, const int32_t ldai, const int32_t kb, const int32_t nb, const TYPE1 *devBInput, const int32_t ldbi, TYPE1 *devAOutput, const int32_t ldao, TYPE2 *devASplit, const int32_t ldas, short *devASpExp, const int32_t ldase, TYPE1 *devAMax, TYPE1 *devAtmp, const int32_t ldat, TYPE1 *devBtmp, const int32_t ldbt, TYPE1 *devE, TYPE1 *devBe, TYPE1 *devB1, TYPE1 *devB2);
template <typename TYPE1, typename TYPE2> int32_t ozblasSplitSparse (ozblasHandle_t *oh, const char major, const int32_t m, const TYPE1 *devInput, const int32_t *devRowptr, TYPE1 *devOutput, TYPE2 *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, TYPE1 *devMax);
template <typename TYPE1, typename TYPE2> int32_t ozblasGlobalSum (ozblasHandle_t *oh, const int32_t m, const int32_t n, const short *devASpExp, const int32_t ldas, const int32_t numsplitA, const short *devBSpExp, const int32_t ldbs, const int32_t numsplitB, TYPE2 *devCsplit, const int32_t llsc, const int32_t ldsc, TYPE1 *devC, const int32_t ldc, const TYPE1 alpha, const TYPE1 beta, const int32_t maxlevel, const int32_t sumOrder);

template <typename TYPE1, typename TYPE2> int32_t ozblasLocalFsum (const int32_t m, const int32_t n, const short *devASpExp, const short *devBSpExp, const TYPE2 *devCsplit, const int32_t ldcs, TYPE1 *devCtmp, const int32_t ldct, const int32_t ic);
template <typename TYPE> int32_t ozblasAxpby (const int32_t m, const int32_t n, const TYPE *devCsplit, const int32_t ldsc, TYPE *devC, const int32_t ldc, const TYPE alpha, const TYPE beta);
template <typename TYPE1, typename TYPE2> int32_t ozblasLocalFsum3 (const int32_t m, const int32_t n, const short *devASpExp, const short *devBSpExp, const TYPE2 *devCsplit, const int32_t ldcs, TYPE1 *devCtmp, const int32_t ldct, TYPE2 *devCtmp1, const int32_t ldct1, TYPE2 *devCtmp2, const int32_t ldct2, TYPE2 *devCtmp3, const int32_t ldct3, const int32_t ic);
template <typename TYPE1, typename TYPE15, typename TYPE2> int32_t ozblasLocalFsum3 (const int32_t m, const int32_t n, const short *devASpExp, const short *devBSpExp, const TYPE15 *devCsplit, const int32_t ldcs, TYPE1 *devCtmp, const int32_t ldct, TYPE2 *devCtmp1, const int32_t ldct1, TYPE2 *devCtmp2, const int32_t ldct2, TYPE2 *devCtmp3, const int32_t ldct3, const int32_t ic);

//=============================================
// BLAS Wrapper
//=============================================
// IAMAX
int32_t blasRiamax (const int32_t n, const float* x, const int32_t incx);
int32_t blasRiamax (const int32_t n, const double* x, const int32_t incx);
#if defined (FLOAT128)
int32_t blasRiamax (const int32_t n, const __float128* x, const int32_t incx);
#endif
// ASUM
float blasRasum (const int32_t n, const float* x, const int32_t incx);
double blasRasum (const int32_t n, const double* x, const int32_t incx);
#if defined (FLOAT128)
__float128 blasRasum (const int32_t n, const __float128* x, const int32_t incx);
#endif
// SCAL
void blasRscal (const int32_t n, const float alpha, float* x, const int32_t incx);
void blasRscal (const int32_t n, const double alpha, double* x, const int32_t incx);
#if defined (FLOAT128)
void blasRscal (const int32_t n, const __float128 alpha, __float128* x, const int32_t incx);
#endif
// AXPY
void blasRaxpy (const int32_t n, const float alpha, const float* x, const int32_t incx, float* y, const int32_t incy);
void blasRaxpy (const int32_t n, const double alpha, const double* x, const int32_t incx, double* y, const int32_t incy);
#if defined (FLOAT128)
void blasRaxpy (const int32_t n, const __float128 alpha, const __float128* x, const int32_t incx, __float128* y, const int32_t incy);
#endif
// DOT
float blasRdot (const int32_t n, const float* x, const int32_t incx, const float* y, const int32_t incy);
double blasRdot (const int32_t n, const double* x, const int32_t incx, const double* y, const int32_t incy);
float blasRdotX (const int32_t n, const float* x, const int32_t incx, const float* y, const int32_t incy);
double blasRdotX (const int32_t n, const double* x, const int32_t incx, const double* y, const int32_t incy);
#if defined (FLOAT128)
__float128 blasRdot (const int32_t n, const __float128* x, const int32_t incx, const __float128* y, const int32_t incy);
__float128 blasRdotX (const int32_t n, const __float128* x, const int32_t incx, const __float128* y, const int32_t incy);
#endif
// NRM2
float blasRnrm2 (const int32_t n, const float* x, const int32_t incx);
double blasRnrm2 (const int32_t n, const double* x, const int32_t incx);
float blasRnrm2X (const int32_t n, const float* x, const int32_t incx);
double blasRnrm2X (const int32_t n, const double* x, const int32_t incx);
#if defined (FLOAT128)
__float128 blasRnrm2 (const int32_t n, const __float128* x, const int32_t incx);
__float128 blasRnrm2X (const int32_t n, const __float128* x, const int32_t incx);
#endif
// GEMV
void blasRgemv (const char trans, const int32_t m, const int32_t n, const float alpha, const float* A, const int32_t lda, const float* x, const int32_t incx, const float beta, float* y, const int32_t incy);
void blasRgemv (const char trans, const int32_t m, const int32_t n, const double alpha, const double* A, const int32_t lda, const double* x, const int32_t incx, const double beta, double* y, const int32_t incy);
#if defined (FLOAT128)
void blasRgemv (const char trans, const int32_t m, const int32_t n, const __float128 alpha, const __float128* A, const int32_t lda, const __float128* x, const int32_t incx, const __float128 beta, __float128* y, const int32_t incy);
#endif
// GEMM
void blasRgemm (const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const float alpha, const float* A, const int32_t lda, const float* B, const int32_t ldb, const float beta, float* C, const int32_t ldc);
void blasRgemm (const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const double alpha, const double* A, const int32_t lda, const double* B, const int32_t ldb, const double beta, double* C, const int32_t ldc);
#if defined (FLOAT128)
void blasRgemm (const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const __float128 alpha, const __float128* A, const int32_t lda, const __float128* B, const int32_t ldb, const __float128 beta, __float128* C, const int32_t ldc);
#endif
void blasRgemm_x2 (const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const float alpha, const float* A, const int32_t lda, const float* B, const int32_t ldb, const float beta, float* C, const int32_t ldc);
void blasRgemm_x2 (const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const double alpha, const double* A, const int32_t lda, const double* B, const int32_t ldb, const double beta, double* C, const int32_t ldc);
// CUDA-GEMM
#if defined (CUBLAS)
void cublasRgemm (cublasHandle_t handle, cublasOperation_t transA, cublasOperation_t transB, const int32_t m, const int32_t n, const int32_t k, const float* alpha, const float* A, const int32_t lda, const float* B, const int32_t ldb, const float* beta, float* C, const int32_t ldc);
void cublasRgemm (cublasHandle_t handle, cublasOperation_t transA, cublasOperation_t transB, const int32_t m, const int32_t n, const int32_t k, const double* alpha, const double* A, const int32_t lda, const double* B, const int32_t ldb, const double* beta, double* C, const int32_t ldc);
#endif
// GEMM-BATCH
void blasRgemmBatch (const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const float alpha, const float** A, const int32_t lda, const float** B, const int32_t ldb, const float beta, float** C, const int32_t ldc, const int32_t grp, const int32_t cnt);
void blasRgemmBatch (const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const double alpha, const double** A, const int32_t lda, const double** B, const int32_t ldb, const double beta, double** C, const int32_t ldc, const int32_t grp, const int32_t cnt);
// CSRMV
void blasRcsrmv (const char trans, const int32_t m, const int32_t n, const int32_t nnz, const float alpha, const char *descrA, const float *A, const int32_t *devAcolind, const int32_t *devArowptr, const float *X, const float beta, float *Y);
void blasRcsrmv (const char trans, const int32_t m, const int32_t n, const int32_t nnz, const double alpha, const char *descrA, const double *A, const int32_t *devAcolind, const int32_t *devArowptr, const double *X, const double beta, double *Y);
void blasRcsrmvX (const char trans, const int32_t m, const int32_t n, const int32_t nnz, const float alpha, const char *descrA, const float *A, const int32_t *devAcolind, const int32_t *devArowptr, const float *X, const float beta, float *Y);
void blasRcsrmvX (const char trans, const int32_t m, const int32_t n, const int32_t nnz, const double alpha, const char *descrA, const double *A, const int32_t *devAcolind, const int32_t *devArowptr, const double *X, const double beta, double *Y);
#if defined (FLOAT128)
void blasRcsrmv (const char trans, const int32_t m, const int32_t n, const int32_t nnz, const __float128 alpha, const char *descrA, const __float128 *A, const int32_t *devAcolind, const int32_t *devArowptr, const __float128 *X, const __float128 beta, __float128 *Y);
void blasRcsrmvX (const char trans, const int32_t m, const int32_t n, const int32_t nnz, const __float128 alpha, const char *descrA, const __float128 *A, const int32_t *devAcolind, const int32_t *devArowptr, const __float128 *X, const __float128 beta, __float128 *Y);
#endif
// CSRMM
void blasRcsrmm (const char trans, const int32_t m, const int32_t n, const int32_t k, const int32_t nnz, const float alpha, const char *descrA, const float *A, const int32_t *devAcolind, const int32_t *devArowptr, const float *B, const int32_t ldb, const float beta, float *C, const int32_t ldc);
void blasRcsrmm (const char trans, const int32_t m, const int32_t n, const int32_t k, const int32_t nnz, const double alpha, const char *descrA, const double *A, const int32_t *devAcolind, const int32_t *devArowptr, const double *B, const int32_t ldb, const double beta, double *C, const int32_t ldc);
void blasRcsrmm_x2 (const char trans, const int32_t m, const int32_t n, const int32_t k, const int32_t nnz, const float alpha, const char *descrA, const float *A, const int32_t *devAcolind, const int32_t *devArowptr, const float *B, const int32_t ldb, const float beta, float *C, const int32_t ldc);
void blasRcsrmm_x2 (const char trans, const int32_t m, const int32_t n, const int32_t k, const int32_t nnz, const double alpha, const char *descrA, const double *A, const int32_t *devAcolind, const int32_t *devArowptr, const double *B, const int32_t ldb, const double beta, double *C, const int32_t ldc);
// OMATCOPY
void blasRomatcopy (const char trans, const int32_t m, const int32_t n, const float* A, const int32_t lda, float* B, const int32_t ldb);
void blasRomatcopy (const char trans, const int32_t m, const int32_t n, const double* A, const int32_t lda, double* B, const int32_t ldb);
#if defined (FLOAT128)
void blasRomatcopy (const char trans, const int32_t m, const int32_t n, const __float128* A, const int32_t lda, __float128* B, const int32_t ldb);
#endif

#endif
