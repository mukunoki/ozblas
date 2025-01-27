#ifndef CUOZBLAS_INTERNAL_H
#define CUOZBLAS_INTERNAL_H

//=============================================
// cuozblas_aux.cpp
//=============================================
int32_t cucheckTrans (const char tran);
template <typename TYPE1, typename TYPE2> int32_t curangeCheck (const int32_t m, const int32_t n, const TYPE1 *mat, const int32_t ld);
double cutimer ();
int32_t cumemCheck (cuozblasHandle_t *oh);
void cuozblasMatAddrAlloc (cuozblasHandle_t *oh, const int32_t m, const int32_t n, const int32_t size, void **dev, int32_t &lds);
void cuozblasVecAddrAlloc (cuozblasHandle_t *oh, const int32_t m, const int32_t size, void **dev);
template <typename TYPE> void cuozblasCopyVec (const int32_t n, const TYPE *devIn, TYPE *devOut);
//void PrintMat (const int32_t m, const int32_t n, const double *devC, const int32_t ldd);
//void PrintMatInt (const int32_t m, const int32_t n, const int32_t *devC, const int32_t ldd);
cublasOperation_t ToCublasOp (const char tran);
cusparseOperation_t ToCusparseOp (const char tran);
//char FromCublasOp (CBLAS_TRANSPOSE tran);
__device__ int32_t double_to_int (const double input, const int32_t rho);
__device__ double int_to_double (const int32_t input);

//=============================================
// cuozblas_XXX
//=============================================

int32_t cugetPitchSize (int32_t n);
void cucounterInit (cuozblasHandle_t *oh);

template <typename TYPE1, typename TYPE2> int32_t cuozblasSplit (cuozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const TYPE1 *devInput, const int32_t ldi, TYPE1 *devOutput, const int32_t ldo, TYPE2 *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, TYPE1 *devMax);
template <typename TYPE1, typename TYPE2> int32_t cuozblasSplit3 (cuozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const TYPE1 *devInput, const int32_t ldi, TYPE2 *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, TYPE2 *devMax, TYPE2 *devTmpD1, const int32_t ldt1, TYPE2 *devTmpD2, const int32_t ldt2, TYPE2 *devTmpD3, const int32_t ldt3, TYPE2 *devTmp, const int32_t ldt);
template <typename TYPE1, typename TYPE2> int32_t cuozblasSplitA (cuozblasHandle_t *oh, const char major, const int32_t ma, const int32_t ka, const TYPE1 *devAInput, const int32_t ldai, const int32_t kb, const int32_t nb, const TYPE1 *devBInput, const int32_t ldbi, TYPE1 *devAOutput, const int32_t ldao, TYPE2 *devASplit, const int32_t ldas, short *devASpExp, const int32_t ldase, TYPE1 *devAMax, TYPE1 *devAtmp, const int32_t ldat, TYPE1 *devBtmp, const int32_t ldbt, TYPE1 *devE, TYPE1 *devBe, TYPE1 *devB1, TYPE1 *devB2);
template <typename TYPE1, typename TYPE2> int32_t cuozblasSplitSparse (cuozblasHandle_t *oh, const char major, const int32_t m, const TYPE1 *devInput, const int32_t *devRowptr, TYPE1 *devOutput, TYPE2 *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, TYPE1 *devMax);
template <typename TYPE1, typename TYPE3> int32_t cuozblasGlobalSum (cuozblasHandle_t *oh, const int32_t m, const int32_t n, const int32_t k, const short *devASpExp, const int32_t ldas, const int32_t numsplitA, const short *devBSpExp, const int32_t ldbs, const int32_t numsplitB, TYPE3 *devCsplit, const int32_t llsc, const int32_t ldsc, TYPE1 *devC, const int32_t ldc, const TYPE1 alpha, const TYPE1 beta, const int32_t maxlevel, const int32_t sumOrder);
template <typename TYPE1, typename TYPE2> int32_t cuozblasLocalFsum (const int32_t m, const int32_t n, const short *devASpExp, const short *devBSpExp, const TYPE2 *devCsplit, const int32_t ldcs, TYPE1 *devCtmp, const int32_t ldct, const int32_t ic);
template <typename TYPE> int32_t cuozblasAxpby (const int32_t m, const int32_t n, const TYPE *devCsplit, const int32_t ldsc, TYPE *devC, const int32_t ldc, const TYPE alpha, const TYPE beta);
template <typename TYPE1, typename TYPE2> int32_t cuozblasLocalFsum3 (const int32_t m, const int32_t n, const short *devASpExp, const short *devBSpExp, const TYPE2 *devCsplit, const int32_t ldcs, TYPE1 *devCtmp, const int32_t ldct, TYPE2 *devCtmp1, const int32_t ldct1, TYPE2 *devCtmp2, const int32_t ldct2, TYPE2 *devCtmp3, const int32_t ldct3, const int32_t ic);
template <typename TYPE1, typename TYPE15, typename TYPE2> int32_t cuozblasLocalFsum3 (const int32_t m, const int32_t n, const short *devASpExp, const short *devBSpExp, const TYPE15 *devCsplit, const int32_t ldcs, TYPE1 *devCtmp, const int32_t ldct, TYPE2 *devCtmp1, const int32_t ldct1, TYPE2 *devCtmp2, const int32_t ldct2, TYPE2 *devCtmp3, const int32_t ldct3, const int32_t ic);

//=============================================
// BLAS Wrapper
//=============================================
// IAMAX
void blasRiamax (cublasHandle_t ch, const int32_t n, const float* x, const int32_t incx, int32_t* ret);
void blasRiamax (cublasHandle_t ch, const int32_t n, const double* x, const int32_t incx, int32_t* ret);
// ASUM
void blasRasum (cublasHandle_t ch, const int32_t n, const float* x, const int32_t incx, float* ret);
void blasRasum (cublasHandle_t ch, const int32_t n, const double* x, const int32_t incx, double* ret);
// SCAL
void blasRscal (cublasHandle_t ch, const int32_t n, const float alpha, float* x, const int32_t incx);
void blasRscal (cublasHandle_t ch, const int32_t n, const double alpha, double* x, const int32_t incx);
// AXPY
void blasRaxpy (cublasHandle_t ch, const int32_t n, const float alpha, const float* x, const int32_t incx, float* y, const int32_t incy);
void blasRaxpy (cublasHandle_t ch, const int32_t n, const double alpha, const double* x, const int32_t incx, double* y, const int32_t incy);
// DOT
void blasRdot (cublasHandle_t ch, const int32_t n, const float* x, const int32_t incx, const float* y, const int32_t incy, float* ret);
void blasRdot (cublasHandle_t ch, const int32_t n, const double* x, const int32_t incx, const double* y, const int32_t incy, double* ret);
void blasRdotX (cublasHandle_t ch, const int32_t n, const float* x, const float* y, float* w, float* ret);
void blasRdotX (cublasHandle_t ch, const int32_t n, const double* x, const double* y, double* w, double* ret);
// NRM2
void blasRnrm2 (cublasHandle_t ch, const int32_t n, const float* x, const int32_t incx, float* ret);
void blasRnrm2 (cublasHandle_t ch, const int32_t n, const double* x, const int32_t incx, double* ret);
void blasRnrm2X (cublasHandle_t ch, const int32_t n, const float* x, float* w, float* ret);
void blasRnrm2X (cublasHandle_t ch, const int32_t n, const double* x, double* w, double* ret);
// GEMV
void blasRgemv (cublasHandle_t ch, const char trans, const int32_t m, const int32_t n, const float alpha, const float* A, const int32_t lda, const float* x, const int32_t incx, const float beta, float* y, const int32_t incy);
void blasRgemv (cublasHandle_t ch, const char trans, const int32_t m, const int32_t n, const double alpha, const double* A, const int32_t lda, const double* x, const int32_t incx, const double beta, double* y, const int32_t incy);
// GEMM
void blasRgemm (cublasHandle_t ch, const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const int32_t alpha, const int32_t* A, const int32_t lda, const int32_t* B, const int32_t ldb, const int32_t beta, int32_t* C, const int32_t ldc);
void blasRgemm (cublasHandle_t ch, const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const float alpha, const half* A, const int32_t lda, const half* B, const int32_t ldb, const float beta, float* C, const int32_t ldc);
void blasRgemm (cublasHandle_t ch, const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const float alpha, const float* A, const int32_t lda, const float* B, const int32_t ldb, const float beta, float* C, const int32_t ldc);
void blasRgemm (cublasHandle_t ch, const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const double alpha, const double* A, const int32_t lda, const double* B, const int32_t ldb, const double beta, double* C, const int32_t ldc);
// GEMM-BATCH
void blasRgemmBatch (cublasHandle_t ch, const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const int32_t alpha, const int32_t** A, const int32_t lda, const int32_t** B, const int32_t ldb, const int32_t beta, int32_t** C, const int32_t ldc, const int32_t grp, const int32_t cnt);
void blasRgemmBatch (cublasHandle_t ch, const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const float alpha, const half** A, const int32_t lda, const half** B, const int32_t ldb, const float beta, float** C, const int32_t ldc, const int32_t grp, const int32_t cnt);
void blasRgemmBatch (cublasHandle_t ch, const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const float alpha, const float** A, const int32_t lda, const float** B, const int32_t ldb, const float beta, float** C, const int32_t ldc, const int32_t grp, const int32_t cnt);
void blasRgemmBatch (cublasHandle_t ch, const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const double alpha, const double** A, const int32_t lda, const double** B, const int32_t ldb, const double beta, double** C, const int32_t ldc, const int32_t grp, const int32_t cnt);
// CSRMV
void blasRcsrmv (cusparseHandle_t ch, const char trans, const int32_t m, const int32_t n, const int32_t nnz, const float alpha, const cusparseMatDescr_t descrA, const float *A, const int32_t *devAcolind, const int32_t *devArowptr, const float *X, const float beta, float *Y);
void blasRcsrmv (cusparseHandle_t ch, const char trans, const int32_t m, const int32_t n, const int32_t nnz, const double alpha, const cusparseMatDescr_t descrA, const double *A, const int32_t *devAcolind, const int32_t *devArowptr, const double *X, const double beta, double *Y);
void blasRcsrmvX (cusparseHandle_t ch, const char trans, const int32_t m, const int32_t n, const int32_t nnz, const float alpha, const cusparseMatDescr_t descrA, const float *A, const int32_t *devAcolind, const int32_t *devArowptr, const float *X, const float beta, float *Y, const int32_t flag);
// CSRMM
void blasRcsrmvX (cusparseHandle_t ch, const char trans, const int32_t m, const int32_t n, const int32_t nnz, const double alpha, const cusparseMatDescr_t descrA, const double *A, const int32_t *devAcolind, const int32_t *devArowptr, const double *X, const double beta, double *Y, const int32_t flag);
// CSRMM
void blasRcsrmm (cusparseHandle_t ch, const char trans, const int32_t m, const int32_t n, const int32_t k, const int32_t nnz, const float alpha, const cusparseMatDescr_t descrA, const float *A, const int32_t *devAcolind, const int32_t *devArowptr, const float *B, const int32_t ldb, const float beta, float *C, const int32_t ldc);
void blasRcsrmm (cusparseHandle_t ch, const char trans, const int32_t m, const int32_t n, const int32_t k, const int32_t nnz, const double alpha, const cusparseMatDescr_t descrA, const double *A, const int32_t *devAcolind, const int32_t *devArowptr, const double *B, const int32_t ldb, const double beta, double *C, const int32_t ldc);
void blasRcsrmm_x (cusparseHandle_t ch, const char trans, const int32_t m, const int32_t n, const int32_t k, const int32_t nnz, const float alpha, const cusparseMatDescr_t descrA, const float *A, const int32_t *devAcolind, const int32_t *devArowptr, const float *B, const int32_t ldb, const float beta, float *C, const int32_t ldc);
void blasRcsrmm_x (cusparseHandle_t ch, const char trans, const int32_t m, const int32_t n, const int32_t k, const int32_t nnz, const double alpha, const cusparseMatDescr_t descrA, const double *A, const int32_t *devAcolind, const int32_t *devArowptr, const double *B, const int32_t ldb, const double beta, double *C, const int32_t ldc);
void blasRcsrmm_x2 (cusparseHandle_t ch, const char trans, const int32_t m, const int32_t n, const int32_t k, const int32_t nnz, const float alpha, const cusparseMatDescr_t descrA, const float *A, const int32_t *devAcolind, const int32_t *devArowptr, const float *B, const int32_t ldb, const float beta, float *C, const int32_t ldc);
void blasRcsrmm_x2 (cusparseHandle_t ch, const char trans, const int32_t m, const int32_t n, const int32_t k, const int32_t nnz, const double alpha, const cusparseMatDescr_t descrA, const double *A, const int32_t *devAcolind, const int32_t *devArowptr, const double *B, const int32_t ldb, const double beta, double *C, const int32_t ldc);
// OMATCOPY
void blasRomatcopy (cublasHandle_t ch, const char trans, const int32_t m, const int32_t n, const float* A, const int32_t lda, float* B, const int32_t ldb);
void blasRomatcopy (cublasHandle_t ch, const char trans, const int32_t m, const int32_t n, const double* A, const int32_t lda, double* B, const int32_t ldb);

//=============================================
// myBLAS 
//=============================================
template <typename TYPE1, typename TYPE2> int32_t blasRgemmSkinny_x2 ( cuozblasHandle_t *oh, const char transA, const char transB, const size_t m, const size_t n, const size_t k, const TYPE2 alpha, const TYPE1 *A, const size_t lda, const TYPE1 *B, const size_t ldb, const TYPE2 beta, TYPE2 *C, const size_t ldc);
template <typename TYPE1, typename TYPE2> int32_t blasRgemmSkinny ( cuozblasHandle_t *oh, const char transA, const char transB, const size_t m, const size_t n, const size_t k, const TYPE2 alpha, const TYPE1 *A, const size_t lda, const TYPE1 *B, const size_t ldb, const TYPE2 beta, TYPE2 *C, const size_t ldc);
#endif
