#include "ozblas_common.h"
#include "eft.h"
#include <immintrin.h>

//#define AVX256 // use AVX256 for both GEMM and CSRMV/M
#define MKL_CSR // if MKL's CSRMV is available

#if defined (AVX256)
#include "ozblas_blas_wrapper_avx256.hpp"
#endif
#if defined (AVX512)
#include "ozblas_blas_wrapper_avx512.hpp"
#endif

// =========================================
// BLAS Wrappers
// =========================================
// IAMAX
int32_t blasRiamax (const int32_t n, const float* x, const int32_t incx) {
	return cblas_isamax (n, x, incx);
}

int32_t blasRiamax (const int32_t n, const double* x, const int32_t incx) {
	return cblas_idamax (n, x, incx);
}

#if defined (FLOAT128)
int32_t blasRiamax (const int32_t n, const __float128* x, const int32_t incx) {
	#if defined (MPLAPACK)
	return iRamax (n, (__float128*)x, incx) - 1; // MPLAPACK uses 1-based index
	#else
	fprintf (OUTPUT, "OzBLAS error: iRamax (binary128) is not available. Use MPLAPACK.\n");
	exit(1);
	#endif
}
#endif

// ASUM
float blasRasum (const int32_t n, const float* x, const int32_t incx) {
	return cblas_sasum (n, x, incx);
}

double blasRasum (const int32_t n, const double* x, const int32_t incx) {
	return cblas_dasum (n, x, incx);
}

#if defined (FLOAT128)
__float128 blasRasum (const int32_t n, const __float128* x, const int32_t incx) {
	#if defined (MPLAPACK)
	return Rasum (n, (__float128*)x, incx);
	#else
	fprintf (OUTPUT, "OzBLAS error: Rasum (binary128) is not available. Use MPLAPACK.\n");
	exit(1);
	#endif
}
#endif

// SCAL
void blasRscal (const int32_t n, const float alpha, float* x, const int32_t incx) {
	cblas_sscal (n, alpha, x, incx);
}

void blasRscal (const int32_t n, const double alpha, double* x, const int32_t incx) {
	cblas_dscal (n, alpha, x, incx);
}

#if defined (FLOAT128)
void blasRscal (const int32_t n, const __float128 alpha, __float128* x, const int32_t incx) {
	#if defined (MPLAPACK)
	Rscal (n, alpha, (__float128*)x, incx);
	#else
	fprintf (OUTPUT, "OzBLAS error: Rscal (binary128) is not available. Use MPLAPACK.\n");
	exit(1);
	#endif
}
#endif

// AXPY
void blasRaxpy (const int32_t n, const float alpha, const float* x, const int32_t incx, float* y, const int32_t incy) {
	cblas_saxpy (n, alpha, x, incx, y, incy);
}

void blasRaxpy (const int32_t n, const double alpha, const double* x, const int32_t incx, double* y, const int32_t incy) {
	cblas_daxpy (n, alpha, x, incx, y, incy);
}

#if defined (FLOAT128)
void blasRaxpy (const int32_t n, const __float128 alpha, const __float128* x, const int32_t incx, __float128* y, const int32_t incy) {
	#if defined (MPLAPACK)
	Raxpy (n, alpha, (__float128*)x, incx, (__float128*)y, incy);
	#else
	fprintf (OUTPUT, "OzBLAS error: Raxpy (binary128) is not available. Use MPLAPACK.\n");
	exit(1);
	#endif
}
#endif

// GEMV
void blasRgemv (const char trans, const int32_t m, const int32_t n, const float alpha, const float* A, const int32_t lda, const float* x, const int32_t incx, const float beta, float* y, const int32_t incy) {
	cblas_sgemv (CblasColMajor, ToCblasOp(trans), m, n, alpha, A, lda, x, incx, beta, y, incy);
}

void blasRgemv (const char trans, const int32_t m, const int32_t n, const double alpha, const double* A, const int32_t lda, const double* x, const int32_t incx, const double beta, double* y, const int32_t incy) {
	cblas_dgemv (CblasColMajor, ToCblasOp(trans), m, n, alpha, A, lda, x, incx, beta, y, incy);
}

#if defined (FLOAT128)
void blasRgemv (const char trans, const int32_t m, const int32_t n, const __float128 alpha, const __float128* A, const int32_t lda, const __float128* x, const int32_t incx, const __float128 beta, __float128* y, const int32_t incy) {
	#if defined (MPLAPACK)
	Rgemv (&trans, m, n, alpha, (__float128*)A, lda, (__float128*)x, incx, beta, y, incy);
	#else
	fprintf (OUTPUT, "OzBLAS error: Rgemv (binary128) is not available. Use MPLAPACK.\n");
	exit(1);
	#endif
}
#endif

// GEMM
void dgemm_tn_skinny_x2 (
	const int32_t m,
	const int32_t n,
	const int32_t k,
	const double alpha,
	const double* A,
	const int32_t lda,
	const double* B,
	const int32_t ldb,
	const double beta,
	double* C,
	const int32_t ldc,
	const int32_t flag
	) {
	for (int32_t nn = 0; nn < n; nn++) {
		for (int32_t mm = 0; mm < m; mm++) {
			double2 Gr;
			Gr.x = 0.;
			Gr.y = 0.;
			#pragma omp parallel
			{
				double2 Lr;
				Lr.x = 0.;
				Lr.y = 0.;
				#pragma omp for
				for (int32_t kk = 0; kk < k; kk++) {
					double Ar = A[mm * lda + kk];
					double Br = B[nn * ldb + kk];
					Dot2i (Ar, Br, Lr);
				}
				#pragma omp critical
				{
					Gr = QuadAdd (Gr, Lr);
				}
			}
			if (beta == 0.)
				Gr = QuadMul (alpha, Gr);
			else
				Gr = QuadAdd (QuadMul (alpha, Gr), QuadMul (beta, C[nn * ldc + mm]));
			C[nn * ldc + mm] = Gr.x;
			if (flag) C[nn * ldc + mm + n * ldc] = Gr.y;
		}
	}
}

void blasRgemm (const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const float alpha, const float* A, const int32_t lda, const float* B, const int32_t ldb, const float beta, float* C, const int32_t ldc) {
	cblas_sgemm (CblasColMajor, ToCblasOp(transA), ToCblasOp(transB), m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void blasRgemm (const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const double alpha, const double* A, const int32_t lda, const double* B, const int32_t ldb, const double beta, double* C, const int32_t ldc) {
	if (transA == 't' && transB == 'n' && m < 10 && n < 10) {
		#if defined (AVX256)
		dgemm_tn_skinny_avx256 (m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
		#else
		cblas_dgemm (CblasColMajor, ToCblasOp(transA), ToCblasOp(transB), m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
		#endif
	} else {
		cblas_dgemm (CblasColMajor, ToCblasOp(transA), ToCblasOp(transB), m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	}
}

#if defined (FLOAT128)
void blasRgemm (const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const __float128 alpha, const __float128* A, const int32_t lda, const __float128* B, const int32_t ldb, const __float128 beta, __float128* C, const int32_t ldc) {
	#if defined (MPLAPACK)
	Rgemm (&transA, &transB, m, n, k, alpha, (__float128*)A, lda, (__float128*)B, ldb, beta, C, ldc);
	#else
	fprintf (OUTPUT, "OzBLAS error: Rgemm (binary128) is not available. Use MPLAPACK.\n");
	exit(1);
	#endif
}
#endif

void blasRgemm_x2 (const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const float alpha, const float* A, const int32_t lda, const float* B, const int32_t ldb, const float beta, float* C, const int32_t ldc) {
	fprintf (OUTPUT, "sgemm_x2 under construction.\n");
}

void blasRgemm_x2 (const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const double alpha, const double* A, const int32_t lda, const double* B, const int32_t ldb, const double beta, double* C, const int32_t ldc) {
	#if defined (AVX256)
	dgemm_tn_skinny_x2_avx256 (m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1);
	#else
	dgemm_tn_skinny_x2 (m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1);
	#endif
}

// CUDA-GEMM
#if defined (CUBLAS)
void cublasRgemm (cublasHandle_t handle, cublasOperation_t transA, cublasOperation_t transB, const int32_t m, const int32_t n, const int32_t k, const float* alpha, const float* A, const int32_t lda, const float* B, const int32_t ldb, const float* beta, float* C, const int32_t ldc) {
	cublasSgemm (handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void cublasRgemm (cublasHandle_t handle, cublasOperation_t transA, cublasOperation_t transB, const int32_t m, const int32_t n, const int32_t k, const double* alpha, const double* A, const int32_t lda, const double* B, const int32_t ldb, const double* beta, double* C, const int32_t ldc) {
	cublasDgemm (handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
#endif

// GEMM-BATCH
void blasRgemmBatch (const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const float alpha, const float** A, const int32_t lda, const float** B, const int32_t ldb, const float beta, float** C, const int32_t ldc, const int32_t grp, const int32_t cnt) {
	#if defined (MKL)
	CBLAS_TRANSPOSE transA_ = ToCblasOp (transA);
	CBLAS_TRANSPOSE transB_ = ToCblasOp (transB);
	cblas_sgemm_batch (CblasColMajor, &transA_, &transB_, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, grp, &cnt);
	#else
	fprintf (OUTPUT, "OzBLAS error: GEMM_BATCH is not available.\n");
	exit(1);
	#endif
}

void blasRgemmBatch (const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const double alpha, const double** A, const int32_t lda, const double** B, const int32_t ldb, const double beta, double** C, const int32_t ldc, const int32_t grp, const int32_t cnt) {
	#if defined (MKL)
	CBLAS_TRANSPOSE transA_ = ToCblasOp (transA);
	CBLAS_TRANSPOSE transB_ = ToCblasOp (transB);
	cblas_dgemm_batch (CblasColMajor, &transA_, &transB_, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, grp, &cnt);
	#else
	fprintf (OUTPUT, "OzBLAS error: GEMM_BATCH is not available.\n");
	exit(1);
	#endif
}
// --------------------------------------------------------------
// CSRMM
// --------------------------------------------------------------
// ==========================================
// General ==================================
// ==========================================
template <typename TYPE>
void csrmm_n (
	const int32_t m,
	const int32_t n,
	const int32_t k,
	const TYPE alpha,
	const TYPE* csrValA,
	const int32_t* csrColIndA,
	const int32_t* csrRowPtrA,
	const TYPE* B,
	const int32_t ldb,
	const TYPE beta,
	TYPE* C,
	const int32_t ldc
) {
	#pragma omp parallel for schedule (static) 
	for(int32_t rowid = 0; rowid < m; rowid++) {
		for (int32_t j = 0; j < n; j++) {
			TYPE Tr = 0.;
			for(int32_t i = csrRowPtrA[rowid]; i < csrRowPtrA[rowid+1]; i++) {
				TYPE Ar = csrValA[i];
				TYPE Br = B[j * ldb + csrColIndA[i]];
				Tr = fma1 (Ar, Br, Tr);
			}
			if (beta == 0.)
				C[j * ldc + rowid] = alpha * Tr;
			else
				C[j * ldc + rowid] = fma1 (alpha, Tr, (beta * C[j * ldc + rowid]));
		}
	}
}

void csrmm_n_x2 (
	const int32_t m,
	const int32_t n,
	const int32_t k,
	const double alpha,
	const double* csrValA,
	const int32_t* csrColIndA,
	const int32_t* csrRowPtrA,
	const double* B,
	const int32_t ldb,
	const double beta,
	double* C,
	const int32_t ldc,
	const int32_t flag
) {
	#pragma omp parallel for schedule (static) 
	for(int32_t rowid = 0; rowid < m; rowid++) {
		for (int32_t j = 0; j < n; j++) {
			double2 Tr;
			Tr.x = Tr.y = 0.;
			for(int32_t i = csrRowPtrA[rowid]; i < csrRowPtrA[rowid+1]; i++) {
				double Ar = csrValA[i];
				double Br = B[j * ldb + csrColIndA[i]];
				Dot2i (Ar, Br, Tr);
			}
			if (beta == 0.)
				Tr = QuadMul (alpha, Tr);
			else
				Tr = QuadAdd (QuadMul (alpha, Tr), QuadMul (beta, C[j * ldc + rowid]));
			C[j * ldc + rowid] = Tr.x;
			if (flag) C[j * ldc + rowid + ldc * n] = Tr.y;
		}
	}
}

// ==========================================
// L=2 ======================================
// ==========================================
#define LL 2
void csrmm_n_l2 (
	const int32_t m,
	const double alpha,
	const double* csrValA,
	const int32_t* csrColIndA,
	const int32_t* csrRowPtrA,
	const double* B,
	const int32_t ldb,
	const double beta,
	double* C,
	const int32_t ldc
) {
	#pragma omp parallel for
	for(int32_t rowid = 0; rowid < m; rowid++) {
		double Tr[LL];
		//#pragma unroll 
		for (int32_t j = 0; j < LL; j++) 
			Tr[j] = 0.;
		for(int32_t i = csrRowPtrA[rowid]; i < csrRowPtrA[rowid+1]; i++) {
			double Ar = csrValA[i];
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				double Br = B[j * ldb + csrColIndA[i]];
				Tr[j] = fma1 (Ar, Br, Tr[j]);
			}
		}
		if (beta == 0.) {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) 
				C[j * ldc + rowid] = alpha * Tr[j];
		} else {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) 
				C[j * ldc + rowid] = fma1 (alpha, Tr[j], (beta * C[j * ldc + rowid]));
		}
	}
}

void csrmm_n_x2_l2 (
	const int32_t m,
	const double alpha,
	const double* csrValA,
	const int32_t* csrColIndA,
	const int32_t* csrRowPtrA,
	const double* B,
	const int32_t ldb,
	const double beta,
	double* C,
	const int32_t ldc
) {
	#pragma omp parallel for
	for(int32_t rowid = 0; rowid < m; rowid++) {
		double2 Tr[LL];
		//#pragma unroll
		for (int32_t j = 0; j < LL; j++) 
			Tr[j].x = Tr[j].y = 0.;
		for(int32_t i = csrRowPtrA[rowid]; i < csrRowPtrA[rowid+1]; i++) {
			double Ar = csrValA[i];
			//#pragma unroll
			for (int32_t j = 0; j < LL; j++) {
				double Br = B[j * ldb + csrColIndA[i]];
				Dot2i (Ar, Br, Tr[j]);
			}
		}
		if (beta == 0.) {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				Tr[j] = QuadMul (alpha, Tr[j]);
				C[j * ldc + rowid] = Tr[j].x;
				C[j * ldc + rowid + ldc * LL] = Tr[j].y;
			}
		} else {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				Tr[j] = QuadAdd (QuadMul (alpha, Tr[j]), QuadMul (beta, C[j * ldc + rowid]));
				C[j * ldc + rowid] = Tr[j].x;
				C[j * ldc + rowid + ldc * LL] = Tr[j].y;
			}
		}
	}
}
#undef LL

// ==========================================
// L=3 ======================================
// ==========================================
#define LL 3
void csrmm_n_l3 (
	const int32_t m,
	const double alpha,
	const double* csrValA,
	const int32_t* csrColIndA,
	const int32_t* csrRowPtrA,
	const double* B,
	const int32_t ldb,
	const double beta,
	double* C,
	const int32_t ldc
) {
	#pragma omp parallel for
	for(int32_t rowid = 0; rowid < m; rowid++) {
		double Tr[LL];
		//#pragma unroll 
		for (int32_t j = 0; j < LL; j++) 
			Tr[j] = 0.;
		for(int32_t i = csrRowPtrA[rowid]; i < csrRowPtrA[rowid+1]; i++) {
			double Ar = csrValA[i];
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				double Br = B[j * ldb + csrColIndA[i]];
				Tr[j] = fma1 (Ar, Br, Tr[j]);
			}
		}
		//#pragma unroll 
		for (int32_t j = 0; j < 3; j++) {
			C[j * ldc + rowid] = fma1 (alpha, Tr[j], (beta * C[j * ldc + rowid]));
		}
	}
}

void csrmm_n_x2_l3 (
	const int32_t m,
	const double alpha,
	const double* csrValA,
	const int32_t* csrColIndA,
	const int32_t* csrRowPtrA,
	const double* B,
	const int32_t ldb,
	const double beta,
	double* C,
	const int32_t ldc
) {
	#pragma omp parallel for
	for(int32_t rowid = 0; rowid < m; rowid++) {
		double2 Tr[LL];
		//#pragma unroll
		for (int32_t j = 0; j < LL; j++) 
			Tr[j].x = Tr[j].y = 0.;
		for(int32_t i = csrRowPtrA[rowid]; i < csrRowPtrA[rowid+1]; i++) {
			double Ar = csrValA[i];
			//#pragma unroll
			for (int32_t j = 0; j < LL; j++) {
				double Br = B[j * ldb + csrColIndA[i]];
				Dot2i (Ar, Br, Tr[j]);
			}
		}
		if (beta == 0.) {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				Tr[j] = QuadMul (alpha, Tr[j]);
				C[j * ldc + rowid] = Tr[j].x;
				C[j * ldc + rowid + ldc * LL] = Tr[j].y;
			}
		} else {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				Tr[j] = QuadAdd (QuadMul (alpha, Tr[j]), QuadMul (beta, C[j * ldc + rowid]));
				C[j * ldc + rowid] = Tr[j].x;
				C[j * ldc + rowid + ldc * LL] = Tr[j].y;
			}
		}
	}
}
#undef LL

// ==========================================
// L=4 ======================================
// ==========================================
#define LL 4
void csrmm_n_l4 (
	const int32_t m,
	const double alpha,
	const double* csrValA,
	const int32_t* csrColIndA,
	const int32_t* csrRowPtrA,
	const double* B,
	const int32_t ldb,
	const double beta,
	double* C,
	const int32_t ldc
) {
	#pragma omp parallel for
	for(int32_t rowid = 0; rowid < m; rowid++) {
		double Tr[LL];
		//#pragma unroll 
		for (int32_t j = 0; j < LL; j++) 
			Tr[j] = 0.;
		for(int32_t i = csrRowPtrA[rowid]; i < csrRowPtrA[rowid+1]; i++) {
			double Ar = csrValA[i];
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				double Br = B[j * ldb + csrColIndA[i]];
				Tr[j] = fma1 (Ar, Br, Tr[j]);
			}
		}
		if (beta == 0.) {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) 
				C[j * ldc + rowid] = alpha * Tr[j];
		} else {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) 
				C[j * ldc + rowid] = fma1 (alpha, Tr[j], (beta * C[j * ldc + rowid]));
		}
	}
}

void csrmm_n_x2_l4 (
	const int32_t m,
	const double alpha,
	const double* csrValA,
	const int32_t* csrColIndA,
	const int32_t* csrRowPtrA,
	const double* B,
	const int32_t ldb,
	const double beta,
	double* C,
	const int32_t ldc
) {
	#pragma omp parallel for
	for(int32_t rowid = 0; rowid < m; rowid++) {
		double2 Tr[LL];
		//#pragma unroll
		for (int32_t j = 0; j < LL; j++) 
			Tr[j].x = Tr[j].y = 0.;
		for(int32_t i = csrRowPtrA[rowid]; i < csrRowPtrA[rowid+1]; i++) {
			double Ar = csrValA[i];
			//#pragma unroll
			for (int32_t j = 0; j < LL; j++) {
				double Br = B[j * ldb + csrColIndA[i]];
				Dot2i (Ar, Br, Tr[j]);
			}
		}
		if (beta == 0.) {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				Tr[j] = QuadMul (alpha, Tr[j]);
				C[j * ldc + rowid] = Tr[j].x;
				C[j * ldc + rowid + ldc * LL] = Tr[j].y;
			}
		} else {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				Tr[j] = QuadAdd (QuadMul (alpha, Tr[j]), QuadMul (beta, C[j * ldc + rowid]));
				C[j * ldc + rowid] = Tr[j].x;
				C[j * ldc + rowid + ldc * LL] = Tr[j].y;
			}
		}
	}
}
#undef LL

// ==========================================
// L=5 ======================================
// ==========================================
#define LL 5
void csrmm_n_l5 (
	const int32_t m,
	const double alpha,
	const double* csrValA,
	const int32_t* csrColIndA,
	const int32_t* csrRowPtrA,
	const double* B,
	const int32_t ldb,
	const double beta,
	double* C,
	const int32_t ldc
) {
	#pragma omp parallel for
	for(int32_t rowid = 0; rowid < m; rowid++) {
		double Tr[LL];
		//#pragma unroll 
		for (int32_t j = 0; j < LL; j++) 
			Tr[j] = 0.;
		for(int32_t i = csrRowPtrA[rowid]; i < csrRowPtrA[rowid+1]; i++) {
			double Ar = csrValA[i];
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				double Br = B[j * ldb + csrColIndA[i]];
				Tr[j] = fma1 (Ar, Br, Tr[j]);
			}
		}
		if (beta == 0.) {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) 
				C[j * ldc + rowid] = alpha * Tr[j];
		} else {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) 
				C[j * ldc + rowid] = fma1 (alpha, Tr[j], (beta * C[j * ldc + rowid]));
		}
	}
}

void csrmm_n_x2_l5 (
	const int32_t m,
	const double alpha,
	const double* csrValA,
	const int32_t* csrColIndA,
	const int32_t* csrRowPtrA,
	const double* B,
	const int32_t ldb,
	const double beta,
	double* C,
	const int32_t ldc
) {
	#pragma omp parallel for
	for(int32_t rowid = 0; rowid < m; rowid++) {
		double2 Tr[LL];
		//#pragma unroll
		for (int32_t j = 0; j < LL; j++) 
			Tr[j].x = Tr[j].y = 0.;
		for(int32_t i = csrRowPtrA[rowid]; i < csrRowPtrA[rowid+1]; i++) {
			double Ar = csrValA[i];
			//#pragma unroll
			for (int32_t j = 0; j < LL; j++) {
				double Br = B[j * ldb + csrColIndA[i]];
				Dot2i (Ar, Br, Tr[j]);
			}
		}
		if (beta == 0.) {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				Tr[j] = QuadMul (alpha, Tr[j]);
				C[j * ldc + rowid] = Tr[j].x;
				C[j * ldc + rowid + ldc * LL] = Tr[j].y;
			}
		} else {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				Tr[j] = QuadAdd (QuadMul (alpha, Tr[j]), QuadMul (beta, C[j * ldc + rowid]));
				C[j * ldc + rowid] = Tr[j].x;
				C[j * ldc + rowid + ldc * LL] = Tr[j].y;
			}
		}
	}
}
#undef LL

void blasRcsrmm (const char trans, const int32_t m, const int32_t n, const int32_t k, const int32_t nnz, const float alpha, const char *descrA, const float *A, const int32_t *devAcolind, const int32_t *devArowptr, const float *B, const int32_t ldb, const float beta, float *C, const int32_t ldc) {
	csrmm_n (m, n, k, alpha, A, devAcolind, devArowptr, B, ldb, beta, C, ldc);
	//fprintf (OUTPUT, "OzBLAS warning: CSRMM not implemented.\n");
}

void blasRcsrmm (const char trans, const int32_t m, const int32_t n, const int32_t k, const int32_t nnz, const double alpha, const char *descrA, const double *A, const int32_t *devAcolind, const int32_t *devArowptr, const double *B, const int32_t ldb, const double beta, double *C, const int32_t ldc) {
	#if defined (MKL) && defined (MKL_CSR)
	int32_t expected_calls = 1;
	struct matrix_descr descrA_matrix;
	sparse_matrix_t csrA;
	mkl_sparse_d_create_csr (&csrA, SPARSE_INDEX_BASE_ZERO, m, k, (int32_t*)devArowptr, (int32_t*)devArowptr+1, (int32_t*)devAcolind, (double*)A);
	descrA_matrix.type = SPARSE_MATRIX_TYPE_GENERAL;//SPARSE_MATRIX_TYPE_SYMMETRIC;
	mkl_sparse_set_mm_hint (csrA, SPARSE_OPERATION_NON_TRANSPOSE, descrA_matrix, SPARSE_LAYOUT_COLUMN_MAJOR, n, expected_calls);
	mkl_sparse_d_mm (SPARSE_OPERATION_NON_TRANSPOSE, alpha, csrA, descrA_matrix, SPARSE_LAYOUT_COLUMN_MAJOR, B, n, ldb, beta, C, ldc);
	mkl_sparse_destroy (csrA);
	#elif defined (AVX256)
	switch (n) {
		case 2: 
			csrmm_n_avx256_l2 (m, alpha, A, devAcolind, devArowptr, B, ldb, beta, C, ldc);
			break;
		case 3: 
			csrmm_n_avx256_l3 (m, alpha, A, devAcolind, devArowptr, B, ldb, beta, C, ldc);
			break;
		case 4: 
			csrmm_n_avx256_l4 (m, alpha, A, devAcolind, devArowptr, B, ldb, beta, C, ldc);
			break;
		case 5: 
			csrmm_n_avx256_l5 (m, alpha, A, devAcolind, devArowptr, B, ldb, beta, C, ldc);
			break;
		default:
			csrmm_n_avx256 (m, n, k, alpha, A, devAcolind, devArowptr, B, ldb, beta, C, ldc);
	}
	#else
	switch (n) {
		case 2: 
			csrmm_n_l2 (m, alpha, A, devAcolind, devArowptr, B, ldb, beta, C, ldc);
			break;
		case 3: 
			csrmm_n_l3 (m, alpha, A, devAcolind, devArowptr, B, ldb, beta, C, ldc);
			break;
		case 4: 
			csrmm_n_l4 (m, alpha, A, devAcolind, devArowptr, B, ldb, beta, C, ldc);
			break;
		case 5: 
			csrmm_n_l5 (m, alpha, A, devAcolind, devArowptr, B, ldb, beta, C, ldc);
			break;
		default:
			csrmm_n (m, n, k, alpha, A, devAcolind, devArowptr, B, ldb, beta, C, ldc);
	}
	#endif
}

void blasRcsrmm_x2 (const char trans, const int32_t m, const int32_t n, const int32_t k, const int32_t nnz, const float alpha, const char *descrA, const float *A, const int32_t *devAcolind, const int32_t *devArowptr, const float *B, const int32_t ldb, const float beta, float *C, const int32_t ldc) {
	fprintf (OUTPUT, "csrmm_x2 under construction.\n");
}

void blasRcsrmm_x2 (const char trans, const int32_t m, const int32_t n, const int32_t k, const int32_t nnz, const double alpha, const char *descrA, const double *A, const int32_t *devAcolind, const int32_t *devArowptr, const double *B, const int32_t ldb, const double beta, double *C, const int32_t ldc) {
	#if defined (AVX256)
	switch (n) {
		case 2: 
			csrmm_n_x2_avx256_l2 (m, alpha, A, devAcolind, devArowptr, B, ldb, beta, C, ldc);
			break;
		case 3: 
			csrmm_n_x2_avx256_l3 (m, alpha, A, devAcolind, devArowptr, B, ldb, beta, C, ldc);
			break;
		case 4: 
			csrmm_n_x2_avx256_l4 (m, alpha, A, devAcolind, devArowptr, B, ldb, beta, C, ldc);
			break;
		case 5: 
			csrmm_n_x2_avx256_l5 (m, alpha, A, devAcolind, devArowptr, B, ldb, beta, C, ldc);
			break;
		default:
			csrmm_n_x2_avx256 (m, n, k, alpha, A, devAcolind, devArowptr, B, ldb, beta, C, ldc, 1);
	}
	#elif defined (AVX512)
	switch (n) {
		case 2: 
			csrmm_n_x2_avx512_l2 (m, alpha, A, devAcolind, devArowptr, B, ldb, beta, C, ldc);
			break;
		case 3: 
			csrmm_n_x2_avx512_l3 (m, alpha, A, devAcolind, devArowptr, B, ldb, beta, C, ldc);
			break;
		case 4: 
			csrmm_n_x2_avx512_l4 (m, alpha, A, devAcolind, devArowptr, B, ldb, beta, C, ldc);
			break;
		case 5: 
			csrmm_n_x2_avx512_l5 (m, alpha, A, devAcolind, devArowptr, B, ldb, beta, C, ldc);
			break;
		default:
			csrmm_n_x2_avx512 (m, n, k, alpha, A, devAcolind, devArowptr, B, ldb, beta, C, ldc, 1);
	}
	#else
	switch (n) {
		case 2: 
			csrmm_n_x2_l2 (m, alpha, A, devAcolind, devArowptr, B, ldb, beta, C, ldc);
			break;
		case 3: 
			csrmm_n_x2_l3 (m, alpha, A, devAcolind, devArowptr, B, ldb, beta, C, ldc);
			break;
		case 4: 
			csrmm_n_x2_l4 (m, alpha, A, devAcolind, devArowptr, B, ldb, beta, C, ldc);
			break;
		case 5: 
			csrmm_n_x2_l5 (m, alpha, A, devAcolind, devArowptr, B, ldb, beta, C, ldc);
			break;
		default:
			csrmm_n_x2 (m, n, k, alpha, A, devAcolind, devArowptr, B, ldb, beta, C, ldc, 1);
	}
	#endif
}

// --------------------------------------------------------------
// CSRMV
// --------------------------------------------------------------
template <typename TYPE>
void csrmv (
	const int32_t m,
	const int32_t n,
	const TYPE alpha,
	const TYPE* csrValA,
	const int32_t* csrColIndA,
	const int32_t* csrRowPtrA,
	const TYPE* x,
	const TYPE beta,
	TYPE* y
) {
	#pragma omp parallel for
	for(int32_t rowid = 0; rowid < m; rowid++) {
		TYPE Tr = 0.;
		for(int32_t i = csrRowPtrA[rowid]; i < csrRowPtrA[rowid+1]; i++) {
			TYPE Ar = csrValA[i];
			TYPE Br = x[csrColIndA[i]];
			Tr = fma1 (Ar, Br, Tr);
		}
		if (beta == 0.)
			y[rowid] = alpha * Tr;
		else
			y[rowid] = fma1 (alpha, Tr, (beta * y[rowid]));
	}
}
#if defined (FLOAT128)
template void csrmv (const int32_t m, const int32_t n, const __float128 alpha, const __float128* matA, const int32_t* matAind, const int32_t* matAptr, const __float128* x, const __float128 beta, __float128* y);
#endif
template void csrmv (const int32_t m, const int32_t n, const double alpha, const double* matA, const int32_t* matAind, const int32_t* matAptr, const double* x, const double beta, double* y);
template void csrmv (const int32_t m, const int32_t n, const float alpha, const float* matA, const int32_t* matAind, const int32_t* matAptr, const float* x, const float beta, float* y);

void blasRcsrmv (const char trans, const int32_t m, const int32_t n, const int32_t nnz, const float alpha, const char *descrA, const float *A, const int32_t *devAcolind, const int32_t *devArowptr, const float *X, const float beta, float *Y) {
	#if defined (MKL) && defined (MKL_CSR)
	int32_t expected_calls = 1;
	struct matrix_descr descrA_matrix;
	sparse_matrix_t csrA;
	mkl_sparse_s_create_csr (&csrA, SPARSE_INDEX_BASE_ZERO, m, n, (int32_t*)devArowptr, (int32_t*)devArowptr+1, (int32_t*)devAcolind, (float*)A);
	descrA_matrix.type = SPARSE_MATRIX_TYPE_GENERAL;//SPARSE_MATRIX_TYPE_SYMMETRIC;
	mkl_sparse_set_mv_hint (csrA, SPARSE_OPERATION_NON_TRANSPOSE, descrA_matrix, expected_calls);
	mkl_sparse_s_mv (SPARSE_OPERATION_NON_TRANSPOSE, alpha, csrA, descrA_matrix, X, beta, Y);
	mkl_sparse_destroy (csrA);
	#else
	fprintf (OUTPUT, "OzBLAS warning: in-house CSRMV is used.\n");
	csrmv (m, n, alpha, A, devAcolind, devArowptr, X, beta, Y);
	#endif
}

void blasRcsrmv (const char trans, const int32_t m, const int32_t n, const int32_t nnz, const double alpha, const char *descrA, const double *A, const int32_t *devAcolind, const int32_t *devArowptr, const double *X, const double beta, double *Y) {
	#if defined (MKL) && defined (MKL_CSR)
	int32_t expected_calls = 1;
	struct matrix_descr descrA_matrix;
	sparse_matrix_t csrA;
	mkl_sparse_d_create_csr (&csrA, SPARSE_INDEX_BASE_ZERO, m, n, (int32_t*)devArowptr, (int32_t*)devArowptr+1, (int32_t*)devAcolind, (double*)A);
	descrA_matrix.type = SPARSE_MATRIX_TYPE_GENERAL;//SPARSE_MATRIX_TYPE_SYMMETRIC;
	mkl_sparse_set_mv_hint (csrA, SPARSE_OPERATION_NON_TRANSPOSE, descrA_matrix, expected_calls);
	mkl_sparse_d_mv (SPARSE_OPERATION_NON_TRANSPOSE, alpha, csrA, descrA_matrix, X, beta, Y);
	mkl_sparse_destroy (csrA);
	#elif defined (AVX256)
	csrmm_n_avx256 (m, 1, n, alpha, A, devAcolind, devArowptr, X, n, beta, Y, m);
	#elif defined (AVX512)
	csrmm_n_avx512 (m, 1, n, alpha, A, devAcolind, devArowptr, X, n, beta, Y, m);
	#else
	csrmv (m, n, alpha, A, devAcolind, devArowptr, X, beta, Y);
	#endif
}

void blasRcsrmvX (const char trans, const int32_t m, const int32_t n, const int32_t nnz, const float alpha, const char *descrA, const float *A, const int32_t *devAcolind, const int32_t *devArowptr, const float *X, const float beta, float *Y) {
	fprintf (OUTPUT, "blasRcsrmvX is not available.\n");
	exit(1);
}

void blasRcsrmvX (const char trans, const int32_t m, const int32_t n, const int32_t nnz, const double alpha, const char *descrA, const double *A, const int32_t *devAcolind, const int32_t *devArowptr, const double *X, const double beta, double *Y) {
	#if defined (AVX256)
	csrmm_n_x2_avx256 (m, 1, n, alpha, A, devAcolind, devArowptr, X, n, beta, Y, m, 0);
	#elif defined (AVX512)
	csrmm_n_x2_avx512 (m, 1, n, alpha, A, devAcolind, devArowptr, X, n, beta, Y, m, 0);
	#else
	csrmm_n_x2 (m, 1, n, alpha, A, devAcolind, devArowptr, X, n, beta, Y, m, 0);
	#endif
}

#if defined (FLOAT128)
void blasRcsrmv (const char trans, const int32_t m, const int32_t n, const int32_t nnz, const __float128 alpha, const char *descrA, const __float128 *A, const int32_t *devAcolind, const int32_t *devArowptr, const __float128 *X, const __float128 beta, __float128 *Y) {
	csrmv (m, n, alpha, A, devAcolind, devArowptr, X, beta, Y);
}

void blasRcsrmvX (const char trans, const int32_t m, const int32_t n, const int32_t nnz, const __float128 alpha, const char *descrA, const __float128 *A, const int32_t *devAcolind, const int32_t *devArowptr, const __float128 *X, const __float128 beta, __float128 *Y) {
	fprintf (OUTPUT, "blasRcsrmvX is not available.\n");
	exit(1);
}
#endif


// --------------------------------------------------------------
// OMATCOPY
// --------------------------------------------------------------
void blasRomatcopy (const char trans, const int32_t m, const int32_t n, const float* A, const int32_t lda, float* B, const int32_t ldb) {
	#if defined (MKL)
	mkl_somatcopy ('c', trans, m, n, 1., A, lda, B, ldb);
	#elif defined (SSL2)
	fprintf (OUTPUT, "OzBLAS error: omatcopy is not available.\n");
	exit(1);
	#else
	cblas_somatcopy (CblasColMajor, ToCblasOp(trans), m, n, 1., A, lda, B, ldb);
	#endif
}

void blasRomatcopy (const char trans, const int32_t m, const int32_t n, const double* A, const int32_t lda, double* B, const int32_t ldb) {
	#if defined (MKL)
	mkl_domatcopy ('c', trans, m, n, 1., A, lda, B, ldb);
	#elif defined (SSL2)
	fprintf (OUTPUT, "OzBLAS error: omatcopy is not available.\n");
	exit(1);
	#else
	cblas_domatcopy (CblasColMajor, ToCblasOp(trans), m, n, 1., A, lda, B, ldb);
	#endif
}

#if defined (FLOAT128)
#include <complex>
void blasRomatcopy (const char trans, const int32_t m, const int32_t n, const __float128* A, const int32_t lda, __float128* B, const int32_t ldb) {
	#if defined (MKL)
	MKL_Complex16 zone;
	zone.real = 1.;
	zone.imag = 0.;
	mkl_zomatcopy ('c', trans, m, n, zone, (MKL_Complex16*)A, lda, (MKL_Complex16*)B, ldb);
	#elif defined (SSL2)
	fprintf (OUTPUT, "OzBLAS error: omatcopy is not available.\n");
	exit(1);
	#else
	const double done = 1.;
	cblas_zomatcopy (CblasColMajor, ToCblasOp(trans), m, n, &done, (const double*)A, lda, (double*)B, ldb);
	#endif
}
#endif

// DOT
float blasRdot (const int32_t n, const float* x, const int32_t incx, const float* y, const int32_t incy) {
	return cblas_sdot (n, x, incx, y, incy);
}

float blasRdotX (const int32_t n, const float* x, const int32_t incx, const float* y, const int32_t incy) {
	fprintf (OUTPUT, "blasRdotX is not available.\n");
	exit(1);
}

double blasRdot (const int32_t n, const double* x, const int32_t incx, const double* y, const int32_t incy) {
	return cblas_ddot (n, x, incx, y, incy);
}

__float128 blasRdotX (const int32_t n, const __float128* x, const int32_t incx, const __float128* y, const int32_t incy) {
	fprintf (OUTPUT, "blasRdotX is not available.\n");
	exit(1);
}

double blasRdotX (const int32_t n, const double* x, const int32_t incx, const double* y, const int32_t incy) {
	double ret;
	#if defined (AVX256)
	dgemm_tn_skinny_x2_avx256 (1, 1, n, 1., x, 0, y, 0, 0., &ret, 0, 0);
	#else
	dgemm_tn_skinny_x2 (1, 1, n, 1., x, 0, y, 0, 0., &ret, 0, 0);
	#endif
	return ret;
}

#if defined (FLOAT128)
__float128 blasRdot (const int32_t n, const __float128* x, const int32_t incx, const __float128* y, const int32_t incy) {
	#if defined (MPLAPACK)
	return Rdot (n, (__float128*)x, incx, (__float128*)y, incy);
	#else
	fprintf (OUTPUT, "OzBLAS error: Rdot (binary128) is not available. Use MPLAPACK.\n");
	exit(1);
	#endif
}
#endif

// NRM2
float blasRnrm2 (const int32_t n, const float* x, const int32_t incx) {
	return cblas_snrm2 (n, x, incx);
}

double blasRnrm2 (const int32_t n, const double* x, const int32_t incx) {
	return cblas_dnrm2 (n, x, incx);
}

float blasRnrm2X (const int32_t n, const float* x, const int32_t incx) {
	fprintf (OUTPUT, "blasRnrm2X is not available.\n");
	exit(1);
}

double blasRnrm2X (const int32_t n, const double* x, const int32_t incx) {
	return sqrt (blasRdotX (n, x, 1, x, 1));
}

__float128 blasRnrm2X (const int32_t n, const __float128* x, const int32_t incx) {
	fprintf (OUTPUT, "blasRnrm2X is not available.\n");
	exit(1);
}

#if defined (FLOAT128)
__float128 blasRnrm2 (const int32_t n, const __float128* x, const int32_t incx) {
	#if defined (MPLAPACK)
	return Rnrm2 (n, (__float128*)x, incx);
	#else
	fprintf (OUTPUT, "OzBLAS error: Rnrm2 (binary128) is not available. Use MPLAPACK.\n");
	exit(1);
	#endif
}
#endif


