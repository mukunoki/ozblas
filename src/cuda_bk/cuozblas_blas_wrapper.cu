#include "cuozblas_common.h"
#include "cueft.h"
#define NTX 512
#define DOT_NBX 512
//#define CUSPARSE_CSR // CSRMV is available only in old cuSparse 

// =========================================
// BLAS Wrappers
// =========================================
// IAMAX
void blasRiamax (cublasHandle_t ch, const int32_t n, const float* x, const int32_t incx, int32_t* ret) {
	cublasIsamax (ch, n, x, incx, ret);
}
void blasRiamax (cublasHandle_t ch, const int32_t n, const double* x, const int32_t incx, int32_t* ret) {
	cublasIdamax (ch, n, x, incx, ret);
}
// ASUM
void blasRasum (cublasHandle_t ch, const int32_t n, const float* x, const int32_t incx, float* ret) {
	cublasSasum (ch, n, x, incx, ret);
}
void blasRasum (cublasHandle_t ch, const int32_t n, const double* x, const int32_t incx, double* ret) {
	cublasDasum (ch, n, x, incx, ret);
}
// SCAL
void blasRscal (cublasHandle_t ch, const int32_t n, const float alpha, float* x, const int32_t incx) {
	cublasSscal (ch, n, &alpha, x, incx);
}
void blasRscal (cublasHandle_t ch, const int32_t n, const double alpha, double* x, const int32_t incx) {
	cublasDscal (ch, n, &alpha, x, incx);
}
// AXPY
void blasRaxpy (cublasHandle_t ch, const int32_t n, const float alpha, const float* x, const int32_t incx, float* y, const int32_t incy) {
	cublasSaxpy (ch, n, &alpha, x, incx, y, incy);
}
void blasRaxpy (cublasHandle_t ch, const int32_t n, const double alpha, const double* x, const int32_t incx, double* y, const int32_t incy) {
	cublasDaxpy (ch, n, &alpha, x, incx, y, incy);
}
// DOT
__global__ void
sum_x_kernel (
	const size_t n,
	double2 *w
) {
	const size_t iTx = threadIdx.x;
	const size_t iBx = blockIdx.x;
	const size_t nBx = gridDim.x;
	const size_t nTx = blockDim.x;
	const size_t addrx = iBx * nTx + iTx;
	size_t i;
	__shared__ double2 Ts2[NTX];
	double2 Tr2 = make_double2 (0., 0.);

	for (i = addrx; i < n; i += nTx * nBx) 
		Tr2 = cuQuadAdd (Tr2, w[i]);
	Ts2[iTx] = Tr2;
	__syncthreads ();

	#pragma unroll
	for (i = nTx/2; i > 0; i >>= 1) {
		if (iTx < i) 
			Ts2[iTx] = cuQuadAdd (Ts2[iTx], Ts2[iTx+i]);
		__syncthreads ();
	}
	if (iTx == 0 && iBx == 0) 
		w[0].x = Ts2[0].x + Ts2[0].y;
}

__global__ void
dotX_kernel (
	const size_t n,
	const double * __restrict__ x,
	const double * __restrict__ y,
	double2 *w
) {
	const size_t iTx = threadIdx.x;
	const size_t iBx = blockIdx.x;
	const size_t nBx = gridDim.x;
	const size_t nTx = blockDim.x;
	const size_t addrx = iBx * nTx + iTx;
	size_t i;
	register double2 Tr2 = make_double2 (0., 0.);
	__shared__ double2 Ts2[NTX];

	// DOT part -----------------------------------------------
	for (i = addrx; i < n; i += nTx * nBx) 
		cuDot2i (x[i], y[i], Tr2);
	Ts2[iTx] = Tr2;
	__syncthreads ();

	// SUM part -----------------------------------------------
	#pragma unroll
	for (i = nTx/2; i > 0; i >>= 1) {
		if (iTx < i) 
			Ts2[iTx] = cuQuadAdd (Ts2[iTx], Ts2[iTx+i]);
		__syncthreads ();
	}
	if (iTx == 0) 
		w[iBx] = Ts2[0];
}

__host__
void blasDdotX (cublasHandle_t ch, const int32_t n, const double* x, const double* y, double* w, double* ret) {
	size_t ntx = NTX;
	size_t nbx = min ((int)ceil((float)n/ntx), DOT_NBX);
	dotX_kernel <<< dim3(nbx), dim3(ntx) >>> (n, x, y, (double2*)w);
	sum_x_kernel <<< dim3(1), dim3(ntx) >>> (nbx, (double2*)w);
	cudaMemcpy (ret, (double*)w, sizeof(double), cudaMemcpyDeviceToHost);
}

void blasRdot (cublasHandle_t ch, const int32_t n, const float* x, const int32_t incx, const float* y, const int32_t incy, float* ret) {
	cublasSdot (ch, n, x, incx, y, incy, ret);
}
void blasRdot (cublasHandle_t ch, const int32_t n, const double* x, const int32_t incx, const double* y, const int32_t incy, double* ret) {
	cublasDdot (ch, n, x, incx, y, incy, ret);
}
void blasRdotX (cublasHandle_t ch, const int32_t n, const float* x, const float* y, float* w,float* ret) {
	fprintf (OUTPUT, "OzBLAS error: blasRdotX is not available.\n");
	exit(1);
}
void blasRdotX (cublasHandle_t ch, const int32_t n, const double* x, const double* y, double* w, double* ret) {
	blasDdotX (ch, n, x, y, w, ret);
}
// NRM2
void blasRnrm2 (cublasHandle_t ch, const int32_t n, const float* x, const int32_t incx, float* ret) {
	cublasSnrm2 (ch, n, x, incx, ret);
}
void blasRnrm2 (cublasHandle_t ch, const int32_t n, const double* x, const int32_t incx, double* ret) {
	cublasDnrm2 (ch, n, x, incx, ret);
}
void blasRnrm2X (cublasHandle_t ch, const int32_t n, const float* x, float* w, float* ret) {
	fprintf (OUTPUT, "OzBLAS error: blasRnrm2X is not available.\n");
	exit(1);
}
void blasRnrm2X (cublasHandle_t ch, const int32_t n, const double* x, double* w, double* ret) {
	double ret_;
	blasDdotX (ch, n, x, x, w, &ret_);
	ret[0] = sqrt(ret_);
}
// GEMV
void blasRgemv (cublasHandle_t ch, const char trans, const int32_t m, const int32_t n, const float alpha, const float* A, const int32_t lda, const float* x, const int32_t incx, const float beta, float* y, const int32_t incy) {
	cublasSgemv (ch, ToCublasOp(trans), m, n, &alpha, A, lda, x, incx, &beta, y, incy);
}
void blasRgemv (cublasHandle_t ch, const char trans, const int32_t m, const int32_t n, const double alpha, const double* A, const int32_t lda, const double* x, const int32_t incx, const double beta, double* y, const int32_t incy) {
	cublasDgemv (ch, ToCublasOp(trans), m, n, &alpha, A, lda, x, incx, &beta, y, incy);
}
// GEMM
void blasRgemm (cublasHandle_t ch, const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const float alpha, const float* A, const int32_t lda, const float* B, const int32_t ldb, const float beta, float* C, const int32_t ldc) {
	cublasSgemm (ch, ToCublasOp(transA), ToCublasOp(transB), m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}
void blasRgemm (cublasHandle_t ch, const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const double alpha, const double* A, const int32_t lda, const double* B, const int32_t ldb, const double beta, double* C, const int32_t ldc) {
	cublasDgemm (ch, ToCublasOp(transA), ToCublasOp(transB), m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}
// GEMM-BATCH
void blasRgemmBatch (cublasHandle_t ch, const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const float alpha, const float** A, const int32_t lda, const float** B, const int32_t ldb, const float beta, float** C, const int32_t ldc, const int32_t grp, const int32_t cnt) {
	cublasSgemmBatched (ch, ToCublasOp(transA), ToCublasOp(transB), m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc, cnt);
}
void blasRgemmBatch (cublasHandle_t ch, const char transA, const char transB, const int32_t m, const int32_t n, const int32_t k, const double alpha, const double** A, const int32_t lda, const double** B, const int32_t ldb, const double beta, double** C, const int32_t ldc, const int32_t grp, const int32_t cnt) {
	cublasDgemmBatched (ch, ToCublasOp(transA), ToCublasOp(transB), m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc, cnt);
}
// CSRMV
__device__ __forceinline__ double shuffleGet (double &val, const int n) {
	return __hiloint2double(
		__shfl_xor_sync(0xFFFFFFFF, __double2hiint(val), n, 32),
		__shfl_xor_sync(0xFFFFFFFF, __double2loint(val), n, 32));
}

__device__ __forceinline__
void shuffleAdd (double &val, const int32_t n) {
	val += __hiloint2double(
		__shfl_xor_sync(0xFFFFFFFF, __double2hiint(val), n, 32),
		__shfl_xor_sync(0xFFFFFFFF, __double2loint(val), n, 32));
}
__device__ __forceinline__
void shuffleAdd (float &val, const int32_t n) {
	val += __shfl_xor_sync (0xFFFFFFFF, val, n, 32);
}

template <typename TYPE>
__global__
void csrmv_n_kernel (
	const int32_t m,
	const TYPE alpha,
	const TYPE * csrValA,
	const int32_t * csrRowPtrA,
	const int32_t * csrColIndA,
	const TYPE * x,
	const TYPE beta,
	TYPE *y
) {
	const int32_t tx = threadIdx.x;
	const int32_t tid = blockDim.x * blockIdx.x + tx;
	const int32_t rowid = tid / 32;
	const int32_t lane  = tid % 32; 
	register TYPE Xr1, Ar1, Tr1;

	if (rowid < m) {
		Tr1 = 0.;
		for (int32_t i = csrRowPtrA[rowid] + lane; i < csrRowPtrA[rowid+1]; i += 32) {
			Ar1 = csrValA[i];
			Xr1 = x[csrColIndA[i]];
			Tr1 = FMA (Ar1, Xr1, Tr1);
		}
		#pragma unroll
		for (int32_t i = 16; i > 0; i >>= 1) 
			shuffleAdd (Tr1, i);
		if (lane == 0) {
			if (beta == 0.)
				y[rowid] = alpha * Tr1;
			else
				y[rowid] = FMA (alpha, Tr1, MUL (beta, y[rowid]));
		}
	}
}

template <typename TYPE>
__host__
int32_t csrmv_n (
	const int32_t m,
	const TYPE alpha,
	const TYPE *csrValA,
	const int32_t *csrRowPtrA,
	const int32_t *csrColIndA,
	const TYPE *x,
	const TYPE beta,
	TYPE *y
) {
	int32_t ntx = 512;
	int32_t nbx = ceil (float(m) / (ntx/32));
	dim3 threads = dim3 (ntx);
	dim3 grid = dim3 (nbx);
	csrmv_n_kernel <<< grid, threads >>> (m, alpha, csrValA, csrRowPtrA, csrColIndA, x, beta, y);
	return 0;
}

__global__
void csrmv_n_x_kernel (
	const int32_t m,
	const double alpha,
	const double * csrValA,
	const int32_t * csrRowPtrA,
	const int32_t * csrColIndA,
	const double * x,
	const double beta,
	double *y,
	const int32_t flag
) {
	const int32_t tx = threadIdx.x;
	const int32_t tid = blockDim.x * blockIdx.x + tx;
	const int32_t rowid = tid / 32;
	const int32_t lane  = tid % 32; 
	register double Xr1, Ar1;
	register double2 Tr1;

	if (rowid < m) {
		Tr1 = make_double2 (0., 0.);
		for (int32_t i = csrRowPtrA[rowid] + lane; i < csrRowPtrA[rowid+1]; i += 32) {
			Ar1 = csrValA[i];
			Xr1 = x[csrColIndA[i]];
			cuDot2i (Ar1, Xr1, Tr1);
		}
		#pragma unroll
		for (int32_t i = 16; i > 0; i >>= 1) {
			double q;
			cuTwoSum (Tr1.x, shuffleGet (Tr1.x, i), Tr1.x, q);
			Tr1.y = Tr1.y + (q + shuffleGet (Tr1.y, i));
		}
		if (lane == 0) {
			double2 ret;
			if (beta == 0.)
				ret = cuQuadMul (alpha, Tr1);
			else
				ret = cuQuadAdd (cuQuadMul (alpha, Tr1), cuQuadMul (beta, y[rowid]));
			if (flag) {
				y[rowid*2] = ret.x;
				y[rowid*2+1] = ret.y;
			} else {
				y[rowid] = ret.x;
			}
		}
	}
}

__host__
int32_t csrmv_n_x (
	const int32_t m,
	const double alpha,
	const double * csrValA,
	const int32_t * csrRowPtrA,
	const int32_t * csrColIndA,
	const double * x,
	const double beta,
	double *y,
	const int32_t flag
) {
	int32_t ntx = 512;
	int32_t nbx = ceil (float(m) / (ntx/32));
	dim3 threads = dim3 (ntx);
	dim3 grid = dim3 (nbx);
	csrmv_n_x_kernel <<< grid, threads >>> (m, alpha, csrValA, csrRowPtrA, csrColIndA, x, beta, y, flag);
	return 0;
}

void blasRcsrmv (cusparseHandle_t ch, const char trans, const int32_t m, const int32_t n, const int32_t nnz, const float alpha, const cusparseMatDescr_t descrA, const float *A, const int32_t *devAcolind, const int32_t *devArowptr, const float *X, const float beta, float *Y) {
	#if defined CUSPARSE_CSR
	//cusparseScsrmv (ch, ToCusparseOp(trans), m, n, nnz, &alpha, descrA, A, devArowptr, devAcolind, X, &beta, Y);
	fprintf (OUTPUT, "blasRcsrmv is not implemented\n");
	#else
	csrmv_n (m, alpha, A, devArowptr, devAcolind, X, beta, Y);
	#endif
}
void blasRcsrmv (cusparseHandle_t ch, const char trans, const int32_t m, const int32_t n, const int32_t nnz, const double alpha, const cusparseMatDescr_t descrA, const double *A, const int32_t *devAcolind, const int32_t *devArowptr, const double *X, const double beta, double *Y) {
	#if defined CUSPARSE_CSR
	//cusparseDcsrmv (ch, ToCusparseOp(trans), m, n, nnz, &alpha, descrA, A, devArowptr, devAcolind, X, &beta, Y);
	cusparseSpMatDescr_t matA;
	cusparseDnVecDescr_t vecX, vecY;
	void* dBuffer = NULL;
	size_t bufsize = 0;
	cusparseCreateCsr (&matA, m, n, nnz, (void*)devArowptr, (void*)devAcolind, (void*)A, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
	cusparseCreateDnVec (&vecX, n, (void*)X, CUDA_R_64F);
	cusparseCreateDnVec (&vecY, m, (void*)Y, CUDA_R_64F);
	cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
	cusparseSpMV_bufferSize (ch, op, &alpha, matA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufsize);
    cudaMalloc (&dBuffer, bufsize);
	cusparseSpMV (ch, op, &alpha, matA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
    cudaFree (&dBuffer);
	cusparseDestroySpMat (matA);
	cusparseDestroyDnVec (vecX);
	cusparseDestroyDnVec (vecY);
	#else
	csrmv_n (m, alpha, A, devArowptr, devAcolind, X, beta, Y);
	#endif
}
void blasRcsrmvX (cusparseHandle_t ch, const char trans, const int32_t m, const int32_t n, const int32_t nnz, const float alpha, const cusparseMatDescr_t descrA, const float *A, const int32_t *devAcolind, const int32_t *devArowptr, const float *X, const float beta, float *Y, const int32_t flag) {
	//csrmv_n_x (m, alpha, A, devArowptr, devAcolind, X, beta, Y, flag);
	fprintf (OUTPUT, "OzBLAS error: blasRcsrmvX is not available.\n");
	exit(1);
}
void blasRcsrmvX (cusparseHandle_t ch, const char trans, const int32_t m, const int32_t n, const int32_t nnz, const double alpha, const cusparseMatDescr_t descrA, const double *A, const int32_t *devAcolind, const int32_t *devArowptr, const double *X, const double beta, double *Y, const int32_t flag) {
	csrmv_n_x (m, alpha, A, devArowptr, devAcolind, X, beta, Y, flag);
}

__global__
void csrmm_n_x_kernel (
	const int32_t m,
	const int32_t n,
	const int32_t k,
	const float alpha,
	const float * __restrict__ csrValA,
	const int32_t * __restrict__ csrRowPtrA,
	const int32_t * __restrict__ csrColIndA,
	const float * __restrict__ B,
	const int32_t ldb,
	const float beta,
	float *C,
	const int32_t ldc
) {
	const int32_t tx = threadIdx.x;
	const int32_t tid = blockDim.x * blockIdx.x + tx;
	const int32_t rowid = tid / 32;
	const int32_t lane  = tid % 32; 
	register double Br1, Ar1;
	register double Tr1;

	if (rowid < m) {
		for (int32_t j = 0; j < n; j++) {
			Tr1 = 0.;
			for (int32_t i = csrRowPtrA[rowid] + lane; i < csrRowPtrA[rowid+1]; i += 32) {
				Ar1 = (double) csrValA[i];
				Br1 = (double) B[j * ldb + csrColIndA[i]];
				Tr1 = __fma_rn (Ar1, Br1, Tr1);
			}
			#pragma unroll
			for (int32_t i = 16; i > 0; i >>= 1) 
				shuffleAdd (Tr1, i);
			if (lane == 0) {
				if (beta == 0.)
					C[j * ldc + rowid] = (double)alpha * Tr1;
				else
					C[j * ldc + rowid] = __fma_rn ((double)alpha, Tr1, __dmul_rn ((double)beta, (double)C[j * ldc + rowid]));
			}
		}
	}
}

__global__
void csrmm_n_x_kernel (
	const int32_t m,
	const int32_t n,
	const int32_t k,
	const double alpha,
	const double * __restrict__ csrValA,
	const int32_t * __restrict__ csrRowPtrA,
	const int32_t * __restrict__ csrColIndA,
	const double * __restrict__ B,
	const int32_t ldb,
	const double beta,
	double *C,
	const int32_t ldc
) {
	const int32_t tx = threadIdx.x;
	const int32_t tid = blockDim.x * blockIdx.x + tx;
	const int32_t rowid = tid / 32;
	const int32_t lane  = tid % 32; 
	register double Br1, Ar1;
	register double2 Tr1;

	if (rowid < m) {
		for (int32_t j = 0; j < n; j++) {
			Tr1 = make_double2 (0., 0.);
			for (int32_t i = csrRowPtrA[rowid] + lane; i < csrRowPtrA[rowid+1]; i += 32) {
				Ar1 = csrValA[i];
				Br1 = B[j * ldb + csrColIndA[i]];
				cuDot2i (Ar1, Br1, Tr1);
			}
			#pragma unroll
			for (int32_t i = 16; i > 0; i >>= 1) {
				double q;
				cuTwoSum (Tr1.x, shuffleGet (Tr1.x, i), Tr1.x, q);
				Tr1.y = Tr1.y + (q + shuffleGet (Tr1.y, i));
			}
			if (lane == 0) {
				if (beta == 0.)
					C[j * ldc + rowid] = __dmul_rn (alpha, __dadd_rn (Tr1.x, Tr1.y));
				else
					C[j * ldc + rowid] = __fma_rn (alpha, __dadd_rn (Tr1.x, Tr1.y), __dmul_rn (beta, C[j * ldc + rowid]));
			}
		}
	}
}

__global__
void csrmm_n_x2_kernel (
	const int32_t m,
	const int32_t n,
	const int32_t k,
	const double alpha,
	const double * __restrict__ csrValA,
	const int32_t * __restrict__ csrRowPtrA,
	const int32_t * __restrict__ csrColIndA,
	const double * __restrict__ B,
	const int32_t ldb,
	const double beta,
	double *C,
	const int32_t ldc
) {
	const int32_t tx = threadIdx.x;
	const int32_t tid = blockDim.x * blockIdx.x + tx;
	const int32_t rowid = tid / 32;
	const int32_t lane  = tid % 32; 
	register double Br1, Ar1;
	register double2 Tr1;

	if (rowid < m) {
		for (int32_t j = 0; j < n; j++) {
			Tr1 = make_double2 (0., 0.);
			for (int32_t i = csrRowPtrA[rowid] + lane; i < csrRowPtrA[rowid+1]; i += 32) {
				Ar1 = csrValA[i];
				Br1 = B[j * ldb + csrColIndA[i]];
				cuDot2i (Ar1, Br1, Tr1);
			}
			#pragma unroll
			for (int32_t i = 16; i > 0; i >>= 1) {
				double q;
				cuTwoSum (Tr1.x, shuffleGet (Tr1.x, i), Tr1.x, q);
				Tr1.y = Tr1.y + (q + shuffleGet (Tr1.y, i));
			}
			if (lane == 0) {
				if (beta == 0.)
					Tr1 = cuQuadMul (alpha, Tr1);
				else
					Tr1 = cuQuadAdd (cuQuadMul (alpha, Tr1), cuQuadMul (beta, C[j * ldc + rowid]));
				C[j * ldc + rowid] = Tr1.x;
				C[j * ldc + rowid + ldc * n] = Tr1.y;
			}
		}
	}
}

__host__
int32_t csrmm_n_x2 (
	const char trans,
	const int32_t m,
	const int32_t n,
	const int32_t k,
	const double alpha,
	const double * csrValA,
	const int32_t * csrRowPtrA,
	const int32_t * csrColIndA,
	const double * B,
	const int32_t ldb,
	const double beta,
	double *C,
	const int32_t ldc
) {
	if (trans == 'T' || trans == 't') {
		fprintf (OUTPUT, "OzBLAS error: CSRMV-T is not available.\n");
		exit(1);
	}

	int32_t ntx = 512;
	int32_t nbx = ceil (float(m) / (ntx/32));
	dim3 threads = dim3 (ntx);
	dim3 grid = dim3 (nbx);
	csrmm_n_x2_kernel <<< grid, threads >>> (m, n, k, alpha, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);

	return 0;
}

template <typename TYPE>
__host__
int32_t csrmm_n_x (
	const char trans,
	const int32_t m,
	const int32_t n,
	const int32_t k,
	const TYPE alpha,
	const TYPE * csrValA,
	const int32_t * csrRowPtrA,
	const int32_t * csrColIndA,
	const TYPE * B,
	const int32_t ldb,
	const TYPE beta,
	TYPE *C,
	const int32_t ldc
) {
	if (trans == 'T' || trans == 't') {
		fprintf (OUTPUT, "OzBLAS error: CSRMV-T is not available.\n");
		exit(1);
	}

	int32_t ntx = 512;
	int32_t nbx = ceil (float(m) / (ntx/32));
	dim3 threads = dim3 (ntx);
	dim3 grid = dim3 (nbx);
	csrmm_n_x_kernel <<< grid, threads >>> (m, n, k, alpha, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);

	return 0;
}

__global__
void csrmm_n_kernel (
	const int32_t m,
	const int32_t n,
	const int32_t k,
	const float alpha,
	const float * __restrict__ csrValA,
	const int32_t * __restrict__ csrRowPtrA,
	const int32_t * __restrict__ csrColIndA,
	const float * __restrict__ B,
	const int32_t ldb,
	const float beta,
	float *C,
	const int32_t ldc
) {
	const int32_t tx = threadIdx.x;
	const int32_t tid = blockDim.x * blockIdx.x + tx;
	const int32_t rowid = tid / 32;
	const int32_t lane  = tid % 32; 
	register float Br1, Ar1, Tr1;

	if (rowid < m) {
		for (int32_t j = 0; j < n; j++) {
			Tr1 = 0.;
			for (int32_t i = csrRowPtrA[rowid] + lane; i < csrRowPtrA[rowid+1]; i += 32) {
				Ar1 = csrValA[i];
				Br1 = B[j * ldb + csrColIndA[i]];
				Tr1 = __fma_rn (Ar1, Br1, Tr1);
			}
			#pragma unroll
			for (int32_t i = 16; i > 0; i >>= 1) 
				shuffleAdd (Tr1, i);
			if (lane == 0) {
				if (beta == 0.)
					C[j * ldc + rowid] = __dmul_rn (alpha, Tr1);
				else
					C[j * ldc + rowid] = __fma_rn (alpha, Tr1, __dmul_rn (beta, C[j * ldc + rowid]));
			}
		}
	}
}

__global__
void csrmm_n_kernel (
	const int32_t m,
	const int32_t n,
	const int32_t k,
	const double alpha,
	const double * __restrict__ csrValA,
	const int32_t * __restrict__ csrRowPtrA,
	const int32_t * __restrict__ csrColIndA,
	const double * __restrict__ B,
	const int32_t ldb,
	const double beta,
	double *C,
	const int32_t ldc
) {
	const int32_t tx = threadIdx.x;
	const int32_t tid = blockDim.x * blockIdx.x + tx;
	const int32_t rowid = tid / 32;
	const int32_t lane  = tid % 32; 
	register double Br1, Ar1, Tr1;

	if (rowid < m) {
		for (int32_t j = 0; j < n; j++) {
			Tr1 = 0.;
			for (int32_t i = csrRowPtrA[rowid] + lane; i < csrRowPtrA[rowid+1]; i += 32) {
				Ar1 = csrValA[i];
				Br1 = B[j * ldb + csrColIndA[i]];
				Tr1 = __fma_rn (Ar1, Br1, Tr1);
			}
			#pragma unroll
			for (int32_t i = 16; i > 0; i >>= 1) 
				shuffleAdd (Tr1, i);
			if (lane == 0) {
				if (beta == 0.)
					C[j * ldc + rowid] = __dmul_rn (alpha, Tr1);
				else	
					C[j * ldc + rowid] = __fma_rn (alpha, Tr1, __dmul_rn (beta, C[j * ldc + rowid]));
			}
		}
	}
}

template <typename TYPE>
__host__
int32_t csrmm_n (
	const char trans,
	const int32_t m,
	const int32_t n,
	const int32_t k,
	const TYPE alpha,
	const TYPE * csrValA,
	const int32_t * csrRowPtrA,
	const int32_t * csrColIndA,
	const TYPE * B,
	const int32_t ldb,
	const TYPE beta,
	TYPE *C,
	const int32_t ldc
) {
	if (trans == 'T' || trans == 't') {
		fprintf (OUTPUT, "OzBLAS error: CSRMV-T is not available.\n");
		exit(1);
	}

	int32_t ntx = 512;
	int32_t nbx = ceil (float(m) / (ntx/32));
	dim3 threads = dim3 (ntx);
	dim3 grid = dim3 (nbx);
	csrmm_n_kernel <<< grid, threads >>> (m, n, k, alpha, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);

	return 0;
}


/*
 * reg block ver
__global__
void csrmm_n_x_kernel (
	const int32_t m,
	const int32_t n,
	const int32_t k,
	const float alpha,
	const float * __restrict__ csrValA,
	const int32_t * __restrict__ csrRowPtrA,
	const int32_t * __restrict__ csrColIndA,
	const float * __restrict__ B,
	const int32_t ldb,
	const float beta,
	float *C,
	const int32_t ldc
) {
	const int32_t iTx = threadIdx.x;
	const int32_t iTy = threadIdx.y;
	const int32_t nTx = blockDim.x;
	const int32_t nTy = blockDim.y;
	const int32_t rowid = nTy * blockIdx.y + iTy;
	const int32_t lane  = iTx % nTx; 
	register double Ar1, Br1, Tr1[16]; // 16 < n

	if (rowid < m) {
		for (int32_t j = 0; j < n; j++) 
			Tr1[j] = 0.;
		for (int32_t i = csrRowPtrA[rowid] + lane; i < csrRowPtrA[rowid+1]; i += nTx) {
			Ar1 = (double)csrValA[i];
			for (int32_t j = 0; j < n; j++) {
				Br1 = (double)B[j * ldb + csrColIndA[i]];
				Tr1[j] = __fma_rn (Ar1, Br1, Tr1[j]);
			}
		}
		for (int32_t j = 0; j < n; j++) {
			#pragma unroll
			for (int32_t i = nTx/2; i > 0; i >>= 1) 
				shuffleAdd (Tr1[j], i);
			if (lane == 0) 
				C[j * ldc + rowid] = __fma_rn(alpha, Tr1[j], __dmul_rn(beta, C[j * ldc + rowid]));
		}
	}
}

__global__
void csrmm_n_x_kernel (
	const int32_t m,
	const int32_t n,
	const int32_t k,
	const double alpha,
	const double * __restrict__ csrValA,
	const int32_t * __restrict__ csrRowPtrA,
	const int32_t * __restrict__ csrColIndA,
	const double * __restrict__ B,
	const int32_t ldb,
	const double beta,
	double *C,
	const int32_t ldc
) {
	const int32_t iTx = threadIdx.x;
	const int32_t iTy = threadIdx.y;
	const int32_t nTx = blockDim.x;
	const int32_t nTy = blockDim.y;
	const int32_t rowid = nTy * blockIdx.y + iTy;
	const int32_t lane  = iTx % nTx; 
	register double Ar1, Br1; // 16 < n
	register double2 Tr1[16];

	if (rowid < m) {
		for (int32_t j = 0; j < n; j++) 
			Tr1[j] = make_double2 (0., 0.);
		for (int32_t i = csrRowPtrA[rowid] + lane; i < csrRowPtrA[rowid+1]; i += nTx) {
			Ar1 = csrValA[i];
			for (int32_t j = 0; j < n; j++) {
				Br1 = B[j * ldb + csrColIndA[i]];
				cuDot2i (Ar1, Br1, Tr1[j]);
			}
		}
		for (int32_t j = 0; j < n; j++) {
			#pragma unroll
			for (int32_t i = 16; i > 0; i >>= 1) {
				double q;
				cuTwoSum (Tr1[j].x, shuffleGet (Tr1[j].x, i), Tr1[j].x, q);
				Tr1[j].y = Tr1[j].y + (q + shuffleGet (Tr1[j].y, i));
			}
			if (lane == 0) 
				C[j * ldc + rowid] = __dmul_rn (alpha, __dadd_rn (Tr1[j].x, Tr1[j].y)) + __dmul_rn (beta, C[j * ldc + rowid]);
		}
	}
}
	
template <typename TYPE>
__host__
int32_t csrmm_n_x (
	const char trans,
	const int32_t m,
	const int32_t n,
	const int32_t k,
	const TYPE alpha,
	const TYPE * csrValA,
	const int32_t * csrRowPtrA,
	const int32_t * csrColIndA,
	const TYPE * B,
	const int32_t ldb,
	const TYPE beta,
	TYPE *C,
	const int32_t ldc
) {
	if (trans == 'T' || trans == 't') {
		fprintf (OUTPUT, "OzBLAS error: CSRMV-T is not available.\n");
		exit(1);
	}
	if (n > 16) {
		fprintf (OUTPUT, "csrmm_n error n>16\n");
		exit(1);
	}

	int32_t ntx = 32;
	int32_t nty = 8;
	int32_t nbx = 1;
	int32_t nby = ceil ((float)m / nty);
	dim3 threads = dim3 (ntx, nty);
	dim3 grid = dim3 (nbx, nby);
	csrmm_n_x_kernel <<< grid, threads >>> (m, n, k, alpha, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);

	return 0;
}

template <typename TYPE>
__global__
void csrmm_n_kernel (
	const int32_t m,
	const int32_t n,
	const int32_t k,
	const TYPE alpha,
	const TYPE * __restrict__ csrValA,
	const int32_t * __restrict__ csrRowPtrA,
	const int32_t * __restrict__ csrColIndA,
	const TYPE * __restrict__ B,
	const int32_t ldb,
	const TYPE beta,
	TYPE *C,
	const int32_t ldc
) {
	const int32_t iTx = threadIdx.x;
	const int32_t iTy = threadIdx.y;
	const int32_t nTx = blockDim.x;
	const int32_t nTy = blockDim.y;
	const int32_t rowid = nTy * blockIdx.y + iTy;
	const int32_t lane  = iTx % nTx; 
	register TYPE Ar1, Br1, Tr1[16]; // 16 < n

	if (rowid < m) {
		for (int32_t j = 0; j < n; j++) 
			Tr1[j] = 0.;
		for (int32_t i = csrRowPtrA[rowid] + lane; i < csrRowPtrA[rowid+1]; i += nTx) {
			Ar1 = csrValA[i];
			for (int32_t j = 0; j < n; j++) {
				Br1 = B[j * ldb + csrColIndA[i]];
				Tr1[j] = FMA (Ar1, Br1, Tr1[j]);
			}
		}
		for (int32_t j = 0; j < n; j++) {
			#pragma unroll
			for (int32_t i = nTx/2; i > 0; i >>= 1) 
				shuffleAdd (Tr1[j], i);
			if (lane == 0) 
				C[j * ldc + rowid] = alpha * Tr1[j] + beta * C[j * ldc + rowid];
		}
	}
}

template <typename TYPE>
__host__
int32_t csrmm_n (
	const char trans,
	const int32_t m,
	const int32_t n,
	const int32_t k,
	const TYPE alpha,
	const TYPE * csrValA,
	const int32_t * csrRowPtrA,
	const int32_t * csrColIndA,
	const TYPE * B,
	const int32_t ldb,
	const TYPE beta,
	TYPE *C,
	const int32_t ldc
) {
	if (trans == 'T' || trans == 't') {
		fprintf (OUTPUT, "OzBLAS error: CSRMV-T is not available.\n");
		exit(1);
	}
	if (n > 16) {
		fprintf (OUTPUT, "csrmm_n error n>16\n");
		exit(1);
	}

	int32_t ntx = 32;
	int32_t nty = 8;
	int32_t nbx = 1;
	int32_t nby = ceil ((float)m / nty);
	dim3 threads = dim3 (ntx, nty);
	dim3 grid = dim3 (nbx, nby);
	csrmm_n_kernel <<< grid, threads >>> (m, n, k, alpha, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);

	return 0;
}
*/

void blasRcsrmm (cusparseHandle_t ch, const char transA, const int32_t m, const int32_t n, const int32_t k, const int32_t nnz, const float alpha, const cusparseMatDescr_t descrA, const float *A, const int32_t *devAcolind, const int32_t *devArowptr, const float *B, const int32_t ldb, const float beta, float *C, const int32_t ldc) {
	#if defined CUSPARSE_CSR
//	cusparseScsrmm (ch, ToCusparseOp(transA), m, n, k, nnz, &alpha, descrA, A, devArowptr, devAcolind, B, ldb, &beta, C, ldc);
	fprintf (OUTPUT, "blasRcsrmm is not implemented\n");
	#else
	csrmm_n (transA, m, n, k, alpha, A, devArowptr, devAcolind, B, ldb, beta, C, ldc);
	#endif
}
void blasRcsrmm (cusparseHandle_t ch, const char transA, const int32_t m, const int32_t n, const int32_t k, const int32_t nnz, const double alpha, const cusparseMatDescr_t descrA, const double *A, const int32_t *devAcolind, const int32_t *devArowptr, const double *B, const int32_t ldb, const double beta, double *C, const int32_t ldc) {
	#if defined CUSPARSE_CSR
	//cusparseDcsrmm (ch, ToCusparseOp(transA), m, n, k, nnz, &alpha, descrA, A, devArowptr, devAcolind, B, ldb, &beta, C, ldc);
	cusparseSpMatDescr_t matA;
	cusparseDnMatDescr_t matB, matC;
	void* dBuffer = NULL;
	size_t bufsize = 0;
	cusparseCreateCsr (&matA, m, k, nnz, (void*)devArowptr, (void*)devAcolind, (void*)A, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
	cusparseCreateDnMat (&matB, k, n, ldb, (void*)B, CUDA_R_64F, CUSPARSE_ORDER_COL);
	cusparseCreateDnMat (&matC, m, n, ldc, (void*)C, CUDA_R_64F, CUSPARSE_ORDER_COL);
	cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
	cusparseSpMM_bufferSize (ch, op, op, &alpha, matA, matB, &beta, matC, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &bufsize);
    cudaMalloc (&dBuffer, bufsize);
	cusparseSpMM (ch, op, op, &alpha, matA, matB, &beta, matC, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);
    cudaFree (&dBuffer);
	cusparseDestroySpMat (matA);
	cusparseDestroyDnMat (matB);
	cusparseDestroyDnMat (matC);
	#else
	csrmm_n (transA, m, n, k, alpha, A, devArowptr, devAcolind, B, ldb, beta, C, ldc);
	#endif
}
void blasRcsrmm_x (cusparseHandle_t ch, const char transA, const int32_t m, const int32_t n, const int32_t k, const int32_t nnz, const float alpha, const cusparseMatDescr_t descrA, const float *A, const int32_t *devAcolind, const int32_t *devArowptr, const float *B, const int32_t ldb, const float beta, float *C, const int32_t ldc) {
	csrmm_n_x (transA, m, n, k, alpha, A, devArowptr, devAcolind, B, ldb, beta, C, ldc);
}
// x2: output is double
void blasRcsrmm_x2 (cusparseHandle_t ch, const char transA, const int32_t m, const int32_t n, const int32_t k, const int32_t nnz, const float alpha, const cusparseMatDescr_t descrA, const float *A, const int32_t *devAcolind, const int32_t *devArowptr, const float *B, const int32_t ldb, const float beta, float *C, const int32_t ldc) {
	fprintf (OUTPUT, "csrmm_n_x2 is not implemented\n");
//	csrmm_n_x2 (transA, m, n, k, alpha, A, devArowptr, devAcolind, B, ldb, beta, C, ldc);
}
void blasRcsrmm_x (cusparseHandle_t ch, const char transA, const int32_t m, const int32_t n, const int32_t k, const int32_t nnz, const double alpha, const cusparseMatDescr_t descrA, const double *A, const int32_t *devAcolind, const int32_t *devArowptr, const double *B, const int32_t ldb, const double beta, double *C, const int32_t ldc) {
	csrmm_n_x (transA, m, n, k, alpha, A, devArowptr, devAcolind, B, ldb, beta, C, ldc);
}
// x2: output is double-double (as double pointer)
void blasRcsrmm_x2 (cusparseHandle_t ch, const char transA, const int32_t m, const int32_t n, const int32_t k, const int32_t nnz, const double alpha, const cusparseMatDescr_t descrA, const double *A, const int32_t *devAcolind, const int32_t *devArowptr, const double *B, const int32_t ldb, const double beta, double *C, const int32_t ldc) {
	csrmm_n_x2 (transA, m, n, k, alpha, A, devArowptr, devAcolind, B, ldb, beta, C, ldc);
}

// --------------------------------------------------------------
// OMATCOPY
// --------------------------------------------------------------
void blasRomatcopy (cublasHandle_t ch, const char trans, const int32_t m, const int32_t n, const float* A, const int32_t lda, float* B, const int32_t ldb) {
	float alpha = 1.;
	float beta = 0.;
	float* np = nullptr;
	// note: m and n are exchanged from omatcopy
	cublasSgeam (ch, ToCublasOp(trans), ToCublasOp('n'), n, m, &alpha, A, lda, &beta, np, n, B, ldb);
}

void blasRomatcopy (cublasHandle_t ch, const char trans, const int32_t m, const int32_t n, const double* A, const int32_t lda, double* B, const int32_t ldb) {
	double alpha = 1.;
	double beta = 0.;
	double* np = nullptr;
	// note: m and n are exchanged from omatcopy
	cublasDgeam (ch, ToCublasOp(trans), ToCublasOp('n'), n, m, &alpha, A, lda, &beta, np, n, B, ldb);
}

