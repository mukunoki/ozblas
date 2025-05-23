#define MYGEMM_NTX1 512
#define MYGEMM_NTX2 512
#define MYGEMM_NTX3 256
#define MYGEMM_NTX4 128
#define MYGEMM_NTX5 64
#define MYGEMM_NTX6 64
#define SPLIT_VEC_NBX 512

#include "../cuozblas_common.h"
#include "myblas_dot_kernel.cu"
#include "myblas_dot_1_kernel.cu"
#include "myblas_dot_2_kernel.cu"
#include "myblas_dot_3_kernel.cu"
#include "myblas_dot_4_kernel.cu"
#include "myblas_dot_5_kernel.cu"
#include "myblas_dot_6_kernel.cu"

//__host__ __inline__ size_t ceilingEx (const float n, const size_t c) {
//	return (size_t)ceil(n/c)*c;
//}


template <typename TYPE1, typename TYPE2>
int32_t blasRgemmSkinny_x2 (
	cuozblasHandle_t *oh,
	const char transA,
	const char transB,
	const size_t m,
	const size_t n,
	const size_t k,
	const TYPE2 alpha,
	const TYPE1 *A,
	const size_t lda,
	const TYPE1 *B,
	const size_t ldb,
	const TYPE2 beta,
	TYPE2 *C,
	const size_t ldc
) {
	size_t ntx, nbx;
	if (m == n && max (m, n) <= 6) {
		double2 *W = (double2*)(oh->devWork + oh->memAddr);
		oh->memAddr += sizeof(double2) * m * n * SPLIT_VEC_NBX;
		cumemCheck (oh);
		switch (m) {
			case 1:
				ntx = MYGEMM_NTX1;
				nbx = min ((int)ceil((float)k/ntx), SPLIT_VEC_NBX);
				myblas_dot_x_kernel_1  <<< dim3(nbx), dim3(ntx) >>> (k, A, lda, B, ldb, W, nbx);
				myblas_sum_x2_kernel_1 <<< dim3(1),   dim3(ntx) >>> (nbx, W, nbx, C);
				break;
			case 2:
				ntx = MYGEMM_NTX2;
				nbx = min ((int)ceil((float)k/ntx), SPLIT_VEC_NBX);
				myblas_dot_x_kernel_2  <<< dim3(nbx), dim3(ntx) >>> (k, A, lda, B, ldb, W, nbx);
				myblas_sum_x2_kernel_2 <<< dim3(1),   dim3(ntx) >>> (nbx, W, nbx, C);
				break;
			case 3:
				ntx = MYGEMM_NTX3;
				nbx = min ((int)ceil((float)k/ntx), SPLIT_VEC_NBX);
				myblas_dot_x_kernel_3  <<< dim3(nbx), dim3(ntx) >>> (k, A, lda, B, ldb, W, nbx);
				myblas_sum_x2_kernel_3 <<< dim3(1),   dim3(ntx) >>> (nbx, W, nbx, C);
				break;
			case 4:
				ntx = MYGEMM_NTX4;
				nbx = min ((int)ceil((float)k/ntx), SPLIT_VEC_NBX);
				myblas_dot_x_kernel_4  <<< dim3(nbx), dim3(ntx) >>> (k, A, lda, B, ldb, W, nbx);
				myblas_sum_x2_kernel_4 <<< dim3(1),   dim3(ntx) >>> (nbx, W, nbx, C);
				break;
			case 5:
				ntx = MYGEMM_NTX5;
				nbx = min ((int)ceil((float)k/ntx), SPLIT_VEC_NBX);
				myblas_dot_x_kernel_5  <<< dim3(nbx), dim3(ntx) >>> (k, A, lda, B, ldb, W, nbx);
				myblas_sum_x2_kernel_5 <<< dim3(1),   dim3(ntx) >>> (nbx, W, nbx, C);
				break;
			case 6:
				ntx = MYGEMM_NTX6;
				nbx = min ((int)ceil((float)k/ntx), SPLIT_VEC_NBX);
				myblas_dot_x_kernel_6  <<< dim3(nbx), dim3(ntx) >>> (k, A, lda, B, ldb, W, nbx);
				myblas_sum_x2_kernel_6 <<< dim3(1),   dim3(ntx) >>> (nbx, W, nbx, C);
				break;
		}
	} else if (max (m, n) <= 6) { // m != n 
		ntx = MYGEMM_NTX6;
		nbx = min ((int)ceil((float)k/ntx), SPLIT_VEC_NBX);
		dim3 threads = dim3 (ntx);
		dim3 grid = dim3 (nbx);
		double2 *W = (double2*)(oh->devWork + oh->memAddr);
		oh->memAddr += sizeof(double2) * m * n * SPLIT_VEC_NBX;
		cumemCheck (oh);
		myblas_dot_x_kernel  <<< grid, threads >>> (m, n, k, A, lda, B, ldb, W, nbx);
		myblas_sum_x2_kernel <<< dim3(1), threads >>> (m, n, nbx, W, nbx, C);
	} else {
		fprintf (OUTPUT, "OzBLAS error: GemmSkinny (m,n > 6) is not implemented (m=%d, n=%d).\n", m, n);
		exit (1);
	}
	return 0;
}

template <>
int32_t blasRgemmSkinny_x2 <half, float> (
	cuozblasHandle_t *oh,
	const char transA,
	const char transB,
	const size_t m,
	const size_t n,
	const size_t k,
	const float alpha,
	const half *A,
	const size_t lda,
	const half *B,
	const size_t ldb,
	const float beta,
	float *C,
	const size_t ldc
) {
	fprintf (OUTPUT, "blasRgemmSkinny_x2 for half is not available\n");
    exit(1);
}

template <>
int32_t blasRgemmSkinny_x2 <int32_t, int32_t> (
	cuozblasHandle_t *oh,
	const char transA,
	const char transB,
	const size_t m,
	const size_t n,
	const size_t k,
	const int32_t alpha,
	const int32_t *A,
	const size_t lda,
	const int32_t *B,
	const size_t ldb,
	const int32_t beta,
	int32_t *C,
	const size_t ldc
) {
	fprintf (OUTPUT, "blasRgemmSkinny_x2 for int is not available\n");
    exit(1);
}

template int32_t blasRgemmSkinny_x2 <double, double> ( cuozblasHandle_t *oh, const char transA, const char transB, const size_t m, const size_t n, const size_t k, const double alpha, const double *A, const size_t lda, const double *B, const size_t ldb, const double beta, double *C, const size_t ldc);
template int32_t blasRgemmSkinny_x2 <float, float> ( cuozblasHandle_t *oh, const char transA, const char transB, const size_t m, const size_t n, const size_t k, const float alpha, const float *A, const size_t lda, const float *B, const size_t ldb, const float beta, float *C, const size_t ldc);


template <typename TYPE1, typename TYPE2>
int32_t blasRgemmSkinny (
	cuozblasHandle_t *oh,
	const char transA,
	const char transB,
	const size_t m,
	const size_t n,
	const size_t k,
	const TYPE2 alpha,
	const TYPE1 *A,
	const size_t lda,
	const TYPE1 *B,
	const size_t ldb,
	const TYPE2 beta,
	TYPE2 *C,
	const size_t ldc
) {
	size_t ntx, nbx;
	if (m == n && max (m, n) <= 6) {
		TYPE1 *W = (TYPE1*)(oh->devWork + oh->memAddr);
		oh->memAddr += sizeof(TYPE1) * m * n * SPLIT_VEC_NBX;
		cumemCheck (oh);
		switch (m) {
			case 1:
				ntx = MYGEMM_NTX1;
				nbx = min ((int)ceil((float)k/ntx), SPLIT_VEC_NBX);
				myblas_dot_kernel_1 <<< dim3(nbx), dim3(ntx) >>> (k, A, lda, B, ldb, W, nbx);
				myblas_sum_kernel_1 <<< dim3(1),   dim3(ntx) >>> (nbx, W, nbx, C);
				break;
			case 2:
				ntx = MYGEMM_NTX2;
				nbx = min ((int)ceil((float)k/ntx), SPLIT_VEC_NBX);
				myblas_dot_kernel_2 <<< dim3(nbx), dim3(ntx) >>> (k, A, lda, B, ldb, W, nbx);
				myblas_sum_kernel_2 <<< dim3(1),   dim3(ntx) >>> (nbx, W, nbx, C);
				break;
			case 3:
				ntx = MYGEMM_NTX3;
				nbx = min ((int)ceil((float)k/ntx), SPLIT_VEC_NBX);
				myblas_dot_kernel_3 <<< dim3(nbx), dim3(ntx) >>> (k, A, lda, B, ldb, W, nbx);
				myblas_sum_kernel_3 <<< dim3(1),   dim3(ntx) >>> (nbx, W, nbx, C);
				break;
			case 4:
				ntx = MYGEMM_NTX4;
				nbx = min ((int)ceil((float)k/ntx), SPLIT_VEC_NBX);
				myblas_dot_kernel_4 <<< dim3(nbx), dim3(ntx) >>> (k, A, lda, B, ldb, W, nbx);
				myblas_sum_kernel_4 <<< dim3(1),   dim3(ntx) >>> (nbx, W, nbx, C);
				break;
			case 5:
				ntx = MYGEMM_NTX5;
				nbx = min ((int)ceil((float)k/ntx), SPLIT_VEC_NBX);
				myblas_dot_kernel_5 <<< dim3(nbx), dim3(ntx) >>> (k, A, lda, B, ldb, W, nbx);
				myblas_sum_kernel_5 <<< dim3(1),   dim3(ntx) >>> (nbx, W, nbx, C);
				break;
			case 6:
				ntx = MYGEMM_NTX6;
				nbx = min ((int)ceil((float)k/ntx), SPLIT_VEC_NBX);
				myblas_dot_kernel_6 <<< dim3(nbx), dim3(ntx) >>> (k, A, lda, B, ldb, W, nbx);
				myblas_sum_kernel_6 <<< dim3(1),   dim3(ntx) >>> (nbx, W, nbx, C);
				break;
		}
	} else if (max (m, n) <= 6) { // m != n 
		ntx = MYGEMM_NTX6;
		nbx = min ((int)ceil((float)k/ntx), SPLIT_VEC_NBX);
		dim3 threads = dim3 (ntx);
		dim3 grid = dim3 (nbx);
		TYPE1 *W = (TYPE1*)(oh->devWork + oh->memAddr);
		oh->memAddr += sizeof(TYPE1) * m * n * SPLIT_VEC_NBX;
		cumemCheck (oh);
		myblas_dot_kernel <<< grid, threads >>> (m, n, k, A, lda, B, ldb, W, nbx);
		myblas_sum_kernel <<< dim3(1), threads >>> (m, n, nbx, W, nbx, C);
	} else {
		blasRgemm (oh->ch, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	}
	return 0;
}


template <>
int32_t blasRgemmSkinny <half, float> (
	cuozblasHandle_t *oh,
	const char transA,
	const char transB,
	const size_t m,
	const size_t n,
	const size_t k,
	const float alpha,
	const half *A,
	const size_t lda,
	const half *B,
	const size_t ldb,
	const float beta,
	float *C,
	const size_t ldc
) {
	blasRgemm (oh->ch, 't', 'n', m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	return 0;
}

/*
template <>
int32_t blasRgemmSkinny <int32_t, int32_t> (
	cuozblasHandle_t *oh,
	const char transA,
	const char transB,
	const size_t m,
	const size_t n,
	const size_t k,
	const int32_t alpha,
	const int32_t *A,
	const size_t lda,
	const int32_t *B,
	const size_t ldb,
	const int32_t beta,
	int32_t *C,
	const size_t ldc
) {
	blasRgemm (oh->ch, 't', 'n', m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	return 0;
	//fprintf (OUTPUT, "blasRgemmSkinny for int is not available\n");
    //exit(1);
}
*/

template int32_t blasRgemmSkinny <double, double> ( cuozblasHandle_t *oh, const char transA, const char transB, const size_t m, const size_t n, const size_t k, const double alpha, const double *A, const size_t lda, const double *B, const size_t ldb, const double beta, double *C, const size_t ldc);
template int32_t blasRgemmSkinny <float, float> ( cuozblasHandle_t *oh, const char transA, const char transB, const size_t m, const size_t n, const size_t k, const float alpha, const float *A, const size_t lda, const float *B, const size_t ldb, const float beta, float *C, const size_t ldc);

