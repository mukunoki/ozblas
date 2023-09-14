#include "cuozblas_common.h"

// =========================================
// Split
// Based on K.Ozaki et al., "Error-free transformations of matrix multiplication
// by using fast routines of matrix multiplication and its applications", 2012
// =========================================

// Number of threads and blocks for CUDA kernels
#define SPLIT_N_NTX 32
#define SPLIT_N_NTY 16
#define SPLIT_T_NTX 512
#define SPLIT_T_NTY 1
#define SPLIT_VEC_NTX 512
#define SPLIT_VEC_NBX 512

#define CONST 0.75 // for splitting

/*
template <typename TYPE>
__device__
TYPE NextPowTwo (const TYPE p) {
	constexpr int32_t epse_type = getEpse <TYPE> ();
	return scalbn (p, epse_type) - (TYPE)((scalbn (1., epse_type) - 1) * p);
}
*/

template <typename TYPE1, typename TYPE2>
__host__ __device__
int32_t getRho (const int32_t dim, const int32_t splitEpsModeFlag) {
	constexpr int32_t epse_type1 = getEpse <TYPE1> ();
	constexpr int32_t epse_type2 = getEpse <TYPE2> ();
	constexpr int32_t epse_type22 = getEpse2 <TYPE2> ();
	switch (splitEpsModeFlag) {
		case 2: // Dot2
			return ceil((epse_type1-(epse_type22-log2(1.*dim))/2)); // standard-ver with Dot2
			break;
		case 9: // overflow-mode
			return ceil((epse_type1-(epse_type2-log2(2.*sqrt(dim)))/2)); // overflow-ver
			break;
		default: // non-overflow-mode
			return ceil((epse_type1-(epse_type2-log2(1.*dim))/2)); // standard-ver
	}
}

// =========================================

template <typename TYPE>
__global__
void cuozblasFindMaxCKernel (
	const int32_t m,
	const int32_t n,
	const TYPE * __restrict__ devInput,
	const int32_t ldi,
	TYPE *devMax
) {
	const int32_t iBx = blockIdx.x;
	const int32_t iBy = blockIdx.y;
	const int32_t nTx = SPLIT_T_NTX;//blockDim.x;
	const int32_t nTy = SPLIT_T_NTY;//blockDim.y;
	const int32_t iTx = threadIdx.x;
	const int32_t iTy = threadIdx.y;
	const int32_t addrx = iBx * nTx + iTx;
	const int32_t addry = iBy * nTy + iTy;
	TYPE max, input, tmp;
	__shared__ TYPE shm[SPLIT_T_NTX];

	if (addry < n){
		max = 0.;
		for (int32_t i = addrx; i < m; i += nTx) {
			input = devInput[addry * ldi + i];
			if (max < fabs(input)) max = fabs(input);
		}
		shm[iTx] = max;
		__syncthreads ();
		#pragma unroll
		for (int32_t i = nTx/2; i > 0; i>>=1) {
			tmp = shm[iTx+i];
			if (iTx < i && shm[iTx] < tmp) shm[iTx] = tmp;
			__syncthreads ();
		}
		if (iTx == 0) devMax[addry] = shm[0];
	} 
}

template <typename TYPE>
__host__
void cuozblasFindMaxDevice (
	const char major,
	const int32_t m, 
	const int32_t n,
	const TYPE * __restrict__ devInput,
	const int32_t ldi, 
	TYPE *devMax
) {
	int32_t ntx, nty, nbx, nby;
	dim3 threads, grid;
	if (major == 'r') {
		fprintf (OUTPUT, "OzBLAS error: cuozblasFindMaxRKernel is not available.\n");
	} else {
		ntx = SPLIT_T_NTX;
		nty = SPLIT_T_NTY;
		nbx = 1;
		nby = ceil (float(n) / nty);
		threads = dim3 (ntx, nty);
		grid = dim3 (nbx, nby);
		cuozblasFindMaxCKernel <<< grid, threads >>>(m, n, devInput, ldi, devMax);
	}
}

// max for m
template <typename TYPE>
__global__
void cuozblasSplitCKernel (
	const int32_t m,
	const int32_t n,
	const int32_t rho,
	const TYPE * __restrict__ devInput,
	const int32_t ldi,
	TYPE *devOutput,
	const int32_t ldo,
	TYPE *devSplit,
	const int32_t lds,
	short *devSpExp,
	TYPE *devMax,
	const int32_t splitShift
) {
	const int32_t iBx = blockIdx.x;
	const int32_t iBy = blockIdx.y;
	const int32_t nTx = SPLIT_T_NTX;//blockDim.x;
	const int32_t nTy = SPLIT_T_NTY;//blockDim.y;
	const int32_t iTx = threadIdx.x;
	const int32_t iTy = threadIdx.y;
	const int32_t addrx = iBx * nTx + iTx;
	const int32_t addry = iBy * nTy + iTy;
	__shared__ TYPE shm[SPLIT_T_NTX];

	if (addry < n){
		//const TYPE sigma = CONST * scalbn (1., rho) * NextPowTwo <TYPE> (devMax[addry]) * splitShift;
		const short tau = ceil(log2(fabs(devMax[addry])));
		const TYPE sigma = CONST * scalbn (1., rho + tau) * splitShift;
		TYPE max = 0.;
		for (int32_t i = addrx; i < m; i += nTx) {
			TYPE input = devInput[addry * ldi + i];
			//const TYPE tmp = (input + sigma);
			//const TYPE split = (tmp - sigma);
			const TYPE split = ((input + sigma) - sigma);
			input = input - split;
			devSplit[addry * lds + i] = split;
			devOutput[addry * ldo + i] = input;
			max = MAX(max, fabs(input));
		}
		shm[iTx] = max;
		__syncthreads ();
		#pragma unroll
		for (int32_t i = nTx/2; i > 0; i>>=1) {
			if (iTx < i && shm[iTx] < shm[iTx+i]) shm[iTx] = shm[iTx+i];
			__syncthreads ();
		}
		if (iTx == 0) devMax[addry] = shm[0];
	}
}

template <typename TYPE1, typename TYPE2>
__global__
void cuozblasSplitCKernel (
	const int32_t m,
	const int32_t n,
	const int32_t rho,
	const TYPE1 * __restrict__ devInput,
	const int32_t ldi,
	TYPE1 *devOutput,
	const int32_t ldo,
	TYPE2 *devSplit,
	const int32_t lds,
	short *devSpExp,
	TYPE1 *devMax,
	const int32_t splitShift
) {
	const int32_t iBx = blockIdx.x;
	const int32_t iBy = blockIdx.y;
	const int32_t nTx = SPLIT_T_NTX;//blockDim.x;
	const int32_t nTy = SPLIT_T_NTY;//blockDim.y;
	const int32_t iTx = threadIdx.x;
	const int32_t iTy = threadIdx.y;
	const int32_t addrx = iBx * nTx + iTx;
	const int32_t addry = iBy * nTy + iTy;
	__shared__ TYPE1 shm[SPLIT_T_NTX];

	if (addry < n){
		const short tau = devSpExp[addry] = ceil(log2(fabs(devMax[addry])));
		const TYPE1 sigma = CONST * scalbn (1., rho + tau) * splitShift;
		TYPE1 max = 0.;
		for (int32_t i = addrx; i < m; i += nTx) {
			TYPE1 input = devInput[addry * ldi + i];
			//const TYPE1 tmp = input + sigma;
			//const TYPE1 split = tmp - sigma;
			const TYPE1 split = (input + sigma) - sigma;
			input = input - split;
			devSplit[addry * lds + i] = scalbn(split, -tau);
			devOutput[addry * ldo + i] = input;
			max = MAX(max, fabs(input));
		}
		shm[iTx] = max;
		__syncthreads ();
		#pragma unroll
		for (int32_t i = nTx/2; i > 0; i>>=1) {
			if (iTx < i && shm[iTx] < shm[iTx+i]) shm[iTx] = shm[iTx+i];
			__syncthreads ();
		}
		if (iTx == 0) devMax[addry] = shm[0];
	}
}

template <typename TYPE>
__global__
void cuozblasSplitVecSubKernel (
	int32_t n,
	TYPE *devMax,
	const TYPE * __restrict__ devWorkCommon
) {
	const int32_t iTx = threadIdx.x;
	const int32_t nTx = SPLIT_VEC_NTX;//blockDim.x;
	__shared__ TYPE shm[SPLIT_VEC_NTX];

	TYPE max = 0.;
	if (iTx < n) max = devWorkCommon[iTx];
	//if (iTx < n) max = devMax[iTx];
	for (int32_t i = iTx; i < n; i += nTx) {
	//for (int32_t i = nTx + iTx; i < n; i += nTx) {
		TYPE tmp = devWorkCommon[i];
		max = MAX(max, fabs(tmp));
	}
	shm[iTx] = max;
	__syncthreads ();
	#pragma unroll
	for (int32_t i = nTx/2; i > 0; i >>= 1) {
		if (iTx < i && shm[iTx] < shm[iTx+i]) shm[iTx] = shm[iTx+i];
		__syncthreads ();
	}
	if (iTx == 0) devMax[0] = shm[0];
}

template <typename TYPE>
__global__
void cuozblasSplitVecKernel (
	const int32_t n,
	const int32_t rho,
	const TYPE * __restrict__ devInput,
	TYPE *devOutput,
	TYPE *devSplit,
	short *devSpExp, // ignored
	TYPE *devMax,
	TYPE *devWorkCommon,
	const int32_t splitShift
) {
	const int32_t iTx = threadIdx.x;
	const int32_t iBx = blockIdx.x;
	const int32_t nTx = SPLIT_VEC_NTX;//blockDim.x;
	const int32_t addrx = iBx * nTx + iTx;
	__shared__ TYPE shm[SPLIT_VEC_NTX];

	//const TYPE sigma = CONST * scalbn (1., rho) * NextPowTwo <TYPE> (devMax[0]) * splitShift;
	const short tau = ceil(log2(fabs(devMax[0])));
	const TYPE sigma = CONST * scalbn (1., rho + tau) * splitShift;
	TYPE max_ = 0.;
	for (int32_t i = addrx; i < n; i += nTx * gridDim.x) {
		TYPE input = devInput[i];
		const TYPE split = (input + sigma) - sigma;
		input = input - split;
		devSplit[i] = split;
		devOutput[i] = input;
		max_ = MAX(max_, fabs(input));
	}
	shm[iTx] = max_;
	__syncthreads ();
	#pragma unroll
	for (int32_t i = nTx/2; i > 0; i>>=1) {
		if (iTx < i && shm[iTx] < shm[iTx+i]) shm[iTx] = shm[iTx+i];
		__syncthreads ();
	}
	if (iTx == 0) devWorkCommon[iBx] = shm[0];
}

template <typename TYPE1, typename TYPE2>
__global__
void cuozblasSplitVecKernel (
	const int32_t n,
	const int32_t rho,
	const TYPE1 * __restrict__ devInput,
	TYPE1 *devOutput,
	TYPE2 *devSplit,
	short *devSpExp,
	TYPE1 *devMax,
	TYPE1 *devWorkCommon,
	const int32_t splitShift
) {
	const int32_t iTx = threadIdx.x;
	const int32_t iBx = blockIdx.x;
	const int32_t nTx = SPLIT_VEC_NTX;//blockDim.x;
	const int32_t addrx = iBx * nTx + iTx;
	__shared__ TYPE1 shm[SPLIT_VEC_NTX];

	const short tau = devSpExp[0] = ceil(log2(fabs(devMax[0]))); // note: here, devMax needs fabs
	const TYPE1 sigma = CONST * scalbn (1., rho + tau) * splitShift;
	TYPE1 max_ = 0.;
	for (int32_t i = addrx; i < n; i += nTx * gridDim.x) {
		TYPE1 input = devInput[i];
		const TYPE1 split = (input + sigma) - sigma;
		input = input - split;
		devSplit[i] = scalbn (split, -tau);
		devOutput[i] = input;
		max_ = MAX(max_, fabs(input));
	}
	shm[iTx] = max_;
	__syncthreads ();
	#pragma unroll
	for (int32_t i = nTx/2; i > 0; i>>=1) {
		if (iTx < i && shm[iTx] < shm[iTx+i]) shm[iTx] = shm[iTx+i];
		__syncthreads ();
	}
	if (iTx == 0) devWorkCommon[iBx] = shm[0];
}

template <typename TYPE1, typename TYPE2>
__host__
void cuozblasSplitDevice (
	cuozblasHandle_t *oh,
	const char major,
	const int32_t m, 
	const int32_t n,
	const TYPE1 *devInput, // input matrix (devAwrk) 
	const int32_t ldi, // leading dimension of input matrix
	TYPE1 *devOutput, // output matrix (devAwrk)
	const int32_t ldo,
	TYPE2 *devSplit, // split matrices (output): this includes NumSplitMax matrices
	const int32_t lds, // leading dimension of split matrix (# of cols)
	short *devSpExp, // exponent of split matrix
	TYPE1 *devMax,
	int32_t splitEpsModeFlag,
	const int32_t splitShift
) {
	int32_t ntx, nty, nbx, nby;
	dim3 threads, grid;
	const int32_t dim = (major == 'r') ? n : m;
	const int32_t rho = getRho <TYPE1, TYPE2> (dim, splitEpsModeFlag);

	if ((major == 'r' && m == 1) || (major == 'c' && n == 1)) {
		ntx = SPLIT_VEC_NTX;
		threads = dim3 (ntx);
		nbx = MIN (ceil((float)dim/ntx), SPLIT_VEC_NBX); 
		grid = dim3 (nbx);
		TYPE1 *work = (TYPE1*)oh->devWorkCommon;
		cuozblasSplitVecKernel <<< grid, threads >>> (dim, rho, devInput, devOutput, devSplit, devSpExp, devMax, work, splitShift);
		grid = dim3 (1);
		cuozblasSplitVecSubKernel <<< grid, threads >>> (nbx, devMax, work);
	} else {
		if (major == 'r') {
			fprintf (OUTPUT, "OzBLAS error: cuozblasSplitRKernel is not available.\n");
		} else {
			ntx = SPLIT_T_NTX;
			nty = SPLIT_T_NTY;
			nbx = 1;
			nby = ceil (float(n) / nty);
			threads = dim3 (ntx, nty);
			grid = dim3 (nbx, nby);
			cuozblasSplitCKernel <<< grid, threads >>> (m, n, rho, devInput, ldi, devOutput, ldo, devSplit, lds, devSpExp, devMax, splitShift);
		} 
	}
}

template <typename TYPE1, typename TYPE2>
__host__
int32_t cuozblasSplit (
	cuozblasHandle_t *oh,
	const char major,
	const int32_t m,
	const int32_t n,
	const TYPE1 *devInput,
	const int32_t ldi,
	TYPE1 *devOutput,
	const int32_t ldo,
	TYPE2 *devSplit,
	const int32_t lds,
	short *devSpExp, 
	const int32_t ldse,
	TYPE1 *devMax
) {
	// FindMax^(0)
	if ((major == 'r' && m == 1) || (major == 'c' && n == 1)) {
		int32_t ptrMax = 0;
		blasRiamax (oh->ch, ((major == 'r') ? n : m), devInput, 1, &ptrMax);
		cudaMemcpy (devMax, &devInput[ptrMax-1], sizeof(TYPE1), cudaMemcpyDeviceToDevice);
	} else {
		cuozblasFindMaxDevice (major, m, n, devInput, ldi, devMax);
	}
	TYPE1 *hst = new TYPE1[m];

	// Split^(0) & FindMax^(1)
	cuozblasSplitDevice (oh, major, m, n, devInput, ldi, devOutput, ldo, devSplit, lds, devSpExp, devMax, oh->splitEpsModeFlag, oh->splitShift);
	const int32_t maxS = (oh->nSplitMax > 0) ? oh->nSplitMax : NumSplitDefaultMax;
	int32_t s;
	for (s = 1; s < maxS; s++) {
		TYPE1 check = 0.;
		if ((major == 'r' && m == 1) || (major == 'c' && n == 1))
			cudaMemcpy (&check, devMax, sizeof(TYPE1), cudaMemcpyDeviceToHost);
		else
			blasRasum (oh->ch, ((major == 'r') ? m : n), devMax, 1, &check);
		if (check == 0.) return s;
		// Split^(i) & FindMax^(i+1)
		cuozblasSplitDevice (oh, major, m, n, devOutput, ldo, devOutput, ldo, &devSplit[lds*n*s], lds, &devSpExp[ldse*s], devMax, oh->splitEpsModeFlag, oh->splitShift);
	}
	if (oh->splitModeFlag > 0)
		fprintf (OUTPUT, "OzBLAS error: infSplit is failed.\n");

	delete[] hst;
	return s;
}
template int32_t cuozblasSplit <double, double> (cuozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const double *devInput, const int32_t ldi, double *devOutput, const int32_t ldo, double *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, double *devMax);
template int32_t cuozblasSplit <double, float> (cuozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const double *devInput, const int32_t ldi, double *devOutput, const int32_t ldo, float *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, double *devMax);
template int32_t cuozblasSplit <float, float> (cuozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const float *devInput, const int32_t ldi, float *devOutput, const int32_t ldo, float *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, float *devMax);
template int32_t cuozblasSplit <float, double> (cuozblasHandle_t *oh, const char major, const int32_t m, const int32_t n, const float *devInput, const int32_t ldi, float *devOutput, const int32_t ldo, double *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, float *devMax);

// =========================================
// Sparse
// =========================================

template <typename TYPE>
__global__
void cuozblasFindMaxSparseNKernel (
	const int32_t m,
	const TYPE * __restrict__ devInput,
	const int32_t * __restrict__ devRowptr,
	TYPE *devMax
) {
	const int32_t iBx = blockIdx.x;
	const int32_t iTx = threadIdx.x;
	const int32_t tid = SPLIT_T_NTX * iBx + iTx;
	const int32_t rowid = tid / 32;
	const int32_t lane  = tid % 32; 
	__shared__ TYPE shm[SPLIT_T_NTX];

	if (rowid < m){
		TYPE max = 0.;
		for (int32_t i = devRowptr[rowid] + lane; i < devRowptr[rowid+1]; i += 32) 
			max = MAX(max, fabs(devInput[i]));
		shm[(iTx/32)*32+lane] = max;
		__syncthreads ();
		if (lane == 0) {
			max = shm[(iTx/32)*32];
			#pragma unroll
			for (int32_t i = 1; i < 32; i++) 
				max = MAX(max, fabs(shm[(iTx/32)*32+i]));
			devMax[rowid] = max;
		}
	}
}

template <typename TYPE>
__host__
void cuozblasFindMaxSparseDevice (
	const char major,
	const int32_t m, 
	const TYPE *devInput,
	const int32_t *devRowptr,
	TYPE *devMax
) {
	if (major == 'r') {
		int32_t ntx = SPLIT_T_NTX;
		int32_t nbx = ceil (float(m) / (ntx/32));
		dim3 threads = dim3 (ntx);
		dim3 grid = dim3 (nbx);
		cuozblasFindMaxSparseNKernel <<< grid, threads >>> (m, devInput, devRowptr, devMax);
	} else {
		fprintf (OUTPUT, "OzBLAS error: Split-T is not implemented.\n");
		exit (1);
	}
}

template <typename TYPE>
__global__
void cuozblasSplitSparseNKernel (
	const int32_t m,
	const TYPE * __restrict__ devInput,
	const int32_t * __restrict__ devRowptr,
	TYPE *devOutput,
	TYPE *devSplit,
	short *devSpExp, // ignored
	TYPE *devMax,
	const int32_t splitEpsModeFlag,
	const int32_t splitShift
) {
	const int32_t iBx = blockIdx.x;
	const int32_t iTx = threadIdx.x;
	const int32_t tid = SPLIT_T_NTX * iBx + iTx;
	const int32_t rowid = tid / 32;
	const int32_t lane  = tid % 32; 
	__shared__ TYPE shm[SPLIT_T_NTX];

	if (rowid < m){
		const int32_t dim = devRowptr[rowid+1] - devRowptr[rowid];
		const int32_t rho = getRho <TYPE, TYPE> (dim, splitEpsModeFlag);
		//const TYPE sigma = CONST * scalbn (1., rho) * NextPowTwo <TYPE> (devMax[rowid]) / splitShift;
		const short tau = ceil(log2(fabs(devMax[rowid])));
		const TYPE sigma = CONST * scalbn (1., rho + tau) / splitShift;
		TYPE max = 0.;
		for (int32_t i = devRowptr[rowid] + lane; i < devRowptr[rowid+1]; i += 32) {
			TYPE input = devInput[i];
			const TYPE split = (input + sigma) - sigma;
			input = input - split;
			devSplit[i] = split;
			devOutput[i] = input;
			max = MAX(max, fabs(input));
		}
		shm[(iTx/32)*32+lane] = max;
		__syncthreads ();
		if (lane == 0) {
			max = shm[(iTx/32)*32];
			#pragma unroll
			for (int32_t i = 1; i < 32; i++) 
				max = MAX(max, fabs(shm[(iTx/32)*32+i]));
			devMax[rowid] = max;
		}
	}
}

template <typename TYPE1, typename TYPE2>
__global__
void cuozblasSplitSparseNKernel (
	const int32_t m,
	const TYPE1 * __restrict__ devInput,
	const int32_t * __restrict__ devRowptr,
	TYPE1 *devOutput,
	TYPE2 *devSplit,
	short *devSpExp,
	TYPE1 *devMax,
	const int32_t splitEpsModeFlag,
	const int32_t splitShift
) {
	const int32_t iBx = blockIdx.x;
	const int32_t iTx = threadIdx.x;
	const int32_t tid = SPLIT_T_NTX * iBx + iTx;
	const int32_t rowid = tid / 32;
	const int32_t lane  = tid % 32; 
	__shared__ TYPE1 shm[SPLIT_T_NTX];

	if (rowid < m){
		const int32_t dim = devRowptr[rowid+1] - devRowptr[rowid];
		const int32_t rho = getRho <TYPE1, TYPE2> (dim, splitEpsModeFlag);
		const short tau = devSpExp[rowid] = ceil(log2(devMax[rowid]));
		const TYPE1 sigma = CONST * scalbn (1., rho+tau) / splitShift;
		TYPE1 max = 0.;
		for (int32_t i = devRowptr[rowid] + lane; i < devRowptr[rowid+1]; i += 32) {
			TYPE1 input = devInput[i];
			const TYPE1 split = (input + sigma) - sigma;
			input = input - split;
			devSplit[i] = scalbn(split, -tau);
			devOutput[i] = input;
			max = MAX(max, fabs(input));
		}
		shm[(iTx/32)*32+lane] = max;
		__syncthreads ();
		if (lane == 0) {
			max = shm[(iTx/32)*32];
			#pragma unroll
			for (int32_t i = 1; i < 32; i++) 
				max = MAX(max, fabs(shm[(iTx/32)*32+i]));
			devMax[rowid] = max;
		}
	}
}

template <typename TYPE1, typename TYPE2>
__host__
void cuozblasSplitSparseDevice (
	const char major,
	const int32_t m, 
	const TYPE1 *devInput, 
	const int32_t *devRowptr,
	TYPE1 *devOutput,
	TYPE2 *devSplit, 
	const int32_t lds, 
	short *devSpExp, 
	TYPE1 *devMax,
	int32_t splitEpsModeFlag,
	const int32_t splitShift
) {
	if (major == 'r') {
		int32_t ntx = SPLIT_T_NTX;
		int32_t nbx = ceil (float(m) / (ntx/32));
		dim3 threads = dim3 (ntx);
		dim3 grid = dim3 (nbx);
		cuozblasSplitSparseNKernel <<< grid, threads >>> (m, devInput, devRowptr, devOutput, devSplit, devSpExp, devMax, splitEpsModeFlag, splitShift);
	} else {
		fprintf (OUTPUT, "OzBLAS error: SplitT is not implemented.\n");
		exit (1);
	}
}

template <typename TYPE1, typename TYPE2>
__host__
int32_t cuozblasSplitSparse (
	cuozblasHandle_t *oh,
	const char major,
	const int32_t m,
	const TYPE1 *devInput,
	const int32_t *devRowptr,
	TYPE1 *devOutput,
	TYPE2 *devSplit,
	const int32_t lds,
	short *devSpExp, 
	const int32_t ldse,
	TYPE1 *devMax
) {
	// cuozblasFindMax^(0)
	cuozblasFindMaxSparseDevice (major, m, devInput, devRowptr, devMax);
	// Split^(0) & cuozblasFindMax^(1)
	cuozblasSplitSparseDevice (major, m, devInput, devRowptr, devOutput, &devSplit[0], lds, devSpExp, devMax, oh->splitEpsModeFlag, oh->splitShift);
	const int32_t maxS = (oh->nSplitMax > 0) ? oh->nSplitMax : NumSplitDefaultMax;
	int32_t s;
	for (s = 1; s < maxS; s++) {
		TYPE1 check;
		blasRasum (oh->ch, m, devMax, 1, &check);
		if (check == 0) return s;
		// Split^(i) & cuozblasFindMax^(i+1)
		cuozblasSplitSparseDevice (major, m, devOutput, devRowptr, devOutput, &devSplit[lds*s], lds, &devSpExp[ldse*s], devMax, oh->splitEpsModeFlag, oh->splitShift);
	}
	if (oh->splitModeFlag > 0)
		fprintf (OUTPUT, "OzBLAS error: infSplit is failed.\n");
	return s;
}
template int32_t cuozblasSplitSparse <double, double> (cuozblasHandle_t *oh, const char major, const int32_t m, const double *devInput, const int32_t *devRowptr, double *devOutput, double *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, double *devMax);
template int32_t cuozblasSplitSparse <double, float> (cuozblasHandle_t *oh, const char major, const int32_t m, const double *devInput, const int32_t *devRowptr, double *devOutput, float *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, double *devMax);
template int32_t cuozblasSplitSparse <float, float> (cuozblasHandle_t *oh, const char major, const int32_t m, const float *devInput, const int32_t *devRowptr, float *devOutput, float *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, float *devMax);
template int32_t cuozblasSplitSparse <float, double> (cuozblasHandle_t *oh, const char major, const int32_t m, const float *devInput, const int32_t *devRowptr, float *devOutput, double *devSplit, const int32_t lds, short *devSpExp, const int32_t ldse, float *devMax);

