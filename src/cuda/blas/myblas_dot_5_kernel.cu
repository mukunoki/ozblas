#include "../cueft.h"

// ================================
// SUM
// ================================

__global__ void
myblas_sum_x2_kernel_5 (
	const size_t n,
	const double2 * __restrict__ w,
	const size_t ldw,
	float *result
) {
}

__global__ void
myblas_sum_x2_kernel_5 (
	const size_t n,
	const double2 * __restrict__ w,
	const size_t ldw,
	double *result
) {
	const size_t iTx = threadIdx.x;
	const size_t iBx = blockIdx.x;
	const size_t nBx = gridDim.x;
	const size_t nTx = MYGEMM_NTX5;//blockDim.x;
	const size_t addrx = iBx * nTx + iTx;
	size_t i, ix, iy;
	__shared__ double2 Ts2[5*5*MYGEMM_NTX5];
	double2 Tr2[5*5];

	#pragma unroll
	for (ix = 0; ix < 5; ix++) {
		#pragma unroll
		for (iy = 0; iy < 5; iy++) {
			Tr2[iy*5+ix] = make_double2 (0., 0.);
		}
	}

	for (i = addrx; i < n; i += nTx * nBx) {
		#pragma unroll
		for (ix = 0; ix < 5; ix++) {
			#pragma unroll
			for (iy = 0; iy < 5; iy++) {
				Tr2[iy*5+ix] = cuQuadAdd (Tr2[iy*5+ix], w[ldw*(iy*5+ix)+i]);
			}
		}
	}
	#pragma unroll
	for (ix = 0; ix < 5; ix++) {
		#pragma unroll
		for (iy = 0; iy < 5; iy++) {
			Ts2[nTx*(iy*5+ix)+iTx] = Tr2[iy*5+ix];
		}
	}
	__syncthreads ();

	#pragma unroll
	for (i = nTx/2; i > 0; i >>= 1) {
		if (iTx < i) {
			#pragma unroll
			for (ix = 0; ix < 5; ix++) {
				#pragma unroll
				for (iy = 0; iy < 5; iy++) {
					Ts2[nTx*(iy*5+ix)+iTx] = cuQuadAdd (Ts2[nTx*(iy*5+ix)+iTx], Ts2[nTx*(iy*5+ix)+iTx+i]);
				}
			}
		}
		__syncthreads ();
	}
	if (iTx == 0 && iBx == 0) {
		#pragma unroll
		for (ix = 0; ix < 5; ix++) {
			#pragma unroll
			for (iy = 0; iy < 5; iy++) {
				result[iy*5+ix] = Ts2[nTx*(iy*5+ix)].x;
				result[iy*5+ix+5*5] = Ts2[nTx*(iy*5+ix)].y;
			}
		}
	}
}

__global__ void
myblas_sum_x_kernel_5 (
	const size_t n,
	const double2 * __restrict__ w,
	const size_t ldw,
	float *result
) {
}

__global__ void
myblas_sum_x_kernel_5 (
	const size_t n,
	const double2 * __restrict__ w,
	const size_t ldw,
	double *result
) {
	const size_t iTx = threadIdx.x;
	const size_t iBx = blockIdx.x;
	const size_t nBx = gridDim.x;
	const size_t nTx = MYGEMM_NTX5;//blockDim.x;
	const size_t addrx = iBx * nTx + iTx;
	size_t i, ix, iy;
	__shared__ double2 Ts2[5*5*MYGEMM_NTX5];
	double2 Tr2[5*5];

	#pragma unroll
	for (ix = 0; ix < 5; ix++) {
		#pragma unroll
		for (iy = 0; iy < 5; iy++) {
			Tr2[iy*5+ix] = make_double2 (0., 0.);
		}
	}

	for (i = addrx; i < n; i += nTx * nBx) {
		#pragma unroll
		for (ix = 0; ix < 5; ix++) {
			#pragma unroll
			for (iy = 0; iy < 5; iy++) {
				Tr2[iy*5+ix] = cuQuadAdd (Tr2[iy*5+ix], w[ldw*(iy*5+ix)+i]);
			}
		}
	}
	#pragma unroll
	for (ix = 0; ix < 5; ix++) {
		#pragma unroll
		for (iy = 0; iy < 5; iy++) {
			Ts2[nTx*(iy*5+ix)+iTx] = Tr2[iy*5+ix];
		}
	}
	__syncthreads ();

	#pragma unroll
	for (i = nTx/2; i > 0; i >>= 1) {
		if (iTx < i) {
			#pragma unroll
			for (ix = 0; ix < 5; ix++) {
				#pragma unroll
				for (iy = 0; iy < 5; iy++) {
					Ts2[nTx*(iy*5+ix)+iTx] = cuQuadAdd (Ts2[nTx*(iy*5+ix)+iTx], Ts2[nTx*(iy*5+ix)+iTx+i]);
				}
			}
		}
		__syncthreads ();
	}
	if (iTx == 0) {
		#pragma unroll
		for (ix = 0; ix < 5; ix++) {
			#pragma unroll
			for (iy = 0; iy < 5; iy++) {
				result[nBx*(iy*5+ix)+iBx] = Ts2[nTx*(iy*5+ix)].x + Ts2[nTx*(iy*5+ix)].y;
			}
		}
	}
}

template <typename TYPE>
__global__ void
myblas_sum_kernel_5 (
	const size_t n,
	const TYPE *w,
	const size_t ldw,
	TYPE *result
) {
	const size_t iTx = threadIdx.x;
	const size_t iBx = blockIdx.x;
	const size_t nBx = gridDim.x;
	const size_t nTx = MYGEMM_NTX5;//blockDim.x;
	const size_t addrx = iBx * nTx + iTx;
	size_t i, ix, iy;
	__shared__ TYPE Ts1[5*5*MYGEMM_NTX5];
	TYPE Tr1[5*5] = {0.};

	for (i = addrx; i < n; i += nTx * nBx) {
		#pragma unroll
		for (ix = 0; ix < 5; ix++) {
			#pragma unroll
			for (iy = 0; iy < 5; iy++) {
				Tr1[iy*5+ix] += w[ldw*(iy*5+ix)+i];
			}
		}
	}
	#pragma unroll
	for (ix = 0; ix < 5; ix++) {
		#pragma unroll
		for (iy = 0; iy < 5; iy++) {
			Ts1[nTx*(iy*5+ix)+iTx] = Tr1[iy*5+ix];
		}
	}
	__syncthreads ();

	#pragma unroll
	for (i = nTx/2; i > 0; i >>= 1) {
		if (iTx < i) {
			#pragma unroll
			for (ix = 0; ix < 5; ix++) {
				#pragma unroll
				for (iy = 0; iy < 5; iy++) {
					Ts1[nTx*(iy*5+ix)+iTx] += Ts1[nTx*(iy*5+ix)+iTx+i];
				}
			}
		}
		__syncthreads ();
	}
	if (iTx == 0 && iBx == 0) {
		#pragma unroll
		for (ix = 0; ix < 5; ix++) {
			#pragma unroll
			for (iy = 0; iy < 5; iy++) {
				result[iy*5+ix] = Ts1[nTx*(iy*5+ix)];
			}
		}
	}
}

// ================================
// DOT
// ================================

template <typename TYPE>
__global__ void
myblas_dot_kernel_5 (
	const size_t n,
	const TYPE * __restrict__ x,
	const size_t ldx,
	const TYPE * __restrict__ y,
	const size_t ldy,
	TYPE *w,
	const size_t ldw
) {
	const size_t iTx = threadIdx.x;
	const size_t iBx = blockIdx.x;
	const size_t nBx = gridDim.x;
	const size_t nTx = MYGEMM_NTX5;//blockDim.x;
	const size_t addrx = iBx * nTx + iTx;
	size_t i, ix, iy;
	__shared__ TYPE Ts1[5*5*MYGEMM_NTX5];
	TYPE Xr1[5], Yr1[5];
	TYPE Tr1[5*5] = {0.};

	for (i = addrx; i < n; i += nTx * nBx) {
		#pragma unroll
		for (ix = 0; ix < 5; ix++) {
			Xr1[ix] = x[ix * ldx + i];
		}
		#pragma unroll
		for (iy = 0; iy < 5; iy++) {
			Yr1[iy] = y[iy * ldy + i];
		}
		#pragma unroll
		for (ix = 0; ix < 5; ix++) {
			#pragma unroll
			for (iy = 0; iy < 5; iy++) {
				Tr1[iy*5+ix] += Xr1[ix] * Yr1[iy];
			}
		}
	}
	#pragma unroll
	for (ix = 0; ix < 5; ix++) {
		#pragma unroll
		for (iy = 0; iy < 5; iy++) {
			Ts1[nTx*(iy*5+ix)+iTx] = Tr1[iy*5+ix];
		}
	}
	__syncthreads ();

	#pragma unroll
	for (i = nTx/2; i > 0; i >>= 1) {
		if (iTx < i) {
			#pragma unroll
			for (ix = 0; ix < 5; ix++) {
				#pragma unroll
				for (iy = 0; iy < 5; iy++) {
					Ts1[nTx*(iy*5+ix)+iTx] += Ts1[nTx*(iy*5+ix)+iTx+i];
				}
			}
		}
		__syncthreads ();
	}

	if (iTx == 0) {
		#pragma unroll
		for (ix = 0; ix < 5; ix++) {
			#pragma unroll
			for (iy = 0; iy < 5; iy++) {
				w[ldw*(iy*5+ix)+iBx] = Ts1[nTx*(iy*5+ix)];
			}
		}
	}
}

__global__ void
myblas_dot_x_kernel_5 (
	const size_t n,
	const float * __restrict__ x,
	const size_t ldx,
	const float * __restrict__ y,
	const size_t ldy,
	double2 *w,
	const size_t ldw
) {
}

__global__ void
myblas_dot_x_kernel_5 (
	const size_t n,
	const double * __restrict__ x,
	const size_t ldx,
	const double * __restrict__ y,
	const size_t ldy,
	double2 *w,
	const size_t ldw
) {
	const size_t iTx = threadIdx.x;
	const size_t iBx = blockIdx.x;
	const size_t nBx = gridDim.x;
	const size_t nTx = MYGEMM_NTX5;//blockDim.x;
	const size_t addrx = iBx * nTx + iTx;
	size_t i, ix, iy;
	register double Xr1[5], Yr1[5];
	register double2 Tr2[5*5];
	__shared__ double2 Ts2[5*5*MYGEMM_NTX5];

	#pragma unroll
	for (ix = 0; ix < 5; ix++) {
		#pragma unroll
		for (iy = 0; iy < 5; iy++) {
			Tr2[iy*5+ix] = make_double2 (0., 0.);
		}
	}

	// DOT part -----------------------------------------------
	for (i = addrx; i < n; i += nTx * nBx) {
		#pragma unroll
		for (ix = 0; ix < 5; ix++) 
			Xr1[ix] = x[ix * ldx + i];
		#pragma unroll
		for (iy = 0; iy < 5; iy++) 
			Yr1[iy] = y[iy * ldy + i];
		#pragma unroll
		for (ix = 0; ix < 5; ix++) {
			#pragma unroll
			for (iy = 0; iy < 5; iy++) {
				cuDot2i (Xr1[ix], Yr1[iy], Tr2[iy*5+ix]);
			}
		}
	}
	#pragma unroll
	for (ix = 0; ix < 5; ix++) {
		#pragma unroll
		for (iy = 0; iy < 5; iy++) {
			Ts2[nTx*(iy*5+ix)+iTx] = Tr2[iy*5+ix];
		}
	}
	__syncthreads ();

	// SUM part -----------------------------------------------
	#pragma unroll
	for (i = nTx/2; i > 0; i >>= 1) {
		if (iTx < i) {
			#pragma unroll
			for (ix = 0; ix < 5; ix++) {
				#pragma unroll
				for (iy = 0; iy < 5; iy++) {
					Ts2[nTx*(iy*5+ix)+iTx] = cuQuadAdd (Ts2[nTx*(iy*5+ix)+iTx], Ts2[nTx*(iy*5+ix)+iTx+i]);
				}
			}
		}
		__syncthreads ();
	}

	if (iTx == 0) {
		#pragma unroll
		for (ix = 0; ix < 5; ix++) {
			#pragma unroll
			for (iy = 0; iy < 5; iy++) {
				w[ldw*(iy*5+ix)+iBx] = Ts2[nTx*(iy*5+ix)];
			}
		}
	}
}



