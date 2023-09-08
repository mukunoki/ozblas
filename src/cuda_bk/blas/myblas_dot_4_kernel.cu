#include "../cueft.h"

// ================================
// SUM
// ================================

__global__ void
myblas_sum_x2_kernel_4 (
	const size_t n,
	const double2 * __restrict__ w,
	const size_t ldw,
	float *result
) {
}

__global__ void
myblas_sum_x2_kernel_4 (
	const size_t n,
	const double2 * __restrict__ w,
	const size_t ldw,
	double *result
) {
	const size_t iTx = threadIdx.x;
	const size_t iBx = blockIdx.x;
	const size_t nBx = gridDim.x;
	const size_t nTx = MYGEMM_NTX4;//blockDim.x;
	const size_t addrx = iBx * nTx + iTx;
	size_t i, ix, iy;
	__shared__ double2 Ts2[4*4*MYGEMM_NTX4];
	double2 Tr2[4*4];

	#pragma unroll
	for (ix = 0; ix < 4; ix++) {
		#pragma unroll
		for (iy = 0; iy < 4; iy++) {
			Tr2[iy*4+ix] = make_double2 (0., 0.);
		}
	}

	for (i = addrx; i < n; i += nTx * nBx) {
		#pragma unroll
		for (ix = 0; ix < 4; ix++) {
			#pragma unroll
			for (iy = 0; iy < 4; iy++) {
				Tr2[iy*4+ix] = cuQuadAdd (Tr2[iy*4+ix], w[ldw*(iy*4+ix)+i]);
			}
		}
	}
	#pragma unroll
	for (ix = 0; ix < 4; ix++) {
		#pragma unroll
		for (iy = 0; iy < 4; iy++) {
			Ts2[nTx*(iy*4+ix)+iTx] = Tr2[iy*4+ix];
		}
	}
	__syncthreads ();

	#pragma unroll
	for (i = nTx/2; i > 0; i >>= 1) {
		if (iTx < i) {
			#pragma unroll
			for (ix = 0; ix < 4; ix++) {
				#pragma unroll
				for (iy = 0; iy < 4; iy++) {
					Ts2[nTx*(iy*4+ix)+iTx] = cuQuadAdd (Ts2[nTx*(iy*4+ix)+iTx], Ts2[nTx*(iy*4+ix)+iTx+i]);
				}
			}
		}
		__syncthreads ();
	}
	if (iTx == 0 && iBx == 0) {
		#pragma unroll
		for (ix = 0; ix < 4; ix++) {
			#pragma unroll
			for (iy = 0; iy < 4; iy++) {
				result[iy*4+ix] = Ts2[nTx*(iy*4+ix)].x;
				result[iy*4+ix+4*4] = Ts2[nTx*(iy*4+ix)].y;
			}
		}
	}
}

__global__ void
myblas_sum_x_kernel_4 (
	const size_t n,
	const double2 * __restrict__ w,
	const size_t ldw,
	float *result
) {
}

__global__ void
myblas_sum_x_kernel_4 (
	const size_t n,
	const double2 * __restrict__ w,
	const size_t ldw,
	double *result
) {
	const size_t iTx = threadIdx.x;
	const size_t iBx = blockIdx.x;
	const size_t nBx = gridDim.x;
	const size_t nTx = MYGEMM_NTX4;//blockDim.x;
	const size_t addrx = iBx * nTx + iTx;
	size_t i, ix, iy;
	__shared__ double2 Ts2[4*4*MYGEMM_NTX4];
	double2 Tr2[4*4];

	#pragma unroll
	for (ix = 0; ix < 4; ix++) {
		#pragma unroll
		for (iy = 0; iy < 4; iy++) {
			Tr2[iy*4+ix] = make_double2 (0., 0.);
		}
	}

	for (i = addrx; i < n; i += nTx * nBx) {
		#pragma unroll
		for (ix = 0; ix < 4; ix++) {
			#pragma unroll
			for (iy = 0; iy < 4; iy++) {
				Tr2[iy*4+ix] = cuQuadAdd (Tr2[iy*4+ix], w[ldw*(iy*4+ix)+i]);
			}
		}
	}
	#pragma unroll
	for (ix = 0; ix < 4; ix++) {
		#pragma unroll
		for (iy = 0; iy < 4; iy++) {
			Ts2[nTx*(iy*4+ix)+iTx] = Tr2[iy*4+ix];
		}
	}
	__syncthreads ();

	#pragma unroll
	for (i = nTx/2; i > 0; i >>= 1) {
		if (iTx < i) {
			#pragma unroll
			for (ix = 0; ix < 4; ix++) {
				#pragma unroll
				for (iy = 0; iy < 4; iy++) {
					Ts2[nTx*(iy*4+ix)+iTx] = cuQuadAdd (Ts2[nTx*(iy*4+ix)+iTx], Ts2[nTx*(iy*4+ix)+iTx+i]);
				}
			}
		}
		__syncthreads ();
	}
	if (iTx == 0) {
		#pragma unroll
		for (ix = 0; ix < 4; ix++) {
			#pragma unroll
			for (iy = 0; iy < 4; iy++) {
				result[nBx*(iy*4+ix)+iBx] = Ts2[nTx*(iy*4+ix)].x + Ts2[nTx*(iy*4+ix)].y;
			}
		}
	}
}

template <typename TYPE>
__global__ void
myblas_sum_kernel_4 (
	const size_t n,
	const TYPE *w,
	const size_t ldw,
	TYPE *result
) {
	const size_t iTx = threadIdx.x;
	const size_t iBx = blockIdx.x;
	const size_t nBx = gridDim.x;
	const size_t nTx = MYGEMM_NTX4;//blockDim.x;
	const size_t addrx = iBx * nTx + iTx;
	size_t i, ix, iy;
	__shared__ TYPE Ts1[4*4*MYGEMM_NTX4];
	TYPE Tr1[4*4] = {0.};

	for (i = addrx; i < n; i += nTx * nBx) {
		#pragma unroll
		for (ix = 0; ix < 4; ix++) {
			#pragma unroll
			for (iy = 0; iy < 4; iy++) {
				Tr1[iy*4+ix] += w[ldw*(iy*4+ix)+i];
			}
		}
	}
	#pragma unroll
	for (ix = 0; ix < 4; ix++) {
		#pragma unroll
		for (iy = 0; iy < 4; iy++) {
			Ts1[nTx*(iy*4+ix)+iTx] = Tr1[iy*4+ix];
		}
	}
	__syncthreads ();

	#pragma unroll
	for (i = nTx/2; i > 0; i >>= 1) {
		if (iTx < i) {
			#pragma unroll
			for (ix = 0; ix < 4; ix++) {
				#pragma unroll
				for (iy = 0; iy < 4; iy++) {
					Ts1[nTx*(iy*4+ix)+iTx] += Ts1[nTx*(iy*4+ix)+iTx+i];
				}
			}
		}
		__syncthreads ();
	}
	if (iTx == 0 && iBx == 0) {
		#pragma unroll
		for (ix = 0; ix < 4; ix++) {
			#pragma unroll
			for (iy = 0; iy < 4; iy++) {
				result[iy*4+ix] = Ts1[nTx*(iy*4+ix)];
			}
		}
	}
}

// ================================
// DOT
// ================================

template <typename TYPE>
__global__ void
myblas_dot_kernel_4 (
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
	const size_t nTx = MYGEMM_NTX4;//blockDim.x;
	const size_t addrx = iBx * nTx + iTx;
	size_t i, ix, iy;
	__shared__ TYPE Ts1[4*4*MYGEMM_NTX4];
	TYPE Xr1[4], Yr1[4];
	TYPE Tr1[4*4] = {0.};

	for (i = addrx; i < n; i += nTx * nBx) {
		#pragma unroll
		for (ix = 0; ix < 4; ix++) {
			Xr1[ix] = x[ix * ldx + i];
		}
		#pragma unroll
		for (iy = 0; iy < 4; iy++) {
			Yr1[iy] = y[iy * ldy + i];
		}
		#pragma unroll
		for (ix = 0; ix < 4; ix++) {
			#pragma unroll
			for (iy = 0; iy < 4; iy++) {
				Tr1[iy*4+ix] += Xr1[ix] * Yr1[iy];
			}
		}
	}
	#pragma unroll
	for (ix = 0; ix < 4; ix++) {
		#pragma unroll
		for (iy = 0; iy < 4; iy++) {
			Ts1[nTx*(iy*4+ix)+iTx] = Tr1[iy*4+ix];
		}
	}
	__syncthreads ();

	#pragma unroll
	for (i = nTx/2; i > 0; i >>= 1) {
		if (iTx < i) {
			#pragma unroll
			for (ix = 0; ix < 4; ix++) {
				#pragma unroll
				for (iy = 0; iy < 4; iy++) {
					Ts1[nTx*(iy*4+ix)+iTx] += Ts1[nTx*(iy*4+ix)+iTx+i];
				}
			}
		}
		__syncthreads ();
	}

	if (iTx == 0) {
		#pragma unroll
		for (ix = 0; ix < 4; ix++) {
			#pragma unroll
			for (iy = 0; iy < 4; iy++) {
				w[ldw*(iy*4+ix)+iBx] = Ts1[nTx*(iy*4+ix)];
			}
		}
	}
}

__global__ void
myblas_dot_x_kernel_4 (
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
myblas_dot_x_kernel_4 (
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
	const size_t nTx = MYGEMM_NTX4;//blockDim.x;
	const size_t addrx = iBx * nTx + iTx;
	size_t i, ix, iy;
	register double Xr1[4], Yr1[4];
	register double2 Tr2[4*4];
	__shared__ double2 Ts2[4*4*MYGEMM_NTX4];

	#pragma unroll
	for (ix = 0; ix < 4; ix++) {
		#pragma unroll
		for (iy = 0; iy < 4; iy++) {
			Tr2[iy*4+ix] = make_double2 (0., 0.);
		}
	}

	// DOT part -----------------------------------------------
	for (i = addrx; i < n; i += nTx * nBx) {
		#pragma unroll
		for (ix = 0; ix < 4; ix++) 
			Xr1[ix] = x[ix * ldx + i];
		#pragma unroll
		for (iy = 0; iy < 4; iy++) 
			Yr1[iy] = y[iy * ldy + i];
		#pragma unroll
		for (ix = 0; ix < 4; ix++) {
			#pragma unroll
			for (iy = 0; iy < 4; iy++) {
				cuDot2i (Xr1[ix], Yr1[iy], Tr2[iy*4+ix]);
			}
		}
	}
	#pragma unroll
	for (ix = 0; ix < 4; ix++) {
		#pragma unroll
		for (iy = 0; iy < 4; iy++) {
			Ts2[nTx*(iy*4+ix)+iTx] = Tr2[iy*4+ix];
		}
	}
	__syncthreads ();

	// SUM part -----------------------------------------------
	#pragma unroll
	for (i = nTx/2; i > 0; i >>= 1) {
		if (iTx < i) {
			#pragma unroll
			for (ix = 0; ix < 4; ix++) {
				#pragma unroll
				for (iy = 0; iy < 4; iy++) {
					Ts2[nTx*(iy*4+ix)+iTx] = cuQuadAdd (Ts2[nTx*(iy*4+ix)+iTx], Ts2[nTx*(iy*4+ix)+iTx+i]);
				}
			}
		}
		__syncthreads ();
	}

	if (iTx == 0) {
		#pragma unroll
		for (ix = 0; ix < 4; ix++) {
			#pragma unroll
			for (iy = 0; iy < 4; iy++) {
				w[ldw*(iy*4+ix)+iBx] = Ts2[nTx*(iy*4+ix)];
			}
		}
	}
}



