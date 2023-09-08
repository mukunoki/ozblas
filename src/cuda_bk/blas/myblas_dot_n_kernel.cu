#include "../cueft.h"

// ================================
// SUM
// ================================

__global__ void
myblas_sum_x2_kernel__NBK_ (
	const size_t n,
	const double2 * __restrict__ w,
	const size_t ldw,
	float *result
) {
}

__global__ void
myblas_sum_x2_kernel__NBK_ (
	const size_t n,
	const double2 * __restrict__ w,
	const size_t ldw,
	double *result
) {
	const size_t iTx = threadIdx.x;
	const size_t iBx = blockIdx.x;
	const size_t nBx = gridDim.x;
	const size_t nTx = DOTGEMM_NTX;//blockDim.x;
	const size_t addrx = iBx * nTx + iTx;
	size_t i, ix, iy;
	__shared__ double2 Ts2[_NBK_*_NBK_*DOTGEMM_NTX];
	double2 Tr2[_NBK_*_NBK_];

	#pragma unroll
	for (ix = 0; ix < _NBK_; ix++) {
		#pragma unroll
		for (iy = 0; iy < _NBK_; iy++) {
			Tr2[iy*_NBK_+ix] = make_double2 (0., 0.);
		}
	}

	for (i = addrx; i < n; i += nTx * nBx) {
		#pragma unroll
		for (ix = 0; ix < _NBK_; ix++) {
			#pragma unroll
			for (iy = 0; iy < _NBK_; iy++) {
				Tr2[iy*_NBK_+ix] = cuQuadAdd (Tr2[iy*_NBK_+ix], w[ldw*(iy*_NBK_+ix)+i]);
			}
		}
	}
	#pragma unroll
	for (ix = 0; ix < _NBK_; ix++) {
		#pragma unroll
		for (iy = 0; iy < _NBK_; iy++) {
			Ts2[nTx*(iy*_NBK_+ix)+iTx] = Tr2[iy*_NBK_+ix];
		}
	}
	__syncthreads ();

	#pragma unroll
	for (i = nTx/2; i > 0; i >>= 1) {
		if (iTx < i) {
			#pragma unroll
			for (ix = 0; ix < _NBK_; ix++) {
				#pragma unroll
				for (iy = 0; iy < _NBK_; iy++) {
					Ts2[nTx*(iy*_NBK_+ix)+iTx] = cuQuadAdd (Ts2[nTx*(iy*_NBK_+ix)+iTx], Ts2[nTx*(iy*_NBK_+ix)+iTx+i]);
				}
			}
		}
		__syncthreads ();
	}
	if (iTx == 0 && iBx == 0) {
		#pragma unroll
		for (ix = 0; ix < _NBK_; ix++) {
			#pragma unroll
			for (iy = 0; iy < _NBK_; iy++) {
				result[iy*_NBK_+ix] = Ts2[nTx*(iy*_NBK_+ix)].x;
				result[iy*_NBK_+ix+_NBK_*_NBK_] = Ts2[nTx*(iy*_NBK_+ix)].y;
			}
		}
	}
}

__global__ void
myblas_sum_x_kernel__NBK_ (
	const size_t n,
	const double2 * __restrict__ w,
	const size_t ldw,
	float *result
) {
}

__global__ void
myblas_sum_x_kernel__NBK_ (
	const size_t n,
	const double2 * __restrict__ w,
	const size_t ldw,
	double *result
) {
	const size_t iTx = threadIdx.x;
	const size_t iBx = blockIdx.x;
	const size_t nBx = gridDim.x;
	const size_t nTx = DOTGEMM_NTX;//blockDim.x;
	const size_t addrx = iBx * nTx + iTx;
	size_t i, ix, iy;
	__shared__ double2 Ts2[_NBK_*_NBK_*DOTGEMM_NTX];
	double2 Tr2[_NBK_*_NBK_];

	#pragma unroll
	for (ix = 0; ix < _NBK_; ix++) {
		#pragma unroll
		for (iy = 0; iy < _NBK_; iy++) {
			Tr2[iy*_NBK_+ix] = make_double2 (0., 0.);
		}
	}

	for (i = addrx; i < n; i += nTx * nBx) {
		#pragma unroll
		for (ix = 0; ix < _NBK_; ix++) {
			#pragma unroll
			for (iy = 0; iy < _NBK_; iy++) {
				Tr2[iy*_NBK_+ix] = cuQuadAdd (Tr2[iy*_NBK_+ix], w[ldw*(iy*_NBK_+ix)+i]);
			}
		}
	}
	#pragma unroll
	for (ix = 0; ix < _NBK_; ix++) {
		#pragma unroll
		for (iy = 0; iy < _NBK_; iy++) {
			Ts2[nTx*(iy*_NBK_+ix)+iTx] = Tr2[iy*_NBK_+ix];
		}
	}
	__syncthreads ();

	#pragma unroll
	for (i = nTx/2; i > 0; i >>= 1) {
		if (iTx < i) {
			#pragma unroll
			for (ix = 0; ix < _NBK_; ix++) {
				#pragma unroll
				for (iy = 0; iy < _NBK_; iy++) {
					Ts2[nTx*(iy*_NBK_+ix)+iTx] = cuQuadAdd (Ts2[nTx*(iy*_NBK_+ix)+iTx], Ts2[nTx*(iy*_NBK_+ix)+iTx+i]);
				}
			}
		}
		__syncthreads ();
	}
	if (iTx == 0) {
		#pragma unroll
		for (ix = 0; ix < _NBK_; ix++) {
			#pragma unroll
			for (iy = 0; iy < _NBK_; iy++) {
				result[nBx*(iy*_NBK_+ix)+iBx] = Ts2[nTx*(iy*_NBK_+ix)].x + Ts2[nTx*(iy*_NBK_+ix)].y;
			}
		}
	}
}

template <typename TYPE>
__global__ void
myblas_sum_kernel__NBK_ (
	const size_t n,
	const TYPE *w,
	const size_t ldw,
	TYPE *result
) {
	const size_t iTx = threadIdx.x;
	const size_t iBx = blockIdx.x;
	const size_t nBx = gridDim.x;
	const size_t nTx = DOTGEMM_NTX;//blockDim.x;
	const size_t addrx = iBx * nTx + iTx;
	size_t i, ix, iy;
	__shared__ TYPE Ts1[_NBK_*_NBK_*DOTGEMM_NTX];
	TYPE Tr1[_NBK_*_NBK_] = {0.};

	for (i = addrx; i < n; i += nTx * nBx) {
		#pragma unroll
		for (ix = 0; ix < _NBK_; ix++) {
			#pragma unroll
			for (iy = 0; iy < _NBK_; iy++) {
				Tr1[iy*_NBK_+ix] += w[ldw*(iy*_NBK_+ix)+i];
			}
		}
	}
	#pragma unroll
	for (ix = 0; ix < _NBK_; ix++) {
		#pragma unroll
		for (iy = 0; iy < _NBK_; iy++) {
			Ts1[nTx*(iy*_NBK_+ix)+iTx] = Tr1[iy*_NBK_+ix];
		}
	}
	__syncthreads ();

	#pragma unroll
	for (i = nTx/2; i > 0; i >>= 1) {
		if (iTx < i) {
			#pragma unroll
			for (ix = 0; ix < _NBK_; ix++) {
				#pragma unroll
				for (iy = 0; iy < _NBK_; iy++) {
					Ts1[nTx*(iy*_NBK_+ix)+iTx] += Ts1[nTx*(iy*_NBK_+ix)+iTx+i];
				}
			}
		}
		__syncthreads ();
	}
	if (iTx == 0 && iBx == 0) {
		#pragma unroll
		for (ix = 0; ix < _NBK_; ix++) {
			#pragma unroll
			for (iy = 0; iy < _NBK_; iy++) {
				result[iy*_NBK_+ix] = Ts1[nTx*(iy*_NBK_+ix)];
			}
		}
	}
}

// ================================
// DOT
// ================================

template <typename TYPE>
__global__ void
myblas_dot_kernel__NBK_ (
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
	const size_t nTx = DOTGEMM_NTX;//blockDim.x;
	const size_t addrx = iBx * nTx + iTx;
	size_t i, ix, iy;
	__shared__ TYPE Ts1[_NBK_*_NBK_*DOTGEMM_NTX];
	TYPE Xr1[_NBK_], Yr1[_NBK_];
	TYPE Tr1[_NBK_*_NBK_] = {0.};

	for (i = addrx; i < n; i += nTx * nBx) {
		#pragma unroll
		for (ix = 0; ix < _NBK_; ix++) {
			Xr1[ix] = x[ix * ldx + i];
		}
		#pragma unroll
		for (iy = 0; iy < _NBK_; iy++) {
			Yr1[iy] = y[iy * ldy + i];
		}
		#pragma unroll
		for (ix = 0; ix < _NBK_; ix++) {
			#pragma unroll
			for (iy = 0; iy < _NBK_; iy++) {
				Tr1[iy*_NBK_+ix] += Xr1[ix] * Yr1[iy];
			}
		}
	}
	#pragma unroll
	for (ix = 0; ix < _NBK_; ix++) {
		#pragma unroll
		for (iy = 0; iy < _NBK_; iy++) {
			Ts1[nTx*(iy*_NBK_+ix)+iTx] = Tr1[iy*_NBK_+ix];
		}
	}
	__syncthreads ();

	#pragma unroll
	for (i = nTx/2; i > 0; i >>= 1) {
		if (iTx < i) {
			#pragma unroll
			for (ix = 0; ix < _NBK_; ix++) {
				#pragma unroll
				for (iy = 0; iy < _NBK_; iy++) {
					Ts1[nTx*(iy*_NBK_+ix)+iTx] += Ts1[nTx*(iy*_NBK_+ix)+iTx+i];
				}
			}
		}
		__syncthreads ();
	}

	if (iTx == 0) {
		#pragma unroll
		for (ix = 0; ix < _NBK_; ix++) {
			#pragma unroll
			for (iy = 0; iy < _NBK_; iy++) {
				w[ldw*(iy*_NBK_+ix)+iBx] = Ts1[nTx*(iy*_NBK_+ix)];
			}
		}
	}
}

__global__ void
myblas_dot_x_kernel__NBK_ (
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
myblas_dot_x_kernel__NBK_ (
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
	const size_t nTx = DOTGEMM_NTX;//blockDim.x;
	const size_t addrx = iBx * nTx + iTx;
	size_t i, ix, iy;
	register double Xr1[_NBK_], Yr1[_NBK_];
	register double2 Tr2[_NBK_*_NBK_];
	__shared__ double2 Ts2[_NBK_*_NBK_*DOTGEMM_NTX];

	#pragma unroll
	for (ix = 0; ix < _NBK_; ix++) {
		#pragma unroll
		for (iy = 0; iy < _NBK_; iy++) {
			Tr2[iy*_NBK_+ix] = make_double2 (0., 0.);
		}
	}

	// DOT part -----------------------------------------------
	for (i = addrx; i < n; i += nTx * nBx) {
		#pragma unroll
		for (ix = 0; ix < _NBK_; ix++) 
			Xr1[ix] = x[ix * ldx + i];
		#pragma unroll
		for (iy = 0; iy < _NBK_; iy++) 
			Yr1[iy] = y[iy * ldy + i];
		#pragma unroll
		for (ix = 0; ix < _NBK_; ix++) {
			#pragma unroll
			for (iy = 0; iy < _NBK_; iy++) {
				cuDot2i (Xr1[ix], Yr1[iy], Tr2[iy*_NBK_+ix]);
			}
		}
	}
	#pragma unroll
	for (ix = 0; ix < _NBK_; ix++) {
		#pragma unroll
		for (iy = 0; iy < _NBK_; iy++) {
			Ts2[nTx*(iy*_NBK_+ix)+iTx] = Tr2[iy*_NBK_+ix];
		}
	}
	__syncthreads ();

	// SUM part -----------------------------------------------
	#pragma unroll
	for (i = nTx/2; i > 0; i >>= 1) {
		if (iTx < i) {
			#pragma unroll
			for (ix = 0; ix < _NBK_; ix++) {
				#pragma unroll
				for (iy = 0; iy < _NBK_; iy++) {
					Ts2[nTx*(iy*_NBK_+ix)+iTx] = cuQuadAdd (Ts2[nTx*(iy*_NBK_+ix)+iTx], Ts2[nTx*(iy*_NBK_+ix)+iTx+i]);
				}
			}
		}
		__syncthreads ();
	}

	if (iTx == 0) {
		#pragma unroll
		for (ix = 0; ix < _NBK_; ix++) {
			#pragma unroll
			for (iy = 0; iy < _NBK_; iy++) {
				w[ldw*(iy*_NBK_+ix)+iBx] = Ts2[nTx*(iy*_NBK_+ix)];
			}
		}
	}
}



