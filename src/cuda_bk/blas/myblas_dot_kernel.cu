#include "../cueft.h"
#define _NBK 6

template <typename TYPE>
__global__ void
myblas_sum_kernel (
	const size_t sx,
	const size_t sy,
	const size_t n,
	TYPE *w,
	const size_t ldw,
	TYPE *result
) {
	const size_t iTx = threadIdx.x;
	const size_t iBx = blockIdx.x;
	const size_t nBx = gridDim.x;
	const size_t nTx = MYGEMM_NTX6;//blockDim.x;
	const size_t addrx = iBx * nTx + iTx;
	size_t i, ix, iy;
	__shared__ TYPE Ts1[_NBK*_NBK*MYGEMM_NTX6];
	TYPE Tr1[_NBK*_NBK] = {0.};

	for (i = addrx; i < n; i += nTx * nBx) 
		for (ix = 0; ix < sx; ix++) 
			for (iy = 0; iy < sy; iy++) 
				Tr1[iy*_NBK+ix] += w[ldw*(iy*sx+ix)+i];
	for (ix = 0; ix < sx; ix++) 
		for (iy = 0; iy < sy; iy++) 
			Ts1[nTx*(iy*sx+ix)+iTx] = Tr1[iy*_NBK+ix];
	__syncthreads ();

	#pragma unroll
	for (i = nTx/2; i > 0; i >>= 1) {
		if (iTx < i)
			for (ix = 0; ix < sx; ix++) 
				for (iy = 0; iy < sy; iy++) 
					Ts1[nTx*(iy*sx+ix)+iTx] += Ts1[nTx*(iy*sx+ix)+iTx+i];
		__syncthreads ();
	}
	if (iTx == 0 && iBx == 0) {
		for (ix = 0; ix < sx; ix++) 
			for (iy = 0; iy < sy; iy++) 
				result[iy*sx+ix] = Ts1[nTx*(iy*sx+ix)];
	}
}

template <typename TYPE>
__global__ void
myblas_dot_kernel (
	const size_t sx,
	const size_t sy,
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
	const size_t nTx = MYGEMM_NTX6;//blockDim.x;
	const size_t addrx = iBx * nTx + iTx;
	size_t i, ix, iy;
	__shared__ TYPE Ts1[_NBK*_NBK*MYGEMM_NTX6];
	TYPE Xr1[_NBK], Yr1[_NBK];
	TYPE Tr1[_NBK*_NBK] = {0.};

	for (i = addrx; i < n; i += nTx * nBx) {
		for (ix = 0; ix < sx; ix++) 
			Xr1[ix] = x[ix * ldx + i];
		for (iy = 0; iy < sy; iy++) 
			Yr1[iy] = y[iy * ldy + i];
		for (ix = 0; ix < sx; ix++) 
			for (iy = 0; iy < sy; iy++) 
				Tr1[iy*_NBK+ix] += Xr1[ix] * Yr1[iy];
	}
	for (ix = 0; ix < sx; ix++) 
		for (iy = 0; iy < sy; iy++) 
			Ts1[nTx*(iy*sx+ix)+iTx] = Tr1[iy*_NBK+ix];
	__syncthreads ();

	#pragma unroll
	for (i = nTx/2; i > 0; i >>= 1) {
		if (iTx < i)
			for (ix = 0; ix < sx; ix++) 
				for (iy = 0; iy < sy; iy++) 
					Ts1[nTx*(iy*sx+ix)+iTx] += Ts1[nTx*(iy*sx+ix)+iTx+i];
		__syncthreads ();
	}

	if (iTx == 0) 
		for (ix = 0; ix < sx; ix++) 
			for (iy = 0; iy < sy; iy++) 
				w[ldw*(iy*sx+ix)+iBx] = Ts1[nTx*(iy*sx+ix)];

}

// Dot2 kernel

__global__ void
myblas_sum_x_kernel (
	const size_t sx,
	const size_t sy,
	const size_t n,
	const double2 * __restrict__ w,
	const size_t ldw,
	float *result
) {
}

__global__ void
myblas_sum_x2_kernel (
	const size_t sx,
	const size_t sy,
	const size_t n,
	const double2 * __restrict__ w,
	const size_t ldw,
	float *result
) {
}

__global__ void
myblas_sum_x2_kernel (
	const size_t sx,
	const size_t sy,
	const size_t n,
	const double2 * __restrict__ w,
	const size_t ldw,
	double *result
) {
	const size_t iTx = threadIdx.x;
	const size_t iBx = blockIdx.x;
	const size_t nBx = gridDim.x;
	const size_t nTx = MYGEMM_NTX6;//blockDim.x;
	const size_t addrx = iBx * nTx + iTx;
	size_t i, ix, iy;
	__shared__ double2 Ts2[_NBK*_NBK*MYGEMM_NTX6];
	double2 Tr2[_NBK*_NBK];

	#pragma unroll
	for (ix = 0; ix < _NBK; ix++) {
		#pragma unroll
		for (iy = 0; iy < _NBK; iy++) {
			Tr2[iy*_NBK+ix] = make_double2 (0., 0.);
		}
	}

	for (i = addrx; i < n; i += nTx * nBx) {
		for (ix = 0; ix < sx; ix++) {
			for (iy = 0; iy < sy; iy++) {
				Tr2[iy*_NBK+ix] = cuQuadAdd (Tr2[iy*_NBK+ix], w[ldw*(iy*sx+ix)+i]);
			}
		}
	}
	for (ix = 0; ix < sx; ix++) {
		for (iy = 0; iy < sy; iy++) {
			Ts2[nTx*(iy*sx+ix)+iTx] = Tr2[iy*_NBK+ix];
		}
	}
	__syncthreads ();

	#pragma unroll
	for (i = nTx/2; i > 0; i >>= 1) {
		if (iTx < i) {
			for (ix = 0; ix < sx; ix++) {
				for (iy = 0; iy < sy; iy++) {
					Ts2[nTx*(iy*sx+ix)+iTx] = cuQuadAdd (Ts2[nTx*(iy*sx+ix)+iTx], Ts2[nTx*(iy*sx+ix)+iTx+i]);
				}
			}
		}
		__syncthreads ();
	}
	if (iTx == 0 && iBx == 0) {
		for (ix = 0; ix < sx; ix++) {
			for (iy = 0; iy < sy; iy++) {
				result[iy*sx+ix] = Ts2[nTx*(iy*sx+ix)].x;
				result[iy*sx+ix+sx*sy] = Ts2[nTx*(iy*sx+ix)].y;
			}
		}
	}
}

__global__ void
myblas_sum_x_kernel (
	const size_t sx,
	const size_t sy,
	const size_t n,
	const double2 * __restrict__ w,
	const size_t ldw,
	double *result
) {
	const size_t iTx = threadIdx.x;
	const size_t iBx = blockIdx.x;
	const size_t nBx = gridDim.x;
	const size_t nTx = MYGEMM_NTX6;//blockDim.x;
	const size_t addrx = iBx * nTx + iTx;
	size_t i, ix, iy;
	__shared__ double2 Ts2[_NBK*_NBK*MYGEMM_NTX6];
	double2 Tr2[_NBK*_NBK];

	#pragma unroll
	for (ix = 0; ix < _NBK; ix++) {
		#pragma unroll
		for (iy = 0; iy < _NBK; iy++) {
			Tr2[iy*_NBK+ix] = make_double2 (0., 0.);
		}
	}

	for (i = addrx; i < n; i += nTx * nBx) {
		for (ix = 0; ix < sx; ix++) {
			for (iy = 0; iy < sy; iy++) {
				Tr2[iy*_NBK+ix] = cuQuadAdd (Tr2[iy*_NBK+ix], w[ldw*(iy*sx+ix)+i]);
			}
		}
	}
	for (ix = 0; ix < sx; ix++) {
		for (iy = 0; iy < sy; iy++) {
			Ts2[nTx*(iy*sx+ix)+iTx] = Tr2[iy*_NBK+ix];
		}
	}
	__syncthreads ();

	#pragma unroll
	for (i = nTx/2; i > 0; i >>= 1) {
		if (iTx < i) {
			for (ix = 0; ix < sx; ix++) {
				for (iy = 0; iy < sy; iy++) {
					Ts2[nTx*(iy*sx+ix)+iTx] = cuQuadAdd (Ts2[nTx*(iy*sx+ix)+iTx], Ts2[nTx*(iy*sx+ix)+iTx+i]);
				}
			}
		}
		__syncthreads ();
	}
	if (iTx == 0 && iBx == 0) {
		for (ix = 0; ix < sx; ix++) {
			for (iy = 0; iy < sy; iy++) {
				result[iy*sx+ix] = Ts2[nTx*(iy*sx+ix)].x + Ts2[nTx*(iy*sx+ix)].y;
			}
		}
	}
}

__global__ void
myblas_dot_x_kernel (
	const size_t sx,
	const size_t sy,
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
myblas_dot_x_kernel (
	const size_t sx,
	const size_t sy,
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
	const size_t nTx = MYGEMM_NTX6;//blockDim.x;
	const size_t addrx = iBx * nTx + iTx;
	size_t i, ix, iy;
	register double Xr1[_NBK], Yr1[_NBK];
	register double2 Tr2[_NBK*_NBK];
	__shared__ double2 Ts2[_NBK*_NBK*MYGEMM_NTX6];

	#pragma unroll
	for (ix = 0; ix < _NBK; ix++) {
		#pragma unroll
		for (iy = 0; iy < _NBK; iy++) {
			Tr2[iy*_NBK+ix] = make_double2 (0., 0.);
		}
	}

	// DOT part -----------------------------------------------
	for (i = addrx; i < n; i += nTx * nBx) {
		for (ix = 0; ix < sx; ix++) 
			Xr1[ix] = x[ix * ldx + i];
		for (iy = 0; iy < sy; iy++) 
			Yr1[iy] = y[iy * ldy + i];
		for (ix = 0; ix < sx; ix++) {
			for (iy = 0; iy < sy; iy++) {
				cuDot2i (Xr1[ix], Yr1[iy], Tr2[iy*_NBK+ix]);
			}
		}
	}
	for (ix = 0; ix < sx; ix++) {
		for (iy = 0; iy < sy; iy++) {
			Ts2[nTx*(iy*sx+ix)+iTx] = Tr2[iy*_NBK+ix];
		}
	}
	__syncthreads ();

	// SUM part -----------------------------------------------
	#pragma unroll
	for (i = nTx/2; i > 0; i >>= 1) {
		if (iTx < i) {
			for (ix = 0; ix < sx; ix++) {
				for (iy = 0; iy < sy; iy++) {
					Ts2[nTx*(iy*sx+ix)+iTx] = cuQuadAdd (Ts2[nTx*(iy*sx+ix)+iTx], Ts2[nTx*(iy*sx+ix)+iTx+i]);
				}
			}
		}
		__syncthreads ();
	}

	if (iTx == 0) {
		for (ix = 0; ix < sx; ix++) {
			for (iy = 0; iy < sy; iy++) {
				w[ldw*(iy*sx+ix)+iBx] = Ts2[nTx*(iy*sx+ix)];
			}
		}
	}
}



