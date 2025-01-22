#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#include <stdio.h>
#define OUTPUT stdout // stderr
#include "../cueft.h"

#define NTX_SGEMM 16//16
#define NTY_SGEMM 16//16
#define REGBKA_SGEMM 8//8
#define REGBKB_SGEMM 16//16
#define KBK_SGEMM 16//16

__global__ void cumyblas_igemm_nn_kernel (
	const size_t m,
	const size_t n,
	const size_t k,
	const int32_t alpha,
	const int32_t *Ag,
	const size_t lda,
	const int32_t *Bg,
	const size_t ldb,
	const int32_t beta,
	int32_t *Cg,
	const size_t ldc
) {
	int32_t jk, ki, ia, ib;
	const int32_t iBx = blockIdx.x;
	const int32_t iBy = blockIdx.y;
	const int32_t nTx = blockDim.x;
	const int32_t nTy = blockDim.y;
	const int32_t iTx = threadIdx.x;
	const int32_t iTy = threadIdx.y;
	const int32_t addrx = (iBx*nTx+iTx)*REGBKA_SGEMM;
	const int32_t addry = (iBy*nTy+iTy)*REGBKB_SGEMM;
	__shared__ int32_t As1[NTY_SGEMM][NTX_SGEMM*REGBKA_SGEMM+1];
	__shared__ int32_t Bs1[NTX_SGEMM*REGBKB_SGEMM][NTY_SGEMM+1];
	register int32_t Ar1[REGBKA_SGEMM], Br1[REGBKB_SGEMM], Cr1, Tr1[REGBKB_SGEMM][REGBKA_SGEMM];

	#pragma unroll 
    for (ib = 0; ib < REGBKB_SGEMM; ib++) {
		#pragma unroll 
        for (ia = 0; ia < REGBKA_SGEMM; ia++) {
    	    Tr1[ib][ia] = 0.;
        }
    }

	for (jk = 0; jk < k; jk += KBK_SGEMM) {
		#pragma unroll 
        for (ia = 0; ia < REGBKA_SGEMM; ia++)
    		As1[iTy][iTx+ia*nTx] = Ag[(jk+iTy)*lda + (addrx+ia)];
		#pragma unroll 
        for (ib = 0; ib < REGBKB_SGEMM; ib++)
    	    Bs1[iTx+ib*nTx][iTy] = Bg[(addry+ib)*ldb + (iTx+jk)];
		__syncthreads ();
		#pragma unroll 
		for (ki = 0; ki < KBK_SGEMM; ki++) {
		    #pragma unroll 
            for (ia = 0; ia < REGBKA_SGEMM; ia++)
		    	Ar1[ia] = As1[ki][iTx+ia*nTx];
		    #pragma unroll 
            for (ib = 0; ib < REGBKB_SGEMM; ib++)
			    Br1[ib] = Bs1[ki+ib*nTx][iTy];
		    #pragma unroll 
            for (ib = 0; ib < REGBKB_SGEMM; ib++)
		        #pragma unroll 
                for (ia = 0; ia < REGBKA_SGEMM; ia++)
			        Tr1[ib][ia] = Ar1[ia] * Br1[ib] + Tr1[ib][ia]; 
		}
		__syncthreads ();
	}

	if (addrx < m && addry < n) {
		#pragma unroll 
        for (ib = 0; ib < REGBKB_SGEMM; ib++) {
		    #pragma unroll 
            for (ia = 0; ia < REGBKA_SGEMM; ia++) {
	            Cr1 = Cg[(addry+ib)*ldc + addrx+ia];
		        Cr1 = (alpha * Tr1[ib][ia] + (beta * Cr1));
    	        Cg[(addry+ib)*ldc + addrx+ia] = Cr1;
            }
        }
	}
    
}


__global__ void cumyblas_sgemm_nn_kernel (
	const size_t m,
	const size_t n,
	const size_t k,
	const float alpha,
	const float *Ag,
	const size_t lda,
	const float *Bg,
	const size_t ldb,
	const float beta,
	float *Cg,
	const size_t ldc
) {
	int32_t jk, ki, ia, ib;
	const int32_t iBx = blockIdx.x;
	const int32_t iBy = blockIdx.y;
	const int32_t nTx = blockDim.x;
	const int32_t nTy = blockDim.y;
	const int32_t iTx = threadIdx.x;
	const int32_t iTy = threadIdx.y;
	const int32_t addrx = (iBx*nTx+iTx)*REGBKA_SGEMM;
	const int32_t addry = (iBy*nTy+iTy)*REGBKB_SGEMM;
	__shared__ float As1[NTY_SGEMM][NTX_SGEMM*REGBKA_SGEMM+1];
	__shared__ float Bs1[NTX_SGEMM*REGBKB_SGEMM][NTY_SGEMM+1];
	register float Ar1[REGBKA_SGEMM], Br1[REGBKB_SGEMM], Cr1, Tr1[REGBKB_SGEMM][REGBKA_SGEMM];

	#pragma unroll 
    for (ib = 0; ib < REGBKB_SGEMM; ib++) {
		#pragma unroll 
        for (ia = 0; ia < REGBKA_SGEMM; ia++) {
    	    Tr1[ib][ia] = 0.;
        }
    }

	for (jk = 0; jk < k; jk += KBK_SGEMM) {
		#pragma unroll 
        for (ia = 0; ia < REGBKA_SGEMM; ia++)
    		As1[iTy][iTx+ia*nTx] = Ag[(jk+iTy)*lda + (addrx+ia)];
		#pragma unroll 
        for (ib = 0; ib < REGBKB_SGEMM; ib++)
    	    Bs1[iTx+ib*nTx][iTy] = Bg[(addry+ib)*ldb + (iTx+jk)];
		__syncthreads ();
		#pragma unroll 
		for (ki = 0; ki < KBK_SGEMM; ki++) {
		    #pragma unroll 
            for (ia = 0; ia < REGBKA_SGEMM; ia++)
		    	Ar1[ia] = As1[ki][iTx+ia*nTx];
		    #pragma unroll 
            for (ib = 0; ib < REGBKB_SGEMM; ib++)
			    Br1[ib] = Bs1[ki+ib*nTx][iTy];
		    #pragma unroll 
            for (ib = 0; ib < REGBKB_SGEMM; ib++)
		        #pragma unroll 
                for (ia = 0; ia < REGBKA_SGEMM; ia++)
			        Tr1[ib][ia] = Ar1[ia] * Br1[ib] + Tr1[ib][ia]; 
		}
		__syncthreads ();
	}

	if (addrx < m && addry < n) {
		#pragma unroll 
        for (ib = 0; ib < REGBKB_SGEMM; ib++) {
		    #pragma unroll 
            for (ia = 0; ia < REGBKA_SGEMM; ia++) {
	            Cr1 = Cg[(addry+ib)*ldc + addrx+ia];
		        Cr1 = (alpha * Tr1[ib][ia] + (beta * Cr1));
    	        Cg[(addry+ib)*ldc + addrx+ia] = Cr1;
            }
        }
	}
    
}

#define NTX_DGEMM 16//16
#define NTY_DGEMM 16//16
#define KBK_DGEMM 16//16
#define REGBKA_DGEMM 2//2
#define REGBKB_DGEMM 4//4

__global__ void cumyblas_dgemm_nn_kernel (
	const size_t m,
	const size_t n,
	const size_t k,
	const double alpha,
	const double *Ag,
	const size_t lda,
	const double *Bg,
	const size_t ldb,
	const double beta,
	double *Cg,
	const size_t ldc
) {
	int32_t jk, ki, ia, ib;
	const int32_t iBx = blockIdx.x;
	const int32_t iBy = blockIdx.y;
	const int32_t nTx = blockDim.x;
	const int32_t nTy = blockDim.y;
	const int32_t iTx = threadIdx.x;
	const int32_t iTy = threadIdx.y;
	const int32_t addrx = (iBx*nTx+iTx)*REGBKA_DGEMM;
	const int32_t addry = (iBy*nTy+iTy)*REGBKB_DGEMM;
	__shared__ double As1[NTY_DGEMM][NTX_DGEMM*REGBKA_DGEMM+1];
	__shared__ double Bs1[NTX_DGEMM*REGBKB_DGEMM][NTY_DGEMM+1];
	register double Ar1[REGBKA_DGEMM], Br1[REGBKB_DGEMM], Cr1, Tr1[REGBKB_DGEMM][REGBKA_DGEMM];

	#pragma unroll 
    for (ib = 0; ib < REGBKB_DGEMM; ib++) {
		#pragma unroll 
        for (ia = 0; ia < REGBKA_DGEMM; ia++) {
    	    Tr1[ib][ia] = 0.;
        }
    }

	for (jk = 0; jk < k; jk += KBK_DGEMM) {
		#pragma unroll 
        for (ia = 0; ia < REGBKA_DGEMM; ia++) {
    		//As1[iTy][iTx*REGBKA_DGEMM+ia] = double_to_float3 (Ag[(jk+iTy)*lda + (addrx+ia)]);
    		As1[iTy][iTx+ia*nTx] = Ag[(jk+iTy)*lda + (addrx+ia)];
        }
		#pragma unroll 
        for (ib = 0; ib < REGBKB_DGEMM; ib++) {
    	    //Bs1[iTx*REGBKB_DGEMM+ib][iTy] = double_to_float3 (Bg[(addry+ib)*ldb + (iTx+jk)]);
    	    Bs1[iTx+ib*nTx][iTy] = Bg[(addry+ib)*ldb + (iTx+jk)];
        }
		__syncthreads ();
		#pragma unroll 
		for (ki = 0; ki < KBK_DGEMM; ki++) {
		    #pragma unroll 
            for (ia = 0; ia < REGBKA_DGEMM; ia++)
		    	//Ar1[ia] = As1[ki][iTx*REGBKA_DGEMM+ia];
		    	Ar1[ia] = As1[ki][iTx+ia*nTx];
		    #pragma unroll 
            for (ib = 0; ib < REGBKB_DGEMM; ib++)
			    //Br1[ib] = Bs1[ki*REGBKB_DGEMM+ib][iTy];
			    Br1[ib] = Bs1[ki+ib*nTx][iTy];
		    #pragma unroll 
            for (ib = 0; ib < REGBKB_DGEMM; ib++)
		        #pragma unroll 
                for (ia = 0; ia < REGBKA_DGEMM; ia++)
			        Tr1[ib][ia] = Ar1[ia] * Br1[ib] + Tr1[ib][ia];
		}
		__syncthreads ();
	}

	if (addrx < m && addry < n) {
		#pragma unroll 
        for (ib = 0; ib < REGBKB_DGEMM; ib++) {
		    #pragma unroll 
            for (ia = 0; ia < REGBKA_DGEMM; ia++) {
	            Cr1 = Cg[(addry+ib)*ldc + addrx+ia];
		        Cr1 = alpha * Tr1[ib][ia] + beta * Cr1;
    	        Cg[(addry+ib)*ldc + addrx+ia] = Cr1;
            }
        }
	}
}

#define NTX_QTW_DGEMM 16//16
#define NTY_QTW_DGEMM 16//16
#define KBK_QTW_DGEMM 16//16
#define REGBKA_QTW_DGEMM 2//2
#define REGBKB_QTW_DGEMM 4//4

__global__ void cumyblas_qtw_dgemm_nn_kernel (
	const size_t m,
	const size_t n,
	const size_t k,
	const double alpha,
	const double *Ag,
	const size_t lda,
	const double *Bg,
	const size_t ldb,
	const double beta,
	double *Cg,
	const size_t ldc
) {
	int32_t jk, ki, ia, ib;
	const int32_t iBx = blockIdx.x;
	const int32_t iBy = blockIdx.y;
	const int32_t nTx = blockDim.x;
	const int32_t nTy = blockDim.y;
	const int32_t iTx = threadIdx.x;
	const int32_t iTy = threadIdx.y;
	const int32_t addrx = (iBx*nTx+iTx)*REGBKA_QTW_DGEMM;
	const int32_t addry = (iBy*nTy+iTy)*REGBKB_QTW_DGEMM;
	__shared__ float3 As1[NTY_QTW_DGEMM][NTX_QTW_DGEMM*REGBKA_QTW_DGEMM+1];
	__shared__ float3 Bs1[NTX_QTW_DGEMM*REGBKB_QTW_DGEMM][NTY_QTW_DGEMM+1];
	register float3 Ar1[REGBKA_QTW_DGEMM], Br1[REGBKB_QTW_DGEMM], Cr1, Tr1[REGBKB_QTW_DGEMM][REGBKA_QTW_DGEMM];

	#pragma unroll 
    for (ib = 0; ib < REGBKB_QTW_DGEMM; ib++) {
		#pragma unroll 
        for (ia = 0; ia < REGBKA_QTW_DGEMM; ia++) {
    	    Tr1[ib][ia] = make_float3 (0., 0., 0.);
        }
    }

	for (jk = 0; jk < k; jk += KBK_QTW_DGEMM) {
		#pragma unroll 
        for (ia = 0; ia < REGBKA_QTW_DGEMM; ia++) {
    		//As1[iTy][iTx*REGBKA_QTW_DGEMM+ia] = double_to_float3 (Ag[(jk+iTy)*lda + (addrx+ia)]);
    		As1[iTy][iTx+ia*nTx] = double_to_float3 (Ag[(jk+iTy)*lda + (addrx+ia)]); // faster
        }
		#pragma unroll 
        for (ib = 0; ib < REGBKB_QTW_DGEMM; ib++) {
    	    //Bs1[iTx*REGBKB_QTW_DGEMM+ib][iTy] = double_to_float3 (Bg[(addry+ib)*ldb + (iTx+jk)]);
    	    Bs1[iTx+ib*nTx][iTy] = double_to_float3 (Bg[(addry+ib)*ldb + (iTx+jk)]); // faster
        }
		__syncthreads ();
		#pragma unroll 
		for (ki = 0; ki < KBK_QTW_DGEMM; ki++) {
		    #pragma unroll 
            for (ia = 0; ia < REGBKA_QTW_DGEMM; ia++)
		    	//Ar1[ia] = As1[ki][iTx*REGBKA_QTW_DGEMM+ia];
		    	Ar1[ia] = As1[ki][iTx+ia*nTx];
		    #pragma unroll 
            for (ib = 0; ib < REGBKB_QTW_DGEMM; ib++)
			    //Br1[ib] = Bs1[ki*REGBKB_QTW_DGEMM+ib][iTy];
			    Br1[ib] = Bs1[ki+ib*nTx][iTy];
		    #pragma unroll 
            for (ib = 0; ib < REGBKB_QTW_DGEMM; ib++)
		        #pragma unroll 
                for (ia = 0; ia < REGBKA_QTW_DGEMM; ia++)
			        Tr1[ib][ia] = cuQTWadd (cuQTWmul (Ar1[ia], Br1[ib]), Tr1[ib][ia]); 
		}
		__syncthreads ();
	}

	if (addrx < m && addry < n) {
		#pragma unroll 
        for (ib = 0; ib < REGBKB_QTW_DGEMM; ib++) {
		    #pragma unroll 
            for (ia = 0; ia < REGBKA_QTW_DGEMM; ia++) {
	            Cr1 = double_to_float3 (Cg[(addry+ib)*ldc + addrx+ia]);
		        Cr1 = cuQTWadd(cuQTWmul (double_to_float3 (alpha), Tr1[ib][ia]), cuQTWmul(double_to_float3 (beta), Cr1));
    	        Cg[(addry+ib)*ldc + addrx+ia] = float3_to_double (Cr1);
            }
        }
	}
    
}

#define NTX_QTWGEMM 16//16
#define NTY_QTWGEMM 16//16
#define KBK_QTWGEMM 16//16
#define REGBKA_QTWGEMM 2//2
#define REGBKB_QTWGEMM 4//4

__global__ void cumyblas_qtwgemm_nn_kernel (
	const size_t m,
	const size_t n,
	const size_t k,
	const double alpha,
	const float3 *Ag,
	const size_t lda,
	const float3 *Bg,
	const size_t ldb,
	const double beta,
	float3 *Cg,
	const size_t ldc
) {
	int32_t jk, ki, ia, ib;
	const int32_t iBx = blockIdx.x;
	const int32_t iBy = blockIdx.y;
	const int32_t nTx = blockDim.x;
	const int32_t nTy = blockDim.y;
	const int32_t iTx = threadIdx.x;
	const int32_t iTy = threadIdx.y;
	const int32_t addrx = (iBx*nTx+iTx)*REGBKA_QTWGEMM;
	const int32_t addry = (iBy*nTy+iTy)*REGBKB_QTWGEMM;
	__shared__ float3 As1[NTY_QTWGEMM][NTX_QTWGEMM*REGBKA_QTWGEMM+1];
	__shared__ float3 Bs1[NTX_QTWGEMM*REGBKB_QTWGEMM][NTY_QTWGEMM+1];
	register float3 Ar1[REGBKA_QTWGEMM], Br1[REGBKB_QTWGEMM], Cr1, Tr1[REGBKB_QTWGEMM][REGBKA_QTWGEMM];

	#pragma unroll 
    for (ib = 0; ib < REGBKB_QTWGEMM; ib++) {
		#pragma unroll 
        for (ia = 0; ia < REGBKA_QTWGEMM; ia++) {
    	    Tr1[ib][ia] = make_float3 (0., 0., 0.);
        }
    }

	for (jk = 0; jk < k; jk += KBK_QTWGEMM) {
		#pragma unroll 
        for (ia = 0; ia < REGBKA_QTWGEMM; ia++) {
    		As1[iTy][iTx*REGBKA_QTWGEMM+ia] = Ag[(jk+iTy)*lda + (addrx+ia)]; // faster
    		//As1[iTy][iTx+ia*nTx] = Ag[(jk+iTy)*lda + (addrx+ia)];
        }
		#pragma unroll 
        for (ib = 0; ib < REGBKB_QTWGEMM; ib++) {
    	    Bs1[iTx*REGBKB_QTWGEMM+ib][iTy] = Bg[(addry+ib)*ldb + (iTx+jk)]; // faster
    	    //Bs1[iTx+ib*nTx][iTy] = Bg[(addry+ib)*ldb + (iTx+jk)];
        }
		__syncthreads ();
		#pragma unroll 
		for (ki = 0; ki < KBK_QTWGEMM; ki++) {
		    #pragma unroll 
            for (ia = 0; ia < REGBKA_QTWGEMM; ia++)
		    	Ar1[ia] = As1[ki][iTx*REGBKA_QTWGEMM+ia];
		    	//Ar1[ia] = As1[ki][iTx+ia*nTx];
		    #pragma unroll 
            for (ib = 0; ib < REGBKB_QTWGEMM; ib++)
			    Br1[ib] = Bs1[ki*REGBKB_QTWGEMM+ib][iTy];
			    //Br1[ib] = Bs1[ki+ib*nTx][iTy];
		    #pragma unroll 
            for (ib = 0; ib < REGBKB_QTWGEMM; ib++)
		        #pragma unroll 
                for (ia = 0; ia < REGBKA_QTWGEMM; ia++)
			        Tr1[ib][ia] = cuQTWadd (cuQTWmul (Ar1[ia], Br1[ib]), Tr1[ib][ia]); 
		}
		__syncthreads ();
	}

	if (addrx < m && addry < n) {
		#pragma unroll 
        for (ib = 0; ib < REGBKB_QTWGEMM; ib++) {
		    #pragma unroll 
            for (ia = 0; ia < REGBKA_QTWGEMM; ia++) {
	            Cr1 = Cg[(addry+ib)*ldc + addrx+ia];
		        Cr1 = cuQTWadd(cuQTWmul (double_to_float3 (alpha), Tr1[ib][ia]), cuQTWmul(double_to_float3 (beta), Cr1));
    	        Cg[(addry+ib)*ldc + addrx+ia] = Cr1;
            }
        }
	}
    
}


/*
__global__ void cumyblas_qtw_gemm_nn_kernel (
	const size_t m,
	const size_t n,
	const size_t k,
	const double alpha,
	const double *Ag,
	const size_t lda,
	const double *Bg,
	const size_t ldb,
	const double beta,
	double *Cg,
	const size_t ldc
) {
	int32_t jk, ki;
	const int32_t iBx = blockIdx.x;
	const int32_t iBy = blockIdx.y;
	const int32_t nTx = blockDim.x;
	const int32_t nTy = blockDim.y;
	const int32_t iTx = threadIdx.x;
	const int32_t iTy = threadIdx.y;
	const int32_t addrx = iBx*nTx+iTx;
	const int32_t addry = iBy*nTy+iTy;
	__shared__ float3 As1[NTY_DGEMM][NTX_DGEMM+1];
	__shared__ float3 Bs1[NTX_DGEMM][NTY_DGEMM+1];
	register   float3 Ar1, Br1, Cr1, Tr1;

	double* addrCg = &Cg[addry*ldc + addrx];
	Tr1 = make_float3 (0., 0., 0.);

	for (jk = 0; jk < k-KBK; jk += KBK) {
		//As1[iTy][iTx]   = make_float3(Ag[(jk+iTy)*lda + MIN(m-1,addrx)],0,0);
		//Bs1[iTx][iTy]   = make_float3(Bg[MIN(n-1,addry)*ldb + (iTx+jk)],0,0);
		As1[iTy][iTx] = double_to_float3 (Ag[(jk+iTy)*lda + MIN(m-1,addrx)]);
		Bs1[iTx][iTy] = double_to_float3 (Bg[MIN(n-1,addry)*ldb + (iTx+jk)]);
		__syncthreads ();
		#pragma unroll 
		for (ki = 0; ki < KBK; ki++) {
			Ar1 = As1[ki][iTx];
			Br1 = Bs1[ki][iTy];
			//Tr1.x = Ar1.x * Br1.x + Tr1.x; 
			Tr1 = cuQTWadd (cuQTWmul (Ar1, Br1), Tr1); 
		}
		__syncthreads ();
	} {
		As1[iTy][iTx]   = double_to_float3 (Ag[MIN(k-1,jk+iTy)*lda + MIN(m-1,addrx)]);
		Bs1[iTx][iTy]   = double_to_float3 (Bg[MIN(n-1,addry)*ldb + MIN(k-1,iTx+jk)]);
		__syncthreads ();
		#pragma unroll 
		for (ki = 0; ki+jk < k; ki++) {
			Ar1 = As1[ki][iTx];
			Br1 = Bs1[ki][iTy];
			//Tr1.x = Ar1.x * Br1.x + Tr1.x; 
			Tr1 = cuQTWadd (cuQTWmul (Ar1, Br1), Tr1); 
		}
		__syncthreads ();
	}

	if (addrx < m && addry < n) {
	    Cr1 = double_to_float3 (addrCg[0]);
		Cr1 = cuQTWadd(cuQTWmul (double_to_float3 (alpha), Tr1), cuQTWmul(double_to_float3 (beta), Cr1));
		//addrCg[0] = Cr1.x + Cr1.y + Cr1.z;
		addrCg[0] = float3_to_double (Cr1);
	}
    
}
*/

#define NTX_CONV 16
#define NTY_CONV 16

__global__ void cumyblas_double_to_float3_kernel (
	const size_t m,
	const size_t n,
	const double * __restrict__ Ag,
	const size_t lda,
	float3 *Cg,
	const size_t ldc
) {
	size_t ix, iy;
	const size_t iBx = blockIdx.x;
	const size_t iBy = blockIdx.y;
	const size_t iTx = threadIdx.x;
	const size_t iTy = threadIdx.y;
	const size_t nTx = blockDim.x;
	const size_t nTy = blockDim.y;
	ix = iBx * nTx + iTx;
	iy = iBy * nTy + iTy;
	if (ix < m && iy < n) {
		Cg[iy * ldc + ix] = double_to_float3 (Ag[iy * lda + ix]);
	}
}

__global__ void cumyblas_float3_to_double_kernel (
	const size_t m,
	const size_t n,
	const float3 * __restrict__ Ag,
	const size_t lda,
	double *Cg,
	const size_t ldc
) {
	size_t ix, iy;
	const size_t iBx = blockIdx.x;
	const size_t iBy = blockIdx.y;
	const size_t iTx = threadIdx.x;
	const size_t iTy = threadIdx.y;
	const size_t nTx = blockDim.x;
	const size_t nTy = blockDim.y;
	ix = iBx * nTx + iTx;
	iy = iBy * nTy + iTy;
	if (ix < m && iy < n) {
		Cg[iy * ldc + ix] = float3_to_double (Ag[iy * lda + ix]);
	}
}

__host__ int32_t
cumyblas_double_to_float3 (
	const size_t m,
    const size_t n,
	const double *a,
    const size_t lda,
	float3 *c,
    const size_t ldc
) {
	dim3 threads = dim3 (NTX_CONV, NTY_CONV);
	dim3 grid = dim3 (ceil((float)m/NTX_CONV), ceil((float)n/NTY_CONV));
	cumyblas_double_to_float3_kernel <<< grid, threads >>> (m, n, a, lda, c, ldc);
	return 0;
}

__host__ int32_t
cumyblas_float3_to_double (
	const size_t m,
    const size_t n,
	const float3 *a,
    const size_t lda,
	double *c,
    const size_t ldc
) {
	dim3 threads = dim3 (NTX_CONV, NTY_CONV);
	dim3 grid = dim3 (ceil((float)m/NTX_CONV), ceil((float)n/NTY_CONV));
	cumyblas_float3_to_double_kernel <<< grid, threads >>> (m, n, a, lda, c, ldc);
	return 0;
}

__host__ int32_t
cumyblas_qtw_dgemm (
	const char transA,
    const char transB,
	const size_t m,
    const size_t n,
    const size_t k,
	const double alpha,
	const double *a,
    const size_t lda,
	const double *b,
    const size_t ldb,
	const double beta,
	double *c,
    const size_t ldc
) {
    if (transA != 'n' && transA != 'N') {
        if (transB != 'n' && transB != 'N') {
	        fprintf (OUTPUT, "MyGEMM is available only for NN\n");
        	exit(1);
        }
    }
	dim3 threads = dim3 (NTX_QTWGEMM, NTY_QTWGEMM);
	dim3 grid = dim3 (ceil((float)m/NTX_QTWGEMM/REGBKA_QTW_DGEMM), ceil((float)n/NTY_QTWGEMM/REGBKB_QTW_DGEMM));
	cumyblas_qtw_dgemm_nn_kernel <<< grid, threads >>> (m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
	return 0;
}

__host__ int32_t
cumyblas_qtwgemm (
	const char transA,
    const char transB,
	const size_t m,
    const size_t n,
    const size_t k,
	const double alpha,
	const float3 *a,
    const size_t lda,
	const float3 *b,
    const size_t ldb,
	const double beta,
	float3 *c,
    const size_t ldc
) {
    if (transA != 'n' && transA != 'N') {
        if (transB != 'n' && transB != 'N') {
	        fprintf (OUTPUT, "MyGEMM is available only for NN\n");
        	exit(1);
        }
    }
	dim3 threads = dim3 (NTX_QTWGEMM, NTY_QTWGEMM);
	dim3 grid = dim3 (ceil((float)m/NTX_QTWGEMM/REGBKA_QTWGEMM), ceil((float)n/NTY_QTWGEMM/REGBKB_QTWGEMM));
	cumyblas_qtwgemm_nn_kernel <<< grid, threads >>> (m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
	return 0;
}

__host__ int32_t
cumyblas_dgemm (
	const char transA,
    const char transB,
	const size_t m,
    const size_t n,
    const size_t k,
	const double alpha,
	const double *a,
    const size_t lda,
	const double *b,
    const size_t ldb,
	const double beta,
	double *c,
    const size_t ldc
) {
    if (transA != 'n' && transA != 'N') {
        if (transB != 'n' && transB != 'N') {
	        fprintf (OUTPUT, "MyGEMM is available only for NN\n");
        	exit(1);
        }
    }
	dim3 threads = dim3 (NTX_DGEMM, NTY_DGEMM);
    dim3 grid = dim3 (ceil((float)m/NTX_DGEMM/REGBKA_DGEMM), ceil((float)n/NTY_DGEMM/REGBKB_DGEMM));
	cumyblas_dgemm_nn_kernel <<< grid, threads >>> (m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
	return 0;
}

__host__ int32_t
cumyblas_sgemm (
	const char transA,
    const char transB,
	const size_t m,
    const size_t n,
    const size_t k,
	const float alpha,
	const float *a,
    const size_t lda,
	const float *b,
    const size_t ldb,
	const float beta,
	float *c,
    const size_t ldc
) {
    if (transA != 'n' && transA != 'N') {
        if (transB != 'n' && transB != 'N') {
	        fprintf (OUTPUT, "MyGEMM is available only for NN\n");
        	exit(1);
        }
    }
	dim3 threads = dim3 (NTX_SGEMM, NTY_SGEMM);
	dim3 grid = dim3 (ceil((float)m/NTX_SGEMM/REGBKA_SGEMM), ceil((float)n/NTY_SGEMM/REGBKB_SGEMM));
	cumyblas_sgemm_nn_kernel <<< grid, threads >>> (m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
	return 0;
}

/*
__host__ int32_t
cumyblas_igemm (
	const char transA,
    const char transB,
	const size_t m,
    const size_t n,
    const size_t k,
	const int32_t alpha,
	const int32_t *a,
    const size_t lda,
	const int32_t *b,
    const size_t ldb,
	const int32_t beta,
	int32_t *c,
    const size_t ldc
) {
printf ("cumyblas_igemm");
	dim3 threads = dim3 (NTX_SGEMM, NTY_SGEMM);
	dim3 grid = dim3 (ceil((float)m/NTX_SGEMM/REGBKA_SGEMM), ceil((float)n/NTY_SGEMM/REGBKB_SGEMM));
    if (transB == 'n' || transB == 'N') {
        if (transA == 'n' || transA == 'N') {
	        cumyblas_igemm_nn_kernel <<< grid, threads >>> (m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        } else {
	        cumyblas_igemm_tn_kernel <<< grid, threads >>> (m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }
    } else {
	    fprintf (OUTPUT, "GEMM-TT and NT is not available\n");
	    exit(1);
    }
	return 0;
}
*/


__global__ void cumyblas_igemm_nn_1_1_kernel (
	const size_t m,
	const size_t n,
	const size_t k,
	const int32_t alpha,
	const int32_t *Ag,
	const size_t lda,
	const int32_t *Bg,
	const size_t ldb,
	const int32_t beta,
	int32_t *Cg,
	const size_t ldc
) {
	const int32_t iBx = blockIdx.x;
	const int32_t nTx = blockDim.x;
	const int32_t iTx = threadIdx.x;
	const int32_t addrx = iBx*nTx+iTx;
    if (addrx == 0) {
        for (int32_t j = 0; j < n; j++) {
            for (int32_t i = 0; i < m; i++) {
                int32_t Tr1 = 0.;
                for (int32_t ki = 0; ki < k; ki++) {
        		    int32_t Ar1 = Ag[ki*lda + i];
        	        int32_t Br1 = Bg[j*ldb + ki];
                    Tr1 = Ar1 * Br1 + Tr1;
                }
                int32_t Cr1 = Cg[j*ldc + i];
     	        Cg[j*ldc + i] = alpha * Tr1 + beta * Cr1;
            }
        }
    }
}

__global__ void cumyblas_igemm_tn_1_1_kernel (
	const size_t m,
	const size_t n,
	const size_t k,
	const int32_t alpha,
	const int32_t *Ag,
	const size_t lda,
	const int32_t *Bg,
	const size_t ldb,
	const int32_t beta,
	int32_t *Cg,
	const size_t ldc
) {
	const int32_t iBx = blockIdx.x;
	const int32_t nTx = blockDim.x;
	const int32_t iTx = threadIdx.x;
	const int32_t addrx = iBx*nTx+iTx;
    if (addrx == 0) {
//        printf ("===igemm1x1===\n");
        for (int32_t j = 0; j < n; j++) {
            for (int32_t i = 0; i < m; i++) {
                int32_t Tr1 = 0.;
                for (int32_t ki = 0; ki < k; ki++) {
        		    int32_t Ar1 = Ag[i*lda + ki];
//                    printf ("===igemm1x1=== Ar1=%d\n", Ar1);
        	        int32_t Br1 = Bg[j*ldb + ki];
//                    printf ("===igemm1x1=== Br1=%d\n", Br1);
                    Tr1 = Ar1 * Br1 + Tr1;
                }
                int32_t Cr1 = Cg[j*ldc + i];
     	        Cg[j*ldc + i] = alpha * Tr1 + beta * Cr1;
//                printf ("===igemm1x1=== Cr1=%d\n", Cg[j*ldc + i]);
            }
        }
    }
}

__host__ int32_t
cumyblas_igemm (
	const char transA,
    const char transB,
	const size_t m,
    const size_t n,
    const size_t k,
	const int32_t alpha,
	const int32_t *a,
    const size_t lda,
	const int32_t *b,
    const size_t ldb,
	const int32_t beta,
	int32_t *c,
    const size_t ldc
) {
//    printf ("*cumyblas_igemm\n");
	dim3 threads = dim3 (1);
	dim3 grid = dim3 (1);
    if (transB == 'n' || transB == 'N') {
        if (transA == 't' || transA == 'T')
        	cumyblas_igemm_tn_1_1_kernel <<< grid, threads >>> (m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        else
        	cumyblas_igemm_nn_1_1_kernel <<< grid, threads >>> (m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } else {
	    fprintf (OUTPUT, "MyGEMM is available only for NN and TN\n");
        exit(1);
    }
	return 0;
}


/*
__global__ void cumyblas_sgemm_nn_1_1_kernel (
	const size_t m,
	const size_t n,
	const size_t k,
	const float alpha,
	const float *Ag,
	const size_t lda,
	const float *Bg,
	const size_t ldb,
	const float beta,
	float *Cg,
	const size_t ldc
) {
	const int32_t iBx = blockIdx.x;
	const int32_t nTx = blockDim.x;
	const int32_t iTx = threadIdx.x;
	const int32_t addrx = iBx*nTx+iTx;
    if (addrx == 0) {
        printf ("===igemm1x1===\n");
        for (int32_t j = 0; j < n; j++) {
            for (int32_t i = 0; i < m; i++) {
                float Tr1 = 0.;
                for (int32_t ki = 0; ki < k; ki++) {
        		    float Ar1 = Ag[ki*lda + i];
                    printf ("===igemm1x1=== Ar1=%d\n", Ar1);
        	        float Br1 = Bg[j*ldb + ki];
                    printf ("===igemm1x1=== Br1=%d\n", Br1);
                    Tr1 = Ar1 * Br1 + Tr1;
                }
                float Cr1 = Cg[j*ldc + i];
     	        Cg[j*ldc + i] = alpha * Tr1 + beta * Cr1;
                printf ("===igemm1x1=== Cr1=%d\n", Cg[j*ldc + i]);
            }
        }
    }
}

__global__ void cumyblas_sgemm_tn_1_1_kernel (
	const size_t m,
	const size_t n,
	const size_t k,
	const float alpha,
	const float *Ag,
	const size_t lda,
	const float *Bg,
	const size_t ldb,
	const float beta,
	float *Cg,
	const size_t ldc
) {
	const int32_t iBx = blockIdx.x;
	const int32_t nTx = blockDim.x;
	const int32_t iTx = threadIdx.x;
	const int32_t addrx = iBx*nTx+iTx;
    if (addrx == 0) {
        printf ("===igemm1x1===\n");
        for (int32_t j = 0; j < n; j++) {
            for (int32_t i = 0; i < m; i++) {
                float Tr1 = 0.;
                for (int32_t ki = 0; ki < k; ki++) {
        		    float Ar1 = Ag[i*lda + ki];
                    printf ("===igemm1x1=== Ar1=%d\n", Ar1);
        	        float Br1 = Bg[j*ldb + ki];
                    printf ("===igemm1x1=== Br1=%d\n", Br1);
                    Tr1 = Ar1 * Br1 + Tr1;
                }
                float Cr1 = Cg[j*ldc + i];
     	        Cg[j*ldc + i] = alpha * Tr1 + beta * Cr1;
                printf ("===igemm1x1=== Cr1=%d\n", Cg[j*ldc + i]);
            }
        }
    }
}

__host__ int32_t
cumyblas_sgemm (
	const char transA,
    const char transB,
	const size_t m,
    const size_t n,
    const size_t k,
	const float alpha,
	const float *a,
    const size_t lda,
	const float *b,
    const size_t ldb,
	const float beta,
	float *c,
    const size_t ldc
) {
	dim3 threads = dim3 (1);
	dim3 grid = dim3 (1);
    if (transB == 'n' || transB == 'N') {
        if (transA == 't' || transA == 'T')
        	cumyblas_sgemm_tn_1_1_kernel <<< grid, threads >>> (m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        else
        	cumyblas_sgemm_nn_1_1_kernel <<< grid, threads >>> (m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } else {
	    fprintf (OUTPUT, "MyGEMM is available only for NN and TN\n");
        exit(1);
    }
	return 0;
}
*/
