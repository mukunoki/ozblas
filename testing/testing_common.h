#ifndef TESTING_COMMON
#define TESTING_COMMON

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// header ------------------------------------------
#include "testing_setting.h"
#undef _IEEE_

#define __STDC_WANT_IEC_60559_TYPES_EXT__
#include <cstdio>
#include <cstdlib>
#include <cmath> 

#include <string.h>
#include <stdint.h>
#include <float.h>
//#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <sys/time.h>
#include <unistd.h>
#include <getopt.h>
#include <ctype.h>
#include <sys/utsname.h>

#if defined (CSRMV) || defined (CSRMM) || defined (CG)
extern "C" {
#include <bebop/smc/sparse_matrix.h>
#include <bebop/smc/sparse_matrix_ops.h>
#include <bebop/smc/csr_matrix.h>
}
#endif

#if defined (MKL)
//#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_trans.h>
//#include <mkl_spblas.h> // conflict with Bebop SMC
#elif defined (SSL2)
#include "/opt/FJSVxtclanga/tcsds-1.2.33/include/cblas.h"
#else
#include <cblas.h>
#endif

#if defined (ARM)
#define __float128 long double
#if defined (POWER_API) // only on Fugaku
#include "pwr.h"
#endif
#endif

#if defined (CUBLAS) 
#include <cublas_v2.h>
#elif defined (CUOZBLAS)
#include <cuozblas.h>
#elif defined (OZBLAS)
#include <ozblas.h>
#elif defined (CBLAS)
#include <ozblas.h> // for blasRcsrmv/mm in ozblas.h
#elif defined (CUMYBLAS)
#include <cuozblas.h>
#endif

#if defined (CUDA)
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#endif

#if !defined (MPLAPACK)
#define mpreal double
#endif

// ===========================================================
// common for FP_TYPE
#if defined (PREC_Q_D) || defined (PREC_Q_S)
#define PREC_Q
#elif defined (PREC_D_D) || defined (PREC_D_S) || defined (PREC_D_H) || defined (PREC_D_I)
#define PREC_D
#elif defined (PREC_S_S) || defined (PREC_S_H) || defined (PREC_S_D)
#define PREC_S
#elif defined (PREC_H_H) 
#define PREC_H
#elif defined (PREC_32F_PDT_32F)
#define PREC_S
#elif defined (PREC_32F_TC_32F)
#define PREC_S
#elif defined (PREC_32F_TC_32TF)
#define PREC_S
#elif defined (PREC_16F_TC_32F)
#define PREC_H
#elif defined (PREC_16F_PDT_32F)
#define PREC_H
#elif defined (PREC_16F_TC_16F)
#define PREC_H
#elif defined (PREC_16F_PDT_16F)
#define PREC_H
#elif defined (PREC_64F_PDT_64F)
#define PREC_D
#elif defined (PREC_64F_TC_64F)
#define PREC_D
#endif
// ===========================================================
	
#if defined (MPLAPACK)
#if defined (PREC_Q)
#define MPFR_WANT_FLOAT128
#define _Float128 __float128 // this is needed if MPFR is >= 4.1.0
#endif
#include <mplapack/mpblas_mpfr.h>
#include <mpfr.h> // this is after above
#endif
#undef _Float128

// +++++++++++++++++++++
// FP_TYPE == binary128
// +++++++++++++++++++++
#if defined (PREC_Q)
#define FP_TYPE	__float128
#if defined (ARM)
#define FABS fabs
#define FP_MAX LDBL_MAX
#define FP_MIN LDBL_MIN
#define F_PI 3.1415926535897932384626433832795029L;
#else
#include <quadmath.h>
#define FABS fabsq
#define FP_MAX FLT128_MAX
#define FP_MIN FLT128_MIN
#define F_PI M_PIq
#endif
#if defined (MPLAPACK)
#include <mplapack/mpblas__Float128.h>
#endif

// +++++++++++++++++++++
// FP_TYPE == double-double
// +++++++++++++++++++++
#elif defined (PREC_DD)
#define FP_TYPE dd_real
#define FABS fabs
#if defined (MPLAPACK)
#include <qd/dd_real.h>
#include <mplapack/mpblas_dd.h>
#endif
#define FP_MAX DBL_MAX
#define FP_MIN DBL_MIN
#define F_PI dd_real::_pi

// +++++++++++++++++++++
// FP_TYPE == binary64
// +++++++++++++++++++++
#elif defined (PREC_D)
#define FP_TYPE	double
#define FABS fabs
#define FP_MAX DBL_MAX
#define FP_MIN DBL_MIN
#define F_PI M_PI

// +++++++++++++++++++++
// FP_TYPE == binary32
// +++++++++++++++++++++
#elif defined (PREC_S)
#define FP_TYPE	float
#define FABS fabs
#define FP_MAX FLT_MAX
#define FP_MIN FLT_MIN
#define F_PI M_PI

// +++++++++++++++++++++
// FP_TYPE == binary16
// +++++++++++++++++++++
#elif defined (PREC_H)
//#include "half-2.1.0/half.hpp"
//using half_float::half;
#define FP_TYPE __fp16//_Float16
#define FABS fabs
#define FP_MAX 65504.
#define FP_MIN (2.e-14)-1
#define F_PI M_PI

#endif

// ===========================================================
// reference BLAS
#if defined (MPLAPACK)
#define refRdot												Rdot
#define refRgemv(tran,m,n,al,A,lda,X,ix,bt,Y,iy)			Rgemv(&tran,m,n,al,A,lda,X,ix,bt,Y,iy)
#define refRgemm(tranA,tranB,m,n,k,al,A,lda,B,ldb,bt,C,ldc) Rgemm(&tranA,&tranB,m,n,k,al,A,lda,B,ldb,bt,C,ldc)
#else
#define refRdot	cblas_ddot
#define refRgemv(tran,m,n,al,A,lda,X,ix,bt,Y,iy)			cblas_dgemv(CblasColMajor,ToCblasOp(tran),m,n,al,A,lda,X,ix,bt,Y,iy)
#define refRgemm(tranA,tranB,m,n,k,al,A,lda,B,ldb,bt,C,ldc) cblas_dgemm(CblasColMajor,ToCblasOp(tranA),ToCblasOp(tranB),m,n,k,al,A,lda,B,ldb,bt,C,ldc)
#endif

// ===========================================================
// argment transformation
#if defined (OZBLAS) 
#define trgRdot(ha,n,x,ix,y,iy)								rdot(&ha,n,x,ix,y,iy)
#define trgRnrm2(ha,n,x,ix)									rnrm2(&ha,n,x,ix)
#define trgRaxpy(ha,n,a,x,ix,y,iy)							raxpy(&ha,n,a,x,ix,y,iy)
#define	trgRgemv(ha,ta,m,n,al,a,la,x,ix,bt,y,iy)			rgemv(&ha,ta,m,n,al,a,la,x,ix,bt,y,iy)
#define trgRgemm(ha,ta,tb,m,n,k,al,A,lda,B,ldb,bt,C,ldc)	rgemm(&ha,ta,tb,m,n,k,al,A,lda,B,ldb,bt,C,ldc)
#define	trgRcsrmv(ha,ta,m,n,nnz,al,da,a,ci,rp,x,bt,y)		rcsrmv(&ha,ta,m,n,nnz,al,da,a,ci,rp,x,bt,y)
#define	trgRcsrmvSplitA(ha,ta,m,n,nnz,da,a,rp)				rcsrmvSplitA(&ha,ta,m,n,nnz,da,a,rp)
#define	trgRcg(ha,ta,n,nnz,da,a,ci,rp,b,x,itr,tol)			rcg(&ha,ta,n,nnz,da,a,ci,rp,b,x,itr,tol)	
#elif defined (CBLAS)
#define trgRdot(ha,n,x,ix,y,iy)								rdot(n,x,ix,y,iy)
#define trgRnrm2(ha,n,x,ix)									rnrm2(n,x,ix)
#define trgRaxpy(ha,n,a,x,ix,y,iy)							raxpy(n,a,x,ix,y,iy)
#define	trgRgemv(ha,ta,m,n,al,a,la,x,ix,bt,y,iy)			rgemv(CblasColMajor,ToCblasOp(ta),m,n,al,a,la,x,ix,bt,y,iy)
#define trgRgemm(ha,ta,tb,m,n,k,al,A,lda,B,ldb,bt,C,ldc)	rgemm(CblasColMajor,ToCblasOp(ta),ToCblasOp(tb),m,n,k,al,A,lda,B,ldb,bt,C,ldc)
#define	trgRcsrmv(ha,ta,m,n,nnz,al,da,a,ci,rp,x,bt,y)		rcsrmv(ta,m,n,nnz,al,da,a,ci,rp,x,bt,y)
#define	trgRcsrmm(ha,ta,m,l,n,nnz,al,da,a,ci,rp,x,ldx,bt,y,ldy)		rcsrmm(ta,m,l,n,nnz,al,da,a,ci,rp,x,ldx,bt,y,ldy)
//#define	trgRcg(ha,ta,n,nnz,da,a,ci,rp,b,x,itr,tol)			rcg(&ha,ta,n,nnz,da,a,ci,rp,b,x,itr,tol)	
#elif defined (CUOZBLAS)
#define trgRdot(ha,n,x,ix,y,iy,r)							rdot(&ha,n,x,ix,y,iy,r)
#define trgRnrm2(ha,n,x,ix,r)								rnrm2(&ha,n,x,ix,r)
#define trgRaxpy(ha,n,a,x,ix,y,iy)							raxpy(&ha,n,a,x,ix,y,iy)
#define	trgRgemv(ha,ta,m,n,al,a,la,x,ix,bt,y,iy)			rgemv(&ha,ta,m,n,al,a,la,x,ix,bt,y,iy)
#define trgRgemm(ha,ta,tb,m,n,k,al,A,lda,B,ldb,bt,C,ldc)	rgemm(&ha,ta,tb,m,n,k,al,A,lda,B,ldb,bt,C,ldc)
#define	trgRcsrmv(ha,ta,m,n,nnz,al,da,a,ci,rp,x,bt,y)		rcsrmv(&ha,ta,m,n,nnz,al,da,a,ci,rp,x,bt,y)
#define	trgRcsrmvSplitA(ha,ta,m,n,nnz,da,a,rp)				rcsrmvSplitA(&ha,ta,m,n,nnz,da,a,rp)
#define	trgRcg(ha,ta,n,nnz,da,a,ci,rp,b,x,itr,tol)			rcg(&ha,ta,n,nnz,da,a,ci,rp,b,x,itr,tol)	
#define	trgRcgfr(ha,ta,n,nnz,da,a,ci,rp,b,x,itr,tol)		rcgfr(&ha,ta,n,nnz,da,a,ci,rp,b,x,itr,tol)	
#define	trgRcgfr_x(ha,ta,n,nnz,da,a,ci,rp,b,x,itr,tol)		rcgfr_x(&ha,ta,n,nnz,da,a,ci,rp,b,x,itr,tol)	
#elif defined (CUBLAS) && defined (GEMMEX)
#define trgRgemm(ha,ta,tb,m,n,k,al,A,lda,B,ldb,bt,C,ldc)	cublasGemmEx(ha,ToCublasOp(ta),ToCublasOp(tb),m,n,k,&al,A,DATA_TYPE_A,lda,B,DATA_TYPE_B,ldb,&bt,C,DATA_TYPE_C,ldc,COMP_TYPE,ALGO_TYPE)
#elif defined (CUBLAS)
#define trgRdot(ha,n,x,ix,y,iy,r)							rdot(ha,n,x,ix,y,iy,r)
#define trgRnrm2(ha,n,x,ix)									rnrm2(ha,n,x,ix)
#define trgRaxpy(ha,n,a,x,ix,y,iy)							raxpy(ha,n,a,x,ix,y,iy)
#define	trgRgemv(ha,ta,m,n,al,a,la,x,ix,bt,y,iy)			rgemv(ha,ToCublasOp(ta),m,n,&al,a,la,x,ix,&bt,y,iy)
#define trgRgemm(ha,ta,tb,m,n,k,al,A,lda,B,ldb,bt,C,ldc)	rgemm(ha,ToCublasOp(ta),ToCublasOp(tb),m,n,k,&al,A,lda,B,ldb,&bt,C,ldc)
#define	trgRcsrmv(ha,ta,m,n,nnz,al,da,a,ci,rp,x,bt,y)		rcsrmv(ha,ToCusparseOp(ta),m,n,nnz,al,da,a,ci,rp,x,bt,y)
//#define	trgRcg(ha,ta,n,nnz,da,a,ci,rp,b,x,itr,tol)			rcg(&ha,ta,n,nnz,da,a,ci,rp,b,x,itr,tol)	
#elif defined (MPBLAS)
#define trgRdot(ha,n,x,ix,y,iy)								rdot(n,x,ix,y,iy)
#define trgRnrm2(ha,n,x,ix)									rnrm2(n,x,ix)
#define trgRaxpy(ha,n,a,x,ix,y,iy)							raxpy(ha,n,a,x,ix,y,iy)
#define	trgRgemv(ha,ta,m,n,al,a,la,x,ix,bt,y,iy)			rgemv(&ta,m,n,al,a,la,x,ix,bt,y,iy)
#define trgRgemm(ha,ta,tb,m,n,k,al,A,lda,B,ldb,bt,C,ldc)	rgemm(&ta,&tb,m,n,k,al,A,lda,B,ldb,bt,C,ldc)
//#define	trgRcg(ha,ta,n,nnz,da,a,ci,rp,b,x,itr,tol)			rcg(&ha,ta,n,nnz,da,a,ci,rp,b,x,itr,tol)	
#elif defined (CUMYBLAS)
#define trgRgemm(ha,ta,tb,m,n,k,al,A,lda,B,ldb,bt,C,ldc)	rgemm(ta,tb,m,n,k,al,A,lda,B,ldb,bt,C,ldc)
#endif

// ===========================================================
#if defined (CUOZBLAS)
#define ozblasHandle_t		cuozblasHandle_t
#define ozblasCreate 		cuozblasCreate
#define ozblasDestroy		cuozblasDestroy
#endif

// ================================================
#if defined (PREC_Q_D)
#if defined (CUOZBLAS)
#define rcg				cuozblasRcg<__float128,double>
#define rdot			cuozblasRdot<__float128,double,double>
#define rnrm2			cuozblasRnrm2<__float128,double,double>
#define raxpy			cuozblasRaxpy<__float128>
#define rgemv			cuozblasRgemv<__float128,double>
#define rgemm			cuozblasRgemm<__float128,double,double>
#define rcsrmv			cuozblasRcsrmv<__float128,double>
#define rcsrmvSplitA	cuozblasRcsrmvSplitA<__float128,double>
#elif defined (OZBLAS)
#define rcg				ozblasRcg<__float128,double>
#define rdot			ozblasRdot<__float128,double,double>
#define rnrm2			ozblasRnrm2<__float128,double,double>
#define rgemv			ozblasRgemv<__float128,double,double>
#define rgemm			ozblasRgemm<__float128,double,double>
#define rcsrmv			ozblasRcsrmv<__float128,double>
#define rcsrmvSplitA	ozblasRcsrmvSplitA<__float128,double>
#elif defined (MPBLAS)
#define	rdot			Rdot
#define	rgemv			Rgemv
#define	rgemm			Rgemm
#endif

// ================================================
#elif defined (PREC_Q_S)
#if defined (CUOZBLAS)
#define rcg				cuozblasRcg<__float128,float>
#define rdot			cuozblasRdot<__float128,float,float>
#define rnrm2			cuozblasRnrm2<__float128,float,float>
#define raxpy			cuozblasRaxpy<__float128>
#define rgemv			cuozblasRgemv<__float128,float,float>
#define rgemm			cuozblasRgemm<__float128,float,float>
#define rcsrmv			cuozblasRcsrmv<__float128,float>
#define rcsrmvSplitA	cuozblasRcsrmvSplitA<__float128,float>
#elif defined (OZBLAS)
#define rcg				ozblasRcg<__float128,float>
#define rdot			ozblasRdot<__float128,float,float>
#define rnrm2			ozblasRnrm2<__float128,float,float>
#define rgemv			ozblasRgemv<__float128,float,float>
#define rgemm			ozblasRgemm<__float128,float,float>
#define rcsrmv			ozblasRcsrmv<__float128,float>
#define rcsrmvSplitA	ozblasRcsrmvSplitA<__float128,float>
#endif

// ================================================
#elif defined (PREC_D_S)
#if defined (CUOZBLAS)
#define rcg				cuozblasRcg<double,float>
#define rdot			cuozblasRdot<double,float,float>
#define rnrm2			cuozblasRnrm2<double,float,float>
#define raxpy			cuozblasRaxpy<double>
#define rgemv			cuozblasRgemv<double,float,float>
#define rgemm			cuozblasRgemm<double,float,float>
#define rcsrmv			cuozblasRcsrmv<double,float>
#define rcsrmvSplitA	cuozblasRcsrmvSplitA<double,float>
#elif defined (OZBLAS)
#define rcg				ozblasRcg<double,float>
#define rdot			ozblasRdot<double,float,float>
#define rnrm2			ozblasRnrm2<double,float,float>
#define rgemv			ozblasRgemv<double,float,float>
#define rgemm			ozblasRgemm<double,float,float>
#define rcsrmv			ozblasRcsrmv<double,float>
#define rcsrmvSplitA	ozblasRcsrmvSplitA<double,float>
#endif

// ================================================
#elif defined (PREC_D_H)
#if defined (CUOZBLAS)
#define rcg				cuozblasRcg<double,half>
#define rdot			cuozblasRdot<double,half,float>
#define rnrm2			cuozblasRnrm2<double,half,float>
#define raxpy			cuozblasRaxpy<double>
#define rgemv			cuozblasRgemv<double,half,float>
#define rgemm			cuozblasRgemm<double,half,float>
#define rcsrmv			cuozblasRcsrmv<double,half>
#define rcsrmvSplitA	cuozblasRcsrmvSplitA<double,half>
#elif defined (OZBLAS)
/*
#define rcg				ozblasRcg<double,half>
#define rdot			ozblasRdot<double,half>
#define rnrm2			ozblasRnrm2<double,half>
#define rgemv			ozblasRgemv<double,half>
#define rgemm			ozblasRgemm<double,half>
#define rcsrmv			ozblasRcsrmv<double,half>
#define rcsrmvSplitA	ozblasRcsrmvSplitA<double,half>
*/
#endif

// ================================================
#elif defined (PREC_D_D)
#if defined (CUOZBLAS)
#define rcg				cuozblasRcg<double,double>
#define rcgfr			cuozblasRcgfr
#define rcgfr_x			cuozblasRcgfr_x
#define rdot			cuozblasRdot<double,double,double>
#define rnrm2			cuozblasRnrm2<double,double,double>
#define raxpy			cuozblasRaxpy<double>
#define rgemv			cuozblasRgemv<double,double,double>
//#define rgemm			cuozblasDgemm
#define rgemm			cuozblasRgemm<double,double,double>
#define rcsrmv			cuozblasRcsrmv<double,double>
#define rcsrmvSplitA	cuozblasRcsrmvSplitA<double,double>
#elif defined (OZBLAS)
#define rcg				ozblasRcg<double,double>
#define rdot			ozblasRdot<double,double,double>
#define rnrm2			ozblasRnrm2<double,double,double>
#define rgemv			ozblasRgemv<double,double,double>
//#define rgemv			ozblasDgemv
#define rgemm			ozblasRgemm<double,double,double>
#define rcsrmv			ozblasRcsrmv<double,double>
#define rcsrmvSplitA	ozblasRcsrmvSplitA<double,double>
#elif defined (CUBLAS)
//#define rcg			cublasDcg
#define rdot			cublasDdot
#define rnrm2			cublasDnrm2
#define rgemv			cublasDgemv
#define rgemm			cublasDgemm
//#define rcsrmv		cublasDcsrmv
#elif defined (CBLAS)
//#define rcg			cblas_dcg
#define rdot			cblas_ddot
#define rnrm2			cblas_dnrm2
#define rgemv			cblas_dgemv
#define rgemm			cblas_dgemm
#define rcsrmv			blasRcsrmv
//#define rcsrmm			blasRcsrmm
#define rcsrmm			blasRcsrmm_x2
#endif

// ================================================
#elif defined (PREC_S_S)
#if defined (CUOZBLAS)
#define rcg				cuozblasRcg<float,float>
#define rdot			cuozblasRdot<float,float,float>
#define rnrm2			cuozblasRnrm2<float,float,float>
#define raxpy			cuozblasRaxpy<float>
#define rgemv			cuozblasRgemv<float,float,float>
#define rgemm			cuozblasRgemm<float,float,float>
#define rcsrmv			cuozblasRcsrmv<float,float>
#define rcsrmvSplitA	cuozblasRcsrmvSplitA<float,float>
#elif defined (OZBLAS)
#define rcg				ozblasRcg<float,float>
#define rdot			ozblasRdot<float,float,float>
#define rnrm2			ozblasRnrm2<float,float,float>
#define rgemv			ozblasRgemv<float,float,float>
#define rgemm			ozblasRgemm<float,float,float>
#define rcsrmv			ozblasRcsrmv<float,float>
#define rcsrmvSplitA	ozblasRcsrmvSplitA<float,float>
#elif defined (CUBLAS)
//#define rcg			cublasScg
#define rdot			cublasSdot
#define rnrm2			cublasSnrm2
#define rgemv			cublasSgemv
#define rgemm			cublasSgemm
//#define rcsrmv		cublasScsrmv
#elif defined (CBLAS)
//#define rcg			cblas_scg
#define rdot			cblas_sdot
#define rnrm2			cblas_snrm2
#define rgemv			cblas_sgemv
#define rgemm			cblas_sgemm
#define rcsrmv			blasRcsrmv
#define rcsrmm			blasRcsrmm
#endif

// ================================================
#elif defined (PREC_S_H)
#if defined (CUOZBLAS)
#define rcg				cuozblasRcg<float,half>
#define rdot			cuozblasRdot<float,half,float>
#define rnrm2			cuozblasRnrm2<float,half,float>
#define raxpy			cuozblasRaxpy<float>
#define rgemv			cuozblasRgemv<float,half,float>
#define rgemm			cuozblasRgemm<float,half,float>
#define rcsrmv			cuozblasRcsrmv<float,half>
#define rcsrmvSplitA	cuozblasRcsrmvSplitA<float,half>
#elif defined (OZBLAS)
/*
#define rcg				ozblasRcg<float,half>
#define rdot			ozblasRdot<float,half>
#define rnrm2			ozblasRnrm2<float,half>
#define rgemv			ozblasRgemv<float,half>
#define rgemm			ozblasRgemm<float,half>
#define rcsrmv			ozblasRcsrmv<float,half>
#define rcsrmvSplitA	ozblasRcsrmvSplitA<float,half>
*/
#endif

// ================================================
#elif defined (PREC_S_D)
#if defined (CUOZBLAS)
#define rcg				cuozblasRcg<float,double>
#define rdot			cuozblasRdot<float,double,double>
#define rnrm2			cuozblasRnrm2<float,double,double>
#define raxpy			cuozblasRaxpy<float>
#define rgemv			cuozblasRgemv<float,double,double>
#define rgemm			cuozblasRgemm<float,double,double>
#define rcsrmv			cuozblasRcsrmv<float,double>
#define rcsrmvSplitA	cuozblasRcsrmvSplitA<float,double>
#elif defined (OZBLAS)
#define rcg				ozblasRcg<float,double>
#define rdot			ozblasRdot<float,double,double>
#define rnrm2			ozblasRnrm2<float,double,double>
#define rgemv			ozblasRgemv<float,double,double>
#define rgemm			ozblasRgemm<float,double,double>
#define rcsrmv			ozblasRcsrmv<float,double>
#define rcsrmvSplitA	ozblasRcsrmvSplitA<float,double>
#endif

// ================================================
#elif defined (PREC_H_H)
#if defined (CUOZBLAS)
#elif defined (OZBLAS)
#elif defined (CUBLAS)
#elif defined (CBLAS)
//#define rcg			cblas_scg
#define rdot			fjcblas_dot_r16
//#define rnrm2			cblas_snrm2
#define rgemv			fjcblas_gemv_r16
#define rgemm			fjcblas_gemm_r16
//#define rcsrmv		cblas_scsrmv
#endif


// ================================================
#elif defined (PREC_DD)
#if defined (MPBLAS)
#define	rdot			Rdot
#define	rgemv			Rgemv
#define	rgemm			Rgemm
#endif

#endif

// ================================================
// GemmEx
#if defined (CUBLAS) && defined (GEMMEX)
#define rgemm			cublasGemmEx
// ----------------------------
#if defined (PREC_32F_PDT_32F)
#define DATA_TYPE_A CUDA_R_32F 
#define DATA_TYPE_B CUDA_R_32F 
#define DATA_TYPE_C CUDA_R_32F 
#define COMP_TYPE CUBLAS_COMPUTE_32F_PEDANTIC 
#define ALGO_TYPE CUBLAS_GEMM_DEFAULT
// ----------------------------
#elif defined (PREC_32F_TC_32F)
#define DATA_TYPE_A CUDA_R_32F 
#define DATA_TYPE_B CUDA_R_32F 
#define DATA_TYPE_C CUDA_R_32F 
#define COMP_TYPE CUBLAS_COMPUTE_32F
#define ALGO_TYPE CUBLAS_GEMM_DEFAULT
// ----------------------------
#elif defined (PREC_32F_TC_32TF)
#define DATA_TYPE_A CUDA_R_32F 
#define DATA_TYPE_B CUDA_R_32F 
#define DATA_TYPE_C CUDA_R_32F 
#define COMP_TYPE CUBLAS_COMPUTE_32F_FAST_TF32
#define ALGO_TYPE CUBLAS_GEMM_DEFAULT
// ----------------------------
#elif defined (PREC_16F_TC_32F)
#define DATA_TYPE_A CUDA_R_16F 
#define DATA_TYPE_B CUDA_R_16F 
#define DATA_TYPE_C CUDA_R_16F 
#define COMP_TYPE CUBLAS_COMPUTE_32F
#define ALGO_TYPE CUBLAS_GEMM_DEFAULT
// ----------------------------
#elif defined (PREC_16F_PDT_32F)
#define DATA_TYPE_A CUDA_R_16F 
#define DATA_TYPE_B CUDA_R_16F 
#define DATA_TYPE_C CUDA_R_16F 
#define COMP_TYPE CUBLAS_COMPUTE_32F_PEDANTIC
#define ALGO_TYPE CUBLAS_GEMM_DEFAULT
// ----------------------------
#elif defined (PREC_16F_TC_16F)
#define DATA_TYPE_A CUDA_R_16F 
#define DATA_TYPE_B CUDA_R_16F 
#define DATA_TYPE_C CUDA_R_16F 
#define COMP_TYPE CUBLAS_COMPUTE_16F
#define ALGO_TYPE CUBLAS_GEMM_DEFAULT
// ----------------------------
#elif defined (PREC_16F_PDT_16F)
#define DATA_TYPE_A CUDA_R_16F 
#define DATA_TYPE_B CUDA_R_16F 
#define DATA_TYPE_C CUDA_R_16F 
#define COMP_TYPE CUBLAS_COMPUTE_16F_PEDANTIC
#define ALGO_TYPE CUBLAS_GEMM_DEFAULT
// ----------------------------
#elif defined (PREC_64F_PDT_64F)
#define DATA_TYPE_A CUDA_R_64F 
#define DATA_TYPE_B CUDA_R_64F 
#define DATA_TYPE_C CUDA_R_64F 
#define COMP_TYPE CUBLAS_COMPUTE_64F_PEDANTIC 
#define ALGO_TYPE CUBLAS_GEMM_DEFAULT
// ----------------------------
#elif defined (PREC_64F_TC_64F)
#define DATA_TYPE_A CUDA_R_64F 
#define DATA_TYPE_B CUDA_R_64F 
#define DATA_TYPE_C CUDA_R_64F 
#define COMP_TYPE CUBLAS_COMPUTE_64F
#define ALGO_TYPE CUBLAS_GEMM_DEFAULT
// ----------------------------
#endif
#endif

#if defined (CUBLAS) && defined (PREC_D_D)
#define trgRgemm(ha,ta,tb,m,n,k,al,A,lda,B,ldb,bt,C,ldc)	cublasGemmEx (ha,ToCublasOp(ta),ToCublasOp(tb),m,n,k,(const void*)&al,(const void*)A,CUDA_R_64F, lda,(const void*)B,CUDA_R_64F, ldb,(const void*)&bt,(void*)C,CUDA_R_64F, ldc, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT)
#endif

#endif
