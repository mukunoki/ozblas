#include "testing_common.h"

// ----- testing handler ---------------------------------------------------------------
struct testingHandle_t {
	#if defined (CUDA) 
	cudaDeviceProp devprop;
	#endif
	int32_t devno;
	int32_t cc;
	double device_bytes_per_sec;
	double device_flops_per_sec_sp;
	double device_flops_per_sec_dp;
	int32_t Cores;
	int32_t DPCores;
	// testing parameter
	char mode; // c=check, p=perf
	double routine_flops;
	double routine_bytes;
	int32_t driverVersion; 
	int32_t runtimeVersion;
	int32_t nodisp;
	int32_t nSplitMax;
	int32_t splitModeFlag;
	// splitModeFlag is needed when you want to do infSplit with specified degree to save memory
	int32_t fastModeFlag;
	int32_t reproModeFlag;
	int32_t sumModeFlag;
	int32_t useBatchedGemmFlag;
	int32_t trueresFlag;
	int32_t splitEpsModeFlag;
	int32_t precxFlag;
	int32_t verbose;
	int32_t trunc;

	// dim
	int32_t dim_start;
	int32_t dim_stop;
	int32_t dim_step;
	int32_t dim_m_const;
	int32_t dim_n_const;
	int32_t dim_k_const;
	int32_t dim_m_dev;
	int32_t dim_n_dev;
	int32_t dim_k_dev;
	int32_t dim_m_hst;
	int32_t dim_n_hst;
	int32_t dim_k_hst;
	int32_t incx;
	int32_t incy;
	int32_t lda_dev;
	int32_t ldb_dev;
	int32_t ldc_dev;
	int32_t dim_step_mode;

	// options
	char tranA; // N, T, C
	char tranB; // N, T, C
	char mtx_file[256];
	int32_t phi; // for input data
	int32_t maxiter; // for iterative solvers
	float tol; //tolerance
};

// ----- CUBLAS const value function ---------------------------------------------------------------
#if defined (CUDA)
inline cublasOperation_t ToCublasOp (const char tran) {
	if (tran == 'N' || tran == 'n') return CUBLAS_OP_N;
	if (tran == 'T' || tran == 't') return CUBLAS_OP_T;
	if (tran == 'C' || tran == 'c') return CUBLAS_OP_C;
	return CUBLAS_OP_N; //default
}
inline cusparseOperation_t ToCusparseOp (const char tran) {
	if (tran == 'N' || tran == 'n') return CUSPARSE_OPERATION_NON_TRANSPOSE;
	if (tran == 'T' || tran == 't') return CUSPARSE_OPERATION_TRANSPOSE;
	if (tran == 'C' || tran == 'c') return CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
	return CUSPARSE_OPERATION_NON_TRANSPOSE; //default
}
#endif
inline CBLAS_TRANSPOSE ToCblasOp (const char tran) {
	if (tran == 'N' || tran == 'n') return CblasNoTrans;
	if (tran == 'T' || tran == 't') return CblasTrans;
	if (tran == 'C' || tran == 'c') return CblasConjTrans;
	return CblasNoTrans;
}

#if !defined (CG)
void get_routine_theoretial_performance (testingHandle_t *th) {
	int32_t s = sizeof(FP_TYPE);
	int32_t n = th->dim_n_dev;
	int32_t m; (void)m;
	m = th->dim_m_dev;
	int32_t k; (void)k;
	k = th->dim_k_dev;
	th->routine_flops = (double) ROUTINE_FLOPS (m, n, k);
	th->routine_bytes = (double) ROUTINE_BYTES (m, n, k, s);
}
#endif

void dim_hst_setup (testingHandle_t *th) {
	th->dim_m_hst = th->dim_stop;
	th->dim_n_hst = th->dim_stop;
	th->dim_k_hst = th->dim_stop;
	if (th->dim_m_const > 0) th->dim_m_hst = th->dim_m_const;
	if (th->dim_n_const > 0) th->dim_n_hst = th->dim_n_const;
	if (th->dim_k_const > 0) th->dim_k_hst = th->dim_k_const;
}

void dim_dev_setup (testingHandle_t *th) {
	th->dim_m_dev = th->dim_start;
	th->dim_n_dev = th->dim_start;
	th->dim_k_dev = th->dim_start;
	if (th->dim_m_const > 0) th->dim_m_dev = th->dim_m_const;
	if (th->dim_n_const > 0) th->dim_n_dev = th->dim_n_const;
	if (th->dim_k_const > 0) th->dim_k_dev = th->dim_k_const;
}

void dim_dev_increment (testingHandle_t *th) {
	if (th->dim_m_const == 0) {
		switch (th->dim_step_mode) {
			case 1: th->dim_m_dev *= 10; break;
			case 2: th->dim_m_dev *= 2; break;
			default: th->dim_m_dev += th->dim_step; break;
		}
	}
	if (th->dim_n_const == 0) {
		switch (th->dim_step_mode) {
			case 1: th->dim_n_dev *= 10; break;
			case 2: th->dim_n_dev *= 2; break;
			default: th->dim_n_dev += th->dim_step; break;
		}
	}
	if (th->dim_k_const == 0) {
		switch (th->dim_step_mode) {
			case 1: th->dim_k_dev *= 10; break;
			case 2: th->dim_k_dev *= 2; break;
			default: th->dim_k_dev += th->dim_step; break;
		}
	}
}

__inline__ double gettime () {
	struct timeval tv;
	#if defined (CUDA)
	cudaDeviceSynchronize ();
	#endif
	gettimeofday (&tv, NULL);
	return tv.tv_sec + (double) tv.tv_usec * 1.0e-6;
}

double getMinTime (
	const double* times,
	const int nloop
) {
	double min = times[0];
	for (int32_t i = 1; i < nloop; i++) {
		double tmp = times[i];
		if (min > tmp) min = tmp;
	}
	return min;
}

double getMaxTime (
	const double* times,
	const int nloop
) {
	double max = times[0];
	for (int32_t i = 1; i < nloop; i++) {
		double tmp = times[i];
		max = MAX (max, tmp);
	}
	return max;
}

int32_t get_num_sp_cores (int32_t cc) {
	switch (cc) {
		case 300: return 192;
		case 320: return 192;
		case 350: return 192;
		case 370: return 192;
		case 500: return 128;
		case 520: return 128;
		case 530: return 128;
		case 600: return 64;
		case 610: return 128;
		case 620: return 128;
		case 700: return 64;
		default: return 0;
	}
}

int32_t get_num_dp_cores (int32_t cc) {
	switch (cc) {
		case 300: return 64;
		case 320: return 64;
		case 350: return 64;
		case 370: return 64;
		case 500: return 0;
		case 520: return 4;
		case 530: return 0;
		case 600: return 32;
		case 610: return 4;
		case 620: return 4;
		case 700: return 32;
		default: return 0;
	}
}

void dev_setting (testingHandle_t *th) {
#if defined (CUDA)
	int32_t driverVersion = 0;
	(void)driverVersion;
	int32_t runtimeVersion = 0;
	(void)runtimeVersion;
	int32_t devcnt;
	(void)devcnt;
	cudaGetDeviceCount (&devcnt);
	cudaDeviceProp devprop;  
	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);
	th->driverVersion = driverVersion;
	th->runtimeVersion = runtimeVersion;
	cudaSetDevice (th->devno);
	cudaGetDeviceProperties (&devprop, th->devno);
	th->devprop = devprop;
	th->cc = devprop.major * 100 + devprop.minor * 10;
	th->Cores = get_num_sp_cores (th->cc);
	th->DPCores = get_num_dp_cores (th->cc);
	th->device_bytes_per_sec = (double)th->devprop.memoryClockRate * 2. * 1000. * (double)th->devprop.memoryBusWidth / 8.;
	th->device_flops_per_sec_sp = (double)th->devprop.clockRate * 1000. * 2. * (double)th->devprop.multiProcessorCount * th->Cores;
	th->device_flops_per_sec_dp = (double)th->devprop.clockRate * 1000. * 2. * (double)th->devprop.multiProcessorCount * th->DPCores;
#endif
}

void testingCreate (
	int32_t argc,
	char **argv,
	testingHandle_t *th
	#if defined (OZBLAS) || defined (CUOZBLAS) 
	, ozblasHandle_t *oh
	#endif
) {
 	// default ---------------------------
	th->devno = 0;
	th->nodisp = 0;
	th->mode = 'c';
	th->tranA = 'N';
	th->tranB = 'N';
	th->phi = 1;
	th->maxiter = 100;
	th->trueresFlag = 0;
	th->verbose = 10;
	th->tol = 1e-8;
	th->trunc = 0;

	// ozblasHandle_t
	th->nSplitMax = 0;
	th->splitModeFlag = 0;
	th->fastModeFlag = 0;
	th->reproModeFlag = 0;
	th->sumModeFlag = 0;
	th->useBatchedGemmFlag = 0;
	th->splitEpsModeFlag = 0;
	th->precxFlag = 0;

	// dim
	th->dim_start	= 16;
	th->dim_stop	= 128;
	th->dim_step	= 16;
	th->dim_m_dev = 0;
	th->dim_n_dev = 0;
	th->dim_k_dev = 0;
	th->dim_m_hst = 0;
	th->dim_n_hst = 0;
	th->dim_k_hst = 0;
	th->dim_m_const = 0;
	th->dim_n_const = 0;
	th->dim_k_const = 0;
	th->incx = 1;
	th->incy = 1;
	th->lda_dev = 0;
	th->ldb_dev = 0;
	th->ldc_dev = 0;
	th->dim_step_mode = 0;

	#if defined (CUOZBLAS)
//	th->pmode = CUBLAS_POINTER_MODE_VALUE;
	#endif

 	// args ---------------------------
	int32_t opt, opt_index;
	optind = 1;
	struct option long_options[] = {
		{"transa", required_argument, 0, 'a'},
		{"transb", required_argument, 0, 'b'},
		{"usebatchedgemm", required_argument, NULL, 'c'},
		{"degree", required_argument, 0, 'd'},
		{"phi", required_argument, NULL, 'e'},
		{"mtx", required_argument, NULL, 'f'},
		{"help", no_argument, NULL, 'g'},
		{"repromode", required_argument, NULL, 'h'},
		{"fastmode", required_argument, NULL, 'i'},
		{"mnk", required_argument, NULL, 'j'},
		{"k", required_argument, NULL, 'k'},
		{"trueres", required_argument, NULL, 'l'},
		{"m", required_argument, NULL, 'm'},
		{"n", required_argument, NULL, 'n'},
		{"mode", required_argument, NULL, 'o'},
		{"usegpu", required_argument, NULL, 'p'},
		{"maxiter", required_argument, NULL, 'q'},
		{"range", required_argument, NULL, 'r'},
		{"trunc", required_argument, NULL, 's'},
		{"summode", required_argument, NULL, 't'},
		{"splitmode", required_argument, NULL, 'u'},
		{"verbose", required_argument, NULL, 'v'},
		{"nodisp", required_argument, NULL, 'w'},
		{"splitepsmode", required_argument, NULL, 'x'},
		{"precx", required_argument, NULL, 'y'},
		{"tol", required_argument, NULL, 'z'},
		{0, 0, 0, 0}
	};
	while ((opt = getopt_long (argc, argv, "abcdefghijklmnopqrstuwxyz", long_options, &opt_index)) != -1){
		switch (opt) {
			case 'a':{ // N, T, C
				if (optarg[0]) th->tranA= optarg[0];
				break;
			} case 'b':{ // N, T, C
				if (optarg[0]) th->tranB= optarg[0];
				break;
			} case 'c':{
				th->useBatchedGemmFlag = atoi(optarg);
				break;
			} case 'd':{
				th->nSplitMax = atoi(optarg);
				break;
			} case 'e':{
				th->phi = atoi(optarg);
				break;
			} case 'f':{
				strcpy (th->mtx_file, optarg);
				break;
			} case 'g':{
				printf ("[Test-setting]\n");
				printf ("--mode={c,p} c: check, p: performance (default: c)\n");
				printf ("--nodisp={0, 1}, 0: off, 1:on\n");
				printf ("\n");
				printf ("[OzBLAS-parameters]\n");
				printf ("--degree=N (maximum number of split matrices)\n");
				printf ("--fastmode={0,1}, 0: off, 1: on (no effect on DOT)\n");
				printf ("--usebatchedgemm={0,1}, 0: off, 1: on (no effect on DOT)\n");
				printf ("--summode={0,1,2,3}, 0: GlobalFsum, 1: GlobalNearsum, 2: LocalFsum, 3: LocalFsum3\n");
				printf ("  (1 is not available on mixed-precision routines, 3 is available only when usebatchedgemm=0)\n");
				printf ("--splitmode={0,1,3}, 0: normal, 1: inf-split, 3: split3\n");
				printf ("\n");
				printf ("[Input (problem size)]\n");
				printf ("--mnk=N (problem size)\n");
				printf ("--m=N\n");
				printf ("--n=N\n");
				printf ("--k=N\n");
				printf ("--range=START:END:STEP\n");
				printf ("\n");
				printf ("[Input (condition)]\n");
				printf ("--trunc=N (N-bit truncation for inputs)\n");
				printf ("--phi=N (phi or erange for initialization)\n");
				printf ("--transa={n,t}: transpose A\n");
				printf ("--transb={n,t}: transpose B\n");
				printf ("--mtx=*.mtx (sparse matrix file)\n");
				printf ("\n");
				printf ("[CG solvers]\n");
				printf ("--maxiter=N (the maximum number of iterations)\n");
				printf ("--tol=F (tolerance for stopping iterations)\n");
				printf ("--trueres={0,1}, 0: off, 1: on\n");
				printf ("--verbose={0,1}, 0: off, 1: on\n");
				exit (0);
				break;
			} case 'h':{
				th->reproModeFlag = atoi(optarg);
				break;
			} case 'i':{
				th->fastModeFlag = atoi(optarg);
				break;
			} case 'j':{
				th->dim_start = atoi (optarg);
				th->dim_stop = atoi (optarg);
				th->dim_step = 1;
				break;
			} case 'k':{
				th->dim_k_const = atoi(optarg);
				break;
			} case 'l':{
				th->trueresFlag = atoi(optarg);
				break;
			} case 'm':{
				th->dim_m_const = atoi(optarg);
				break;
			} case 'n':{
				th->dim_n_const = atoi(optarg);
				break;
			} case 'o':{
				th->mode = optarg[0];
				break;
			} case 'q':{
				th->maxiter = atoi(optarg);
				break;
			} case 'r':{
				char *str, *c_start, *c_stop, *c_step;
				char par[] = ":";
				str = optarg;
				c_start = strtok (str, par);
				c_stop  = strtok (NULL, par);
				c_step	= strtok (NULL, par);

				th->dim_start	= atoi (c_start);
				th->dim_stop	= atoi (c_stop);

				if (strcmp (c_step, "pow") == 0)
					th->dim_step_mode	= 2;
				else 
					th->dim_step	= atoi (c_step);
				break;
			} case 's':{
				th->trunc = atoi(optarg);
				break;
			} case 't':{
				th->sumModeFlag = atoi(optarg);
				break;
			} case 'u':{
				th->splitModeFlag = atoi(optarg);
				break;
			} case 'v':{
				th->verbose = atoi(optarg);
				break;
			} case 'w':{
				th->nodisp = atoi(optarg);
				break;
			} case 'x':{
				th->splitEpsModeFlag = atoi(optarg);
				break;
			} case 'y':{
				th->precxFlag = atoi(optarg);
				break;
			} case 'z':{
				th->tol = (float)atof(optarg);
				break;
			}
		}
	}
	dev_setting (th);

	#if defined (CUOZBLAS) || defined (OZBLAS)
	oh->nSplitMax = th->nSplitMax;
	oh->splitModeFlag = th->splitModeFlag;
	oh->fastModeFlag = th->fastModeFlag;
	oh->reproModeFlag = th->reproModeFlag;
	oh->sumModeFlag = th->sumModeFlag;
	oh->useBatchedGemmFlag = th->useBatchedGemmFlag;
	oh->splitEpsModeFlag = th->splitEpsModeFlag;
	oh->precxFlag = th->precxFlag;
	#endif
}

// =========================================
// Print floating-point value with bit representation
// =========================================
typedef union{
	double d;
	uint64_t i;
} d_and_i;

void printBitsB64 (double val) {
	d_and_i di;
	di.d = val;
	// sign
	printf ("%d", (int)((di.i >> 63) & 1));
	printf ("|");
	// exponent
	for (int i = 62; i >= 62-10; i--) 
		printf ("%d", (int)((di.i >> i) & 1));
	printf ("|");
	// fraction
	for (int i = 62-11; i >= 0; i--) 
		printf ("%d", (int)((di.i >> i) & 1));
	printf (" : ");
	printf ("%+1.18e\n", val);
}

#if defined (CUDA)
union b16bit {
	half f;
	int16_t i;
};
#else
/*
union b16bit {
	__fp16 f;
	int16_t i;
};
*/
#endif
union b32bit {
	float f;
	int32_t i;
};
union b64bit {
	double f;
	int64_t i;
};
#if defined (PREC_Q)
union b128bit {
	__float128 f;
	__int128_t i;
};

typedef union{
	__float128 q;
	__int128 i;
} q_and_i;

void printBitsB128 (__float128 val) {
	q_and_i qi;
	qi.q = val;
	// sign
	printf ("%d", (int)((qi.i >> 127) & 1));
	printf ("|");
	// exponent
	for (int i = 126; i >= 126-14; i--) 
		printf ("%d", (int)((qi.i >> i) & 1));
	printf ("|");
	// fraction
	for (int i = 126-15; i >= 0; i--) 
		printf ("%d", (int)((qi.i >> i) & 1));
	printf (" : ");
	#if defined (ARM)
	printf ("%1.3Le\n", val);
	#else
	char buf[128];
	quadmath_snprintf(buf, sizeof(buf), "%+1.30Qe", val);
	puts(buf);
	#endif
}
#endif

// =========================================
// for libquadmath
// =========================================
template <typename TYPE>
TYPE isinf1 (const TYPE v) {
	return std::isinf(v);
}
template float isinf1 <float> (const float v);
template double isinf1 <double> (const double v);
#if defined (PREC_Q)
template <>
__float128 isinf1 (const __float128 v) {
	return isinfq(v);
}
#endif

template <typename TYPE>
TYPE isnan1 (const TYPE v) {
	return std::isnan(v);
}
template float isnan1 <float> (const float v);
template double isnan1 <double> (const double v);
#if defined (PREC_Q)
template <>
__float128 isnan1 (const __float128 v) {
	return isnanq(v);
}
#endif

// =========================================
// Check, conversion, etc.
// =========================================
#if defined (MPLAPACK)
double toDouble (mpreal mp) {
	return mpfr_get_d(mp, MPFR_RNDN);
}
#endif

double toDouble (const FP_TYPE fp) {
	double dp;
	#if defined (PREC_DD)
	dp = fp.x[0];
	#else
	dp = fp;
	#endif
	return dp;
}

mpreal toMpreal (const FP_TYPE fp) {
	mpreal mp;
	#if defined (MPLAPACK)
	#if defined (PREC_Q)
	mpfr_set_float128 (mp, fp, MPFR_RNDN);
	#elif defined (PREC_DD)
	mpreal mp2;
	mpfr_set_d (mp,  fp.x[0], MPFR_RNDN);
	mpfr_set_d (mp2, fp.x[1], MPFR_RNDN);
	mp += mp2;
	#else
	mpfr_set_d (mp, (double)fp, MPFR_RNDN);
	#endif
	#else
	mp = fp;
	#endif
	return mp;
}

__inline__ void
QuickTwoSum (
	const double a,
	const double b,
	double &s,
	double &e
) {
	double t;
	s = a + b;
	t = s - a;
	e = b - t;
}

FP_TYPE ToFptype (mpreal mp) {
	FP_TYPE fp;
	#if defined (MPLAPACK)
	#if defined (PREC_Q)
	fp = mpfr_get_float128 (mp, MPFR_RNDN);
	#elif defined (PREC_DD)
	double hi = toDouble (mp);
	mpreal mp2 = (mp - toMpreal(hi));
	double lo = toDouble (mp2);
	double hi_, lo_;
	QuickTwoSum (hi, lo, hi_, lo_);
	fp.x[0] = hi_;
	fp.x[1] = lo_;
	#else
	fp = toDouble (mp);
	#endif
	#else
	fp = mp;
	#endif
	return fp;
}

FP_TYPE truncate (const FP_TYPE valf, const int32_t trunc) {
	#if defined (PREC_DD) || defined (PREC_MPFR)
	fprintf (stderr, "Error: truncation is not supported on DD and MPFR.\n");
	exit (1);
	#else
	#if defined (PREC_Q)
	union b128bit bb;
	#elif defined (PREC_D)
	union b64bit bb;
	#elif defined (PREC_S)
	union b32bit bb;
	#elif defined (PREC_H)
	union b16bit bb;
	#endif
	bb.f = valf;
	bb.i = bb.i >> trunc;	
	bb.i = bb.i << trunc;	
	return bb.f;
	#endif
}

void mublasInitMat (
	testingHandle_t *th,
	const int32_t m,
	const int32_t n,
	int32_t ld,
	FP_TYPE *mat,
	const double phi,
	const int32_t mode,
	const int32_t trunc,
	const int32_t seed
) {
	int32_t j;
	if (ld < m) ld = m;

	srand48(seed);
	switch (mode) {
		case 0:	// init with constant
			#pragma omp parallel for 
			for (j = 0; j < n; j++) {
				for (int32_t i = 0; i < m; i++) {
					FP_TYPE val = phi;
					if (trunc > 0) val = truncate (val, trunc); // significand truncation
					mat[j*ld+i] = val;
				}
			}
			if (!th->nodisp) printf ("#\tinput: initialized with const-mode\t");
			break;
		case 1:	// drand48 +[0,1)
			//to check reproducibility in Fsum, disable thread parallelism
			//#pragma omp parallel for 
			for (j = 0; j < n; j++) {
				for (int32_t i = 0; i < m; i++) {
					FP_TYPE val = drand48();
					if (trunc > 0) val = truncate (val, trunc); // significand truncation
					mat[j*ld+i] = val;
				}
			}
			if (!th->nodisp) printf ("#\tinput: initialized with drand48 [0,1)\t");
			break;
		case 2:	// phi-mode
			printf ("!!! WARNING !!! phi-mode only supports double-precision.\n");
			//#pragma omp parallel for
			for (j = 0; j < n; j++) {
				for (int32_t i = 0; i < m; i++) {
					double mu = 0.0, sigma = 1.0;
					double r1 = drand48(); //rand()/((FP_TYPE)RAND_MAX+1.);
					double r2 = drand48(); //rand()/((FP_TYPE)RAND_MAX+1.);
					double x1 = mu + (std::sqrt (-2. * log(r1)) * sin (2. * M_PI * r2)) * sigma;
					double val = (r1 - 0.5) * exp (phi * x1);
					mat[j*ld+i] = (trunc > 0) ? (FP_TYPE)truncate (val, trunc) : (FP_TYPE)val; // significand truncation
				}
			}
			if (!th->nodisp) printf ("#\tinput: initialized with phi-mode\t");
			break;
		case 3: // erange-mode : initialize with +-1e(phi/2)
			if (phi == 0.) {
				fprintf (stderr, "error: phi must be > 0 on erange-mode.\n");
				exit(1);
			}
			FP_TYPE f_pi;
			#if defined (PREC_MPFR)
			mpfr_const_pi (f_pi, MPFR_RNDN);
			#else
			f_pi = F_PI;
			#endif
			//#pragma omp parallel for 
			for (j = 0; j < n; j++) {
				for (int32_t i = 0; i < m; i++) {
					FP_TYPE valf = (drand48()*9+1) * std::pow(10,rand()%(int32_t)phi);
					valf = ((rand() % 2) ? 1.:-1.) * ((double)M_PI/f_pi) * valf;
					mat[j*ld+i] = (FP_TYPE)((trunc > 0) ? truncate (valf, trunc) : valf); // significand truncation
				}
			}
			/*
			for (j = 0; j < n; j++) {
				int32_t i;
				for (i = 0; i < m; i++) {
					FP_TYPE valf = mat[j*ld+i];
					printf ("%d: ",j*n+i);
					#if defined (PREC_DD)
					printBitsB64 (valf.x[0]);
					printBitsB64 (valf.x[1]);
					#elif defined (PREC_Q_D) || defined (PREC_Q_S)
					printBitsB128 (valf);
					#else
					printBitsB64 (valf);
					#endif
				}
			}
			*/
			if (!th->nodisp) {
				printf ("#\tinput: initialized with erange-mode");
				if (trunc > 0) printf (" (with truncation %d)\t", trunc);
				else printf ("\t");
			}
			break;
	}	
	if (!th->nodisp && mode > 1) {
		FP_TYPE amax = 0., amin = FP_MAX;
#if !defined (PREC_DD)
		#pragma omp parallel for reduction(max:amax) reduction(min:amin)
#endif
		for (j = 0; j < n; j++) {
			FP_TYPE amax_local = 0.;
			FP_TYPE amin_local = FP_MAX; 
			for (int32_t i = 0; i < m; i++) {
				FP_TYPE tmp = FABS (mat[j*ld+i]);
				amax_local = MAX (amax_local, tmp);
				amin_local = MIN (amin_local, tmp);
			}
			amax = MAX (amax_local, amax);
			amin = MIN (amin_local, amin);
		}
		#if defined (PREC_Q) && !defined (ARM)
		char buf[128];
		if (amax == amin) {
			quadmath_snprintf(buf, sizeof(buf), "%.2Qf", amax); // amax == amin
			printf ("val = %s\n", buf);
		} else {
			FP_TYPE erange = log10q(amax) - log10q(amin);
			quadmath_snprintf(buf, sizeof(buf), "%.2Qf", erange);
			printf ("range = %s (max = ", buf);
			quadmath_snprintf(buf, sizeof(buf), "%.3Qe", amax); 
			printf ("%s, min = ", buf);
			quadmath_snprintf(buf, sizeof(buf), "%.3Qe", amin); 
			printf ("%s)\n", buf);
		}
		#else
		if (amax == amin) {
			printf ("val = %1.3e\n", toDouble (amax)); // amax == amin
		} else {
			FP_TYPE erange = log10 (toDouble (amax)) - log10 (toDouble (amin));
			printf ("range = %.2f (max = %1.3e, min = %1.3e)\n", toDouble (erange), toDouble (amax), toDouble (amin));
		}
		#endif
	} else {
		if (!th->nodisp) printf ("\n");
	}
}

void mublasConvMat (
	const int32_t rd,
	const int32_t cd,
	const FP_TYPE *src,
	const int32_t lds,
	mpreal *dst,
	const int32_t ldd
) {
	#pragma omp parallel for
	for (int32_t j = 0; j < cd; j++) {
		for (int32_t i = 0; i < rd; i++) 
			dst[j*ldd+i] = toMpreal (src[j*lds+i]);
	}
}

void mublasCopyMat (
	const int32_t rd,
	const int32_t cd,
	const FP_TYPE *src,
	const int32_t lds,
	FP_TYPE *dst,
	const int32_t ldd
) {
	#pragma omp parallel for
	for (int32_t j = 0; j < cd; j++) {
		for (int32_t i = 0; i < rd; i++) 
			dst[j*ldd+i] = src[j*lds+i];
	}
}

// check routine (|trg-ref|/|ref|) -------------------------------------------------
void mublasCheckMatrix (
	testingHandle_t *th,
	const size_t m,
	const size_t n,
	const FP_TYPE *target,
	const int32_t ldt,
	const mpreal *mpref,
	const int32_t ldr
) {
	size_t i, j, errcnt = 0, nicnt = 0;
	// comparison on mpreal
	#if defined (MPLAPACK) && defined (MPFR_CHECK)
	mpreal trg, ref, tmp_r0, tmp_r1, dif, rerr, rerrmax = 0.;
    for (j = 0; j < n; j++) {
    	for (i = 0; i < m; i++) {
			ref = mpref[j*ldr+i];
			FP_TYPE trgf = target[j*ldt+i];
			trg = toMpreal (trgf);
			dif = trg - ref;
			rerr = (ref == toMpreal(0.)) ? toMpreal(0.) : fabs (dif / ref);
			//printf ("[%lu,%lu] trg=%a mpref=%a dif=%+1.18e rel=%+1.18e\n", j, i, toDouble (trg), toDouble (ref), toDouble (dif), toDouble (rerr));
			if (rerrmax < rerr) rerrmax = rerr;
			#if defined (PREC_DD)
			if (isinf1(trgf.x[0]) || isnan1(trgf.x[0])) {
			#else
			if (isinf1(trgf) || isnan1(trgf)) {
			#endif
				nicnt++;
			} else {
				if (dif != 0) errcnt++;
			}
		}
    }
	#else
	// comparison on FP_TYPE
	FP_TYPE trgf, ref, dif, rerr, rerrmax = 0.;
    for (j = 0; j < n; j++) {
    	for (i = 0; i < m; i++) {
			ref = ToFptype (mpref[j*ldr+i]);
			trgf = target[j*ldt+i];
			dif = trgf - ref;
			rerr = (ref == 0.) ? 0. : FABS (dif / ref);
			//if (rerr != 0.)	printf("[%ld,%ld] trg=%1.16e ref=%1.16e diff=%1.3e (%1.3e)\n", j, i, (double)trgf, (double)ref, (double)dif, (double)rerr);
			if (rerrmax < rerr) rerrmax = rerr;
			#if defined (PREC_DD)
			if (isinf1(trgf.x[0]) || isnan1(trgf.x[0])) {
			#elif defined (PREC_H_H)
			if (isinf1((float)trgf) || isnan1((float)trgf)) {
			#else
			if (isinf1(trgf) || isnan1(trgf)) {
			#endif
				nicnt++;
			} else {
				if (dif != 0) errcnt++;
			}
		}
    }
	#endif
//	if (!th->nodisp) {
		if (nicnt > 0)  
			printf("\tNan/Inf\t%zd\t", nicnt);
		else
			printf("\t%1.4e\t%zd\t", toDouble (rerrmax), errcnt);
//	}
    /*else {
		printf ("%d\t%d\t%d", th->dim_m_dev, th->dim_n_dev, th->dim_k_dev);
		if (th->sumModeFlag == 1) {
			if (errcnt == 0 && nicnt == 0)
				printf ("\tOK");
			else 
				printf ("\tNG");
			printf ("\t%1.4e (%a)\t%zd\n", toDouble (rerrmax), toDouble (rerrmax), errcnt);
		} else {
			#if defined (PREC_Q)
			if (fabs(toDouble (rerrmax)) < 1e-30 && nicnt == 0)
			#elif defined (PREC_D)
			if (fabs(toDouble (rerrmax)) < 1e-14 && nicnt == 0)
			#elif defined (PREC_S)
			if (fabs(toDouble (rerrmax)) < 1e-6 && nicnt == 0)
			#endif
				printf ("\tOK");
			else {
				printf ("\tNG");
			}
			if (nicnt > 0)  
				printf("\tNan/Inf\t%zd\n", nicnt);
			else
				printf ("\t%1.4e (%a)\t%zd\n", toDouble (rerrmax), toDouble (rerrmax), errcnt);
		}
	}
    */
}

// print routine -------------------------------------------------
void print_routine_name () {
	#if defined (PREC_S_S)
	printf ("S");
	#elif defined (PREC_S_D)
	printf ("SD");
	#elif defined (PREC_D_S)
	printf ("DS");
	#elif defined (PREC_D_D)
	printf ("D");
	#elif defined (PREC_Q_D)
	printf ("QD");
	#elif defined (PREC_Q_S)
	printf ("QS");
	#elif defined (PREC_H)
	printf ("H");
	#else
	printf ("unknown-precision,");
	#endif
	#if defined (DOT)
	printf ("DOT\n");
	#elif defined (GEMV)
	printf ("GEMV\n");
	#elif defined (CSRMV)
	printf ("CSRMV\n");
	#elif defined (CSRMM)
	printf ("CSRMM\n");
	#elif defined (GEMM) || defined (GEMMEX)
	printf ("GEMM\n");
	#elif defined (CG)
	printf ("CG\n");
	#else
	printf ("unknown-routine\n");
	#endif
}

void print_library_name () {
	#if defined (CUOZBLAS)
	printf ("cuOzBLAS (using cuBLAS)\n");
	#elif defined (OZBLAS)
	printf ("OzBLAS");
	#if defined (MKL)
	//MKLVersion Version;
    //mkl_get_version(&Version);
	//printf (" (using MKL %d.%d.%d)", Version.MajorVersion, Version.MinorVersion, Version.UpdateVersion);
	printf (" (using MKL");
	#elif defined (OpenBLAS)
	printf (" (using OpenBLAS");
	#elif defined (SSL2)
	printf (" (using SSL2");
	#endif
	#if defined (NVBLAS)
	printf (" with NVBLAS)\n");
	#else
	printf (")\n");
	#endif
	#else
	printf ("unknown-library\n");
	#endif
}

void print_info1 (
	testingHandle_t *th
) {
	if (!th->nodisp) {
		system("echo -n '# Testing info -----------------------------\n'");
		system("echo -n '# \tDate:\t'; date");
		system("echo -n '# \tCPU:\t'; cat /proc/cpuinfo | grep 'model name' | head -n 1");
		system("echo -n '# \tOMP:\tOMP_NUM_THREADS = ' $OMP_NUM_THREADS");
		system("echo ', MKL_NUM_THREADS = ' $MKL_NUM_THREADS");
		system("echo -n '# \tOS:\t'; uname -a");
		system("echo -n '# \tHost:\t'; hostname");
		printf("# \tCompiler:\t");
		#if defined (__INTEL_COMPILER)
		printf("ICC %d\n", __INTEL_COMPILER);
		#elif defined (__FUJITSU)
		printf("FCC\n");
		#elif defined (__GNUC__)
		printf("GCC %d\n", __GNUC__);
		#endif
		#if defined (CUDA) 
		printf("# \tDevice[%d]\t%s\n", th->devno, th->devprop.name);
		printf("# \t- Theoretical Peak Memory Bandwidth:\t%3.1f [GB/s]\n", th->device_bytes_per_sec * 1.e-9);
		printf("# \t- Theoretical Peak Performance (SP):\t%4.1f [GFlops]\n", th->device_flops_per_sec_sp * 1.e-9);
		printf("# \t- Theoretical Peak Performance (DP):\t%4.1f [GFlops]\n", th->device_flops_per_sec_dp * 1.e-9);
		printf("# \t- Total Gmem:\t%4.1f [GB]\n", th->devprop.totalGlobalMem * 1e-9);
		printf("# \t- CC:\t%d.%d\n", th->devprop.major, th->devprop.minor);
		printf("# \t- MultiProcessorCount:\t%d\n", th->devprop.multiProcessorCount);
		printf("# \tDriver / Runtime:\t%d, %d\n", th->driverVersion, th->runtimeVersion);
		#endif
		printf ("#\tRoutine:\t");
		print_routine_name ();
		printf ("#\tLibrary:\t");
		print_library_name ();
		#if !defined (CG)
		if (th->mode == 'c') {
			printf ("#\tMode:\tchecking with ");
			#if defined (MPLAPACK)
			printf ("MPBLAS (MPFR %zd-bit)\n", mpfr_get_default_prec());
			#else
			printf ("CBLAS (FP64)\n");
			#endif
		} else {
			printf ("#\tmode: performance evaluation\n");
		}
		printf ("#\tNLOOP / WLOOP:\t%d/%d\n", NLOOP, WLOOP);
		#endif
		#if defined (CUOZBLAS) || defined (OZBLAS)
		printf ("# OzBLAS -----------------------------------\n");
		printf ("#\tWork-mem size = %1.1e (bytes)\n", WORK_MEM_SIZE);
		printf ("#\tDegree = %d\n", th->nSplitMax);
		printf ("#\tsplitMode = %d (0:none, 1:infSplit (warn when infSplit is failed), 3:fastSplit+infSplit)\n", th->splitModeFlag);
		// splitModeFlag is needed when you want to do infSplit with specified degree to save memory
		printf ("#\tFastMode = %d\n", th->fastModeFlag);
		printf ("#\tReproMode = %d\n", th->reproModeFlag);
		printf ("#\tSumMode = %d (0:GlobalFSum, 1:GlobalNearsum, 2:LocalFsum, 3:LocalFsum3)\n", th->sumModeFlag);
		printf ("#\tUseBatchedGemm = %d\n", th->useBatchedGemmFlag);
		printf ("#\tSplitEpsMode = %d\n", th->splitEpsModeFlag);
		printf ("#\tprecx = %d\n", th->precxFlag);
		#endif
		#if defined (CG)
		printf ("# CG parameter -----------------------------\n");
		printf ("#\ttol:\t%1.1e\n", th->tol);
		printf ("#\tmaxiter:\t%d\n", th->maxiter);
		#else
		printf ("# BLAS parameter ---------------------------\n");
		printf ("#\ttranA:\t%c\n", th->tranA);
		printf ("#\ttranB:\t%c\n", th->tranB);
		#endif
		printf ("# Problem setting --------------------------\n");
		printf ("#\tphi = %d (on erange-mode, [1e0,1e%d))\n", th->phi, th->phi);
		#if !defined (CSRMV) 
		printf ("#\tdim_start:\t%d\n", th->dim_start);
		printf ("#\tdim_stop:\t%d\n", th->dim_stop);
		printf ("#\tdim_step:\t%d\n", th->dim_step);
		if (th->dim_m_const == 0) printf ("#\tM:\trange\n");
		else printf ("#\tM:\t%d\n", th->dim_m_const);
		if (th->dim_n_const == 0) printf ("#\tN:\trange\n");
		else printf ("#\tN:\t%d\n", th->dim_n_const);
		if (th->dim_k_const == 0) printf ("#\tK:\trange\n");
		else printf ("#\tK:\t%d\n", th->dim_k_const);
		#endif
	}
}

void print_info2 (
	testingHandle_t *th
) {
	if (!th->nodisp) {
		printf ("# Evaluation result ------------------------\n");
		if (th->mode == 'p') printf ("# (* shows the value at the last execution, ** shows the value of the last block at the last execution)\n");
		#if defined (CSRMV) || defined (CSRMM) || defined (CG)
		printf ("# Mat\tM\tN\tNNZ");
		#else
		printf ("# M\tN\tK");
		#endif
		#if !defined (CG)
		if (th->mode == 'p') {
			printf ("\tTime[sec]\tPerf[GFlops]\tPerf[GB/s]\titr");
			#if defined (POWER_API)
			printf ("\tNode-power[W]\tNode-energy[J]\tNode-power-M[W]\tNode-energy-M[J]\tCores-power[W]\tCores-energy[J]\tMem-power[W]\tMem-energy[J]");
			#endif
		} else {
			printf("\terr(rlt.max)\tnum-err");
		}
		#endif
	}
	#if defined (CUOZBLAS) || defined (OZBLAS)
	printf("\t*t_SpltA\t*t_SpltB\t*t_Comp \t*t_Sum  \t*t_Other\t*t_Total\t*#sA\t*#sB\t*#GEMMs\t*Mem[GB]\tmbk\tnbk");
	#endif
    printf("\n");
}

#if defined (CUOZBLAS) || defined (OZBLAS)
void print_info3 (
	testingHandle_t *th,
	ozblasHandle_t *ha
) {
//	if (!th->nodisp) {
		double t_other = ha->t_total - ha->t_SplitA - ha->t_SplitB - ha->t_comp - ha->t_sum;
		printf ("\t%1.2e\t%1.2e\t%1.2e\t%1.2e\t%1.2e\t%1.2e", ha->t_SplitA, ha->t_SplitB, ha->t_comp, ha->t_sum, t_other, ha->t_total);
		printf ("\t%1.1f\t%1.1f\t%1.1f\t%1.1e", ha->nSplitA, ha->nSplitB, ha->nSplitC, ha->memAddr*1.e-9);
		printf ("\t%d\t%d", ha->mbk, ha->nbk);
	//	printf ("\n");
//	}
}
#endif

