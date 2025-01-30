#define ROUTINE_FLOPS(m,n,k)	((double)2.*m*n+3.*m)
#define ROUTINE_BYTES(m,n,k,s)	(((double)2.*m+n+m*n)*s) 
#include "testing_common.h"
#include "testing_common.cpp"

int32_t
main (int32_t argc, char **argv)
{
	int32_t i;
	int32_t lda_hst;
	int32_t cda_dev, cda_hst;
	int32_t rda_dev, rda_hst;
	int32_t vlx_dev, vlx_hst;
	int32_t vly_dev, vly_hst;
	double t0, t1, sec;

	#if defined (POWER_API)
	// only on Fugaku
	PWR_Cntxt cntxt = NULL;
	PWR_Cntxt cntxt_m = NULL;
	PWR_Obj obj_node = NULL;
	PWR_Obj obj_node_m = NULL; // measured
	PWR_Obj obj_cores[4] = {NULL, NULL, NULL, NULL};
	PWR_Obj obj_mem[4] = {NULL, NULL, NULL, NULL};
	double energy0_node = 0;
	double energy1_node = 0;
	double energy0_node_m = 0; // measured
	double energy1_node_m = 0; // measured
	double energy0_cores[4] = {0};
	double energy1_cores[4] = {0};
	double energy0_mem[4] = {0};
	double energy1_mem[4] = {0};
	PWR_Time tt;

	PWR_CntxtInit (PWR_CNTXT_DEFAULT, PWR_ROLE_APP, "app", &cntxt); // initialization
	PWR_CntxtInit (PWR_CNTXT_FX1000, PWR_ROLE_APP, "app", &cntxt_m); // initialization
	PWR_CntxtGetObjByName (cntxt, "plat.node", &obj_node); // get object
	PWR_CntxtGetObjByName (cntxt_m, "plat.node", &obj_node_m); // get object
	PWR_CntxtGetObjByName (cntxt, "plat.node.cpu.cmg0.cores", &obj_cores[0]); // get object
	PWR_CntxtGetObjByName (cntxt, "plat.node.cpu.cmg1.cores", &obj_cores[1]); // get object
	PWR_CntxtGetObjByName (cntxt, "plat.node.cpu.cmg2.cores", &obj_cores[2]); // get object
	PWR_CntxtGetObjByName (cntxt, "plat.node.cpu.cmg3.cores", &obj_cores[3]); // get object
	PWR_CntxtGetObjByName (cntxt, "plat.node.mem0", &obj_mem[0]); // get object
	PWR_CntxtGetObjByName (cntxt, "plat.node.mem1", &obj_mem[1]); // get object
	PWR_CntxtGetObjByName (cntxt, "plat.node.mem2", &obj_mem[2]); // get object
	PWR_CntxtGetObjByName (cntxt, "plat.node.mem3", &obj_mem[3]); // get object
	#endif

	#if defined (MPLAPACK)
    mpfr_set_default_prec(MPFR_PREC);
//	mpreal::set_default_prec(MPFR_PREC);
	#endif

// testing setup ------------------------------
	testingHandle_t th;
	#if defined (CUOZBLAS) || defined (OZBLAS)
	ozblasHandle_t ha;
	ozblasCreate (&ha, WORK_MEM_SIZE); 
	testingCreate (argc, argv, &th, &ha);
	#elif defined (CUBLAS)
	cublasHandle_t ha;
	cublasCreate (&ha); 
	cublasSetPointerMode(ha, CUBLAS_POINTER_MODE_HOST);
	testingCreate (argc, argv, &th);
	#else
	testingCreate (argc, argv, &th);
	#endif
	print_info1 (&th);
// --------------------------------------------

// memory setup -------------------------------
	dim_hst_setup (&th);
	rda_hst = th.dim_stop;
	cda_hst = th.dim_stop;
	vlx_hst = th.dim_stop;
	vly_hst = th.dim_stop;
	if (th.tranA == 'N' || th.tranA == 'n') {
		if (th.dim_n_const > 0) vlx_hst = th.dim_n_const;
		if (th.dim_m_const > 0) vly_hst = th.dim_m_const;
	} else {
		if (th.dim_m_const > 0) vlx_hst = th.dim_m_const;
		if (th.dim_n_const > 0) vly_hst = th.dim_n_const;
	}
	if (th.dim_m_const > 0) rda_hst = th.dim_m_const;
	if (th.dim_n_const > 0) cda_hst = th.dim_n_const;
	lda_hst = rda_hst;
	// malloc host memory
	FP_TYPE alpha, beta;
	FP_TYPE *hst_A = new FP_TYPE[lda_hst * cda_hst];
	FP_TYPE *hst_X = new FP_TYPE[vlx_hst * th.incx];
	FP_TYPE *hst_Y = new FP_TYPE[vly_hst * th.incy];
	FP_TYPE *hst_Y_t = new FP_TYPE[vly_hst * th.incy];
	// initialize (0:const, 1:drand48, 2:phi, 3:erange)
	mublasInitMat (&th, 1, 1, 1, &alpha, 1., 0, 0, 0);
	mublasInitMat (&th, 1, 1, 1, &beta, 0., 0, 0, 0);
	if (th.trunc != 0)  // for reduced-precision performance evaluation
		printf ("### !!! Truncated inputs !!!\n");
	mublasInitMat (&th, rda_hst, cda_hst, lda_hst, hst_A, th.phi, 3, th.trunc, 1);
	mublasInitMat (&th, vlx_hst * th.incx, 1, 0, hst_X, th.phi, 3, th.trunc, 2);
	mublasInitMat (&th, vly_hst * th.incy, 1, 0, hst_Y, 0., 0, 0, 0);

// --------------------------------------------

	if (!th.nodisp && th.fastModeFlag && th.dim_start) 
		printf ("### Warning: when m=n=1, it is computed as DOT, which ignores fastmode and batchedGemm.\n");
	print_info2 (&th);

// evaluation ---------------------------------
	dim_dev_setup (&th);
	while (1) {
		if ((th.dim_m_const == 0 && th.dim_m_dev > th.dim_stop) ||
		    (th.dim_n_const == 0 && th.dim_n_dev > th.dim_stop)) break;
		if (th.dim_m_dev == 0 || th.dim_n_dev == 0) {
		} else {

		// dim setup
		if (th.tranA == 'N' || th.tranA == 'n') {
			vlx_dev = th.dim_n_dev;
			vly_dev = th.dim_m_dev;
		} else {
			vlx_dev = th.dim_m_dev;
			vly_dev = th.dim_n_dev;
		}
		rda_dev = th.dim_m_dev;
		cda_dev = th.dim_n_dev;

		FP_TYPE *dev_A = hst_A;
		FP_TYPE *dev_X = hst_X;
		FP_TYPE *dev_Y = hst_Y;
		int32_t lda_dev = lda_hst;
		#if defined (CUDA)
		int32_t sizeType = sizeof (FP_TYPE);
		// malloc device memory
		size_t pitch;
		cudaMallocPitch ((void **) &dev_A, &pitch, sizeType * rda_dev, cda_dev);
		lda_dev = pitch/sizeType;
		cudaMalloc ((void **) &dev_X, sizeType * vlx_dev * th.incx);
		cudaMalloc ((void **) &dev_Y, sizeType * vly_dev * th.incy);
		// memcpy from hst to device
		cublasSetMatrix (rda_dev, cda_dev, sizeType, hst_A, lda_hst, dev_A, lda_dev);
		cublasSetVector (vlx_dev * th.incx, sizeType, hst_X, 1, dev_X, 1);
		// ---------------------------------------------
		#endif

		printf ("%d\t%d\t--", th.dim_m_dev, th.dim_n_dev);
		get_routine_theoretial_performance (&th);

		// execution ---------------------------------
		if (th.mode == 'p') {
			for (i = 0; i < WLOOP; i++) // warm up
				trgRgemv (ha, th.tranA, rda_dev, cda_dev, alpha, dev_A, lda_dev, dev_X, 1, beta, dev_Y, 1);
			#if defined (POWER_API)
			PWR_ObjAttrGetValue(obj_node, PWR_ATTR_ENERGY, &energy0_node, &tt);
			PWR_ObjAttrGetValue(obj_node_m, PWR_ATTR_MEASURED_ENERGY, &energy0_node_m, &tt);
			for (i = 0; i < NUM_CMG; i++) {
				PWR_ObjAttrGetValue(obj_cores[i], PWR_ATTR_ENERGY, &energy0_cores[i], &tt);
				PWR_ObjAttrGetValue(obj_mem[i], PWR_ATTR_ENERGY, &energy0_mem[i], &tt); 
			}
			#endif
			t0 = gettime ();
			for (i = 0; i < NLOOP; i++) 
				trgRgemv (ha, th.tranA, rda_dev, cda_dev, alpha, dev_A, lda_dev, dev_X, 1, beta, dev_Y, 1);
			t1 = gettime ();
			sec = (t1 - t0) / NLOOP;
			printf ("\t%1.3e\t%1.3e\t%1.3e\t%d", sec, (th.routine_flops / sec) * 1.0e-9, (th.routine_bytes / sec) * 1.0e-9, NLOOP);
			#if defined (POWER_API)
			PWR_ObjAttrGetValue(obj_node, PWR_ATTR_ENERGY, &energy1_node, &tt); 
			PWR_ObjAttrGetValue(obj_node_m, PWR_ATTR_MEASURED_ENERGY, &energy1_node_m, &tt); 
			double nsec = sec * NLOOP;
			double energy_node = energy1_node - energy0_node;
			double energy_node_m = energy1_node_m - energy0_node_m;
			double energy_cores_all = 0.;
			double energy_mem_all = 0.;
			double avg_power_node = energy_node / nsec;
			double avg_power_node_m = energy_node_m / nsec;
			double avg_power_cores = 0.;
			double avg_power_mem = 0.;
			for (i = 0; i < NUM_CMG; i++) {
				PWR_ObjAttrGetValue(obj_cores[i], PWR_ATTR_ENERGY, &energy1_cores[i], &tt); 
				PWR_ObjAttrGetValue(obj_mem[i], PWR_ATTR_ENERGY, &energy1_mem[i], &tt); 
				double energy_cores = energy1_cores[i] - energy0_cores[i];
				energy_cores_all += energy_cores;
				avg_power_cores += energy_cores / nsec;
				double energy_mem = energy1_mem[i] - energy0_mem[i];
				energy_mem_all += energy_mem;
				avg_power_mem += energy_mem / nsec;
			}
			avg_power_cores /= NUM_CMG;
			avg_power_mem /= NUM_CMG;
			printf ("\t%1.3e\t%1.3e", avg_power_node, energy_node);
			printf ("\t%1.3e\t%1.3e", avg_power_node_m, energy_node_m);
			printf ("\t%1.3e\t%1.3e", avg_power_cores, energy_cores_all);
			printf ("\t%1.3e\t%1.3e", avg_power_mem, energy_mem_all);
			#endif
		}
		if (th.mode == 'c') {
			trgRgemv (ha, th.tranA, rda_dev, cda_dev, alpha, dev_A, lda_dev, dev_X, 1, beta, dev_Y, 1);
			#if defined (CUDA)
			cublasGetVector (vly_dev * th.incy, sizeType, dev_Y, 1, hst_Y_t, 1);
			#else
			mublasCopyMat (vly_dev * th.incy, 1, dev_Y, 0, hst_Y_t, 0);
			#endif
		}
		// -------------------------------------------

		#if defined (CUDA)
		cudaFree (dev_A);
		cudaFree (dev_X);
		cudaFree (dev_Y);
		#endif
		if (th.mode == 'c') {
			mpreal alpha_r = 0;
			mpreal beta_r = 0;
			mpreal *hst_A_r = new mpreal[rda_dev * cda_dev];
			mpreal *hst_X_r = new mpreal[vlx_dev * th.incx];
			mpreal *hst_Y_r = new mpreal[vly_dev * th.incy];
			mublasConvMat (1, 1, &alpha, 0, &alpha_r, 0);
			mublasConvMat (1, 1, &beta, 0, &beta_r, 0);
			mublasConvMat (rda_dev, cda_dev, hst_A, lda_hst, hst_A_r, rda_dev);
			mublasConvMat (vlx_dev * th.incx, 1, hst_X, 0, hst_X_r, 0);
			mublasConvMat (vly_dev * th.incy, 1, hst_Y, 0, hst_Y_r, 0);
			refRgemv (th.tranA,rda_dev,cda_dev,alpha_r,hst_A_r,rda_dev,hst_X_r,1,beta_r,hst_Y_r,1);
			mublasCheckMatrix (&th, vly_dev * th.incy, 1, hst_Y_t, vly_dev * th.incy, hst_Y_r, vly_dev * th.incy);
			delete[]hst_A_r;
			delete[]hst_X_r;
			delete[]hst_Y_r;
		}

		#if defined (CUOZBLAS) || defined (OZBLAS)
		print_info3 (&th, &ha);
		#endif

		}
		dim_dev_increment (&th);
		printf ("\n");
		if (th.dim_m_const && th.dim_n_const) break;
	}
// --------------------------------------------
	delete[]hst_A;
	delete[]hst_X;
	delete[]hst_Y;
	delete[]hst_Y_t;

	#if defined (CUOZBLAS) || defined (OZBLAS)
	ozblasDestroy (&ha);
	#elif defined (CUBLAS)
	cublasDestroy (ha);
	#endif
// --------------------------------------------

	#if defined (POWER_API)
	PWR_CntxtDestroy(cntxt);
	PWR_CntxtDestroy(cntxt_m);
	#endif

	return 0;
}
