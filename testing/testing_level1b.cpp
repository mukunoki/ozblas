#define ROUTINE_FLOPS(m,n,k)	((double)2.*n)
#define ROUTINE_BYTES(m,n,k,s)	((double)2.*n*s)
#include "testing_common.h"
#include "testing_common.cpp"

int32_t
main (int32_t argc, char **argv)
{
	int32_t i;
	double t0, t1, sec;
	mpreal hst_result_r;

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
	FP_TYPE hst_result_t = 0.;
	#if defined (CUDA)
	FP_TYPE *dev_X, *dev_Y;
	#endif

	// dim setup
	dim_hst_setup (&th);
	// malloc host memory
	FP_TYPE *hst_X = new FP_TYPE[th.dim_n_hst * th.incx];
	FP_TYPE *hst_Y = new FP_TYPE[th.dim_n_hst * th.incy];
	// initialize (1:val, 2:phi, 3:erange)
	if (th.trunc != 0)  // for reduced-precision performance evaluation
		printf ("### !!! Truncated inputs !!!\n");
	mublasInitMat (&th, th.dim_n_hst*th.incx, 1, 0, hst_X, th.phi, 3, th.trunc);
	mublasInitMat (&th, th.dim_n_hst*th.incy, 1, 0, hst_Y, th.phi, 3, th.trunc); 
// --------------------------------------------

	if (!th.nodisp && th.fastModeFlag) printf ("### Warning: fastmode is ignored on DOT because DOT is computed by a GEMM.\n");
	if (!th.nodisp && th.useBatchedGemmFlag) printf ("### Warning: batchedGemm is ignored on DOT because DOT is computed by a GEMM.\n");
	print_info2 (&th);

// evaluation ---------------------------------
	dim_dev_setup (&th);
	while (1) {
		if (th.dim_n_const == 0 && th.dim_n_dev > th.dim_stop) break;

		#if defined (CUDA)
		int32_t sizeType = sizeof (FP_TYPE);
		cudaMalloc ((void **) &dev_X, sizeType * th.dim_n_dev * th.incx);
		cudaMalloc ((void **) &dev_Y, sizeType * th.dim_n_dev * th.incy);
		cublasSetVector(th.dim_n_dev * th.incx, sizeType, hst_X, 1, dev_X, 1);
		cublasSetVector(th.dim_n_dev * th.incy, sizeType, hst_Y, 1, dev_Y, 1);
		#endif

		if (!th.nodisp) printf ("--\t%d\t--", th.dim_n_dev);
		get_routine_theoretial_performance (&th);
		// execution ---------------------------------
		if (th.mode == 'p') {
			for (i = 0; i < WLOOP; i++) { // warm up
				#if defined (CUBLAS) || defined (CUOZBLAS) 
				trgRdot (ha, th.dim_n_dev, dev_X, 1, dev_Y, 1, &hst_result_t);
				#else
				hst_result_t = trgRdot (ha, th.dim_n_dev, hst_X, 1, hst_Y, 1);
				#endif
			}
			#if defined (POWER_API)
			PWR_ObjAttrGetValue(obj_node, PWR_ATTR_ENERGY, &energy0_node, &tt);
			PWR_ObjAttrGetValue(obj_node_m, PWR_ATTR_MEASURED_ENERGY, &energy0_node_m, &tt);
			for (i = 0; i < NUM_CMG; i++) {
				PWR_ObjAttrGetValue(obj_cores[i], PWR_ATTR_ENERGY, &energy0_cores[i], &tt);
				PWR_ObjAttrGetValue(obj_mem[i], PWR_ATTR_ENERGY, &energy0_mem[i], &tt); 
			}
			#endif
			t0 = gettime ();
			for (i = 0; i < NLOOP; i++) {
				#if defined (CUBLAS) || defined (CUOZBLAS) 
				trgRdot (ha, th.dim_n_dev, dev_X, 1, dev_Y, 1, &hst_result_t); 
				#else
				hst_result_t = trgRdot (ha, th.dim_n_dev, hst_X, 1, hst_Y, 1);
				#endif
			}
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
			#if defined (CUBLAS) || defined (CUOZBLAS) 
			trgRdot (ha, th.dim_n_dev, dev_X, 1, dev_Y, 1, &hst_result_t); 
			#else
			hst_result_t = trgRdot (ha, th.dim_n_dev, hst_X, 1, hst_Y, 1);
			#endif
		}
		// -------------------------------------------

		#if defined (CUDA)
		cudaFree (dev_X);
		cudaFree (dev_Y);
		#endif

		if (th.mode == 'c') {
			mpreal *hst_X_r = new mpreal[th.dim_n_hst];
			mpreal *hst_Y_r = new mpreal[th.dim_n_hst];
			mublasConvMat (th.dim_n_hst*th.incx, 1, hst_X, 0, hst_X_r, 0);
			mublasConvMat (th.dim_n_hst*th.incy, 1, hst_Y, 0, hst_Y_r, 0);
			hst_result_r = refRdot (th.dim_n_dev,hst_X_r,1,hst_Y_r,1);
			mublasCheckMatrix (&th, 1, 1, &hst_result_t, 1, &hst_result_r, 1);
			delete[]hst_X_r;
			delete[]hst_Y_r;
		}

		#if defined (CUOZBLAS) || defined (OZBLAS)
		print_info3 (&th, &ha);
		#else
		printf ("\n");
		#endif

		dim_dev_increment (&th);
		if (th.dim_m_const && th.dim_n_const && th.dim_k_const) break;
	}
// --------------------------------------------

// shutdown -----------------------------------
	delete[]hst_X;
	delete[]hst_Y;

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
