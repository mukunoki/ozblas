#define ROUTINE_FLOPS(m,n,k)	((double)2.*k+2.*m+m)
#define ROUTINE_BYTES(m,n,k,s)	((double)(k+2.*m+n)*s+(k+m+1.)*4.) 
#include "testing_common.h"
#include "testing_common.cpp"

void mpfrCsrmv (
	testingHandle_t th, int32_t m, int32_t n, int32_t nnz,
	mpreal alpha, mpreal* matA, int32_t* matAind, int32_t* matAptr,
	mpreal* x, mpreal beta, mpreal* y
	) {
	#pragma omp parallel for 
	for(int32_t j = 0; j < m; j++) {
		mpreal t = 0.;
		for(int32_t i = matAptr[j]; i < matAptr[j+1]; i++) 
			t = t + matA[i] * x[matAind[i]];
		y[j] = alpha * t + beta * y[j];
	}
}

int32_t
main (int32_t argc, char **argv)
{
	int32_t i;
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

// library setup ------------------------------
	#if defined (CUDA)
	int32_t sizeType = sizeof (FP_TYPE);
	int32_t sizeInt = sizeof (int32_t);
	cusparseMatDescr_t descrA;
	cusparseCreateMatDescr (&descrA);
	cusparseSetMatType (descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase (descrA, CUSPARSE_INDEX_BASE_ZERO);
	#else
	char descrA[4];
	descrA[0] = 'G';
	descrA[3] = 'C';
	#endif
// --------------------------------------------

// testing setup ------------------------------
	testingHandle_t th;
	#if defined (CUOZBLAS) || defined (OZBLAS)
	ozblasHandle_t ha;
	ozblasCreate (&ha, WORK_MEM_SIZE);
	testingCreate (argc, argv, &th, &ha);
	#else
	testingCreate (argc, argv, &th);
	#endif
	if (th.mtx_file == NULL || !strlen(th.mtx_file)) {
	    fprintf (stderr, "error: input matrix is not specified.\n");
		exit (1);
	}
	print_info1 (&th);
// --------------------------------------------

// memory setup -------------------------------
	struct sparse_matrix_t* hst_A__ = load_sparse_matrix (MATRIX_MARKET, th.mtx_file);
	sparse_matrix_expand_symmetric_storage (hst_A__);
	int errcode = sparse_matrix_convert (hst_A__, CSR);
	if (errcode != 0) {
	    fprintf (stderr, "error: conversion failed.\n");
		free (hst_A__);
		exit (1);
	}
	struct csr_matrix_t* hst_A_ = (struct csr_matrix_t*) hst_A__->repr;
	int32_t m = hst_A_->m;
	int32_t n = hst_A_->n;
	int32_t nnz = hst_A_->nnz;

	// malloc host memory
	FP_TYPE alpha, beta;
	FP_TYPE *hst_A = new FP_TYPE[nnz];
	FP_TYPE *hst_X = new FP_TYPE[n];
	FP_TYPE *hst_Y = new FP_TYPE[m];
	FP_TYPE *hst_Y_t = new FP_TYPE[m];
	int32_t *hst_Colind = hst_A_->colidx;
	int32_t *hst_Rowptr = hst_A_->rowptr;
	double *hst_Aptr = (double*) hst_A_->values;
	for (int32_t i = 0; i < nnz; i++) 
		hst_A[i] = (FP_TYPE) hst_Aptr[i];
	// initialize (0:const, 1:drand48, 2:phi, 3:erange)
	mublasInitMat (&th, 1, 1, 0, &alpha, 1., 0, 0, 0);
	mublasInitMat (&th, 1, 1, 0, &beta, 0., 0, 0, 0);
	//mublasInitMat (&th, n, 1, 0, hst_X, 1., 1, 0, 1);
	mublasInitMat (&th, n, 1, 0, hst_X, th.phi, 3, 0, 1);
	mublasInitMat (&th, m, 1, 0, hst_Y, 0., 0, 0, 0);
// --------------------------------------------

	print_info2 (&th);

// evaluation ---------------------------------
	FP_TYPE *dev_A = hst_A;
	FP_TYPE *dev_X = hst_X;
	FP_TYPE *dev_Y = hst_Y;
	int32_t *dev_Rowptr = hst_Rowptr;
	int32_t *dev_Colind = hst_Colind;
	#if defined (CUDA)
	// malloc device memory
	cudaMalloc ((void **) &dev_A, sizeType * nnz);
	cudaMalloc ((void **) &dev_Colind, sizeInt * nnz);
	cudaMalloc ((void **) &dev_Rowptr, sizeInt * (m+1));
	cudaMalloc ((void **) &dev_X, sizeType * n);
	cudaMalloc ((void **) &dev_Y, sizeType * m);
	cudaMemcpy (dev_A, hst_A, sizeType * nnz, cudaMemcpyHostToDevice);
	cudaMemcpy (dev_Colind, hst_Colind, sizeInt * nnz, cudaMemcpyHostToDevice);
	cudaMemcpy (dev_Rowptr, hst_Rowptr, sizeInt * (m+1), cudaMemcpyHostToDevice);
	cublasSetVector (n, sizeType, hst_X, 1, dev_X, 1);
	cublasSetVector (m, sizeType, hst_Y, 1, dev_Y, 1);
	#endif

	printf ("%s\t%d\t%d\t%d", th.mtx_file, m, n, nnz);
	th.dim_n_dev = n;
	th.dim_m_dev = m;
	th.dim_k_dev = nnz;
	get_routine_theoretial_performance (&th);
	
	// execution ---------------------------------
	if (th.mode == 'p') {
		for (i = 0; i < WLOOP; i++) // warm up
			trgRcsrmv (ha, th.tranA, m, n, nnz, alpha, descrA, dev_A, dev_Colind, dev_Rowptr, dev_X, beta, dev_Y);
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
			trgRcsrmv (ha, th.tranA, m, n, nnz, alpha, descrA, dev_A, dev_Colind, dev_Rowptr, dev_X, beta, dev_Y);
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
		trgRcsrmv (ha, th.tranA, m, n, nnz, alpha, descrA, dev_A, dev_Colind, dev_Rowptr, dev_X, beta, dev_Y);
		#if defined (CUDA)
		cublasGetVector (m, sizeType, dev_Y, 1, hst_Y_t, 1);
		#else
		mublasCopyMat (m, 1, dev_Y, 0, hst_Y_t, 0);
		#endif
	}
	// -------------------------------------------

	#if defined (CUDA)
	cudaFree (dev_A);
	cudaFree (dev_Colind);
	cudaFree (dev_Rowptr);
	cudaFree (dev_X);
	cudaFree (dev_Y);
	#endif
	if (th.mode == 'c') {
		mpreal alpha_r = 0;
		mpreal beta_r = 0;
		mpreal *hst_A_r = new mpreal[nnz];
		mpreal *hst_X_r = new mpreal[n];
		mpreal *hst_Y_r = new mpreal[m];
		mublasConvMat (1, 1, &alpha, 0, &alpha_r, 0);
		mublasConvMat (1, 1, &beta, 0, &beta_r, 0);
		mublasConvMat (nnz, 1, hst_A, 0, hst_A_r, 0);
		mublasConvMat (n, 1, hst_X, 0, hst_X_r, 0);
		mublasConvMat (m, 1, hst_Y, 0, hst_Y_r, 0);
		mpfrCsrmv (th, m, n, nnz, alpha_r, hst_A_r, hst_Colind, hst_Rowptr, hst_X_r, beta_r, hst_Y_r);
		mublasCheckMatrix (&th, m, 1, hst_Y_t, m, hst_Y_r, m);
		delete[]hst_A_r;
		delete[]hst_X_r;
		delete[]hst_Y_r;
	}
// --------------------------------------------

// shutdown -----------------------------------
	delete[]hst_A;
	delete[]hst_X;
	delete[]hst_Y;
	delete[]hst_Y_t;

	#if defined (CUOZBLAS) || defined (OZBLAS)
	ozblasDestroy (&ha);
	print_info3 (&th, &ha);
	#endif
	printf ("\n");
// --------------------------------------------
	destroy_sparse_matrix(hst_A__);

	#if defined (POWER_API)
	PWR_CntxtDestroy(cntxt);
	PWR_CntxtDestroy(cntxt_m);
	#endif

	return 0;
}

