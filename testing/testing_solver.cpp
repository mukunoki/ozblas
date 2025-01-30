#include "testing_common.h"
#include "testing_common.cpp"

int32_t main (int32_t argc, char **argv) {

// library setup ------------------------------
	#if defined (CUOZBLAS)
	cusparseMatDescr_t descrA;
	cusparseCreateMatDescr (&descrA);
	cusparseSetMatType (descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase (descrA, CUSPARSE_INDEX_BASE_ZERO);
	#endif
	#if defined (CUOZBLAS) || defined (OZBLAS)
	ozblasHandle_t ha;
	ozblasCreate (&ha, WORK_MEM_SIZE);
	#endif

// testing setup ------------------------------
	testingHandle_t th;
	testingCreate (argc, argv, &th, &ha);
	print_info1 (&th);

// memory setup -------------------------------
	struct sparse_matrix_t* hst_A__ = load_sparse_matrix (MATRIX_MARKET, th.mtx_file);
	sparse_matrix_expand_symmetric_storage (hst_A__);
	int32_t errcode = sparse_matrix_convert (hst_A__, CSR);
	if (errcode != 0) {
	    fprintf (stderr, "err: conversion failed.\n");
		free (hst_A__);
		exit (1);
	}
	struct csr_matrix_t* hst_A_ = (struct csr_matrix_t*) hst_A__->repr;
	int32_t m = hst_A_->m;
	int32_t n = hst_A_->n;
	int32_t nnz = hst_A_->nnz;

	// malloc host memory
	FP_TYPE *hst_A = new FP_TYPE[nnz];
	FP_TYPE *hst_X = new FP_TYPE[n];
	FP_TYPE *hst_B = new FP_TYPE[n];
	int32_t *hst_Colind = hst_A_->colidx;
	int32_t *hst_Rowptr = hst_A_->rowptr;
	double *hst_Aptr = (double*) hst_A_->values;
	for (int32_t i = 0; i < nnz; i++) 
		hst_A[i] = (FP_TYPE) hst_Aptr[i];

	#if defined (OZBLAS) 
	char descrA[4];
	descrA[0] = 'G';
	descrA[3] = 'C';
	#endif

	ha.trueresFlag = th.trueresFlag;
	ha.verbose = th.verbose;

	// initialize
	mublasInitMat (&th, n, 1, 0, hst_X, 1., 0, 0, 0);
	mublasInitMat (&th, n, 1, 0, hst_B, 1., 0, 0, 0);

	// for verbose
	if (ha.verbose > 0) {
		if (th.maxiter % ha.verbose != 0) {
			fprintf (stderr, "maxiter mod verbose != 0\n");
			exit(1);
		}
		ha.cg_verbose1 = new FP_TYPE[(int32_t)ceil((float)th.maxiter/ha.verbose)+2];
		if (ha.trueresFlag)
			ha.cg_verbose2 = new FP_TYPE[(int32_t)ceil((float)th.maxiter/ha.verbose)+2];
		else
			ha.cg_verbose2 = new double[(int32_t)ceil((float)th.maxiter/ha.verbose)+2];
	}

// --------------------------------------------

	//print_info2 (&th);
	if (ha.verbose == 0) 
		printf ("#matrix\tm\tn\tnnz\titer\tsec\tgflops\tgbs\ttrueres\t(hex)\tSpMV/MM\tDOT/NRM\tAXPY/SCAL\tSum\tSpltVec\tSpltMat\tOther\n");

// evaluation ---------------------------------
	FP_TYPE *dev_A = hst_A;
	FP_TYPE *dev_X = hst_X;
	FP_TYPE *dev_B = hst_B;
	int32_t *dev_Rowptr = hst_Rowptr;
	int32_t *dev_Colind = hst_Colind;
	#if defined (CUOZBLAS)
	int32_t sizeType = sizeof (FP_TYPE);
	int32_t sizeInt = sizeof (int32_t);
	// malloc device memory
	cudaMalloc ((void **) &dev_A, sizeType * nnz);
	cudaMalloc ((void **) &dev_X, sizeType * n);
	cudaMalloc ((void **) &dev_B, sizeType * n);
	cudaMalloc ((void **) &dev_Colind, sizeInt * nnz);
	cudaMalloc ((void **) &dev_Rowptr, sizeInt * (m+1));
	cudaMemcpy (dev_A, hst_A, sizeType * nnz, cudaMemcpyHostToDevice);
	cudaMemcpy (dev_Colind, hst_Colind, sizeInt * nnz, cudaMemcpyHostToDevice);
	cudaMemcpy (dev_Rowptr, hst_Rowptr, sizeInt * (m+1), cudaMemcpyHostToDevice);
	cublasSetVector (n, sizeType, hst_X, 1, dev_X, 1);
	cublasSetVector (n, sizeType, hst_B, 1, dev_B, 1);
	#endif

    if (th.verbose) printf ("## ");
	printf ("%s\t%d\t%d\t%d\t", th.mtx_file, m, n, nnz);
	double t0 = gettime ();
	trgRcg (ha, th.tranA, n, nnz, descrA, dev_A, dev_Colind, dev_Rowptr, dev_B, dev_X, th.maxiter, th.tol);
	double t1 = gettime () - t0;
	
// result -----------------------------------
	// ======================================
	// computation of true residual
	// Note: computation is performed on splitModeFlag=1 (infSplit)
	ha.splitModeFlag = 1;
	FP_TYPE nrmb, trueres;
	#if defined (CUOZBLAS)
	trgRnrm2 (ha, n, dev_B, 1, &nrmb);
	#else
	nrmb = trgRnrm2 (ha, n, dev_B, 1);
	#endif
	trgRcsrmv (ha, th.tranA, n, n, nnz, -1., descrA, dev_A, dev_Colind, dev_Rowptr, dev_X, 1., dev_B); // b = b-Ax 
	#if defined (CUOZBLAS)
	trgRnrm2 (ha, n, dev_B, 1, &trueres);
	trueres = trueres / nrmb;
	#else
	trueres = trgRnrm2 (ha, n, dev_B, 1) / nrmb;
	#endif
	// ======================================

	if (ha.verbose > 0) {
		if (ha.trueresFlag) 
			printf ("\n## iter\t||r_i||/||b||\t||b-Ax||/||b||\n");
		else
			printf ("\n## iter\t||r_i||/||b||\ttime(sec)\n");
        for (int32_t i = 0; i <= (int32_t)ceil((float)ha.cg_numiter/ha.verbose); i++) {
			FP_TYPE* ptr1 = (FP_TYPE*)ha.cg_verbose1+i;
			if (ha.trueresFlag) {
				FP_TYPE* ptr2 = (FP_TYPE*)ha.cg_verbose2+i;
			    printf ("%d\t%1.3e\t%1.3e\n", i*ha.verbose, toDouble(ptr1[0]), toDouble(ptr2[0]));
			} else {
				double* ptr2 = (double*)ha.cg_verbose2+i;
			    printf ("%d\t%1.3e\t%1.3e\n", i*ha.verbose, toDouble(ptr1[0]), toDouble(ptr2[0]));
			}
		}
	}

	// below, O(n) and O(nnz) alone are considered
	// in while-loop, 1SpMV, 2DOT, 3AXPY, 1SCAL
	double gflops = 1.e-9 * (2. * nnz + 12. * n) * ha.cg_numiter / t1;
	// Flops: SpMV: 2nnz+n (note: beta==0 in while loop) + DOT: 2*2n + AXPY: 3*2n + SCAL: n
	double gbs = 1.e-9 * (12. * nnz + 140. * n) * ha.cg_numiter / t1;
	// B/s: SpMV: 8(nnz+2n)+4(nnz+n) + DOT: 2*8*2n + AXPY: 8*3*3n + SCAL: 8*2n
	
	if (ha.verbose == 0) {
    	double tloc_SpMV_SpMM_total = ha.t_SpMV_SpMM_total;
    	double tloc_DOT_NRM2_total = ha.t_DOT_NRM2_total;
    	double tloc_AXPY_SCAL_total = ha.t_AXPY_SCAL_total;
    	double tloc_Sum_total = ha.t_Sum_total;
    	double tloc_SplitVec_total = ha.t_SplitVec_total;
    	double tloc_SplitMat_total = ha.t_SplitMat_total;
    	double tloc_Other_total = t1 -tloc_SpMV_SpMM_total -tloc_DOT_NRM2_total -tloc_AXPY_SCAL_total
    								 -tloc_Sum_total -tloc_SplitVec_total -tloc_SplitMat_total;
        printf ("%d\t", ha.cg_numiter);
		printf ("%1.3e\t", t1);
		printf ("%1.3e\t", gflops);
		printf ("%1.3e\t", gbs);
		printf ("%1.3e\t(%a)\t", toDouble(trueres), toDouble(trueres));
    	printf ("%1.3e\t", tloc_SpMV_SpMM_total);
    	printf ("%1.3e\t", tloc_DOT_NRM2_total);
    	printf ("%1.3e\t", tloc_AXPY_SCAL_total);
    	printf ("%1.3e\t", tloc_Sum_total);
    	printf ("%1.3e\t", tloc_SplitVec_total);
    	printf ("%1.3e\t", tloc_SplitMat_total);
    	printf ("%1.3e\n", tloc_Other_total);
    }

	if (!th.nodisp) {
	    printf ("## total time =\t%1.3e", t1);
    	if (ha.trueresFlag) printf (" (including true residual computation)");
    	printf ("\n");
    	printf ("## total iter =\t%d\n", ha.cg_numiter);
    	printf ("## time per iter =\t%1.3e\n", t1/ha.cg_numiter);
    	printf ("## GFlops/s =\t%1.3e\n", gflops);
    	printf ("## GB/s =\t%1.3e\n", gbs);
    	char buf[128];
    	printf ("## ||b-Ax||/||b|| =\t%1.3e\n", toDouble(trueres));
    }

// --------------------------------------------

// shutdown -----------------------------------
	#if defined (CUOZBLAS)
	cudaFree (dev_A);
	cudaFree (dev_Rowptr);
	cudaFree (dev_Colind);
	cudaFree (dev_X);
	cudaFree (dev_B);
	#endif
	delete[]hst_A;
	delete[]hst_X;
	delete[]hst_B;
	if (ha.verbose > 0) {
		delete[](FP_TYPE*)ha.cg_verbose1;
		if (ha.trueresFlag)
			delete[](FP_TYPE*)ha.cg_verbose2;
		else
			delete[](double*)ha.cg_verbose2;
	}

	#if defined (CUOZBLAS) || defined (OZBLAS)
	ozblasDestroy (&ha);
	#endif
	destroy_sparse_matrix(hst_A__);
// --------------------------------------------

	return 0;
}

