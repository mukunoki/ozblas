#include "cuozblas_common.h"

void cucounterInit (cuozblasHandle_t *oh) {
	if (oh->initialized == 0) {
		fprintf (stderr, "OzBLAS error: OzBLAS is not initialized (call cuozblasCreate).\n");
		exit(1);
	}
	oh->splitShift = 1; 
	oh->t_SplitA = 0.;
	oh->t_SplitB = 0.;
	oh->t_comp = 0.;
	oh->t_sum = 0.;
	oh->t_total = 0.;
	oh->nSplitA = 0;
	oh->nSplitB = 0;
	oh->nSplitC = 0;
	oh->mbk = 0;
	oh->nbk = 0;
	oh->memAddr = oh->memMaskSplitA;
}

void cuozblasCreate (cuozblasHandle_t *oh, uint64_t WorkSizeBytes) {
	cublasCreate (&oh->ch);
	cublasSetPointerMode(oh->ch, CUBLAS_POINTER_MODE_HOST);
	cusparseCreate (&oh->csh);
	cusparseSetPointerMode(oh->csh, CUSPARSE_POINTER_MODE_HOST);

	oh->workSizeBytes = WorkSizeBytes;

	// default
	oh->nSplitMax = 0;
	// for CG
	oh->memMaskSplitA = 0; // disable pre-split of matA 
	oh->splitShift = 1; // default 

	// Flag
	oh->splitModeFlag = 0;
	oh->fastModeFlag = 0;
	oh->reproModeFlag = 1;
	oh->sumModeFlag = 0;
	oh->useBatchedGemmFlag = 1;
	oh->splitEpsModeFlag = 0;

	// work memory allocation
	cudaMalloc ((void **) &oh->devWorkCommon, sizeof(double)*512);// for DOT and NRM2, and vecSplit, '512' = SPLIT_VEC_NTX in cuozblas_split.cu
	cudaMalloc ((void **) &oh->devWork, oh->workSizeBytes);
	if (oh->devWork == NULL) {
		fprintf (OUTPUT, "OzBLAS error: work memory allocation error (%1.3e Bytes requested).\n", (double)oh->workSizeBytes);
		exit (1);
	}

	// batched BLAS
	if (oh->useBatchedGemmFlag) {
		char *hstBatchAddr_;
		cudaMallocHost ((void **) &hstBatchAddr_, sizeof(double*) * NumSplitDefaultMax * NumSplitDefaultMax * 3);
		oh->hstBatchAddr = hstBatchAddr_;
		char *devBatchAddr_;
		cudaMalloc ((void **) &devBatchAddr_, sizeof(double*) * NumSplitDefaultMax * NumSplitDefaultMax * 3);
		oh->devBatchAddr = devBatchAddr_;
	}

	oh->initialized = 1;
}

void cuozblasDestroy (cuozblasHandle_t *oh) {
	cudaFree (oh->devWork);
	cudaFree (oh->devWorkCommon);
	if (oh->useBatchedGemmFlag) {
		cudaFreeHost (oh->hstBatchAddr);
		cudaFree (oh->devBatchAddr);
	}
	cublasDestroy (oh->ch);
	cusparseDestroy (oh->csh);
}
