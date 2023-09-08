#include "ozblas_common.h"

void counterInit (ozblasHandle_t *oh) {
	if (oh->initialized == 0) {
		fprintf (stderr, "OzBLAS error: OzBLAS is not initialized (call ozblasCreate).\n");
		exit(1);
	}
	oh->splitShift = 1; 
	oh->t_SplitA = 0.;
	oh->t_SplitB = 0.;
	oh->t_comp = 0.;
	oh->t_sum = 0.;
	oh->t_total = 0.;
	oh->n_comp = 0.;
	oh->nSplitA = 0;
	oh->nSplitB = 0;
	oh->nSplitC = 0;
	oh->mbk = 0;
	oh->nbk = 0;
	oh->memAddr = oh->memMaskSplitA;
}

void ozblasCreate (ozblasHandle_t *oh, uint64_t WorkSizeBytes) {
	#if defined (CUBLAS)
	cublasCreate (&oh->ch);
	cublasSetPointerMode(oh->ch, CUBLAS_POINTER_MODE_HOST);
	#endif

	oh->workSizeBytes = WorkSizeBytes;

	// default
	oh->nSplitMax = 0;
	// for CG
	oh->memMaskSplitA = 0; // disable pre-split of matA 
	oh->splitShift = 1; // default (no-Splitshift)

	// Flag
	oh->splitModeFlag = 0;
	oh->fastModeFlag = 0;
	oh->reproModeFlag = 1;
	oh->sumModeFlag = 0;
	oh->useBatchedGemmFlag = 0;
	oh->splitEpsModeFlag = 0;
	oh->precxFlag = 0;

	#if defined (MKL)
	oh->useBatchedGemmFlag = 1;
	#else
	oh->useBatchedGemmFlag = 0; // currently OpenBLAS does not support it
	#endif

	// work memory allocation
	char *devWork_ = (char*) malloc (oh->workSizeBytes);
	if (devWork_ == NULL) {
		fprintf (OUTPUT, "OzBLAS error: work memory allocation error (%1.3e Bytes requested).\n", (double)oh->workSizeBytes);
		exit (1);
	}
	oh->devWork = devWork_;

	oh->initialized = 1;
}

void ozblasDestroy (ozblasHandle_t *oh) {
	#if defined (CUBLAS)
	cublasDestroy (oh->ch);
	#endif

	oh->initialized = 0;

	free (oh->devWork);
}

