#if defined (AVX512)
#define VL512 8

// =========================================
// AVX horizontal add
// =========================================
static inline double hadd_avx512 (__m512d a) {
	__m256d b = _mm256_add_pd (_mm512_castpd512_pd256(a), _mm512_extractf64x4_pd (a,1));
	__m128d d = _mm_add_pd (_mm256_castpd256_pd128(b), _mm256_extractf128_pd (b,1));
	double *f = (double*)&d;
	return _mm_cvtsd_f64(d) + f[1];
}

static inline double2 hadd_x2_avx512 (__m512d aHi, __m512d aLo) {
	double SrHi[VL512], SrLo[VL512];
	_mm512_store_pd (SrHi, aHi);
	_mm512_store_pd (SrLo, aLo);
	double2 Sr;
	Sr.x = Sr.y = 0.;
	//#pragma unroll 
	for (int32_t l = 0; l < VL512; l++) {
		double2 Pr;
		Pr.x = SrHi[l];
		Pr.y = SrLo[l];
		Sr = QuadAdd (Sr, Pr);
	}
	return Sr;
}

// =========================================
// CSRMM
// =========================================
void csrmm_n_avx512 (
	const int32_t m,
	const int32_t n,
	const int32_t k,
	const double alpha,
	const double* csrValA,
	const int32_t* csrColIndA,
	const int32_t* csrRowPtrA,
	const double* B,
	const int32_t ldb,
	const double beta,
	double* C,
	const int32_t ldc
) {
	#pragma omp parallel for schedule (static) 
	for(int32_t rowid = 0; rowid < m; rowid++) {
		for (int32_t j = 0; j < n; j++) {
			__m512d Tr = _mm512_set_pd (0.,0.,0.,0.,0.,0.,0.,0.);
			int32_t i;
			for(i = csrRowPtrA[rowid]; i < csrRowPtrA[rowid+1] - (VL512-1); i += VL512) {
				__m512d Ar = _mm512_load_pd (&csrValA[i]);
				__m512d Br = _mm512_set_pd (
								B[j * ldb + csrColIndA[i+7]],
								B[j * ldb + csrColIndA[i+6]],
								B[j * ldb + csrColIndA[i+5]],
								B[j * ldb + csrColIndA[i+4]],
								B[j * ldb + csrColIndA[i+3]],
								B[j * ldb + csrColIndA[i+2]],
								B[j * ldb + csrColIndA[i+1]],
								B[j * ldb + csrColIndA[i+0]]);
				Tr = _mm512_fmadd_pd (Ar, Br, Tr);
			}
			//#pragma unroll 
			for (int32_t l = VL512-1; l > 0; l--) {
				if (i == csrRowPtrA[rowid+1] - l) {
					__m512d Ar = _mm512_load_pd (&csrValA[i]);
					double Br2[VL512] = {0.};
					for (int32_t ll = l; ll > 0; ll--) 
						Br2[VL512-ll] = B[j * ldb + csrColIndA[i+ll-1]];
					__m512d Br3 = _mm512_set_pd (Br2[0], Br2[1], Br2[2], Br2[3], Br2[4], Br2[5], Br2[6], Br2[7]);
					Tr = _mm512_fmadd_pd (Ar, Br3, Tr);
        	        i += l;
				}
            }
			if (beta == 0.)
				C[j * ldc + rowid] = alpha * hadd_avx512 (Tr);
			else
				C[j * ldc + rowid] = fma (alpha, hadd_avx512 (Tr), (beta * C[j * ldc + rowid]));
		}
	}
}

void csrmm_n_x2_avx512 (
	const int32_t m,
	const int32_t n,
	const int32_t k,
	const double alpha,
	const double* csrValA,
	const int32_t* csrColIndA,
	const int32_t* csrRowPtrA,
	const double* B,
	const int32_t ldb,
	const double beta,
	double* C,
	const int32_t ldc,
	const int32_t flag
) {
	#pragma omp parallel for schedule (static) 
	for(int32_t rowid = 0; rowid < m; rowid++) {
		for (int32_t j = 0; j < n; j++) {
			__m512d TrHi = _mm512_set_pd (0.,0.,0.,0.,0.,0.,0.,0.);
			__m512d TrLo = _mm512_set_pd (0.,0.,0.,0.,0.,0.,0.,0.);
			int32_t i;
			for(i = csrRowPtrA[rowid]; i < csrRowPtrA[rowid+1] - (VL512-1); i += VL512) {
				__m512d Ar = _mm512_load_pd (&csrValA[i]);
				__m512d Br = _mm512_set_pd (
								B[j * ldb + csrColIndA[i+7]],
								B[j * ldb + csrColIndA[i+6]],
								B[j * ldb + csrColIndA[i+5]],
								B[j * ldb + csrColIndA[i+4]],
								B[j * ldb + csrColIndA[i+3]],
								B[j * ldb + csrColIndA[i+2]],
								B[j * ldb + csrColIndA[i+1]],
								B[j * ldb + csrColIndA[i+0]]);
				Dot2_avx512 (Ar, Br, TrHi, TrLo);
			}
			//#pragma unroll 
			for (int32_t l = VL512-1; l > 0; l--) {
				if (i == csrRowPtrA[rowid+1] - l) {
					__m512d Ar = _mm512_load_pd (&csrValA[i]);
					double Br2[VL512] = {0.};
					for (int32_t ll = l; ll > 0; ll--) 
						Br2[VL512-ll] = B[j * ldb + csrColIndA[i+ll-1]];
					__m512d Br3 = _mm512_set_pd (Br2[0], Br2[1], Br2[2], Br2[3], Br2[4], Br2[5], Br2[6], Br2[7]);
					Dot2_avx512 (Ar, Br3, TrHi, TrLo);
					i += l;
				}
            }
			double2 Sr;
			if (beta == 0.)
				Sr = QuadMul (alpha, hadd_x2_avx512 (TrHi, TrLo));
			else
				Sr = QuadAdd (QuadMul (alpha, hadd_x2_avx512 (TrHi, TrLo)), QuadMul (beta, C[j * ldc + rowid]));
			C[j * ldc + rowid] = Sr.x;
			if (flag) C[j * ldc + rowid + ldc * n] = Sr.y;
		}
	}
}

// ==========================================
// L=2 ======================================
// ==========================================
#define LL 2
void csrmm_n_avx512_l2 (
	const int32_t m,
	const double alpha,
	const double* csrValA,
	const int32_t* csrColIndA,
	const int32_t* csrRowPtrA,
	const double* B,
	const int32_t ldb,
	const double beta,
	double* C,
	const int32_t ldc
) {
	#pragma omp parallel for schedule (static) 
	for(int32_t rowid = 0; rowid < m; rowid++) {
		__m512d Tr[LL];
		//#pragma unroll 
		for (int32_t j = 0; j < LL; j++) 
			Tr[j] = _mm512_set_pd (0.,0.,0.,0.,0.,0.,0.,0.);
		int32_t i;
		for(i = csrRowPtrA[rowid]; i < csrRowPtrA[rowid+1] - (VL512-1); i += VL512) {
			__m512d Ar = _mm512_load_pd (&csrValA[i]);
			__m512d Br;
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				Br = _mm512_set_pd (
								B[j * ldb + csrColIndA[i+7]],
								B[j * ldb + csrColIndA[i+6]],
								B[j * ldb + csrColIndA[i+5]],
								B[j * ldb + csrColIndA[i+4]],
								B[j * ldb + csrColIndA[i+3]],
								B[j * ldb + csrColIndA[i+2]],
								B[j * ldb + csrColIndA[i+1]],
								B[j * ldb + csrColIndA[i+0]]);
				Tr[j] = _mm512_fmadd_pd (Ar, Br, Tr[j]);
			}
		}
		//#pragma unroll 
		for (int32_t l = VL512-1; l > 0; l--) {
			if (i == csrRowPtrA[rowid+1] - l) {
				__m512d Ar = _mm512_load_pd (&csrValA[i]);
				//#pragma unroll 
				for (int32_t j = 0; j < LL; j++) {
					double Br2[VL512] = {0.};
					for (int32_t ll = l; ll > 0; ll--) 
						Br2[VL512-ll] = B[j * ldb + csrColIndA[i+ll-1]];
					__m512d Br3 = _mm512_set_pd (Br2[0], Br2[1], Br2[2], Br2[3], Br2[4], Br2[5], Br2[6], Br2[7]);
					Tr[j] = _mm512_fmadd_pd (Ar, Br3, Tr[j]);
				}
				i += l;
            }
		}
		if (beta == 0.) {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) 
				C[j * ldc + rowid] = alpha * hadd_avx512 (Tr[j]);
		} else {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) 
				C[j * ldc + rowid] = fma (alpha, hadd_avx512 (Tr[j]), (beta * C[j * ldc + rowid]));
		}
	}
}

void csrmm_n_x2_avx512_l2 (
	const int32_t m,
	const double alpha,
	const double* csrValA,
	const int32_t* csrColIndA,
	const int32_t* csrRowPtrA,
	const double* B,
	const int32_t ldb,
	const double beta,
	double* C,
	const int32_t ldc
) {
	#pragma omp parallel for schedule (static) 
	for(int32_t rowid = 0; rowid < m; rowid++) {
		__m512d TrHi[LL];
		__m512d TrLo[LL];
		//#pragma unroll 
		for (int32_t j = 0; j < LL; j++) {
			TrHi[j] = _mm512_set_pd (0.,0.,0.,0.,0.,0.,0.,0.);
			TrLo[j] = _mm512_set_pd (0.,0.,0.,0.,0.,0.,0.,0.);
		}
		int32_t i;
		for(i = csrRowPtrA[rowid]; i < csrRowPtrA[rowid+1] - (VL512-1); i += VL512) {
			__m512d Ar = _mm512_load_pd (&csrValA[i]);
			__m512d Br;
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				Br = _mm512_set_pd (
								B[j * ldb + csrColIndA[i+7]],
								B[j * ldb + csrColIndA[i+6]],
								B[j * ldb + csrColIndA[i+5]],
								B[j * ldb + csrColIndA[i+4]],
								B[j * ldb + csrColIndA[i+3]],
								B[j * ldb + csrColIndA[i+2]],
								B[j * ldb + csrColIndA[i+1]],
								B[j * ldb + csrColIndA[i+0]]);
				Dot2_avx512 (Ar, Br, TrHi[j], TrLo[j]);
			}
		}
		//#pragma unroll 
		for (int32_t l = VL512-1; l > 0; l--) {
			if (i == csrRowPtrA[rowid+1] - l) {
				__m512d Ar = _mm512_load_pd (&csrValA[i]);
				//#pragma unroll 
				for (int32_t j = 0; j < LL; j++) {
					double Br2[VL512] = {0.};
					for (int32_t ll = l; ll > 0; ll--) 
						Br2[VL512-ll] = B[j * ldb + csrColIndA[i+ll-1]];
					__m512d Br3 = _mm512_set_pd (Br2[0], Br2[1], Br2[2], Br2[3], Br2[4], Br2[5], Br2[6], Br2[7]);
					Dot2_avx512 (Ar, Br3, TrHi[j], TrLo[j]);
				}
				i += l;
            }
		}
		if (beta == 0.) {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				double2 Sr = hadd_x2_avx512 (TrHi[j], TrLo[j]);
				Sr = QuadMul (alpha, Sr);
				C[j * ldc + rowid] = Sr.x;
				C[j * ldc + rowid + ldc * LL] = Sr.y;
			}
		} else {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				double2 Sr = hadd_x2_avx512 (TrHi[j], TrLo[j]);
				Sr = QuadAdd (QuadMul (alpha, Sr), QuadMul (beta, C[j * ldc + rowid]));
				C[j * ldc + rowid] = Sr.x;
				C[j * ldc + rowid + ldc * LL] = Sr.y;
			}
		}
	}
}
#undef LL

// ==========================================
// L=3 ======================================
// ==========================================
#define LL 3
void csrmm_n_avx512_l3 (
	const int32_t m,
	const double alpha,
	const double* csrValA,
	const int32_t* csrColIndA,
	const int32_t* csrRowPtrA,
	const double* B,
	const int32_t ldb,
	const double beta,
	double* C,
	const int32_t ldc
) {
	#pragma omp parallel for schedule (static) 
	for(int32_t rowid = 0; rowid < m; rowid++) {
		__m512d Tr[LL];
		//#pragma unroll 
		for (int32_t j = 0; j < LL; j++) 
			Tr[j] = _mm512_set_pd (0.,0.,0.,0.,0.,0.,0.,0.);
		int32_t i;
		for(i = csrRowPtrA[rowid]; i < csrRowPtrA[rowid+1] - (VL512-1); i += VL512) {
			__m512d Ar = _mm512_load_pd (&csrValA[i]);
			__m512d Br;
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				Br = _mm512_set_pd (
								B[j * ldb + csrColIndA[i+7]],
								B[j * ldb + csrColIndA[i+6]],
								B[j * ldb + csrColIndA[i+5]],
								B[j * ldb + csrColIndA[i+4]],
								B[j * ldb + csrColIndA[i+3]],
								B[j * ldb + csrColIndA[i+2]],
								B[j * ldb + csrColIndA[i+1]],
								B[j * ldb + csrColIndA[i+0]]);
				Tr[j] = _mm512_fmadd_pd (Ar, Br, Tr[j]);
			}
		}
		//#pragma unroll 
		for (int32_t l = VL512-1; l > 0; l--) {
			if (i == csrRowPtrA[rowid+1] - l) {
				__m512d Ar = _mm512_load_pd (&csrValA[i]);
				//#pragma unroll 
				for (int32_t j = 0; j < LL; j++) {
					double Br2[VL512] = {0.};
					for (int32_t ll = l; ll > 0; ll--) 
						Br2[VL512-ll] = B[j * ldb + csrColIndA[i+ll-1]];
					__m512d Br3 = _mm512_set_pd (Br2[0], Br2[1], Br2[2], Br2[3], Br2[4], Br2[5], Br2[6], Br2[7]);
					Tr[j] = _mm512_fmadd_pd (Ar, Br3, Tr[j]);
				}
				i += l;
            }
		}
		if (beta == 0.) {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) 
				C[j * ldc + rowid] = alpha * hadd_avx512 (Tr[j]);
		} else {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) 
				C[j * ldc + rowid] = fma (alpha, hadd_avx512 (Tr[j]), (beta * C[j * ldc + rowid]));
		}
	}
}

void csrmm_n_x2_avx512_l3 (
	const int32_t m,
	const double alpha,
	const double* csrValA,
	const int32_t* csrColIndA,
	const int32_t* csrRowPtrA,
	const double* B,
	const int32_t ldb,
	const double beta,
	double* C,
	const int32_t ldc
) {
	#pragma omp parallel for schedule (static) 
	for(int32_t rowid = 0; rowid < m; rowid++) {
		__m512d TrHi[LL];
		__m512d TrLo[LL];
		//#pragma unroll 
		for (int32_t j = 0; j < LL; j++) {
			TrHi[j] = _mm512_set_pd (0.,0.,0.,0.,0.,0.,0.,0.);
			TrLo[j] = _mm512_set_pd (0.,0.,0.,0.,0.,0.,0.,0.);
		}
		int32_t i;
		for(i = csrRowPtrA[rowid]; i < csrRowPtrA[rowid+1] - (VL512-1); i += VL512) {
			__m512d Ar = _mm512_load_pd (&csrValA[i]);
			__m512d Br;
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				Br = _mm512_set_pd (
								B[j * ldb + csrColIndA[i+7]],
								B[j * ldb + csrColIndA[i+6]],
								B[j * ldb + csrColIndA[i+5]],
								B[j * ldb + csrColIndA[i+4]],
								B[j * ldb + csrColIndA[i+3]],
								B[j * ldb + csrColIndA[i+2]],
								B[j * ldb + csrColIndA[i+1]],
								B[j * ldb + csrColIndA[i+0]]);
				Dot2_avx512 (Ar, Br, TrHi[j], TrLo[j]);
			}
		}
		//#pragma unroll 
		for (int32_t l = VL512-1; l > 0; l--) {
			if (i == csrRowPtrA[rowid+1] - l) {
				__m512d Ar = _mm512_load_pd (&csrValA[i]);
				//#pragma unroll 
				for (int32_t j = 0; j < LL; j++) {
					double Br2[VL512] = {0.};
					for (int32_t ll = l; ll > 0; ll--) 
						Br2[VL512-ll] = B[j * ldb + csrColIndA[i+ll-1]];
					__m512d Br3 = _mm512_set_pd (Br2[0], Br2[1], Br2[2], Br2[3], Br2[4], Br2[5], Br2[6], Br2[7]);
					Dot2_avx512 (Ar, Br3, TrHi[j], TrLo[j]);
				}
				i += l;
            }
		}
		if (beta == 0.) {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				double2 Sr = hadd_x2_avx512 (TrHi[j], TrLo[j]);
				Sr = QuadMul (alpha, Sr);
				C[j * ldc + rowid] = Sr.x;
				C[j * ldc + rowid + ldc * LL] = Sr.y;
			}
		} else {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				double2 Sr = hadd_x2_avx512 (TrHi[j], TrLo[j]);
				Sr = QuadAdd (QuadMul (alpha, Sr), QuadMul (beta, C[j * ldc + rowid]));
				C[j * ldc + rowid] = Sr.x;
				C[j * ldc + rowid + ldc * LL] = Sr.y;
			}
		}
	}
}
#undef LL

// ==========================================
// L=4 ======================================
// ==========================================
#define LL 4
void csrmm_n_avx512_l4 (
	const int32_t m,
	const double alpha,
	const double* csrValA,
	const int32_t* csrColIndA,
	const int32_t* csrRowPtrA,
	const double* B,
	const int32_t ldb,
	const double beta,
	double* C,
	const int32_t ldc
) {
	#pragma omp parallel for schedule (static) 
	for(int32_t rowid = 0; rowid < m; rowid++) {
		__m512d Tr[LL];
		//#pragma unroll 
		for (int32_t j = 0; j < LL; j++) 
			Tr[j] = _mm512_set_pd (0.,0.,0.,0.,0.,0.,0.,0.);
		int32_t i;
		for(i = csrRowPtrA[rowid]; i < csrRowPtrA[rowid+1] - (VL512-1); i += VL512) {
			__m512d Ar = _mm512_load_pd (&csrValA[i]);
			__m512d Br;
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				Br = _mm512_set_pd (
								B[j * ldb + csrColIndA[i+7]],
								B[j * ldb + csrColIndA[i+6]],
								B[j * ldb + csrColIndA[i+5]],
								B[j * ldb + csrColIndA[i+4]],
								B[j * ldb + csrColIndA[i+3]],
								B[j * ldb + csrColIndA[i+2]],
								B[j * ldb + csrColIndA[i+1]],
								B[j * ldb + csrColIndA[i+0]]);
				Tr[j] = _mm512_fmadd_pd (Ar, Br, Tr[j]);
			}
		}
		//#pragma unroll 
		for (int32_t l = VL512-1; l > 0; l--) {
			if (i == csrRowPtrA[rowid+1] - l) {
				__m512d Ar = _mm512_load_pd (&csrValA[i]);
				//#pragma unroll 
				for (int32_t j = 0; j < LL; j++) {
					double Br2[VL512] = {0.};
					for (int32_t ll = l; ll > 0; ll--) 
						Br2[VL512-ll] = B[j * ldb + csrColIndA[i+ll-1]];
					__m512d Br3 = _mm512_set_pd (Br2[0], Br2[1], Br2[2], Br2[3], Br2[4], Br2[5], Br2[6], Br2[7]);
					Tr[j] = _mm512_fmadd_pd (Ar, Br3, Tr[j]);
				}
				i += l;
            }
		}
		if (beta == 0.) {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) 
				C[j * ldc + rowid] = alpha * hadd_avx512 (Tr[j]);
		} else {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) 
				C[j * ldc + rowid] = fma (alpha, hadd_avx512 (Tr[j]), (beta * C[j * ldc + rowid]));
		}
	}
}

void csrmm_n_x2_avx512_l4 (
	const int32_t m,
	const double alpha,
	const double* csrValA,
	const int32_t* csrColIndA,
	const int32_t* csrRowPtrA,
	const double* B,
	const int32_t ldb,
	const double beta,
	double* C,
	const int32_t ldc
) {
	#pragma omp parallel for schedule (static) 
	for(int32_t rowid = 0; rowid < m; rowid++) {
		__m512d TrHi[LL];
		__m512d TrLo[LL];
		//#pragma unroll 
		for (int32_t j = 0; j < LL; j++) {
			TrHi[j] = _mm512_set_pd (0.,0.,0.,0.,0.,0.,0.,0.);
			TrLo[j] = _mm512_set_pd (0.,0.,0.,0.,0.,0.,0.,0.);
		}
		int32_t i;
		for(i = csrRowPtrA[rowid]; i < csrRowPtrA[rowid+1] - (VL512-1); i += VL512) {
			__m512d Ar = _mm512_load_pd (&csrValA[i]);
			__m512d Br;
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				Br = _mm512_set_pd (
								B[j * ldb + csrColIndA[i+7]],
								B[j * ldb + csrColIndA[i+6]],
								B[j * ldb + csrColIndA[i+5]],
								B[j * ldb + csrColIndA[i+4]],
								B[j * ldb + csrColIndA[i+3]],
								B[j * ldb + csrColIndA[i+2]],
								B[j * ldb + csrColIndA[i+1]],
								B[j * ldb + csrColIndA[i+0]]);
				Dot2_avx512 (Ar, Br, TrHi[j], TrLo[j]);
			}
		}
		//#pragma unroll 
		for (int32_t l = VL512-1; l > 0; l--) {
			if (i == csrRowPtrA[rowid+1] - l) {
				__m512d Ar = _mm512_load_pd (&csrValA[i]);
				//#pragma unroll 
				for (int32_t j = 0; j < LL; j++) {
					double Br2[VL512] = {0.};
					for (int32_t ll = l; ll > 0; ll--) 
						Br2[VL512-ll] = B[j * ldb + csrColIndA[i+ll-1]];
					__m512d Br3 = _mm512_set_pd (Br2[0], Br2[1], Br2[2], Br2[3], Br2[4], Br2[5], Br2[6], Br2[7]);
					Dot2_avx512 (Ar, Br3, TrHi[j], TrLo[j]);
				}
				i += l;
            }
		}
		if (beta == 0.) {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				double2 Sr = hadd_x2_avx512 (TrHi[j], TrLo[j]);
				Sr = QuadMul (alpha, Sr);
				C[j * ldc + rowid] = Sr.x;
				C[j * ldc + rowid + ldc * LL] = Sr.y;
			}
		} else {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				double2 Sr = hadd_x2_avx512 (TrHi[j], TrLo[j]);
				Sr = QuadAdd (QuadMul (alpha, Sr), QuadMul (beta, C[j * ldc + rowid]));
				C[j * ldc + rowid] = Sr.x;
				C[j * ldc + rowid + ldc * LL] = Sr.y;
			}
		}
	}
}
#undef LL

// ==========================================
// L=5 ======================================
// ==========================================
#define LL 5
void csrmm_n_avx512_l5 (
	const int32_t m,
	const double alpha,
	const double* csrValA,
	const int32_t* csrColIndA,
	const int32_t* csrRowPtrA,
	const double* B,
	const int32_t ldb,
	const double beta,
	double* C,
	const int32_t ldc
) {
	#pragma omp parallel for schedule (static) 
	for(int32_t rowid = 0; rowid < m; rowid++) {
		__m512d Tr[LL];
		//#pragma unroll 
		for (int32_t j = 0; j < LL; j++) 
			Tr[j] = _mm512_set_pd (0.,0.,0.,0.,0.,0.,0.,0.);
		int32_t i;
		for(i = csrRowPtrA[rowid]; i < csrRowPtrA[rowid+1] - (VL512-1); i += VL512) {
			__m512d Ar = _mm512_load_pd (&csrValA[i]);
			__m512d Br;
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				Br = _mm512_set_pd (
								B[j * ldb + csrColIndA[i+7]],
								B[j * ldb + csrColIndA[i+6]],
								B[j * ldb + csrColIndA[i+5]],
								B[j * ldb + csrColIndA[i+4]],
								B[j * ldb + csrColIndA[i+3]],
								B[j * ldb + csrColIndA[i+2]],
								B[j * ldb + csrColIndA[i+1]],
								B[j * ldb + csrColIndA[i+0]]);
				Tr[j] = _mm512_fmadd_pd (Ar, Br, Tr[j]);
			}
		}
		//#pragma unroll 
		for (int32_t l = VL512-1; l > 0; l--) {
			if (i == csrRowPtrA[rowid+1] - l) {
				__m512d Ar = _mm512_load_pd (&csrValA[i]);
				//#pragma unroll 
				for (int32_t j = 0; j < LL; j++) {
					double Br2[VL512] = {0.};
					for (int32_t ll = l; ll > 0; ll--) 
						Br2[VL512-ll] = B[j * ldb + csrColIndA[i+ll-1]];
					__m512d Br3 = _mm512_set_pd (Br2[0], Br2[1], Br2[2], Br2[3], Br2[4], Br2[5], Br2[6], Br2[7]);
					Tr[j] = _mm512_fmadd_pd (Ar, Br3, Tr[j]);
				}
				i += l;
            }
		}
		if (beta == 0.) {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) 
				C[j * ldc + rowid] = alpha * hadd_avx512 (Tr[j]);
		} else {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) 
				C[j * ldc + rowid] = fma (alpha, hadd_avx512 (Tr[j]), (beta * C[j * ldc + rowid]));
		}
	}
}

void csrmm_n_x2_avx512_l5 (
	const int32_t m,
	const double alpha,
	const double* csrValA,
	const int32_t* csrColIndA,
	const int32_t* csrRowPtrA,
	const double* B,
	const int32_t ldb,
	const double beta,
	double* C,
	const int32_t ldc
) {
	#pragma omp parallel for schedule (static) 
	for(int32_t rowid = 0; rowid < m; rowid++) {
		__m512d TrHi[LL];
		__m512d TrLo[LL];
		//#pragma unroll 
		for (int32_t j = 0; j < LL; j++) {
			TrHi[j] = _mm512_set_pd (0.,0.,0.,0.,0.,0.,0.,0.);
			TrLo[j] = _mm512_set_pd (0.,0.,0.,0.,0.,0.,0.,0.);
		}
		int32_t i;
		for(i = csrRowPtrA[rowid]; i < csrRowPtrA[rowid+1] - (VL512-1); i += VL512) {
			__m512d Ar = _mm512_load_pd (&csrValA[i]);
			__m512d Br;
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				Br = _mm512_set_pd (
								B[j * ldb + csrColIndA[i+7]],
								B[j * ldb + csrColIndA[i+6]],
								B[j * ldb + csrColIndA[i+5]],
								B[j * ldb + csrColIndA[i+4]],
								B[j * ldb + csrColIndA[i+3]],
								B[j * ldb + csrColIndA[i+2]],
								B[j * ldb + csrColIndA[i+1]],
								B[j * ldb + csrColIndA[i+0]]);
				Dot2_avx512 (Ar, Br, TrHi[j], TrLo[j]);
			}
		}
		//#pragma unroll 
		for (int32_t l = VL512-1; l > 0; l--) {
			if (i == csrRowPtrA[rowid+1] - l) {
				__m512d Ar = _mm512_load_pd (&csrValA[i]);
				//#pragma unroll 
				for (int32_t j = 0; j < LL; j++) {
					double Br2[VL512] = {0.};
					for (int32_t ll = l; ll > 0; ll--) 
						Br2[VL512-ll] = B[j * ldb + csrColIndA[i+ll-1]];
					__m512d Br3 = _mm512_set_pd (Br2[0], Br2[1], Br2[2], Br2[3], Br2[4], Br2[5], Br2[6], Br2[7]);
					Dot2_avx512 (Ar, Br3, TrHi[j], TrLo[j]);
				}
				i += l;
            }
		}
		if (beta == 0.) {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				double2 Sr = hadd_x2_avx512 (TrHi[j], TrLo[j]);
				Sr = QuadMul (alpha, Sr);
				C[j * ldc + rowid] = Sr.x;
				C[j * ldc + rowid + ldc * LL] = Sr.y;
			}
		} else {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				double2 Sr = hadd_x2_avx512 (TrHi[j], TrLo[j]);
				Sr = QuadAdd (QuadMul (alpha, Sr), QuadMul (beta, C[j * ldc + rowid]));
				C[j * ldc + rowid] = Sr.x;
				C[j * ldc + rowid + ldc * LL] = Sr.y;
			}
		}
	}
}
#undef LL

#endif
