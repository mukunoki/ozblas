#if defined (AVX256)
#define VL256 4

// =========================================
// AVX horizontal add
// =========================================
static inline double hadd_avx256 (__m256d a) {
	__m256d b = _mm256_hadd_pd (a, _mm256_permute2f128_pd(a, a, 1));
	b = _mm256_hadd_pd (b, b);
	return _mm_cvtsd_f64 (_mm256_castpd256_pd128(b));
}

static inline double2 hadd_x2_avx256 (__m256d vHi, __m256d vLo) {
	__m128d aHi = _mm256_extractf128_pd (vHi, 1);
	__m128d bHi = _mm256_castpd256_pd128 (vHi);
	__m128d aLo = _mm256_extractf128_pd (vLo, 1);
	__m128d bLo = _mm256_castpd256_pd128 (vLo);
	__m128d tHi, tLo;
	QuadAdd_avx128 (aHi, aLo, bHi, bLo, tHi, tLo); 
	double sHi[2], sLo[2];
	_mm_store_pd (sHi, tHi);
	_mm_store_pd (sLo, tLo);
	double2 c;
	QuadAdd_ (sHi[0], sLo[0], sHi[1], sLo[1], c.x, c.y);
	return c;
}

// =========================================
// DGEMM_TN_SKINNY
// =========================================
void dgemm_tn_skinny_avx256 (
	const int32_t m,
	const int32_t n,
	const int32_t k,
	const double alpha,
	const double* A,
	const int32_t lda,
	const double* B,
	const int32_t ldb,
	const double beta,
	double* C,
	const int32_t ldc
	) {
	for (int32_t nn = 0; nn < n; nn++) {
		for (int32_t mm = 0; mm < m; mm++) {
			__m256d Gr = _mm256_set_pd (0.,0.,0.,0.);
			double Tr = 0.;
			int32_t kb = (int32_t)floor((float)k/VL256);
			#pragma omp parallel
			{
				__m256d Lr = _mm256_set_pd (0.,0.,0.,0.);
				#pragma omp for
				for (int32_t kk = 0; kk < kb; kk++) {
					__m256d Ar = _mm256_load_pd (&A[mm * lda + kk * VL256]);
					__m256d Br = _mm256_load_pd (&B[nn * ldb + kk * VL256]);
					Lr = _mm256_fmadd_pd (Ar, Br, Lr);
				}
				#pragma omp critical
				{
					Gr = _mm256_add_pd (Lr, Gr);
				}
			}
			for (int32_t kk = kb * VL256; kk < k; kk++) {
				double Ar = A[mm * lda + kk];
				double Br = B[nn * ldb + kk];
				Tr = fma (Ar, Br, Tr);
			}
			if (beta == 0.)
				C[nn * ldc + mm] = alpha * (hadd_avx256 (Gr) + Tr);
			else 
				C[nn * ldc + mm] = fma (alpha, (hadd_avx256 (Gr) + Tr), (beta * C[nn * ldc + mm]));
		}
	}
}

void dgemm_tn_skinny_x2_avx256 (
	const int32_t m,
	const int32_t n,
	const int32_t k,
	const double alpha,
	const double* A,
	const int32_t lda,
	const double* B,
	const int32_t ldb,
	const double beta,
	double* C,
	const int32_t ldc,
	const int32_t flag
	) {
	for (int32_t nn = 0; nn < n; nn++) {
		for (int32_t mm = 0; mm < m; mm++) {
			__m256d GrHi = _mm256_set_pd (0.,0.,0.,0.);
			__m256d GrLo = _mm256_set_pd (0.,0.,0.,0.);
			double2 Tr;
			Tr.x = Tr.y = 0.;
			int32_t kb = (int32_t)floor((float)k/VL256);
			#pragma omp parallel
			{
				__m256d LrHi = _mm256_set_pd (0.,0.,0.,0.);
				__m256d LrLo = _mm256_set_pd (0.,0.,0.,0.);
				#pragma omp for
				for (int32_t kk = 0; kk < kb; kk++) {
					__m256d Ar = _mm256_load_pd (&A[mm * lda + kk * VL256]);
					__m256d Br = _mm256_load_pd (&B[nn * ldb + kk * VL256]);
					Dot2_avx256 (Ar, Br, LrHi, LrLo);
				}
				#pragma omp critical
				{
					QuadAdd_avx256 (GrHi, GrLo, LrHi, LrLo, GrHi, GrLo);
				}
			}
			for (int32_t kk = kb * VL256; kk < k; kk++) {
				double Ar = A[mm * lda + kk];
				double Br = B[nn * ldb + kk];
				Dot2i (Ar, Br, Tr);
			}
			double2 Gr = QuadAdd (hadd_x2_avx256 (GrHi, GrLo), Tr);
			if (beta == 0.)
				Gr = QuadMul (alpha, Gr);
			else
				Gr = QuadAdd (QuadMul (alpha, Gr), QuadMul (beta, C[nn * ldc + mm]));
			C[nn * ldc + mm] = Gr.x;
			if (flag) C[nn * ldc + mm + n * ldc] = Gr.y;
		}
	}
}

// =========================================
// CSRMM
// =========================================
void csrmm_n_avx256 (
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
			__m256d Tr = _mm256_set_pd (0.,0.,0.,0.);
			int32_t i;
			for(i = csrRowPtrA[rowid]; i < csrRowPtrA[rowid+1] - (VL256-1); i += VL256) {
				__m256d Ar = _mm256_load_pd (&csrValA[i]);
				__m256d Br = _mm256_set_pd (
								B[j * ldb + csrColIndA[i+3]],
								B[j * ldb + csrColIndA[i+2]],
								B[j * ldb + csrColIndA[i+1]],
								B[j * ldb + csrColIndA[i+0]]);
				Tr = _mm256_fmadd_pd (Ar, Br, Tr);
			}
			//#pragma unroll 
			for (int32_t l = VL256-1; l > 0; l--) {
				if (i == csrRowPtrA[rowid+1] - l) {
					__m256d Ar = _mm256_load_pd (&csrValA[i]);
					double Br2[VL256] = {0.};
					for (int32_t ll = l; ll > 0; ll--) 
						Br2[VL256-ll] = B[j * ldb + csrColIndA[i+ll-1]];
					__m256d Br3 = _mm256_set_pd (Br2[0], Br2[1], Br2[2], Br2[3]);
					Tr = _mm256_fmadd_pd (Ar, Br3, Tr);
        	        i += l;
				}
            }
			if (beta == 0.)
				C[j * ldc + rowid] = alpha * hadd_avx256 (Tr);
			else
				C[j * ldc + rowid] = fma (alpha, hadd_avx256 (Tr), (beta * C[j * ldc + rowid]));
		}
	}
}

void csrmm_n_x2_avx256 (
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
			__m256d TrHi = _mm256_set_pd (0.,0.,0.,0.);
			__m256d TrLo = _mm256_set_pd (0.,0.,0.,0.);
			int32_t i;
			for(i = csrRowPtrA[rowid]; i < csrRowPtrA[rowid+1] - (VL256-1); i += VL256) {
				__m256d Ar = _mm256_load_pd (&csrValA[i]);
				__m256d Br = _mm256_set_pd (
								B[j * ldb + csrColIndA[i+3]],
								B[j * ldb + csrColIndA[i+2]],
								B[j * ldb + csrColIndA[i+1]],
								B[j * ldb + csrColIndA[i+0]]);
				Dot2_avx256 (Ar, Br, TrHi, TrLo);
			}
			//#pragma unroll 
			for (int32_t l = VL256-1; l > 0; l--) {
				if (i == csrRowPtrA[rowid+1] - l) {
					__m256d Ar = _mm256_load_pd (&csrValA[i]);
					double Br2[VL256] = {0.};
					for (int32_t ll = l; ll > 0; ll--) 
						Br2[VL256-ll] = B[j * ldb + csrColIndA[i+ll-1]];
					__m256d Br3 = _mm256_set_pd (Br2[0], Br2[1], Br2[2], Br2[3]);
					Dot2_avx256 (Ar, Br3, TrHi, TrLo);
					i += l;
				}
            }
			double2 Sr;
			if (beta == 0.)
				Sr = QuadMul (alpha, hadd_x2_avx256 (TrHi, TrLo));
			else
				Sr = QuadAdd (QuadMul (alpha, hadd_x2_avx256 (TrHi, TrLo)), QuadMul (beta, C[j * ldc + rowid]));
			C[j * ldc + rowid] = Sr.x;
			if (flag) C[j * ldc + rowid + ldc * n] = Sr.y;
		}
	}
}

// ==========================================
// L=2 ======================================
// ==========================================
#define LL 2
void csrmm_n_avx256_l2 (
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
		__m256d Tr[LL];
		//#pragma unroll 
		for (int32_t j = 0; j < LL; j++) 
			Tr[j] = _mm256_set_pd (0.,0.,0.,0.);
		int32_t i;
		for(i = csrRowPtrA[rowid]; i < csrRowPtrA[rowid+1] - (VL256-1); i += VL256) {
			__m256d Ar = _mm256_load_pd (&csrValA[i]);
			__m256d Br;
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				Br = _mm256_set_pd (
								B[j * ldb + csrColIndA[i+3]],
								B[j * ldb + csrColIndA[i+2]],
								B[j * ldb + csrColIndA[i+1]],
								B[j * ldb + csrColIndA[i+0]]);
				Tr[j] = _mm256_fmadd_pd (Ar, Br, Tr[j]);
			}
		}
		//#pragma unroll 
		for (int32_t l = VL256-1; l > 0; l--) {
			if (i == csrRowPtrA[rowid+1] - l) {
				__m256d Ar = _mm256_load_pd (&csrValA[i]);
				//#pragma unroll 
				for (int32_t j = 0; j < LL; j++) {
					double Br2[VL256] = {0.};
					for (int32_t ll = l; ll > 0; ll--) 
						Br2[VL256-ll] = B[j * ldb + csrColIndA[i+ll-1]];
					__m256d Br3 = _mm256_set_pd (Br2[0], Br2[1], Br2[2], Br2[3]);
					Tr[j] = _mm256_fmadd_pd (Ar, Br3, Tr[j]);
				}
				i += l;
            }
		}
		if (beta == 0.) {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) 
				C[j * ldc + rowid] = alpha * hadd_avx256 (Tr[j]);
		} else {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) 
				C[j * ldc + rowid] = fma (alpha, hadd_avx256 (Tr[j]), (beta * C[j * ldc + rowid]));
		}
	}
}

void csrmm_n_x2_avx256_l2 (
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
		__m256d TrHi[LL];
		__m256d TrLo[LL];
		//#pragma unroll 
		for (int32_t j = 0; j < LL; j++) {
			TrHi[j] = _mm256_set_pd (0.,0.,0.,0.);
			TrLo[j] = _mm256_set_pd (0.,0.,0.,0.);
		}
		int32_t i;
		for(i = csrRowPtrA[rowid]; i < csrRowPtrA[rowid+1] - (VL256-1); i += VL256) {
			__m256d Ar = _mm256_load_pd (&csrValA[i]);
			__m256d Br;
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				Br = _mm256_set_pd (
								B[j * ldb + csrColIndA[i+3]],
								B[j * ldb + csrColIndA[i+2]],
								B[j * ldb + csrColIndA[i+1]],
								B[j * ldb + csrColIndA[i+0]]);
				Dot2_avx256 (Ar, Br, TrHi[j], TrLo[j]);
			}
		}
		//#pragma unroll 
		for (int32_t l = VL256-1; l > 0; l--) {
			if (i == csrRowPtrA[rowid+1] - l) {
				__m256d Ar = _mm256_load_pd (&csrValA[i]);
				//#pragma unroll 
				for (int32_t j = 0; j < LL; j++) {
					double Br2[VL256] = {0.};
					for (int32_t ll = l; ll > 0; ll--) 
						Br2[VL256-ll] = B[j * ldb + csrColIndA[i+ll-1]];
					__m256d Br3 = _mm256_set_pd (Br2[0], Br2[1], Br2[2], Br2[3]);
					Dot2_avx256 (Ar, Br3, TrHi[j], TrLo[j]);
				}
				i += l;
            }
		}
		if (beta == 0.) {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				double2 Sr = hadd_x2_avx256 (TrHi[j], TrLo[j]);
				Sr = QuadMul (alpha, Sr);
				C[j * ldc + rowid] = Sr.x;
				C[j * ldc + rowid + ldc * LL] = Sr.y;
			}
		} else {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				double2 Sr = hadd_x2_avx256 (TrHi[j], TrLo[j]);
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
void csrmm_n_avx256_l3 (
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
		__m256d Tr[LL];
		//#pragma unroll 
		for (int32_t j = 0; j < LL; j++) 
			Tr[j] = _mm256_set_pd (0.,0.,0.,0.);
		int32_t i;
		for(i = csrRowPtrA[rowid]; i < csrRowPtrA[rowid+1] - (VL256-1); i += VL256) {
			__m256d Ar = _mm256_load_pd (&csrValA[i]);
			__m256d Br;
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				Br = _mm256_set_pd (
								B[j * ldb + csrColIndA[i+3]],
								B[j * ldb + csrColIndA[i+2]],
								B[j * ldb + csrColIndA[i+1]],
								B[j * ldb + csrColIndA[i+0]]);
				Tr[j] = _mm256_fmadd_pd (Ar, Br, Tr[j]);
			}
		}
		//#pragma unroll 
		for (int32_t l = VL256-1; l > 0; l--) {
			if (i == csrRowPtrA[rowid+1] - l) {
				__m256d Ar = _mm256_load_pd (&csrValA[i]);
				//#pragma unroll 
				for (int32_t j = 0; j < LL; j++) {
					double Br2[VL256] = {0.};
					for (int32_t ll = l; ll > 0; ll--) 
						Br2[VL256-ll] = B[j * ldb + csrColIndA[i+ll-1]];
					__m256d Br3 = _mm256_set_pd (Br2[0], Br2[1], Br2[2], Br2[3]);
					Tr[j] = _mm256_fmadd_pd (Ar, Br3, Tr[j]);
				}
				i += l;
            }
		}
		if (beta == 0.) {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) 
				C[j * ldc + rowid] = alpha * hadd_avx256 (Tr[j]);
		} else {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) 
				C[j * ldc + rowid] = fma (alpha, hadd_avx256 (Tr[j]), (beta * C[j * ldc + rowid]));
		}
	}
}

void csrmm_n_x2_avx256_l3 (
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
		__m256d TrHi[LL];
		__m256d TrLo[LL];
		//#pragma unroll 
		for (int32_t j = 0; j < LL; j++) {
			TrHi[j] = _mm256_set_pd (0.,0.,0.,0.);
			TrLo[j] = _mm256_set_pd (0.,0.,0.,0.);
		}
		int32_t i;
		for(i = csrRowPtrA[rowid]; i < csrRowPtrA[rowid+1] - (VL256-1); i += VL256) {
			__m256d Ar = _mm256_load_pd (&csrValA[i]);
			__m256d Br;
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				Br = _mm256_set_pd (
								B[j * ldb + csrColIndA[i+3]],
								B[j * ldb + csrColIndA[i+2]],
								B[j * ldb + csrColIndA[i+1]],
								B[j * ldb + csrColIndA[i+0]]);
				Dot2_avx256 (Ar, Br, TrHi[j], TrLo[j]);
			}
		}
		//#pragma unroll 
		for (int32_t l = VL256-1; l > 0; l--) {
			if (i == csrRowPtrA[rowid+1] - l) {
				__m256d Ar = _mm256_load_pd (&csrValA[i]);
				//#pragma unroll 
				for (int32_t j = 0; j < LL; j++) {
					double Br2[VL256] = {0.};
					for (int32_t ll = l; ll > 0; ll--) 
						Br2[VL256-ll] = B[j * ldb + csrColIndA[i+ll-1]];
					__m256d Br3 = _mm256_set_pd (Br2[0], Br2[1], Br2[2], Br2[3]);
					Dot2_avx256 (Ar, Br3, TrHi[j], TrLo[j]);
				}
				i += l;
            }
		}
		if (beta == 0.) {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				double2 Sr = hadd_x2_avx256 (TrHi[j], TrLo[j]);
				Sr = QuadMul (alpha, Sr);
				C[j * ldc + rowid] = Sr.x;
				C[j * ldc + rowid + ldc * LL] = Sr.y;
			}
		} else {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				double2 Sr = hadd_x2_avx256 (TrHi[j], TrLo[j]);
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
void csrmm_n_avx256_l4 (
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
		__m256d Tr[LL];
		//#pragma unroll 
		for (int32_t j = 0; j < LL; j++) 
			Tr[j] = _mm256_set_pd (0.,0.,0.,0.);
		int32_t i;
		for(i = csrRowPtrA[rowid]; i < csrRowPtrA[rowid+1] - (VL256-1); i += VL256) {
			__m256d Ar = _mm256_load_pd (&csrValA[i]);
			__m256d Br;
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				Br = _mm256_set_pd (
								B[j * ldb + csrColIndA[i+3]],
								B[j * ldb + csrColIndA[i+2]],
								B[j * ldb + csrColIndA[i+1]],
								B[j * ldb + csrColIndA[i+0]]);
				Tr[j] = _mm256_fmadd_pd (Ar, Br, Tr[j]);
			}
		}
		//#pragma unroll 
		for (int32_t l = VL256-1; l > 0; l--) {
			if (i == csrRowPtrA[rowid+1] - l) {
				__m256d Ar = _mm256_load_pd (&csrValA[i]);
				//#pragma unroll 
				for (int32_t j = 0; j < LL; j++) {
					double Br2[VL256] = {0.};
					for (int32_t ll = l; ll > 0; ll--) 
						Br2[VL256-ll] = B[j * ldb + csrColIndA[i+ll-1]];
					__m256d Br3 = _mm256_set_pd (Br2[0], Br2[1], Br2[2], Br2[3]);
					Tr[j] = _mm256_fmadd_pd (Ar, Br3, Tr[j]);
				}
				i += l;
            }
		}
		if (beta == 0.) {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) 
				C[j * ldc + rowid] = alpha * hadd_avx256 (Tr[j]);
		} else {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) 
				C[j * ldc + rowid] = fma (alpha, hadd_avx256 (Tr[j]), (beta * C[j * ldc + rowid]));
		}
	}
}

void csrmm_n_x2_avx256_l4 (
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
		__m256d TrHi[LL];
		__m256d TrLo[LL];
		//#pragma unroll 
		for (int32_t j = 0; j < LL; j++) {
			TrHi[j] = _mm256_set_pd (0.,0.,0.,0.);
			TrLo[j] = _mm256_set_pd (0.,0.,0.,0.);
		}
		int32_t i;
		for(i = csrRowPtrA[rowid]; i < csrRowPtrA[rowid+1] - (VL256-1); i += VL256) {
			__m256d Ar = _mm256_load_pd (&csrValA[i]);
			__m256d Br;
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				Br = _mm256_set_pd (
								B[j * ldb + csrColIndA[i+3]],
								B[j * ldb + csrColIndA[i+2]],
								B[j * ldb + csrColIndA[i+1]],
								B[j * ldb + csrColIndA[i+0]]);
				Dot2_avx256 (Ar, Br, TrHi[j], TrLo[j]);
			}
		}
		//#pragma unroll 
		for (int32_t l = VL256-1; l > 0; l--) {
			if (i == csrRowPtrA[rowid+1] - l) {
				__m256d Ar = _mm256_load_pd (&csrValA[i]);
				//#pragma unroll 
				for (int32_t j = 0; j < LL; j++) {
					double Br2[VL256] = {0.};
					for (int32_t ll = l; ll > 0; ll--) 
						Br2[VL256-ll] = B[j * ldb + csrColIndA[i+ll-1]];
					__m256d Br3 = _mm256_set_pd (Br2[0], Br2[1], Br2[2], Br2[3]);
					Dot2_avx256 (Ar, Br3, TrHi[j], TrLo[j]);
				}
				i += l;
            }
		}
		if (beta == 0.) {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				double2 Sr = hadd_x2_avx256 (TrHi[j], TrLo[j]);
				Sr = QuadMul (alpha, Sr);
				C[j * ldc + rowid] = Sr.x;
				C[j * ldc + rowid + ldc * LL] = Sr.y;
			}
		} else {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				double2 Sr = hadd_x2_avx256 (TrHi[j], TrLo[j]);
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
void csrmm_n_avx256_l5 (
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
		__m256d Tr[LL];
		//#pragma unroll 
		for (int32_t j = 0; j < LL; j++) 
			Tr[j] = _mm256_set_pd (0.,0.,0.,0.);
		int32_t i;
		for(i = csrRowPtrA[rowid]; i < csrRowPtrA[rowid+1] - (VL256-1); i += VL256) {
			__m256d Ar = _mm256_load_pd (&csrValA[i]);
			__m256d Br;
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				Br = _mm256_set_pd (
								B[j * ldb + csrColIndA[i+3]],
								B[j * ldb + csrColIndA[i+2]],
								B[j * ldb + csrColIndA[i+1]],
								B[j * ldb + csrColIndA[i+0]]);
				Tr[j] = _mm256_fmadd_pd (Ar, Br, Tr[j]);
			}
		}
		//#pragma unroll 
		for (int32_t l = VL256-1; l > 0; l--) {
			if (i == csrRowPtrA[rowid+1] - l) {
				__m256d Ar = _mm256_load_pd (&csrValA[i]);
				//#pragma unroll 
				for (int32_t j = 0; j < LL; j++) {
					double Br2[VL256] = {0.};
					for (int32_t ll = l; ll > 0; ll--) 
						Br2[VL256-ll] = B[j * ldb + csrColIndA[i+ll-1]];
					__m256d Br3 = _mm256_set_pd (Br2[0], Br2[1], Br2[2], Br2[3]);
					Tr[j] = _mm256_fmadd_pd (Ar, Br3, Tr[j]);
				}
				i += l;
            }
		}
		if (beta == 0.) {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) 
				C[j * ldc + rowid] = alpha * hadd_avx256 (Tr[j]);
		} else {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) 
				C[j * ldc + rowid] = fma (alpha, hadd_avx256 (Tr[j]), (beta * C[j * ldc + rowid]));
		}
	}
}

void csrmm_n_x2_avx256_l5 (
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
		__m256d TrHi[LL];
		__m256d TrLo[LL];
		//#pragma unroll 
		for (int32_t j = 0; j < LL; j++) {
			TrHi[j] = _mm256_set_pd (0.,0.,0.,0.);
			TrLo[j] = _mm256_set_pd (0.,0.,0.,0.);
		}
		int32_t i;
		for(i = csrRowPtrA[rowid]; i < csrRowPtrA[rowid+1] - (VL256-1); i += VL256) {
			__m256d Ar = _mm256_load_pd (&csrValA[i]);
			__m256d Br;
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				Br = _mm256_set_pd (
								B[j * ldb + csrColIndA[i+3]],
								B[j * ldb + csrColIndA[i+2]],
								B[j * ldb + csrColIndA[i+1]],
								B[j * ldb + csrColIndA[i+0]]);
				Dot2_avx256 (Ar, Br, TrHi[j], TrLo[j]);
			}
		}
		//#pragma unroll 
		for (int32_t l = VL256-1; l > 0; l--) {
			if (i == csrRowPtrA[rowid+1] - l) {
				__m256d Ar = _mm256_load_pd (&csrValA[i]);
				//#pragma unroll 
				for (int32_t j = 0; j < LL; j++) {
					double Br2[VL256] = {0.};
					for (int32_t ll = l; ll > 0; ll--) 
						Br2[VL256-ll] = B[j * ldb + csrColIndA[i+ll-1]];
					__m256d Br3 = _mm256_set_pd (Br2[0], Br2[1], Br2[2], Br2[3]);
					Dot2_avx256 (Ar, Br3, TrHi[j], TrLo[j]);
				}
				i += l;
            }
		}
		if (beta == 0.) {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				double2 Sr = hadd_x2_avx256 (TrHi[j], TrLo[j]);
				Sr = QuadMul (alpha, Sr);
				C[j * ldc + rowid] = Sr.x;
				C[j * ldc + rowid + ldc * LL] = Sr.y;
			}
		} else {
			//#pragma unroll 
			for (int32_t j = 0; j < LL; j++) {
				double2 Sr = hadd_x2_avx256 (TrHi[j], TrLo[j]);
				Sr = QuadAdd (QuadMul (alpha, Sr), QuadMul (beta, C[j * ldc + rowid]));
				C[j * ldc + rowid] = Sr.x;
				C[j * ldc + rowid + ldc * LL] = Sr.y;
			}
		}
	}
}
#undef LL

#endif
