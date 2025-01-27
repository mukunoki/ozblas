#include "ozblas_common.h"

// =========================================
// Correctly-rounded summation
// with NearSum
// from "ACCURATE FLOATING-POINT SUMMATION" by S.M.RUMP, T.OGITA, S.OISHI (2005)
// http://www.ti3.tu-harburg.de/paper/rump/RuOgOi06.pdf
// =========================================

template <typename TYPE>
void ozblasTransform (
	const int32_t n,
	TYPE *vec,
	const int32_t ld,
	TYPE rho,
	TYPE &tau1,
	TYPE &tau2
) {
	constexpr int32_t emin = getEmin <TYPE> ();
	constexpr int32_t epse = getEpse <TYPE> ();

	int32_t i, m;
	TYPE tmp, mu, sigma, t, tau;

	mu = fabs(vec[0]);
	for (i = 1; i < n; i++)
		mu = MAX (mu, fabs(vec[i*ld])); 
	if ((n == 0) || (mu == 0.)) {
		tau1 = rho;
		tau2 = 0.;
		return;
	}

	m = ceil (std::log2((double)(n+2)));
	sigma = scalbn1 (1., m+ceil(std::log2(mu)));
	t = rho;
	while (1) {
		// ExtractVector
		tau = 0.;
		for (i = 0; i < n; i++) {
			tmp = (sigma + vec[i*ld]) - sigma;
			vec[i*ld] -= tmp;  // <- output
			tau += tmp;
		}
		// here, tau1 = t1
		tau1 = t + tau;
		if ((sigma <= scalbn1 (1., emin)) || (fabs(tau1) >= scalbn1 (1., 2*m+1-epse)*sigma)) {
			//FastTwoSum (t, tau, &tau1, &tau2);
			//tau1 = t + tau
			tau2 = tau - (tau1 - t);
			return;
		}
		sigma = scalbn1 (1., m-epse) * sigma;
		t = tau1;
	} 
}

template <typename TYPE>
void ozblasTransformK (
	const int32_t n,
	TYPE *vec,
	const int32_t ld,
	TYPE rho,
	TYPE &res,
	TYPE &r
) { 
	TYPE tmp = 0., tau1, tau2;
	ozblasTransform (n, vec, ld, rho, tau1, tau2);
	for (int32_t i = 0; i < n; i++)
		tmp += vec[i*ld];
	res = tau1 + (tau2 + tmp);
	r = tau2 - (res - tau1);
}

template <typename TYPE>
__inline__
TYPE getSign (TYPE v) {
	return (v < 0) ? -1.:1.;
}

template <typename TYPE>
TYPE ozblasNearsum (
	const int32_t n,
	TYPE *vec,
	const int32_t ld
) {
	constexpr int32_t epse = getEpse <TYPE> ();
	TYPE tmp, res, res2, r, r2, mu, delta, delta2;
	TYPE eps = scalbn1 (1., -epse);

	ozblasTransformK (n, vec, ld, (TYPE)0., res, r);
	ozblasTransformK (n, vec, ld, r, delta, r2);
	if (delta == 0) 
		return res;
	res2 = res + getSign (delta) * eps * fabs(res);
	if (res2 == res) {
		mu = getSign (delta) * eps * fabs(res);
		res2 = res + 2. * getSign (delta) * eps * fabs(res);
	} else {
		mu = (res2 - res) / 2.;
	}
	if (fabs(delta) < fabs(mu)) 
		return res;
	if (fabs(delta) > fabs(mu)) 
		return res2;
	ozblasTransformK (n, vec, ld, r2, delta2, tmp);
	if (delta2 == 0) 
		return res + mu;
	if (getSign (delta2) == getSign (mu))
		return res2;
	return res;
}

/*
template <typename TYPE>
void ozblasTransformNpara (
	const int32_t n,
	TYPE *vec,
	const int32_t ld,
	TYPE rho,
	TYPE &tau1,
	TYPE &tau2
) {
	constexpr int32_t emin = getEmin <TYPE> ();
	constexpr int32_t epse = getEpse <TYPE> ();

	int32_t i, m;
	TYPE tmp, mu, sigma, t, tau;

	mu = fabs(vec[0]);
	for (i = 1; i < n; i++)
		mu = MAX (mu, fabs(vec[i*ld])); 
	if ((n == 0) || (mu == 0.)) {
		tau1 = rho;
		tau2 = 0.;
		return;
	}

	m = ceil (std::log2((double)(n+2)));
	sigma = scalbn1 (1., m+ceil(std::log2(mu)));
	t = rho;
	while (1) {
		// ExtractVector
		tau = 0.;
		#pragma omp parallel for private (tmp) reduction(+:tau)
		for (i = 0; i < n; i++) {
			tmp = (sigma + vec[i*ld]) - sigma;
			vec[i*ld] -= tmp;  // <- output
			tau += tmp;
		}
		// here, tau1 = t1
		tau1 = t + tau;
		if ((sigma <= scalbn1 (1., emin)) || (fabs(tau1) >= scalbn1 (1., 2*m+1-epse)*sigma)) {
			//FastTwoSum (t, tau, &tau1, &tau2);
			//tau1 = t + tau
			tau2 = tau - (tau1 - t);
			return;
		}
		sigma = scalbn1 (1., m-epse) * sigma;
		t = tau1;
	} 
}

template <typename TYPE>
void ozblasTransformKNpara (
	const int32_t n,
	TYPE *vec,
	const int32_t ld,
	TYPE rho,
	TYPE &res,
	TYPE &r
) { 
	TYPE tmp = 0., tau1, tau2;
	ozblasTransformNpara (n, vec, ld, rho, tau1, tau2);
	int32_t i;
	#pragma omp parallel for reduction(+:tmp)
	for (i = 0; i < n; i++) {
		tmp += vec[i*ld];
	}
	res = tau1 + (tau2 + tmp);
	r = tau2 - (res - tau1);
}

template <typename TYPE>
TYPE ozblasNearsumNpara (
	const int32_t n,
	TYPE *vec,
	const int32_t ld
) {
	constexpr int32_t epse = getEpse <TYPE> ();
	TYPE tmp, res, res2, r, r2, mu, delta, delta2;
	TYPE eps = scalbn1 (1., -epse);

	ozblasTransformKNpara (n, vec, ld, (TYPE)0., res, r);
	ozblasTransformKNpara (n, vec, ld, r, delta, r2);
	if (delta == 0) 
		return res;
	res2 = res + getSign (delta) * eps * fabs(res);
	if (res2 == res) {
		mu = getSign (delta) * eps * fabs(res);
		res2 = res + 2. * getSign (delta) * eps * fabs(res);
	} else {
		mu = (res2 - res) / 2.;
	}
	if (fabs(delta) < fabs(mu)) 
		return res;
	if (fabs(delta) > fabs(mu)) 
		return res2;
	ozblasTransformKNpara (n, vec, ld, r2, delta2, tmp);
	if (delta2 == 0) 
		return res + mu;
	if (getSign (delta2) == getSign (mu))
		return res2;
	return res;
}
*/

template <typename TYPE>
void ozblasGlobalNearsumKernel (
	const int32_t m,
	const int32_t n,
	const short *devASpExp,
	const int32_t ldase,
	const int32_t nSplitA,
	const short *devBSpExp,
	const int32_t ldbse,
	const int32_t nSplitB,
	TYPE *devCsplit,
	const int32_t llsc,
	const int32_t ldsc,
	TYPE *devC,
	const int32_t ldc,
	const TYPE alpha,
	const TYPE beta,
	const int32_t maxlevel,
	const int32_t sumOrder, 
	const int32_t split3FlagA,
	const int32_t split3FlagB,
	int32_t *check
) {
	#pragma omp parallel for
	for (int32_t addry = 0; addry < n; addry++) {
		for (int32_t addrx = 0; addrx < m; addrx++) {
			TYPE t = 0.;
			int32_t ic = 0;
			for (int32_t ik = 0; ik <= maxlevel; ik++) {
				for (int32_t ia = 0; ia < nSplitA; ia++) {
					for (int32_t ib = 0; ib < nSplitB; ib++) {
						if (ik == ia + ib) 
							ic++;
					}
				}
			}
			t = ozblasNearsum (ic, &devCsplit[addry * ldsc + addrx], llsc);
			devC[addry * ldc + addrx] = fma1 (alpha, t, (beta * devC[addry * ldc + addrx]));
		}
	}
}

template <typename TYPE1, typename TYPE2>
void ozblasGlobalNearsumKernel (
	const int32_t m,
	const int32_t n,
	const short *devASpExp,
	const int32_t ldase,
	const int32_t nSplitA,
	const short *devBSpExp,
	const int32_t ldbse,
	const int32_t nSplitB,
	TYPE2 *devCsplit,
	const int32_t llsc,
	const int32_t ldsc,
	TYPE1 *devC,
	const int32_t ldc,
	const TYPE1 alpha,
	const TYPE1 beta,
	const int32_t maxlevel,
	const int32_t sumOrder, 
	const int32_t split3FlagA,
	const int32_t split3FlagB,
	int32_t *check
) {
	fprintf (OUTPUT, "OzBLAS error: ozblasGlobalNearsumKernel is not available when TYPE1 != TYPE2.\n");
	exit(1);
	/*
	#pragma omp parallel for
	for (int32_t addry = 0; addry < n; addry++) {
		for (int32_t addrx = 0; addrx < m; addrx++) {
			TYPE1 t = 0.;
			int32_t ic = 0;
			for (int32_t ik = 0; ik <= maxlevel; ik++) {
				for (int32_t ia = 0; ia < nSplitA; ia++) {
					short seA = devASpExp[ldase*ia+addrx];
					for (int32_t ib = 0; ib < nSplitB; ib++) {
						if (ik == ia + ib) {
							// any order is OK in nearsum
							int32_t it = (sumOrder == 1) ? ic : (nSplitA * ib + ia);
							TYPE1 c = (TYPE1)devCsplit[llsc * it + addry * ldsc + addrx];
							short seB = devBSpExp[ldbse*ib+addry];
							devCsplit[llsc * it + addry * ldsc + addrx] = scalbn1 (c, seA+seB);
							ic++;
						}
					}
				}
			}
			t = ozblasNearsum (ic, &devCsplit[addry * ldsc + addrx], llsc);
			devC[addry * ldc + addrx] = fma1 (alpha, t, (beta * devC[addry * ldc + addrx]));
		}
	}
	*/
}

// =========================================
// Floating-point summation (FSum)
// =========================================

// mono-precision, Exp is ignored
template <typename TYPE>
void ozblasGlobalFsumKernel (
	const int32_t m,
	const int32_t n,
	const short *devASpExp,
	const int32_t ldase,
	const int32_t nSplitA,
	const short *devBSpExp,
	const int32_t ldbse,
	const int32_t nSplitB,
	const TYPE *devCsplit,
	const int32_t llsc,
	const int32_t ldsc,
	TYPE *devC,
	const int32_t ldc,
	const TYPE alpha,
	const TYPE beta,
	const int32_t maxlevel,
	const int32_t sumOrder, 
	const int32_t split3FlagA,
	const int32_t split3FlagB,
	int32_t *check
) {
printf ("sum mono\n");
	#pragma omp parallel for
	for (int32_t addry = 0; addry < n; addry++) {
		for (int32_t addrx = 0; addrx < m; addrx++) {
			TYPE t = 0.;
			int32_t ic = 0;
			for (int32_t ik = 0; ik <= maxlevel; ik++) {
				for (int32_t ia = 0; ia < nSplitA; ia++) {
					for (int32_t ib = 0; ib < nSplitB; ib++) {
						if (ik == ia + ib) {
							// order must be cared
							int32_t it;
							switch (sumOrder) {
								case 2: it = nSplitA * ib + ia; break;
								case 3: it = nSplitB * ia + ib; break;
								default: it = ic; break;
							}
							TYPE c = devCsplit[llsc * it + addry * ldsc + addrx];
							t += c;
							ic++;
						}
					}
				}
			}
			devC[addry * ldc + addrx] = fma1 (alpha, t, (beta * devC[addry * ldc + addrx]));
		}
	}
}

template <typename TYPE1, typename TYPE2>
void ozblasGlobalFsumKernel (
	const int32_t m,
	const int32_t n,
	const short *devASpExp,
	const int32_t ldase,
	const int32_t nSplitA,
	const short *devBSpExp,
	const int32_t ldbse,
	const int32_t nSplitB,
	const TYPE2 *devCsplit,
	const int32_t llsc,
	const int32_t ldsc,
	TYPE1 *devC,
	const int32_t ldc,
	const TYPE1 alpha,
	const TYPE1 beta,
	const int32_t maxlevel,
	const int32_t sumOrder, 
	const int32_t split3FlagA,
	const int32_t split3FlagB,
	int32_t *check
) {
printf ("sum multi\n");
	#pragma omp parallel for
	for (int32_t addry = 0; addry < n; addry++) {
		for (int32_t addrx = 0; addrx < m; addrx++) {
			TYPE1 t = 0.;
			int32_t ic = 0;
			for (int32_t ik = 0; ik <= maxlevel; ik++) {
				for (int32_t ia = 0; ia < nSplitA; ia++) {
					short seA = (split3FlagA) ? 0:devASpExp[ldase*ia+addrx];
					for (int32_t ib = 0; ib < nSplitB; ib++) {
						if (ik == ia + ib) {
							int32_t it;
							switch (sumOrder) {
								case 2: it = nSplitA * ib + ia; break;
								case 3: it = nSplitB * ia + ib; break;
								default: it = ic; break;
							}
							TYPE1 c = (TYPE1)devCsplit[llsc * it + addry * ldsc + addrx];
							short seB = (split3FlagB) ? 0:devBSpExp[ldbse*ib+addry];
							t += scalbn1 (c, seA+seB);
							ic++;
						}
					}
				}
			}
			devC[addry * ldc + addrx] = fma1 (alpha, t, (beta * devC[addry * ldc + addrx]));
		}
	}
}

template <typename TYPE1, typename TYPE2>
int32_t ozblasGlobalSum (
	ozblasHandle_t *oh,
	const int32_t m,
	const int32_t n,
	const short *devASpExp,
	const int32_t ldase,
	const int32_t nSplitA,
	const short *devBSpExp,
	const int32_t ldbse,
	const int32_t nSplitB,
	TYPE2 *devCsplit, // [llsc * nSplitA * nSplitB]
	const int32_t llsc, // = ldsc * n
	const int32_t ldsc,
	TYPE1 *devC,
	const int32_t ldc,
	const TYPE1 alpha,
	const TYPE1 beta,
	const int32_t maxlevel,
	const int32_t sumOrder,
	const int32_t split3FlagA,
	const int32_t split3FlagB
) {
	int32_t check = 0;
	if (oh->sumModeFlag == 1) { // Nearsum
//		if (m == 1 && n == 1) { // for DOT, not effective for performance...
//			devC[0] = alpha * ozblasNearsumNpara (nSplitA*nSplitB, &devCsplit[0], llsc) + beta * devC[0];
//		} else {
		ozblasGlobalNearsumKernel (m, n, devASpExp, ldase, nSplitA, devBSpExp, ldbse, nSplitB,
								  devCsplit, llsc, ldsc, devC, ldc, alpha, beta, maxlevel, sumOrder, split3FlagA, split3FlagB, &check);
//		}
	} else { // Fsum
		ozblasGlobalFsumKernel (m, n, devASpExp, ldase, nSplitA, devBSpExp, ldbse, nSplitB,
							  devCsplit, llsc, ldsc, devC, ldc, alpha, beta, maxlevel, sumOrder, split3FlagA, split3FlagB, &check);
	}
	return check;
}
template int32_t ozblasGlobalSum <__float128, double> (ozblasHandle_t *oh, const int32_t m, const int32_t n, const short *devASpExp, const int32_t ldase, const int32_t nSplitA, const short *devBSpExp, const int32_t ldbse, const int32_t nSplitB, double *devCsplit, const int32_t llsc, const int32_t ldsc, __float128 *devC, const int32_t ldc, const __float128 alpha, const __float128 beta, const int32_t maxlevel, const int32_t sumOrder, const int32_t split3FlagA, const int32_t split3FlagB);
template int32_t ozblasGlobalSum <__float128, float> (ozblasHandle_t *oh, const int32_t m, const int32_t n, const short *devASpExp, const int32_t ldase, const int32_t nSplitA, const short *devBSpExp, const int32_t ldbse, const int32_t nSplitB, float *devCsplit, const int32_t llsc, const int32_t ldsc, __float128 *devC, const int32_t ldc, const __float128 alpha, const __float128 beta, const int32_t maxlevel, const int32_t sumOrder, const int32_t split3FlagA, const int32_t split3FlagB);
template int32_t ozblasGlobalSum <double, double> (ozblasHandle_t *oh, const int32_t m, const int32_t n, const short *devASpExp, const int32_t ldase, const int32_t nSplitA, const short *devBSpExp, const int32_t ldbse, const int32_t nSplitB, double *devCsplit, const int32_t llsc, const int32_t ldsc, double *devC, const int32_t ldc, const double alpha, const double beta, const int32_t maxlevel, const int32_t sumOrder, const int32_t split3FlagA, const int32_t split3FlagB);
template int32_t ozblasGlobalSum <double, float> (ozblasHandle_t *oh, const int32_t m, const int32_t n, const short *devASpExp, const int32_t ldase, const int32_t nSplitA, const short *devBSpExp, const int32_t ldbse, const int32_t nSplitB, float *devCsplit, const int32_t llsc, const int32_t ldsc, double *devC, const int32_t ldc, const double alpha, const double beta, const int32_t maxlevel, const int32_t sumOrder, const int32_t split3FlagA, const int32_t split3FlagB);
template int32_t ozblasGlobalSum <float, float> (ozblasHandle_t *oh, const int32_t m, const int32_t n, const short *devASpExp, const int32_t ldase, const int32_t nSplitA, const short *devBSpExp, const int32_t ldbse, const int32_t nSplitB, float *devCsplit, const int32_t llsc, const int32_t ldsc, float *devC, const int32_t ldc, const float alpha, const float beta, const int32_t maxlevel, const int32_t sumOrder, const int32_t split3FlagA, const int32_t split3FlagB);
template int32_t ozblasGlobalSum <float, double> (ozblasHandle_t *oh, const int32_t m, const int32_t n, const short *devASpExp, const int32_t ldase, const int32_t nSplitA, const short *devBSpExp, const int32_t ldbse, const int32_t nSplitB, double *devCsplit, const int32_t llsc, const int32_t ldsc, float *devC, const int32_t ldc, const float alpha, const float beta, const int32_t maxlevel, const int32_t sumOrder, const int32_t split3FlagA, const int32_t split3FlagB);


// ==============================================
// For Quadruple-precision
// ==============================================
template <typename TYPE1, typename TYPE2>
int32_t ozblasLocalFsum (
	const int32_t m,
	const int32_t n,
	const short *devASpExp,
	const short *devBSpExp,
	const TYPE2 *devCsplit,
	const int32_t ldcs,
	TYPE1 *devCtmp,
	const int32_t ldct,
	const int32_t ic,
	const int32_t split3FlagA,
	const int32_t split3FlagB
) {
	int32_t check = 0;
	#pragma omp parallel for
	for (int32_t addry = 0; addry < n; addry++) {
		short seB = (split3FlagB) ? 0:devBSpExp[addry];
		for (int32_t addrx = 0; addrx < m; addrx++) {
			TYPE1 c = devCsplit[addry * ldcs + addrx];
			short seA = (split3FlagA) ? 0:devASpExp[addrx];
			c = scalbn1 (c, seA+seB);
			if (ic == 0) 
				devCtmp[addry * ldct + addrx] = c;
			else
				devCtmp[addry * ldct + addrx] += c;
		}
	}
	return check;
}
template int32_t ozblasLocalFsum <__float128, double> (const int32_t m, const int32_t n, const short *devASpExp, const short *devBSpExp, const double *devCsplit, const int32_t ldcs, __float128 *devCtmp, const int32_t ldct, const int32_t ic, const int32_t split3FlagA, const int32_t split3FlagB);
template int32_t ozblasLocalFsum <__float128, float> (const int32_t m, const int32_t n, const short *devASpExp, const short *devBSpExp, const float *devCsplit, const int32_t ldcs, __float128 *devCtmp, const int32_t ldct, const int32_t ic, const int32_t split3FlagA, const int32_t split3FlagB);
template int32_t ozblasLocalFsum <double, double> (const int32_t m, const int32_t n, const short *devASpExp, const short *devBSpExp, const double *devCsplit, const int32_t ldcs, double *devCtmp, const int32_t ldct, const int32_t ic, const int32_t split3FlagA, const int32_t split3FlagB);
template int32_t ozblasLocalFsum <double, float> (const int32_t m, const int32_t n, const short *devASpExp, const short *devBSpExp, const float *devCsplit, const int32_t ldcs, double *devCtmp, const int32_t ldct, const int32_t ic, const int32_t split3FlagA, const int32_t split3FlagB);
template int32_t ozblasLocalFsum <float, float> (const int32_t m, const int32_t n, const short *devASpExp, const short *devBSpExp, const float *devCsplit, const int32_t ldcs, float *devCtmp, const int32_t ldct, const int32_t ic, const int32_t split3FlagA, const int32_t split3FlagB);
template int32_t ozblasLocalFsum <float, double> (const int32_t m, const int32_t n, const short *devASpExp, const short *devBSpExp, const double *devCsplit, const int32_t ldcs, float *devCtmp, const int32_t ldct, const int32_t ic, const int32_t split3FlagA, const int32_t split3FlagB);

// AXPBY: y=ax+by
template <typename TYPE>
int32_t ozblasAxpby (
	const int32_t m,
	const int32_t n,
	const TYPE *devCsplit,
	const int32_t ldsc,
	TYPE *devC,
	const int32_t ldc,
	const TYPE alpha,
	const TYPE beta
) {
	// note: check is not implemented
	int32_t checkGlobal = 0;
	#pragma omp parallel for
	for (int32_t addry = 0; addry < n; addry++) {
		for (int32_t addrx = 0; addrx < m; addrx++) {
			TYPE t = devCsplit[addry * ldsc + addrx];
			devC[addry * ldc + addrx] = alpha * t + beta * devC[addry * ldc + addrx];
		}
	}
	return checkGlobal;
}
template int32_t ozblasAxpby (const int32_t m, const int32_t n, const __float128 *devCsplit, const int32_t ldsc, __float128 *devC, const int32_t ldc, const __float128 alpha, const __float128 beta);
template int32_t ozblasAxpby (const int32_t m, const int32_t n, const double *devCsplit, const int32_t ldsc, double *devC, const int32_t ldc, const double alpha, const double beta);
template int32_t ozblasAxpby (const int32_t m, const int32_t n, const float *devCsplit, const int32_t ldsc, float *devC, const int32_t ldc, const float alpha, const float beta);

// ==============================================
// For Quadruple-precision
// FastSum with 3 binary64 bins
// ==============================================

template <typename TYPE>
__inline__ void
TwoSum (TYPE a, TYPE b, TYPE &s, TYPE &e)
{
	TYPE v;
	s = a + b;
	v = s - a;
	e = (a - (s - v)) + (b - v);
}

template <typename TYPE1, typename TYPE2>
int32_t ozblasLocalFsum3 (
	const int32_t m,
	const int32_t n,
	const short *devASpExp,
	const short *devBSpExp,
	const TYPE2 *devCsplit,
	const int32_t ldcs,
	TYPE1 *devCtmp,
	const int32_t ldct,
	TYPE2 *devCtmp1,
	const int32_t ldct1,
	TYPE2 *devCtmp2,
	const int32_t ldct2,
	TYPE2 *devCtmp3,
	const int32_t ldct3,
	const int32_t ic,
	const int32_t split3FlagA,
	const int32_t split3FlagB
) {
	int32_t checkGlobal = 0;
	#pragma omp parallel for
	for (int32_t addry = 0; addry < n; addry++) {
		int32_t checkLocal = 0;
		#pragma omp atomic read
		checkLocal = checkGlobal;
		if (checkLocal) continue;

		short seB = (split3FlagB) ? 0:devBSpExp[addry];
		for (int32_t addrx = 0; addrx < m; addrx++) {
			TYPE2 c = devCsplit[addry * ldcs + addrx]; 
			short seA = (split3FlagA) ? 0:devASpExp[addrx];
			c = scalbn1 (c, seA+seB); // here, TYPE2
			if (ic == 0) { // fisrt
				devCtmp1[addry * ldct1 + addrx] = c;
				devCtmp2[addry * ldct2 + addrx] = 0.;
				devCtmp3[addry * ldct3 + addrx] = 0.;
			} else {
				TYPE2 e1, e2, t1, t2, t3;
				t1 = devCtmp1[addry * ldct1 + addrx];
				t2 = devCtmp2[addry * ldct2 + addrx];
				t3 = devCtmp3[addry * ldct3 + addrx];
				TwoSum (t1, c, t1, e1);
				TwoSum (t2, e1, t2, e2);
				t3 += e2;
				if (ic == -1) { // last
					// check overflow (this check should be done on TYPE2)
					if (std::isinf(t1) || std::isnan(t1) || std::isinf(t2) || std::isnan(t2) || std::isinf(t3) || std::isnan(t3)) checkLocal = 1; // computation failed
					else devCtmp[addry * ldct + addrx] = (TYPE1)t1 + ((TYPE1)t2 + (TYPE1)t3);
				} else {
					devCtmp1[addry * ldct1 + addrx] = t1;
					devCtmp2[addry * ldct2 + addrx] = t2;
					devCtmp3[addry * ldct3 + addrx] = t3;
				}
			}
		}

		if (checkLocal) {
			// here you can write re-do codes with TYPE1
			#pragma omp atomic write
			checkGlobal = 1;
		}
	}
	return checkGlobal;
}
template int32_t ozblasLocalFsum3 <__float128, double> (const int32_t m, const int32_t n, const short *devASpExp, const short *devBSpExp, const double *devCsplit, const int32_t ldcs, __float128 *devCtmp, const int32_t ldct, double *devCtmp1, const int32_t ldct1, double *devCtmp2, const int32_t ldct2, double *devCtmp3, const int32_t ldct3, const int32_t ic, const int32_t split3FlagA, const int32_t split3FlagB);
template int32_t ozblasLocalFsum3 <__float128, float> (const int32_t m, const int32_t n, const short *devASpExp, const short *devBSpExp, const float *devCsplit, const int32_t ldcs, __float128 *devCtmp, const int32_t ldct, float *devCtmp1, const int32_t ldct1, float *devCtmp2, const int32_t ldct2, float *devCtmp3, const int32_t ldct3, const int32_t ic, const int32_t split3FlagA, const int32_t split3FlagB);
template int32_t ozblasLocalFsum3 <double, double> (const int32_t m, const int32_t n, const short *devASpExp, const short *devBSpExp, const double *devCsplit, const int32_t ldcs, double *devCtmp, const int32_t ldct, double *devCtmp1, const int32_t ldct1, double *devCtmp2, const int32_t ldct2, double *devCtmp3, const int32_t ldct3, const int32_t ic, const int32_t split3FlagA, const int32_t split3FlagB);
template int32_t ozblasLocalFsum3 <double, float> (const int32_t m, const int32_t n, const short *devASpExp, const short *devBSpExp, const float *devCsplit, const int32_t ldcs, double *devCtmp, const int32_t ldct, float *devCtmp1, const int32_t ldct1, float *devCtmp2, const int32_t ldct2, float *devCtmp3, const int32_t ldct3, const int32_t ic, const int32_t split3FlagA, const int32_t split3FlagB);
template int32_t ozblasLocalFsum3 <float, double> (const int32_t m, const int32_t n, const short *devASpExp, const short *devBSpExp, const double *devCsplit, const int32_t ldcs, float *devCtmp, const int32_t ldct, double *devCtmp1, const int32_t ldct1, double *devCtmp2, const int32_t ldct2, double *devCtmp3, const int32_t ldct3, const int32_t ic, const int32_t split3FlagA, const int32_t split3FlagB);
template int32_t ozblasLocalFsum3 <float, float> (const int32_t m, const int32_t n, const short *devASpExp, const short *devBSpExp, const float *devCsplit, const int32_t ldcs, float *devCtmp, const int32_t ldct, float *devCtmp1, const int32_t ldct1, float *devCtmp2, const int32_t ldct2, float *devCtmp3, const int32_t ldct3, const int32_t ic, const int32_t split3FlagA, const int32_t split3FlagB);

template <typename TYPE1, typename TYPE15, typename TYPE2>
int32_t ozblasLocalFsum3 (
	const int32_t m,
	const int32_t n,
	const short *devASpExp,
	const short *devBSpExp,
	const TYPE15 *devCsplit,
	const int32_t ldcs,
	TYPE1 *devCtmp,
	const int32_t ldct,
	TYPE2 *devCtmp1,
	const int32_t ldct1,
	TYPE2 *devCtmp2,
	const int32_t ldct2,
	TYPE2 *devCtmp3,
	const int32_t ldct3,
	const int32_t ic,
	const int32_t split3FlagA,
	const int32_t split3FlagB
) {
	int32_t checkGlobal = 0;
	#pragma omp parallel for
	for (int32_t addry = 0; addry < n; addry++) {
		int32_t checkLocal = 0;
		#pragma omp atomic read
		checkLocal = checkGlobal;
		if (checkLocal) continue;

		short seB = (split3FlagB) ? 0:devBSpExp[addry];
		for (int32_t addrx = 0; addrx < m; addrx++) {
			TYPE2 c = (TYPE2)devCsplit[addry * ldcs + addrx];
			short seA = (split3FlagA) ? 0:devASpExp[addrx];
			c = scalbn1 (c, seA+seB);
			if (ic == 0) { // fisrt
				devCtmp1[addry * ldct1 + addrx] = c;
				devCtmp2[addry * ldct2 + addrx] = 0.;
				devCtmp3[addry * ldct3 + addrx] = 0.;
			} else {
				TYPE2 e1, e2, t1, t2, t3;
				t1 = devCtmp1[addry * ldct1 + addrx];
				t2 = devCtmp2[addry * ldct2 + addrx];
				t3 = devCtmp3[addry * ldct3 + addrx];
				TwoSum (t1, c, t1, e1);
				TwoSum (t2, e1, t2, e2);
				t3 += e2;
				if (ic == -1) { // last
					// check overflow (this check should be done on TYPE2)
					if (std::isinf(t1) || std::isnan(t1) || std::isinf(t2) || std::isnan(t2) || std::isinf(t3) || std::isnan(t3)) checkLocal = 1; // computation failed
					else devCtmp[addry * ldct + addrx] = (TYPE1)t1 + ((TYPE1)t2 + (TYPE1)t3);
				} else {
					devCtmp1[addry * ldct1 + addrx] = t1;
					devCtmp2[addry * ldct2 + addrx] = t2;
					devCtmp3[addry * ldct3 + addrx] = t3;
				}
			}
		}

		if (checkLocal) {
			// here you can write re-do codes with TYPE1
			#pragma omp atomic write
			checkGlobal = 1;
		}
	}
	return checkGlobal;
}
template int32_t ozblasLocalFsum3 <__float128, float, double> (const int32_t m, const int32_t n, const short *devASpExp, const short *devBSpExp, const float *devCsplit, const int32_t ldcs, __float128 *devCtmp, const int32_t ldct, double *devCtmp1, const int32_t ldct1, double *devCtmp2, const int32_t ldct2, double *devCtmp3, const int32_t ldct3, const int32_t ic, const int32_t split3FlagA, const int32_t split3FlagB);
template int32_t ozblasLocalFsum3 <double, float, double> (const int32_t m, const int32_t n, const short *devASpExp, const short *devBSpExp, const float *devCsplit, const int32_t ldcs, double *devCtmp, const int32_t ldct, double *devCtmp1, const int32_t ldct1, double *devCtmp2, const int32_t ldct2, double *devCtmp3, const int32_t ldct3, const int32_t ic, const int32_t split3FlagA, const int32_t split3FlagB);
template int32_t ozblasLocalFsum3 <float, float, double> (const int32_t m, const int32_t n, const short *devASpExp, const short *devBSpExp, const float *devCsplit, const int32_t ldcs, float *devCtmp, const int32_t ldct, double *devCtmp1, const int32_t ldct1, double *devCtmp2, const int32_t ldct2, double *devCtmp3, const int32_t ldct3, const int32_t ic, const int32_t split3FlagA, const int32_t split3FlagB);

