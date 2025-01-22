#ifndef CUEFT_H
#define CUEFT_H

// ===================================
// EFT (double)
// ===================================

__device__ __forceinline__ void
cuTwoSum
	(double a, double b, double &s, double &e)
{
	double v;
	s = __dadd_rn (a, b);
	v = __dsub_rn (s, a);
	e = __dadd_rn ( __dsub_rn(a, __dsub_rn (s, v)), __dsub_rn (b, v));
}

__device__ __forceinline__ void
cuQuickTwoSum
	(double a, double b, double &s, double &e)
{
	s = __dadd_rn (a, b);
	e = __dsub_rn (b, __dsub_rn (s, a));
}

__device__ __forceinline__ void
cuTwoProdFMA
	(double a, double b, double &p, double &e)
{
	p = __dmul_rn (a, b);
	e = __fma_rn (a, b, -p);
}

__device__ __forceinline__ void
cuDot2i
	(const double a, const double b, double2 &c)
{
	double h, r, q;
	cuTwoProdFMA (a, b, h, r);
	cuTwoSum (c.x, h, c.x, q);
	c.y = __dadd_rn (c.y, __dadd_rn (q, r));
}

// ===================================
// EFT (float)
// ===================================

__device__ __forceinline__ void
cuTwoSumF
	(float a, float b, float &s, float &e)
{
	float v;
	s = __fadd_rn (a, b);
	v = __fsub_rn (s, a);
	e = __fadd_rn ( __fsub_rn(a, __fsub_rn (s, v)), __fsub_rn (b, v));
}

__device__ __forceinline__ void
cuQuickTwoSumF
	(float a, float b, float &s, float &e)
{
	s = __fadd_rn (a, b);
	e = __fsub_rn (b, __fsub_rn (s, a));
}

__device__ __forceinline__ void
cuTwoProdFMAF
	(float a, float b, float &p, float &e)
{
	p = __fmul_rn (a, b);
	e = __fmaf_rn (a, b, -p);
}

// ===================================
// Double-double
// ===================================

__device__ __forceinline__ double2
cuQuadAdd
	(const double2 a, const double2 b) {
	double2 t, c;
	cuTwoSum (a.x, b.x, t.x, t.y);
	t.y = __dadd_rn (t.y, __dadd_rn (a.y, b.y));
	cuQuickTwoSum (t.x, t.y, c.x, c.y);
	return c;
}

__device__ __forceinline__ double2
cuQuadMul
	(const double2 a, const double2 b) {
	double2 t, c;
	cuTwoProdFMA (a.x, b.x, t.x, t.y);
    t.y = __fma_rn (a.x, b.y, t.y);
    t.y = __fma_rn (a.y, b.x, t.y);
    cuQuickTwoSum (t.x, t.y, c.x, c.y);
	return c;
}

__device__ __forceinline__ double2
cuQuadMul
	(const double a, const double2 b) {
	double2 t, c;
	cuTwoProdFMA (a, b.x, t.x, t.y);
	t.y = __fma_rn (a, b.y, t.y);
	cuQuickTwoSum (t.x, t.y, c.x, c.y);
	return c;
}

__device__ __forceinline__ double2
cuQuadMul
	(const double a, const double b) {
	double2 c;
	cuTwoProdFMA (a, b, c.x, c.y);
	return c;
}

// ===================================
// Double-float
// ===================================

__device__ __forceinline__ float2
cuQuadAddF
	(const float2 a, const float2 b) {
	float2 t, c;
	cuTwoSumF (a.x, b.x, t.x, t.y);
	t.y = __fadd_rn (t.y, __fadd_rn (a.y, b.y));
	cuQuickTwoSumF (t.x, t.y, c.x, c.y);
	return c;
}

__device__ __forceinline__ float2
cuQuadMulF
	(const float2 a, const float2 b) {
	float2 t, c;
	cuTwoProdFMAF (a.x, b.x, t.x, t.y);
    t.y = __fmaf_rn (a.x, b.y, t.y);
    t.y = __fmaf_rn (a.y, b.x, t.y);
    cuQuickTwoSumF (t.x, t.y, c.x, c.y);
	return c;
}

// ===================================
// Pair Arithmetic
// ===================================

__device__ __forceinline__ double2
cuQuadAddPA
	(const double2 a, const double2 b) {
	double2 c;
	cuTwoSum (a.x, b.x, c.x, c.y);
	c.y = __dadd_rn (c.y, __dadd_rn (a.y, b.y));
	return c;
}

__device__ __forceinline__ double2
cuQuadMulPA
	(const double2 a, const double2 b) {
	double2 c;
	cuTwoProdFMA (a.x, b.x, c.x, c.y);
    c.y = __fma_rn (a.x, b.y, c.y);
    c.y = __fma_rn (a.y, b.x, c.y);
	return c;
}

// ===================================
// Quasi Triple Word Arithmetic
// ===================================

__device__ __forceinline__ float3
cuQTWadd
	(const float3 a, const float3 b) {
	float3 c, e;
	cuTwoSumF (a.x, b.x, c.x, e.x);
	cuTwoSumF (a.y, b.y, c.y, e.y);
	cuTwoSumF (c.y, e.x, c.y, e.z);
	c.z = __fadd_rn (__fadd_rn (__fadd_rn (a.z, b.z), e.y), e.z);
	return c;
}

__device__ __forceinline__ float3
cuQTWmul
	(const float3 a, const float3 b) {
	float3 c;
    float e1, e2, e3, e4, e5, t2, t3;
	cuTwoProdFMAF (a.x, b.x, c.x, e1);
	cuTwoProdFMAF (a.x, b.y, t2, e2);
	cuTwoProdFMAF (a.y, b.x, t3, e3);
	cuTwoSumF (t2, t3, c.y, e4);
	cuTwoSumF (c.y, e1, c.y, e5);
    c.z = __fadd_rn (__fadd_rn (__fadd_rn (__fmaf_rn (a.z, b.x, e2), __fmaf_rn (a.y, b.y, e3)), __fmaf_rn (a.x, b.z, e4)), e5);
	return c;
}

// ===================================
// Conversion from/to double
// ===================================

__device__ __forceinline__ float2 double_to_float2 (const double d) {
    float h = (float)d;
    float l = d - (double)h;
	return make_float2 (h, l);
}

__device__ __forceinline__ float3 double_to_float3 (const double d) {
    float h = (float)d;
    float m = d - (double)h;
    float l = d - (double)h - (double)m;
	return make_float3 (h, m, l);
}

__device__ __inline__ double float3_to_double (const float3 df) {
	return ((double)df.z + (double)df.y) + (double)df.x;
}

__device__ __inline__ double float2_to_double (const float2 df) {
	return (double)df.y + (double)df.x;
}

/*
#define SPLITTER_DF 536870913.0 //2^29+1
__device__ __forceinline__ double2
double_to_double2
	(const double d)
{
	double t, h, l;
	t = __dmul_rn (SPLITTER_DF, d);
	h = __dsub_rn (t, (__fma_rn (SPLITTER_DF, d, -d)));
	l = __dsub_rn (d, h);
	return make_double2 (h, l);
}

__device__ __forceinline__ float3
double_to_float3
	(const double d)
{
    double2 t, t2;
    t = double_to_double2 (d);
    t2 = double_to_double2 (t.y);
	return make_float3 ((float)t.x, (float)t2.x, (float)t2.y);
    //return make_float3 ((float)d, 0., 0.);
}
*/

// ===================================
// Triple word
// ===================================

/*
void cuVecSum (const int n, const float* x, float* y) 
{
    float s2;
    float s1 = x[n-1];
    for (int i = n-2; i > 0; i--) {
        cuTwoSum (x[i], s1, s2, y[i+1])
        s1 = s2;
    }
    y[0] = s[0];
}

void cuVSEB (const int n, const float* e, float* y) 
{
    float r, epst, eps2;
    int j = 0;
    float eps1 = e[0];
    for (int i = 0; i < n-3; i++) {
        cuTwoSum (eps1, e[i+1], r, epst)
        if (epst != 0) {
            y[j] = r;
            eps2 = epst;
            j++;
        } else {
            eps2 = r;
        }
        eps1 = eps2;
    }
    cuTwoSum (eps1, e[n-1], y[j], y[j+1]);
    for (int k = j+2; k < n-1; k++) {
        y[k] = 0.;
    }
}

void merge3 (const float* a, const float* b, float* z) 
{
    int ia = 0, ib = 0;
    for (int i = 0; i < 6; i++) {
        if (a[ia] > b[ib]) {
            z[i] = a[ia];
            ia++;
        } else {
            z[i] = b[ib];
            ib++;
        }
    }
}

__device__ __forceinline__ float3
cuTWadd
	(const float3 a, const float3 b) {
    float z[6], e[6], c[3];
    merge3 (a, b, z);
    cuVecSum (6, z, e);
    cuVSEB (3, e, c);
	return float3 (c[0], c[1], c[2]);
}

__device__ __forceinline__ float3
cuTWmul
	(const float3 a, const float3 b) {
	float3 c;
    float e1, e2, e3, e4, e5, t2, t3;
	cuTwoProdFMAF (a.x, b.x, c.x, e1);
	cuTwoProdFMAF (a.x, b.y, t2, e2);
	cuTwoProdFMAF (a.y, b.x, t3, e3);
	cuTwoSumF (t2, t3, c.y, e4);
	cuTwoSumF (c.y, e1, c.y, e5);
    c.z = __fadd_rn (__fadd_rn (__fadd_rn (__fmaf_rn (a.z, b.x, e2), __fmaf_rn (a.y, b.y, e3)), __fmaf_rn (a.x, b.z, e4)), e5);
	return c;
}

*/



























#endif
