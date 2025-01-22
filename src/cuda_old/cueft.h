#ifndef CUEFT_H
#define CUEFT_H

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

#endif
