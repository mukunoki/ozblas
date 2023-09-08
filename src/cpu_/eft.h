#ifndef EFT_H
#define EFT_H

#include <immintrin.h>

struct double2 {
	double x;
	double y;
};

__inline__ void
TwoSum
	(double a, double b, double &s, double &e)
{
	double v;
	s = a + b;
	v = s - a;
	e = (a - (s - v)) + (b - v);
}

__inline__ void
TwoSum_avx128
	(__m128d a, __m128d b, __m128d &s, __m128d &e)
{
	__m128d v;
	s = _mm_add_pd (a, b);
	v = _mm_sub_pd (s, a);
	e = _mm_add_pd (_mm_sub_pd (a, _mm_sub_pd (s, v)), _mm_sub_pd (b, v));
}

__inline__ void
TwoSum_avx256
	(__m256d a, __m256d b, __m256d &s, __m256d &e)
{
	__m256d v;
	s = _mm256_add_pd (a, b);
	v = _mm256_sub_pd (s, a);
	e = _mm256_add_pd (_mm256_sub_pd (a, _mm256_sub_pd (s, v)), _mm256_sub_pd (b, v));
}

__inline__ void
TwoSum_avx512
	(__m512d a, __m512d b, __m512d &s, __m512d &e)
{
	__m512d v;
	s = _mm512_add_pd (a, b);
	v = _mm512_sub_pd (s, a);
	e = _mm512_add_pd (_mm512_sub_pd (a, _mm512_sub_pd (s, v)), _mm512_sub_pd (b, v));
}

__inline__ void
QuickTwoSum
	(double a, double b, double &s, double &e)
{
	s = a + b;
	e = b - (s - a);
}

__inline__ void
QuickTwoSum_avx128
	(__m128d a, __m128d b, __m128d &s, __m128d &e)
{
	s = _mm_add_pd (a, b);
	e = _mm_sub_pd (b, _mm_sub_pd (s, a));
}

__inline__ void
QuickTwoSum_avx256
	(__m256d a, __m256d b, __m256d &s, __m256d &e)
{
	s = _mm256_add_pd (a, b);
	e = _mm256_sub_pd (b, _mm256_sub_pd (s, a));
}

__inline__ void
TwoProdFMA
	(double a, double b, double &p, double &e)
{
	p = a * b;
	e = fma (a, b, -p);
}

__inline__ void
TwoProdFMA_avx256
	(__m256d a, __m256d b, __m256d &p, __m256d &e)
{
	p = _mm256_mul_pd (a, b);
	e = _mm256_fmadd_pd (a, b, -p);
}

__inline__ void
TwoProdFMA_avx512
	(__m512d a, __m512d b, __m512d &p, __m512d &e)
{
	p = _mm512_mul_pd (a, b);
	e = _mm512_fmadd_pd (a, b, -p);
}

__inline__ void
Dot2i
	(const double a, const double b, double2 &c)
{
	double h, r, q;
	TwoProdFMA (a, b, h, r);
	TwoSum (c.x, h, c.x, q);
	c.y = c.y + (q + r);
}

__inline__ void
Dot2_avx256
	(const __m256d a, const __m256d b, __m256d &ch, __m256d &cl)
{
	__m256d h, r, q;
	TwoProdFMA_avx256 (a, b, h, r);
	TwoSum_avx256 (ch, h, ch, q);
	cl = _mm256_add_pd (cl, _mm256_add_pd (q, r));
}

__inline__ void
Dot2_avx512
	(const __m512d a, const __m512d b, __m512d &ch, __m512d &cl)
{
	__m512d h, r, q;
	TwoProdFMA_avx512 (a, b, h, r);
	TwoSum_avx512 (ch, h, ch, q);
	cl = _mm512_add_pd (cl, _mm512_add_pd (q, r));
}

__inline__ double2
QuadAdd
	(const double2 a, const double2 b) {
	double2 t, c;
	TwoSum (a.x, b.x, t.x, t.y);
	t.y = t.y + (a.y + b.y);
	QuickTwoSum (t.x, t.y, c.x, c.y);
	return c;
}

__inline__ void
QuadAdd_
	(const double ax, const double ay, const double bx, const double by, double &cx, double &cy) {
	double tx, ty;
	TwoSum (ax, bx, tx, ty);
	ty = ty + (ay + by);
	QuickTwoSum (tx, ty, cx, cy);
}

__inline__ void
QuadAdd_avx128
	(const __m128d aHi, const __m128d aLo, const __m128d bHi, const __m128d bLo, __m128d &cHi, __m128d &cLo) {
	__m128d tHi, tLo;
	TwoSum_avx128 (aHi, bHi, tHi, tLo);
	tLo = _mm_add_pd (tLo, _mm_add_pd (aLo, bLo));
	QuickTwoSum_avx128 (tHi, tLo, cHi, cLo);
}

__inline__ void
QuadAdd_avx256
	(const __m256d aHi, const __m256d aLo, const __m256d bHi, const __m256d bLo, __m256d &cHi, __m256d &cLo) {
	__m256d tHi, tLo;
	TwoSum_avx256 (aHi, bHi, tHi, tLo);
	tLo = _mm256_add_pd (tLo, _mm256_add_pd (aLo, bLo));
	QuickTwoSum_avx256 (tHi, tLo, cHi, cLo);
}

__inline__ double2
QuadMul
	(const double a, const double2 b) {
	double2 t, c;
	TwoProdFMA (a, b.x, t.x, t.y);
	t.y = fma (a, b.y, t.y);
	QuickTwoSum (t.x, t.y, c.x, c.y);
	return c;
}

__inline__ double2
QuadMul
	(const double a, const double b) {
	double2 c;
	TwoProdFMA (a, b, c.x, c.y);
	return c;
}

#endif
