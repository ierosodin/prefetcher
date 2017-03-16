#ifndef TRANSPOSE_IMPL
#define TRANSPOSE_IMPL

#ifdef NAIVE
void transpose(int *src, int *dst, int w, int h)
{
    for (int x = 0; x < w; x++)
        for (int y = 0; y < h; y++)
            *(dst + x * h + y) = *(src + y * w + x);
}
#endif

#ifdef SSE
void transpose(int *src, int *dst, int w, int h)
{
    for (int x = 0; x < w; x += 4) {
        for (int y = 0; y < h; y += 4) {
            __m128i I0 = _mm_loadu_si128((__m128i *)(src + (y + 0) * w + x));
            __m128i I1 = _mm_loadu_si128((__m128i *)(src + (y + 1) * w + x));
            __m128i I2 = _mm_loadu_si128((__m128i *)(src + (y + 2) * w + x));
            __m128i I3 = _mm_loadu_si128((__m128i *)(src + (y + 3) * w + x));
            __m128i T0 = _mm_unpacklo_epi32(I0, I1);
            __m128i T1 = _mm_unpacklo_epi32(I2, I3);
            __m128i T2 = _mm_unpackhi_epi32(I0, I1);
            __m128i T3 = _mm_unpackhi_epi32(I2, I3);
            I0 = _mm_unpacklo_epi64(T0, T1);
            I1 = _mm_unpackhi_epi64(T0, T1);
            I2 = _mm_unpacklo_epi64(T2, T3);
            I3 = _mm_unpackhi_epi64(T2, T3);
            _mm_storeu_si128((__m128i *)(dst + ((x + 0) * h) + y), I0);
            _mm_storeu_si128((__m128i *)(dst + ((x + 1) * h) + y), I1);
            _mm_storeu_si128((__m128i *)(dst + ((x + 2) * h) + y), I2);
            _mm_storeu_si128((__m128i *)(dst + ((x + 3) * h) + y), I3);
        }
    }
}
#endif

#ifdef SSE_PREFETCH
void transpose(int *src, int *dst, int w, int h)
{
    for (int x = 0; x < w; x += 4) {
        for (int y = 0; y < h; y += 4) {
#define PFDIST  8
            _mm_prefetch(src+(y + PFDIST + 0) *w + x, _MM_HINT_T1);
            _mm_prefetch(src+(y + PFDIST + 1) *w + x, _MM_HINT_T1);
            _mm_prefetch(src+(y + PFDIST + 2) *w + x, _MM_HINT_T1);
            _mm_prefetch(src+(y + PFDIST + 3) *w + x, _MM_HINT_T1);

            __m128i I0 = _mm_loadu_si128 ((__m128i *)(src + (y + 0) * w + x));
            __m128i I1 = _mm_loadu_si128 ((__m128i *)(src + (y + 1) * w + x));
            __m128i I2 = _mm_loadu_si128 ((__m128i *)(src + (y + 2) * w + x));
            __m128i I3 = _mm_loadu_si128 ((__m128i *)(src + (y + 3) * w + x));
            __m128i T0 = _mm_unpacklo_epi32(I0, I1);
            __m128i T1 = _mm_unpacklo_epi32(I2, I3);
            __m128i T2 = _mm_unpackhi_epi32(I0, I1);
            __m128i T3 = _mm_unpackhi_epi32(I2, I3);
            I0 = _mm_unpacklo_epi64(T0, T1);
            I1 = _mm_unpackhi_epi64(T0, T1);
            I2 = _mm_unpacklo_epi64(T2, T3);
            I3 = _mm_unpackhi_epi64(T2, T3);
            _mm_storeu_si128((__m128i *)(dst + ((x + 0) * h) + y), I0);
            _mm_storeu_si128((__m128i *)(dst + ((x + 1) * h) + y), I1);
            _mm_storeu_si128((__m128i *)(dst + ((x + 2) * h) + y), I2);
            _mm_storeu_si128((__m128i *)(dst + ((x + 3) * h) + y), I3);
        }
    }
}
#endif

#ifdef AVX
void transpose(int *src, int *dst, int w, int h)
{
    for (int x = 0; x < w; x += 8) {
        for (int y = 0; y < h; y += 8) {
            __m256 I0 = _mm256_loadu_ps((src + (y + 0) * w + x));
            __m256 I1 = _mm256_loadu_ps((src + (y + 1) * w + x));
            __m256 I2 = _mm256_loadu_ps((src + (y + 2) * w + x));
            __m256 I3 = _mm256_loadu_ps((src + (y + 3) * w + x));
            __m256 I4 = _mm256_loadu_ps((src + (y + 4) * w + x));
            __m256 I5 = _mm256_loadu_ps((src + (y + 5) * w + x));
            __m256 I6 = _mm256_loadu_ps((src + (y + 6) * w + x));
            __m256 I7 = _mm256_loadu_ps((src + (y + 7) * w + x));
            __m256 T0 = _mm256_unpacklo_ps(I0, I1);
            __m256 T1 = _mm256_unpacklo_ps(I2, I3);
            __m256 T2 = _mm256_unpackhi_ps(I0, I1);
            __m256 T3 = _mm256_unpackhi_ps(I2, I3);
            __m256 T4 = _mm256_unpacklo_ps(I4, I5);
            __m256 T5 = _mm256_unpacklo_ps(I6, I7);
            __m256 T6 = _mm256_unpackhi_ps(I4, I5);
            __m256 T7 = _mm256_unpackhi_ps(I6, I7);
            __m256d D0 = _mm256_castps_pd(T0);
            __m256d D1 = _mm256_castps_pd(T1);
            __m256d D2 = _mm256_castps_pd(T2);
            __m256d D3 = _mm256_castps_pd(T3);
            __m256d D4 = _mm256_castps_pd(T4);
            __m256d D5 = _mm256_castps_pd(T5);
            __m256d D6 = _mm256_castps_pd(T6);
            __m256d D7 = _mm256_castps_pd(T7);
            __m256d J0 = _mm256_unpacklo_pd(D0, D1);
            __m256d J1 = _mm256_unpackhi_pd(D0, D1);
            __m256d J2 = _mm256_unpacklo_pd(D2, D3);
            __m256d J3 = _mm256_unpackhi_pd(D2, D3);
            __m256d J4 = _mm256_unpacklo_pd(D4, D5);
            __m256d J5 = _mm256_unpackhi_pd(D4, D5);
            __m256d J6 = _mm256_unpacklo_pd(D6, D7);
            __m256d J7 = _mm256_unpackhi_pd(D6, D7);
            D0 = _mm256_permute2f128_pd(J0, J4, 0x20);
            D1 = _mm256_permute2f128_pd(J1, J5, 0x20);
            D2 = _mm256_permute2f128_pd(J2, J6, 0x20);
            D3 = _mm256_permute2f128_pd(J3, J7, 0x20);
            D4 = _mm256_permute2f128_pd(J0, J4, 0x31);
            D5 = _mm256_permute2f128_pd(J1, J5, 0x31);
            D6 = _mm256_permute2f128_pd(J2, J6, 0x31);
            D7 = _mm256_permute2f128_pd(J3, J7, 0x31);
            I0 = _mm256_castpd_ps(D0);
            I1 = _mm256_castpd_ps(D1);
            I2 = _mm256_castpd_ps(D2);
            I3 = _mm256_castpd_ps(D3);
            I4 = _mm256_castpd_ps(D4);
            I5 = _mm256_castpd_ps(D5);
            I6 = _mm256_castpd_ps(D6);
            I7 = _mm256_castpd_ps(D7);
            _mm256_store_ps((dst + ((x + 0) * h) + y), I0);
            _mm256_store_ps((dst + ((x + 1) * h) + y), I1);
            _mm256_store_ps((dst + ((x + 2) * h) + y), I2);
            _mm256_store_ps((dst + ((x + 3) * h) + y), I3);
            _mm256_store_ps((dst + ((x + 4) * h) + y), I4);
            _mm256_store_ps((dst + ((x + 5) * h) + y), I5);
            _mm256_store_ps((dst + ((x + 6) * h) + y), I6);
            _mm256_store_ps((dst + ((x + 7) * h) + y), I7);
        }
    }

}
#endif

#endif /* TRANSPOSE_IMPL */
