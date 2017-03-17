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
            I0 = _mm256_shuffle_ps(T0, T1, _MM_SHUFFLE(1,0,1,0));
            I1 = _mm256_shuffle_ps(T0, T1, _MM_SHUFFLE(3,2,3,2));
            I2 = _mm256_shuffle_ps(T2, T3, _MM_SHUFFLE(1,0,1,0));
            I3 = _mm256_shuffle_ps(T2, T3, _MM_SHUFFLE(3,2,3,2));
            I4 = _mm256_shuffle_ps(T4, T5, _MM_SHUFFLE(1,0,1,0));
            I5 = _mm256_shuffle_ps(T4, T5, _MM_SHUFFLE(3,2,3,2));
            I6 = _mm256_shuffle_ps(T6, T6, _MM_SHUFFLE(1,0,1,0));
            I7 = _mm256_shuffle_ps(T7, T7, _MM_SHUFFLE(3,2,3,2));
            T0 = _mm256_permute2f128_ps(I0, I4, 0x20);
            T1 = _mm256_permute2f128_ps(I1, I5, 0x20);
            T2 = _mm256_permute2f128_ps(I2, I6, 0x20);
            T3 = _mm256_permute2f128_ps(I3, I7, 0x20);
            T4 = _mm256_permute2f128_ps(I0, I4, 0x31);
            T5 = _mm256_permute2f128_ps(I1, I5, 0x31);
            T6 = _mm256_permute2f128_ps(I2, I6, 0x31);
            T7 = _mm256_permute2f128_ps(I3, I7, 0x31);
            _mm256_store_ps((dst + ((x + 0) * h) + y), T0);
            _mm256_store_ps((dst + ((x + 1) * h) + y), T1);
            _mm256_store_ps((dst + ((x + 2) * h) + y), T2);
            _mm256_store_ps((dst + ((x + 3) * h) + y), T3);
            _mm256_store_ps((dst + ((x + 4) * h) + y), T4);
            _mm256_store_ps((dst + ((x + 5) * h) + y), T5);
            _mm256_store_ps((dst + ((x + 6) * h) + y), T6);
            _mm256_store_ps((dst + ((x + 7) * h) + y), T7);
        }
    }

}
#endif

#endif /* TRANSPOSE_IMPL */
