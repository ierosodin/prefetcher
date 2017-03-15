#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>

#include <xmmintrin.h>

/* provide the implementations of naive_transpose,
 * sse_transpose, sse_prefetch_transpose
 */

#include "impl.c"

static double diff_in_us(struct timespec t1, struct timespec t2)
{
    struct timespec diff;
    if (t2.tv_nsec-t1.tv_nsec < 0) {
        diff.tv_sec  = t2.tv_sec - t1.tv_sec - 1;
        diff.tv_nsec = t2.tv_nsec - t1.tv_nsec + 1000000000;
    } else {
        diff.tv_sec  = t2.tv_sec - t1.tv_sec;
        diff.tv_nsec = t2.tv_nsec - t1.tv_nsec;
    }
    return (diff.tv_sec * 1000000.0 + diff.tv_nsec / 1000.0);
}

int main(int argc, char const *argv[])
{
    int TEST_W = 4096, TEST_H = 4096;
    int args;

    while (-1 != (args = getopt(argc, argv, "w:h:"))) {
        switch (args) {
            case 'w':
                TEST_W = atoi(optarg);
                break;
            case 'h':
                TEST_H = atoi(optarg);
                break;
        }
    }

    struct timespec start, end;
    int *src  = (int *) malloc(sizeof(int) * TEST_W * TEST_H);
    int *out = (int *) malloc(sizeof(int) * TEST_W * TEST_H);

    srand(time(NULL));
    for (int y = 0; y < TEST_H; y++)
        for (int x = 0; x < TEST_W; x++)
            *(src + y * TEST_W + x) = rand();

    clock_gettime(CLOCK_REALTIME, &start);
    transpose(src, out, TEST_W, TEST_H);
    clock_gettime(CLOCK_REALTIME, &end);
    printf("%lf\n", diff_in_us(start, end)/1000000);

    free(out);
    free(src);

    return 0;
}
