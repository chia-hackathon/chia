/* Insertion sort over many short runs.  The inner loop's exit is a highly
   biased branch and the comparison is not, so the two levels of a predictor
   are exercised against each other rather than in agreement. */
#include <stdio.h>

#define RUNS 20000
#define LEN 64

static int buf[LEN];

int main(void)
{
    unsigned long seed = 987654321;
    long checksum = 0;
    for (int r = 0; r < RUNS; r++) {
        for (int i = 0; i < LEN; i++) {
            seed = seed * 6364136223846793005UL + 1442695040888963407UL;
            buf[i] = (int)((seed >> 33) & 0xffff);
        }
        for (int i = 1; i < LEN; i++) {
            int key = buf[i], j = i - 1;
            while (j >= 0 && buf[j] > key) { buf[j + 1] = buf[j]; j--; }
            buf[j + 1] = key;
        }
        checksum += buf[0] + buf[LEN - 1];
    }
    printf("checksum %ld\n", checksum);
    return 0;
}
