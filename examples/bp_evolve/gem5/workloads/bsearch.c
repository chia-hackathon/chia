/* Binary search over a sorted array, repeated.  Chosen because every level of
   the search is a data-dependent conditional branch with roughly even odds and
   no exploitable local pattern -- it is close to the worst case for a history
   predictor, so a design that only wins on loop-heavy code will not win here. */
#include <stdio.h>
#include <stdlib.h>

#define N 65536
#define QUERIES 400000

static int a[N];

int main(void)
{
    for (int i = 0; i < N; i++) a[i] = i * 3;

    unsigned long seed = 12345;
    long found = 0;
    for (int q = 0; q < QUERIES; q++) {
        seed = seed * 1103515245 + 12345;
        int key = (int)((seed >> 16) % (N * 3));
        int lo = 0, hi = N - 1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (a[mid] == key) { found++; break; }
            if (a[mid] < key) lo = mid + 1;
            else hi = mid - 1;
        }
    }
    printf("found %ld\n", found);
    return 0;
}
