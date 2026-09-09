/* Dijkstra on a dense grid graph, with a linear scan for the minimum.  The
   scan's "is this the new minimum" branch is unpredictable and frequent, which
   is what makes shortest-path code a standard branch-prediction workload. */
#include <stdio.h>
#include <string.h>

#define V 900
#define INF 1000000000

static int w[V][V];
static int dist[V];
static char done[V];

int main(void)
{
    unsigned long seed = 24680;
    for (int i = 0; i < V; i++)
        for (int j = 0; j < V; j++) {
            seed = seed * 1103515245 + 12345;
            w[i][j] = (i == j) ? 0 : (int)((seed >> 16) % 100 + 1);
        }

    long total = 0;
    for (int src = 0; src < 8; src++) {
        for (int i = 0; i < V; i++) { dist[i] = INF; done[i] = 0; }
        dist[src] = 0;
        for (int it = 0; it < V; it++) {
            int best = -1, bd = INF;
            for (int i = 0; i < V; i++)
                if (!done[i] && dist[i] < bd) { bd = dist[i]; best = i; }
            if (best < 0) break;
            done[best] = 1;
            for (int j = 0; j < V; j++)
                if (!done[j] && dist[best] + w[best][j] < dist[j])
                    dist[j] = dist[best] + w[best][j];
        }
        for (int i = 0; i < V; i++) if (dist[i] < INF) total += dist[i];
    }
    printf("total %ld\n", total);
    return 0;
}
