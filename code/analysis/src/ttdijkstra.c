/*
 * ttdijkstra.c — bounded multi-source Dijkstra (Dial bucket queue) on a
 * 0.01-degree global friction grid.
 *
 * Friction units: minutes per metre (Weiss et al. 2020 MAP surface).
 * Output: cumulative travel time in minutes, capped at maxmin; unreached = 1e9.
 *
 * Usage:
 *   ttdijkstra friction.bin sources.bin out.bin nrows ncols maxmin [mode]
 *     mode 0 = 8-neighbour (default)
 *     mode 1 = 4-neighbour (sensitivity)
 *
 * friction.bin : float32, nrows*ncols, row-major, north-up
 * sources.bin  : uint8,   nrows*ncols, 1 = source cell
 * out.bin      : float32, nrows*ncols
 *
 * Compile: gcc -O3 -march=native -funroll-loops -o ttdijkstra ttdijkstra.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define BIG 1e9f

static double dx_m[40000];   /* per-row east-west metres */
static double dy_m[40000];   /* per-row north-south metres */
static double dd_m[40000];   /* per-row diagonal metres */

typedef struct {
    int32_t *v;
    size_t n, cap;
} bucket;

static void bpush(bucket *b, int32_t x) {
    if (b->n == b->cap) {
        b->cap = b->cap ? b->cap * 2 : 4096;
        b->v = (int32_t *)realloc(b->v, b->cap * sizeof(int32_t));
        if (!b->v) { fprintf(stderr, "OOM bucket\n"); exit(1); }
    }
    b->v[b->n++] = x;
}

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s friction.bin sources.bin out.bin nrows ncols maxmin [mode]\n", argv[0]);
        return 2;
    }
    const char *fric_path = argv[1], *src_path = argv[2], *out_path = argv[3];
    long nrows = atol(argv[4]), ncols = atol(argv[5]);
    double maxmin = atof(argv[6]);
    int mode = (argc > 7) ? atoi(argv[7]) : 0;
    long N = nrows * ncols;
    fprintf(stderr, "grid %ld x %ld = %ld cells, maxmin=%.0f, mode=%d\n", nrows, ncols, N, maxmin, mode);

    float *fric = (float *)malloc((size_t)N * sizeof(float));
    float *dist = (float *)malloc((size_t)N * sizeof(float));
    uint8_t *src = (uint8_t *)malloc((size_t)N);
    if (!fric || !dist || !src) { fprintf(stderr, "OOM main arrays\n"); return 1; }

    FILE *f = fopen(fric_path, "rb");
    if (!f) { perror("friction"); return 1; }
    if (fread(fric, sizeof(float), (size_t)N, f) != (size_t)N) { fprintf(stderr, "short read friction\n"); return 1; }
    fclose(f);
    f = fopen(src_path, "rb");
    if (!f) { perror("sources"); return 1; }
    if (fread(src, 1, (size_t)N, f) != (size_t)N) { fprintf(stderr, "short read sources\n"); return 1; }
    fclose(f);

    for (long i = 0; i < N; i++) dist[i] = BIG;

    /* per-row distances: 30 arc-second grid (1/120 deg), north-up from lat_top */
    const double PIX = 1.0 / 120.0;
    for (long r = 0; r < nrows; r++) {
        double lat = 85.0 - PIX * ((double)r + 0.5);
        double cl = cos(lat * M_PI / 180.0);
        if (cl < 1e-6) cl = 1e-6;
        dx_m[r] = 111320.0 * PIX * cl;
        dy_m[r] = 110574.0 * PIX;
        dd_m[r] = sqrt(dx_m[r] * dx_m[r] + dy_m[r] * dy_m[r]);
    }

    long nb = (long)maxmin + 2;
    bucket *bk = (bucket *)calloc((size_t)nb, sizeof(bucket));
    if (!bk) { fprintf(stderr, "OOM buckets\n"); return 1; }

    long nsrc = 0;
    for (long i = 0; i < N; i++) {
        if (src[i]) {
            if (fric[i] == -9999.0f || fric[i] < 0.0f) continue;
            dist[i] = 0.0f;
            bpush(&bk[0], (int32_t)i);
            nsrc++;
        }
    }
    fprintf(stderr, "sources: %ld\n", nsrc);

    /* neighbour offsets */
    int off_r[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    int off_c[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    int nmax = (mode == 1) ? 4 : 8;
    int off4_r[4] = {-1, 0, 0, 1};
    int off4_c[4] = {0, -1, 1, 0};

    long popped = 0, relaxed = 0, improved = 0;
    for (long t = 0; t < nb; t++) {
        bucket *b = &bk[t];
        size_t k = 0;
        while (k < b->n) {
            int32_t c = b->v[k++];
            float dc = dist[c];
            if (dc < (float)t) continue;          /* stale: already settled at an earlier bucket */
            popped++;
            long cr = c / ncols, cc = c % ncols;
            float fc = fric[c];
            for (int q = 0; q < nmax; q++) {
                int dr = (mode == 1) ? off4_r[q] : off_r[q];
                int dcc = (mode == 1) ? off4_c[q] : off_c[q];
                long nr = cr + dr, nc = cc + dcc;
                if (nr < 0 || nr >= nrows || nc < 0 || nc >= ncols) continue;
                long ni = nr * ncols + nc;
                float fn = fric[ni];
                if (fn == -9999.0f || fn < 0.0f) continue;
                double step = (dr != 0 && dcc != 0) ? dd_m[cr] : (dr == 0 ? dx_m[cr] : dy_m[cr]);
                float nd = dc + (float)(0.5 * (fc + fn) * step);
                relaxed++;
                if (nd < dist[ni] && nd <= (float)maxmin) {
                    dist[ni] = nd;
                    long bt = (long)nd;
                    if (bt >= nb) bt = nb - 1;
                    bpush(&bk[bt], (int32_t)ni);
                    improved++;
                }
            }
        }
        free(b->v); b->v = NULL; b->n = b->cap = 0;
        if ((t % 30) == 0)
            fprintf(stderr, "  t=%ld popped=%ld improved=%ld\n", t, popped, improved);
    }

    FILE *o = fopen(out_path, "wb");
    if (!o) { perror("out"); return 1; }
    fwrite(dist, sizeof(float), (size_t)N, o);
    fclose(o);
    fprintf(stderr, "done. popped=%ld relaxed=%ld improved=%ld\n", popped, relaxed, improved);
    return 0;
}
