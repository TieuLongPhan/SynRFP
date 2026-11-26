#include <stdint.h>

/* All candidates are ordered by original source position, then panel position. */
void nearest(const uint64_t *bits, const int32_t *density, const int32_t *labels,
             const int32_t *train, int n_train, const int32_t *query, int n_query,
             const int32_t *pred_d, const int32_t *pred_du, int32_t *out) {
    for (int q = 0; q < n_query; q++) {
        const int i = query[q];
        const uint64_t *left = bits + (int64_t)i * 32;
        int best[3] = {-1, -1, -1};
        int num[3] = {-1, -1, -1};
        int den[3] = {1, 1, 1};
        int ties[3] = {0, 0, 0};
        int tie_matches[3] = {0, 0, 0};
        int cross_num[3] = {0, 0, 0};
        int cross_den[3] = {1, 1, 1};
        for (int t = 0; t < n_train; t++) {
            const int j = train[t];
            const uint64_t *right = bits + (int64_t)j * 32;
            int d = 0, u = 0;
            for (int w = 0; w < 16; w++) {
                d += __builtin_popcountll(left[w] & right[w]);
                u += __builtin_popcountll(left[w+16] & right[w+16]);
            }
            int all_num[3] = {d, u, d+u};
            const int nd = density[2*i] + density[2*j] - d;
            const int nu = density[2*i+1] + density[2*j+1] - u;
            int all_den[3] = {nd, nu, nd+nu};
            for (int c = 0; c < 3; c++) {
                if (!all_den[c]) {
                    all_num[c] = 1;
                    all_den[c] = 1;
                }
            }
            for (int c = 0; c < 3; c++) {
                const int diff = all_num[c]*den[c] - num[c]*all_den[c];
                if (diff < 0) continue;
                if (diff > 0) {
                    best[c] = j;
                    num[c] = all_num[c];
                    den[c] = all_den[c];
                    const int other = c == 0 ? 1 : 0;
                    cross_num[c] = all_num[other];
                    cross_den[c] = all_den[other];
                    ties[c] = tie_matches[c] = 0;
                }
                ties[c]++;
                const int prediction = c == 0 ? pred_d[q] : pred_du[q];
                tie_matches[c] += labels[j] == prediction;
            }
        }
        for (int c = 0; c < 3; c++) {
            int32_t *row = out + (q*3 + c)*7;
            row[0] = best[c];
            row[1] = num[c];
            row[2] = den[c];
            row[3] = ties[c];
            row[4] = tie_matches[c];
            row[5] = cross_num[c];
            row[6] = cross_den[c];
        }
    }
}
