#include <stddef.h>
#include <stdint.h>

void squared_distances(const uint64_t *rows, size_t n_rows,
                       const uint64_t *candidates, size_t n_candidates,
                       float weight_sq, float *distances) {
    for (size_t c = 0; c < n_candidates; c++) {
        const uint64_t *candidate = candidates + 32*c;
        for (size_t i = 0; i < n_rows; i++) {
            const uint64_t *row = rows + 32*i;
            int delta = 0;
            int pooled = 0;
            for (int j = 0; j < 16; j++) {
                delta += __builtin_popcountll(row[j] ^ candidate[j]);
                pooled += __builtin_popcountll(row[j+16] ^ candidate[j+16]);
            }
            distances[c*n_rows+i] = (float)delta + weight_sq*(float)pooled;
        }
    }
}
