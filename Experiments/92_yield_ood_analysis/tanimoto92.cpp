// Exact maximum binary Tanimoto with a population-count upper bound.
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace {
template <int Width>
int overlap(const uint64_t* a, const uint64_t* b, int words) {
    int total = 0;
    const int length = Width ? Width : words;
    for (int i = 0; i < length; ++i)
        total += __builtin_popcountll(a[i] & b[i]);
    return total;
}

template <int Width>
void nearest(const uint64_t* query, const int32_t* query_sizes, int nq,
             const uint64_t* train, const int32_t* stops, int words,
             int bit_count, float* result) {
    #pragma omp parallel for schedule(dynamic, 32)
    for (int i = 0; i < nq; ++i) {
        const int q = query_sizes[i];
        if (q == 0) {
            result[i] = stops[1] > stops[0] ? 1.0f : 0.0f;
            continue;
        }
        int lo = q, hi = q + 1;
        int best_overlap = 0, best_union = 1;
        const auto* left = query + std::size_t(i) * words;
        while ((lo > 0 || hi <= bit_count) && best_overlap < best_union) {
            // The best possible Tanimoto is min(q,t)/max(q,t). Walk
            // count groups in decreasing upper bound and stop when beaten.
            const int t = (hi > bit_count || (lo > 0 && lo * hi >= q * q))
                          ? lo-- : hi++;
            if (std::min(q,t) * best_union <= best_overlap * std::max(q,t))
                break;
            for (int j = stops[t]; j < stops[t+1]; ++j) {
                const auto* right = train + std::size_t(j) * words;
                const int shared = overlap<Width>(left,right,words);
                const int size = q + t - shared;
                if (shared * best_union > best_overlap * size) {
                    best_overlap = shared;
                    best_union = size;
                }
                if (shared == size) break;
            }
        }
        result[i] = float(best_overlap) / best_union;
    }
}
}

extern "C" void max_tanimoto(const uint64_t* query, const int32_t* query_sizes, int nq,
                              const uint64_t* train, const int32_t* stops, int words,
                              int bit_count, float* result) {
    if (words == 32)
        nearest<32>(query,query_sizes,nq,train,stops,words,bit_count,result);
    else if (words == 64)
        nearest<64>(query,query_sizes,nq,train,stops,words,bit_count,result);
    else
        nearest<0>(query,query_sizes,nq,train,stops,words,bit_count,result);
}
