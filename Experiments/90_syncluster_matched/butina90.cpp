// Label-free replay of RDKit Butina with reordering and descending index ties.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {
constexpr int bit_word_size = 64;
constexpr int resolution = 1000;

std::size_t lower_start(std::size_t i) { return i * (i - 1) / 2; }

uint16_t distance_code(int intersection, int bit_union) {
    const double distance = bit_union ? 1.0 - double(intersection) / bit_union : 0.0;
    int threshold = int(resolution * distance);
    // Compare with the exact binary64 cutoff used by native Python.
    if (double(threshold) / resolution < distance) ++threshold;
    if (threshold > 0 && double(threshold - 1) / resolution >= distance) --threshold;
    return uint16_t(threshold);
}

void assign_bit(uint64_t* row, int bit) {
    row[bit / bit_word_size] |= uint64_t(1) << (bit % bit_word_size);
}

void remove_bit(uint64_t* row, int bit) {
    row[bit / bit_word_size] &= ~(uint64_t(1) << (bit % bit_word_size));
}
}

extern "C" void make_distances(const uint64_t* bits, int n, int words, uint16_t* codes) {
    std::vector<int> counts(n);
    for (int i = 0; i < n; ++i) {
        for (int w = 0; w < words; ++w)
            counts[i] += __builtin_popcountll(bits[std::size_t(i) * words + w]);
    }
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 1; i < n; ++i) {
        const auto* left = bits + std::size_t(i) * words;
        for (int j = 0; j < i; ++j) {
            const auto* right = bits + std::size_t(j) * words;
            int intersection = 0;
            for (int w = 0; w < words; ++w)
                intersection += __builtin_popcountll(left[w] & right[w]);
            codes[lower_start(i) + j] = distance_code(intersection, counts[i] + counts[j] - intersection);
        }
    }
}

extern "C" int cluster_distances(const uint16_t* codes, int n, int threshold, int* labels, int* centers) {
    std::fill(labels, labels + n, -1);
    if (threshold == resolution) {
        std::fill(labels, labels + n, 0);
        centers[0] = n - 1;
        return 1;
    }
    const int words = (n + bit_word_size - 1) / bit_word_size;
    std::vector<uint64_t> neighborhood(std::size_t(n) * words);
    std::vector<uint64_t> unseen(words, 0);
    std::vector<int> degree(n, 0);
    for (int i = 0; i < n; ++i) {
        assign_bit(unseen.data(), i);
        auto* row = neighborhood.data() + std::size_t(i) * words;
        const auto* distances = codes + lower_start(i);
        for (int j = 0; j < i; ++j) {
            if (distances[j] > threshold) continue;
            assign_bit(row, j);
            assign_bit(neighborhood.data() + std::size_t(j) * words, i);
            ++degree[i];
            ++degree[j];
        }
    }

    int n_clusters = 0;
    std::vector<int> members;
    while (true) {
        int centroid = -1;
        int max_degree = 0;
        for (int i = n - 1; i >= 0; --i) {
            if (labels[i] < 0 && degree[i] > max_degree) {
                centroid = i;
                max_degree = degree[i];
            }
        }
        if (centroid < 0) break;

        centers[n_clusters] = centroid;
        members.clear();
        members.push_back(centroid);
        remove_bit(unseen.data(), centroid);
        const auto* row = neighborhood.data() + std::size_t(centroid) * words;
        for (int w = 0; w < words; ++w) {
            uint64_t candidates = row[w] & unseen[w];
            unseen[w] &= ~candidates;
            while (candidates) {
                members.push_back(w * bit_word_size + __builtin_ctzll(candidates));
                candidates &= candidates - 1;
            }
        }
        for (int member : members) {
            labels[member] = n_clusters;
            const auto* neighbors = neighborhood.data() + std::size_t(member) * words;
            for (int w = 0; w < words; ++w) {
                uint64_t remaining = neighbors[w] & unseen[w];
                while (remaining) {
                    --degree[w * bit_word_size + __builtin_ctzll(remaining)];
                    remaining &= remaining - 1;
                }
            }
        }
        ++n_clusters;
    }
    for (int i = n - 1; i >= 0; --i) {
        if (labels[i] >= 0) continue;
        centers[n_clusters] = i;
        labels[i] = n_clusters++;
    }
    return n_clusters;
}
