#pragma once
#include "core/common.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <random>
#include <fstream>
#include <limits>

#if HAS_AOCL_BLAS
#include <cblas.h>
#endif

#if HAS_SUPERKMEANS
#include "superkmeans/superkmeans.h"
#include "superkmeans/hierarchical_superkmeans.h"
#endif

namespace score {

namespace impl {

inline int capped_assign_block_size(int num_centroids, size_t scratch_cap_bytes = 128ull * 1024ull * 1024ull) {
    constexpr int kDefaultAssignBlockSize = 50000;

    if (num_centroids <= 0) return 1;

    size_t bytes_per_row = (size_t)num_centroids * sizeof(float);
    size_t capped_rows = std::max<size_t>(1, scratch_cap_bytes / bytes_per_row);
    return (int)std::min<size_t>(kDefaultAssignBlockSize, capped_rows);
}

} // namespace impl

// Per-subspace Inverted File Index (flat K-means)
// Replaces IMI's Cartesian product with a single flat K-means clustering.
// Centroid ranking uses BLAS/SGEMM during build-time batch assignment and
// prepared SoA SIMD for query-time dynamic activation.
// Ranking uses score(c) = 0.5*||c||² - <x,c>; the omitted 0.5*||x||² term
// is constant across centroids, so ordering is identical to squared L2 distance.
class IVFIndex {
public:
    IVFIndex() = default;

    void init(int subspace_dim, int num_centroids, int subspace_id) {
        sub_dim_ = subspace_dim;
        K_ = num_centroids;
        subspace_id_ = subspace_id;
        centroids_.resize((size_t)K_ * sub_dim_);
        centroid_rank_bias_.resize(K_);
        cell_sizes_.resize(K_, 0);
        cell_offsets_.resize(K_ + 1, 0);
    }

    // Train K centroids. Uses HSKM with n_threads=1 to eliminate OMP barrier overhead
    // (BLIS sgemm_ handles multi-threading internally via its own pthread pool).
    // use_internal_rotation=true: let SuperKMeans apply its own DCT rotation for better
    // K-means convergence, then unrotate centroids back to caller's space.
    // Use when passing raw (un-pre-rotated) data and need raw-space centroids back.
    void train(const float* subspace_data, int n, int max_iter = 10, uint64_t seed = 42,
               bool use_internal_rotation = false) {
#if HAS_SUPERKMEANS
        train_hskm(subspace_data, n, max_iter, seed, use_internal_rotation);
#else
        train_blas(subspace_data, n, max_iter, seed);
#endif
        precompute_centroid_rank_bias();
    }

    // BLAS-accelerated K-means: blocked SGEMM assignment + centroid update
    // Features: early termination on convergence, pre-allocated buffers,
    // SIMD-accelerated argmin (AVX-512).
    void train_blas(const float* data, int n, int max_iter = 10, uint64_t seed = 42) {
        init_centroids_forgy(data, n, seed);
        precompute_centroid_rank_bias();

        std::vector<int> assignments(n, 0);

        // Large blocks amortize BLIS SGEMM call overhead (~100μs/call).
        // BLIS handles internal cache blocking; our block_size is only the outer tile.
        int block_size = 50000;

        // Pre-allocate reusable buffers (avoid per-iteration allocation)
        std::vector<float> ip_block((size_t)block_size * K_);
        int nt = omp_get_max_threads();
        std::vector<float> partial_centroids_buf((size_t)nt * K_ * sub_dim_);
        std::vector<int> partial_counts_buf((size_t)nt * K_);
        std::vector<float> old_centroids((size_t)K_ * sub_dim_);

        for (int iter = 0; iter < max_iter; iter++) {
            // Save old centroids for convergence check
            std::copy(centroids_.begin(), centroids_.end(), old_centroids.begin());

            // Block-based assignment on training data
            for (int b = 0; b < n; b += block_size) {
                int block_n = std::min(block_size, n - b);
                const float* block_data = data + (size_t)b * sub_dim_;

#if HAS_AOCL_BLAS
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            block_n, K_, sub_dim_, 1.0f,
                            block_data, sub_dim_,
                            centroids_.data(), sub_dim_,
                            0.0f, ip_block.data(), K_);
#else
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < block_n; i++) {
                    for (int c = 0; c < K_; c++) {
                        float dot = 0;
                        const float* pt = block_data + (size_t)i * sub_dim_;
                        const float* ct = centroids_.data() + (size_t)c * sub_dim_;
                        for (int d = 0; d < sub_dim_; d++)
                            dot += pt[d] * ct[d];
                        ip_block[(size_t)i * K_ + c] = dot;
                    }
                }
#endif
                // Argmin: find nearest centroid per vector.
                // score(c) = 0.5*||c||² - <x,c> preserves squared-L2 ordering.
                for (int i = 0; i < block_n; i++) {
                    int gi = b + i;
                    const float* ip_row = ip_block.data() + (size_t)i * K_;
#if defined(__AVX512F__)
                    // AVX-512: process 16 centroids per iteration, ~16x fewer comparisons
                    __m512 best_dist = _mm512_set1_ps(std::numeric_limits<float>::max());
                    __m512i best_idx = _mm512_setzero_si512();

                    int c = 0;
                    for (; c + 16 <= K_; c += 16) {
                        __m512 bias = _mm512_loadu_ps(centroid_rank_bias_.data() + c);
                        __m512 ip = _mm512_loadu_ps(ip_row + c);
                        __m512 dist = _mm512_sub_ps(bias, ip);
                        __mmask16 lt = _mm512_cmp_ps_mask(dist, best_dist, _CMP_LT_OQ);
                        best_dist = _mm512_mask_mov_ps(best_dist, lt, dist);
                        best_idx = _mm512_mask_mov_epi32(best_idx, lt,
                                    _mm512_set1_epi32(c));
                        // Encode lane offset: add [0,1,...,15] to base index
                        best_idx = _mm512_mask_add_epi32(best_idx, lt,
                                    _mm512_set1_epi32(c),
                                    _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15));
                    }
                    // Horizontal reduction: find min across 16 lanes
                    float dists_arr[16]; int idx_arr[16];
                    _mm512_storeu_ps(dists_arr, best_dist);
                    _mm512_storeu_si512(idx_arr, best_idx);
                    float best = dists_arr[0]; int best_c = idx_arr[0];
                    for (int j = 1; j < 16; j++) {
                        if (dists_arr[j] < best) { best = dists_arr[j]; best_c = idx_arr[j]; }
                    }
                    // Handle tail
                    for (; c < K_; c++) {
                        float dist = centroid_rank_score(c, ip_row[c]);
                        if (dist < best) { best = dist; best_c = c; }
                    }
#else
                    float best = std::numeric_limits<float>::max();
                    int best_c = 0;
                    for (int c = 0; c < K_; c++) {
                        float dist = centroid_rank_score(c, ip_row[c]);
                        if (dist < best) { best = dist; best_c = c; }
                    }
#endif
                    assignments[gi] = best_c;
                }
            }

            // Centroid update with pre-allocated buffers
            update_centroids_prealloced(data, assignments.data(), n,
                                        partial_centroids_buf, partial_counts_buf);
            precompute_centroid_rank_bias();

            // Early termination: check max centroid drift
            float max_drift = 0;
            for (int c = 0; c < K_; c++) {
                float drift = 0;
                const float* nc = centroids_.data() + (size_t)c * sub_dim_;
                const float* oc = old_centroids.data() + (size_t)c * sub_dim_;
                for (int d = 0; d < sub_dim_; d++) {
                    float diff = nc[d] - oc[d];
                    drift += diff * diff;
                }
                if (drift > max_drift) max_drift = drift;
            }
            if (max_drift < 1e-6f) break;
        }
    }

    // Assign a single subspace vector to nearest centroid
    int assign(const float* subspace_vec) const {
        float best_score = std::numeric_limits<float>::max();
        int best_c = 0;
        for (int c = 0; c < K_; c++) {
            float dot = 0;
            const float* cent = centroids_.data() + (size_t)c * sub_dim_;
            for (int d = 0; d < sub_dim_; d++) {
                dot += subspace_vec[d] * cent[d];
            }
            float score = centroid_rank_score(c, dot);
            if (score < best_score) { best_score = score; best_c = c; }
        }
        return best_c;
    }

    // Batch assign using BLAS SGEMM (blocked to avoid OOM on large N)
    void assign_batch_strided_with_scratch_cap(const float* subspace_data, int row_stride,
                                               int* cell_ids, int n,
                                               size_t scratch_cap_bytes) const {
#if HAS_AOCL_BLAS
        // Cap scratch per subspace so concurrent Step 6 workers do not each materialize
        // a ~500MB ip_block at K=2500.
        int block_size = impl::capped_assign_block_size(K_, scratch_cap_bytes);

        // Blocked SGEMM + argmin
        std::vector<float> ip_block((size_t)block_size * K_);
        bool pack_strided_input = row_stride != sub_dim_ && sub_dim_ >= 32;
        std::vector<float> packed_block;
        if (pack_strided_input) packed_block.resize((size_t)block_size * (size_t)sub_dim_);
        for (int b = 0; b < n; b += block_size) {
            int block_n = std::min(block_size, n - b);
            const float* block_data = subspace_data + (size_t)b * row_stride;
            int block_lda = row_stride;
            if (pack_strided_input) {
                for (int i = 0; i < block_n; ++i) {
                    std::memcpy(packed_block.data() + (size_t)i * (size_t)sub_dim_,
                                block_data + (size_t)i * (size_t)row_stride,
                                (size_t)sub_dim_ * sizeof(float));
                }
                block_data = packed_block.data();
                block_lda = sub_dim_;
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        block_n, K_, sub_dim_, 1.0f,
                        block_data, block_lda,
                        centroids_.data(), sub_dim_,
                        0.0f, ip_block.data(), K_);

            for (int i = 0; i < block_n; i++) {
                int gi = b + i;
                const float* ip_row = ip_block.data() + (size_t)i * K_;
#if defined(__AVX512F__)
                __m512 best_dist = _mm512_set1_ps(std::numeric_limits<float>::max());
                __m512i best_idx = _mm512_setzero_si512();
                const __m512i lane_offsets = _mm512_setr_epi32(
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                int c = 0;
                for (; c + 16 <= K_; c += 16) {
                    __m512 bias = _mm512_loadu_ps(centroid_rank_bias_.data() + c);
                    __m512 ip = _mm512_loadu_ps(ip_row + c);
                    __m512 dist = _mm512_sub_ps(bias, ip);
                    __mmask16 lt = _mm512_cmp_ps_mask(dist, best_dist, _CMP_LT_OQ);
                    best_dist = _mm512_mask_mov_ps(best_dist, lt, dist);
                    __m512i idx = _mm512_add_epi32(_mm512_set1_epi32(c), lane_offsets);
                    best_idx = _mm512_mask_mov_epi32(best_idx, lt, idx);
                }

                float dists_arr[16];
                int idx_arr[16];
                _mm512_storeu_ps(dists_arr, best_dist);
                _mm512_storeu_si512(idx_arr, best_idx);
                float best = dists_arr[0];
                int best_c = idx_arr[0];
                for (int j = 1; j < 16; ++j) {
                    if (dists_arr[j] < best ||
                        (dists_arr[j] == best && idx_arr[j] < best_c)) {
                        best = dists_arr[j];
                        best_c = idx_arr[j];
                    }
                }
                for (; c < K_; ++c) {
                    float dist = centroid_rank_score(c, ip_row[c]);
                    if (dist < best || (dist == best && c < best_c)) { best = dist; best_c = c; }
                }
#else
                float best = std::numeric_limits<float>::max();
                int best_c = 0;
                for (int c = 0; c < K_; c++) {
                    float dist = centroid_rank_score(c, ip_row[c]);
                    if (dist < best) { best = dist; best_c = c; }
                }
#endif
                cell_ids[gi] = best_c;
            }
        }
#else
        #pragma omp parallel for schedule(dynamic, 256)
        for (int i = 0; i < n; i++) {
            cell_ids[i] = assign(subspace_data + (size_t)i * row_stride);
        }
#endif
    }

    void assign_batch_strided_simd_soa(const float* subspace_data, int row_stride,
                                       int* cell_ids, int n,
                                       bool parallel = true) const {
        if (n <= 0) return;
        bool use_soa = simd_prepared_ && centroids_soa_.size() == (size_t)K_ * (size_t)sub_dim_;
        #pragma omp parallel for schedule(static) if(parallel)
        for (int i = 0; i < n; ++i) {
            const float* row = subspace_data + (size_t)i * (size_t)row_stride;
#if defined(__AVX512F__)
            if (use_soa) {
                __m512 best_dist = _mm512_set1_ps(std::numeric_limits<float>::max());
                __m512i best_idx = _mm512_setzero_si512();
                const __m512i lane_offsets = _mm512_setr_epi32(
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                int c = 0;
                for (; c + 16 <= K_; c += 16) {
                    __m512 dot = _mm512_setzero_ps();
                    for (int d = 0; d < sub_dim_; ++d) {
                        __m512 x = _mm512_set1_ps(row[d]);
                        __m512 cent = _mm512_loadu_ps(centroids_soa_.data() + (size_t)d * (size_t)K_ + c);
                        dot = _mm512_add_ps(dot, _mm512_mul_ps(x, cent));
                    }
                    __m512 dist = _mm512_sub_ps(_mm512_loadu_ps(centroid_rank_bias_.data() + c), dot);
                    __mmask16 lt = _mm512_cmp_ps_mask(dist, best_dist, _CMP_LT_OQ);
                    best_dist = _mm512_mask_mov_ps(best_dist, lt, dist);
                    __m512i idx = _mm512_add_epi32(_mm512_set1_epi32(c), lane_offsets);
                    best_idx = _mm512_mask_mov_epi32(best_idx, lt, idx);
                }

                float dists_arr[16];
                int idx_arr[16];
                _mm512_storeu_ps(dists_arr, best_dist);
                _mm512_storeu_si512(idx_arr, best_idx);
                float best = dists_arr[0];
                int best_c = idx_arr[0];
                for (int j = 1; j < 16; ++j) {
                    if (dists_arr[j] < best ||
                        (dists_arr[j] == best && idx_arr[j] < best_c)) {
                        best = dists_arr[j];
                        best_c = idx_arr[j];
                    }
                }
                for (; c < K_; ++c) {
                    float dot = 0.0f;
                    for (int d = 0; d < sub_dim_; ++d) {
                        dot += row[d] * centroids_soa_[(size_t)d * (size_t)K_ + (size_t)c];
                    }
                    float dist = centroid_rank_score(c, dot);
                    if (dist < best) {
                        best = dist;
                        best_c = c;
                    }
                }
                cell_ids[i] = best_c;
                continue;
            }
#endif
            float best = std::numeric_limits<float>::max();
            int best_c = 0;
            for (int c = 0; c < K_; ++c) {
                float dot = 0.0f;
                if (use_soa) {
                    for (int d = 0; d < sub_dim_; ++d) {
                        dot += row[d] * centroids_soa_[(size_t)d * (size_t)K_ + (size_t)c];
                    }
                } else {
                    const float* cent = centroids_.data() + (size_t)c * (size_t)sub_dim_;
                    for (int d = 0; d < sub_dim_; ++d) {
                        dot += row[d] * cent[d];
                    }
                }
                float dist = centroid_rank_score(c, dot);
                if (dist < best) {
                    best = dist;
                    best_c = c;
                }
            }
            cell_ids[i] = best_c;
        }
    }

    void assign_batch_strided_simd_points16(const float* subspace_data, int row_stride,
                                            int* cell_ids, int n) const {
        if (n <= 0) return;
#if defined(__AVX512F__)
        if (sub_dim_ > 64) {
            #pragma omp parallel for schedule(dynamic, 256)
            for (int i = 0; i < n; ++i) {
                cell_ids[i] = assign(subspace_data + (size_t)i * (size_t)row_stride);
            }
            return;
        }

        constexpr int kTileRows = 16;
        int full_tiles = n / kTileRows;
        #pragma omp parallel for schedule(static)
        for (int tile = 0; tile < full_tiles; ++tile) {
            int base = tile * kTileRows;
            alignas(64) float x_tile[64 * kTileRows];
            for (int d = 0; d < sub_dim_; ++d) {
                float* dst = x_tile + (size_t)d * kTileRows;
                for (int lane = 0; lane < kTileRows; ++lane) {
                    dst[lane] = subspace_data[(size_t)(base + lane) * (size_t)row_stride + (size_t)d];
                }
            }

            __m512 best_dist = _mm512_set1_ps(std::numeric_limits<float>::max());
            __m512i best_idx = _mm512_setzero_si512();
            for (int c = 0; c < K_; ++c) {
                const float* cent = centroids_.data() + (size_t)c * (size_t)sub_dim_;
                __m512 dot = _mm512_setzero_ps();
                for (int d = 0; d < sub_dim_; ++d) {
                    __m512 x = _mm512_load_ps(x_tile + (size_t)d * kTileRows);
                    __m512 cv = _mm512_set1_ps(cent[d]);
                    dot = _mm512_add_ps(dot, _mm512_mul_ps(x, cv));
                }
                __m512 dist = _mm512_sub_ps(_mm512_set1_ps(centroid_rank_bias_[c]), dot);
                __mmask16 lt = _mm512_cmp_ps_mask(dist, best_dist, _CMP_LT_OQ);
                best_dist = _mm512_mask_mov_ps(best_dist, lt, dist);
                best_idx = _mm512_mask_mov_epi32(best_idx, lt, _mm512_set1_epi32(c));
            }
            _mm512_storeu_si512(cell_ids + base, best_idx);
        }

        int tail_start = full_tiles * kTileRows;
        for (int i = tail_start; i < n; ++i) {
            cell_ids[i] = assign(subspace_data + (size_t)i * (size_t)row_stride);
        }
#else
        #pragma omp parallel for schedule(dynamic, 256)
        for (int i = 0; i < n; ++i) {
            cell_ids[i] = assign(subspace_data + (size_t)i * (size_t)row_stride);
        }
#endif
    }

    void assign_batch_with_scratch_cap(const float* subspace_data, int* cell_ids, int n,
                                       size_t scratch_cap_bytes) const {
        assign_batch_strided_with_scratch_cap(subspace_data, sub_dim_, cell_ids, n, scratch_cap_bytes);
    }

    void assign_batch(const float* subspace_data, int* cell_ids, int n) const {
        assign_batch_with_scratch_cap(subspace_data, cell_ids, n, 128ull * 1024ull * 1024ull);
    }

    // Dynamic Activation: find top-L nearest centroids.
    // Query-time activation uses prepared SoA centroids only; QueryEngine::load_index()
    // calls prepare_simd_centroids() for every loaded subspace.
    // Overload writing into caller's pre-allocated buffer (zero-alloc after warmup).
    void dynamic_activation(const float* query_subspace, int top_l,
                            std::vector<std::pair<float, int>>& result) const {
        if (!simd_prepared_) {
            throw std::runtime_error("IVF dynamic_activation requires prepare_simd_centroids()");
        }

        auto& scratch = thread_local_query_scratch();
        scratch.select_vals.resize(K_);
        scratch.select_ids.resize(K_);
        auto& select_vals = scratch.select_vals;
        auto& select_ids = scratch.select_ids;

        // Inline SIMD ranking path: exact ordering by 0.5*||c||² - <q,c>.
        // The omitted 0.5*||q||² term is constant across centroids in this subspace.
#if defined(__AVX512F__)
        int start_dim = 0;
        if (sub_dim_ > 0) {
            __m512 q_val = _mm512_set1_ps(query_subspace[0]);
            const float* cent_d = centroids_soa_.data();
            int k = 0;
            for (; k + 16 <= K_; k += 16) {
                __m512 c_val = _mm512_loadu_ps(cent_d + k);
                __m512 bias_val = _mm512_loadu_ps(centroid_rank_bias_.data() + k);
                _mm512_storeu_ps(select_vals.data() + k, _mm512_fnmadd_ps(q_val, c_val, bias_val));
            }
            for (; k < K_; k++) {
                select_vals[(size_t)k] = std::fma(-query_subspace[0], cent_d[k],
                                                  centroid_rank_bias_[(size_t)k]);
            }
            start_dim = 1;
        } else {
            std::copy(centroid_rank_bias_.begin(), centroid_rank_bias_.end(), select_vals.begin());
        }
        for (int d = start_dim; d < sub_dim_; d++) {
            __m512 q_val = _mm512_set1_ps(query_subspace[d]);
            const float* cent_d = centroids_soa_.data() + (size_t)d * K_;
            int k = 0;
            for (; k + 16 <= K_; k += 16) {
                __m512 c_val = _mm512_loadu_ps(cent_d + k);
                __m512 dist_old = _mm512_loadu_ps(select_vals.data() + k);
                _mm512_storeu_ps(select_vals.data() + k, _mm512_fnmadd_ps(q_val, c_val, dist_old));
            }
            for (; k < K_; k++) {
                select_vals[(size_t)k] = std::fma(-query_subspace[d], cent_d[k], select_vals[(size_t)k]);
            }
        }
#else
        for (int k = 0; k < K_; k++) {
            float dist = centroid_rank_score(k, 0.0f);
            for (int d = 0; d < sub_dim_; d++) {
                dist = std::fma(-query_subspace[d], centroids_soa_[(size_t)d * K_ + k], dist);
            }
            select_vals[(size_t)k] = dist;
        }
#endif
        int actual_top = std::min(top_l, K_);
#if defined(__AVX512F__)
        int c = 0;
        const __m512i lane = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                               8, 9, 10, 11, 12, 13, 14, 15);
        for (; c + 16 <= K_; c += 16) {
            _mm512_storeu_si512(select_ids.data() + c,
                                _mm512_add_epi32(_mm512_set1_epi32(c), lane));
        }
        for (; c < K_; ++c) select_ids[(size_t)c] = c;
#else
        for (int c = 0; c < K_; c++) select_ids[(size_t)c] = c;
#endif

        if (actual_top < K_) {
            argpartition_smallest_avx512(select_vals.data(), select_ids.data(), (size_t)K_, (size_t)actual_top);
        }
        result.resize((size_t)actual_top);
        for (int c = 0; c < actual_top; ++c) {
            result[(size_t)c] = {select_vals[(size_t)c], select_ids[(size_t)c]};
        }
        std::sort(result.begin(), result.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
    }

    // Prepare centroids for inline SIMD rank-score evaluation (exact ordering, no quantization).
    // Transposes centroids from AoS to SoA layout for cache-friendly SIMD batch processing.
    void prepare_simd_centroids() {
        centroids_soa_.resize((size_t)K_ * sub_dim_);
        for (int k = 0; k < K_; k++)
            for (int d = 0; d < sub_dim_; d++)
                centroids_soa_[d * K_ + k] = centroids_[(size_t)k * sub_dim_ + d];
        simd_prepared_ = true;
    }

    // Compute posting list sizes from assignments
    void compute_cell_sizes(const int* cell_ids, int n) {
        std::fill(cell_sizes_.begin(), cell_sizes_.end(), 0);
        for (int i = 0; i < n; i++) {
            cell_sizes_[cell_ids[i]]++;
        }
        cell_offsets_[0] = 0;
        for (int c = 0; c < K_; c++) {
            cell_offsets_[c + 1] = cell_offsets_[c] + cell_sizes_[c];
        }
    }

    // Accessors
    int subspace_dim() const { return sub_dim_; }
    int K() const { return K_; }
    size_t k_total() const { return (size_t)K_; }
    int subspace_id() const { return subspace_id_; }
    const std::vector<size_t>& cell_sizes() const { return cell_sizes_; }
    const std::vector<size_t>& cell_offsets() const { return cell_offsets_; }
    const std::vector<float>& centroids() const { return centroids_; }

    size_t centroid_buffer_bytes() const {
        return centroids_.capacity() * sizeof(float) +
               centroid_rank_bias_.capacity() * sizeof(float) +
               centroids_soa_.capacity() * sizeof(float);
    }

    size_t metadata_buffer_bytes() const {
        return cell_sizes_.capacity() * sizeof(size_t) +
               cell_offsets_.capacity() * sizeof(size_t);
    }

    size_t buffer_bytes() const {
        return centroid_buffer_bytes() + metadata_buffer_bytes();
    }

    // Serialize
    void save(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&sub_dim_), sizeof(int));
        out.write(reinterpret_cast<const char*>(&K_), sizeof(int));
        out.write(reinterpret_cast<const char*>(&subspace_id_), sizeof(int));
        out.write(reinterpret_cast<const char*>(centroids_.data()),
                  centroids_.size() * sizeof(float));
        out.write(reinterpret_cast<const char*>(cell_sizes_.data()),
                  cell_sizes_.size() * sizeof(size_t));
    }

    void load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) throw std::runtime_error("Cannot open: " + path);
        in.read(reinterpret_cast<char*>(&sub_dim_), sizeof(int));
        in.read(reinterpret_cast<char*>(&K_), sizeof(int));
        in.read(reinterpret_cast<char*>(&subspace_id_), sizeof(int));

        centroids_.resize((size_t)K_ * sub_dim_);
        in.read(reinterpret_cast<char*>(centroids_.data()),
                centroids_.size() * sizeof(float));

        cell_sizes_.resize(K_);
        cell_offsets_.resize(K_ + 1);
        in.read(reinterpret_cast<char*>(cell_sizes_.data()),
                cell_sizes_.size() * sizeof(size_t));

        // Recompute offsets
        cell_offsets_[0] = 0;
        for (int c = 0; c < K_; c++)
            cell_offsets_[c + 1] = cell_offsets_[c] + cell_sizes_[c];

        precompute_centroid_rank_bias();
    }

private:
    int sub_dim_ = 0;
    int K_ = 0;
    int subspace_id_ = 0;

    std::vector<float> centroids_;          // K x sub_dim, row-major
    std::vector<float> centroid_rank_bias_;  // K, precomputed 0.5*||c||² for ranking
    std::vector<size_t> cell_sizes_;
    std::vector<size_t> cell_offsets_;

    // SoA SIMD centroid data (populated by prepare_simd_centroids())
    std::vector<float> centroids_soa_;  // [dim * K + k] layout for cache-friendly SIMD
    bool simd_prepared_ = false;

    struct QueryScratch {
        std::vector<float> select_vals;
        std::vector<int> select_ids;
    };

    static QueryScratch& thread_local_query_scratch() {
        thread_local QueryScratch scratch;
        return scratch;
    }

    inline float centroid_rank_score(int c, float dot) const {
        return centroid_rank_bias_[(size_t)c] - dot;
    }

    static float median3(float a, float b, float c) {
        if (b < a) std::swap(a, b);
        if (c < b) std::swap(b, c);
        if (b < a) std::swap(a, b);
        return b;
    }

    static float quantile9_pivot(const float* vals, size_t first, size_t last, size_t target) {
        float sample[9];
        size_t span = last - first;
        for (size_t i = 0; i < 9; ++i) {
            size_t pos = first + (i * (span - 1)) / 8;
            sample[i] = vals[pos];
        }
        for (size_t i = 1; i < 9; ++i) {
            float v = sample[i];
            size_t j = i;
            while (j > 0 && v < sample[j - 1]) {
                sample[j] = sample[j - 1];
                --j;
            }
            sample[j] = v;
        }
        size_t rank = (target * 8) / std::max<size_t>(1, span - 1);
        rank = std::min<size_t>(7, std::max<size_t>(1, rank));
        return sample[rank];
    }

    static void sort_soa_range(float* vals, int* ids, size_t first, size_t last) {
        for (size_t i = first + 1; i < last; ++i) {
            float v = vals[i];
            int id = ids[i];
            size_t j = i;
            while (j > first && v < vals[j - 1]) {
                vals[j] = vals[j - 1];
                ids[j] = ids[j - 1];
                --j;
            }
            vals[j] = v;
            ids[j] = id;
        }
    }

#if defined(__AVX512F__)
    static __mmask16 lane_mask_16(uint32_t count) {
        return count >= 16 ? (__mmask16)0xffffu : (__mmask16)((1u << count) - 1u);
    }

    struct KVBuffer16 {
        __m512 val;
        __m512i idx;
        static constexpr uint32_t nval = 16;
    };

    struct KVPartialBuffer16 {
        __m512 val;
        __m512i idx;
        uint32_t nval = 0;
    };

    struct KVPartialBuffer32 {
        KVBuffer16 b0;
        KVBuffer16 b1;
        uint32_t nval = 0;

        template <typename BufferT>
        inline void append(const BufferT& partial) {
            const uint32_t partial_nval = partial.nval;
            if (partial_nval == 0) return;

            if (nval == 0) {
                b0.val = partial.val;
                b0.idx = partial.idx;
                nval = partial_nval;
            } else if (nval + partial_nval <= 16) {
                __mmask16 mask_new = (__mmask16)(lane_mask_16(partial_nval) << nval);
                b0.val = _mm512_mask_expand_ps(b0.val, mask_new, partial.val);
                b0.idx = _mm512_mask_expand_epi32(b0.idx, mask_new, partial.idx);
                nval += partial_nval;
            } else if (nval <= 16) {
                const uint32_t space_in_b0 = 16 - nval;
                if (space_in_b0 > 0) {
                    __mmask16 mask_fill = (__mmask16)(lane_mask_16(space_in_b0) << nval);
                    b0.val = _mm512_mask_expand_ps(b0.val, mask_fill, partial.val);
                    b0.idx = _mm512_mask_expand_epi32(b0.idx, mask_fill, partial.idx);
                }

                const uint32_t remaining = partial_nval - space_in_b0;
                if (remaining > 0) {
                    __mmask16 mask_rest = (__mmask16)(lane_mask_16(remaining) << space_in_b0);
                    b1.val = _mm512_maskz_compress_ps(mask_rest, partial.val);
                    b1.idx = _mm512_maskz_compress_epi32(mask_rest, partial.idx);
                }
                nval = 16 + remaining;
            } else {
                const uint32_t pos_in_b1 = nval - 16;
                __mmask16 mask_new = (__mmask16)(lane_mask_16(partial_nval) << pos_in_b1);
                b1.val = _mm512_mask_expand_ps(b1.val, mask_new, partial.val);
                b1.idx = _mm512_mask_expand_epi32(b1.idx, mask_new, partial.idx);
                nval += partial_nval;
            }
        }

        inline KVBuffer16 pop_front_16() {
            KVBuffer16 ret = b0;
            b0 = b1;
            nval -= 16;
            return ret;
        }
    };
#endif

    static size_t partition_two_way_soa(float* vals, int* ids,
                                        size_t first, size_t last, float pivot) {
#if defined(__AVX512F__)
        constexpr size_t B = 16;
        const size_t n = last - first;
        if (n < B) {
            size_t left = first;
            size_t right = last;
            while (left < right) {
                while (left < right && vals[left] < pivot) ++left;
                while (left < right) {
                    --right;
                    if (vals[right] < pivot) break;
                }
                if (left >= right) break;
                std::swap(vals[left], vals[right]);
                std::swap(ids[left], ids[right]);
                ++left;
            }
            return left;
        }

        float* base_vals = vals + first;
        int* base_ids = ids + first;
        const __m512 pivot_vec = _mm512_set1_ps(pivot);

        KVPartialBuffer32 buffer_left;
        KVPartialBuffer32 buffer_right;

        auto load = [base_vals, base_ids](size_t i) {
            KVBuffer16 ret;
            ret.val = _mm512_loadu_ps(base_vals + i);
            ret.idx = _mm512_loadu_epi32(base_ids + i);
            return ret;
        };

        auto store = [base_vals, base_ids](size_t i, const KVBuffer16& x) {
            _mm512_storeu_ps(base_vals + i, x.val);
            _mm512_storeu_epi32(base_ids + i, x.idx);
        };

        auto partial_load = [base_vals, base_ids](size_t addr, size_t count) {
            KVPartialBuffer16 ret;
            ret.nval = (uint32_t)count;
            if (count == 0) return ret;
            const __mmask16 mask = lane_mask_16((uint32_t)count);
            ret.val = _mm512_maskz_loadu_ps(mask, base_vals + addr);
            ret.idx = _mm512_maskz_loadu_epi32(mask, base_ids + addr);
            return ret;
        };

        auto partial_store = [base_vals, base_ids](size_t addr, const KVPartialBuffer16& buf) {
            if (buf.nval == 0) return;
            const __mmask16 mask = lane_mask_16(buf.nval);
            _mm512_mask_storeu_ps(base_vals + addr, mask, buf.val);
            _mm512_mask_storeu_epi32(base_ids + addr, mask, buf.idx);
        };

        auto fill_buffers = [&](const KVBuffer16& x) {
            const __mmask16 mask_lt = _mm512_cmp_ps_mask(x.val, pivot_vec, _CMP_LT_OQ);
            const __mmask16 mask_ge = (__mmask16)(~(uint32_t)mask_lt & 0xffffu);

            KVPartialBuffer16 left_part;
            left_part.val = _mm512_maskz_compress_ps(mask_lt, x.val);
            left_part.idx = _mm512_maskz_compress_epi32(mask_lt, x.idx);
            left_part.nval = (uint32_t)_mm_popcnt_u32((uint32_t)mask_lt);

            KVPartialBuffer16 right_part;
            right_part.val = _mm512_maskz_compress_ps(mask_ge, x.val);
            right_part.idx = _mm512_maskz_compress_epi32(mask_ge, x.idx);
            right_part.nval = (uint32_t)_mm_popcnt_u32((uint32_t)mask_ge);

            buffer_left.append(left_part);
            buffer_right.append(right_part);
        };

        auto fill_buffers_partial = [&](const KVPartialBuffer16& x) {
            if (x.nval == 0) return;
            const __mmask16 valid_mask = lane_mask_16(x.nval);
            const __mmask16 mask_lt = (__mmask16)(_mm512_cmp_ps_mask(x.val, pivot_vec, _CMP_LT_OQ) & valid_mask);
            const __mmask16 mask_ge = (__mmask16)(~(uint32_t)mask_lt & (uint32_t)valid_mask);

            KVPartialBuffer16 left_part;
            left_part.val = _mm512_maskz_compress_ps(mask_lt, x.val);
            left_part.idx = _mm512_maskz_compress_epi32(mask_lt, x.idx);
            left_part.nval = (uint32_t)_mm_popcnt_u32((uint32_t)mask_lt);

            KVPartialBuffer16 right_part;
            right_part.val = _mm512_maskz_compress_ps(mask_ge, x.val);
            right_part.idx = _mm512_maskz_compress_epi32(mask_ge, x.idx);
            right_part.nval = (uint32_t)_mm_popcnt_u32((uint32_t)mask_ge);

            buffer_left.append(left_part);
            buffer_right.append(right_part);
        };

        size_t left_ptr = 0;
        size_t right_ptr = n;
        KVPartialBuffer16 x_partial;

        if (n >= 2 * B) {
            KVBuffer16 x = load(left_ptr);
            fill_buffers(x);

            x = load(right_ptr - B);
            fill_buffers(x);

            while (left_ptr + 3 * B <= right_ptr) {
                if (buffer_left.nval >= B) {
                    KVBuffer16 to_store = buffer_left.pop_front_16();
                    store(left_ptr, to_store);
                    left_ptr += B;
                    x = load(left_ptr);
                    fill_buffers(x);
                } else {
                    KVBuffer16 to_store = buffer_right.pop_front_16();
                    store(right_ptr - B, to_store);
                    right_ptr -= B;
                    x = load(right_ptr - B);
                    fill_buffers(x);
                }
            }

            const size_t remaining = right_ptr - left_ptr - 2 * B;
            x_partial = partial_load(left_ptr + B, remaining);

            if (buffer_left.nval >= B) {
                KVBuffer16 to_store = buffer_left.pop_front_16();
                store(left_ptr, to_store);
                left_ptr += B;
            }
            if (buffer_right.nval >= B) {
                KVBuffer16 to_store = buffer_right.pop_front_16();
                store(right_ptr - B, to_store);
                right_ptr -= B;
            }
        } else {
            KVBuffer16 x = load(0);
            fill_buffers(x);
            x_partial = partial_load(B, n - B);
        }

        fill_buffers_partial(x_partial);

        if (buffer_left.nval >= B) {
            KVBuffer16 to_store = buffer_left.pop_front_16();
            store(left_ptr, to_store);
            left_ptr += B;
        }

        if (buffer_left.nval > 0) {
            KVPartialBuffer16 left_remainder;
            left_remainder.val = buffer_left.b0.val;
            left_remainder.idx = buffer_left.b0.idx;
            left_remainder.nval = buffer_left.nval;
            partial_store(left_ptr, left_remainder);
            left_ptr += left_remainder.nval;
        }

        const size_t ret = left_ptr;

        if (buffer_right.nval >= B) {
            KVBuffer16 to_store = buffer_right.pop_front_16();
            store(left_ptr, to_store);
            left_ptr += B;
        }

        if (buffer_right.nval > 0) {
            KVPartialBuffer16 right_remainder;
            right_remainder.val = buffer_right.b0.val;
            right_remainder.idx = buffer_right.b0.idx;
            right_remainder.nval = buffer_right.nval;
            partial_store(left_ptr, right_remainder);
        }

        return first + ret;
#else
        size_t left = first;
        size_t right = last;
        while (left < right) {
            while (left < right && vals[left] < pivot) ++left;
            while (left < right) {
                --right;
                if (vals[right] < pivot) break;
            }
            if (left >= right) break;
            std::swap(vals[left], vals[right]);
            std::swap(ids[left], ids[right]);
            ++left;
        }
        return left;
#endif
    }

    static void argpartition_smallest_avx512(float* vals, int* ids, size_t n, size_t k) {
        if (k == 0 || k >= n) return;

        size_t first = 0;
        size_t last = n;
        size_t target = k;
        int depth_limit = 0;
        for (size_t m = n; m > 1; m >>= 1) depth_limit += 2;

        while (last - first > 64) {
            if (depth_limit-- <= 0) {
                sort_soa_range(vals, ids, first, last);
                return;
            }

            size_t mid = first + ((last - first) >> 1);
            float pivot = (last - first >= 256)
                ? quantile9_pivot(vals, first, last, target)
                : median3(vals[first], vals[mid], vals[last - 1]);
            size_t cut = partition_two_way_soa(vals, ids, first, last, pivot);

            if (cut <= first || cut >= last) {
                sort_soa_range(vals, ids, first, last);
                return;
            }

            size_t left_count = cut - first;
            if (target < left_count) {
                last = cut;
            } else if (target > left_count) {
                first = cut;
                target -= left_count;
            } else {
                return;
            }
        }

        sort_soa_range(vals, ids, first, last);
    }

    void precompute_centroid_rank_bias() {
        centroid_rank_bias_.resize(K_);
        for (int c = 0; c < K_; c++) {
            float nsq = 0;
            const float* cent = centroids_.data() + (size_t)c * sub_dim_;
            for (int d = 0; d < sub_dim_; d++) nsq += cent[d] * cent[d];
            centroid_rank_bias_[c] = 0.5f * nsq;
        }
    }

    // Forgy initialization: pick K random data points as initial centroids
    void init_centroids_forgy(const float* data, int n, uint64_t seed) {
        std::mt19937 rng(seed);
        std::vector<int> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        // Partial Fisher-Yates: only shuffle first K elements
        for (int i = 0; i < K_ && i < n; i++) {
            int j = i + rng() % (n - i);
            std::swap(indices[i], indices[j]);
        }
        for (int c = 0; c < K_; c++) {
            std::copy(data + (size_t)indices[c] * sub_dim_,
                      data + (size_t)(indices[c] + 1) * sub_dim_,
                      centroids_.data() + (size_t)c * sub_dim_);
        }
    }

    // Update centroids with pre-allocated buffers (avoids alloc in hot loop)
    void update_centroids_prealloced(const float* data, const int* assignments, int n,
                                      std::vector<float>& partial_centroids,
                                      std::vector<int>& partial_counts) {
        int nt = omp_get_max_threads();
        size_t pc_sz = (size_t)nt * K_ * sub_dim_;
        size_t cnt_sz = (size_t)nt * K_;

        std::memset(partial_centroids.data(), 0, pc_sz * sizeof(float));
        std::memset(partial_counts.data(), 0, cnt_sz * sizeof(int));

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            float* my_cents = partial_centroids.data() + (size_t)tid * K_ * sub_dim_;
            int* my_counts = partial_counts.data() + (size_t)tid * K_;

            #pragma omp for schedule(static)
            for (int i = 0; i < n; i++) {
                int c = assignments[i];
                my_counts[c]++;
                const float* pt = data + (size_t)i * sub_dim_;
                float* cent = my_cents + (size_t)c * sub_dim_;
                for (int d = 0; d < sub_dim_; d++)
                    cent[d] += pt[d];
            }
        }

        #pragma omp parallel for schedule(static)
        for (int c = 0; c < K_; c++) {
            float* cent = centroids_.data() + (size_t)c * sub_dim_;
            std::fill(cent, cent + sub_dim_, 0.0f);
            int total = 0;
            for (int t = 0; t < nt; t++) {
                total += partial_counts[(size_t)t * K_ + c];
                const float* pc = partial_centroids.data() + ((size_t)t * K_ + c) * sub_dim_;
                for (int d = 0; d < sub_dim_; d++)
                    cent[d] += pc[d];
            }
            if (total > 0) {
                float inv = 1.0f / total;
                for (int d = 0; d < sub_dim_; d++)
                    cent[d] *= inv;
            }
        }
    }

#if HAS_SUPERKMEANS
    // HierarchicalSuperKMeans training (kept for reference/fallback)
    void train_hskm(const float* data, int n, int max_iter, uint64_t seed,
                    bool use_internal_rotation = false) {
        if (K_ >= 128) {
            skmeans::HierarchicalSuperKMeansConfig hcfg;
            hcfg.iters_mesoclustering = 3;
            hcfg.iters_fineclustering = 5;
            hcfg.iters_refinement = 0;  // meso+fine sufficient; refinement adds <1% recall
            hcfg.seed = seed;
            // n_threads=1: disables all OMP parallel regions inside SuperKMeans
            // (guarded by `if (n_threads > 1)`). BLIS sgemm_ still uses its own
            // pthread pool for multi-threaded GEMM. This eliminates ~80% of time
            // that was spent in OMP barrier spin-wait.
            hcfg.n_threads = 1;
            hcfg.verbose = false;
            // use_internal_rotation: let SuperKMeans apply DCT/QR internally for better
            // convergence, then unrotate centroids back to caller's space.
            hcfg.data_already_rotated = !use_internal_rotation;
            hcfg.unrotate_centroids = use_internal_rotation;

            skmeans::HierarchicalSuperKMeans<skmeans::Quantization::f32,
                                            skmeans::DistanceFunction::l2>
                hskmeans(K_, sub_dim_, hcfg);
            auto sk_centroids = hskmeans.Train(data, n);
            centroids_.assign(sk_centroids.begin(), sk_centroids.end());
        } else {
            skmeans::SuperKMeansConfig cfg;
            cfg.iters = max_iter;
            cfg.seed = seed;
            cfg.n_threads = 1;
            cfg.verbose = false;
            cfg.data_already_rotated = !use_internal_rotation;
            cfg.unrotate_centroids = use_internal_rotation;

            skmeans::SuperKMeans<skmeans::Quantization::f32,
                                 skmeans::DistanceFunction::l2>
                skmeans_obj(K_, sub_dim_, cfg);
            auto sk_centroids = skmeans_obj.Train(data, n);
            centroids_.assign(sk_centroids.begin(), sk_centroids.end());
        }
    }
#endif
};

} // namespace score
