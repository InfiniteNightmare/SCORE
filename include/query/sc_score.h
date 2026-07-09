#pragma once
#include "core/common.h"
#include "update/update_state.h"
#include <vector>
#include <algorithm>
#include <cstring>

#if defined(__AVX512BW__)
#include <immintrin.h>
#endif

namespace score {

struct SCSelectStats {
    size_t requested = 0;
    size_t nonzero_scores = 0;
    int max_score = 0;
    int threshold_score = 0;
    size_t n_above_threshold = 0;
    size_t threshold_bucket = 0;
    size_t need_at_threshold = 0;
    size_t selected_above_threshold = 0;
    size_t selected_at_threshold = 0;
    size_t selected_total = 0;
    size_t discarded_at_threshold = 0;
};

// SC-score: collision count per point across subspaces
class SCScore {
public:
    SCScore() = default;

    void init(size_t num_points, int num_subspaces) {
        (void)num_subspaces;
        n_ = num_points;
        scores_.resize(n_, 0);
    }

    void init_uninitialized(size_t num_points, int num_subspaces) {
        (void)num_subspaces;
        n_ = num_points;
        scores_.resize_uninitialized(n_);
    }

    void advise_scores_hugepage() {
        scores_.advise_hugepage();
    }

    void reset() {
        std::memset(scores_.data(), 0, n_);
    }

    // Increment scores for all points in a posting list (one subspace hit)
    void increment(const uint32_t* point_ids, size_t count) {
        uint8_t* __restrict__ sc = scores_.data();
        constexpr size_t PREFETCH = 16;
        size_t i = 0;
        for (; i + PREFETCH < count; i++) {
            __builtin_prefetch(&sc[point_ids[i + PREFETCH]], 1, 1);
            sc[point_ids[i]]++;
        }
        for (; i < count; i++) {
            sc[point_ids[i]]++;
        }
    }

    // Weighted increment: CRISP-style scoring where close cells get higher weight
    // 4-way unroll: process 4 entries per iteration for better ILP
    void increment_weighted(const uint32_t* point_ids,
                            size_t count,
                            uint8_t weight,
                            int prefetch_distance = 64) {
        uint8_t* __restrict__ sc = scores_.data();
        if (prefetch_distance == 32) {
            constexpr size_t PREFETCH = 64;
            size_t i = 0;
            for (; i + PREFETCH < count; i += 4) {
                __builtin_prefetch(&sc[point_ids[i + PREFETCH]], 1, 1);
                sc[point_ids[i]] += weight;
                sc[point_ids[i + 1]] += weight;
                sc[point_ids[i + 2]] += weight;
                sc[point_ids[i + 3]] += weight;
            }
            for (; i < count; i++) {
                sc[point_ids[i]] += weight;
            }
            return;
        }

        if (prefetch_distance == 64) {
            constexpr size_t PREFETCH = 64;
            constexpr size_t PREFETCH2 = 128;
            size_t i = 0;
            for (; i + PREFETCH2 < count; i += 4) {
                __builtin_prefetch(&sc[point_ids[i + PREFETCH]], 1, 1);
                __builtin_prefetch(&sc[point_ids[i + PREFETCH2]], 1, 1);
                sc[point_ids[i]] += weight;
                sc[point_ids[i + 1]] += weight;
                sc[point_ids[i + 2]] += weight;
                sc[point_ids[i + 3]] += weight;
            }
            for (; i < count; i++) {
                sc[point_ids[i]] += weight;
            }
            return;
        }

        if (prefetch_distance <= 0) {
            size_t i = 0;
            for (; i + 3 < count; i += 4) {
                sc[point_ids[i]] += weight;
                sc[point_ids[i + 1]] += weight;
                sc[point_ids[i + 2]] += weight;
                sc[point_ids[i + 3]] += weight;
            }
            for (; i < count; i++) {
                sc[point_ids[i]] += weight;
            }
            return;
        }

        const size_t PREFETCH = (size_t)prefetch_distance;
        size_t i = 0;
        for (; i + 3 < count; i += 4) {
            if (i + PREFETCH < count) {
                __builtin_prefetch(&sc[point_ids[i + PREFETCH]], 1, 1);
            }
            sc[point_ids[i]] += weight;
            sc[point_ids[i + 1]] += weight;
            sc[point_ids[i + 2]] += weight;
            sc[point_ids[i + 3]] += weight;
        }
        for (; i < count; i++) {
            sc[point_ids[i]] += weight;
        }
    }

    void increment_weighted_live(const uint32_t* point_ids,
                                 size_t count,
                                 uint8_t weight,
                                 const LiveTable& live,
                                 int prefetch_distance = 64) {
        uint8_t* __restrict__ sc = scores_.data();
        size_t prefetch = prefetch_distance > 0 ? (size_t)prefetch_distance : 0;
        size_t i = 0;
        for (; i + 3 < count; i += 4) {
            if (prefetch != 0 && i + prefetch < count) {
                __builtin_prefetch(&sc[point_ids[i + prefetch]], 1, 1);
            }
            uint32_t p0 = point_ids[i];
            uint32_t p1 = point_ids[i + 1];
            uint32_t p2 = point_ids[i + 2];
            uint32_t p3 = point_ids[i + 3];
            if (live.is_live(p0)) sc[p0] += weight;
            if (live.is_live(p1)) sc[p1] += weight;
            if (live.is_live(p2)) sc[p2] += weight;
            if (live.is_live(p3)) sc[p3] += weight;
        }
        for (; i < count; ++i) {
            uint32_t pid = point_ids[i];
            if (live.is_live(pid)) sc[pid] += weight;
        }
    }

    // Select top-beta*N candidate point IDs by score
    std::vector<int> select_top(float beta, SCSelectStats* stats = nullptr) const {
        size_t num_candidates = std::max(1, (int)(beta * n_));
        return select_top_n(num_candidates, stats);
    }

    // SIMD-accelerated selection: skip zero-score blocks
    std::vector<int> select_top_n(size_t num_candidates, SCSelectStats* stats = nullptr) const {
        std::vector<int> result;
        select_top_n_into(num_candidates, stats, result);
        return result;
    }

    void select_top_n_into(size_t num_candidates,
                           SCSelectStats* stats,
                           std::vector<int>& result) const {
#if defined(__AVX512BW__)
        select_top_n_avx512_into(num_candidates, stats, result);
#else
        select_top_n_scalar_into(num_candidates, stats, result);
#endif
    }

    void select_top_n_live_into(size_t num_candidates,
                                const LiveTable& live,
                                SCSelectStats* stats,
                                std::vector<int>& result) const {
#if defined(__AVX512BW__)
        select_top_n_live_avx512_into(num_candidates, live, stats, result);
#else
        select_top_n_live_scalar_into(num_candidates, live, stats, result);
#endif
    }

    void select_top_n_range_into(size_t num_candidates,
                                 size_t begin,
                                 size_t end,
                                 SCSelectStats* stats,
                                 std::vector<int>& result) const {
        if (begin > end || end > n_) {
            throw std::invalid_argument("SCScore select range is outside score array");
        }
        if (begin == end) {
            result.clear();
            if (stats) {
                *stats = SCSelectStats{};
                stats->requested = num_candidates;
            }
            return;
        }
        if (begin == 0 && end == n_) {
            select_top_n_into(num_candidates, stats, result);
            return;
        }
#if defined(__AVX512BW__)
        select_top_n_range_avx512_into(num_candidates, begin, end, stats, result);
#else
        select_top_n_range_scalar_into(num_candidates, begin, end, stats, result);
#endif
    }

    uint8_t score(int point_id) const { return scores_[point_id]; }

    void clear_scores_for_pids(const std::vector<uint32_t>& point_ids) {
        clear_scores_for_ids(point_ids.data(), point_ids.size());
    }

    void clear_scores_for_ids(const uint32_t* point_ids, size_t count) {
        uint8_t* __restrict__ sc = scores_.data();
        if (count >= 128) {
            constexpr size_t PREFETCH = 32;
            size_t i = 0;
            for (; i + PREFETCH + 3 < count; i += 4) {
                __builtin_prefetch(&sc[point_ids[i + PREFETCH]], 1, 1);
                uint32_t p0 = point_ids[i];
                uint32_t p1 = point_ids[i + 1];
                uint32_t p2 = point_ids[i + 2];
                uint32_t p3 = point_ids[i + 3];
                if ((size_t)p0 >= n_ || (size_t)p1 >= n_ ||
                    (size_t)p2 >= n_ || (size_t)p3 >= n_) {
                    throw std::out_of_range("SCScore clear PID exceeds score array size");
                }
                sc[p0] = 0;
                sc[p1] = 0;
                sc[p2] = 0;
                sc[p3] = 0;
            }
            for (; i < count; ++i) {
                uint32_t pid = point_ids[i];
                if ((size_t)pid >= n_) {
                    throw std::out_of_range("SCScore clear PID exceeds score array size");
                }
                sc[pid] = 0;
            }
            return;
        }
        for (size_t i = 0; i < count; ++i) {
            uint32_t pid = point_ids[i];
            if ((size_t)pid >= n_) {
                throw std::out_of_range("SCScore clear PID exceeds score array size");
            }
            sc[pid] = 0;
        }
    }

    void clear_scores_for_valid_pids(const uint32_t* point_ids, size_t count) {
        uint8_t* __restrict__ sc = scores_.data();
        if (count >= 128) {
            constexpr size_t PREFETCH = 64;
            size_t i = 0;
            for (; i + PREFETCH + 3 < count; i += 4) {
                __builtin_prefetch(&sc[point_ids[i + PREFETCH]], 1, 1);
                sc[point_ids[i]] = 0;
                sc[point_ids[i + 1]] = 0;
                sc[point_ids[i + 2]] = 0;
                sc[point_ids[i + 3]] = 0;
            }
            for (; i < count; ++i) {
                sc[point_ids[i]] = 0;
            }
            return;
        }
        for (size_t i = 0; i < count; ++i) {
            sc[point_ids[i]] = 0;
        }
    }

    void clear_scores_for_valid_pids(const std::vector<uint32_t>& point_ids) {
        clear_scores_for_valid_pids(point_ids.data(), point_ids.size());
    }

    // Access raw scores array
    const hugepage_vector_uint8& scores() const { return scores_; }
    hugepage_vector_uint8& mutable_scores() { return scores_; }
    size_t num_points() const { return n_; }

private:
    static void fill_select_hist_stats(SCSelectStats* stats,
                                       size_t requested,
                                       const uint32_t hist[256],
                                       int max_score,
                                       int threshold,
                                       size_t n_above,
                                       size_t need_at) {
        if (!stats) return;
        stats->requested = requested;
        stats->nonzero_scores = 0;
        for (int s = 1; s <= max_score; ++s) stats->nonzero_scores += hist[s];
        stats->max_score = max_score;
        stats->threshold_score = threshold;
        stats->n_above_threshold = n_above;
        stats->threshold_bucket = threshold > 0 ? hist[threshold] : 0;
        stats->need_at_threshold = need_at;
    }

    static void finish_select_hist_stats(SCSelectStats* stats,
                                          size_t selected_above,
                                          size_t selected_at,
                                          size_t selected_total) {
        if (!stats) return;
        stats->selected_above_threshold = selected_above;
        stats->selected_at_threshold = selected_at;
        stats->selected_total = selected_total;
        stats->discarded_at_threshold = stats->threshold_bucket > selected_at
            ? stats->threshold_bucket - selected_at
            : 0;
    }

#if defined(__AVX512BW__)
    struct SelectCountInfo {
        size_t nonzero = 0;
        int max_score = 0;
        int threshold = 1;
        size_t n_above = 0;
        size_t threshold_bucket = 0;
        size_t need_at = 0;
    };

    struct MaxNonzeroInfo {
        int max_score = 0;
        size_t nonzero = 0;
    };

    size_t count_ge_avx512(uint8_t threshold) const {
        const __m512i thresh_vec = _mm512_set1_epi8((char)threshold);
        size_t count = 0;
        size_t n_aligned = n_ & ~63ULL;
        for (size_t i = 0; i < n_aligned; i += 64) {
            __m512i v = _mm512_loadu_si512(scores_.data() + i);
            count += (size_t)__builtin_popcountll(_mm512_cmpge_epu8_mask(v, thresh_vec));
        }
        for (size_t i = n_aligned; i < n_; ++i) {
            if (scores_[i] >= threshold) ++count;
        }
        return count;
    }

    size_t count_ge_range_avx512(uint8_t threshold, size_t begin, size_t end) const {
        const __m512i thresh_vec = _mm512_set1_epi8((char)threshold);
        size_t count = 0;
        size_t i = begin;
        size_t vector_end = begin + ((end - begin) & ~63ULL);
        for (; i < vector_end; i += 64) {
            __m512i v = _mm512_loadu_si512(scores_.data() + i);
            count += (size_t)__builtin_popcountll(_mm512_cmpge_epu8_mask(v, thresh_vec));
        }
        for (; i < end; ++i) {
            if (scores_[i] >= threshold) ++count;
        }
        return count;
    }

    MaxNonzeroInfo max_and_nonzero_avx512() const {
        size_t n_aligned = n_ & ~63ULL;
        __m512i max_vec = _mm512_setzero_si512();
        const __m512i zero_vec = _mm512_setzero_si512();
        MaxNonzeroInfo info;
        for (size_t i = 0; i < n_aligned; i += 64) {
            __m512i v = _mm512_loadu_si512(scores_.data() + i);
            max_vec = _mm512_max_epu8(max_vec, v);
            info.nonzero += (size_t)__builtin_popcountll(_mm512_cmpneq_epu8_mask(v, zero_vec));
        }
        alignas(64) uint8_t tmp[64];
        _mm512_store_si512((__m512i*)tmp, max_vec);
        for (uint8_t v : tmp) info.max_score = std::max(info.max_score, (int)v);
        for (size_t i = n_aligned; i < n_; ++i) {
            int score = (int)scores_[i];
            info.max_score = std::max(info.max_score, score);
            if (score != 0) ++info.nonzero;
        }
        return info;
    }

    MaxNonzeroInfo max_and_nonzero_range_avx512(size_t begin, size_t end) const {
        size_t i = begin;
        size_t vector_end = begin + ((end - begin) & ~63ULL);
        __m512i max_vec = _mm512_setzero_si512();
        const __m512i zero_vec = _mm512_setzero_si512();
        MaxNonzeroInfo info;
        for (; i < vector_end; i += 64) {
            __m512i v = _mm512_loadu_si512(scores_.data() + i);
            max_vec = _mm512_max_epu8(max_vec, v);
            info.nonzero += (size_t)__builtin_popcountll(_mm512_cmpneq_epu8_mask(v, zero_vec));
        }
        alignas(64) uint8_t tmp[64];
        _mm512_store_si512((__m512i*)tmp, max_vec);
        for (uint8_t v : tmp) info.max_score = std::max(info.max_score, (int)v);
        for (; i < end; ++i) {
            int score = (int)scores_[i];
            info.max_score = std::max(info.max_score, score);
            if (score != 0) ++info.nonzero;
        }
        return info;
    }

    bool build_select_count_info_avx512(size_t num_candidates, SelectCountInfo& info) const {
        MaxNonzeroInfo max_nonzero = max_and_nonzero_avx512();
        info.max_score = max_nonzero.max_score;
        info.nonzero = max_nonzero.nonzero;
        if (info.max_score == 0) return false;
        constexpr size_t kUnknownCount = (size_t)-1;
        size_t count_cache[256];
        std::fill(count_cache, count_cache + 256, kUnknownCount);
        count_cache[1] = info.nonzero;
        auto count_ge_cached = [&](int threshold) -> size_t {
            size_t& cached = count_cache[(size_t)threshold];
            if (cached == kUnknownCount) cached = count_ge_avx512((uint8_t)threshold);
            return cached;
        };

        if (num_candidates > info.nonzero) {
            info.threshold = 1;
        } else {
            int lo = 1, hi = info.max_score;
            while (lo < hi) {
                int mid = lo + (hi - lo + 1) / 2;
                if (count_ge_cached(mid) >= num_candidates) lo = mid;
                else hi = mid - 1;
            }
            info.threshold = lo;
        }
        size_t at_or_above = count_ge_cached(info.threshold);
        info.n_above = info.threshold < 255
            ? count_ge_cached(info.threshold + 1)
            : 0;
        info.threshold_bucket = at_or_above - info.n_above;
        info.need_at = num_candidates > info.n_above ? num_candidates - info.n_above : 0;
        return true;
    }

    bool build_select_count_info_range_avx512(size_t num_candidates,
                                              size_t begin,
                                              size_t end,
                                              SelectCountInfo& info) const {
        MaxNonzeroInfo max_nonzero = max_and_nonzero_range_avx512(begin, end);
        info.max_score = max_nonzero.max_score;
        info.nonzero = max_nonzero.nonzero;
        if (info.max_score == 0) return false;
        constexpr size_t kUnknownCount = (size_t)-1;
        size_t count_cache[256];
        std::fill(count_cache, count_cache + 256, kUnknownCount);
        count_cache[1] = info.nonzero;
        auto count_ge_cached = [&](int threshold) -> size_t {
            size_t& cached = count_cache[(size_t)threshold];
            if (cached == kUnknownCount) {
                cached = count_ge_range_avx512((uint8_t)threshold, begin, end);
            }
            return cached;
        };

        if (num_candidates > info.nonzero) {
            info.threshold = 1;
        } else {
            int lo = 1, hi = info.max_score;
            while (lo < hi) {
                int mid = lo + (hi - lo + 1) / 2;
                if (count_ge_cached(mid) >= num_candidates) lo = mid;
                else hi = mid - 1;
            }
            info.threshold = lo;
        }
        size_t at_or_above = count_ge_cached(info.threshold);
        info.n_above = info.threshold < 255
            ? count_ge_cached(info.threshold + 1)
            : 0;
        info.threshold_bucket = at_or_above - info.n_above;
        info.need_at = num_candidates > info.n_above ? num_candidates - info.n_above : 0;
        return true;
    }

    size_t count_ge_live_avx512(uint8_t threshold, const LiveTable& live) const {
        const __m512i thresh_vec = _mm512_set1_epi8((char)threshold);
        const std::vector<uint64_t>& live_words = live.words();
        const size_t scan_n = live.max_pid();
        const size_t n_aligned = scan_n & ~63ULL;
        size_t count = 0;
        for (size_t i = 0, word_idx = 0; i < n_aligned; i += 64, ++word_idx) {
            uint64_t live_mask = live_words[word_idx];
            if (live_mask == 0) continue;
            __m512i v = _mm512_loadu_si512(scores_.data() + i);
            count += (size_t)__builtin_popcountll(
                (uint64_t)_mm512_cmpge_epu8_mask(v, thresh_vec) & live_mask);
        }
        for (size_t i = n_aligned; i < scan_n; ++i) {
            if (live.is_live((uint32_t)i) && scores_[i] >= threshold) ++count;
        }
        return count;
    }

    MaxNonzeroInfo max_and_nonzero_live_avx512(const LiveTable& live) const {
        const std::vector<uint64_t>& live_words = live.words();
        const size_t scan_n = live.max_pid();
        const size_t n_aligned = scan_n & ~63ULL;
        __m512i max_vec = _mm512_setzero_si512();
        const __m512i zero_vec = _mm512_setzero_si512();
        MaxNonzeroInfo info;
        for (size_t i = 0, word_idx = 0; i < n_aligned; i += 64, ++word_idx) {
            uint64_t live_mask = live_words[word_idx];
            if (live_mask == 0) continue;
            __m512i v = _mm512_loadu_si512(scores_.data() + i);
            info.nonzero += (size_t)__builtin_popcountll(
                (uint64_t)_mm512_cmpneq_epu8_mask(v, zero_vec) & live_mask);
            __m512i live_v = _mm512_maskz_mov_epi8((__mmask64)live_mask, v);
            max_vec = _mm512_max_epu8(max_vec, live_v);
        }
        alignas(64) uint8_t tmp[64];
        _mm512_store_si512((__m512i*)tmp, max_vec);
        for (uint8_t v : tmp) info.max_score = std::max(info.max_score, (int)v);
        for (size_t i = n_aligned; i < scan_n; ++i) {
            if (!live.is_live((uint32_t)i)) continue;
            int score = (int)scores_[i];
            info.max_score = std::max(info.max_score, score);
            if (score != 0) ++info.nonzero;
        }
        return info;
    }

    bool build_select_count_info_live_avx512(size_t num_candidates,
                                             const LiveTable& live,
                                             SelectCountInfo& info) const {
        MaxNonzeroInfo max_nonzero = max_and_nonzero_live_avx512(live);
        info.max_score = max_nonzero.max_score;
        info.nonzero = max_nonzero.nonzero;
        if (info.max_score == 0) return false;
        constexpr size_t kUnknownCount = (size_t)-1;
        size_t count_cache[256];
        std::fill(count_cache, count_cache + 256, kUnknownCount);
        count_cache[1] = info.nonzero;
        auto count_ge_cached = [&](int threshold) -> size_t {
            size_t& cached = count_cache[(size_t)threshold];
            if (cached == kUnknownCount) {
                cached = count_ge_live_avx512((uint8_t)threshold, live);
            }
            return cached;
        };

        if (num_candidates > info.nonzero) {
            info.threshold = 1;
        } else {
            int lo = 1, hi = info.max_score;
            while (lo < hi) {
                int mid = lo + (hi - lo + 1) / 2;
                if (count_ge_cached(mid) >= num_candidates) lo = mid;
                else hi = mid - 1;
            }
            info.threshold = lo;
        }
        size_t at_or_above = count_ge_cached(info.threshold);
        info.n_above = info.threshold < 255
            ? count_ge_cached(info.threshold + 1)
            : 0;
        info.threshold_bucket = at_or_above - info.n_above;
        info.need_at = num_candidates > info.n_above ? num_candidates - info.n_above : 0;
        return true;
    }

    static void fill_select_count_stats(SCSelectStats* stats,
                                        size_t requested,
                                        const SelectCountInfo& info) {
        if (!stats) return;
        stats->requested = requested;
        stats->nonzero_scores = info.nonzero;
        stats->max_score = info.max_score;
        stats->threshold_score = info.threshold;
        stats->n_above_threshold = info.n_above;
        stats->threshold_bucket = info.threshold_bucket;
        stats->need_at_threshold = info.need_at;
    }

    void select_top_n_avx512_into(size_t num_candidates,
                                  SCSelectStats* stats,
                                  std::vector<int>& result) const {
        SelectCountInfo info;
        if (!build_select_count_info_avx512(num_candidates, info)) {
            result.clear();
            return;
        }
        int threshold = info.threshold;
        const size_t ties_to_select = std::min(info.need_at, info.threshold_bucket);
        fill_select_count_stats(stats, num_candidates, info);

        const size_t expected_total = info.n_above + ties_to_select;
        result.resize(expected_total);
        int* out_ptr = result.data();
        size_t selected_total = 0;
        size_t selected_above = 0;
        size_t selected_at = 0;

        const __m512i thresh_vec = _mm512_set1_epi8((char)threshold);
        const bool has_above_threshold = threshold < 255;
        const __m512i thresh_plus_vec = _mm512_set1_epi8((char)(has_above_threshold ? threshold + 1 : 0));
        size_t n_aligned = n_ & ~63ULL;

        for (size_t i = 0; i < n_aligned; i += 64) {
            __m512i v = _mm512_loadu_si512(scores_.data() + i);
            __mmask64 above_mask = 0;
            __mmask64 at_mask = 0;
            if (selected_at < ties_to_select) {
                __mmask64 ge_mask = _mm512_cmpge_epu8_mask(v, thresh_vec);
                if (ge_mask == 0) continue;
                above_mask = has_above_threshold
                    ? _mm512_cmpge_epu8_mask(v, thresh_plus_vec)
                    : 0;
                at_mask = ge_mask & ~above_mask;
            } else {
                if (!has_above_threshold) break;
                above_mask = _mm512_cmpge_epu8_mask(v, thresh_plus_vec);
                if (above_mask == 0) continue;
            }

            __mmask64 am = above_mask;
            while (am) {
                out_ptr[selected_total++] = (int)(i + __builtin_ctzll(am));
                selected_above++;
                am &= am - 1;
            }

            while (at_mask && selected_at < ties_to_select) {
                out_ptr[selected_total++] = (int)(i + __builtin_ctzll(at_mask));
                selected_at++;
                at_mask &= at_mask - 1;
            }

            if (selected_above >= info.n_above && selected_at >= ties_to_select) break;
        }
        for (size_t i = n_aligned;
             i < n_ && (selected_above < info.n_above || selected_at < ties_to_select);
             i++) {
            uint8_t s = scores_[i];
            if (s > threshold) {
                out_ptr[selected_total++] = (int)i;
                selected_above++;
            } else if (s == (uint8_t)threshold && selected_at < ties_to_select) {
                out_ptr[selected_total++] = (int)i;
                selected_at++;
            }
        }

        result.resize(selected_total);
        finish_select_hist_stats(stats, selected_above, selected_at, result.size());
    }

    void select_top_n_live_avx512_into(size_t num_candidates,
                                       const LiveTable& live,
                                       SCSelectStats* stats,
                                       std::vector<int>& result) const {
        if ((size_t)live.max_pid() > n_) {
            throw std::invalid_argument("LiveTable max_pid exceeds SCScore size");
        }
        SelectCountInfo info;
        if (!build_select_count_info_live_avx512(num_candidates, live, info)) {
            result.clear();
            if (stats) {
                *stats = SCSelectStats{};
                stats->requested = num_candidates;
            }
            return;
        }
        int threshold = info.threshold;
        const size_t ties_to_select = std::min(info.need_at, info.threshold_bucket);
        fill_select_count_stats(stats, num_candidates, info);

        const size_t expected_total = info.n_above + ties_to_select;
        result.resize(expected_total);
        int* out_ptr = result.data();
        size_t selected_total = 0;
        size_t selected_above = 0;
        size_t selected_at = 0;

        const std::vector<uint64_t>& live_words = live.words();
        const __m512i thresh_vec = _mm512_set1_epi8((char)threshold);
        const bool has_above_threshold = threshold < 255;
        const __m512i thresh_plus_vec =
            _mm512_set1_epi8((char)(has_above_threshold ? threshold + 1 : 0));
        const size_t scan_n = live.max_pid();
        const size_t n_aligned = scan_n & ~63ULL;

        for (size_t i = 0, word_idx = 0; i < n_aligned; i += 64, ++word_idx) {
            uint64_t live_mask_word = live_words[word_idx];
            if (live_mask_word == 0) continue;
            __m512i v = _mm512_loadu_si512(scores_.data() + i);
            __mmask64 live_mask = (__mmask64)live_mask_word;
            __mmask64 above_mask = 0;
            __mmask64 at_mask = 0;
            if (selected_at < ties_to_select) {
                __mmask64 ge_mask = _mm512_cmpge_epu8_mask(v, thresh_vec) & live_mask;
                if (ge_mask == 0) continue;
                above_mask = has_above_threshold
                    ? (_mm512_cmpge_epu8_mask(v, thresh_plus_vec) & live_mask)
                    : 0;
                at_mask = ge_mask & ~above_mask;
            } else {
                if (!has_above_threshold) break;
                above_mask = _mm512_cmpge_epu8_mask(v, thresh_plus_vec) & live_mask;
                if (above_mask == 0) continue;
            }

            __mmask64 am = above_mask;
            while (am) {
                out_ptr[selected_total++] = (int)(i + __builtin_ctzll((uint64_t)am));
                selected_above++;
                am &= am - 1;
            }

            while (at_mask && selected_at < ties_to_select) {
                out_ptr[selected_total++] = (int)(i + __builtin_ctzll((uint64_t)at_mask));
                selected_at++;
                at_mask &= at_mask - 1;
            }

            if (selected_above >= info.n_above && selected_at >= ties_to_select) break;
        }
        for (size_t i = n_aligned;
             i < scan_n && (selected_above < info.n_above || selected_at < ties_to_select);
             ++i) {
            if (!live.is_live((uint32_t)i)) continue;
            uint8_t s = scores_[i];
            if (s > threshold) {
                out_ptr[selected_total++] = (int)i;
                selected_above++;
            } else if (s == (uint8_t)threshold && selected_at < ties_to_select) {
                out_ptr[selected_total++] = (int)i;
                selected_at++;
            }
        }

        result.resize(selected_total);
        finish_select_hist_stats(stats, selected_above, selected_at, result.size());
    }

    void select_top_n_range_avx512_into(size_t num_candidates,
                                        size_t begin,
                                        size_t end,
                                        SCSelectStats* stats,
                                        std::vector<int>& result) const {
        SelectCountInfo info;
        if (!build_select_count_info_range_avx512(num_candidates, begin, end, info)) {
            result.clear();
            if (stats) {
                *stats = SCSelectStats{};
                stats->requested = num_candidates;
            }
            return;
        }
        int threshold = info.threshold;
        const size_t ties_to_select = std::min(info.need_at, info.threshold_bucket);
        fill_select_count_stats(stats, num_candidates, info);

        const size_t expected_total = info.n_above + ties_to_select;
        result.resize(expected_total);
        int* out_ptr = result.data();
        size_t selected_total = 0;
        size_t selected_above = 0;
        size_t selected_at = 0;

        const __m512i thresh_vec = _mm512_set1_epi8((char)threshold);
        const bool has_above_threshold = threshold < 255;
        const __m512i thresh_plus_vec =
            _mm512_set1_epi8((char)(has_above_threshold ? threshold + 1 : 0));
        size_t i = begin;
        size_t vector_end = begin + ((end - begin) & ~63ULL);

        for (; i < vector_end; i += 64) {
            __m512i v = _mm512_loadu_si512(scores_.data() + i);
            __mmask64 above_mask = 0;
            __mmask64 at_mask = 0;
            if (selected_at < ties_to_select) {
                __mmask64 ge_mask = _mm512_cmpge_epu8_mask(v, thresh_vec);
                if (ge_mask == 0) continue;
                above_mask = has_above_threshold
                    ? _mm512_cmpge_epu8_mask(v, thresh_plus_vec)
                    : 0;
                at_mask = ge_mask & ~above_mask;
            } else {
                if (!has_above_threshold) break;
                above_mask = _mm512_cmpge_epu8_mask(v, thresh_plus_vec);
                if (above_mask == 0) continue;
            }

            __mmask64 am = above_mask;
            while (am) {
                out_ptr[selected_total++] = (int)(i + __builtin_ctzll((uint64_t)am));
                selected_above++;
                am &= am - 1;
            }

            while (at_mask && selected_at < ties_to_select) {
                out_ptr[selected_total++] = (int)(i + __builtin_ctzll((uint64_t)at_mask));
                selected_at++;
                at_mask &= at_mask - 1;
            }

            if (selected_above >= info.n_above && selected_at >= ties_to_select) break;
        }
        for (; i < end && (selected_above < info.n_above || selected_at < ties_to_select); ++i) {
            uint8_t s = scores_[i];
            if (s > threshold) {
                out_ptr[selected_total++] = (int)i;
                selected_above++;
            } else if (s == (uint8_t)threshold && selected_at < ties_to_select) {
                out_ptr[selected_total++] = (int)i;
                selected_at++;
            }
        }

        result.resize(selected_total);
        finish_select_hist_stats(stats, selected_above, selected_at, result.size());
    }

#endif

    // Scalar (non-AVX512) top-N selection. The three public variants below differ
    // only in scan range, an optional live-pid predicate, and whether an
    // all-zero score board resets the stats struct — captured here as parameters
    // so the histogram/threshold/two-pass-select logic has one source of truth.
    template <typename Pred>
    void select_top_n_scalar_core(size_t num_candidates, size_t begin, size_t end,
                                  Pred include, bool reset_stats_on_empty,
                                  SCSelectStats* stats,
                                  std::vector<int>& result) const {
        uint32_t hist[256] = {};
        int max_score = 0;
        for (size_t i = begin; i < end; ++i) {
            if (!include(i)) continue;
            uint8_t s = scores_[i];
            hist[s]++;
            if (s > max_score) max_score = s;
        }

        if (max_score == 0) {
            result.clear();
            if (reset_stats_on_empty && stats) {
                *stats = SCSelectStats{};
                stats->requested = num_candidates;
            }
            return;
        }

        size_t accumulated = 0;
        int threshold = 0;
        for (int s = max_score; s > 0; --s) {
            if (accumulated + hist[s] >= num_candidates) {
                threshold = s;
                break;
            }
            accumulated += hist[s];
        }
        if (threshold == 0 && accumulated < num_candidates) threshold = 1;
        size_t n_above = 0;
        for (int s = max_score; s > threshold; --s) n_above += hist[s];
        size_t need_at = (num_candidates > n_above) ? num_candidates - n_above : 0;
        fill_select_hist_stats(stats, num_candidates, hist, max_score, threshold, n_above, need_at);

        result.clear();
        result.reserve(num_candidates);
        size_t selected_above = 0;
        size_t selected_at = 0;
        for (size_t i = begin; i < end && result.size() < num_candidates; ++i) {
            if (!include(i)) continue;
            if (scores_[i] > threshold) {
                result.push_back((int)i);
                selected_above++;
            }
        }
        for (size_t i = begin; i < end && result.size() < num_candidates; ++i) {
            if (!include(i)) continue;
            if (scores_[i] == threshold) {
                result.push_back((int)i);
                selected_at++;
            }
        }
        finish_select_hist_stats(stats, selected_above, selected_at, result.size());
    }

    void select_top_n_scalar_into(size_t num_candidates,
                                  SCSelectStats* stats,
                                  std::vector<int>& result) const {
        select_top_n_scalar_core(num_candidates, 0, n_,
                                 [](size_t) { return true; }, false, stats, result);
    }

    void select_top_n_live_scalar_into(size_t num_candidates,
                                       const LiveTable& live,
                                       SCSelectStats* stats,
                                       std::vector<int>& result) const {
        if ((size_t)live.max_pid() > n_) {
            throw std::invalid_argument("LiveTable max_pid exceeds SCScore size");
        }
        select_top_n_scalar_core(num_candidates, 0, live.max_pid(),
                                 [&](size_t i) { return live.is_live((uint32_t)i); },
                                 true, stats, result);
    }

    void select_top_n_range_scalar_into(size_t num_candidates,
                                        size_t begin,
                                        size_t end,
                                        SCSelectStats* stats,
                                        std::vector<int>& result) const {
        select_top_n_scalar_core(num_candidates, begin, end,
                                 [](size_t) { return true; }, true, stats, result);
    }

    size_t n_ = 0;
    hugepage_vector_uint8 scores_;
};

} // namespace score
