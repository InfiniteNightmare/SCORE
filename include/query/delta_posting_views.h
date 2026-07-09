#pragma once

#include "query/unsealed_delta_view.h"
#include "update/update_delta.h"

#include <cstdint>
#include <vector>

namespace score {
// Common base for the single-layer posting views. Holds the per-subspace CSR
// (posting_offsets / posting_ids) plus the cell accessors that Phase 4 scans;
// derived views only provide their own build(). UnsealedDeltaView (the L0
// multi-layer view) does NOT derive from this — it has a layered layout.
struct PostingView {
    struct CellSpan {
        const uint32_t* data = nullptr;
        size_t count = 0;
    };

    std::vector<std::vector<uint32_t>> posting_offsets;
    std::vector<std::vector<uint32_t>> posting_ids;
    uint64_t total_postings = 0;

    bool empty() const { return total_postings == 0; }

    size_t cell_size(int m, int cell) const {
        if (m < 0 || (size_t)m >= posting_offsets.size()) return 0;
        const std::vector<uint32_t>& offsets = posting_offsets[(size_t)m];
        if (cell < 0 || (size_t)cell + 1 >= offsets.size()) return 0;
        return (size_t)(offsets[(size_t)cell + 1] - offsets[(size_t)cell]);
    }

    const uint32_t* cell_data(int m, int cell) const {
        if (m < 0 || (size_t)m >= posting_offsets.size()) return nullptr;
        const std::vector<uint32_t>& offsets = posting_offsets[(size_t)m];
        if (cell < 0 || (size_t)cell + 1 >= offsets.size()) return nullptr;
        const std::vector<uint32_t>& ids = posting_ids[(size_t)m];
        return ids.data() + offsets[(size_t)cell];
    }

    CellSpan cell_span(int m, int cell) const {
        if (m < 0 || (size_t)m >= posting_offsets.size()) return {};
        const std::vector<uint32_t>& offsets = posting_offsets[(size_t)m];
        if (cell < 0 || (size_t)cell + 1 >= offsets.size()) return {};
        const uint32_t begin = offsets[(size_t)cell];
        const uint32_t end = offsets[(size_t)cell + 1];
        if (begin == end) return {};
        const std::vector<uint32_t>& ids = posting_ids[(size_t)m];
        return {ids.data() + begin, (size_t)(end - begin)};
    }
};

// Live-filtered posting view over the base (L2) segment: base postings with
// tombstoned pids removed. Built only when updates exist; the raw PostingLists
// is used directly when the index has no updates.
struct BasePostingView : PostingView {
    void build(const PostingLists& base_postings,
               int M,
               const LiveTable& live,
               uint32_t base_n) {
        posting_offsets.clear();
        posting_ids.clear();
        total_postings = 0;
        if (M <= 0 || base_n == 0) return;

        posting_offsets.resize((size_t)M);
        posting_ids.resize((size_t)M);
        for (int m = 0; m < M; ++m) {
            int K_int = base_postings.K(m);
            if (K_int <= 0) {
                throw std::runtime_error("base live posting overlay saw invalid K");
            }
            size_t K = (size_t)K_int;
            if (K > (size_t)std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("base live posting overlay cell count exceeds uint32 range");
            }
            std::vector<uint32_t> sizes(K, 0);
            uint64_t subspace_total = 0;
            for (size_t cell = 0; cell < K; ++cell) {
                const uint32_t* data = base_postings.cell_data(m, (int)cell);
                size_t count = base_postings.cell_size(m, (int)cell);
                uint32_t live_count = 0;
                for (size_t i = 0; i < count; ++i) {
                    uint32_t pid = data[i];
                    if (pid >= base_n) {
                        throw std::runtime_error("base posting PID exceeds base_n");
                    }
                    if (live.is_live(pid)) ++live_count;
                }
                sizes[cell] = live_count;
                subspace_total += live_count;
            }
            if (subspace_total > (uint64_t)std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("base live posting overlay subspace exceeds uint32 range");
            }

            std::vector<uint32_t>& offsets = posting_offsets[(size_t)m];
            offsets.assign(K + 1, 0);
            for (size_t cell = 0; cell < K; ++cell) {
                offsets[cell + 1] = offsets[cell] + sizes[cell];
            }
            std::vector<uint32_t>& ids = posting_ids[(size_t)m];
            ids.assign((size_t)subspace_total, 0);

            std::vector<uint32_t> cursors(K, 0);
            for (size_t cell = 0; cell < K; ++cell) {
                const uint32_t* data = base_postings.cell_data(m, (int)cell);
                size_t count = base_postings.cell_size(m, (int)cell);
                uint32_t offset = offsets[cell];
                for (size_t i = 0; i < count; ++i) {
                    uint32_t pid = data[i];
                    if (!live.is_live(pid)) continue;
                    ids[(size_t)(offset + cursors[cell]++)] = pid;
                }
            }
            total_postings += subspace_total;
        }
    }
};

// Merged posting view across all sealed delta (L1) segments: per-cell union of
// live pids from every sealed segment. Built at publish time so Phase 4 sees one
// logical posting layer regardless of how many sealed segments exist.
struct SealedDeltaPostingView : PostingView {
    void build(const std::vector<std::shared_ptr<const SealedDeltaSegment>>& sealed_deltas,
               const std::vector<size_t>& cells_per_subspace,
               const LiveTable* live) {
        posting_offsets.clear();
        posting_ids.clear();
        total_postings = 0;
        if (sealed_deltas.empty() || cells_per_subspace.empty()) return;

        posting_offsets.resize(cells_per_subspace.size());
        posting_ids.resize(cells_per_subspace.size());
        for (size_t m = 0; m < cells_per_subspace.size(); ++m) {
            size_t K = cells_per_subspace[m];
            if (K > (size_t)std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("sealed delta posting overlay cell count exceeds uint32 range");
            }
            std::vector<uint32_t> sizes(K, 0);
            uint64_t subspace_total = 0;
            for (const auto& segment : sealed_deltas) {
                if (!segment) continue;
                for (size_t cell = 0; cell < K; ++cell) {
                    size_t count = segment->cell_size((uint32_t)m, (uint32_t)cell);
                    if (count == 0) continue;
                    const uint32_t* data = segment->cell_data((uint32_t)m, (uint32_t)cell);
                    if (live != nullptr && segment->live_count() != segment->row_count()) {
                        uint32_t live_count = 0;
                        for (size_t i = 0; i < count; ++i) {
                            if (live->is_live(data[i])) ++live_count;
                        }
                        if (live_count > std::numeric_limits<uint32_t>::max() - sizes[cell]) {
                            throw std::runtime_error("sealed delta posting overlay live cell size exceeds uint32 range");
                        }
                        sizes[cell] += live_count;
                        subspace_total += live_count;
                    } else {
                        if (count > (size_t)std::numeric_limits<uint32_t>::max() - sizes[cell]) {
                            throw std::runtime_error("sealed delta posting overlay cell size exceeds uint32 range");
                        }
                        sizes[cell] += (uint32_t)count;
                        subspace_total += count;
                    }
                }
            }
            if (subspace_total > (uint64_t)std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("sealed delta posting overlay subspace exceeds uint32 range");
            }

            std::vector<uint32_t>& offsets = posting_offsets[m];
            offsets.assign(K + 1, 0);
            for (size_t cell = 0; cell < K; ++cell) {
                offsets[cell + 1] = offsets[cell] + sizes[cell];
            }
            posting_ids[m].assign((size_t)subspace_total, 0);

            std::vector<uint32_t> cursors(K, 0);
            for (const auto& segment : sealed_deltas) {
                if (!segment) continue;
                for (size_t cell = 0; cell < K; ++cell) {
                    size_t count = segment->cell_size((uint32_t)m, (uint32_t)cell);
                    if (count == 0) continue;
                    const uint32_t* data = segment->cell_data((uint32_t)m, (uint32_t)cell);
                    uint32_t offset = offsets[cell] + cursors[cell];
                    if (live != nullptr && segment->live_count() != segment->row_count()) {
                        for (size_t i = 0; i < count; ++i) {
                            uint32_t pid = data[i];
                            if (!live->is_live(pid)) continue;
                            posting_ids[m][(size_t)offset++] = pid;
                            ++cursors[cell];
                        }
                    } else {
                        std::copy(data,
                                  data + count,
                                  posting_ids[m].begin() + (std::ptrdiff_t)offset);
                        cursors[cell] += (uint32_t)count;
                    }
                }
            }
            total_postings += subspace_total;
        }
    }

    void build_appending_to_previous(
        const SealedDeltaPostingView& previous,
        const std::vector<std::shared_ptr<const SealedDeltaSegment>>& appended_deltas,
        const std::vector<size_t>& cells_per_subspace,
        const LiveTable* live) {
        posting_offsets.clear();
        posting_ids.clear();
        total_postings = 0;
        if (cells_per_subspace.empty()) return;
        if (previous.posting_offsets.size() != cells_per_subspace.size() ||
            previous.posting_ids.size() != cells_per_subspace.size()) {
            throw std::runtime_error("sealed delta posting append saw incompatible previous view");
        }

        posting_offsets.resize(cells_per_subspace.size());
        posting_ids.resize(cells_per_subspace.size());
        for (size_t m = 0; m < cells_per_subspace.size(); ++m) {
            size_t K = cells_per_subspace[m];
            if (K > (size_t)std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("sealed delta posting append cell count exceeds uint32 range");
            }
            const std::vector<uint32_t>& previous_offsets = previous.posting_offsets[m];
            const std::vector<uint32_t>& previous_ids = previous.posting_ids[m];
            if (previous_offsets.size() != K + 1 ||
                previous_offsets.empty() ||
                previous_offsets.back() != previous_ids.size()) {
                throw std::runtime_error("sealed delta posting append saw invalid previous offsets");
            }

            std::vector<uint32_t> appended_sizes(K, 0);
            uint64_t appended_total = 0;
            for (const auto& segment : appended_deltas) {
                if (!segment) continue;
                for (size_t cell = 0; cell < K; ++cell) {
                    size_t count = segment->cell_size((uint32_t)m, (uint32_t)cell);
                    if (count == 0) continue;
                    const uint32_t* data = segment->cell_data((uint32_t)m, (uint32_t)cell);
                    if (live != nullptr && segment->live_count() != segment->row_count()) {
                        uint32_t live_count = 0;
                        for (size_t i = 0; i < count; ++i) {
                            if (live->is_live(data[i])) ++live_count;
                        }
                        if (live_count >
                            std::numeric_limits<uint32_t>::max() - appended_sizes[cell]) {
                            throw std::runtime_error("sealed delta posting append live cell size exceeds uint32 range");
                        }
                        appended_sizes[cell] += live_count;
                        appended_total += live_count;
                    } else {
                        if (count >
                            (size_t)std::numeric_limits<uint32_t>::max() -
                                (size_t)appended_sizes[cell]) {
                            throw std::runtime_error("sealed delta posting append cell size exceeds uint32 range");
                        }
                        appended_sizes[cell] += (uint32_t)count;
                        appended_total += count;
                    }
                }
            }
            uint64_t subspace_total = (uint64_t)previous_ids.size() + appended_total;
            if (subspace_total > (uint64_t)std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("sealed delta posting append subspace exceeds uint32 range");
            }

            std::vector<uint32_t>& offsets = posting_offsets[m];
            offsets.assign(K + 1, 0);
            for (size_t cell = 0; cell < K; ++cell) {
                uint32_t previous_count = previous_offsets[cell + 1] - previous_offsets[cell];
                if (previous_count >
                    std::numeric_limits<uint32_t>::max() - appended_sizes[cell]) {
                    throw std::runtime_error("sealed delta posting append cell size exceeds uint32 range");
                }
                uint32_t cell_count = previous_count + appended_sizes[cell];
                if (cell_count > std::numeric_limits<uint32_t>::max() - offsets[cell]) {
                    throw std::runtime_error("sealed delta posting append offsets exceed uint32 range");
                }
                offsets[cell + 1] = offsets[cell] + cell_count;
            }
            posting_ids[m].assign((size_t)subspace_total, 0);

            std::vector<uint32_t> write_offsets(K, 0);
            for (size_t cell = 0; cell < K; ++cell) {
                uint32_t previous_begin = previous_offsets[cell];
                uint32_t previous_end = previous_offsets[cell + 1];
                uint32_t previous_count = previous_end - previous_begin;
                uint32_t offset = offsets[cell];
                if (previous_count != 0) {
                    std::copy(previous_ids.begin() + (std::ptrdiff_t)previous_begin,
                              previous_ids.begin() + (std::ptrdiff_t)previous_end,
                              posting_ids[m].begin() + (std::ptrdiff_t)offset);
                }
                write_offsets[cell] = offset + previous_count;
            }
            for (const auto& segment : appended_deltas) {
                if (!segment) continue;
                for (size_t cell = 0; cell < K; ++cell) {
                    size_t count = segment->cell_size((uint32_t)m, (uint32_t)cell);
                    if (count == 0) continue;
                    const uint32_t* data = segment->cell_data((uint32_t)m, (uint32_t)cell);
                    uint32_t offset = write_offsets[cell];
                    if (live != nullptr && segment->live_count() != segment->row_count()) {
                        for (size_t i = 0; i < count; ++i) {
                            uint32_t pid = data[i];
                            if (!live->is_live(pid)) continue;
                            posting_ids[m][(size_t)offset++] = pid;
                        }
                        write_offsets[cell] = offset;
                    } else {
                        std::copy(data,
                                  data + count,
                                  posting_ids[m].begin() + (std::ptrdiff_t)offset);
                        write_offsets[cell] += (uint32_t)count;
                    }
                }
            }
            for (size_t cell = 0; cell < K; ++cell) {
                if (write_offsets[cell] != offsets[cell + 1]) {
                    throw std::runtime_error("sealed delta posting append wrote wrong cell size");
                }
            }
            total_postings += subspace_total;
        }
    }
};

struct SealedDeltaPidLookup {
    struct Range {
        uint32_t begin = 0;
        uint32_t end = 0;
        uint32_t segment_index = 0;
    };

    uint32_t dense_begin = 0;
    std::vector<uint32_t> dense_segment_indices;
    std::vector<Range> ranges;
    static constexpr uint32_t kMissingSegmentIndex = std::numeric_limits<uint32_t>::max();

    bool empty() const {
        return dense_segment_indices.empty() && ranges.empty();
    }

    void build(const std::vector<std::shared_ptr<const SealedDeltaSegment>>& sealed_deltas) {
        dense_begin = 0;
        dense_segment_indices.clear();
        ranges.clear();
        if (sealed_deltas.size() <= 1) return;

        uint64_t min_pid = std::numeric_limits<uint64_t>::max();
        uint64_t max_pid_end = 0;
        uint64_t total_pid_range = 0;
        uint64_t total_rows = 0;
        ranges.reserve(sealed_deltas.size());
        for (size_t i = 0; i < sealed_deltas.size(); ++i) {
            const auto& segment = sealed_deltas[i];
            if (!segment || segment->pid_count() == 0) continue;
            uint64_t begin = segment->pid_begin();
            uint64_t end = begin + (uint64_t)segment->pid_count();
            if (end > (uint64_t)std::numeric_limits<uint32_t>::max() + 1ull) {
                throw std::runtime_error("sealed delta PID lookup range exceeds uint32 range");
            }
            min_pid = std::min(min_pid, begin);
            max_pid_end = std::max(max_pid_end, end);
            total_pid_range += segment->pid_count();
            total_rows += segment->row_count();
            ranges.push_back({(uint32_t)begin, (uint32_t)end, (uint32_t)i});
        }
        if (ranges.empty()) return;

        std::sort(ranges.begin(), ranges.end(),
                  [](const Range& a, const Range& b) {
                      if (a.begin != b.begin) return a.begin < b.begin;
                      return a.end < b.end;
                  });

        uint64_t span = max_pid_end - min_pid;
        constexpr uint64_t kMaxDenseLookupBytes = 64ull * 1024ull * 1024ull;
        uint64_t dense_bytes = span * sizeof(uint32_t);
        uint64_t density_limit = std::max<uint64_t>(total_pid_range * 4ull, total_rows * 8ull + 1024ull);
        if (span > 0 &&
            span <= (uint64_t)std::numeric_limits<size_t>::max() &&
            span <= density_limit &&
            dense_bytes <= kMaxDenseLookupBytes) {
            dense_begin = (uint32_t)min_pid;
            dense_segment_indices.assign((size_t)span, kMissingSegmentIndex);
            for (size_t segment_index = 0; segment_index < sealed_deltas.size(); ++segment_index) {
                const auto& segment = sealed_deltas[segment_index];
                if (!segment) continue;
                for (uint32_t row = 0; row < segment->row_count(); ++row) {
                    uint32_t pid = segment->pid(row);
                    size_t offset = (size_t)((uint64_t)pid - (uint64_t)dense_begin);
                    if (offset >= dense_segment_indices.size()) {
                        throw std::runtime_error("sealed delta PID lookup row is outside dense range");
                    }
                    if (dense_segment_indices[offset] != kMissingSegmentIndex) {
                        throw std::runtime_error("sealed delta PID lookup found duplicate PID");
                    }
                    dense_segment_indices[offset] = (uint32_t)segment_index;
                }
            }
            ranges.clear();
        }
    }

    bool segment_index_for_pid(uint32_t pid, uint32_t& segment_index_out) const {
        if (!dense_segment_indices.empty()) {
            if (pid < dense_begin) return false;
            uint64_t offset = (uint64_t)pid - (uint64_t)dense_begin;
            if (offset >= (uint64_t)dense_segment_indices.size()) return false;
            uint32_t segment_index = dense_segment_indices[(size_t)offset];
            if (segment_index == kMissingSegmentIndex) return false;
            segment_index_out = segment_index;
            return true;
        }
        auto it = std::upper_bound(
            ranges.begin(),
            ranges.end(),
            pid,
            [](uint32_t value, const Range& range) {
                return value < range.begin;
            });
        if (it == ranges.begin()) return false;
        --it;
        if (pid >= it->begin && pid < it->end) {
            segment_index_out = it->segment_index;
            return true;
        }
        return false;
    }
};

enum class CandidateSelectionStrategy : uint8_t {
    BaseAllLive,
    UpdateAllLive,
    LiveSuffixRange,
    DeadScoresFiltered,
    DeadPidClear,
    LiveBitset,
};
} // namespace score
