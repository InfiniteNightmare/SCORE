#pragma once

#include "core/common.h"
#include "update/update_delta.h"

#include <cstdint>
#include <vector>

namespace score {
struct UnsealedDeltaView {
    struct RowRef {
        const L0Block* block = nullptr;
        uint32_t local_row = 0;
    };

    struct CellSpan {
        const uint32_t* data = nullptr;
        size_t count = 0;
    };

    using PostingVectors = std::vector<std::vector<uint32_t>>;

    struct PostingLayer {
        std::shared_ptr<const PostingVectors> offsets;
        std::shared_ptr<const PostingVectors> ids;
        uint32_t level = 0;
        uint64_t total_ids = 0;
    };

    std::vector<std::shared_ptr<const L0Block>> committed_blocks;
    std::vector<PostingLayer> posting_layers;
    std::vector<std::vector<uint32_t>> posting_cell_counts;
    bool empty() const { return committed_blocks.empty(); }
    static constexpr uint32_t kBulkLayerLevel = 31;
    static constexpr uint64_t kContiguousLayerMaxIds = 1ull << 20;

    void build_pid_lookup(const LiveTable* live = nullptr) {
        block_pid_begins.clear();
        block_lookup_indices.clear();
        pid_lookup_begin = 0;
        pid_to_block_lookup.clear();
        block_pid_begins.reserve(committed_blocks.size());
        block_lookup_indices.reserve(committed_blocks.size());
        all_pids_live_ = true;
        uint32_t min_pid = std::numeric_limits<uint32_t>::max();
        uint32_t max_pid = 0;
        bool ordered_dense_ranges = true;
        uint32_t previous_range_end = 0;
        bool have_previous_range = false;
        for (size_t i = 0; i < committed_blocks.size(); ++i) {
            const auto& block = committed_blocks[i];
            if (!block) continue;
            block_pid_begins.push_back(block->pid_begin());
            block_lookup_indices.push_back(i);
            min_pid = std::min(min_pid, block->pid_begin());
            max_pid = std::max(max_pid, block->pid_end());
            if (!block->dense_pid_order() ||
                (have_previous_range && block->pid_begin() < previous_range_end)) {
                ordered_dense_ranges = false;
            }
            previous_range_end = block->pid_end();
            have_previous_range = true;
            if (live) {
                if (block->dense_pid_order()) {
                    if (!live->range_all_live(block->pid_begin(), block->pid_end())) {
                        all_pids_live_ = false;
                    }
                } else {
                    for (uint32_t row = 0; row < block->row_count(); ++row) {
                        if (!live->is_live(block->pid(row))) {
                            all_pids_live_ = false;
                            break;
                        }
                    }
                }
            }
        }
        if (!ordered_dense_ranges) {
            block_pid_begins.clear();
            block_lookup_indices.clear();
        }
        if (min_pid != std::numeric_limits<uint32_t>::max() && max_pid > min_pid) {
            uint64_t span = (uint64_t)max_pid - (uint64_t)min_pid;
            if (span <= (uint64_t)dense_pid_lookup_max_span()) {
                pid_lookup_begin = min_pid;
                pid_to_block_lookup.assign((size_t)span, kMissingBlockIndex);
                for (uint32_t block_idx = 0;
                     (size_t)block_idx < committed_blocks.size();
                     ++block_idx) {
                    const auto& block = committed_blocks[(size_t)block_idx];
                    if (!block) continue;
                    if (block->dense_pid_order()) {
                        for (uint32_t pid = block->pid_begin(); pid < block->pid_end(); ++pid) {
                            pid_to_block_lookup[(size_t)(pid - pid_lookup_begin)] = block_idx;
                        }
                    } else {
                        for (uint32_t row = 0; row < block->row_count(); ++row) {
                            uint32_t pid = block->pid(row);
                            pid_to_block_lookup[(size_t)(pid - pid_lookup_begin)] = block_idx;
                        }
                    }
                }
            }
        }
    }

    bool build_pid_lookup_incremental_from_previous(const UnsealedDeltaView& previous,
                                                    size_t tail_begin,
                                                    const LiveTable* live = nullptr) {
        if (tail_begin > committed_blocks.size() ||
            tail_begin != previous.committed_blocks.size()) {
            return false;
        }
        block_pid_begins = previous.block_pid_begins;
        block_lookup_indices = previous.block_lookup_indices;
        pid_lookup_begin = previous.pid_lookup_begin;
        pid_to_block_lookup = previous.pid_to_block_lookup;
        all_pids_live_ = previous.all_pids_live_;

        for (size_t block_idx = tail_begin; block_idx < committed_blocks.size(); ++block_idx) {
            const auto& block = committed_blocks[block_idx];
            if (!block) continue;
            if (!block_pid_begins.empty() &&
                block->pid_begin() < block_pid_begins.back()) {
                return false;
            }
            if (!block->dense_pid_order()) return false;
            block_pid_begins.push_back(block->pid_begin());
            block_lookup_indices.push_back(block_idx);
            if (live && !live->range_all_live(block->pid_begin(), block->pid_end())) {
                all_pids_live_ = false;
            }
        }

        if (committed_blocks.empty()) return true;
        if (pid_to_block_lookup.empty()) return true;
        uint64_t lookup_begin = pid_lookup_begin;
        uint64_t lookup_end = lookup_begin + (uint64_t)pid_to_block_lookup.size();
        for (size_t block_idx = tail_begin; block_idx < committed_blocks.size(); ++block_idx) {
            const auto& block = committed_blocks[block_idx];
            if (!block) continue;
            uint64_t begin = block->pid_begin();
            uint64_t end = block->pid_end();
            if (begin < lookup_begin) return false;
            if (end > (uint64_t)std::numeric_limits<uint32_t>::max()) return false;
            if (end > lookup_end) {
                if (end - lookup_begin > (uint64_t)dense_pid_lookup_max_span()) {
                    pid_lookup_begin = 0;
                    pid_to_block_lookup.clear();
                    return true;
                }
                size_t old_size = pid_to_block_lookup.size();
                size_t new_size = (size_t)(end - lookup_begin);
                pid_to_block_lookup.resize(new_size, kMissingBlockIndex);
                lookup_end = end;
                (void)old_size;
            }
            if (block->dense_pid_order()) {
                for (uint32_t pid = block->pid_begin(); pid < block->pid_end(); ++pid) {
                    pid_to_block_lookup[(size_t)((uint64_t)pid - lookup_begin)] =
                        (uint32_t)block_idx;
                }
            } else {
                for (uint32_t row = 0; row < block->row_count(); ++row) {
                    uint32_t pid = block->pid(row);
                    pid_to_block_lookup[(size_t)((uint64_t)pid - lookup_begin)] =
                        (uint32_t)block_idx;
                }
            }
        }
        return true;
    }

    void refresh_all_pids_live(const LiveTable* live) {
        all_pids_live_ = true;
        if (live == nullptr) return;
        for (const auto& block : committed_blocks) {
            if (!block) continue;
            if (block->dense_pid_order()) {
                if (!live->range_all_live(block->pid_begin(), block->pid_end())) {
                    all_pids_live_ = false;
                    return;
                }
            } else {
                for (uint32_t row = 0; row < block->row_count(); ++row) {
                    if (!live->is_live(block->pid(row))) {
                        all_pids_live_ = false;
                        return;
                    }
                }
            }
        }
    }

    template <typename Fn>
    void for_each_cell_span(uint32_t subspace, uint32_t cell, Fn&& fn) const {
        if (!posting_layers.empty()) {
            for (const PostingLayer& layer : posting_layers) {
                if (!layer.offsets || !layer.ids ||
                    (size_t)subspace >= layer.offsets->size() ||
                    (size_t)subspace >= layer.ids->size()) {
                    continue;
                }
                const std::vector<uint32_t>& offsets = (*layer.offsets)[(size_t)subspace];
                if ((size_t)cell + 1 >= offsets.size()) continue;
                const uint32_t begin = offsets[(size_t)cell];
                const uint32_t end = offsets[(size_t)cell + 1];
                if (begin == end) continue;
                const std::vector<uint32_t>& ids = (*layer.ids)[(size_t)subspace];
                fn(CellSpan{ids.data() + begin, (size_t)(end - begin)});
            }
            return;
        }
        CellSpan span = cell_span(subspace, cell);
        if (span.count != 0) fn(span);
    }

    void rebuild_live_cell_counts(const std::vector<size_t>& cells_per_subspace,
                                  const LiveTable* live) {
        live_posting_counts.clear();
        has_live_posting_counts_ = false;
        if (live == nullptr || all_pids_live_) return;
        live_posting_counts.resize(cells_per_subspace.size());
        for (size_t m = 0; m < cells_per_subspace.size(); ++m) {
            size_t K = cells_per_subspace[m];
            std::vector<uint32_t>& counts = live_posting_counts[m];
            counts.assign(K, 0);
            for (size_t cell = 0; cell < K; ++cell) {
                uint32_t live_count = 0;
                for_each_cell_span((uint32_t)m, (uint32_t)cell, [&](CellSpan span) {
                    for (size_t i = 0; i < span.count; ++i) {
                        if (live->is_live(span.data[i])) ++live_count;
                    }
                });
                counts[cell] = live_count;
            }
        }
        has_live_posting_counts_ = true;
    }

    bool try_rebuild_live_cell_counts_incremental(
        const UnsealedDeltaView& previous,
        size_t tail_begin,
        const std::vector<size_t>& cells_per_subspace,
        const LiveTable* previous_live,
        const LiveTable* live,
        const std::vector<uint32_t>* dead_pids) {
        live_posting_counts.clear();
        has_live_posting_counts_ = false;
        if (live == nullptr || all_pids_live_) return true;
        if (dead_pids == nullptr && tail_begin == committed_blocks.size() &&
            previous_live != live) {
            return false;
        }

        if (previous.has_live_posting_counts_) {
            live_posting_counts = previous.live_posting_counts;
        } else if (previous.all_pids_live_ &&
                   previous.posting_cell_counts.size() == cells_per_subspace.size()) {
            live_posting_counts = previous.posting_cell_counts;
        } else {
            return false;
        }
        if (live_posting_counts.size() != cells_per_subspace.size()) {
            live_posting_counts.clear();
            return false;
        }
        for (size_t m = 0; m < cells_per_subspace.size(); ++m) {
            if (live_posting_counts[m].size() != cells_per_subspace[m]) {
                live_posting_counts.clear();
                return false;
            }
        }

        if (dead_pids != nullptr) {
            for (uint32_t pid : *dead_pids) {
                if (previous_live != nullptr && !previous_live->is_live(pid)) continue;
                if (live->is_live(pid)) continue;
                RowRef ref;
                if (!previous.resolve_pid(pid, ref)) continue;
                if (ref.block == nullptr || !ref.block->has_row_cells()) {
                    live_posting_counts.clear();
                    return false;
                }
                for (size_t m = 0; m < cells_per_subspace.size(); ++m) {
                    uint16_t cell = ref.block->row_cell(ref.local_row, (uint32_t)m);
                    if ((size_t)cell >= live_posting_counts[m].size() ||
                        live_posting_counts[m][(size_t)cell] == 0) {
                        live_posting_counts.clear();
                        return false;
                    }
                    --live_posting_counts[m][(size_t)cell];
                }
            }
        }

        for (size_t block_idx = tail_begin; block_idx < committed_blocks.size(); ++block_idx) {
            const auto& block = committed_blocks[block_idx];
            if (!block) continue;
            if (block->has_row_cells()) {
                for (uint32_t row = 0; row < block->row_count(); ++row) {
                    uint32_t pid = block->pid(row);
                    if (!live->is_live(pid)) continue;
                    for (size_t m = 0; m < cells_per_subspace.size(); ++m) {
                        uint16_t cell = block->row_cell(row, (uint32_t)m);
                        if ((size_t)cell >= live_posting_counts[m].size() ||
                            live_posting_counts[m][(size_t)cell] ==
                                std::numeric_limits<uint32_t>::max()) {
                            live_posting_counts.clear();
                            return false;
                        }
                        ++live_posting_counts[m][(size_t)cell];
                    }
                }
            } else {
                if (!live->range_all_live(block->pid_begin(), block->pid_end())) {
                    live_posting_counts.clear();
                    return false;
                }
                for (size_t m = 0; m < cells_per_subspace.size(); ++m) {
                    for (size_t cell = 0; cell < cells_per_subspace[m]; ++cell) {
                        size_t count = block->cell_size((uint32_t)m, (uint32_t)cell);
                        if (count >
                            (size_t)std::numeric_limits<uint32_t>::max() -
                                (size_t)live_posting_counts[m][cell]) {
                            live_posting_counts.clear();
                            return false;
                        }
                        live_posting_counts[m][cell] += (uint32_t)count;
                    }
                }
            }
        }

        has_live_posting_counts_ = true;
        return true;
    }

    void rebuild_posting_cell_counts(const std::vector<size_t>& cells_per_subspace) {
        posting_cell_counts.clear();
        if (posting_layers.empty()) return;
        posting_cell_counts.resize(cells_per_subspace.size());
        for (size_t m = 0; m < cells_per_subspace.size(); ++m) {
            size_t K = cells_per_subspace[m];
            std::vector<uint32_t>& counts = posting_cell_counts[m];
            counts.assign(K, 0);
            for (const PostingLayer& layer : posting_layers) {
                for (size_t cell = 0; cell < K; ++cell) {
                    size_t count = layer_cell_size(layer, (uint32_t)m, (uint32_t)cell);
                    if (count >
                        (size_t)std::numeric_limits<uint32_t>::max() -
                            (size_t)counts[cell]) {
                        throw std::runtime_error("L0 posting cell count exceeds uint32 range");
                    }
                    counts[cell] += (uint32_t)count;
                }
            }
        }
    }

    uint64_t posting_layer_total_ids() const {
        uint64_t total = 0;
        for (const PostingLayer& layer : posting_layers) {
            if (layer.total_ids > std::numeric_limits<uint64_t>::max() - total) {
                throw std::runtime_error("L0 posting layer total exceeds uint64 range");
            }
            total += layer.total_ids;
        }
        return total;
    }

    static uint64_t posting_vectors_total_ids(const PostingVectors& ids) {
        uint64_t total = 0;
        for (const auto& subspace_ids : ids) {
            if ((uint64_t)subspace_ids.size() >
                std::numeric_limits<uint64_t>::max() - total) {
                throw std::runtime_error("L0 posting layer size exceeds uint64 range");
            }
            total += (uint64_t)subspace_ids.size();
        }
        return total;
    }

    static CellSpan layer_cell_span(const PostingLayer& layer,
                                    uint32_t subspace,
                                    uint32_t cell) {
        if (!layer.offsets || !layer.ids ||
            (size_t)subspace >= layer.offsets->size() ||
            (size_t)subspace >= layer.ids->size()) {
            return {};
        }
        const std::vector<uint32_t>& offsets = (*layer.offsets)[(size_t)subspace];
        if ((size_t)cell + 1 >= offsets.size()) return {};
        const uint32_t begin = offsets[(size_t)cell];
        const uint32_t end = offsets[(size_t)cell + 1];
        if (begin == end) return {};
        const std::vector<uint32_t>& ids = (*layer.ids)[(size_t)subspace];
        return {ids.data() + begin, (size_t)(end - begin)};
    }

    static size_t layer_cell_size(const PostingLayer& layer,
                                  uint32_t subspace,
                                  uint32_t cell) {
        if (!layer.offsets || (size_t)subspace >= layer.offsets->size()) return 0;
        const std::vector<uint32_t>& offsets = (*layer.offsets)[(size_t)subspace];
        if ((size_t)cell + 1 >= offsets.size()) return 0;
        return (size_t)(offsets[(size_t)cell + 1] - offsets[(size_t)cell]);
    }

    PostingLayer make_layer(PostingVectors&& offsets,
                            PostingVectors&& ids,
                            uint32_t level = 0) const {
        auto ids_ptr = std::make_shared<PostingVectors>(std::move(ids));
        auto offsets_ptr = std::make_shared<PostingVectors>(std::move(offsets));
        PostingLayer layer;
        layer.offsets = std::move(offsets_ptr);
        layer.ids = std::move(ids_ptr);
        layer.level = level;
        layer.total_ids = posting_vectors_total_ids(*layer.ids);
        return layer;
    }

    static PostingLayer merge_two_layers(const PostingLayer& older,
                                         const PostingLayer& newer,
                                         const std::vector<size_t>& cells_per_subspace) {
        PostingVectors offsets(cells_per_subspace.size());
        PostingVectors ids(cells_per_subspace.size());
        for (size_t m = 0; m < cells_per_subspace.size(); ++m) {
            size_t K = cells_per_subspace[m];
            std::vector<uint32_t> sizes(K, 0);
            size_t total = 0;
            for (size_t cell = 0; cell < K; ++cell) {
                size_t count = layer_cell_size(older, (uint32_t)m, (uint32_t)cell) +
                               layer_cell_size(newer, (uint32_t)m, (uint32_t)cell);
                if (count > (size_t)std::numeric_limits<uint32_t>::max()) {
                    throw std::runtime_error("merged L0 posting layer cell exceeds uint32 range");
                }
                sizes[cell] = (uint32_t)count;
                if (count > std::numeric_limits<size_t>::max() - total) {
                    throw std::runtime_error("merged L0 posting layer exceeds size_t range");
                }
                total += count;
            }
            offsets[m].assign(K + 1, 0);
            for (size_t cell = 0; cell < K; ++cell) {
                if (sizes[cell] > std::numeric_limits<uint32_t>::max() - offsets[m][cell]) {
                    throw std::runtime_error("merged L0 posting layer offsets exceed uint32 range");
                }
                offsets[m][cell + 1] = offsets[m][cell] + sizes[cell];
            }
            ids[m].assign(total, 0);
            std::vector<uint32_t> write_offsets = offsets[m];
            for (size_t cell = 0; cell < K; ++cell) {
                CellSpan older_span = layer_cell_span(older, (uint32_t)m, (uint32_t)cell);
                if (older_span.count != 0) {
                    uint32_t offset = write_offsets[cell];
                    std::copy(older_span.data,
                              older_span.data + older_span.count,
                              ids[m].begin() + (std::ptrdiff_t)offset);
                    write_offsets[cell] += (uint32_t)older_span.count;
                }
                CellSpan newer_span = layer_cell_span(newer, (uint32_t)m, (uint32_t)cell);
                if (newer_span.count != 0) {
                    uint32_t offset = write_offsets[cell];
                    std::copy(newer_span.data,
                              newer_span.data + newer_span.count,
                              ids[m].begin() + (std::ptrdiff_t)offset);
                    write_offsets[cell] += (uint32_t)newer_span.count;
                }
            }
        }
        PostingLayer merged;
        merged.ids = std::make_shared<PostingVectors>(std::move(ids));
        merged.offsets = std::make_shared<PostingVectors>(std::move(offsets));
        merged.level = std::max(older.level, newer.level) + 1;
        merged.total_ids = posting_vectors_total_ids(*merged.ids);
        return merged;
    }

    void normalize_layers(const std::vector<size_t>& cells_per_subspace) {
        if (posting_layers.empty()) return;
        std::vector<PostingLayer> stack;
        stack.reserve(posting_layers.size());
        uint64_t max_merge_bytes = layer_merge_max_posting_bytes();
        for (PostingLayer layer : posting_layers) {
            if (layer.total_ids == 0) continue;
            stack.push_back(std::move(layer));
            while (stack.size() >= 2) {
                PostingLayer& newer = stack.back();
                PostingLayer& older = stack[stack.size() - 2];
                if (older.level != newer.level) break;
                if (max_merge_bytes != 0 &&
                    layer_merge_posting_bytes(older, newer) > max_merge_bytes) {
                    break;
                }
                PostingLayer merged =
                    merge_two_layers(older, newer, cells_per_subspace);
                stack.pop_back();
                stack.pop_back();
                stack.push_back(std::move(merged));
            }
        }
        posting_layers = std::move(stack);
    }

    void consolidate_layers_to_single(const std::vector<size_t>& cells_per_subspace) {
        if (posting_layers.size() <= 1) return;
        PostingVectors offsets(cells_per_subspace.size());
        PostingVectors ids(cells_per_subspace.size());
        std::vector<std::vector<uint32_t>> consolidated_counts(cells_per_subspace.size());
        auto build_subspace = [&](size_t m) {
            size_t K = cells_per_subspace[m];
            std::vector<uint32_t> sizes(K, 0);
            size_t total = 0;
            for (size_t cell = 0; cell < K; ++cell) {
                size_t cell_total = 0;
                for (const PostingLayer& layer : posting_layers) {
                    size_t count = layer_cell_size(layer, (uint32_t)m, (uint32_t)cell);
                    if (count >
                        (size_t)std::numeric_limits<uint32_t>::max() - cell_total) {
                        throw std::runtime_error("consolidated L0 posting cell exceeds uint32 range");
                    }
                    cell_total += count;
                }
                sizes[cell] = (uint32_t)cell_total;
                if (cell_total > std::numeric_limits<size_t>::max() - total) {
                    throw std::runtime_error("consolidated L0 posting layer exceeds size_t range");
                }
                total += cell_total;
            }
            consolidated_counts[m] = sizes;
            offsets[m].assign(K + 1, 0);
            for (size_t cell = 0; cell < K; ++cell) {
                if (sizes[cell] > std::numeric_limits<uint32_t>::max() - offsets[m][cell]) {
                    throw std::runtime_error("consolidated L0 posting offsets exceed uint32 range");
                }
                offsets[m][cell + 1] = offsets[m][cell] + sizes[cell];
            }
            ids[m].assign(total, 0);
            std::vector<uint32_t> write_offsets = offsets[m];
            for (size_t cell = 0; cell < K; ++cell) {
                for (const PostingLayer& layer : posting_layers) {
                    CellSpan span = layer_cell_span(layer, (uint32_t)m, (uint32_t)cell);
                    if (span.count == 0) continue;
                    uint32_t offset = write_offsets[cell];
                    std::copy(span.data,
                              span.data + span.count,
                              ids[m].begin() + (std::ptrdiff_t)offset);
                    write_offsets[cell] += (uint32_t)span.count;
                }
            }
        };

        std::exception_ptr build_error;
        int subspace_count = (int)cells_per_subspace.size();
        int build_threads = std::min(subspace_count, std::min(omp_get_max_threads(), 4));
#pragma omp parallel for schedule(static) num_threads(build_threads) if(subspace_count >= 4 && build_threads > 1)
        for (int mi = 0; mi < subspace_count; ++mi) {
            try {
                build_subspace((size_t)mi);
            } catch (...) {
#pragma omp critical(score_growing_delta_consolidate_error)
                {
                    if (!build_error) build_error = std::current_exception();
                }
            }
        }
        if (build_error) std::rethrow_exception(build_error);

        posting_layers.clear();
        PostingLayer layer =
            make_layer(std::move(offsets), std::move(ids), kBulkLayerLevel);
        if (layer.total_ids != 0) posting_layers.push_back(std::move(layer));
        posting_cell_counts = std::move(consolidated_counts);
    }

    bool maybe_consolidate_layers_for_query(const std::vector<size_t>& cells_per_subspace) {
        size_t min_layers = query_consolidate_min_layers();
        if (min_layers == 0 || posting_layers.size() < min_layers) return false;
        uint64_t total_ids = posting_layer_total_ids();
        if (total_ids >
            std::numeric_limits<uint64_t>::max() / (uint64_t)sizeof(uint32_t)) {
            throw std::runtime_error("L0 posting layer byte size exceeds uint64 range");
        }
        uint64_t total_bytes = total_ids * (uint64_t)sizeof(uint32_t);
        uint64_t max_bytes = query_consolidate_max_posting_bytes();
        if (max_bytes == 0 || total_bytes > max_bytes) return false;
        // Defer the full-rebuild consolidation while one (bulk) layer dominates.
        // consolidate_layers_to_single re-copies ALL postings into a new layer on
        // every call, which dominates publish_ms when a large bulk is reused and
        // only a small tail was appended. The query path scores multi-layer
        // postings correctly (for_each_cell_span), so skipping the O(total) re-copy
        // is a large publish win for the common append-a-small-tail commit. Once the
        // tail grows to a meaningful share (bulk < 75% of total), we re-consolidate
        // so query-time layer indirection stays bounded. Self-regulating: the tail
        // layer is log-merged by normalize_layers and grows until it trips the check.
        uint64_t max_layer_ids = 0;
        for (const PostingLayer& layer : posting_layers) {
            if (layer.total_ids > max_layer_ids) max_layer_ids = layer.total_ids;
        }
        if (posting_layers.size() == 2 && max_layer_ids * 4 > total_ids * 3) {
            return false;
        }
        consolidate_layers_to_single(cells_per_subspace);
        return true;
    }

    void publish_single_layer(PostingVectors&& offsets,
                              PostingVectors&& ids,
                              uint32_t level = 0) {
        posting_layers.clear();
        posting_offsets.clear();
        posting_ids.clear();
        posting_cell_counts.clear();
        PostingLayer layer = make_layer(std::move(offsets), std::move(ids), level);
        if (layer.total_ids != 0) {
            posting_layers.push_back(std::move(layer));
        }
    }

    PostingLayer build_layer_for_block_range(
        size_t block_begin,
        size_t block_end,
        const std::vector<size_t>& cells_per_subspace,
        uint32_t level = 0) const {
        PostingVectors offsets(cells_per_subspace.size());
        PostingVectors ids(cells_per_subspace.size());
        for (size_t m = 0; m < cells_per_subspace.size(); ++m) {
            size_t K = cells_per_subspace[m];
            std::vector<uint32_t> sizes(K, 0);
            size_t total = 0;
            for (size_t block_idx = block_begin; block_idx < block_end; ++block_idx) {
                const auto& block = committed_blocks[block_idx];
                if (!block) continue;
                if (block->has_row_cells()) {
                    total += block->row_count();
                    for (uint32_t row = 0; row < block->row_count(); ++row) {
                        uint16_t cell = block->row_cell(row, (uint32_t)m);
                        sizes[(size_t)cell]++;
                    }
                } else {
                    for (size_t cell = 0; cell < K; ++cell) {
                        size_t count = block->cell_size((uint32_t)m, (uint32_t)cell);
                        sizes[cell] += (uint32_t)count;
                        total += count;
                    }
                }
            }
            offsets[m].assign(K + 1, 0);
            for (size_t cell = 0; cell < K; ++cell) {
                offsets[m][cell + 1] = offsets[m][cell] + sizes[cell];
            }
            ids[m].assign(total, 0);
            std::vector<uint32_t> cursors(K, 0);
            for (size_t block_idx = block_begin; block_idx < block_end; ++block_idx) {
                const auto& block = committed_blocks[block_idx];
                if (!block) continue;
                if (block->has_row_cells()) {
                    for (uint32_t row = 0; row < block->row_count(); ++row) {
                        uint16_t cell = block->row_cell(row, (uint32_t)m);
                        uint32_t offset = offsets[m][(size_t)cell] + cursors[(size_t)cell]++;
                        ids[m][(size_t)offset] = block->pid(row);
                    }
                } else {
                    for (size_t cell = 0; cell < K; ++cell) {
                        size_t count = block->cell_size((uint32_t)m, (uint32_t)cell);
                        if (count == 0) continue;
                        const uint32_t* data = block->cell_data((uint32_t)m, (uint32_t)cell);
                        uint32_t offset = offsets[m][cell] + cursors[cell];
                        std::copy(data, data + count, ids[m].begin() + (std::ptrdiff_t)offset);
                        cursors[cell] += (uint32_t)count;
                    }
                }
            }
        }
        return make_layer(std::move(offsets), std::move(ids), level);
    }

    uint64_t block_range_posting_id_count(
        size_t block_begin,
        size_t block_end,
        const std::vector<size_t>& cells_per_subspace) const {
        uint64_t total = 0;
        for (size_t block_idx = block_begin; block_idx < block_end; ++block_idx) {
            const auto& block = committed_blocks[block_idx];
            if (!block) continue;
            uint64_t block_total = 0;
            if (block->has_row_cells()) {
                block_total = (uint64_t)block->row_count() * (uint64_t)cells_per_subspace.size();
            } else {
                for (size_t m = 0; m < cells_per_subspace.size(); ++m) {
                    for (size_t cell = 0; cell < cells_per_subspace[m]; ++cell) {
                        block_total +=
                            (uint64_t)block->cell_size((uint32_t)m, (uint32_t)cell);
                    }
                }
            }
            if (block_total > std::numeric_limits<uint64_t>::max() - total) {
                throw std::runtime_error("L0 block posting count exceeds uint64 range");
            }
            total += block_total;
        }
        return total;
    }

    bool copy_and_increment_posting_cell_counts(
        const UnsealedDeltaView& previous,
        size_t tail_begin,
        const std::vector<size_t>& cells_per_subspace) {
        if (previous.posting_cell_counts.size() != cells_per_subspace.size()) {
            return false;
        }
        posting_cell_counts = previous.posting_cell_counts;
        for (size_t m = 0; m < cells_per_subspace.size(); ++m) {
            if (posting_cell_counts[m].size() != cells_per_subspace[m]) {
                posting_cell_counts.clear();
                return false;
            }
        }
        for (size_t block_idx = tail_begin; block_idx < committed_blocks.size(); ++block_idx) {
            const auto& block = committed_blocks[block_idx];
            if (!block) continue;
            if (block->has_row_cells()) {
                for (size_t m = 0; m < cells_per_subspace.size(); ++m) {
                    for (uint32_t row = 0; row < block->row_count(); ++row) {
                        uint16_t cell = block->row_cell(row, (uint32_t)m);
                        if ((size_t)cell >= posting_cell_counts[m].size()) {
                            throw std::runtime_error("incremental L0 count cell exceeds subspace width");
                        }
                        if (posting_cell_counts[m][(size_t)cell] ==
                            std::numeric_limits<uint32_t>::max()) {
                            throw std::runtime_error("incremental L0 cell count exceeds uint32 range");
                        }
                        ++posting_cell_counts[m][(size_t)cell];
                    }
                }
            } else {
                for (size_t m = 0; m < cells_per_subspace.size(); ++m) {
                    for (size_t cell = 0; cell < cells_per_subspace[m]; ++cell) {
                        size_t count = block->cell_size((uint32_t)m, (uint32_t)cell);
                        if (count >
                            (size_t)std::numeric_limits<uint32_t>::max() -
                                (size_t)posting_cell_counts[m][cell]) {
                            throw std::runtime_error("incremental L0 cell count exceeds uint32 range");
                        }
                        posting_cell_counts[m][cell] += (uint32_t)count;
                    }
                }
            }
        }
        return true;
    }

    bool build_contiguous_incremental_from_previous(
        const UnsealedDeltaView& previous,
        size_t tail_begin,
        const std::vector<size_t>& cells_per_subspace) {
        if (previous.posting_layers.size() != 1 ||
            !previous.posting_layers.front().offsets ||
            !previous.posting_layers.front().ids) {
            return false;
        }
        const PostingLayer& previous_layer = previous.posting_layers.front();
        const PostingVectors& previous_offsets_all = *previous_layer.offsets;
        const PostingVectors& previous_ids_all = *previous_layer.ids;
        if (previous_offsets_all.size() != cells_per_subspace.size() ||
            previous_ids_all.size() != cells_per_subspace.size()) {
            return false;
        }

        PostingVectors offsets(cells_per_subspace.size());
        PostingVectors ids(cells_per_subspace.size());
        std::vector<std::vector<uint32_t>> counts(cells_per_subspace.size());

        auto build_subspace = [&](size_t m) {
            size_t K = cells_per_subspace[m];
            if (previous_offsets_all[m].size() != K + 1) {
                throw std::runtime_error("incremental L0 lookup cannot reuse incompatible previous postings");
            }
            const std::vector<uint32_t>& previous_offsets = previous_offsets_all[m];
            const std::vector<uint32_t>& previous_ids = previous_ids_all[m];
            std::vector<uint32_t> tail_sizes(K, 0);
            size_t tail_total = 0;
            for (size_t block_idx = tail_begin; block_idx < committed_blocks.size(); ++block_idx) {
                const auto& block = committed_blocks[block_idx];
                if (!block) continue;
                if (block->has_row_cells()) {
                    for (uint32_t row = 0; row < block->row_count(); ++row) {
                        uint16_t cell = block->row_cell(row, (uint32_t)m);
                        if ((size_t)cell >= K) {
                            throw std::runtime_error("incremental L0 tail row cell exceeds subspace width");
                        }
                        if (tail_sizes[(size_t)cell] ==
                            std::numeric_limits<uint32_t>::max()) {
                            throw std::runtime_error("incremental L0 tail cell count exceeds uint32 range");
                        }
                        if (tail_total == std::numeric_limits<size_t>::max()) {
                            throw std::runtime_error("incremental L0 tail total count exceeds size_t range");
                        }
                        ++tail_sizes[(size_t)cell];
                        ++tail_total;
                    }
                } else {
                    for (size_t cell = 0; cell < K; ++cell) {
                        size_t count = block->cell_size((uint32_t)m, (uint32_t)cell);
                        if (count >
                            (size_t)std::numeric_limits<uint32_t>::max() -
                                (size_t)tail_sizes[cell]) {
                            throw std::runtime_error("incremental L0 tail cell count exceeds uint32 range");
                        }
                        if (count > std::numeric_limits<size_t>::max() - tail_total) {
                            throw std::runtime_error("incremental L0 tail total count exceeds size_t range");
                        }
                        tail_sizes[cell] += (uint32_t)count;
                        tail_total += count;
                    }
                }
            }

            std::vector<uint32_t>& subspace_offsets = offsets[m];
            std::vector<uint32_t>& subspace_counts = counts[m];
            subspace_offsets.assign(K + 1, 0);
            subspace_counts.assign(K, 0);
            if (tail_total > std::numeric_limits<size_t>::max() - previous_ids.size()) {
                throw std::runtime_error("incremental L0 lookup subspace exceeds size_t range");
            }
            size_t total = previous_ids.size() + tail_total;
            if (total > (size_t)std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("incremental L0 lookup subspace exceeds uint32 range");
            }
            for (size_t cell = 0; cell < K; ++cell) {
                uint32_t previous_count = previous_offsets[cell + 1] - previous_offsets[cell];
                if (previous_count >
                    std::numeric_limits<uint32_t>::max() - tail_sizes[cell]) {
                    throw std::runtime_error("incremental L0 lookup cell size exceeds uint32 range");
                }
                uint32_t cell_count = previous_count + tail_sizes[cell];
                if (cell_count > std::numeric_limits<uint32_t>::max() - subspace_offsets[cell]) {
                    throw std::runtime_error("incremental L0 lookup offsets exceed uint32 range");
                }
                subspace_counts[cell] = cell_count;
                subspace_offsets[cell + 1] = subspace_offsets[cell] + cell_count;
            }
            ids[m].assign(total, 0);

            std::vector<uint32_t> write_offsets(K, 0);
            for (size_t cell = 0; cell < K; ++cell) {
                uint32_t previous_begin = previous_offsets[cell];
                uint32_t previous_end = previous_offsets[cell + 1];
                uint32_t count = previous_end - previous_begin;
                if (count == 0) continue;
                uint32_t offset = subspace_offsets[cell];
                std::copy(previous_ids.begin() + (std::ptrdiff_t)previous_begin,
                          previous_ids.begin() + (std::ptrdiff_t)previous_end,
                          ids[m].begin() + (std::ptrdiff_t)offset);
                write_offsets[cell] = offset + count;
            }
            for (size_t cell = 0; cell < K; ++cell) {
                if (write_offsets[cell] == 0) write_offsets[cell] = subspace_offsets[cell];
            }
            for (size_t block_idx = tail_begin; block_idx < committed_blocks.size(); ++block_idx) {
                const auto& block = committed_blocks[block_idx];
                if (!block) continue;
                if (block->has_row_cells()) {
                    for (uint32_t row = 0; row < block->row_count(); ++row) {
                        uint16_t cell = block->row_cell(row, (uint32_t)m);
                        uint32_t offset = write_offsets[(size_t)cell]++;
                        ids[m][(size_t)offset] = block->pid(row);
                    }
                } else {
                    for (size_t cell = 0; cell < K; ++cell) {
                        size_t count = block->cell_size((uint32_t)m, (uint32_t)cell);
                        if (count == 0) continue;
                        const uint32_t* data = block->cell_data((uint32_t)m, (uint32_t)cell);
                        uint32_t offset = write_offsets[cell];
                        std::copy(data, data + count, ids[m].begin() + (std::ptrdiff_t)offset);
                        write_offsets[cell] += (uint32_t)count;
                    }
                }
            }
        };

        std::exception_ptr build_error;
        int subspace_count = (int)cells_per_subspace.size();
        int build_threads = std::min(subspace_count, std::min(omp_get_max_threads(), 4));
#pragma omp parallel for schedule(static) num_threads(build_threads) if(subspace_count >= 4 && build_threads > 1)
        for (int mi = 0; mi < subspace_count; ++mi) {
            try {
                build_subspace((size_t)mi);
            } catch (...) {
#pragma omp critical(score_growing_delta_contiguous_incremental_error)
                {
                    if (!build_error) build_error = std::current_exception();
                }
            }
        }
        if (build_error) std::rethrow_exception(build_error);

        posting_layers.clear();
        PostingLayer layer =
            make_layer(std::move(offsets), std::move(ids), kBulkLayerLevel);
        if (layer.total_ids != 0) posting_layers.push_back(std::move(layer));
        posting_cell_counts = std::move(counts);
        return true;
    }

    void build_lookup(const std::vector<size_t>& cells_per_subspace,
                      const LiveTable* live = nullptr,
                      bool release_block_postings = false) {
        live_posting_counts.clear();
        has_live_posting_counts_ = false;
        build_pid_lookup(live);

        posting_offsets.clear();
        posting_ids.clear();
        posting_cell_counts.clear();
        posting_layers.clear();
        if (!committed_blocks.empty()) {
            PostingLayer layer =
                build_layer_for_block_range(
                    0, committed_blocks.size(), cells_per_subspace, kBulkLayerLevel);
            if (layer.total_ids != 0) {
                posting_layers.push_back(std::move(layer));
            }
        }

        rebuild_posting_cell_counts(cells_per_subspace);
        rebuild_live_cell_counts(cells_per_subspace, live);

        if (release_block_postings) {
            for (const auto& block : committed_blocks) {
                if (!block) continue;
                block->release_posting_index();
            }
        }
    }

    void build_incremental_from_previous(
        const UnsealedDeltaView& previous,
        std::vector<std::shared_ptr<const L0Block>> tail_blocks,
        const std::vector<size_t>& cells_per_subspace,
        const LiveTable* live = nullptr,
        bool release_tail_block_postings = false,
        bool force_refresh_liveness = false,
        const LiveTable* previous_live = nullptr,
        const std::vector<uint32_t>* dead_pids = nullptr) {
        committed_blocks = previous.committed_blocks;
        size_t tail_begin = committed_blocks.size();
        committed_blocks.insert(committed_blocks.end(),
                                std::make_move_iterator(tail_blocks.begin()),
                                std::make_move_iterator(tail_blocks.end()));
        if (!build_pid_lookup_incremental_from_previous(previous, tail_begin, live)) {
            build_pid_lookup(live);
        }
        if (force_refresh_liveness) {
            refresh_all_pids_live(live);
        }

        posting_layers.clear();
        posting_offsets.clear();
        posting_ids.clear();
        posting_cell_counts.clear();

        if (!previous.posting_layers.empty()) {
            posting_layers = previous.posting_layers;
        } else if (!previous.posting_offsets.empty() || !previous.posting_ids.empty()) {
            if (previous.posting_offsets.size() != previous.posting_ids.size()) {
                throw std::runtime_error("incremental L0 lookup cannot reuse incompatible previous postings");
            }
            PostingVectors previous_offsets = previous.posting_offsets;
            PostingVectors previous_ids = previous.posting_ids;
            PostingLayer previous_layer =
                make_layer(std::move(previous_offsets), std::move(previous_ids), kBulkLayerLevel);
            if (previous_layer.total_ids != 0) posting_layers.push_back(std::move(previous_layer));
        }

        if (tail_begin < committed_blocks.size()) {
            PostingLayer tail_layer =
                build_layer_for_block_range(
                    tail_begin, committed_blocks.size(), cells_per_subspace, 0);
            if (tail_layer.total_ids != 0) posting_layers.push_back(std::move(tail_layer));
        }
        normalize_layers(cells_per_subspace);
        bool consolidated_for_query = maybe_consolidate_layers_for_query(cells_per_subspace);
        if (!consolidated_for_query &&
            !copy_and_increment_posting_cell_counts(previous, tail_begin, cells_per_subspace)) {
            rebuild_posting_cell_counts(cells_per_subspace);
        }
        if (!try_rebuild_live_cell_counts_incremental(
                previous,
                tail_begin,
                cells_per_subspace,
                previous_live,
                live,
                dead_pids)) {
            rebuild_live_cell_counts(cells_per_subspace, live);
        }

        if (release_tail_block_postings) {
            for (size_t block_idx = tail_begin; block_idx < committed_blocks.size(); ++block_idx) {
                const auto& block = committed_blocks[block_idx];
                if (!block) continue;
                block->release_posting_index();
            }
        }
    }

    bool resolve_pid(uint32_t pid, RowRef& out) const {
        if (!pid_to_block_lookup.empty() &&
            pid >= pid_lookup_begin &&
            (uint64_t)pid - (uint64_t)pid_lookup_begin < (uint64_t)pid_to_block_lookup.size()) {
            uint32_t block_idx = pid_to_block_lookup[(size_t)(pid - pid_lookup_begin)];
            if (block_idx == kMissingBlockIndex ||
                (size_t)block_idx >= committed_blocks.size()) {
                return false;
            }
            const L0Block* block = committed_blocks[(size_t)block_idx].get();
            if (block == nullptr || !block->contains_pid(pid)) return false;
            out.block = block;
            out.local_row = block->local_row_for_pid(pid);
            return true;
        }
        if (block_pid_begins.empty()) {
            for (auto it = committed_blocks.rbegin(); it != committed_blocks.rend(); ++it) {
                const auto& block = *it;
                if (block && block->contains_pid(pid)) {
                    out.block = block.get();
                    out.local_row = block->local_row_for_pid(pid);
                    return true;
                }
            }
            return false;
        }
        auto it = std::upper_bound(block_pid_begins.begin(), block_pid_begins.end(), pid);
        if (it == block_pid_begins.begin()) return false;
        size_t lookup_idx = (size_t)std::distance(block_pid_begins.begin(), it) - 1;
        if (lookup_idx >= block_lookup_indices.size()) return false;
        size_t block_idx = block_lookup_indices[lookup_idx];
        if (block_idx >= committed_blocks.size()) return false;
        const L0Block* block = committed_blocks[block_idx].get();
        if (block == nullptr || !block->contains_pid(pid)) return false;
        out.block = block;
        out.local_row = block->local_row_for_pid(pid);
        return true;
    }

    const L0Block* block_for_pid(uint32_t pid) const {
        RowRef row;
        return resolve_pid(pid, row) ? row.block : nullptr;
    }

    bool rabitq_record_for_pid(uint32_t pid,
                               const char*& record,
                               uint32_t& record_bytes) const {
        RowRef row;
        if (!resolve_pid(pid, row)) return false;
        record = reinterpret_cast<const char*>(row.block->rabitq_record(row.local_row));
        record_bytes = row.block->rabitq_record_bytes();
        return true;
    }

    bool verify_row_for_pid(uint32_t pid,
                            const float*& verify_vector,
                            uint32_t& verify_dim,
                            uint32_t& output_id) const {
        RowRef row;
        if (!resolve_pid(pid, row)) return false;
        verify_vector = row.block->verify_vector(row.local_row);
        verify_dim = row.block->verify_dim();
        output_id = row.block->orig_id(row.local_row);
        return true;
    }

    size_t cell_size(uint32_t subspace, uint32_t cell) const {
        if ((size_t)subspace < posting_cell_counts.size()) {
            const std::vector<uint32_t>& counts = posting_cell_counts[(size_t)subspace];
            if ((size_t)cell < counts.size()) return counts[(size_t)cell];
        }
        if (!posting_layers.empty()) {
            size_t total = 0;
            for (const PostingLayer& layer : posting_layers) {
                size_t count = layer_cell_size(layer, subspace, cell);
                if (count > std::numeric_limits<size_t>::max() - total) {
                    throw std::runtime_error("L0 posting cell size exceeds size_t range");
                }
                total += count;
            }
            return total;
        }
        if ((size_t)subspace >= posting_offsets.size()) return 0;
        const std::vector<uint32_t>& offsets = posting_offsets[(size_t)subspace];
        if ((size_t)cell + 1 >= offsets.size()) return 0;
        return (size_t)(offsets[(size_t)cell + 1] - offsets[(size_t)cell]);
    }

    bool all_pids_live() const { return all_pids_live_; }

    bool has_live_cell_counts() const { return has_live_posting_counts_; }

    size_t live_cell_size(uint32_t subspace, uint32_t cell) const {
        if (!has_live_posting_counts_) return cell_size(subspace, cell);
        if ((size_t)subspace >= live_posting_counts.size()) return 0;
        const std::vector<uint32_t>& counts = live_posting_counts[(size_t)subspace];
        if ((size_t)cell >= counts.size()) return 0;
        return counts[(size_t)cell];
    }

    const uint32_t* cell_data(uint32_t subspace, uint32_t cell) const {
        if (!posting_layers.empty()) {
            if (posting_layers.size() != 1) return nullptr;
            CellSpan span = layer_cell_span(posting_layers.front(), subspace, cell);
            return span.data;
        }
        if ((size_t)subspace >= posting_offsets.size()) return nullptr;
        const std::vector<uint32_t>& offsets = posting_offsets[(size_t)subspace];
        if ((size_t)cell + 1 >= offsets.size()) return nullptr;
        const std::vector<uint32_t>& ids = posting_ids[(size_t)subspace];
        return ids.data() + offsets[(size_t)cell];
    }

    CellSpan cell_span(uint32_t subspace, uint32_t cell) const {
        if (!posting_layers.empty()) {
            if (posting_layers.size() != 1) return {};
            return layer_cell_span(posting_layers.front(), subspace, cell);
        }
        if ((size_t)subspace >= posting_offsets.size()) return {};
        const std::vector<uint32_t>& offsets = posting_offsets[(size_t)subspace];
        if ((size_t)cell + 1 >= offsets.size()) return {};
        const uint32_t begin = offsets[(size_t)cell];
        const uint32_t end = offsets[(size_t)cell + 1];
        if (begin == end) return {};
        const std::vector<uint32_t>& ids = posting_ids[(size_t)subspace];
        return {ids.data() + begin, (size_t)(end - begin)};
    }

    std::vector<uint32_t> block_pid_begins;
    std::vector<size_t> block_lookup_indices;
    uint32_t pid_lookup_begin = 0;
    std::vector<uint32_t> pid_to_block_lookup;
    std::vector<std::vector<uint32_t>> posting_offsets;
    std::vector<std::vector<uint32_t>> posting_ids;
    std::vector<std::vector<uint32_t>> live_posting_counts;
    bool all_pids_live_ = true;
    bool has_live_posting_counts_ = false;
    static constexpr uint32_t kMissingBlockIndex = std::numeric_limits<uint32_t>::max();
    static constexpr size_t kDefaultDensePidLookupMaxSpan = 1048576;
    static constexpr uint64_t kDefaultLayerMergeMaxPostingBytes = 1ull * 1024ull * 1024ull;
    static constexpr size_t kDefaultQueryConsolidateMinLayers = 2;
    static constexpr uint64_t kDefaultQueryConsolidateMaxPostingBytes = 16ull * 1024ull * 1024ull;

    static size_t dense_pid_lookup_max_span() {
        const char* env = std::getenv("SCORE_L0_DENSE_PID_LOOKUP_MAX_SPAN");
        if (env == nullptr || env[0] == '\0') return kDefaultDensePidLookupMaxSpan;
        char* end = nullptr;
        unsigned long long value = std::strtoull(env, &end, 10);
        if (end == env || (end != nullptr && *end != '\0')) {
            return kDefaultDensePidLookupMaxSpan;
        }
        if (value == 0) return 0;
        return (size_t)std::min<unsigned long long>(
            value, (unsigned long long)std::numeric_limits<uint32_t>::max());
    }

    static uint64_t layer_merge_posting_bytes(const PostingLayer& older,
                                              const PostingLayer& newer) {
        uint64_t total_ids = older.total_ids;
        if (newer.total_ids > std::numeric_limits<uint64_t>::max() - total_ids) {
            throw std::runtime_error("L0 posting layer merge size exceeds uint64 range");
        }
        total_ids += newer.total_ids;
        if (total_ids > std::numeric_limits<uint64_t>::max() / (uint64_t)sizeof(uint32_t)) {
            throw std::runtime_error("L0 posting layer merge byte size exceeds uint64 range");
        }
        return total_ids * (uint64_t)sizeof(uint32_t);
    }

    static uint64_t parse_mib_env_or_default(const char* name, uint64_t default_bytes) {
        const char* env = std::getenv(name);
        if (env == nullptr || env[0] == '\0') return default_bytes;
        char* end = nullptr;
        unsigned long long value = std::strtoull(env, &end, 10);
        if (end == env || (end != nullptr && *end != '\0')) return default_bytes;
        if (value == 0) return 0;
        value = std::min<unsigned long long>(value, 4096ull);
        return (uint64_t)value * 1024ull * 1024ull;
    }

    static uint64_t layer_merge_max_posting_bytes() {
        return parse_mib_env_or_default(
            "SCORE_L0_LAYER_MERGE_MB", kDefaultLayerMergeMaxPostingBytes);
    }

    static size_t query_consolidate_min_layers() {
        const char* env = std::getenv("SCORE_L0_QUERY_CONSOLIDATE_MIN_LAYERS");
        if (env == nullptr || env[0] == '\0') return kDefaultQueryConsolidateMinLayers;
        char* end = nullptr;
        unsigned long long value = std::strtoull(env, &end, 10);
        if (end == env || (end != nullptr && *end != '\0')) {
            return kDefaultQueryConsolidateMinLayers;
        }
        if (value == 0) return 0;
        return (size_t)std::max<unsigned long long>(
            2ull, std::min<unsigned long long>(value, 1024ull));
    }

    static uint64_t query_consolidate_max_posting_bytes() {
        return parse_mib_env_or_default(
            "SCORE_L0_QUERY_CONSOLIDATE_MB", kDefaultQueryConsolidateMaxPostingBytes);
    }

};
} // namespace score
