#pragma once

#include "query/base_segment.h"
#include "query/delta_posting_views.h"
#include "update/update_state.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace score {
struct ReaderView {
    uint64_t generation = 0;
    std::shared_ptr<const BaseSegment> base;
    std::shared_ptr<const BasePostingView> base_live_postings;
    bool base_live_postings_exact = false;
    uint32_t base_live_postings_base_live_count = 0;
    std::shared_ptr<const std::vector<uint32_t>> base_live_postings_unfiltered_dead_pids;
    std::vector<std::shared_ptr<const SealedDeltaSegment>> sealed_deltas;
    std::shared_ptr<const SealedDeltaPostingView> sealed_delta_postings;
    std::shared_ptr<const SealedDeltaPidLookup> sealed_delta_pid_lookup;
    std::shared_ptr<const UnsealedDeltaView> growing_delta;
    std::shared_ptr<const LiveTable> live;
    std::shared_ptr<const std::vector<uint32_t>> sparse_dead_pids;
    std::shared_ptr<const std::vector<uint32_t>> selection_dead_pids;
    std::vector<std::vector<uint32_t>> composite_cell_sizes;
    uint32_t base_n = 0;
    uint32_t max_pid = 0;
    uint32_t live_count = 0;
    uint32_t base_live_count = 0;
    uint64_t base_live_fingerprint = 0;
    uint32_t base_dead_prefix_end = 0;
    uint32_t live_suffix_begin = 0;
    bool dead_scores_filtered_for_selection = false;
    CandidateSelectionStrategy candidate_selection_strategy =
        CandidateSelectionStrategy::BaseAllLive;
    uint64_t l0_last_committed_lsn = 0;
    uint64_t l0_wal_replay_offset = 0;

    bool has_deltas() const {
        return !sealed_deltas.empty() || (growing_delta && !growing_delta->empty());
    }
};

struct ResolvedRow {
    enum class SegmentKind {
        Base,
        Delta
    };

    SegmentKind segment = SegmentKind::Base;
    uint32_t global_pid = 0;
    uint32_t output_id = 0;
    const char* rabitq_record = nullptr;
    size_t rabitq_record_bytes = 0;
    const float* verify_vector = nullptr;
    int verify_dim = 0;
};

class RowResolver {
public:
    RowResolver() = default;
    explicit RowResolver(const ReaderView& view) : view_(&view) {}

    bool resolve(uint32_t global_pid, ResolvedRow& row) const {
        if (view_ == nullptr || !view_->base) return false;
        const BaseSegment& base = *view_->base;
        if (base.n >= 0 && global_pid < view_->base_n && global_pid < (uint32_t)base.n) {
            if (base.rabitq_bin_with_assign_ == nullptr || base.verify_vectors_ == nullptr) {
                return false;
            }
            if (!base.perm_to_orig.empty() && global_pid >= base.perm_to_orig.size()) {
                return false;
            }

            row.segment = ResolvedRow::SegmentKind::Base;
            row.global_pid = global_pid;
            row.output_id = base.perm_to_orig.empty()
                ? global_pid
                : base.perm_to_orig[(size_t)global_pid];
            row.rabitq_record = base.rabitq_bin_with_assign_ +
                (size_t)global_pid * base.rabitq_bin_with_assign_record_bytes_;
            row.rabitq_record_bytes = base.rabitq_bin_with_assign_record_bytes_;
            row.verify_vector = base.verify_vectors_ + (size_t)global_pid * base.verify_vectors_dim;
            row.verify_dim = base.verify_vectors_dim;
            return true;
        }

        auto resolve_sealed_delta = [&](const std::shared_ptr<const SealedDeltaSegment>& segment) {
            if (!segment || !segment->contains_pid(global_pid)) return false;
            uint32_t local_row = segment->local_row_for_pid(global_pid);
            row.segment = ResolvedRow::SegmentKind::Delta;
            row.global_pid = global_pid;
            row.output_id = segment->orig_id(local_row);
            row.rabitq_record = reinterpret_cast<const char*>(segment->rabitq_record(local_row));
            row.rabitq_record_bytes = segment->rabitq_record_bytes();
            row.verify_vector = segment->verify_vector(local_row);
            row.verify_dim = (int)segment->verify_dim();
            return true;
        };

        if (view_->sealed_delta_pid_lookup) {
            uint32_t segment_index = 0;
            if (view_->sealed_delta_pid_lookup->segment_index_for_pid(global_pid, segment_index) &&
                (size_t)segment_index < view_->sealed_deltas.size() &&
                resolve_sealed_delta(view_->sealed_deltas[(size_t)segment_index])) {
                return true;
            }
        } else {
            for (const auto& segment : view_->sealed_deltas) {
                if (resolve_sealed_delta(segment)) return true;
            }
        }

        if (view_->growing_delta) {
            UnsealedDeltaView::RowRef l0_row;
            if (view_->growing_delta->resolve_pid(global_pid, l0_row)) {
                const L0Block* block = l0_row.block;
                uint32_t local_row = l0_row.local_row;
                row.segment = ResolvedRow::SegmentKind::Delta;
                row.global_pid = global_pid;
                row.output_id = block->orig_id(local_row);
                row.rabitq_record = reinterpret_cast<const char*>(block->rabitq_record(local_row));
                row.rabitq_record_bytes = block->rabitq_record_bytes();
                row.verify_vector = block->verify_vector(local_row);
                row.verify_dim = (int)block->verify_dim();
                return true;
            }
        }
        return false;
    }

private:
    const ReaderView* view_ = nullptr;
};
} // namespace score
