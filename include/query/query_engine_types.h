#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace score {

struct QueryResult {
    std::vector<std::pair<float, int>> topk;
};

struct QueryProfileStats {
    uint64_t queries = 0;
    double total_us = 0.0;
    double rotate_us = 0.0;
    double activation_us = 0.0;
    double score_reset_us = 0.0;
    double cell_schedule_us = 0.0;
    double posting_score_us = 0.0;
    double candidate_select_us = 0.0;
    double candidate_dead_clear_us = 0.0;
    double candidate_topn_us = 0.0;
    double refine_total_us = 0.0;
    double rabitq_rank_us = 0.0;
    double verify_us = 0.0;
    uint64_t phase4_cells = 0;
    uint64_t selected_candidates = 0;
    uint64_t candidate_dead_clear_pids = 0;
    uint64_t base_posting_ids = 0;
    uint64_t delta_posting_ids = 0;
    uint64_t delta_posting_cells = 0;
    uint64_t inline_delta_posting_cells = 0;
    uint64_t inline_delta_posting_ids = 0;
    uint64_t buffered_delta_posting_cells = 0;
    uint64_t buffered_delta_posting_ids = 0;
    uint64_t resolved_refine_queries = 0;
    uint64_t resolved_base_candidates = 0;
    uint64_t resolved_delta_candidates = 0;

    void add(const QueryProfileStats& other) {
        queries += other.queries;
        total_us += other.total_us;
        rotate_us += other.rotate_us;
        activation_us += other.activation_us;
        score_reset_us += other.score_reset_us;
        cell_schedule_us += other.cell_schedule_us;
        posting_score_us += other.posting_score_us;
        candidate_select_us += other.candidate_select_us;
        candidate_dead_clear_us += other.candidate_dead_clear_us;
        candidate_topn_us += other.candidate_topn_us;
        refine_total_us += other.refine_total_us;
        rabitq_rank_us += other.rabitq_rank_us;
        verify_us += other.verify_us;
        phase4_cells += other.phase4_cells;
        selected_candidates += other.selected_candidates;
        candidate_dead_clear_pids += other.candidate_dead_clear_pids;
        base_posting_ids += other.base_posting_ids;
        delta_posting_ids += other.delta_posting_ids;
        delta_posting_cells += other.delta_posting_cells;
        inline_delta_posting_cells += other.inline_delta_posting_cells;
        inline_delta_posting_ids += other.inline_delta_posting_ids;
        buffered_delta_posting_cells += other.buffered_delta_posting_cells;
        buffered_delta_posting_ids += other.buffered_delta_posting_ids;
        resolved_refine_queries += other.resolved_refine_queries;
        resolved_base_candidates += other.resolved_base_candidates;
        resolved_delta_candidates += other.resolved_delta_candidates;
    }
};

} // namespace score
