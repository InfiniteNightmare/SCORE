#include "query/query_engine.h"
#include "core/common.h"
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace score;

namespace {

void print_usage() {
    std::cerr << "Usage: score_search <index_dir> <query.fvecs> <groundtruth.ivecs> [options]\n";
    std::cerr << "Options:\n";
    std::cerr << "  -topk|-k <int>                  Top-K results (default: 10)\n";
    std::cerr << "  -M <int>                        Number of subspaces (default: 16)\n";
    std::cerr << "  -alpha|-collision_ratio <f>     Collision ratio (default: 0.05)\n";
    std::cerr << "  -beta|-candidate_ratio <f>      Candidate ratio (default: 0.001)\n";
    std::cerr << "  -query_runs <int>               Repeat full query batch N times (default: 1)\n";
    std::cerr << "  -threads <int>                  Number of threads (default: OMP default)\n";
    std::cerr << "  -max_w <int>                    Max tier weight for weighted scoring (default: 3)\n";
    std::cerr << "  -score_prefetch <int>           Phase4 score-array prefetch distance (default: 32)\n";
    std::cerr << "  -profile_query                  Print per-query phase timings\n";
}

int parse_int_option(const std::string& value, const char* option) {
    size_t consumed = 0;
    long long parsed = 0;
    try {
        parsed = std::stoll(value, &consumed, 10);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("invalid integer for ") + option + ": " + value);
    }
    if (consumed != value.size() ||
        parsed < (long long)std::numeric_limits<int>::min() ||
        parsed > (long long)std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string("invalid integer for ") + option + ": " + value);
    }
    return (int)parsed;
}

float parse_float_option(const std::string& value, const char* option) {
    size_t consumed = 0;
    float parsed = 0.0f;
    try {
        parsed = std::stof(value, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("invalid float for ") + option + ": " + value);
    }
    if (consumed != value.size() || !std::isfinite(parsed)) {
        throw std::runtime_error(std::string("invalid float for ") + option + ": " + value);
    }
    return parsed;
}

} // namespace

int main(int argc, char** argv) {
    enable_transparent_huge_pages_for_process();

    if (argc < 4) {
        print_usage();
        return 1;
    }

    std::string index_dir = argv[1];
    std::string query_path = argv[2];
    std::string gt_path = argv[3];

    IndexConfig config;
    int query_runs = 1;

    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        auto require_value = [&](const char* option) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + option);
            }
            return argv[++i];
        };

        try {
            if (arg == "-topk" || arg == "-k") {
                config.topk = parse_int_option(require_value(arg.c_str()), arg.c_str());
                if (config.topk < 0) throw std::runtime_error(arg + " must be non-negative");
            } else if (arg == "-M") {
                config.num_subspaces = parse_int_option(require_value("-M"), "-M");
                if (config.num_subspaces <= 0) throw std::runtime_error("-M must be positive");
            } else if (arg == "-alpha" || arg == "-collision_ratio") {
                config.collision_ratio = parse_float_option(require_value(arg.c_str()), arg.c_str());
                if (config.collision_ratio < 0.0f) {
                    throw std::runtime_error(arg + " must be non-negative");
                }
            } else if (arg == "-beta" || arg == "-candidate_ratio") {
                config.candidate_ratio = parse_float_option(require_value(arg.c_str()), arg.c_str());
                if (config.candidate_ratio < 0.0f) {
                    throw std::runtime_error(arg + " must be non-negative");
                }
            } else if (arg == "-query_runs") {
                query_runs = parse_int_option(require_value("-query_runs"), "-query_runs");
                if (query_runs <= 0) throw std::runtime_error("-query_runs must be positive");
            } else if (arg == "-threads") {
                config.num_threads = parse_int_option(require_value("-threads"), "-threads");
                if (config.num_threads < 0) throw std::runtime_error("-threads must be non-negative");
            } else if (arg == "-max_w") {
                config.max_w = parse_int_option(require_value("-max_w"), "-max_w");
                if (config.max_w <= 0) throw std::runtime_error("-max_w must be positive");
            } else if (arg == "-score_prefetch") {
                config.phase4_score_prefetch =
                    parse_int_option(require_value("-score_prefetch"), "-score_prefetch");
                if (config.phase4_score_prefetch < 0) {
                    throw std::runtime_error("-score_prefetch must be non-negative");
                }
            } else if (arg == "-profile_query") {
                config.query_profile = true;
            } else {
                throw std::runtime_error("unknown option: " + arg);
            }
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
            print_usage();
            return 1;
        }
    }

    if (config.num_threads > 0) {
        omp_set_num_threads(config.num_threads);
    }

    // Load queries
    std::cout << "Loading queries from " << query_path << "..." << std::endl;
    int nq, dim;
    std::vector<float> queries = load_fvecs(query_path, nq, dim);
    std::cout << "Loaded " << nq << " queries, dim=" << dim << std::endl;

    // Load ground truth
    int gt_n, gt_dim;
    std::vector<int> groundtruth = load_ivecs(gt_path, gt_n, gt_dim);
    std::cout << "Loaded " << gt_n << " ground truth entries, top-" << gt_dim << std::endl;

    if (gt_n < nq) {
        std::cerr << "Ground truth has " << gt_n << " rows, fewer than " << nq
                  << " queries; provide a matching groundtruth.ivecs file." << std::endl;
        return 1;
    }

    if (gt_dim < config.topk) {
        std::cerr << "Ground truth top-" << gt_dim << " is smaller than requested Recall@"
                  << config.topk << "; provide a matching groundtruth.ivecs file." << std::endl;
        return 1;
    }

    // Load index
    std::cout << "Loading index from " << index_dir << "..." << std::endl;
    QueryEngine engine;
    engine.init(config);
    engine.load_index(index_dir);
    std::cout << "Index loaded: " << engine.num_vectors() << " vectors, dim=" << engine.dim() << std::endl;

    // Pre-allocate thread engines
    std::cout << "Pre-allocating " << omp_get_max_threads() << " thread engines..." << std::endl;
    engine.search_batch_preallocate();

    // Run search
    std::cout << "\nSearching..." << std::endl;
    if (config.query_profile) engine.reset_profile_stats();
    Timer total_timer;
    total_timer.reset();
    std::vector<QueryResult> results;
    for (int run = 0; run < query_runs; ++run) {
        results = engine.search_batch(queries.data(), nq);
    }
    double total_ms = total_timer.elapsed_ms();

    // Compute recall
    int topk = config.topk;
    int recall_hits = 0;
    int recall_total = 0;

    for (int q = 0; q < nq; q++) {
        std::vector<int> returned_ids;
        for (auto& [dist, pid] : results[q].topk) {
            returned_ids.push_back(pid);
        }

        for (int k = 0; k < topk; k++) {
            int gt_id = groundtruth[(size_t)q * gt_dim + k];
            for (int r = 0; r < (int)returned_ids.size(); r++) {
                if (returned_ids[r] == gt_id) {
                    recall_hits++;
                    break;
                }
            }
            recall_total++;
        }
    }

    float recall = (recall_total > 0) ? (float)recall_hits / recall_total : 0;

    double qps = ((double)nq * (double)query_runs) / (total_ms / 1000.0);

    std::cout << "\n===== Search Results =====" << std::endl;
    std::cout << "Recall@" << topk << ": " << recall << std::endl;
    std::cout << "Query runs: " << query_runs << std::endl;
    std::cout << "QPS: " << qps << std::endl;
    std::cout << "Total time: " << total_ms << " ms" << std::endl;
    if (config.query_profile) {
        QueryProfileStats stats = engine.collect_profile_stats();
        double q = stats.queries > 0 ? (double)stats.queries : 1.0;
        auto avg_ms = [&](double us) { return us / q / 1000.0; };
        std::cout << "\n===== Query Profile =====" << std::endl;
        std::cout << "Profiled queries: " << stats.queries << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "avg_total_ms: " << avg_ms(stats.total_us) << std::endl;
        std::cout << "avg_rotate_ms: " << avg_ms(stats.rotate_us) << std::endl;
        std::cout << "avg_activation_ms: " << avg_ms(stats.activation_us) << std::endl;
        std::cout << "avg_score_reset_ms: " << avg_ms(stats.score_reset_us) << std::endl;
        std::cout << "avg_cell_schedule_ms: " << avg_ms(stats.cell_schedule_us) << std::endl;
        std::cout << "avg_posting_score_ms: " << avg_ms(stats.posting_score_us) << std::endl;
        std::cout << "avg_candidate_select_ms: " << avg_ms(stats.candidate_select_us) << std::endl;
        std::cout << "avg_candidate_dead_clear_ms: "
                  << avg_ms(stats.candidate_dead_clear_us) << std::endl;
        std::cout << "avg_candidate_topn_ms: " << avg_ms(stats.candidate_topn_us) << std::endl;
        std::cout << "avg_refine_total_ms: " << avg_ms(stats.refine_total_us) << std::endl;
        std::cout << "avg_rabitq_rank_ms: " << avg_ms(stats.rabitq_rank_us) << std::endl;
        std::cout << "avg_verify_ms: " << avg_ms(stats.verify_us) << std::endl;
        std::cout << "avg_phase4_cells: " << ((double)stats.phase4_cells / q) << std::endl;
        std::cout << "avg_selected_candidates: " << ((double)stats.selected_candidates / q) << std::endl;
        std::cout << "avg_candidate_dead_clear_pids: "
                  << ((double)stats.candidate_dead_clear_pids / q) << std::endl;
        std::cout << "avg_base_posting_ids: " << ((double)stats.base_posting_ids / q) << std::endl;
        std::cout << "avg_delta_posting_ids: " << ((double)stats.delta_posting_ids / q) << std::endl;
        std::cout << "avg_delta_posting_cells: "
                  << ((double)stats.delta_posting_cells / q) << std::endl;
        std::cout << "avg_inline_delta_posting_cells: "
                  << ((double)stats.inline_delta_posting_cells / q) << std::endl;
        std::cout << "avg_inline_delta_posting_ids: "
                  << ((double)stats.inline_delta_posting_ids / q) << std::endl;
        std::cout << "avg_buffered_delta_posting_cells: "
                  << ((double)stats.buffered_delta_posting_cells / q) << std::endl;
        std::cout << "avg_buffered_delta_posting_ids: "
                  << ((double)stats.buffered_delta_posting_ids / q) << std::endl;
        std::cout << "resolved_refine_query_ratio: "
                  << ((double)stats.resolved_refine_queries / q) << std::endl;
        std::cout << "avg_resolved_base_candidates: "
                  << ((double)stats.resolved_base_candidates / q) << std::endl;
        std::cout << "avg_resolved_delta_candidates: "
                  << ((double)stats.resolved_delta_candidates / q) << std::endl;
    }

    return 0;
}
