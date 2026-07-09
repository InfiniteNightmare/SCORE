#include "core/common.h"
#include "query/query_engine.h"

#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace score;

namespace {

void print_usage() {
    std::cerr << "Usage: score_update_search <index_dir> <query.fvecs> <groundtruth.ivecs> [options]\n"
              << "Options:\n"
              << "  -topk|-k <int>              Top-K results (default: 10)\n"
              << "  -M <int>                    Number of subspaces (default: 16)\n"
              << "  -alpha <float>              Collision ratio (default: 0.05)\n"
              << "  -beta <float>               Candidate ratio (default: 0.001)\n"
              << "  -query_runs <int>           Repeat full query batch N times (default: 1)\n"
              << "  -threads <int>              Query threads (default: OMP default)\n"
              << "  -max_w <int>                Max tier weight (default: 3)\n";
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

double recall_at_k(const std::vector<QueryResult>& results,
                   const std::vector<int>& groundtruth,
                   int gt_dim,
                   int topk) {
    int hits = 0;
    int total = 0;
    for (size_t q = 0; q < results.size(); ++q) {
        for (int k = 0; k < topk; ++k) {
            int gt_id = groundtruth[q * (size_t)gt_dim + (size_t)k];
            for (const auto& hit : results[q].topk) {
                if (hit.second == gt_id) {
                    ++hits;
                    break;
                }
            }
            ++total;
        }
    }
    return total > 0 ? (double)hits / (double)total : 0.0;
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
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + option);
            return argv[++i];
        };
        try {
            if (arg == "-topk" || arg == "-k") {
                config.topk = parse_int_option(require_value(arg.c_str()), arg.c_str());
                if (config.topk <= 0) throw std::runtime_error(arg + " must be positive");
            } else if (arg == "-M") {
                config.num_subspaces = parse_int_option(require_value("-M"), "-M");
                if (config.num_subspaces <= 0) throw std::runtime_error("-M must be positive");
            } else if (arg == "-alpha") {
                config.collision_ratio = parse_float_option(require_value("-alpha"), "-alpha");
                if (config.collision_ratio < 0.0f) throw std::runtime_error("-alpha must be non-negative");
            } else if (arg == "-beta") {
                config.candidate_ratio = parse_float_option(require_value("-beta"), "-beta");
                if (config.candidate_ratio < 0.0f) throw std::runtime_error("-beta must be non-negative");
            } else if (arg == "-query_runs") {
                query_runs = parse_int_option(require_value("-query_runs"), "-query_runs");
                if (query_runs <= 0) throw std::runtime_error("-query_runs must be positive");
            } else if (arg == "-threads") {
                config.num_threads = parse_int_option(require_value("-threads"), "-threads");
                if (config.num_threads < 0) throw std::runtime_error("-threads must be non-negative");
            } else if (arg == "-max_w") {
                config.max_w = parse_int_option(require_value("-max_w"), "-max_w");
                if (config.max_w <= 0) throw std::runtime_error("-max_w must be positive");
            } else {
                throw std::runtime_error("unknown option: " + arg);
            }
        } catch (const std::exception& e) {
            std::cerr << e.what() << "\n";
            print_usage();
            return 1;
        }
    }
    if (config.num_threads > 0) {
        omp_set_num_threads(config.num_threads);
    }

    int nq = 0, dim = 0;
    std::vector<float> queries = load_fvecs(query_path, nq, dim);
    int gt_n = 0, gt_dim = 0;
    std::vector<int> groundtruth = load_ivecs(gt_path, gt_n, gt_dim);
    if (gt_n < nq || gt_dim < config.topk) {
        std::cerr << "ground truth shape is incompatible with queries/topk\n";
        return 1;
    }

    QueryIndexHandle handle;
    handle.init(config);
    handle.load_index(index_dir);
    if (dim != handle.dim()) {
        std::cerr << "query dimension does not match index\n";
        return 1;
    }
    if (nq > 0) {
        handle.search_batch(queries.data(), 1);
    }

    Timer total_timer;
    total_timer.reset();
    std::vector<QueryResult> results;
    for (int run = 0; run < query_runs; ++run) {
        results = handle.search_batch(queries.data(), nq);
    }
    double total_ms = total_timer.elapsed_ms();
    double recall = recall_at_k(results, groundtruth, gt_dim, config.topk);
    double qps = ((double)nq * (double)query_runs) / (total_ms / 1000.0);

    std::cout << "\n===== Search Results =====\n";
    std::cout << "Recall@" << config.topk << ": " << recall << "\n";
    std::cout << "Query runs: " << query_runs << "\n";
    std::cout << "QPS: " << qps << "\n";
    std::cout << "Total time: " << total_ms << " ms\n";
    return 0;
}
