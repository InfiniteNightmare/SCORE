#include "core/common.h"
#include "query/query_engine.h"
#include "update/update_index.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

using namespace score;

namespace {

void print_usage() {
    std::cerr
        << "Usage: score_update_query_mix_bench <index_dir> <query.fvecs> <groundtruth.ivecs> [options]\n"
        << "Required update options:\n"
        << "  -upsert_fbin <file>        Update vectors\n"
        << "  -upsert_ids <ids>          Update orig IDs\n"
        << "  -batch_rows <int>          Rows per update commit\n"
        << "Query options:\n"
        << "  -topk <int>                Default 10\n"
        << "  -M <int>                   Default 16\n"
        << "  -alpha <float>             Default 0.05\n"
        << "  -beta <float>              Default 0.001\n"
        << "  -max_w <int>               Default 15\n"
        << "  -score_prefetch <int>      Default 32\n"
        << "  -query_threads <int>       Default 1\n"
        << "  -query_batches <int>       Minimum full query-set passes while update runs (default 1)\n"
        << "Update options:\n"
        << "  -update_threads <int>      Update thread cap (default OMP default)\n"
        << "  -durability <strict|buffered|none>\n"
        << "                             WAL durability mode (default buffered)\n";
}

uint64_t parse_uint64_option(const std::string& value, const char* option) {
    size_t consumed = 0;
    uint64_t parsed = 0;
    try {
        parsed = std::stoull(value, &consumed, 10);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("invalid unsigned integer for ") + option + ": " + value);
    }
    if (consumed != value.size()) {
        throw std::runtime_error(std::string("invalid unsigned integer for ") + option + ": " + value);
    }
    return parsed;
}

uint32_t parse_uint32_option(const std::string& value, const char* option) {
    uint64_t parsed = parse_uint64_option(value, option);
    if (parsed == 0 || parsed > (uint64_t)std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(std::string("invalid positive uint32 for ") + option + ": " + value);
    }
    return (uint32_t)parsed;
}

int parse_positive_int_option(const std::string& value, const char* option) {
    uint32_t parsed = parse_uint32_option(value, option);
    if (parsed > (uint32_t)std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string("value exceeds int range for ") + option + ": " + value);
    }
    return (int)parsed;
}

double percentile_ms(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    double rank = (p / 100.0) * (double)(values.size() - 1);
    size_t lo = (size_t)rank;
    size_t hi = std::min(lo + 1, values.size() - 1);
    double frac = rank - (double)lo;
    return values[lo] * (1.0 - frac) + values[hi] * frac;
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

struct UpdateWorkerStats {
    std::vector<double> commit_ms;
    double total_s = 0.0;
    // Per-phase commit timing aggregates (only populated when timing is on).
    double prepare_ms_sum = 0.0;
    double encode_ms_sum = 0.0;
    double wal_commit_ms_sum = 0.0;
    double publish_ms_sum = 0.0;
    long published_count = 0;
    long direct_publish_count = 0;
};

void run_update_worker(const std::string& index_dir,
                       const std::string& upsert_fbin,
                       const std::vector<uint32_t>& upsert_ids,
                       int update_n,
                       int update_dim,
                       uint32_t batch_rows,
                       int update_threads,
                       const std::string& durability_mode,
                       QueryIndexHandle* publish_handle,
                       UpdateWorkerStats& stats) {
    std::ifstream row_input(upsert_fbin, std::ios::binary);
    if (!row_input.is_open()) throw std::runtime_error("cannot open update fbin: " + upsert_fbin);
    row_input.seekg((std::streamoff)(sizeof(int) * 3), std::ios::beg);
    if (!row_input) throw std::runtime_error("cannot seek update fbin payload: " + upsert_fbin);

    std::vector<float> rows((size_t)batch_rows * (size_t)update_dim);
    std::vector<uint32_t> ids;
    ids.reserve(batch_rows);
    stats.commit_ms.reserve((upsert_ids.size() + batch_rows - 1) / batch_rows);

    auto total_start = std::chrono::steady_clock::now();
    UpdateIndexWriter writer(index_dir, durability_mode, update_threads, upsert_ids,
                             publish_handle);
    writer.set_collect_commit_timing(true);
    for (size_t offset = 0; offset < upsert_ids.size(); offset += batch_rows) {
        size_t count = std::min<size_t>(batch_rows, upsert_ids.size() - offset);
        size_t float_count = count * (size_t)update_dim;
        if (offset + count > (size_t)update_n) {
            throw std::runtime_error("update row offset exceeds fbin row count");
        }
        row_input.read(reinterpret_cast<char*>(rows.data()), float_count * sizeof(float));
        if (!row_input) throw std::runtime_error("short update fbin stream read: " + upsert_fbin);

        ids.assign(upsert_ids.begin() + (std::ptrdiff_t)offset,
                   upsert_ids.begin() + (std::ptrdiff_t)(offset + count));
        UpdateIndexMemoryBatch batch;
        batch.upsert_rows = rows.data();
        batch.upsert_n = (int)count;
        batch.upsert_dim = update_dim;
        batch.upsert_orig_ids.swap(ids);

        auto start = std::chrono::steady_clock::now();
        writer.commit_batch(batch);
        auto end = std::chrono::steady_clock::now();
        ids.swap(batch.upsert_orig_ids);
        stats.commit_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        const UpdateCommitTiming& timing = writer.last_commit_timing();
        stats.prepare_ms_sum += timing.prepare_ms;
        stats.encode_ms_sum += timing.encode_ms;
        stats.wal_commit_ms_sum += timing.wal_commit_ms;
        stats.publish_ms_sum += timing.publish_ms;
        if (timing.published) ++stats.published_count;
        if (timing.direct_publish) ++stats.direct_publish_count;
    }
    writer.flush_buffered_commits();
    auto total_end = std::chrono::steady_clock::now();
    stats.total_s = std::chrono::duration<double>(total_end - total_start).count();
}

} // namespace

int main(int argc, char** argv) {
    enable_transparent_huge_pages_for_process();
    if (argc < 7) {
        print_usage();
        return 1;
    }

    std::string index_dir = argv[1];
    std::string query_path = argv[2];
    std::string gt_path = argv[3];
    std::string upsert_fbin;
    std::string upsert_ids_path;
    std::string durability_mode = "buffered";
    uint32_t batch_rows = 0;
    int update_threads = 0;
    int query_batches = 1;

    IndexConfig config;
    config.num_threads = 1;

    try {
        for (int i = 4; i < argc; ++i) {
            std::string arg = argv[i];
            auto require_value = [&](const char* option) -> std::string {
                if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + option);
                return argv[++i];
            };
            if (arg == "-upsert_fbin") upsert_fbin = require_value("-upsert_fbin");
            else if (arg == "-upsert_ids") upsert_ids_path = require_value("-upsert_ids");
            else if (arg == "-batch_rows") batch_rows = parse_uint32_option(require_value("-batch_rows"), "-batch_rows");
            else if (arg == "-update_threads") update_threads = parse_positive_int_option(require_value("-update_threads"), "-update_threads");
            else if (arg == "-query_threads") config.num_threads = parse_positive_int_option(require_value("-query_threads"), "-query_threads");
            else if (arg == "-query_batches") query_batches = parse_positive_int_option(require_value("-query_batches"), "-query_batches");
            else if (arg == "-durability") durability_mode = require_value("-durability");
            else if (arg == "-topk") config.topk = parse_positive_int_option(require_value("-topk"), "-topk");
            else if (arg == "-M") config.num_subspaces = parse_positive_int_option(require_value("-M"), "-M");
            else if (arg == "-alpha") config.collision_ratio = std::stof(require_value("-alpha"));
            else if (arg == "-beta") config.candidate_ratio = std::stof(require_value("-beta"));
            else if (arg == "-max_w") config.max_w = parse_positive_int_option(require_value("-max_w"), "-max_w");
            else if (arg == "-score_prefetch") config.phase4_score_prefetch = std::stoi(require_value("-score_prefetch"));
            else throw std::runtime_error("unknown option: " + arg);
        }

        if (upsert_fbin.empty() || upsert_ids_path.empty() || batch_rows == 0) {
            throw std::runtime_error("missing required update options");
        }
        if (!is_valid_update_durability_mode(durability_mode)) {
            throw std::runtime_error("invalid durability mode: " + durability_mode);
        }

        int nq = 0, query_dim = 0;
        std::vector<float> queries = load_fvecs(query_path, nq, query_dim);
        int gt_n = 0, gt_dim = 0;
        std::vector<int> groundtruth = load_ivecs(gt_path, gt_n, gt_dim);
        if (gt_n < nq || gt_dim < config.topk) {
            throw std::runtime_error("ground truth shape is incompatible with queries/topk");
        }

        std::vector<uint32_t> upsert_ids = impl::load_update_ids(upsert_ids_path);
        impl::UpdateFbinMetadata update_meta =
            impl::read_update_fbin_metadata(upsert_fbin);
        if ((int)upsert_ids.size() != update_meta.n) {
            throw std::runtime_error("upsert vector count does not match upsert ID count");
        }
        if (update_meta.dim <= 0 ||
            (size_t)batch_rows > std::numeric_limits<size_t>::max() / (size_t)update_meta.dim) {
            throw std::runtime_error("batch_rows * dim overflows update row buffer");
        }

        QueryIndexHandle handle;
        handle.init(config);
        handle.load_index(index_dir);
        // Warm the per-thread query executor so the first timed query does not
        // pay the one-time rotation/base load cost.
        handle.search_batch(queries.data(), nq);

        std::atomic<bool> update_done{false};
        std::exception_ptr update_error;
        UpdateWorkerStats update_stats;

        auto query_start = std::chrono::steady_clock::now();
        std::thread update_thread([&]() {
            try {
                run_update_worker(index_dir,
                                  upsert_fbin,
                                  upsert_ids,
                                  update_meta.n,
                                  update_meta.dim,
                                  batch_rows,
                                  update_threads,
                                  durability_mode,
                                  &handle,
                                  update_stats);
            } catch (...) {
                update_error = std::current_exception();
            }
            update_done.store(true, std::memory_order_release);
        });

        int query_passes = 0;
        int64_t query_count = 0;
        double recall_sum = 0.0;
        do {
            std::vector<QueryResult> results = handle.search_batch(queries.data(), nq);
            recall_sum += recall_at_k(results, groundtruth, gt_dim, config.topk);
            query_count += nq;
            ++query_passes;
        } while (query_passes < query_batches ||
                 !update_done.load(std::memory_order_acquire));
        auto query_end = std::chrono::steady_clock::now();

        update_thread.join();
        if (update_error) std::rethrow_exception(update_error);

        double query_s = std::chrono::duration<double>(query_end - query_start).count();
        double query_qps = query_s > 0.0 ? (double)query_count / query_s : 0.0;
        double avg_recall = query_passes > 0 ? recall_sum / (double)query_passes : 0.0;
        double commit_sum_ms = std::accumulate(
            update_stats.commit_ms.begin(), update_stats.commit_ms.end(), 0.0);

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "mixed_query_passes=" << query_passes
                  << " query_count=" << query_count
                  << " query_s=" << query_s
                  << " query_qps=" << query_qps
                  << " recall@" << config.topk << "=" << avg_recall
                  << "\n";
        std::cout << "update_rows=" << upsert_ids.size()
                  << " batches=" << update_stats.commit_ms.size()
                  << " batch_rows=" << batch_rows
                  << " update_threads=" << update_threads
                  << " update_s=" << update_stats.total_s
                  << " update_rows_per_s="
                  << (update_stats.total_s > 0.0 ? (double)upsert_ids.size() / update_stats.total_s : 0.0)
                  << " update_commit_s=" << (commit_sum_ms / 1000.0)
                  << "\n";
        std::cout << "commit_ms_avg="
                  << (update_stats.commit_ms.empty() ? 0.0 : commit_sum_ms / update_stats.commit_ms.size())
                  << " p50=" << percentile_ms(update_stats.commit_ms, 50.0)
                  << " p95=" << percentile_ms(update_stats.commit_ms, 95.0)
                  << " p99=" << percentile_ms(update_stats.commit_ms, 99.0)
                  << " max=" << percentile_ms(update_stats.commit_ms, 100.0)
                  << "\n";
        double phase_batches = (double)update_stats.commit_ms.size();
        std::cout << "commit_phase_ms_avg"
                  << " prepare=" << (phase_batches > 0.0 ? update_stats.prepare_ms_sum / phase_batches : 0.0)
                  << " encode=" << (phase_batches > 0.0 ? update_stats.encode_ms_sum / phase_batches : 0.0)
                  << " wal_commit=" << (phase_batches > 0.0 ? update_stats.wal_commit_ms_sum / phase_batches : 0.0)
                  << " publish=" << (phase_batches > 0.0 ? update_stats.publish_ms_sum / phase_batches : 0.0)
                  << " direct_publish=" << update_stats.direct_publish_count
                  << "/" << update_stats.published_count
                  << "\n";
    } catch (const std::exception& e) {
        std::cerr << "score_update_query_mix_bench failed: " << e.what() << "\n";
        print_usage();
        return 1;
    }

    return 0;
}
