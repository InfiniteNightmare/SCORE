#include "query/query_engine.h"
#include "update/update_compaction.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace score;

namespace {

void print_usage() {
    std::cerr
        << "Usage: score_update_stream_bench <index_dir> -upsert_fbin <file> (-upsert_ids <ids> | -mixed_ops <tsv>) [options]\n"
        << "Options:\n"
        << "  -mixed_ops <tsv>           Ordered TSV records: op orig_id row_index; op is insert/upsert/delete\n"
        << "  -batch_rows <int>          Rows per in-process update commit (required)\n"
        << "  -durability <strict|buffered|none>\n"
        << "                             WAL durability mode (default: buffered)\n"
        << "  -threads <int>             Threads for update encoding (default: OMP default)\n"
        << "  -reload_after_commit       Refresh a QueryEngine reader after every commit\n"
        << "  -query_fvecs <file>        Optional queries to run on the reloaded reader\n"
        << "  -groundtruth <file>        Ground truth for optional query recall\n"
        << "  -query_runs <int>          Optional query-set repetitions (default: 1)\n"
        << "  -profile_query             Print aggregate post-stream query phase timings\n"
        << "  -max_update_rows <int>     Consume only the first N update rows\n"
        << "  -flush_fraction <float>\n"
        << "                              Flush L0 when WAL/base reaches this fraction (default: 0.05)\n"
        << "  -compaction_fraction <float>\n"
        << "                              Compact when total overhead reaches this fraction (default: 0.20)\n";
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
    if (parsed > (uint64_t)std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(std::string("value exceeds uint32 range for ") + option + ": " + value);
    }
    return (uint32_t)parsed;
}

double parse_fraction_option(const std::string& value, const char* option) {
    size_t consumed = 0;
    double parsed = 0.0;
    try {
        parsed = std::stod(value, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("invalid floating point value for ") + option + ": " + value);
    }
    if (consumed != value.size() || !std::isfinite(parsed) || parsed < 0.0) {
        throw std::runtime_error(std::string("invalid non-negative fraction for ") + option + ": " + value);
    }
    return parsed;
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

void print_ms_stats(const char* name, const std::vector<double>& values) {
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    std::cout << name << "_avg=" << (values.empty() ? 0.0 : sum / values.size())
              << " p50=" << percentile_ms(values, 50.0)
              << " p95=" << percentile_ms(values, 95.0)
              << " p99=" << percentile_ms(values, 99.0)
              << " max=" << percentile_ms(values, 100.0)
              << "\n";
}

struct MixedUpdateRecord {
    char op = 0;
    uint32_t orig_id = 0;
    uint32_t row_index = 0;
};

char parse_mixed_op(const std::string& value, size_t line_no) {
    if (value == "insert" || value == "i" || value == "I") return 'i';
    if (value == "upsert" || value == "u" || value == "U") return 'u';
    if (value == "delete" || value == "d" || value == "D") return 'd';
    throw std::runtime_error("invalid mixed op at line " + std::to_string(line_no) +
                             ": " + value);
}

std::vector<MixedUpdateRecord> load_mixed_ops(const std::string& path,
                                              uint32_t max_records) {
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("cannot open mixed ops: " + path);
    }
    std::vector<MixedUpdateRecord> records;
    std::string line;
    size_t line_no = 0;
    while (std::getline(input, line)) {
        ++line_no;
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string op_text;
        std::string id_text;
        std::string row_text;
        if (!(iss >> op_text >> id_text)) {
            throw std::runtime_error("invalid mixed op record at line " +
                                     std::to_string(line_no));
        }
        MixedUpdateRecord record;
        record.op = parse_mixed_op(op_text, line_no);
        record.orig_id = parse_uint32_option(id_text, "-mixed_ops orig_id");
        if (record.op != 'd') {
            if (!(iss >> row_text)) {
                throw std::runtime_error("missing row_index for mixed op at line " +
                                         std::to_string(line_no));
            }
            record.row_index = parse_uint32_option(row_text, "-mixed_ops row_index");
        }
        records.push_back(record);
        if (max_records != 0 && records.size() >= max_records) break;
    }
    if (records.empty()) {
        throw std::runtime_error("mixed ops file is empty: " + path);
    }
    return records;
}

void read_fbin_rows_by_index(std::ifstream& input,
                             const std::string& path,
                             const std::vector<uint32_t>& row_indices,
                             int dim,
                             std::vector<float>& rows) {
    rows.resize(row_indices.size() * (size_t)dim);
    const std::streamoff payload_offset = (std::streamoff)(sizeof(int) * 3);
    const std::streamoff row_bytes = (std::streamoff)((size_t)dim * sizeof(float));
    for (size_t i = 0; i < row_indices.size(); ++i) {
        input.seekg(payload_offset + (std::streamoff)row_indices[i] * row_bytes,
                    std::ios::beg);
        if (!input) {
            throw std::runtime_error("cannot seek mixed row source: " + path);
        }
        input.read(reinterpret_cast<char*>(rows.data() + i * (size_t)dim),
                   row_bytes);
        if (!input) {
            throw std::runtime_error("short mixed row source read: " + path);
        }
    }
}

void print_query_profile_stats(const QueryProfileStats& stats) {
    double q = stats.queries > 0 ? (double)stats.queries : 1.0;
    auto avg_ms = [&](double us) { return us / q / 1000.0; };
    std::cout << "\n===== Post-stream Query Profile =====\n";
    std::cout << "Profiled queries: " << stats.queries << "\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "avg_total_ms: " << avg_ms(stats.total_us) << "\n";
    std::cout << "avg_rotate_ms: " << avg_ms(stats.rotate_us) << "\n";
    std::cout << "avg_activation_ms: " << avg_ms(stats.activation_us) << "\n";
    std::cout << "avg_score_reset_ms: " << avg_ms(stats.score_reset_us) << "\n";
    std::cout << "avg_cell_schedule_ms: " << avg_ms(stats.cell_schedule_us) << "\n";
    std::cout << "avg_posting_score_ms: " << avg_ms(stats.posting_score_us) << "\n";
    std::cout << "avg_candidate_select_ms: " << avg_ms(stats.candidate_select_us) << "\n";
    std::cout << "avg_candidate_dead_clear_ms: "
              << avg_ms(stats.candidate_dead_clear_us) << "\n";
    std::cout << "avg_candidate_topn_ms: " << avg_ms(stats.candidate_topn_us) << "\n";
    std::cout << "avg_refine_total_ms: " << avg_ms(stats.refine_total_us) << "\n";
    std::cout << "avg_rabitq_rank_ms: " << avg_ms(stats.rabitq_rank_us) << "\n";
    std::cout << "avg_verify_ms: " << avg_ms(stats.verify_us) << "\n";
    std::cout << "avg_phase4_cells: " << ((double)stats.phase4_cells / q) << "\n";
    std::cout << "avg_selected_candidates: " << ((double)stats.selected_candidates / q) << "\n";
    std::cout << "avg_candidate_dead_clear_pids: "
              << ((double)stats.candidate_dead_clear_pids / q) << "\n";
    std::cout << "avg_base_posting_ids: " << ((double)stats.base_posting_ids / q) << "\n";
    std::cout << "avg_delta_posting_ids: " << ((double)stats.delta_posting_ids / q) << "\n";
    std::cout << "avg_delta_posting_cells: "
              << ((double)stats.delta_posting_cells / q) << "\n";
    std::cout << "avg_inline_delta_posting_cells: "
              << ((double)stats.inline_delta_posting_cells / q) << "\n";
    std::cout << "avg_inline_delta_posting_ids: "
              << ((double)stats.inline_delta_posting_ids / q) << "\n";
    std::cout << "avg_buffered_delta_posting_cells: "
              << ((double)stats.buffered_delta_posting_cells / q) << "\n";
    std::cout << "avg_buffered_delta_posting_ids: "
              << ((double)stats.buffered_delta_posting_ids / q) << "\n";
    std::cout << "resolved_refine_query_ratio: "
              << ((double)stats.resolved_refine_queries / q) << "\n";
    std::cout << "avg_resolved_base_candidates: "
              << ((double)stats.resolved_base_candidates / q) << "\n";
    std::cout << "avg_resolved_delta_candidates: "
              << ((double)stats.resolved_delta_candidates / q) << "\n";
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

    if (argc < 7) {
        print_usage();
        return 1;
    }

    std::string index_dir = argv[1];
    std::string upsert_fbin;
    std::string upsert_ids_path;
    std::string mixed_ops_path;
    std::string query_path;
    std::string groundtruth_path;
    std::string durability_mode = "buffered";
    uint32_t batch_rows = 0;
    uint32_t compact_every_batches = 0;
    uint32_t max_update_rows = 0;
    uint32_t query_runs = 1;
    int threads = 0;
    UpdateCompactionPolicy policy;
    bool reload_after_commit = false;
    bool profile_query = false;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        auto require_value = [&](const char* option) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + option);
            }
            return argv[++i];
        };

        try {
            if (arg == "-upsert_fbin") upsert_fbin = require_value("-upsert_fbin");
            else if (arg == "-upsert_ids") upsert_ids_path = require_value("-upsert_ids");
            else if (arg == "-mixed_ops") mixed_ops_path = require_value("-mixed_ops");
            else if (arg == "-query_fvecs") query_path = require_value("-query_fvecs");
            else if (arg == "-groundtruth") groundtruth_path = require_value("-groundtruth");
            else if (arg == "-durability") durability_mode = require_value("-durability");
            else if (arg == "-batch_rows") {
                batch_rows = parse_uint32_option(require_value("-batch_rows"), "-batch_rows");
                if (batch_rows == 0) throw std::runtime_error("-batch_rows must be positive");
            } else if (arg == "-threads") {
                uint32_t parsed = parse_uint32_option(require_value("-threads"), "-threads");
                if (parsed == 0 || parsed > (uint32_t)std::numeric_limits<int>::max()) {
                    throw std::runtime_error("-threads must fit positive int range");
                }
                threads = (int)parsed;
            } else if (arg == "-compact_every_batches") {
                compact_every_batches =
                    parse_uint32_option(require_value("-compact_every_batches"),
                                        "-compact_every_batches");
            } else if (arg == "-reload_after_commit") {
                reload_after_commit = true;
            } else if (arg == "-profile_query") {
                profile_query = true;
            } else if (arg == "-max_update_rows") {
                max_update_rows =
                    parse_uint32_option(require_value("-max_update_rows"), "-max_update_rows");
                if (max_update_rows == 0) {
                    throw std::runtime_error("-max_update_rows must be positive");
                }
            } else if (arg == "-query_runs") {
                query_runs = parse_uint32_option(require_value("-query_runs"), "-query_runs");
                if (query_runs == 0) throw std::runtime_error("-query_runs must be positive");
            } else if (arg == "-flush_fraction") {
                policy.flush_fraction =
                    parse_fraction_option(require_value("-flush_fraction"),
                                          "-flush_fraction");
            } else if (arg == "-compaction_fraction") {
                policy.compaction_fraction =
                    parse_fraction_option(require_value("-compaction_fraction"),
                                          "-compaction_fraction");
            } else {
                std::cerr << "Unknown option: " << arg << "\n";
                print_usage();
                return 1;
            }
        } catch (const std::exception& e) {
            std::cerr << e.what() << "\n";
            print_usage();
            return 1;
        }
    }

    bool mixed_mode = !mixed_ops_path.empty();
    if (upsert_fbin.empty() || batch_rows == 0 ||
        (mixed_mode ? !upsert_ids_path.empty() : upsert_ids_path.empty())) {
        print_usage();
        return 1;
    }
    if (!is_valid_update_durability_mode(durability_mode)) {
        std::cerr << "invalid durability mode: " << durability_mode << "\n";
        print_usage();
        return 1;
    }
    if (compact_every_batches != 0 &&
        !impl::compaction_policy_has_triggers(policy)) {
        print_usage();
        return 1;
    }
    if (query_path.empty() != groundtruth_path.empty()) {
        std::cerr << "-query_fvecs and -groundtruth must be provided together\n";
        print_usage();
        return 1;
    }
    if (!query_path.empty() && !reload_after_commit) {
        std::cerr << "-query_fvecs requires -reload_after_commit\n";
        print_usage();
        return 1;
    }

    try {
        impl::UpdateFbinMetadata meta =
            impl::read_update_fbin_metadata(upsert_fbin);
        std::vector<uint32_t> upsert_ids;
        std::vector<MixedUpdateRecord> mixed_records;
        std::vector<uint32_t> incoming_ids_hint;
        size_t total_records = 0;
        size_t insert_records = 0;
        size_t upsert_records = 0;
        size_t delete_records = 0;
        if (mixed_mode) {
            mixed_records = load_mixed_ops(mixed_ops_path, max_update_rows);
            for (const MixedUpdateRecord& record : mixed_records) {
                if (record.op == 'i') {
                    ++insert_records;
                } else if (record.op == 'u') {
                    ++upsert_records;
                } else if (record.op == 'd') {
                    ++delete_records;
                }
                if (record.op != 'd' && record.row_index >= (uint32_t)meta.n) {
                    throw std::runtime_error("mixed op row_index exceeds row source vector count");
                }
                incoming_ids_hint.push_back(record.orig_id);
            }
            total_records = mixed_records.size();
        } else {
            upsert_ids = impl::load_update_ids(upsert_ids_path);
            if (max_update_rows != 0) {
                if (max_update_rows > upsert_ids.size() ||
                    max_update_rows > (uint32_t)meta.n) {
                    throw std::runtime_error("-max_update_rows exceeds update input row count");
                }
                upsert_ids.resize(max_update_rows);
                meta.n = (int)max_update_rows;
            }
            if ((int)upsert_ids.size() != meta.n) {
                throw std::runtime_error("upsert vector count does not match upsert ID count");
            }
            total_records = upsert_ids.size();
            upsert_records = upsert_ids.size();
            incoming_ids_hint = upsert_ids;
        }
        if (meta.dim <= 0 ||
            (size_t)batch_rows > std::numeric_limits<size_t>::max() / (size_t)meta.dim) {
            throw std::runtime_error("batch_rows * dim overflows stream bench row buffer");
        }

        constexpr int query_topk = 10;
        int nq = 0, query_dim = 0;
        int gt_n = 0, gt_dim = 0;
        std::vector<float> queries;
        std::vector<int> groundtruth;
        if (!query_path.empty()) {
            queries = load_fvecs(query_path, nq, query_dim);
            groundtruth = load_ivecs(groundtruth_path, gt_n, gt_dim);
            if (gt_n < nq || gt_dim < query_topk) {
                throw std::runtime_error("ground truth shape is incompatible with query benchmark");
            }
        }

        std::ifstream row_input(upsert_fbin, std::ios::binary);
        if (!row_input.is_open()) {
            throw std::runtime_error("cannot open update fbin: " + upsert_fbin);
        }
        if (!mixed_mode) {
            row_input.seekg((std::streamoff)(sizeof(int) * 3), std::ios::beg);
            if (!row_input) {
                throw std::runtime_error("cannot seek update fbin payload: " + upsert_fbin);
            }
        }
        std::vector<float> rows((size_t)batch_rows * (size_t)meta.dim);
        std::vector<uint32_t> ids;
        ids.reserve(batch_rows);
        std::vector<float> insert_rows;
        std::vector<float> upsert_rows;
        std::vector<uint32_t> insert_ids;
        std::vector<uint32_t> batch_upsert_ids;
        std::vector<uint32_t> delete_ids;
        std::vector<uint32_t> insert_row_indices;
        std::vector<uint32_t> upsert_row_indices;

        std::vector<double> commit_ms;
        commit_ms.reserve((total_records + batch_rows - 1) / batch_rows);
        std::vector<double> commit_prepare_ms;
        commit_prepare_ms.reserve((total_records + batch_rows - 1) / batch_rows);
        std::vector<double> commit_encode_ms;
        commit_encode_ms.reserve((total_records + batch_rows - 1) / batch_rows);
        std::vector<double> commit_wal_ms;
        commit_wal_ms.reserve((total_records + batch_rows - 1) / batch_rows);
        std::vector<double> commit_publish_ms;
        commit_publish_ms.reserve((total_records + batch_rows - 1) / batch_rows);
        std::vector<double> commit_internal_total_ms;
        commit_internal_total_ms.reserve((total_records + batch_rows - 1) / batch_rows);
        std::vector<double> reload_ms;
        reload_ms.reserve((total_records + batch_rows - 1) / batch_rows);
        size_t direct_publish_count = 0;
        std::chrono::steady_clock::time_point total_start;
        UpdateCompactionRunStats compaction;
        double compaction_s = 0.0;
        std::unique_ptr<QueryIndexHandle> reload_handle;

        bool has_compaction_policy = impl::compaction_policy_has_triggers(policy);
        uint32_t policy_check_every_batches = compact_every_batches;
        if (policy_check_every_batches == 0 && has_compaction_policy) {
            policy_check_every_batches = 1;
        }

        {
            std::unique_ptr<UpdateService> service;
            auto ensure_service = [&]() -> UpdateService& {
                if (!service) {
                    service = std::make_unique<UpdateService>(
                        index_dir,
                        policy,
                        durability_mode,
                        threads,
                        incoming_ids_hint,
                        reload_handle ? reload_handle.get() : nullptr);
                    service->set_auto_compact_after_commit(false);
                    service->set_collect_commit_timing(true);
                }
                return *service;
            };
            auto close_service = [&]() {
                if (!service) return;
                service->flush_buffered_commits();
                service.reset();
            };
            auto run_policy_compaction =
                [&](UpdateService& active_service, size_t max_passes) {
                auto compaction_start = std::chrono::steady_clock::now();
                UpdateCompactionRunStats stats =
                    active_service.compact_until_quiescent(max_passes);
                auto compaction_end = std::chrono::steady_clock::now();
                compaction_s +=
                    std::chrono::duration<double>(compaction_end - compaction_start).count();
                compaction.absorb(stats);
                if (active_service.requires_reopen_after_compaction()) {
                    service.reset();
                }
                return stats.changed();
            };
            auto drain_delta_compaction = [&](UpdateService& active_service) {
                auto compaction_start = std::chrono::steady_clock::now();
                UpdateCompactionDecision decision =
                    active_service.flush_delta_segments();
                auto compaction_end = std::chrono::steady_clock::now();
                compaction_s +=
                    std::chrono::duration<double>(compaction_end - compaction_start).count();
                compaction.record(decision);
                return decision.changed();
            };
            if (reload_after_commit) {
                IndexConfig reload_config;
                reload_config.num_threads = 1;
                reload_config.query_profile = profile_query;
                reload_handle = std::make_unique<QueryIndexHandle>();
                reload_handle->init(reload_config);
                reload_handle->load_index(index_dir);
            }
            total_start = std::chrono::steady_clock::now();

            uint32_t batches_since_compaction = 0;
            for (size_t offset = 0; offset < total_records; offset += batch_rows) {
                size_t count = std::min<size_t>(batch_rows, total_records - offset);
                UpdateIndexMemoryBatch batch;
                if (mixed_mode) {
                    insert_ids.clear();
                    batch_upsert_ids.clear();
                    delete_ids.clear();
                    insert_row_indices.clear();
                    upsert_row_indices.clear();
                    for (size_t i = offset; i < offset + count; ++i) {
                        const MixedUpdateRecord& record = mixed_records[i];
                        if (record.op == 'i') {
                            insert_ids.push_back(record.orig_id);
                            insert_row_indices.push_back(record.row_index);
                        } else if (record.op == 'u') {
                            batch_upsert_ids.push_back(record.orig_id);
                            upsert_row_indices.push_back(record.row_index);
                        } else {
                            delete_ids.push_back(record.orig_id);
                        }
                    }
                    if (!insert_row_indices.empty()) {
                        read_fbin_rows_by_index(row_input, upsert_fbin, insert_row_indices,
                                                meta.dim, insert_rows);
                        batch.insert_rows = insert_rows.data();
                        batch.insert_n = (int)insert_ids.size();
                        batch.insert_dim = meta.dim;
                        batch.insert_orig_ids.swap(insert_ids);
                    }
                    if (!upsert_row_indices.empty()) {
                        read_fbin_rows_by_index(row_input, upsert_fbin, upsert_row_indices,
                                                meta.dim, upsert_rows);
                        batch.upsert_rows = upsert_rows.data();
                        batch.upsert_n = (int)batch_upsert_ids.size();
                        batch.upsert_dim = meta.dim;
                        batch.upsert_orig_ids.swap(batch_upsert_ids);
                    }
                    batch.delete_orig_ids.swap(delete_ids);
                } else {
                    size_t float_count = count * (size_t)meta.dim;
                    row_input.read(reinterpret_cast<char*>(rows.data()),
                                   float_count * sizeof(float));
                    if (!row_input) {
                        throw std::runtime_error("short update fbin stream read: " + upsert_fbin);
                    }
                    ids.assign(upsert_ids.begin() + (std::ptrdiff_t)offset,
                               upsert_ids.begin() + (std::ptrdiff_t)(offset + count));
                    batch.upsert_rows = rows.data();
                    batch.upsert_n = (int)count;
                    batch.upsert_dim = meta.dim;
                    batch.upsert_orig_ids.swap(ids);
                }

                UpdateService& active_service = ensure_service();
                auto start = std::chrono::steady_clock::now();
                active_service.commit_batch(batch);
                auto end = std::chrono::steady_clock::now();
                UpdateCommitTiming timing = active_service.last_commit_timing();
                if (mixed_mode) {
                    insert_ids.swap(batch.insert_orig_ids);
                    batch_upsert_ids.swap(batch.upsert_orig_ids);
                    delete_ids.swap(batch.delete_orig_ids);
                } else {
                    ids.swap(batch.upsert_orig_ids);
                }
                commit_ms.push_back(
                    std::chrono::duration<double, std::milli>(end - start).count());
                commit_prepare_ms.push_back(timing.prepare_ms);
                commit_encode_ms.push_back(timing.encode_ms);
                commit_wal_ms.push_back(timing.wal_commit_ms);
                commit_publish_ms.push_back(timing.publish_ms);
                commit_internal_total_ms.push_back(timing.total_ms);
                if (reload_after_commit) {
                    reload_ms.push_back(timing.publish_ms);
                    if (timing.direct_publish) ++direct_publish_count;
                }
                ++batches_since_compaction;

                if (policy_check_every_batches != 0 &&
                    batches_since_compaction >= policy_check_every_batches) {
                    run_policy_compaction(active_service, 1);
                    batches_since_compaction = 0;
                }
            }
            if (service && has_compaction_policy && batches_since_compaction != 0) {
                run_policy_compaction(*service, 1);
            }
            if (service && has_compaction_policy) {
                run_policy_compaction(*service, 0);
                drain_delta_compaction(*service);
            }
            close_service();
        }
        auto total_end = std::chrono::steady_clock::now();

        double commit_sum_ms = std::accumulate(commit_ms.begin(), commit_ms.end(), 0.0);
        double total_s = std::chrono::duration<double>(total_end - total_start).count();
        double throughput = total_records == 0 ? 0.0 : (double)total_records / total_s;
        UpdateManifest manifest = load_update_manifest_json_for_index(index_dir);

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "rows=" << total_records
                  << " batches=" << commit_ms.size()
                  << " batch_rows=" << batch_rows
                  << " inserts=" << insert_records
                  << " upserts=" << upsert_records
                  << " deletes=" << delete_records
                  << " compact_every_batches=" << compact_every_batches
                  << " policy_check_every_batches=" << policy_check_every_batches
                  << " durability=" << durability_mode
                  << " total_s=" << total_s
                  << " update_commit_s=" << (commit_sum_ms / 1000.0)
                  << " rows_per_s=" << throughput
                  << "\n";
        std::cout << "commit_ms_avg=" << (commit_ms.empty() ? 0.0 : commit_sum_ms / commit_ms.size())
                  << " p50=" << percentile_ms(commit_ms, 50.0)
                  << " p95=" << percentile_ms(commit_ms, 95.0)
                  << " p99=" << percentile_ms(commit_ms, 99.0)
                  << " max=" << percentile_ms(commit_ms, 100.0)
                  << "\n";
        print_ms_stats("commit_prepare_ms", commit_prepare_ms);
        print_ms_stats("commit_encode_ms", commit_encode_ms);
        print_ms_stats("commit_wal_ms", commit_wal_ms);
        print_ms_stats("commit_publish_ms", commit_publish_ms);
        print_ms_stats("commit_internal_total_ms", commit_internal_total_ms);
        if (reload_after_commit) {
            double reload_sum_ms = std::accumulate(reload_ms.begin(), reload_ms.end(), 0.0);
            std::cout << "reader_publish_ms_avg="
                      << (reload_ms.empty() ? 0.0 : reload_sum_ms / reload_ms.size())
                      << " p50=" << percentile_ms(reload_ms, 50.0)
                      << " p95=" << percentile_ms(reload_ms, 95.0)
                      << " p99=" << percentile_ms(reload_ms, 99.0)
                      << " max=" << percentile_ms(reload_ms, 100.0)
                      << " direct=" << direct_publish_count
                      << " fallback=" << (reload_ms.size() - direct_publish_count)
                      << "\n";
        }
        std::cout << "manifest_max_pid=" << manifest.max_pid
                  << " live_count=" << manifest.live_count
                  << " last_committed_lsn=" << manifest.last_committed_lsn
                  << " sealed_deltas=" << manifest.sealed_deltas.size()
                  << " l0_committed=" << (manifest.last_committed_lsn != 0 ? 1 : 0)
                  << "\n";
        if (reload_after_commit && reload_handle) {
            std::shared_ptr<const ReaderView> view = reload_handle->snapshot();
            if (!view) {
                throw std::runtime_error("reload handle has no published reader snapshot");
            }
            size_t l0_blocks = view->growing_delta
                ? view->growing_delta->committed_blocks.size()
                : 0;
            size_t l0_layers = view->growing_delta
                ? view->growing_delta->posting_layers.size()
                : 0;
            uint64_t l0_layer_ids = view->growing_delta
                ? view->growing_delta->posting_layer_total_ids()
                : 0;
            std::cout << "reader_l0_blocks=" << l0_blocks
                      << " reader_l0_layers=" << l0_layers
                      << " reader_l0_layer_ids=" << l0_layer_ids
                      << " reader_sealed_deltas=" << view->sealed_deltas.size()
                      << "\n";
        }
        if (impl::compaction_policy_has_triggers(policy)) {
            std::cout << "compaction_s=" << compaction_s
                      << " compaction_runs=" << compaction.passes
                      << " compaction_not_needed=" << compaction.no_op_passes
                      << " flushed_l0=" << compaction.flushed_l0
                      << " compaction=" << compaction.compacted
                      << "\n";
        }
        if (reload_handle && !query_path.empty()) {
            if (query_dim != reload_handle->dim()) {
                throw std::runtime_error("query dimension does not match loaded index");
            }
            if (nq > 0) {
                reload_handle->search_batch(queries.data(), 1);
            }
            if (profile_query) reload_handle->reset_profile_stats();
            auto query_start = std::chrono::steady_clock::now();
            double recall_sum = 0.0;
            int64_t query_count = 0;
            for (uint32_t run = 0; run < query_runs; ++run) {
                std::vector<QueryResult> results =
                    reload_handle->search_batch(queries.data(), nq);
                recall_sum += recall_at_k(results, groundtruth, gt_dim, query_topk);
                query_count += nq;
            }
            auto query_end = std::chrono::steady_clock::now();
            double query_s =
                std::chrono::duration<double>(query_end - query_start).count();
            std::cout << "post_stream_query_runs=" << query_runs
                      << " query_count=" << query_count
                      << " query_s=" << query_s
                      << " query_qps="
                      << (query_s > 0.0 ? (double)query_count / query_s : 0.0)
                      << " recall@10=" << (recall_sum / (double)query_runs)
                      << "\n";
            if (profile_query) {
                print_query_profile_stats(reload_handle->collect_profile_stats());
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "score_update_stream_bench failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
