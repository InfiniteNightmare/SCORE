#include "update/update_compaction.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <string>

using namespace score;

namespace {

void print_usage() {
    std::cerr
        << "Usage: score_update_index <index_dir> [operations] [options]\n"
        << "Operations:\n"
        << "  -insert_fbin <file> -insert_ids <ids>\n"
        << "  -delete_ids <ids>\n"
        << "  -upsert_fbin <file> -upsert_ids <ids>\n"
        << "Options:\n"
        << "  -durability <strict|buffered|none>\n"
        << "                             WAL durability mode (default: buffered)\n"
        << "  -batch_rows <int>          Maximum update rows per WAL commit (default: one commit)\n"
        << "  -threads <int>             Threads for update encoding (default: OMP default)\n"
        << "  -flush_fraction <float>\n"
        << "                             Flush L0 when WAL/base reaches this fraction (default: 0.05)\n"
        << "  -compaction_fraction <float>\n"
        << "                             Compact when total overhead reaches this fraction (default: 0.20)\n"
        << "\n"
        << "IDs may be whitespace/comma-separated text uint32 values or raw uint32 binary.\n";
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
        throw std::runtime_error(std::string("invalid fraction for ") + option + ": " + value);
    }
    if (consumed != value.size() || !std::isfinite(parsed) || parsed < 0.0) {
        throw std::runtime_error(std::string("invalid non-negative fraction for ") + option + ": " + value);
    }
    return parsed;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage();
        return 1;
    }

    std::string index_dir = argv[1];
    UpdateIndexBatch batch;
    UpdateCompactionPolicy compaction_policy;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        auto require_value = [&](const char* option) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + option);
            }
            return argv[++i];
        };

        try {
            if (arg == "-insert_fbin") batch.insert_fbin = require_value("-insert_fbin");
            else if (arg == "-insert_ids") batch.insert_ids = require_value("-insert_ids");
            else if (arg == "-delete_ids") batch.delete_ids = require_value("-delete_ids");
            else if (arg == "-upsert_fbin") batch.upsert_fbin = require_value("-upsert_fbin");
            else if (arg == "-upsert_ids") batch.upsert_ids = require_value("-upsert_ids");
            else if (arg == "-durability") {
                batch.durability_mode = require_value("-durability");
            }
            else if (arg == "-batch_rows") {
                uint32_t rows = parse_uint32_option(require_value("-batch_rows"), "-batch_rows");
                if (rows == 0) {
                    throw std::runtime_error("-batch_rows must be positive");
                }
                batch.batch_rows = rows;
            }
            else if (arg == "-threads") {
                uint32_t threads = parse_uint32_option(require_value("-threads"), "-threads");
                if (threads == 0) {
                    throw std::runtime_error("-threads must be positive");
                }
                if (threads > (uint32_t)std::numeric_limits<int>::max()) {
                    throw std::runtime_error("-threads exceeds int range");
                }
                batch.threads = (int)threads;
            }
            else if (arg == "-flush_fraction") {
                compaction_policy.flush_fraction =
                    parse_fraction_option(require_value("-flush_fraction"),
                                          "-flush_fraction");
            } else if (arg == "-compaction_fraction") {
                compaction_policy.compaction_fraction =
                    parse_fraction_option(require_value("-compaction_fraction"),
                                          "-compaction_fraction");
            }
            else {
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

    try {
        commit_update_batch(index_dir, batch);
        UpdateCompactionDecision compaction;
        if (impl::compaction_policy_has_triggers(compaction_policy)) {
            compaction = compact_index_by_policy(index_dir, compaction_policy);
        }
        std::cout << "score_update_index committed batch to " << index_dir;
        if (compaction.changed()) {
            std::cout << " (compaction:";
            if (compaction.flushed_l0) std::cout << " flush_l0";
            if (compaction.compacted) std::cout << " compacted";
            std::cout << ")";
        }
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "score_update_index failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
