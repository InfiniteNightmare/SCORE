#include "update/update_compaction.h"

#include <iostream>
#include <string>

using namespace score;

namespace {

void print_usage() {
    std::cerr
        << "Usage: score_compact_index <index_dir> [-flush|-compaction|-compaction_reorder] [options]\n"
        << "  -flush               Flush L0 into a sealed delta segment\n"
        << "  -compaction          Fold sealed deltas into the base and remove update artifacts\n"
        << "  -compaction_reorder  Compaction that also rebuilds clean PID locality\n"
        << "Options for -compaction_reorder:\n"
        << "  -pid_reorder_cell_key_depth <int>  Reorder cell key depth (default: 2)\n"
        << "  -pid_reorder_key_mode <cluster-first|cell-first|interleave> (default: cluster-first)\n"
        << "  -pid_reorder_subspaces <csv>       Custom subspace key order\n"
        << "  -io_rows <int>                     Rows per file chunk (default: 65536)\n"
        << "  -bucket_buffer_rows <int>          Per-bucket buffered rows (default: 64)\n"
        << "  -rewrite_workers <int>             Large-artifact rewrite workers (default: 4)\n"
        << "  -rewrite_buffer_mb <int>           Large-artifact rewrite buffer (default: 64)\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage();
        return 1;
    }

    std::string index_dir = argv[1];
    std::string mode = argv[2];
    try {
        if (mode == "-flush") {
            if (argc != 3) {
                print_usage();
                return 1;
            }
            bool changed = flush_delta_segments(index_dir);
            std::cout << (changed ? "Flush published sealed delta" : "No L0 records to flush")
                      << " at " << index_dir << std::endl;
            return 0;
        }
        if (mode == "-compaction") {
            if (argc != 3) {
                print_usage();
                return 1;
            }
            bool changed = compact_index(index_dir);
            std::cout << (changed ? "Compaction published compact base"
                                  : "No update manifest; no compaction work")
                      << " at " << index_dir << std::endl;
            return 0;
        }
        if (mode == "-compaction_reorder") {
            CompactionOptions options;
            options.reorder = true;
            for (int i = 3; i < argc; ++i) {
                std::string arg = argv[i];
                if (arg == "-pid_reorder_cell_key_depth" && i + 1 < argc) {
                    options.reorder_config.pid_reorder_cell_key_depth = std::stoi(argv[++i]);
                } else if (arg == "-pid_reorder_key_mode" && i + 1 < argc) {
                    std::string key_mode = argv[++i];
                    if (key_mode == "cluster-first") options.reorder_config.pid_reorder_key_mode = 0;
                    else if (key_mode == "cell-first") options.reorder_config.pid_reorder_key_mode = 1;
                    else if (key_mode == "interleave") options.reorder_config.pid_reorder_key_mode = 2;
                    else {
                        std::cerr << "Unknown pid reorder key mode: " << key_mode << std::endl;
                        return 1;
                    }
                } else if (arg == "-pid_reorder_subspaces" && i + 1 < argc) {
                    parse_int_csv(argv[++i], options.reorder_config.pid_reorder_subspaces);
                } else if (arg == "-io_rows" && i + 1 < argc) {
                    options.io_rows = (size_t)std::stoll(argv[++i]);
                } else if (arg == "-bucket_buffer_rows" && i + 1 < argc) {
                    options.bucket_buffer_rows = (size_t)std::stoll(argv[++i]);
                } else if (arg == "-rewrite_workers" && i + 1 < argc) {
                    options.rewrite_workers = std::stoi(argv[++i]);
                } else if (arg == "-rewrite_buffer_mb" && i + 1 < argc) {
                    size_t mib = (size_t)std::stoull(argv[++i]);
                    if (mib == 0 || mib > (size_t)4096) {
                        std::cerr << "-rewrite_buffer_mb must be in [1, 4096]" << std::endl;
                        return 1;
                    }
                    options.rewrite_buffer_bytes = mib * 1024ull * 1024ull;
                } else {
                    std::cerr << "Unknown or incomplete option: " << arg << std::endl;
                    return 1;
                }
            }
            bool changed = compact_index(index_dir, options);
            std::cout << (changed ? "Reorder compaction published compact base"
                                  : "No update manifest; no compaction work")
                      << " at " << index_dir << std::endl;
            return 0;
        }
        print_usage();
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "score_compact_index failed: " << e.what() << std::endl;
        return 1;
    }
}
