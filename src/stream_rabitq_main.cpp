#include "build/streaming_build.h"

#include <cerrno>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/stat.h>

using namespace score;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: score_stream_rabitq <data.fvecs|data.fbin> <index_dir> [options]\n"
                  << "Options:\n"
                  << "  -rabitq_k <int>       RaBitQ clusters (default: 200)\n"
                  << "  -rabitq_seed <int>    RaBitQ clustering seed (default: 42)\n"
                  << "  -sample_rows <int>    Training sample rows (default: 65536)\n"
                  << "  -block_mib <int>      Streaming source block size in MiB\n";
        return 1;
    }

    std::string data_path = argv[1];
    std::string index_dir = argv[2];
    StreamingRabitqOptions options;
    bool block_mib_set = false;
    unsigned long long block_mib = 0;

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-rabitq_k" && i + 1 < argc) options.rabitq_k = std::stoi(argv[++i]);
        else if (arg == "-rabitq_seed" && i + 1 < argc) options.rabitq_seed = std::stoi(argv[++i]);
        else if (arg == "-sample_rows" && i + 1 < argc) options.sample_rows = std::stoi(argv[++i]);
        else if (arg == "-block_mib" && i + 1 < argc) {
            block_mib = std::stoull(argv[++i]);
            block_mib_set = true;
        }
        else {
            std::cerr << "Unknown or incomplete option: " << arg << std::endl;
            return 1;
        }
    }

    auto [tag, n, dim] = detect_vector_format(data_path);
    if (tag < 0) {
        std::cerr << "Unsupported vector format: " << data_path << std::endl;
        return 1;
    }
    if (block_mib_set) {
        try {
            options.block_rows = streaming_block_rows_for_mib(dim, (uint64_t)block_mib);
        } catch (const std::exception& e) {
            std::cerr << "Invalid -block_mib: " << e.what() << std::endl;
            return 1;
        }
    }
    if (options.block_rows <= 0 || options.sample_rows <= 0 || options.rabitq_k <= 0) {
        std::cerr << "rabitq_k, sample_rows, and block_rows must be positive" << std::endl;
        return 1;
    }

    if (mkdir(index_dir.c_str(), 0755) != 0) {
        std::cerr << "Output index_dir must not already exist for score_stream_rabitq: "
                  << index_dir << " (" << std::strerror(errno) << ")" << std::endl;
        return 1;
    }

    RandomRotation rotation;
    // Match the main builder: shared FhtKac rotation metadata uses seed 42;
    // `rabitq_seed` controls top-level clustering only.
    rotation.init(dim, 42, RotationType::FhtKac);
    try {
        save_rotation_exclusive(rotation, index_dir + "/rotation.bin");

        std::cout << "[StreamRaBitQ] input=" << data_path
                  << " n=" << n
                  << " dim=" << dim
                  << " padded=" << rotation.padded_dim()
                  << " rabitq_k=" << options.rabitq_k
                  << " sample_rows=" << options.sample_rows
                  << " block_rows=" << options.block_rows << std::endl;

        Timer timer;
        timer.reset();
        write_streaming_verify_vectors(data_path, index_dir + "/verify_vectors.fbin",
                                       rotation, options.block_rows);
        write_streaming_rabitq_artifacts(data_path, index_dir, rotation, options);
        std::cout << "[StreamRaBitQ] artifacts written in " << timer.elapsed_ms() << " ms" << std::endl;
        log_build_rss_checkpoint("stream_rabitq_done");
    } catch (const std::exception& e) {
        cleanup_streaming_index_dir(index_dir, {
            "rotation.bin",
            "verify_vectors.fbin",
            "rabitq_centroids.bin",
            "rabitq_assignments.bin",
            "rabitq_bin_data.bin",
        });
        std::cerr << "score_stream_rabitq failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
