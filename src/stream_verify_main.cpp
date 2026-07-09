#include "build/streaming_build.h"

#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/stat.h>

using namespace score;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: score_stream_verify <data.fvecs|data.fbin> <index_dir> [block_mib]" << std::endl;
        std::cerr << "Writes rotation.bin and verify_vectors.fbin using blockwise streaming." << std::endl;
        return 1;
    }

    std::string data_path = argv[1];
    std::string index_dir = argv[2];
    int block_rows = 65536;
    unsigned long long block_mib = 0;
    bool block_mib_set = false;
    if (argc >= 4) {
        block_mib = std::stoull(argv[3]);
        block_mib_set = true;
    }

    auto [tag, n, dim] = detect_vector_format(data_path);
    if (tag < 0) {
        std::cerr << "Unsupported vector format: " << data_path << std::endl;
        return 1;
    }
    if (block_mib_set) {
        try {
            block_rows = streaming_block_rows_for_mib(dim, (uint64_t)block_mib);
        } catch (const std::exception& e) {
            std::cerr << "Invalid block_mib: " << e.what() << std::endl;
            return 1;
        }
    }

    if (mkdir(index_dir.c_str(), 0755) != 0) {
        std::cerr << "Output index_dir must not already exist for score_stream_verify: "
                  << index_dir << " (" << std::strerror(errno) << ")" << std::endl;
        return 1;
    }

    RandomRotation rotation;
    rotation.init(dim, 42, RotationType::FhtKac);
    try {
        save_rotation_exclusive(rotation, index_dir + "/rotation.bin");

        std::cout << "[StreamVerify] input=" << data_path
                  << " n=" << n
                  << " dim=" << dim
                  << " padded=" << rotation.padded_dim()
                  << " block_rows=" << block_rows << std::endl;

        Timer timer;
        timer.reset();
        write_streaming_verify_vectors(data_path, index_dir + "/verify_vectors.fbin", rotation, block_rows);
        std::cout << "[StreamVerify] verify_vectors.fbin written in "
                  << timer.elapsed_ms() << " ms" << std::endl;
        log_build_rss_checkpoint("stream_verify_done");
    } catch (const std::exception& e) {
        cleanup_streaming_index_dir(index_dir, {"rotation.bin", "verify_vectors.fbin"});
        std::cerr << "score_stream_verify failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
