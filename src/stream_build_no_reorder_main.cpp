#include "build/streaming_build.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

using namespace score;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: score_stream_build_no_reorder <data.fbin> <index_dir> [options]\n"
                  << "Options:\n"
                  << "  -M <int>              Subspaces (default: 16)\n"
                  << "  -rabitq_k <int>       RaBitQ clusters (default: 200)\n"
                  << "  -rabitq_seed <int>    RaBitQ clustering seed (default: 42)\n"
                  << "  -ivf_k <int>          IVF cells per subspace (default: 2500)\n"
                  << "  -sample_rows <int>    Shared training sample rows (default: 65536)\n"
                  << "  -block_mib <int>      Streaming source block size in MiB\n"
                  << "  -train_iters <int>    K-means training iterations (default: 10)\n"
                  << "  -subspace_workers <int>  Concurrent IVF subspace workers (default: M/full concurrency)\n"
                  << "  -preserve_posting_assignments  Keep validated assignment artifacts for faster posthoc reorder\n";
        return 1;
    }

    std::string data_path = argv[1];
    std::string index_dir = argv[2];
    StreamingRabitqOptions rabitq_options;
    StreamingIvfPostingOptions ivf_options;
    bool block_mib_set = false;
    unsigned long long block_mib = 0;

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-M" && i + 1 < argc) ivf_options.M = std::stoi(argv[++i]);
        else if (arg == "-rabitq_k" && i + 1 < argc) rabitq_options.rabitq_k = std::stoi(argv[++i]);
        else if (arg == "-rabitq_seed" && i + 1 < argc) rabitq_options.rabitq_seed = std::stoi(argv[++i]);
        else if (arg == "-ivf_k" && i + 1 < argc) ivf_options.ivf_k = std::stoi(argv[++i]);
        else if (arg == "-sample_rows" && i + 1 < argc) {
            int sample_rows = std::stoi(argv[++i]);
            rabitq_options.sample_rows = sample_rows;
            ivf_options.sample_rows = sample_rows;
        } else if (arg == "-block_mib" && i + 1 < argc) {
            block_mib = std::stoull(argv[++i]);
            block_mib_set = true;
        } else if (arg == "-train_iters" && i + 1 < argc) {
            int train_iters = std::stoi(argv[++i]);
            rabitq_options.train_iters = train_iters;
            ivf_options.train_iters = train_iters;
        } else if (arg == "-subspace_workers" && i + 1 < argc) {
            ivf_options.subspace_workers = std::stoi(argv[++i]);
        } else if (arg == "-preserve_posting_assignments") {
            ivf_options.preserve_assignment_artifacts = true;
        } else {
            std::cerr << "Unknown or incomplete option: " << arg << std::endl;
            return 1;
        }
    }

    auto [tag, n, dim] = detect_vector_format(data_path);
    if (tag < 0) {
        std::cerr << "Unsupported vector format: " << data_path << std::endl;
        return 1;
    }
    if (tag != 0) {
        std::cerr << "score_stream_build_no_reorder now requires fbin input for the optimized streaming path: "
                  << data_path << std::endl;
        return 1;
    }
    if (block_mib_set) {
        int block_rows_from_mib = 0;
        try {
            block_rows_from_mib = streaming_block_rows_for_mib(dim, (uint64_t)block_mib);
        } catch (const std::exception& e) {
            std::cerr << "Invalid -block_mib: " << e.what() << std::endl;
            return 1;
        }
        rabitq_options.block_rows = block_rows_from_mib;
        ivf_options.block_rows = block_rows_from_mib;
    }
    if (ivf_options.M <= 0 || ivf_options.ivf_k <= 0 || rabitq_options.rabitq_k <= 0 ||
        ivf_options.sample_rows <= 0 || ivf_options.block_rows <= 0 || ivf_options.train_iters <= 0 ||
        ivf_options.subspace_workers < 0) {
        std::cerr << "M, rabitq_k, ivf_k, sample_rows, block_rows, and train_iters must be positive; subspace_workers must be non-negative" << std::endl;
        return 1;
    }

    if (mkdir(index_dir.c_str(), 0755) != 0) {
        std::cerr << "Output index_dir must not already exist for score_stream_build_no_reorder: "
                  << index_dir << " (" << std::strerror(errno) << ")" << std::endl;
        return 1;
    }

    int dir_fd = ::open(index_dir.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (dir_fd < 0) {
        rmdir(index_dir.c_str());
        std::cerr << "Cannot open output index_dir safely: " << index_dir
                  << " (" << std::strerror(errno) << ")" << std::endl;
        return 1;
    }
    std::string bound_dir = "/proc/self/fd/" + std::to_string(dir_fd);

    try {
        RandomRotation rotation;
        rotation.init(dim, 42, RotationType::FhtKac);
        save_rotation_exclusive(rotation, bound_dir + "/rotation.bin");
        write_rotate_subspace_flag_exclusive(bound_dir);

        std::cout << "[StreamBuildNoReorder] input=" << data_path
                  << " n=" << n
                  << " dim=" << dim
                  << " padded=" << rotation.padded_dim()
                  << " M=" << ivf_options.M
                  << " rabitq_k=" << rabitq_options.rabitq_k
                  << " ivf_k=" << ivf_options.ivf_k
                  << " sample_rows=" << ivf_options.sample_rows
                  << " block_rows=" << ivf_options.block_rows
                  << " subspace_workers=" << ivf_options.subspace_workers
                  << " preserve_posting_assignments=" << (ivf_options.preserve_assignment_artifacts ? 1 : 0)
                  << std::endl;

        Timer timer;
        timer.reset();
        write_streaming_no_reorder_fused_artifacts(data_path, bound_dir, rotation,
                                                   rabitq_options, ivf_options);
        std::cout << "[StreamBuildNoReorder] complete in " << timer.elapsed_ms() << " ms" << std::endl;
        log_build_rss_checkpoint("stream_build_no_reorder_done");
    } catch (const std::exception& e) {
        cleanup_streaming_index_dir(bound_dir, streaming_no_reorder_artifact_names(ivf_options.M));
        ::close(dir_fd);
        rmdir(index_dir.c_str());
        std::cerr << "score_stream_build_no_reorder failed: " << e.what() << std::endl;
        return 1;
    }

    if (::close(dir_fd) != 0) {
        std::cerr << "Failed closing output index_dir: " << index_dir
                  << " (" << std::strerror(errno) << ")" << std::endl;
        return 1;
    }

    return 0;
}
