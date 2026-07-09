#include "build/streaming_build.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

using namespace score;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: score_stream_ivf_posting <data.fvecs|data.fbin> <index_dir> [options]\n"
                  << "Options:\n"
                  << "  -append_existing     Add IVF/postings to an existing streaming index dir\n"
                  << "  -M <int>              Subspaces (default: 16)\n"
                  << "  -ivf_k <int>          IVF cells per subspace (default: 2500)\n"
                  << "  -sample_rows <int>    Training sample rows (default: 65536)\n"
                  << "  -block_mib <int>      Streaming source block size in MiB\n"
                  << "  -train_iters <int>    IVF training iterations (default: 10)\n";
        return 1;
    }

    std::string data_path = argv[1];
    std::string index_dir = argv[2];
    StreamingIvfPostingOptions options;
    bool append_existing = false;
    bool block_mib_set = false;
    unsigned long long block_mib = 0;

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-append_existing") append_existing = true;
        else if (arg == "-M" && i + 1 < argc) options.M = std::stoi(argv[++i]);
        else if (arg == "-ivf_k" && i + 1 < argc) options.ivf_k = std::stoi(argv[++i]);
        else if (arg == "-sample_rows" && i + 1 < argc) options.sample_rows = std::stoi(argv[++i]);
        else if (arg == "-block_mib" && i + 1 < argc) {
            block_mib = std::stoull(argv[++i]);
            block_mib_set = true;
        }
        else if (arg == "-train_iters" && i + 1 < argc) options.train_iters = std::stoi(argv[++i]);
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
    if (options.M <= 0 || options.ivf_k <= 0 || options.sample_rows <= 0 ||
        options.block_rows <= 0 || options.train_iters <= 0) {
        std::cerr << "M, ivf_k, sample_rows, block_rows, and train_iters must be positive" << std::endl;
        return 1;
    }

    if (!append_existing) {
        if (mkdir(index_dir.c_str(), 0755) != 0) {
            std::cerr << "Output index_dir must not already exist for score_stream_ivf_posting: "
                      << index_dir << " (" << std::strerror(errno) << ")" << std::endl;
            return 1;
        }
    }

    int dir_fd = ::open(index_dir.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (dir_fd < 0) {
        if (!append_existing) rmdir(index_dir.c_str());
        std::cerr << "Cannot open output index_dir safely: " << index_dir
                  << " (" << std::strerror(errno) << ")" << std::endl;
        return 1;
    }
    std::string bound_dir = "/proc/self/fd/" + std::to_string(dir_fd);

    RandomRotation rotation;
    bool rotate_flag_created = false;
    try {
        if (append_existing) {
            rotation.load(bound_dir + "/rotation.bin");
            if (rotation.dim() != dim) {
                throw std::runtime_error("existing rotation dimension does not match input data");
            }
        } else {
            rotation.init(dim, 42, RotationType::FhtKac);
            save_rotation_exclusive(rotation, bound_dir + "/rotation.bin");
        }
        std::ifstream flag_probe(bound_dir + "/rotate_subspace.flag");
        if (!flag_probe.good()) {
            write_rotate_subspace_flag_exclusive(bound_dir);
            rotate_flag_created = true;
        }

        std::cout << "[StreamIVFPosting] input=" << data_path
                  << " n=" << n
                  << " dim=" << dim
                  << " padded=" << rotation.padded_dim()
                  << " M=" << options.M
                  << " ivf_k=" << options.ivf_k
                  << " sample_rows=" << options.sample_rows
                  << " block_rows=" << options.block_rows
                  << (append_existing ? " append_existing=1" : "") << std::endl;

        Timer timer;
        timer.reset();
        write_streaming_ivf_posting_artifacts(data_path, bound_dir, rotation, options);
        std::cout << "[StreamIVFPosting] IVF/posting artifacts written in "
                  << timer.elapsed_ms() << " ms" << std::endl;
        log_build_rss_checkpoint("stream_ivf_posting_done");
    } catch (const std::exception& e) {
        if (!append_existing) {
            cleanup_streaming_index_dir(bound_dir, streaming_ivf_posting_artifact_names(options.M));
        } else if (rotate_flag_created) {
            ::unlink((bound_dir + "/rotate_subspace.flag").c_str());
        }
        ::close(dir_fd);
        if (!append_existing) rmdir(index_dir.c_str());
        std::cerr << "score_stream_ivf_posting failed: " << e.what() << std::endl;
        return 1;
    }

    if (::close(dir_fd) != 0) {
        std::cerr << "Failed closing output index_dir: " << index_dir
                  << " (" << std::strerror(errno) << ")" << std::endl;
        return 1;
    }

    return 0;
}
