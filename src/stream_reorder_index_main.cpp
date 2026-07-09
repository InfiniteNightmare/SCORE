#include "build/index_builder.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

using namespace score;

namespace {

int read_assignment_count(const std::string& index_dir) {
    std::ifstream in(index_dir + "/rabitq_assignments.bin", std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("cannot open rabitq_assignments.bin");
    int n = 0;
    in.read(reinterpret_cast<char*>(&n), sizeof(int));
    if (!in || n <= 0) throw std::runtime_error("invalid rabitq_assignments.bin header");
    return n;
}

int read_rabitq_k(const std::string& index_dir) {
    std::ifstream in(index_dir + "/rabitq_centroids.bin", std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("cannot open rabitq_centroids.bin");
    int K = 0, dim = 0, padded_dim = 0;
    in.read(reinterpret_cast<char*>(&K), sizeof(int));
    in.read(reinterpret_cast<char*>(&dim), sizeof(int));
    in.read(reinterpret_cast<char*>(&padded_dim), sizeof(int));
    if (!in || K <= 0 || dim <= 0 || padded_dim <= 0) {
        throw std::runtime_error("invalid rabitq_centroids.bin header");
    }
    return K;
}

int detect_subspace_count(const std::string& index_dir) {
    int M = 0;
    while (true) {
        std::ifstream in(index_dir + "/ivf_subspace_" + std::to_string(M) + ".bin", std::ios::binary);
        if (!in.is_open()) break;
        ++M;
    }
    if (M <= 0) throw std::runtime_error("cannot detect ivf_subspace_*.bin files");
    return M;
}

void cleanup_output_dir_contents(const std::string& index_dir, int M) noexcept {
    const char* scalar_names[] = {
        "rotation.bin",
        "rotate_subspace.flag",
        "subspace_rotation.bin",
        "rabitq_centroids.bin",
        "verify_vectors.fbin",
        "rabitq_assignments.bin",
        "rabitq_bin_data.bin",
        "perm_to_orig.bin",
        "perm_to_orig.raw.tmp",
        "perm_to_orig.scratch.tmp",
        "orig_to_new.reorder.tmp",
    };
    for (const char* name : scalar_names) {
        ::unlink((index_dir + "/" + name).c_str());
    }
    for (int m = 0; m < M; ++m) {
        ::unlink((index_dir + "/ivf_subspace_" + std::to_string(m) + ".bin").c_str());
        ::unlink((index_dir + "/posting_subspace_" + std::to_string(m) + ".bin").c_str());
        ::unlink((index_dir + "/posting_assign_subspace_" + std::to_string(m) + ".reorder.tmp").c_str());
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: score_stream_reorder_index <source_no_reorder_index_dir> <output_reordered_index_dir> [options]\n"
                  << "Options:\n"
                  << "  -M <int>                         Assert detected subspaces match this value\n"
                  << "  -pid_reorder_cell_key_depth <int> Reorder cell key depth (default: 2)\n"
                  << "  -pid_reorder_key_mode <cluster-first|cell-first|interleave> (default: cluster-first)\n"
                  << "  -pid_reorder_subspaces <csv>     Custom subspace key order\n"
                  << "  -io_rows <int>                   Rows per file chunk (default: 65536)\n"
                  << "  -bucket_buffer_rows <int>        Per-bucket buffered rows (default: 64)\n"
                  << "  -rewrite_workers <int>           Large-artifact rewrite workers (default: 4)\n";
        return 1;
    }

    std::string source_dir = argv[1];
    std::string output_dir = argv[2];
    IndexConfig config;
    int M_assert = 0;
    size_t io_rows = 65536;
    size_t bucket_buffer_rows = 64;
    int rewrite_workers = 4;

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-M" && i + 1 < argc) M_assert = std::stoi(argv[++i]);
        else if (arg == "-pid_reorder_cell_key_depth" && i + 1 < argc) config.pid_reorder_cell_key_depth = std::stoi(argv[++i]);
        else if (arg == "-pid_reorder_key_mode" && i + 1 < argc) {
            std::string mode = argv[++i];
            if (mode == "cluster-first") config.pid_reorder_key_mode = 0;
            else if (mode == "cell-first") config.pid_reorder_key_mode = 1;
            else if (mode == "interleave") config.pid_reorder_key_mode = 2;
            else {
                std::cerr << "Unknown pid reorder key mode: " << mode << std::endl;
                return 1;
            }
        } else if (arg == "-pid_reorder_subspaces" && i + 1 < argc) {
            parse_int_csv(argv[++i], config.pid_reorder_subspaces);
        } else if (arg == "-io_rows" && i + 1 < argc) {
            io_rows = (size_t)std::stoll(argv[++i]);
        } else if (arg == "-bucket_buffer_rows" && i + 1 < argc) {
            bucket_buffer_rows = (size_t)std::stoll(argv[++i]);
        } else if (arg == "-rewrite_workers" && i + 1 < argc) {
            rewrite_workers = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown or incomplete option: " << arg << std::endl;
            return 1;
        }
    }

    if (io_rows == 0 || bucket_buffer_rows == 0 || M_assert < 0 || rewrite_workers <= 0) {
        std::cerr << "M assertion must be non-negative; rewrite_workers, io_rows, and bucket_buffer_rows must be positive" << std::endl;
        return 1;
    }
    if (::access((source_dir + "/perm_to_orig.bin").c_str(), F_OK) == 0) {
        std::cerr << "Source index already contains perm_to_orig.bin; refusing double reorder" << std::endl;
        return 1;
    }
    if (mkdir(output_dir.c_str(), 0755) != 0) {
        std::cerr << "Output index_dir must not already exist for score_stream_reorder_index: "
                  << output_dir << " (" << std::strerror(errno) << ")" << std::endl;
        return 1;
    }

    int source_fd = ::open(source_dir.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (source_fd < 0) {
        ::rmdir(output_dir.c_str());
        std::cerr << "Cannot open source index_dir safely: " << source_dir
                  << " (" << std::strerror(errno) << ")" << std::endl;
        return 1;
    }
    int output_fd = ::open(output_dir.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (output_fd < 0) {
        ::close(source_fd);
        ::rmdir(output_dir.c_str());
        std::cerr << "Cannot open output index_dir safely: " << output_dir
                  << " (" << std::strerror(errno) << ")" << std::endl;
        return 1;
    }

    std::string bound_source = "/proc/self/fd/" + std::to_string(source_fd);
    std::string bound_output = "/proc/self/fd/" + std::to_string(output_fd);
    int M = 0;
    try {
        int n = read_assignment_count(bound_source);
        int rabitq_k = read_rabitq_k(bound_source);
        M = detect_subspace_count(bound_source);
        if (M_assert > 0 && M_assert != M) {
            throw std::runtime_error("-M assertion does not match detected source subspace count");
        }
        config.num_subspaces = M;
        config.rabitq_k = rabitq_k;

        impl::PosthocReorderIndexOptions options;
        options.source_index_dir = bound_source;
        options.output_index_dir = bound_output;
        options.n = n;
        options.M = M;
        options.config = config;
        options.io_rows = io_rows;
        options.bucket_buffer_rows = bucket_buffer_rows;
        options.rewrite_workers = rewrite_workers;

        std::cout << "[StreamReorderIndex] source=" << source_dir
                  << " output=" << output_dir
                  << " n=" << n
                  << " M=" << M
                  << " rabitq_k=" << rabitq_k
                  << " key_mode=" << config.pid_reorder_key_mode
                  << " key_depth=" << config.pid_reorder_cell_key_depth
                  << " rewrite_workers=" << rewrite_workers << std::endl;
        Timer timer;
        timer.reset();
        impl::posthoc_reorder_no_reorder_index(options);
        std::cout << "[StreamReorderIndex] complete in " << timer.elapsed_ms() << " ms" << std::endl;
        log_build_rss_checkpoint("stream_reorder_index_done");
    } catch (const std::exception& e) {
        cleanup_output_dir_contents(bound_output, std::max(0, M));
        ::close(source_fd);
        ::close(output_fd);
        std::cerr << "score_stream_reorder_index failed: " << e.what() << std::endl;
        return 1;
    }

    bool close_ok = true;
    if (::close(source_fd) != 0) close_ok = false;
    if (::close(output_fd) != 0) close_ok = false;
    if (!close_ok) {
        std::cerr << "Failed closing index directories" << std::endl;
        return 1;
    }
    return 0;
}
