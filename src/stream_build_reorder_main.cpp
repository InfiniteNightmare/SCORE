#include "build/index_builder.h"
#include "build/streaming_build.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

using namespace score;

namespace {

std::string strip_trailing_slashes(std::string path) {
    while (path.size() > 1 && path.back() == '/') path.pop_back();
    return path;
}

void cleanup_reordered_output_contents(const std::string& index_dir, int M) noexcept {
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
    for (const char* name : scalar_names) ::unlink((index_dir + "/" + name).c_str());
    for (int m = 0; m < M; ++m) {
        ::unlink((index_dir + "/ivf_subspace_" + std::to_string(m) + ".bin").c_str());
        ::unlink((index_dir + "/posting_subspace_" + std::to_string(m) + ".bin").c_str());
    }
}

void cleanup_fused_staging_contents(const std::string& index_dir, int M) noexcept {
    const std::vector<std::string> names = streaming_no_reorder_artifact_names(M);
    for (const std::string& name : names) ::unlink((index_dir + "/" + name).c_str());
    for (int m = 0; m < M; ++m) {
        ::unlink((index_dir + "/posting_assign_subspace_" + std::to_string(m) + ".tmp").c_str());
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: score_stream_build_reorder <data.fbin> <output_reordered_index_dir> [options]\n"
                  << "Options:\n"
                  << "  -M <int>                          Subspaces (default: 16)\n"
                  << "  -rabitq_k <int>                   RaBitQ clusters (default: 200)\n"
                  << "  -rabitq_seed <int>                RaBitQ clustering seed (default: 42)\n"
                  << "  -ivf_k <int>                      IVF cells per subspace (default: 2500)\n"
                  << "  -sample_rows <int>                Shared training sample rows (default: 65536)\n"
                  << "  -block_mib <int>                  Streaming source block size in MiB\n"
                  << "  -train_iters <int>                K-means training iterations (default: 10)\n"
                  << "  -subspace_workers <int>           Concurrent IVF workers (default: M/full concurrency)\n"
                  << "  -pid_reorder_cell_key_depth <int> Reorder cell key depth (default: 2)\n"
                  << "  -pid_reorder_key_mode <cluster-first|cell-first|interleave> (default: cluster-first)\n"
                  << "  -pid_reorder_subspaces <csv>      Custom subspace key order\n"
                  << "  -io_rows <int>                    Rows per file chunk (default: 65536)\n"
                  << "  -bucket_buffer_rows <int>         Per-bucket buffered rows (default: 64)\n"
                  << "  -rewrite_workers <int>            Large-artifact rewrite workers (default: 4)\n";
        return 1;
    }

    std::string data_path = argv[1];
    std::string output_dir = strip_trailing_slashes(argv[2]);
    std::string staging_dir = output_dir + ".staging." + std::to_string(getpid());
    StreamingRabitqOptions rabitq_options;
    StreamingIvfPostingOptions ivf_options;
    ivf_options.materialize_no_reorder_postings = false;
    IndexConfig config;
    config.pid_reorder = true;
    size_t io_rows = 65536;
    size_t bucket_buffer_rows = 64;
    int rewrite_workers = 4;
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
        } else if (arg == "-pid_reorder_cell_key_depth" && i + 1 < argc) {
            config.pid_reorder_cell_key_depth = std::stoi(argv[++i]);
        } else if (arg == "-pid_reorder_key_mode" && i + 1 < argc) {
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

    auto [tag, n_detected, dim] = detect_vector_format(data_path);
    if (tag != 0) {
        std::cerr << "score_stream_build_reorder requires fbin input: " << data_path << std::endl;
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
        ivf_options.subspace_workers < 0 || io_rows == 0 || bucket_buffer_rows == 0 || rewrite_workers <= 0) {
        std::cerr << "build/reorder options must be positive; subspace_workers must be non-negative" << std::endl;
        return 1;
    }
    (void)n_detected;

    if (mkdir(output_dir.c_str(), 0755) != 0) {
        std::cerr << "Output index_dir must not already exist for score_stream_build_reorder: "
                  << output_dir << " (" << std::strerror(errno) << ")" << std::endl;
        return 1;
    }
    int output_fd = ::open(output_dir.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (output_fd < 0) {
        rmdir(output_dir.c_str());
        std::cerr << "Cannot open output directory safely" << std::endl;
        return 1;
    }
    std::string bound_output = "/proc/self/fd/" + std::to_string(output_fd);

    if (mkdir(staging_dir.c_str(), 0755) != 0) {
        cleanup_reordered_output_contents(bound_output, ivf_options.M);
        ::close(output_fd);
        rmdir(output_dir.c_str());
        std::cerr << "Staging index_dir must not already exist for score_stream_build_reorder: "
                  << staging_dir << " (" << std::strerror(errno) << ")" << std::endl;
        return 1;
    }
    int staging_fd = ::open(staging_dir.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (staging_fd < 0) {
        cleanup_reordered_output_contents(bound_output, ivf_options.M);
        ::close(output_fd);
        rmdir(output_dir.c_str());
        rmdir(staging_dir.c_str());
        std::cerr << "Cannot open staging directory safely" << std::endl;
        return 1;
    }
    std::string bound_staging = "/proc/self/fd/" + std::to_string(staging_fd);

    try {
        RandomRotation rotation;
        rotation.init(dim, 42, RotationType::FhtKac);
        save_rotation_exclusive(rotation, bound_staging + "/rotation.bin");
        write_rotate_subspace_flag_exclusive(bound_staging);

        std::cout << "[StreamBuildReorder] input=" << data_path
                  << " output=" << output_dir
                  << " staging=" << staging_dir
                  << " dim=" << dim
                  << " padded=" << rotation.padded_dim()
                  << " M=" << ivf_options.M
                  << " rabitq_k=" << rabitq_options.rabitq_k
                  << " ivf_k=" << ivf_options.ivf_k
                  << " rewrite_workers=" << rewrite_workers << std::endl;

        Timer timer;
        timer.reset();
        StreamingReorderStagingArtifacts staging = write_streaming_no_reorder_fused_artifacts(
            data_path, bound_staging, rotation, rabitq_options, ivf_options);
        double staging_ms = timer.elapsed_ms();
        log_build_rss_checkpoint("stream_build_reorder_after_staging");

        impl::copy_file_exclusive(bound_staging + "/rotation.bin", bound_output + "/rotation.bin");
        impl::copy_file_exclusive(bound_staging + "/rotate_subspace.flag", bound_output + "/rotate_subspace.flag");
        impl::copy_file_exclusive(staging.centroids_path, bound_output + "/rabitq_centroids.bin");
        for (int m = 0; m < staging.M; ++m) {
            impl::copy_file_exclusive(staging.ivf_paths[(size_t)m],
                                        bound_output + "/ivf_subspace_" + std::to_string(m) + ".bin");
        }

        const std::string raw_perm_path = bound_output + "/perm_to_orig.raw.tmp";
        const std::string scratch_perm_path = bound_output + "/perm_to_orig.scratch.tmp";
        impl::DiskPidReorderOptions reorder_options;
        reorder_options.n = staging.n;
        reorder_options.M = staging.M;
        reorder_options.assignment_key_range = (size_t)std::max(1, rabitq_options.rabitq_k);
        reorder_options.cells_per_subspace = staging.cells_per_subspace;
        reorder_options.subspace_assignment_paths = staging.subspace_assignment_paths;
        reorder_options.rabitq_assignments_path = staging.assignments_path;
        reorder_options.perm_to_orig_path = raw_perm_path;
        reorder_options.scratch_perm_path = scratch_perm_path;
        config.num_subspaces = staging.M;
        config.rabitq_k = rabitq_options.rabitq_k;
        reorder_options.config = config;
        reorder_options.io_rows = io_rows;
        reorder_options.bucket_buffer_rows = bucket_buffer_rows;
        Timer reorder_timer;
        reorder_timer.reset();
        log_build_rss_checkpoint("stream_build_reorder_before_reorder");
        impl::build_pid_reorder_perm_disk_staging(reorder_options);
        double reorder_ms = reorder_timer.elapsed_ms();
        log_build_rss_checkpoint("stream_build_reorder_after_reorder");

        impl::ReorderedArtifactFinalizeOptions finalize_options;
        finalize_options.n = staging.n;
        finalize_options.M = staging.M;
        finalize_options.cells_per_subspace = staging.cells_per_subspace;
        finalize_options.subspace_assignment_paths = staging.subspace_assignment_paths;
        finalize_options.raw_perm_to_orig_path = raw_perm_path;
        finalize_options.verify_input_path = staging.verify_path;
        finalize_options.assignments_input_path = staging.assignments_path;
        finalize_options.bin_data_input_path = staging.bin_data_path;
        finalize_options.verify_output_path = bound_output + "/verify_vectors.fbin";
        finalize_options.assignments_output_path = bound_output + "/rabitq_assignments.bin";
        finalize_options.bin_data_output_path = bound_output + "/rabitq_bin_data.bin";
        for (int m = 0; m < staging.M; ++m) {
            finalize_options.posting_output_paths.push_back(bound_output + "/posting_subspace_" + std::to_string(m) + ".bin");
        }
        finalize_options.final_perm_to_orig_path = bound_output + "/perm_to_orig.bin";
        finalize_options.io_rows = io_rows;
        finalize_options.rewrite_workers = rewrite_workers;
        Timer finalize_timer;
        finalize_timer.reset();
        log_build_rss_checkpoint("stream_build_reorder_before_finalize");
        impl::finalize_reordered_artifacts_from_staging(finalize_options);
        double finalize_ms = finalize_timer.elapsed_ms();
        log_build_rss_checkpoint("stream_build_reorder_after_finalize");

        for (const std::string& path : staging.subspace_assignment_paths) ::unlink(path.c_str());
        ::unlink(raw_perm_path.c_str());
        ::unlink(scratch_perm_path.c_str());
        cleanup_fused_staging_contents(bound_staging, staging.M);
        std::cout << "[StreamBuildReorder] staging_ms=" << staging_ms
                  << " reorder_ms=" << reorder_ms
                  << " finalize_ms=" << finalize_ms << std::endl;
        std::cout << "[StreamBuildReorder] complete in " << timer.elapsed_ms() << " ms" << std::endl;
        log_build_rss_checkpoint("stream_build_reorder_done");
    } catch (const std::exception& e) {
        cleanup_reordered_output_contents(bound_output, ivf_options.M);
        cleanup_fused_staging_contents(bound_staging, ivf_options.M);
        ::close(output_fd);
        ::close(staging_fd);
        rmdir(output_dir.c_str());
        rmdir(staging_dir.c_str());
        std::cerr << "score_stream_build_reorder failed: " << e.what() << std::endl;
        return 1;
    }

    bool close_ok = true;
    if (::close(output_fd) != 0) close_ok = false;
    if (::close(staging_fd) != 0) close_ok = false;
    if (rmdir(staging_dir.c_str()) != 0) close_ok = false;
    if (!close_ok) {
        std::cerr << "Failed closing/removing fused build directories" << std::endl;
        return 1;
    }
    return 0;
}
