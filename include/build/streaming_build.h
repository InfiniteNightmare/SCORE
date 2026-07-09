#pragma once

#include "core/common.h"
#include "core/ivf.h"
#include "core/rotation.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <fstream>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include "rabitqlib/quantization/rabitq.hpp"

#if HAS_AOCL_BLAS
extern "C" void bli_thread_set_num_threads(int);
extern "C" int bli_thread_get_num_threads(void);
#endif

namespace score {

inline constexpr size_t streaming_posting_assignment_cell_bytes() {
    return sizeof(uint16_t);
}

inline void write_all_fd(int fd, const void* data, size_t bytes, const std::string& path);

inline void pwrite_all_fd(int fd, const void* data, size_t bytes, off_t offset, const std::string& path);

inline std::vector<float> collect_rotated_samples(const std::string& input_path,
                                                  const RandomRotation& rotation,
                                                  int sample_rows,
                                                  int block_rows,
                                                  std::unique_ptr<class SequentialFileWarmer>* warmer_out = nullptr);

inline bool stream_build_phase_trace_enabled() {
    const char* env = std::getenv("SCORE_STREAM_BUILD_PHASE_TRACE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

inline bool stream_build_assign_simd_compiled() {
#if HAS_AVX512
    return true;
#else
    return false;
#endif
}

inline int stream_build_assign_simd_max_subdim() {
    return 8;
}

inline bool stream_build_assign_points16_compiled() {
#if HAS_AVX512
    return true;
#else
    return false;
#endif
}

inline const char* stream_build_assign_backend_reason(bool compiled,
                                                      int sub_dim,
                                                      int max_subdim,
                                                      bool use_simd,
                                                      bool use_points16) {
    if (use_simd) return "enabled";
    if (use_points16) return "enabled_highdim_points16";
    if (!compiled) return "no_avx512";
    if (sub_dim > max_subdim) return "subdim_over_limit";
    return "disabled";
}

inline void log_stream_build_phase_time(const std::string& label, double elapsed_ms) {
    if (!stream_build_phase_trace_enabled()) return;
    std::cout << "[StreamBuild][Phase] " << label << " " << elapsed_ms << " ms" << std::endl;
}

inline void log_stream_assign_scratch_config(size_t total_budget,
                                             size_t per_worker_budget,
                                             int M,
                                             int subspace_workers,
                                             int K,
                                             int sub_dim,
                                             int block_rows,
                                             bool simd_assign_compiled,
                                             int simd_max_subdim,
                                             bool simd_assign,
                                             bool points16_assign) {
    if (!stream_build_phase_trace_enabled()) return;
    int assign_block_rows = impl::capped_assign_block_size(K, per_worker_budget);
    int sgemm_blocks_per_subspace = assign_block_rows > 0
        ? (block_rows + assign_block_rows - 1) / assign_block_rows
        : 0;
    size_t packed_bytes = sub_dim >= 32 ? (size_t)assign_block_rows * (size_t)sub_dim * sizeof(float) : 0;
    size_t assignments_bytes = (size_t)block_rows * sizeof(int);
    size_t estimated_per_worker_bytes = (size_t)assign_block_rows * (size_t)K * sizeof(float) +
                                        packed_bytes + assignments_bytes;
    std::cout << "[StreamBuild][AssignScratch] total_mib=" << (total_budget / (1024 * 1024))
              << " per_worker_mib=" << (per_worker_budget / (1024 * 1024))
              << " M=" << M
              << " workers=" << subspace_workers
              << " K=" << K
              << " sub_dim=" << sub_dim
              << " block_rows=" << block_rows
              << " assign_block_rows=" << assign_block_rows
              << " sgemm_blocks_per_subspace=" << sgemm_blocks_per_subspace
              << " estimated_per_worker_mib=" << (estimated_per_worker_bytes / (1024 * 1024))
              << " simd_compiled=" << (simd_assign_compiled ? 1 : 0)
              << " simd_max_subdim=" << simd_max_subdim
              << " assign_backend=" << (simd_assign ? "simd_soa" : (points16_assign ? "simd_points16" : "blas"))
              << " assign_backend_reason=" << stream_build_assign_backend_reason(
                     simd_assign_compiled, sub_dim, simd_max_subdim, simd_assign, points16_assign)
              << std::endl;
}

class ScopedThread {
public:
    ScopedThread() = default;
    explicit ScopedThread(std::thread&& thread) noexcept : thread_(std::move(thread)) {}
    ScopedThread(const ScopedThread&) = delete;
    ScopedThread& operator=(const ScopedThread&) = delete;
    ScopedThread(ScopedThread&&) = delete;
    ScopedThread& operator=(ScopedThread&&) = delete;
    ~ScopedThread() {
        if (thread_.joinable()) thread_.join();
    }

    bool joinable() const noexcept { return thread_.joinable(); }
    void join() {
        if (thread_.joinable()) thread_.join();
    }

private:
    std::thread thread_;
};

class ThreadGroup {
public:
    ThreadGroup() = default;
    ThreadGroup(const ThreadGroup&) = delete;
    ThreadGroup& operator=(const ThreadGroup&) = delete;
    ~ThreadGroup() { join_all(); }

    template<typename Fn>
    void emplace_back(Fn&& fn) {
        threads_.emplace_back(std::forward<Fn>(fn));
    }

    void reserve(size_t n) { threads_.reserve(n); }

    void join_all() noexcept {
        for (std::thread& thread : threads_) {
            if (thread.joinable()) thread.join();
        }
    }

private:
    std::vector<std::thread> threads_;
};

inline void write_exclusive_file(const std::string& path, const void* data, size_t bytes) {
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0) {
        throw std::runtime_error("cannot create exclusive file: " + path);
    }
    try {
        write_all_fd(fd, data, bytes, path);
    } catch (...) {
        ::close(fd);
        ::unlink(path.c_str());
        throw;
    }
    if (::close(fd) != 0) {
        ::unlink(path.c_str());
        throw std::runtime_error("failed closing exclusive file: " + path);
    }
}

inline void save_rotation_exclusive(const RandomRotation& rotation, const std::string& path) {
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0) {
        throw std::runtime_error("cannot create exclusive rotation output: " + path);
    }
    try {
        std::string fd_path = "/proc/self/fd/" + std::to_string(fd);
        std::ofstream out(fd_path, std::ios::binary);
        if (!out.is_open()) {
            throw std::runtime_error("cannot open exclusive rotation fd: " + path);
        }
        rotation.save(out);
        out.close();
        if (!out) {
            throw std::runtime_error("failed writing exclusive rotation output: " + path);
        }
    } catch (...) {
        ::close(fd);
        ::unlink(path.c_str());
        throw;
    }
    if (::close(fd) != 0) {
        ::unlink(path.c_str());
        throw std::runtime_error("failed closing exclusive rotation output: " + path);
    }
}

inline void write_rotate_subspace_flag_exclusive(const std::string& index_dir) {
    const char marker[] = "shared_fhtkac\n";
    write_exclusive_file(index_dir + "/rotate_subspace.flag", marker, sizeof(marker) - 1);
}

inline void cleanup_streaming_index_dir(const std::string& index_dir,
                                        const std::vector<std::string>& artifact_names) noexcept {
    for (const std::string& name : artifact_names) {
        std::string path = index_dir + "/" + name;
        if (::unlink(path.c_str()) != 0 && errno != ENOENT) {
            // Cleanup is best-effort; preserve the original exception/error path.
        }
    }
    if (::rmdir(index_dir.c_str()) != 0 && errno != ENOENT) {
        // Fresh output directories should be empty after artifact cleanup. Ignore
        // failures here so callers report the root cause instead of cleanup noise.
    }
}

inline void write_all_fd(int fd, const void* data, size_t bytes, const std::string& path) {
    const char* ptr = static_cast<const char*>(data);
    size_t written = 0;
    while (written < bytes) {
        ssize_t rc = ::write(fd, ptr + written, bytes - written);
        if (rc < 0 && errno == EINTR) continue;
        if (rc <= 0) {
            throw std::runtime_error("failed writing file: " + path);
        }
        written += (size_t)rc;
    }
}

inline void read_all_fd(int fd, void* data, size_t bytes, const std::string& path) {
    char* ptr = static_cast<char*>(data);
    size_t read_bytes = 0;
    while (read_bytes < bytes) {
        ssize_t rc = ::read(fd, ptr + read_bytes, bytes - read_bytes);
        if (rc < 0 && errno == EINTR) continue;
        if (rc <= 0) {
            throw std::runtime_error("failed reading file: " + path);
        }
        read_bytes += (size_t)rc;
    }
}

inline int open_exclusive_fd(const std::string& path) {
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0) {
        throw std::runtime_error("cannot create exclusive file: " + path);
    }
    return fd;
}

inline void pwrite_all_fd(int fd, const void* data, size_t bytes, off_t offset, const std::string& path) {
    const char* ptr = static_cast<const char*>(data);
    size_t written = 0;
    while (written < bytes) {
        ssize_t rc = ::pwrite(fd, ptr + written, bytes - written, offset + (off_t)written);
        if (rc < 0 && errno == EINTR) continue;
        if (rc <= 0) {
            throw std::runtime_error("failed positioned write file: " + path);
        }
        written += (size_t)rc;
    }
}

class VectorBlockReader {
public:
    explicit VectorBlockReader(const std::string& path) : path_(path) {
        auto [tag, n, dim] = detect_vector_format(path);
        if (tag < 0) {
            throw std::runtime_error("unsupported vector file format: " + path);
        }
        format_tag_ = tag;
        n_ = n;
        dim_ = dim;
        if (n_ <= 0 || dim_ < 4 || dim_ > 16384) {
            throw std::runtime_error("invalid vector file dimensions: " + path);
        }
        in_.open(path, std::ios::binary);
        if (!in_.is_open()) {
            throw std::runtime_error("cannot open vector file: " + path);
        }
        if (format_tag_ == 0) {
            in_.seekg(12, std::ios::beg);
        }
    }

    int n() const { return n_; }
    int dim() const { return dim_; }
    int format_tag() const { return format_tag_; }
    int rows_read() const { return rows_read_; }

    int read_block(int max_rows, std::vector<float>& out) {
        if (max_rows <= 0) {
            throw std::runtime_error("read_block max_rows must be positive");
        }
        int remaining = n_ - rows_read_;
        int rows = std::min(max_rows, remaining);
        if (rows > 0 && (size_t)rows > std::numeric_limits<size_t>::max() / (size_t)dim_) {
            throw std::runtime_error("vector block allocation would overflow: " + path_);
        }
        out.resize((size_t)rows * dim_);
        if (rows == 0) return 0;

        if (format_tag_ == 0) {
            in_.read(reinterpret_cast<char*>(out.data()), (size_t)rows * dim_ * sizeof(float));
            if (!in_) {
                throw std::runtime_error("short read from fbin vector block: " + path_);
            }
        } else {
            for (int i = 0; i < rows; ++i) {
                int row_dim = 0;
                in_.read(reinterpret_cast<char*>(&row_dim), sizeof(int));
                if (!in_ || row_dim != dim_) {
                    throw std::runtime_error("invalid fvecs row header in: " + path_);
                }
                in_.read(reinterpret_cast<char*>(out.data() + (size_t)i * dim_),
                         (size_t)dim_ * sizeof(float));
                if (!in_) {
                    throw std::runtime_error("short read from fvecs vector block: " + path_);
                }
            }
        }

        rows_read_ += rows;
        return rows;
    }

private:
    std::string path_;
    std::ifstream in_;
    int format_tag_ = -1;
    int n_ = 0;
    int dim_ = 0;
    int rows_read_ = 0;
};

class PrefetchedVectorBlockReader {
public:
    // Single-consumer prefetch reader. Returned Block::data remains valid until
    // the next next() call or reader destruction.
    struct Block {
        int rows = 0;
        const std::vector<float>* data = nullptr;
    };

    PrefetchedVectorBlockReader(const std::string& path, int max_rows)
        : reader_(path), max_rows_(max_rows) {
        if (max_rows_ <= 0) {
            throw std::runtime_error("prefetched reader max_rows must be positive");
        }
    }
    PrefetchedVectorBlockReader(const PrefetchedVectorBlockReader&) = delete;
    PrefetchedVectorBlockReader& operator=(const PrefetchedVectorBlockReader&) = delete;
    PrefetchedVectorBlockReader(PrefetchedVectorBlockReader&&) = delete;
    PrefetchedVectorBlockReader& operator=(PrefetchedVectorBlockReader&&) = delete;
    ~PrefetchedVectorBlockReader() { stop_worker(); }

    int n() const { return reader_.n(); }
    int dim() const { return reader_.dim(); }
    int format_tag() const { return reader_.format_tag(); }

    void release_buffers() noexcept {
        stop_worker();
        for (std::vector<float>& buffer : buffers_) {
            std::vector<float>().swap(buffer);
        }
    }

    void start() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (started_) {
            throw std::runtime_error("prefetched reader already started");
        }
        start_worker_locked();
    }

    Block next() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!started_) start_worker_locked();
        if (in_use_index_ >= 0) {
            states_[in_use_index_] = BufferState::Empty;
            in_use_index_ = -1;
            cv_.notify_all();
        }
        cv_.wait(lock, [this]() {
            return exception_ || ready_index_unlocked() >= 0 || eof_;
        });
        if (exception_) {
            std::rethrow_exception(exception_);
        }
        int ready = ready_index_unlocked();
        if (ready < 0) return {};
        states_[ready] = BufferState::InUse;
        in_use_index_ = ready;
        return {rows_[(size_t)ready], &buffers_[(size_t)ready]};
    }

private:
    enum class BufferState { Empty, Reading, Ready, InUse };

    int ready_index_unlocked() const {
        for (int i = 0; i < 2; ++i) {
            if (states_[i] == BufferState::Ready) return i;
        }
        return -1;
    }

    int empty_index_unlocked() const {
        for (int i = 0; i < 2; ++i) {
            if (states_[i] == BufferState::Empty) return i;
        }
        return -1;
    }

    void start_worker_locked() {
        started_ = true;
        try {
            worker_ = std::thread([this]() { worker_loop(); });
        } catch (...) {
            started_ = false;
            throw;
        }
        cv_.notify_all();
    }

    void worker_loop() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (true) {
            cv_.wait(lock, [this]() {
                return stop_ || exception_ || (!eof_ && empty_index_unlocked() >= 0);
            });
            if (stop_ || exception_) return;
            int buffer_index = empty_index_unlocked();
            if (buffer_index < 0) continue;
            states_[buffer_index] = BufferState::Reading;
            lock.unlock();

            int rows = 0;
            std::exception_ptr read_exception;
            try {
                rows = reader_.read_block(max_rows_, buffers_[(size_t)buffer_index]);
            } catch (...) {
                read_exception = std::current_exception();
            }

            lock.lock();
            if (read_exception) {
                states_[buffer_index] = BufferState::Empty;
                exception_ = read_exception;
                cv_.notify_all();
                return;
            }
            if (rows == 0) {
                states_[buffer_index] = BufferState::Empty;
                eof_ = true;
                cv_.notify_all();
                return;
            }
            rows_[(size_t)buffer_index] = rows;
            states_[buffer_index] = BufferState::Ready;
            cv_.notify_all();
        }
    }

    void stop_worker() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            cv_.notify_all();
        }
        if (worker_.joinable()) worker_.join();
    }

    VectorBlockReader reader_;
    int max_rows_ = 0;
    std::vector<float> buffers_[2];
    int rows_[2] = {0, 0};
    BufferState states_[2] = {BufferState::Empty, BufferState::Empty};
    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::exception_ptr exception_;
    int in_use_index_ = -1;
    bool started_ = false;
    bool stop_ = false;
    bool eof_ = false;
};

class SequentialFileWarmer {
public:
    SequentialFileWarmer(const std::string& path, off_t offset, size_t chunk_bytes)
        : path_(path), offset_(offset), chunk_bytes_(std::max<size_t>(1, chunk_bytes)) {
        worker_ = std::thread([this]() { warm_loop(); });
    }
    SequentialFileWarmer(const SequentialFileWarmer&) = delete;
    SequentialFileWarmer& operator=(const SequentialFileWarmer&) = delete;
    ~SequentialFileWarmer() { stop(); }

    void stop() noexcept {
        stop_.store(true, std::memory_order_relaxed);
        if (worker_.joinable()) worker_.join();
    }

private:
    void warm_loop() noexcept {
        int fd = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) return;
        if (::lseek(fd, offset_, SEEK_SET) < 0) {
            ::close(fd);
            return;
        }
#if defined(_POSIX_ADVISORY_INFO) || defined(POSIX_FADV_SEQUENTIAL)
        (void)::posix_fadvise(fd, offset_, 0, POSIX_FADV_SEQUENTIAL);
#endif
        std::vector<char> buffer(chunk_bytes_);
        while (!stop_.load(std::memory_order_relaxed)) {
            ssize_t rc = ::read(fd, buffer.data(), buffer.size());
            if (rc < 0 && errno == EINTR) continue;
            if (rc <= 0) break;
        }
        ::close(fd);
    }

    std::string path_;
    off_t offset_ = 0;
    size_t chunk_bytes_ = 0;
    std::atomic<bool> stop_{false};
    std::thread worker_;
};

inline void write_streaming_verify_vectors(const std::string& input_path,
                                           const std::string& verify_path,
                                           const RandomRotation& rotation,
                                           int block_rows) {
    PrefetchedVectorBlockReader reader(input_path, block_rows);
    if (reader.dim() != rotation.dim()) {
        throw std::runtime_error("input dimension does not match rotation dimension");
    }
    if (block_rows <= 0) {
        throw std::runtime_error("streaming verify block_rows must be positive");
    }
    if ((size_t)block_rows > std::numeric_limits<size_t>::max() / (size_t)std::max(reader.dim(), rotation.padded_dim())) {
        throw std::runtime_error("streaming verify block_rows would overflow allocation size");
    }

    int out_fd = ::open(verify_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (out_fd < 0) {
        throw std::runtime_error("cannot open streaming verify output: " + verify_path);
    }

    int n = reader.n();
    int pd = rotation.padded_dim();
    int reserved = 0;
    try {
        write_all_fd(out_fd, &n, sizeof(int), verify_path);
        write_all_fd(out_fd, &pd, sizeof(int), verify_path);
        write_all_fd(out_fd, &reserved, sizeof(int), verify_path);

        std::vector<float> rotated;
        reader.start();
        while (true) {
            auto input_block = reader.next();
            int rows = input_block.rows;
            if (rows == 0) break;
            const std::vector<float>& block = *input_block.data;
            if ((size_t)rows > std::numeric_limits<size_t>::max() / (size_t)pd) {
                throw std::runtime_error("streaming verify rotated block allocation would overflow");
            }
            rotated.assign((size_t)rows * pd, 0.0f);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < rows; ++i) {
                rotation.apply(block.data() + (size_t)i * reader.dim(),
                               rotated.data() + (size_t)i * pd);
            }
            write_all_fd(out_fd, rotated.data(), (size_t)rows * pd * sizeof(float), verify_path);
        }
    } catch (...) {
        ::close(out_fd);
        ::unlink(verify_path.c_str());
        throw;
    }
    if (::close(out_fd) != 0) {
        ::unlink(verify_path.c_str());
        throw std::runtime_error("failed closing streaming verify output: " + verify_path);
    }
}

inline std::vector<float> load_fbin_payload(const std::string& path, int& n, int& dim) {
    auto [tag, n_detected, dim_detected] = detect_vector_format(path);
    if (tag != 0) {
        throw std::runtime_error("expected fbin payload: " + path);
    }
    std::ifstream in(path, std::ios::binary);
    int reserved = 0;
    in.read(reinterpret_cast<char*>(&n), sizeof(int));
    in.read(reinterpret_cast<char*>(&dim), sizeof(int));
    in.read(reinterpret_cast<char*>(&reserved), sizeof(int));
    if (n != n_detected || dim != dim_detected || reserved != 0) {
        throw std::runtime_error("invalid fbin header: " + path);
    }
    std::vector<float> data((size_t)n * dim);
    in.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    if (!in) {
        throw std::runtime_error("short read from fbin payload: " + path);
    }
    return data;
}

struct StreamingRabitqOptions {
    int rabitq_k = 200;
    int rabitq_seed = 42;
    int block_rows = 65536;
    int sample_rows = 65536;
    int train_iters = 10;
};

inline int streaming_block_rows_for_mib(int dim, uint64_t block_mib) {
    if (dim <= 0) {
        throw std::runtime_error("streaming block_mib requires positive dimension");
    }
    if (block_mib == 0 || block_mib > (std::numeric_limits<size_t>::max() / (1024ull * 1024ull))) {
        throw std::runtime_error("streaming block_mib is invalid or too large");
    }
    const size_t block_bytes = (size_t)block_mib * 1024ull * 1024ull;
    const size_t row_bytes = (size_t)dim * sizeof(float);
    const size_t rows = block_bytes / row_bytes;
    if (rows == 0) {
        throw std::runtime_error("streaming block_mib is too small for one input row");
    }
    if (rows > (size_t)std::numeric_limits<int>::max()) {
        throw std::runtime_error("streaming block_mib produces too many rows");
    }
    return (int)rows;
}

struct StreamingIvfPostingOptions {
    int M = 16;
    int ivf_k = 2500;
    int block_rows = 65536;
    int sample_rows = 65536;
    int train_iters = 10;
    int subspace_workers = 0;  // 0 = use all M subspaces concurrently
    bool preserve_assignment_artifacts = false;
    bool materialize_no_reorder_postings = true;
};

struct StreamingReorderStagingArtifacts {
    int n = 0;
    int M = 0;
    int rabitq_k = 0;
    std::vector<size_t> cells_per_subspace;
    std::vector<std::string> subspace_assignment_paths;
    std::vector<std::string> ivf_paths;
    std::string verify_path;
    std::string centroids_path;
    std::string assignments_path;
    std::string bin_data_path;
};

inline std::string preserved_posting_assignment_manifest_name() {
    return "posting_assignments.manifest";
}

inline std::string preserved_posting_assignment_name(int m) {
    return "posting_assign_subspace_" + std::to_string(m) + ".assign.bin";
}

inline void save_ivf_index_exclusive(const std::string& path,
                                     const IVFIndex& ivf,
                                     const std::vector<int>& cell_sizes) {
    if ((int)cell_sizes.size() != ivf.K()) {
        throw std::runtime_error("IVF cell size vector length mismatch: " + path);
    }
    int sub_dim = ivf.subspace_dim();
    int K = ivf.K();
    int subspace_id = ivf.subspace_id();
    std::vector<size_t> size_t_cell_sizes((size_t)K);
    for (int k = 0; k < K; ++k) size_t_cell_sizes[(size_t)k] = (size_t)cell_sizes[(size_t)k];

    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0) {
        throw std::runtime_error("cannot create streaming IVF output: " + path);
    }
    try {
        write_all_fd(fd, &sub_dim, sizeof(int), path);
        write_all_fd(fd, &K, sizeof(int), path);
        write_all_fd(fd, &subspace_id, sizeof(int), path);
        write_all_fd(fd, ivf.centroids().data(), ivf.centroids().size() * sizeof(float), path);
        write_all_fd(fd, size_t_cell_sizes.data(), size_t_cell_sizes.size() * sizeof(size_t), path);
    } catch (...) {
        ::close(fd);
        ::unlink(path.c_str());
        throw;
    }
    if (::close(fd) != 0) {
        ::unlink(path.c_str());
        throw std::runtime_error("failed closing streaming IVF output: " + path);
    }
}

inline void write_posting_header_exclusive(const std::string& path,
                                           int K,
                                           int total,
                                           const std::vector<int>& offsets) {
    if ((int)offsets.size() != K + 1 || offsets.front() != 0 || offsets.back() != total) {
        throw std::runtime_error("invalid posting offsets for: " + path);
    }
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0) {
        throw std::runtime_error("cannot create streaming posting output: " + path);
    }
    try {
        write_all_fd(fd, &K, sizeof(int), path);
        write_all_fd(fd, &total, sizeof(int), path);
        write_all_fd(fd, offsets.data(), (size_t)(K + 1) * sizeof(int), path);
        off_t file_bytes = (off_t)(2 * sizeof(int) + (size_t)(K + 1) * sizeof(int) + (size_t)total * sizeof(int));
        if (::ftruncate(fd, file_bytes) != 0) {
            throw std::runtime_error("failed sizing streaming posting output: " + path);
        }
    } catch (...) {
        ::close(fd);
        ::unlink(path.c_str());
        throw;
    }
    if (::close(fd) != 0) {
        ::unlink(path.c_str());
        throw std::runtime_error("failed closing streaming posting header: " + path);
    }
}

inline int create_posting_output_exclusive(const std::string& path,
                                           int K,
                                           int total,
                                           const std::vector<int>& offsets) {
    if ((int)offsets.size() != K + 1 || offsets.front() != 0 || offsets.back() != total) {
        throw std::runtime_error("invalid posting offsets for: " + path);
    }
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0) {
        throw std::runtime_error("cannot create streaming posting output: " + path);
    }
    try {
        write_all_fd(fd, &K, sizeof(int), path);
        write_all_fd(fd, &total, sizeof(int), path);
        write_all_fd(fd, offsets.data(), (size_t)(K + 1) * sizeof(int), path);
        off_t file_bytes = (off_t)(2 * sizeof(int) + (size_t)(K + 1) * sizeof(int) + (size_t)total * sizeof(int));
        if (::ftruncate(fd, file_bytes) != 0) {
            throw std::runtime_error("failed sizing streaming posting output: " + path);
        }
    } catch (...) {
        ::close(fd);
        ::unlink(path.c_str());
        throw;
    }
    return fd;
}

inline std::vector<std::string> streaming_ivf_posting_artifact_names(int M) {
    std::vector<std::string> names;
    names.reserve((size_t)M * 2 + 1);
    names.push_back("rotation.bin");
    names.push_back("rotate_subspace.flag");
    for (int m = 0; m < M; ++m) {
        names.push_back("ivf_subspace_" + std::to_string(m) + ".bin");
        names.push_back("posting_subspace_" + std::to_string(m) + ".bin");
        names.push_back(preserved_posting_assignment_name(m));
    }
    names.push_back(preserved_posting_assignment_manifest_name());
    return names;
}

inline std::vector<std::string> streaming_no_reorder_artifact_names(int M) {
    std::vector<std::string> names = {
        "rotation.bin",
        "rotate_subspace.flag",
        "verify_vectors.fbin",
        "rabitq_centroids.bin",
        "rabitq_assignments.bin",
        "rabitq_bin_data.bin",
    };
    for (int m = 0; m < M; ++m) {
        names.push_back("ivf_subspace_" + std::to_string(m) + ".bin");
        names.push_back("posting_subspace_" + std::to_string(m) + ".bin");
        names.push_back(preserved_posting_assignment_name(m));
    }
    names.push_back(preserved_posting_assignment_manifest_name());
    return names;
}

inline void extract_subspace_block(const std::vector<float>& rotated,
                                   int rows,
                                   int padded_dim,
                                   int sub_dim,
                                   int subspace,
                                   std::vector<float>& sub_block) {
    sub_block.resize((size_t)rows * sub_dim);
    const int offset = subspace * sub_dim;
    for (int i = 0; i < rows; ++i) {
        std::memcpy(sub_block.data() + (size_t)i * sub_dim,
                    rotated.data() + (size_t)i * padded_dim + offset,
                    (size_t)sub_dim * sizeof(float));
    }
}

inline void write_streaming_ivf_posting_artifacts(const std::string& input_path,
                                                  const std::string& index_dir,
                                                  const RandomRotation& rotation,
                                                  const StreamingIvfPostingOptions& options) {
    if (options.M <= 0 || options.ivf_k <= 0 || options.block_rows <= 0 ||
        options.sample_rows <= 0 || options.train_iters <= 0) {
        throw std::runtime_error("streaming IVF/posting options must be positive");
    }

    PrefetchedVectorBlockReader reader(input_path, options.block_rows);
    int n = reader.n();
    int dim = reader.dim();
    int padded_dim = rotation.padded_dim();
    int M = options.M;
    if (dim != rotation.dim()) {
        throw std::runtime_error("streaming IVF input dimension does not match rotation dimension");
    }
    if (dim % M != 0) {
        throw std::runtime_error("streaming IVF requires dim divisible by M");
    }
    if (n > std::numeric_limits<int>::max()) {
        throw std::runtime_error("streaming postings require int32 total count");
    }
    int sample_n = std::min(n, options.sample_rows);
    if (options.ivf_k > sample_n) {
        throw std::runtime_error("streaming IVF sample_rows must be >= ivf_k");
    }
    if (options.ivf_k > (int)std::numeric_limits<uint16_t>::max()) {
        throw std::runtime_error("streaming IVF ivf_k exceeds uint16 assignment limit");
    }
    if ((size_t)options.block_rows > std::numeric_limits<size_t>::max() / (size_t)std::max(dim, padded_dim)) {
        throw std::runtime_error("streaming IVF block_rows would overflow allocation size");
    }
    int sub_dim = dim / M;
    std::vector<float> samples = collect_rotated_samples(input_path, rotation, sample_n, options.block_rows);
    std::vector<IVFIndex> ivf_indices((size_t)M);
    std::vector<std::vector<int>> cell_sizes((size_t)M, std::vector<int>((size_t)options.ivf_k, 0));
    std::vector<float> sub_samples((size_t)sample_n * sub_dim);

    for (int m = 0; m < M; ++m) {
        for (int i = 0; i < sample_n; ++i) {
            std::memcpy(sub_samples.data() + (size_t)i * sub_dim,
                        samples.data() + (size_t)i * dim + (size_t)m * sub_dim,
                        (size_t)sub_dim * sizeof(float));
        }
        ivf_indices[(size_t)m].init(sub_dim, options.ivf_k, m);
        ivf_indices[(size_t)m].train(sub_samples.data(), sample_n, options.train_iters, (uint64_t)(42 + m));
    }

    std::vector<float> rotated;
    std::vector<float> sub_block;
    std::vector<int> assignments((size_t)options.block_rows);

    reader.start();
    while (true) {
        auto input_block = reader.next();
        int rows = input_block.rows;
        if (rows == 0) break;
        const std::vector<float>& block = *input_block.data;
        rotated.assign((size_t)rows * padded_dim, 0.0f);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < rows; ++i) {
            rotation.apply(block.data() + (size_t)i * dim,
                           rotated.data() + (size_t)i * padded_dim);
        }
        if ((int)assignments.size() < rows) assignments.resize((size_t)rows);
        for (int m = 0; m < M; ++m) {
            extract_subspace_block(rotated, rows, padded_dim, sub_dim, m, sub_block);
            ivf_indices[(size_t)m].assign_batch(sub_block.data(), assignments.data(), rows);
            std::vector<int>& sizes = cell_sizes[(size_t)m];
            for (int i = 0; i < rows; ++i) {
                int cell = assignments[(size_t)i];
                if (cell < 0 || cell >= options.ivf_k) {
                    throw std::runtime_error("streaming IVF assignment out of range");
                }
                ++sizes[(size_t)cell];
            }
        }
    }

    std::vector<std::vector<int>> offsets((size_t)M);
    std::vector<int> posting_fds((size_t)M, -1);
    std::vector<std::string> posting_paths((size_t)M);
    std::vector<std::string> created_paths;
    created_paths.reserve((size_t)M * 2);

    std::vector<std::vector<int>> cursors;
    std::vector<int> block_counts((size_t)options.ivf_k);
    std::vector<int> block_offsets((size_t)options.ivf_k + 1);
    std::vector<int> block_cursors((size_t)options.ivf_k);
    std::vector<uint32_t> ids_chunk((size_t)options.block_rows);
    try {
        for (int m = 0; m < M; ++m) {
            offsets[(size_t)m].assign((size_t)options.ivf_k + 1, 0);
            for (int k = 0; k < options.ivf_k; ++k) {
                offsets[(size_t)m][(size_t)k + 1] = offsets[(size_t)m][(size_t)k] + cell_sizes[(size_t)m][(size_t)k];
            }
            if (offsets[(size_t)m].back() != n) {
                throw std::runtime_error("streaming IVF posting counts do not sum to n");
            }
            std::string ivf_path = index_dir + "/ivf_subspace_" + std::to_string(m) + ".bin";
            save_ivf_index_exclusive(ivf_path, ivf_indices[(size_t)m], cell_sizes[(size_t)m]);
            created_paths.push_back(ivf_path);
            posting_paths[(size_t)m] = index_dir + "/posting_subspace_" + std::to_string(m) + ".bin";
            posting_fds[(size_t)m] = create_posting_output_exclusive(posting_paths[(size_t)m],
                                                                     options.ivf_k, n, offsets[(size_t)m]);
            created_paths.push_back(posting_paths[(size_t)m]);
        }
        cursors = offsets;

        PrefetchedVectorBlockReader fill_reader(input_path, options.block_rows);
        fill_reader.start();
        int global_start = 0;
        while (true) {
            auto input_block = fill_reader.next();
            int rows = input_block.rows;
            if (rows == 0) break;
            const std::vector<float>& block = *input_block.data;
            rotated.assign((size_t)rows * padded_dim, 0.0f);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < rows; ++i) {
                rotation.apply(block.data() + (size_t)i * dim,
                               rotated.data() + (size_t)i * padded_dim);
            }
            if ((int)assignments.size() < rows) assignments.resize((size_t)rows);
            if ((int)ids_chunk.size() < rows) ids_chunk.resize((size_t)rows);

            for (int m = 0; m < M; ++m) {
                extract_subspace_block(rotated, rows, padded_dim, sub_dim, m, sub_block);
                ivf_indices[(size_t)m].assign_batch(sub_block.data(), assignments.data(), rows);
                std::fill(block_counts.begin(), block_counts.end(), 0);
                for (int i = 0; i < rows; ++i) {
                    int cell = assignments[(size_t)i];
                    if (cell < 0 || cell >= options.ivf_k) {
                        throw std::runtime_error("streaming IVF assignment out of range during posting fill");
                    }
                    ++block_counts[(size_t)cell];
                }
                block_offsets[0] = 0;
                for (int k = 0; k < options.ivf_k; ++k) block_offsets[(size_t)k + 1] = block_offsets[(size_t)k] + block_counts[(size_t)k];
                std::fill(block_cursors.begin(), block_cursors.end(), 0);
                for (int i = 0; i < rows; ++i) {
                    int cell = assignments[(size_t)i];
                    ids_chunk[(size_t)block_offsets[(size_t)cell] + (size_t)block_cursors[(size_t)cell]++] = (uint32_t)(global_start + i);
                }

                off_t ids_base = (off_t)(2 * sizeof(int) + (size_t)(options.ivf_k + 1) * sizeof(int));
                for (int k = 0; k < options.ivf_k; ++k) {
                    int count = block_counts[(size_t)k];
                    if (count == 0) continue;
                    int dst = cursors[(size_t)m][(size_t)k];
                    pwrite_all_fd(posting_fds[(size_t)m],
                                  ids_chunk.data() + (size_t)block_offsets[(size_t)k],
                                  (size_t)count * sizeof(uint32_t),
                                  ids_base + (off_t)dst * (off_t)sizeof(uint32_t),
                                  posting_paths[(size_t)m]);
                    cursors[(size_t)m][(size_t)k] += count;
                }
            }
            global_start += rows;
        }
    } catch (...) {
        for (int fd : posting_fds) {
            if (fd >= 0) ::close(fd);
        }
        for (const std::string& path : created_paths) {
            ::unlink(path.c_str());
        }
        throw;
    }

    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < options.ivf_k; ++k) {
            if (cursors[(size_t)m][(size_t)k] != offsets[(size_t)m][(size_t)k + 1]) {
                for (int fd : posting_fds) {
                    if (fd >= 0) ::close(fd);
                }
                for (const std::string& path : created_paths) {
                    ::unlink(path.c_str());
                }
                throw std::runtime_error("streaming posting fill count mismatch");
            }
        }
        if (::close(posting_fds[(size_t)m]) != 0) {
            for (const std::string& path : created_paths) {
                ::unlink(path.c_str());
            }
            throw std::runtime_error("failed closing streaming posting output: " + posting_paths[(size_t)m]);
        }
        posting_fds[(size_t)m] = -1;
    }
}

inline void save_rabitq_centroids(const std::string& path,
                                  const std::vector<float>& rot_centroids,
                                  int k,
                                  int dim,
                                  int padded_dim) {
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0) {
        throw std::runtime_error("cannot create rabitq centroid output: " + path);
    }
    try {
        write_all_fd(fd, &k, sizeof(int), path);
        write_all_fd(fd, &dim, sizeof(int), path);
        write_all_fd(fd, &padded_dim, sizeof(int), path);
        write_all_fd(fd, rot_centroids.data(), (size_t)k * padded_dim * sizeof(float), path);
    } catch (...) {
        ::close(fd);
        ::unlink(path.c_str());
        throw;
    }
    if (::close(fd) != 0) {
        ::unlink(path.c_str());
        throw std::runtime_error("failed closing rabitq centroid output: " + path);
    }
}

inline std::vector<float> collect_rotated_samples(const std::string& input_path,
                                                   const RandomRotation& rotation,
                                                   int sample_rows,
                                                   int block_rows,
                                                   std::unique_ptr<SequentialFileWarmer>* warmer_out) {
    VectorBlockReader reader(input_path);
    if (reader.dim() != rotation.dim()) {
        throw std::runtime_error("sample collection dimension does not match rotation dimension");
    }
    int take = std::min(reader.n(), sample_rows);
    if (take <= 0) {
        throw std::runtime_error("streaming RaBitQ sample_rows must be positive");
    }
    std::vector<float> samples((size_t)take * rotation.dim());

    if (reader.format_tag() == 0) {
        auto target_for_selected = [&](int selected) {
            return (int)(((int64_t)selected * reader.n()) / take);
        };

        if (warmer_out != nullptr) {
            constexpr size_t warm_chunk_bytes = 8ull * 1024ull * 1024ull;
            *warmer_out = std::make_unique<SequentialFileWarmer>(input_path, 12, warm_chunk_bytes);

            struct SampleInputFd {
                int fd = -1;
                explicit SampleInputFd(int opened_fd) : fd(opened_fd) {}
                SampleInputFd(const SampleInputFd&) = delete;
                SampleInputFd& operator=(const SampleInputFd&) = delete;
                ~SampleInputFd() {
                    if (fd >= 0) ::close(fd);
                }
                void close_checked(const std::string& path) {
                    if (fd < 0) return;
                    int old_fd = fd;
                    fd = -1;
                    if (::close(old_fd) != 0) {
                        throw std::runtime_error("failed closing fbin parallel sample input: " + path);
                    }
                }
            } sample_fd(::open(input_path.c_str(), O_RDONLY | O_CLOEXEC));
            if (sample_fd.fd < 0) {
                warmer_out->reset();
                throw std::runtime_error("cannot open fbin for parallel sample collection: " + input_path);
            }
            std::atomic<int> next_selected{0};
            std::atomic<bool> stop_workers{false};
            int worker_count = std::max(1, std::min(8, omp_get_max_threads()));
            std::vector<std::exception_ptr> worker_exceptions((size_t)worker_count);
            auto pread_all = [&](void* data, size_t bytes, off_t offset) {
                char* ptr = static_cast<char*>(data);
                size_t read_bytes = 0;
                while (read_bytes < bytes) {
                    ssize_t rc = ::pread(sample_fd.fd, ptr + read_bytes, bytes - read_bytes, offset + (off_t)read_bytes);
                    if (rc < 0 && errno == EINTR) continue;
                    if (rc <= 0) {
                        throw std::runtime_error("short fbin parallel sample row read from: " + input_path);
                    }
                    read_bytes += (size_t)rc;
                }
            };
            ThreadGroup workers;
            workers.reserve((size_t)worker_count);
            for (int worker = 0; worker < worker_count; ++worker) {
                workers.emplace_back([&, worker]() {
                    std::vector<float> row((size_t)reader.dim());
                    std::vector<float> tmp((size_t)rotation.padded_dim(), 0.0f);
                    try {
                        while (!stop_workers.load(std::memory_order_relaxed)) {
                            int selected = next_selected.fetch_add(1, std::memory_order_relaxed);
                            if (selected >= take) break;
                            int target = target_for_selected(selected);
                            off_t offset = (off_t)12 + (off_t)target * (off_t)reader.dim() * (off_t)sizeof(float);
                            pread_all(row.data(), (size_t)reader.dim() * sizeof(float), offset);
                            std::fill(tmp.begin(), tmp.end(), 0.0f);
                            rotation.apply(row.data(), tmp.data());
                            std::memcpy(samples.data() + (size_t)selected * (size_t)rotation.dim(),
                                        tmp.data(), (size_t)rotation.dim() * sizeof(float));
                        }
                    } catch (...) {
                        worker_exceptions[(size_t)worker] = std::current_exception();
                        stop_workers.store(true, std::memory_order_relaxed);
                    }
                });
            }
            workers.join_all();
            std::exception_ptr worker_exception;
            for (const std::exception_ptr& ex : worker_exceptions) {
                if (ex) {
                    worker_exception = ex;
                    break;
                }
            }
            if (worker_exception) {
                warmer_out->reset();
                std::rethrow_exception(worker_exception);
            }
            sample_fd.close_checked(input_path);
            return samples;
        }

        std::ifstream in(input_path, std::ios::binary);
        if (!in.is_open()) {
            throw std::runtime_error("cannot open fbin for sequential sample collection: " + input_path);
        }
        in.seekg(12, std::ios::beg);
        if (!in) {
            throw std::runtime_error("failed seeking fbin payload for sample collection: " + input_path);
        }
        constexpr size_t max_sample_read_bytes = 32ull * 1024ull * 1024ull;
        size_t row_bytes = (size_t)reader.dim() * sizeof(float);
        int max_sample_read_rows = (int)std::max<size_t>(1, max_sample_read_bytes / std::max<size_t>(1, row_bytes));
        int rows_per_block = std::max(1, std::min(block_rows, max_sample_read_rows));
        std::vector<float> block((size_t)rows_per_block * (size_t)reader.dim());
        std::vector<float> tmp((size_t)rotation.padded_dim(), 0.0f);
        int selected = 0;
        for (int block_start = 0; block_start < reader.n() && selected < take; block_start += rows_per_block) {
            int rows = std::min(rows_per_block, reader.n() - block_start);
            in.read(reinterpret_cast<char*>(block.data()), (size_t)rows * (size_t)reader.dim() * sizeof(float));
            if (!in) {
                throw std::runtime_error("short fbin sample block read from: " + input_path);
            }
            while (selected < take) {
                int target = target_for_selected(selected);
                if (target < block_start) {
                    throw std::runtime_error("sample target order invariant violated");
                }
                if (target >= block_start + rows) break;
                const float* row = block.data() + (size_t)(target - block_start) * (size_t)reader.dim();
                std::fill(tmp.begin(), tmp.end(), 0.0f);
                rotation.apply(row, tmp.data());
                std::memcpy(samples.data() + (size_t)selected * (size_t)rotation.dim(),
                            tmp.data(), (size_t)rotation.dim() * sizeof(float));
                ++selected;
            }
        }
        if (selected != take) {
            throw std::runtime_error("fbin sample collection did not collect all targets from: " + input_path);
        }
        return samples;
    }

    std::ifstream in(input_path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("cannot open vector file for direct sample collection: " + input_path);
    }
    std::vector<float> row((size_t)reader.dim());
    std::vector<float> tmp((size_t)rotation.padded_dim(), 0.0f);
    for (int selected = 0; selected < take; ++selected) {
        int target = (int)(((int64_t)selected * reader.n()) / take);
        std::streamoff offset = 0;
        if (reader.format_tag() == 0) {
            offset = (std::streamoff)12 + (std::streamoff)target * reader.dim() * (std::streamoff)sizeof(float);
        } else {
            offset = (std::streamoff)target * ((std::streamoff)sizeof(int) + reader.dim() * (std::streamoff)sizeof(float));
        }
        in.seekg(offset, std::ios::beg);
        if (!in) {
            throw std::runtime_error("failed seeking sample row in: " + input_path);
        }
        if (reader.format_tag() == 1) {
            int row_dim = 0;
            in.read(reinterpret_cast<char*>(&row_dim), sizeof(int));
            if (!in || row_dim != reader.dim()) {
                throw std::runtime_error("invalid fvecs sample row header in: " + input_path);
            }
        }
        in.read(reinterpret_cast<char*>(row.data()), (size_t)reader.dim() * sizeof(float));
        if (!in) {
            throw std::runtime_error("short sample row read from: " + input_path);
        }
        std::fill(tmp.begin(), tmp.end(), 0.0f);
        rotation.apply(row.data(), tmp.data());
        std::memcpy(samples.data() + (size_t)selected * rotation.dim(),
                    tmp.data(), (size_t)rotation.dim() * sizeof(float));
    }
    return samples;
}

inline StreamingReorderStagingArtifacts write_streaming_no_reorder_fused_artifacts(
    const std::string& input_path,
    const std::string& index_dir,
    const RandomRotation& rotation,
    const StreamingRabitqOptions& rabitq_options,
    const StreamingIvfPostingOptions& ivf_options) {
    if (rabitq_options.block_rows != ivf_options.block_rows ||
        rabitq_options.sample_rows != ivf_options.sample_rows ||
        rabitq_options.train_iters != ivf_options.train_iters) {
        throw std::runtime_error("fused streaming builder requires shared block/sample/train options");
    }
    if (rabitq_options.rabitq_k <= 0 || ivf_options.M <= 0 || ivf_options.ivf_k <= 0 ||
        rabitq_options.block_rows <= 0 || rabitq_options.sample_rows <= 0 || rabitq_options.train_iters <= 0) {
        throw std::runtime_error("fused streaming builder options must be positive");
    }

    VectorBlockReader meta(input_path);
    int n = meta.n();
    int dim = meta.dim();
    int padded_dim = rotation.padded_dim();
    int M = ivf_options.M;
    int block_rows = rabitq_options.block_rows;
    if (meta.format_tag() != 0) {
        throw std::runtime_error("fused streaming no-reorder builder only supports fbin input");
    }
    if (dim != rotation.dim()) {
        throw std::runtime_error("fused streaming input dimension does not match rotation dimension");
    }
    if (dim % M != 0) {
        throw std::runtime_error("fused streaming IVF requires dim divisible by M");
    }
    if (n > std::numeric_limits<int>::max()) {
        throw std::runtime_error("fused streaming postings require int32 total count");
    }
    int sample_n = std::min(n, rabitq_options.sample_rows);
    if (rabitq_options.rabitq_k > sample_n || ivf_options.ivf_k > sample_n) {
        throw std::runtime_error("fused streaming sample_rows must cover all centroid counts");
    }
    if (ivf_options.ivf_k > (int)std::numeric_limits<uint16_t>::max()) {
        throw std::runtime_error("fused streaming IVF ivf_k exceeds uint16 assignment limit");
    }
    if ((size_t)block_rows > std::numeric_limits<size_t>::max() / (size_t)std::max(dim, padded_dim)) {
        throw std::runtime_error("fused streaming block_rows would overflow allocation size");
    }
    const size_t total_assign_scratch_budget = 256ull * 1024ull * 1024ull;
    const size_t min_assign_scratch_per_worker = 16ull * 1024ull * 1024ull;
    int subspace_workers = ivf_options.subspace_workers <= 0 ? M : std::min(M, ivf_options.subspace_workers);
    const size_t assign_scratch_per_worker = std::max(
        min_assign_scratch_per_worker,
        total_assign_scratch_budget / (size_t)subspace_workers);

    Timer phase_timer;
    int sub_dim = dim / M;
    bool simd_assign_compiled = stream_build_assign_simd_compiled();
    int simd_assign_max_subdim = stream_build_assign_simd_max_subdim();
    bool use_simd_assign = simd_assign_compiled && sub_dim <= simd_assign_max_subdim;
    bool use_points16_assign = stream_build_assign_points16_compiled() &&
                               !use_simd_assign && sub_dim >= 32 && sub_dim <= 64;
    log_stream_assign_scratch_config(total_assign_scratch_budget,
                                     assign_scratch_per_worker,
                                     M,
                                     subspace_workers,
                                     ivf_options.ivf_k,
                                     sub_dim,
                                     block_rows,
                                     simd_assign_compiled,
                                     simd_assign_max_subdim,
                                     use_simd_assign,
                                     use_points16_assign);
    phase_timer.reset();
    std::unique_ptr<SequentialFileWarmer> input_warmer;
    std::vector<float> samples = collect_rotated_samples(input_path, rotation, sample_n, block_rows, &input_warmer);
    log_stream_build_phase_time("collect_samples", phase_timer.elapsed_ms());

    IVFIndex top_ivf;
    top_ivf.init(dim, rabitq_options.rabitq_k, 0);
    phase_timer.reset();
    top_ivf.train(samples.data(), sample_n, rabitq_options.train_iters, (uint64_t)rabitq_options.rabitq_seed);
    log_stream_build_phase_time("train_rabitq_top", phase_timer.elapsed_ms());

    std::vector<float> rot_centroids((size_t)rabitq_options.rabitq_k * padded_dim, 0.0f);
    const std::vector<float>& centroids = top_ivf.centroids();
    for (int k = 0; k < rabitq_options.rabitq_k; ++k) {
        std::memcpy(rot_centroids.data() + (size_t)k * padded_dim,
                    centroids.data() + (size_t)k * dim,
                    (size_t)dim * sizeof(float));
    }

    std::vector<IVFIndex> ivf_indices((size_t)M);
    std::vector<std::vector<int>> cell_sizes((size_t)M, std::vector<int>((size_t)ivf_options.ivf_k, 0));
    std::vector<float> sub_samples((size_t)sample_n * sub_dim);
    phase_timer.reset();
    for (int m = 0; m < M; ++m) {
        for (int i = 0; i < sample_n; ++i) {
            std::memcpy(sub_samples.data() + (size_t)i * sub_dim,
                        samples.data() + (size_t)i * dim + (size_t)m * sub_dim,
                        (size_t)sub_dim * sizeof(float));
        }
        ivf_indices[(size_t)m].init(sub_dim, ivf_options.ivf_k, m);
        ivf_indices[(size_t)m].train(sub_samples.data(), sample_n, ivf_options.train_iters, (uint64_t)(42 + m));
        if (use_simd_assign) ivf_indices[(size_t)m].prepare_simd_centroids();
    }
    log_stream_build_phase_time("train_ivf_subspaces", phase_timer.elapsed_ms());
    // Training samples are not needed during the fused encode/write pass.
    // Release them before allocating the large per-block input/rotated/BLAS buffers.
    std::vector<float>().swap(sub_samples);
    std::vector<float>().swap(samples);

    const std::string verify_path = index_dir + "/verify_vectors.fbin";
    const std::string rotation_path = index_dir + "/rotation.bin";
    const std::string rotate_flag_path = index_dir + "/rotate_subspace.flag";
    const std::string centroids_path = index_dir + "/rabitq_centroids.bin";
    const std::string assignments_path = index_dir + "/rabitq_assignments.bin";
    const std::string bin_data_path = index_dir + "/rabitq_bin_data.bin";
    std::vector<std::string> created_paths;
    std::vector<std::string> temp_paths;
    std::vector<std::string> preserved_assignment_paths((size_t)M);
    StreamingReorderStagingArtifacts staging_artifacts;
    staging_artifacts.n = n;
    staging_artifacts.M = M;
    staging_artifacts.rabitq_k = rabitq_options.rabitq_k;
    staging_artifacts.verify_path = verify_path;
    staging_artifacts.centroids_path = centroids_path;
    staging_artifacts.assignments_path = assignments_path;
    staging_artifacts.bin_data_path = bin_data_path;
    std::vector<int> temp_write_fds((size_t)M, -1);
    std::vector<int> temp_read_fds((size_t)M, -1);
    std::vector<int> posting_fds((size_t)M, -1);
    std::vector<std::string> posting_paths((size_t)M);
    int verify_fd = -1;
    int assign_fd = -1;
    int bin_fd = -1;
#if HAS_AOCL_BLAS
    int saved_blis_threads = bli_thread_get_num_threads();
    bool blis_threads_changed = false;
#endif

    auto close_fd_if_open = [](int& fd) noexcept {
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
    };
    auto close_fd_checked = [](int& fd, const std::string& path) {
        if (fd >= 0) {
            if (::close(fd) != 0) {
                fd = -1;
                throw std::runtime_error("failed closing file: " + path);
            }
            fd = -1;
        }
    };
    auto cleanup_paths = [&]() noexcept {
        for (const std::string& path : temp_paths) ::unlink(path.c_str());
        for (const std::string& path : created_paths) ::unlink(path.c_str());
    };

    try {
#if HAS_AOCL_BLAS
        int total_cores = std::max(1, omp_get_max_threads());
        // BLIS thread count is process-global in AOCL. Set it once before the
        // concurrent fused phase and never mutate it inside worker threads.
        int concurrent_blas_consumers = subspace_workers + 1; // IVF workers + top-level RaBitQ assigner
        int blis_threads_per_subspace = std::max(1, total_cores / concurrent_blas_consumers);
        bli_thread_set_num_threads(blis_threads_per_subspace);
        blis_threads_changed = true;
#endif
        if (access(rotation_path.c_str(), F_OK) != 0) {
            save_rotation_exclusive(rotation, rotation_path);
            created_paths.push_back(rotation_path);
        }
        if (access(rotate_flag_path.c_str(), F_OK) != 0) {
            write_rotate_subspace_flag_exclusive(index_dir);
            created_paths.push_back(rotate_flag_path);
        }

        save_rabitq_centroids(centroids_path, rot_centroids, rabitq_options.rabitq_k, dim, padded_dim);
        created_paths.push_back(centroids_path);

        verify_fd = open_exclusive_fd(verify_path);
        created_paths.push_back(verify_path);
        assign_fd = open_exclusive_fd(assignments_path);
        created_paths.push_back(assignments_path);
        bin_fd = open_exclusive_fd(bin_data_path);
        created_paths.push_back(bin_data_path);
        for (int m = 0; m < M; ++m) {
            std::string temp_path = index_dir + "/posting_assign_subspace_" + std::to_string(m) + ".tmp";
            temp_write_fds[(size_t)m] = open_exclusive_fd(temp_path);
            temp_paths.push_back(temp_path);
        }

        int reserved = 0;
        write_all_fd(verify_fd, &n, sizeof(int), verify_path);
        write_all_fd(verify_fd, &padded_dim, sizeof(int), verify_path);
        write_all_fd(verify_fd, &reserved, sizeof(int), verify_path);
        write_all_fd(assign_fd, &n, sizeof(int), assignments_path);
        size_t bin_bytes = rabitqlib::BinDataMap<float>::data_bytes(padded_dim);
        int bb = (int)(sizeof(uint32_t) + bin_bytes);
        write_all_fd(bin_fd, &n, sizeof(int), bin_data_path);
        write_all_fd(bin_fd, &bb, sizeof(int), bin_data_path);

        PrefetchedVectorBlockReader reader(input_path, block_rows);
        std::vector<float> rotated((size_t)block_rows * (size_t)padded_dim);
        std::vector<float> rotated_dim;
        if (padded_dim != dim) rotated_dim.resize((size_t)block_rows * (size_t)dim);
        std::vector<float> sub_block;
        std::vector<int> top_assignments(block_rows);
        std::vector<uint32_t> rabitq_assignments(block_rows);
        std::vector<std::vector<int>> subspace_assignments(
            (size_t)M, std::vector<int>((size_t)block_rows));
        std::vector<std::vector<uint16_t>> subspace_assignments_u16(
            (size_t)M, std::vector<uint16_t>((size_t)block_rows));
        std::vector<char> bin_chunk((size_t)block_rows * (size_t)bb);
        double phase_rotate_write_ms = 0.0;
        double phase_top_assign_ms = 0.0;
        double phase_quantize_write_ms = 0.0;
        double phase_subspace_assign_ms = 0.0;
        Timer sub_phase_timer;

        phase_timer.reset();
        reader.start();
        while (true) {
            auto input_block = reader.next();
            int rows = input_block.rows;
            if (rows == 0) break;
            const std::vector<float>& block = *input_block.data;
            if (rotated.size() < (size_t)rows * (size_t)padded_dim) rotated.resize((size_t)rows * (size_t)padded_dim);
            if (padded_dim != dim && rotated_dim.size() < (size_t)rows * (size_t)dim) rotated_dim.resize((size_t)rows * (size_t)dim);
            if ((int)top_assignments.size() < rows) top_assignments.resize(rows);
            if ((int)rabitq_assignments.size() < rows) rabitq_assignments.resize(rows);
            if (bin_chunk.size() < (size_t)rows * (size_t)bb) bin_chunk.resize((size_t)rows * (size_t)bb);

            sub_phase_timer.reset();
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < rows; ++i) {
                rotation.apply(block.data() + (size_t)i * dim,
                               rotated.data() + (size_t)i * padded_dim);
                if (padded_dim != dim) {
                    std::memcpy(rotated_dim.data() + (size_t)i * dim,
                                rotated.data() + (size_t)i * padded_dim,
                                (size_t)dim * sizeof(float));
                }
            }
            std::exception_ptr verify_write_exception;
            ScopedThread verify_write_thread(std::thread([&]() {
                try {
                    write_all_fd(verify_fd, rotated.data(), (size_t)rows * padded_dim * sizeof(float), verify_path);
                } catch (...) {
                    verify_write_exception = std::current_exception();
                }
            }));
            phase_rotate_write_ms += sub_phase_timer.elapsed_ms();

            std::exception_ptr rabitq_exception;
            std::exception_ptr subspace_exception;
            std::mutex subspace_exception_mutex;
            std::atomic<int> next_subspace{0};
            std::atomic<int> completed_subspaces{0};
            double block_subspace_assign_ms = 0.0;
            auto record_subspace_exception = [&](std::exception_ptr ex) {
                std::lock_guard<std::mutex> lock(subspace_exception_mutex);
                if (!subspace_exception) subspace_exception = ex;
            };
            const int total_cores = std::max(1, omp_get_max_threads());
            const int threads_per_subspace = std::max(1, total_cores / subspace_workers);
            auto assign_one_subspace = [&](int m) {
                std::vector<int>& local_assignments = subspace_assignments[(size_t)m];
                std::vector<uint16_t>& local_assignments_u16 = subspace_assignments_u16[(size_t)m];
                if ((int)local_assignments.size() < rows) {
                    throw std::runtime_error("fused streaming assignment scratch under-sized");
                }
                if ((int)local_assignments_u16.size() < rows) {
                    throw std::runtime_error("fused streaming uint16 assignment scratch under-sized");
                }
                if (use_simd_assign) {
                    ivf_indices[(size_t)m].assign_batch_strided_simd_soa(
                        rotated.data() + (size_t)m * sub_dim, padded_dim,
                        local_assignments.data(), rows);
                } else if (use_points16_assign) {
                    ivf_indices[(size_t)m].assign_batch_strided_simd_points16(
                        rotated.data() + (size_t)m * sub_dim, padded_dim,
                        local_assignments.data(), rows);
                } else {
                    ivf_indices[(size_t)m].assign_batch_strided_with_scratch_cap(
                        rotated.data() + (size_t)m * sub_dim, padded_dim,
                        local_assignments.data(), rows,
                        assign_scratch_per_worker);
                }
                std::vector<int>& sizes = cell_sizes[(size_t)m];
                for (int i = 0; i < rows; ++i) {
                    int cell = local_assignments[(size_t)i];
                    if (cell < 0 || cell >= ivf_options.ivf_k) {
                        throw std::runtime_error("fused streaming IVF assignment out of range");
                    }
                    local_assignments_u16[(size_t)i] = (uint16_t)cell;
                    ++sizes[(size_t)cell];
                }
                write_all_fd(temp_write_fds[(size_t)m], local_assignments_u16.data(),
                             (size_t)rows * sizeof(uint16_t), temp_paths[(size_t)m]);
                int finished = completed_subspaces.fetch_add(1, std::memory_order_acq_rel) + 1;
                if (finished == M) block_subspace_assign_ms = sub_phase_timer.elapsed_ms();
            };
            auto steal_subspace_assignments = [&]() {
                omp_set_num_threads(threads_per_subspace);
                while (true) {
                    int m = next_subspace.fetch_add(1, std::memory_order_relaxed);
                    if (m >= M) break;
                    assign_one_subspace(m);
                }
            };
            sub_phase_timer.reset();
            ScopedThread rabitq_thread(std::thread([&]() {
                try {
                    const float* top_data = (padded_dim == dim) ? rotated.data() : rotated_dim.data();
                    Timer local_timer;
                    local_timer.reset();
                    top_ivf.assign_batch(top_data, top_assignments.data(), rows);
                    phase_top_assign_ms += local_timer.elapsed_ms();
                    local_timer.reset();
                    #pragma omp parallel for schedule(dynamic, 256)
                    for (int i = 0; i < rows; ++i) {
                        uint32_t cid = (uint32_t)top_assignments[(size_t)i];
                        rabitq_assignments[(size_t)i] = cid;
                        char* record = bin_chunk.data() + (size_t)i * (size_t)bb;
                        std::memcpy(record, &cid, sizeof(uint32_t));
                        const float* rot_vec = rotated.data() + (size_t)i * padded_dim;
                        const float* c_rot = rot_centroids.data() + (size_t)cid * padded_dim;
                        rabitqlib::quant::quantize_compact_one_bit<float>(
                            rot_vec, c_rot, padded_dim,
                            record + sizeof(uint32_t),
                            rabitqlib::METRIC_L2);
                    }
                    write_all_fd(assign_fd, rabitq_assignments.data(), (size_t)rows * sizeof(uint32_t), assignments_path);
                    write_all_fd(bin_fd, bin_chunk.data(), (size_t)rows * (size_t)bb, bin_data_path);
                    phase_quantize_write_ms += local_timer.elapsed_ms();
                    steal_subspace_assignments();
                } catch (...) {
                    rabitq_exception = std::current_exception();
                }
            }));

            try {
                int worker_count = std::min(subspace_workers, M);
                ThreadGroup workers;
                workers.reserve((size_t)worker_count);
                for (int worker = 0; worker < worker_count; ++worker) {
                    workers.emplace_back([&]() {
                        try {
                            steal_subspace_assignments();
                        } catch (...) {
                            record_subspace_exception(std::current_exception());
                        }
                    });
                }
                workers.join_all();
                if (subspace_exception) {
                    std::rethrow_exception(subspace_exception);
                }
            } catch (...) {
                rabitq_thread.join();
                verify_write_thread.join();
                throw;
            }
            rabitq_thread.join();
            verify_write_thread.join();
            phase_subspace_assign_ms += block_subspace_assign_ms;
            if (rabitq_exception) std::rethrow_exception(rabitq_exception);
            if (verify_write_exception) std::rethrow_exception(verify_write_exception);
        }
        input_warmer.reset();
        reader.release_buffers();
        std::vector<float>().swap(rotated);
        std::vector<float>().swap(rotated_dim);
        std::vector<float>().swap(sub_block);
        std::vector<int>().swap(top_assignments);
        std::vector<uint32_t>().swap(rabitq_assignments);
        std::vector<std::vector<int>>().swap(subspace_assignments);
        std::vector<std::vector<uint16_t>>().swap(subspace_assignments_u16);
        std::vector<char>().swap(bin_chunk);
        log_stream_build_phase_time("fused_encode_count", phase_timer.elapsed_ms());
        log_stream_build_phase_time("fused.rotate_write", phase_rotate_write_ms);
        log_stream_build_phase_time("fused.top_assign", phase_top_assign_ms);
        log_stream_build_phase_time("fused.quantize_write", phase_quantize_write_ms);
        log_stream_build_phase_time("fused.subspace_assign", phase_subspace_assign_ms);

        close_fd_checked(verify_fd, verify_path);
        close_fd_checked(assign_fd, assignments_path);
        close_fd_checked(bin_fd, bin_data_path);
        for (int m = 0; m < M; ++m) {
            close_fd_checked(temp_write_fds[(size_t)m], temp_paths[(size_t)m]);
        }

        phase_timer.reset();
        std::vector<std::vector<int>> offsets((size_t)M);
        staging_artifacts.cells_per_subspace.assign((size_t)M, (size_t)ivf_options.ivf_k);
        staging_artifacts.subspace_assignment_paths = temp_paths;
        staging_artifacts.ivf_paths.assign((size_t)M, std::string());
        for (int m = 0; m < M; ++m) {
            offsets[(size_t)m].assign((size_t)ivf_options.ivf_k + 1, 0);
            for (int k = 0; k < ivf_options.ivf_k; ++k) {
                offsets[(size_t)m][(size_t)k + 1] = offsets[(size_t)m][(size_t)k] + cell_sizes[(size_t)m][(size_t)k];
            }
            if (offsets[(size_t)m].back() != n) {
                throw std::runtime_error("fused streaming IVF posting counts do not sum to n");
            }
            std::string ivf_path = index_dir + "/ivf_subspace_" + std::to_string(m) + ".bin";
            save_ivf_index_exclusive(ivf_path, ivf_indices[(size_t)m], cell_sizes[(size_t)m]);
            created_paths.push_back(ivf_path);
            staging_artifacts.ivf_paths[(size_t)m] = ivf_path;
        }
        std::vector<IVFIndex>().swap(ivf_indices);
        std::vector<std::vector<int>>().swap(cell_sizes);

        if (!ivf_options.materialize_no_reorder_postings) {
#if HAS_AOCL_BLAS
            if (blis_threads_changed) {
                bli_thread_set_num_threads(saved_blis_threads);
                blis_threads_changed = false;
            }
#endif
            return staging_artifacts;
        }

        for (int m = 0; m < M; ++m) {
            posting_paths[(size_t)m] = index_dir + "/posting_subspace_" + std::to_string(m) + ".bin";
            posting_fds[(size_t)m] = create_posting_output_exclusive(posting_paths[(size_t)m],
                                                                     ivf_options.ivf_k, n, offsets[(size_t)m]);
            created_paths.push_back(posting_paths[(size_t)m]);
            temp_read_fds[(size_t)m] = ::open(temp_paths[(size_t)m].c_str(), O_RDONLY | O_CLOEXEC);
            if (temp_read_fds[(size_t)m] < 0) {
                throw std::runtime_error("cannot open temporary IVF assignments: " + temp_paths[(size_t)m]);
            }
        }

        std::vector<std::vector<int>> cursors = offsets;
        std::vector<int> block_counts((size_t)ivf_options.ivf_k);
        std::vector<int> block_offsets((size_t)ivf_options.ivf_k + 1);
        std::vector<int> block_cursors((size_t)ivf_options.ivf_k);
        std::vector<uint32_t> ids_chunk((size_t)block_rows);
        std::vector<uint16_t> temp_assignments((size_t)block_rows);
        off_t ids_base = (off_t)(2 * sizeof(int) + (size_t)(ivf_options.ivf_k + 1) * sizeof(int));
        for (int global_start = 0; global_start < n; global_start += block_rows) {
            int rows = std::min(block_rows, n - global_start);
            if ((int)ids_chunk.size() < rows) ids_chunk.resize((size_t)rows);
            if ((int)temp_assignments.size() < rows) temp_assignments.resize((size_t)rows);
            for (int m = 0; m < M; ++m) {
                read_all_fd(temp_read_fds[(size_t)m], temp_assignments.data(), (size_t)rows * sizeof(uint16_t), temp_paths[(size_t)m]);
                std::fill(block_counts.begin(), block_counts.end(), 0);
                for (int i = 0; i < rows; ++i) {
                    int cell = (int)temp_assignments[(size_t)i];
                    if (cell < 0 || cell >= ivf_options.ivf_k) {
                        throw std::runtime_error("fused temporary IVF assignment out of range during posting fill");
                    }
                    ++block_counts[(size_t)cell];
                }
                block_offsets[0] = 0;
                for (int k = 0; k < ivf_options.ivf_k; ++k) {
                    block_offsets[(size_t)k + 1] = block_offsets[(size_t)k] + block_counts[(size_t)k];
                }
                std::fill(block_cursors.begin(), block_cursors.end(), 0);
                for (int i = 0; i < rows; ++i) {
                    int cell = (int)temp_assignments[(size_t)i];
                    ids_chunk[(size_t)block_offsets[(size_t)cell] + (size_t)block_cursors[(size_t)cell]++] = (uint32_t)(global_start + i);
                }
                for (int k = 0; k < ivf_options.ivf_k; ++k) {
                    int count = block_counts[(size_t)k];
                    if (count == 0) continue;
                    int dst = cursors[(size_t)m][(size_t)k];
                    pwrite_all_fd(posting_fds[(size_t)m],
                                  ids_chunk.data() + (size_t)block_offsets[(size_t)k],
                                  (size_t)count * sizeof(uint32_t),
                                  ids_base + (off_t)dst * (off_t)sizeof(uint32_t),
                                  posting_paths[(size_t)m]);
                    cursors[(size_t)m][(size_t)k] += count;
                }
            }
        }
        log_stream_build_phase_time("posting_materialize", phase_timer.elapsed_ms());

        for (int m = 0; m < M; ++m) {
            for (int k = 0; k < ivf_options.ivf_k; ++k) {
                if (cursors[(size_t)m][(size_t)k] != offsets[(size_t)m][(size_t)k + 1]) {
                    throw std::runtime_error("fused streaming posting fill count mismatch");
                }
            }
            close_fd_checked(temp_read_fds[(size_t)m], temp_paths[(size_t)m]);
            close_fd_checked(posting_fds[(size_t)m], posting_paths[(size_t)m]);
            if (ivf_options.preserve_assignment_artifacts) {
                preserved_assignment_paths[(size_t)m] = index_dir + "/" + preserved_posting_assignment_name(m);
                if (::link(temp_paths[(size_t)m].c_str(), preserved_assignment_paths[(size_t)m].c_str()) != 0) {
                    throw std::runtime_error("failed preserving temporary IVF assignments: " + temp_paths[(size_t)m]);
                }
                if (::unlink(temp_paths[(size_t)m].c_str()) != 0) {
                    throw std::runtime_error("failed removing linked temporary IVF assignments: " + temp_paths[(size_t)m]);
                }
                created_paths.push_back(preserved_assignment_paths[(size_t)m]);
            } else {
                if (::unlink(temp_paths[(size_t)m].c_str()) != 0) {
                    throw std::runtime_error("failed removing temporary IVF assignments: " + temp_paths[(size_t)m]);
                }
            }
        }
        if (ivf_options.preserve_assignment_artifacts) {
            std::ostringstream manifest;
            manifest << "SCORE_POSTING_ASSIGNMENTS_V2\n";
            manifest << "n " << n << "\n";
            manifest << "M " << M << "\n";
            for (int m = 0; m < M; ++m) {
                manifest << "subspace " << m
                         << " K " << ivf_options.ivf_k
                         << " bytes " << ((size_t)n * streaming_posting_assignment_cell_bytes()) << "\n";
            }
            std::string manifest_payload = manifest.str();
            std::string manifest_path = index_dir + "/" + preserved_posting_assignment_manifest_name();
            write_exclusive_file(manifest_path, manifest_payload.data(), manifest_payload.size());
            created_paths.push_back(manifest_path);
        }
#if HAS_AOCL_BLAS
        if (blis_threads_changed) {
            bli_thread_set_num_threads(saved_blis_threads);
            blis_threads_changed = false;
        }
#endif
    } catch (...) {
#if HAS_AOCL_BLAS
        if (blis_threads_changed) {
            bli_thread_set_num_threads(saved_blis_threads);
            blis_threads_changed = false;
        }
#endif
        close_fd_if_open(verify_fd);
        close_fd_if_open(assign_fd);
        close_fd_if_open(bin_fd);
        for (int& fd : temp_write_fds) close_fd_if_open(fd);
        for (int& fd : temp_read_fds) close_fd_if_open(fd);
        for (int& fd : posting_fds) close_fd_if_open(fd);
        cleanup_paths();
        throw;
    }
    return staging_artifacts;
}

inline void write_streaming_rabitq_artifacts(const std::string& input_path,
                                             const std::string& index_dir,
                                             const RandomRotation& rotation,
                                             const StreamingRabitqOptions& options) {
    if (options.rabitq_k <= 0) {
        throw std::runtime_error("streaming RaBitQ requires positive rabitq_k");
    }
    if (options.block_rows <= 0 || options.sample_rows <= 0) {
        throw std::runtime_error("streaming RaBitQ block_rows/sample_rows must be positive");
    }

    PrefetchedVectorBlockReader reader(input_path, options.block_rows);
    int n = reader.n();
    int dim = reader.dim();
    int padded_dim = rotation.padded_dim();
    if (dim != rotation.dim()) {
        throw std::runtime_error("streaming RaBitQ input dimension does not match rotation dimension");
    }
    if (options.rabitq_k > std::min(n, options.sample_rows)) {
        throw std::runtime_error("streaming RaBitQ sample_rows must be >= rabitq_k");
    }
    if ((size_t)options.block_rows > std::numeric_limits<size_t>::max() / (size_t)std::max(dim, padded_dim)) {
        throw std::runtime_error("streaming RaBitQ block_rows would overflow allocation size");
    }
    std::vector<float> samples = collect_rotated_samples(
        input_path, rotation, options.sample_rows, options.block_rows);
    int sample_n = (int)(samples.size() / (size_t)dim);

    IVFIndex top_ivf;
    top_ivf.init(dim, options.rabitq_k, 0);
    top_ivf.train(samples.data(), sample_n, options.train_iters, (uint64_t)options.rabitq_seed);

    std::vector<float> rot_centroids((size_t)options.rabitq_k * padded_dim, 0.0f);
    const std::vector<float>& centroids = top_ivf.centroids();
    for (int k = 0; k < options.rabitq_k; ++k) {
        std::memcpy(rot_centroids.data() + (size_t)k * padded_dim,
                    centroids.data() + (size_t)k * dim,
                    (size_t)dim * sizeof(float));
    }
    const std::string centroids_path = index_dir + "/rabitq_centroids.bin";
    const std::string assignments_path = index_dir + "/rabitq_assignments.bin";
    const std::string bin_data_path = index_dir + "/rabitq_bin_data.bin";
    save_rabitq_centroids(centroids_path, rot_centroids, options.rabitq_k, dim, padded_dim);

    int assign_fd = ::open(assignments_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (assign_fd < 0) {
        ::unlink(centroids_path.c_str());
        throw std::runtime_error("cannot open streaming assignments output");
    }
    int bin_fd = ::open(bin_data_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (bin_fd < 0) {
        ::close(assign_fd);
        ::unlink(assignments_path.c_str());
        ::unlink(centroids_path.c_str());
        throw std::runtime_error("cannot open streaming binary-code output");
    }
    size_t bin_bytes = rabitqlib::BinDataMap<float>::data_bytes(padded_dim);
    int bb = (int)(sizeof(uint32_t) + bin_bytes);

    try {
        std::vector<float> rotated((size_t)options.block_rows * (size_t)padded_dim);
        std::vector<float> rotated_dim((size_t)options.block_rows * (size_t)dim);
        std::vector<int> int_assignments(options.block_rows);
        std::vector<uint32_t> assignments(options.block_rows);
        std::vector<char> bin_chunk((size_t)options.block_rows * bb);

        write_all_fd(assign_fd, &n, sizeof(int), assignments_path);
        write_all_fd(bin_fd, &n, sizeof(int), bin_data_path);
        write_all_fd(bin_fd, &bb, sizeof(int), bin_data_path);

        reader.start();
        while (true) {
            auto input_block = reader.next();
            int rows = input_block.rows;
            if (rows == 0) break;
            const std::vector<float>& block = *input_block.data;
            if (rotated.size() < (size_t)rows * (size_t)padded_dim) {
                rotated.resize((size_t)rows * (size_t)padded_dim);
            }
            if (rotated_dim.size() < (size_t)rows * (size_t)dim) {
                rotated_dim.resize((size_t)rows * (size_t)dim);
            }
            if ((int)int_assignments.size() < rows) int_assignments.resize(rows);
            if ((int)assignments.size() < rows) assignments.resize(rows);
            if (bin_chunk.size() < (size_t)rows * (size_t)bb) bin_chunk.resize((size_t)rows * (size_t)bb);

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < rows; ++i) {
                rotation.apply(block.data() + (size_t)i * dim,
                               rotated.data() + (size_t)i * padded_dim);
                std::memcpy(rotated_dim.data() + (size_t)i * dim,
                            rotated.data() + (size_t)i * padded_dim,
                            (size_t)dim * sizeof(float));
            }

            top_ivf.assign_batch(rotated_dim.data(), int_assignments.data(), rows);

            #pragma omp parallel for schedule(dynamic, 256)
            for (int i = 0; i < rows; ++i) {
                uint32_t cid = (uint32_t)int_assignments[(size_t)i];
                assignments[(size_t)i] = cid;
                char* out = bin_chunk.data() + (size_t)i * bb;
                std::memcpy(out, &cid, sizeof(uint32_t));
                const float* rot_vec = rotated.data() + (size_t)i * padded_dim;
                const float* c_rot = rot_centroids.data() + (size_t)cid * padded_dim;
                rabitqlib::quant::quantize_compact_one_bit<float>(
                    rot_vec, c_rot, padded_dim,
                    out + sizeof(uint32_t),
                    rabitqlib::METRIC_L2);
            }

            write_all_fd(assign_fd, assignments.data(), (size_t)rows * sizeof(uint32_t), assignments_path);
            write_all_fd(bin_fd, bin_chunk.data(), (size_t)rows * (size_t)bb, bin_data_path);
        }
    } catch (...) {
        ::close(assign_fd);
        ::close(bin_fd);
        ::unlink(assignments_path.c_str());
        ::unlink(bin_data_path.c_str());
        ::unlink(centroids_path.c_str());
        throw;
    }
    if (::close(assign_fd) != 0) {
        ::close(bin_fd);
        ::unlink(assignments_path.c_str());
        ::unlink(bin_data_path.c_str());
        ::unlink(centroids_path.c_str());
        throw std::runtime_error("failed closing streaming assignments output");
    }
    if (::close(bin_fd) != 0) {
        ::unlink(assignments_path.c_str());
        ::unlink(bin_data_path.c_str());
        ::unlink(centroids_path.c_str());
        throw std::runtime_error("failed closing streaming binary-code output");
    }
}

} // namespace score
