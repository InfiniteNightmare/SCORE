#pragma once
#include "core/common.h"
#include "core/rotation.h"
#include "core/rabitq.h"
#include "core/ivf.h"
#include "core/posting_lists.h"
#include <algorithm>
#include <vector>
#include <string>
#include <exception>
#include <fstream>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <unistd.h>

#if SCORE_USE_LIBURING
#include <liburing.h>
#endif

// RaBitQ-Library: for quantize_compact_one_bit (guaranteed compatible with split_single_estdist)
#include "rabitqlib/quantization/rabitq.hpp"

// BLIS thread control (AOCL-BLAS uses pthreads, not OMP)
#if HAS_AOCL_BLAS
extern "C" void bli_thread_set_num_threads(int);
extern "C" int bli_thread_get_num_threads(void);
#endif

namespace score {

namespace impl {
enum class RotatedDimStrideBacking {
    RotPaddedAlias,
    ReusedDataStorage,
};

enum class LateRotatedSource {
    VerifyRotPadded,
    IvfSlabs,
};

struct SharedRotationViews {
    std::vector<float> rot_padded;
    float* dim_stride_rows = nullptr;
    RotatedDimStrideBacking dim_stride_backing = RotatedDimStrideBacking::RotPaddedAlias;

    bool dim_stride_aliases_rot_padded() const {
        return dim_stride_backing == RotatedDimStrideBacking::RotPaddedAlias;
    }

    bool dim_stride_reuses_data_storage() const {
        return dim_stride_backing == RotatedDimStrideBacking::ReusedDataStorage;
    }
};

inline bool should_repack_ivf_from_rot_padded(bool rotate_subspace,
                                              const float* subspace_data,
                                              const float* dim_stride_rotated_view,
                                              const float* data_ptr,
                                              const float* verify_rotated_padded,
                                              int verify_rotated_dim,
                                              int dim) {
    return rotate_subspace && dim_stride_rotated_view != nullptr && subspace_data == dim_stride_rotated_view &&
           dim_stride_rotated_view == data_ptr && verify_rotated_padded != nullptr &&
           verify_rotated_dim > dim;
}

inline bool can_finalize_rotated_outputs_from_ivf_slabs(bool rotate_subspace,
                                                        bool subspace_force_raw,
                                                        const float* subspace_data,
                                                        const float* dim_stride_rotated_view,
                                                        const float* verify_rotated_padded,
                                                        int verify_rotated_dim,
                                                        int dim,
                                                        bool reuse_data_buffer_for_all_sub_data) {
    return rotate_subspace && !subspace_force_raw && reuse_data_buffer_for_all_sub_data &&
           dim_stride_rotated_view != nullptr && subspace_data == dim_stride_rotated_view &&
           verify_rotated_padded != nullptr && verify_rotated_dim == dim;
}

inline void copy_finalize_subspace(float* dst, const float* src, int sub_dim) {
#if HAS_AVX512
    int d = 0;
    for (; d + 16 <= sub_dim; d += 16) {
        _mm512_storeu_ps(dst + d, _mm512_loadu_ps(src + d));
    }
    int rem = sub_dim - d;
    if (rem > 0) {
        __mmask16 tail_mask = (__mmask16)((1u << rem) - 1u);
        _mm512_mask_storeu_ps(dst + d, tail_mask, _mm512_maskz_loadu_ps(tail_mask, src + d));
    }
    return;
#elif HAS_AVX2
    int d = 0;
    for (; d + 8 <= sub_dim; d += 8) {
        _mm256_storeu_ps(dst + d, _mm256_loadu_ps(src + d));
    }
    for (; d + 4 <= sub_dim; d += 4) {
        _mm_storeu_ps(dst + d, _mm_loadu_ps(src + d));
    }
    for (; d < sub_dim; ++d) {
        dst[d] = src[d];
    }
    return;
#elif defined(__SSE__)
    int d = 0;
    for (; d + 4 <= sub_dim; d += 4) {
        _mm_storeu_ps(dst + d, _mm_loadu_ps(src + d));
    }
    for (; d < sub_dim; ++d) {
        dst[d] = src[d];
    }
    return;
#else
    std::memcpy(dst, src, (size_t)sub_dim * sizeof(float));
#endif
}

inline void gather_rotated_rows_from_ivf_slabs(float* dst,
                                               const std::vector<float*>& all_sub_data_ptrs,
                                               const uint32_t* orig_ids,
                                               int count,
                                               int M,
                                               int sub_dim) {
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < count; ++j) {
        uint32_t orig = orig_ids[(size_t)j];
        float* row_dst = dst + (size_t)j * (size_t)M * (size_t)sub_dim;
        for (int m = 0; m < M; ++m) {
            copy_finalize_subspace(row_dst + (size_t)m * (size_t)sub_dim,
                                   all_sub_data_ptrs[m] + (size_t)orig * (size_t)sub_dim,
                                   sub_dim);
        }
    }
}

inline size_t finalize_rows_per_chunk(size_t record_bytes, int thread_count) {
    constexpr size_t kBaseChunkBytes = 256ull * PAGE_SIZE;
    constexpr size_t kTargetRowsPerThread = 64;
    size_t base_rows = std::max<size_t>(1, kBaseChunkBytes / std::max<size_t>(record_bytes, 1));
    size_t parallel_floor = (size_t)std::max(1, thread_count) * kTargetRowsPerThread;
    return std::max(base_rows, parallel_floor);
}

inline std::vector<int> normalize_pid_reorder_subspaces(const IndexConfig& config, int M) {
    std::vector<int> key_subspaces = config.pid_reorder_subspaces;
    if (key_subspaces.empty()) {
        key_subspaces.resize(M);
        std::iota(key_subspaces.begin(), key_subspaces.end(), 0);
    }
    return key_subspaces;
}

inline int pid_reorder_key_depth(const IndexConfig& config, const std::vector<int>& key_subspaces) {
    return std::min((int)key_subspaces.size(), std::max(0, config.pid_reorder_cell_key_depth));
}

inline std::vector<uint32_t> build_pid_reorder_perm_legacy(
    const std::vector<uint32_t>& assignments,
    const int* submajor_cell_ids,
    int n,
    int M,
    const IndexConfig& config) {
    std::vector<uint32_t> perm_to_orig((size_t)n);
    std::iota(perm_to_orig.begin(), perm_to_orig.end(), 0u);

    std::vector<int> key_subspaces = normalize_pid_reorder_subspaces(config, M);
    int key_depth = pid_reorder_key_depth(config, key_subspaces);
    auto cell_id = [&](uint32_t pid, int subspace) -> int {
        return submajor_cell_ids[(size_t)subspace * n + pid];
    };

    std::sort(perm_to_orig.begin(), perm_to_orig.end(), [&](uint32_t a, uint32_t b) {
        if (config.pid_reorder_key_mode == 0) {
            uint32_t ca = assignments[a], cb = assignments[b];
            if (ca != cb) return ca < cb;
            for (int i = 0; i < key_depth; i++) {
                int sa = key_subspaces[i];
                int va = cell_id(a, sa);
                int vb = cell_id(b, sa);
                if (va != vb) return va < vb;
            }
        } else if (config.pid_reorder_key_mode == 1) {
            for (int i = 0; i < key_depth; i++) {
                int sa = key_subspaces[i];
                int va = cell_id(a, sa);
                int vb = cell_id(b, sa);
                if (va != vb) return va < vb;
            }
            uint32_t ca = assignments[a], cb = assignments[b];
            if (ca != cb) return ca < cb;
        } else {
            for (int i = 0; i < key_depth; i++) {
                int sa = key_subspaces[i];
                int va = cell_id(a, sa);
                int vb = cell_id(b, sa);
                if (va != vb) return va < vb;
                uint32_t ca = assignments[a], cb = assignments[b];
                if (ca != cb) return ca < cb;
            }
        }
        return a < b;
    });
    return perm_to_orig;
}

template <typename KeyFn>
inline void stable_counting_pass(std::vector<uint32_t>& perm,
                                 std::vector<uint32_t>& tmp,
                                 size_t key_range,
                                 KeyFn key_fn) {
    if (key_range <= 1 || perm.empty()) return;

    std::vector<uint32_t> offsets(key_range, 0);
    for (uint32_t pid : perm) {
        offsets[(size_t)key_fn(pid)]++;
    }

    uint32_t sum = 0;
    for (size_t k = 0; k < key_range; ++k) {
        uint32_t count = offsets[k];
        offsets[k] = sum;
        sum += count;
    }

    for (uint32_t pid : perm) {
        size_t key = (size_t)key_fn(pid);
        tmp[(size_t)offsets[key]++] = pid;
    }
    perm.swap(tmp);
}

inline std::vector<uint32_t> build_pid_reorder_perm_counting(
    const std::vector<uint32_t>& assignments,
    const int* submajor_cell_ids,
    int n,
    int M,
    const std::vector<size_t>& cells_per_subspace,
    const IndexConfig& config) {
    std::vector<uint32_t> perm_to_orig((size_t)n);
    std::iota(perm_to_orig.begin(), perm_to_orig.end(), 0u);

    std::vector<int> key_subspaces = normalize_pid_reorder_subspaces(config, M);
    int key_depth = pid_reorder_key_depth(config, key_subspaces);
    if (config.pid_reorder_key_mode != 0 && config.pid_reorder_key_mode != 1 && key_depth == 0) {
        return perm_to_orig;
    }

    uint32_t max_assignment = 0;
    for (int i = 0; i < n; ++i) {
        max_assignment = std::max(max_assignment, assignments[(size_t)i]);
    }
    size_t assignment_range = (size_t)max_assignment + 1;

    std::vector<uint32_t> tmp((size_t)n);
    auto assignment_key = [&](uint32_t pid) -> uint32_t {
        return assignments[(size_t)pid];
    };
    auto cell_key = [&](int subspace) {
        return [&, subspace](uint32_t pid) -> uint32_t {
            return (uint32_t)submajor_cell_ids[(size_t)subspace * n + pid];
        };
    };
    auto cell_range = [&](int subspace) -> size_t {
        return cells_per_subspace[(size_t)subspace];
    };

    if (config.pid_reorder_key_mode == 0) {
        for (int i = key_depth - 1; i >= 0; --i) {
            int subspace = key_subspaces[(size_t)i];
            stable_counting_pass(perm_to_orig, tmp, cell_range(subspace), cell_key(subspace));
        }
        stable_counting_pass(perm_to_orig, tmp, assignment_range, assignment_key);
    } else if (config.pid_reorder_key_mode == 1) {
        stable_counting_pass(perm_to_orig, tmp, assignment_range, assignment_key);
        for (int i = key_depth - 1; i >= 0; --i) {
            int subspace = key_subspaces[(size_t)i];
            stable_counting_pass(perm_to_orig, tmp, cell_range(subspace), cell_key(subspace));
        }
    } else {
        for (int i = key_depth - 1; i >= 0; --i) {
            stable_counting_pass(perm_to_orig, tmp, assignment_range, assignment_key);
            int subspace = key_subspaces[(size_t)i];
            stable_counting_pass(perm_to_orig, tmp, cell_range(subspace), cell_key(subspace));
        }
    }
    return perm_to_orig;
}

struct DiskPidReorderOptions {
    int n = 0;
    int M = 0;
    size_t assignment_key_range = 0;
    std::vector<size_t> cells_per_subspace;
    std::vector<std::string> subspace_assignment_paths;
    std::string rabitq_assignments_path;
    std::string perm_to_orig_path;
    std::string scratch_perm_path;
    std::string orig_to_new_path;
    IndexConfig config;
    size_t io_rows = 65536;
    size_t bucket_buffer_rows = 64;
};

struct ReorderedArtifactFinalizeOptions {
    int n = 0;
    int M = 0;
    std::vector<size_t> cells_per_subspace;
    std::vector<std::string> subspace_assignment_paths;
    std::string raw_perm_to_orig_path;

    std::string verify_input_path;
    std::string assignments_input_path;
    std::string bin_data_input_path;

    std::string verify_output_path;
    std::string assignments_output_path;
    std::string bin_data_output_path;
    std::vector<std::string> posting_output_paths;
    std::string final_perm_to_orig_path;
    size_t io_rows = 65536;
    int rewrite_workers = 4;
    size_t max_cached_record_bytes = (size_t)512 * 1024 * 1024;
    size_t parallel_buffer_budget_bytes = 0;
};

struct PosthocReorderIndexOptions {
    std::string source_index_dir;
    std::string output_index_dir;
    int n = 0;
    int M = 0;
    IndexConfig config;
    size_t io_rows = 65536;
    size_t bucket_buffer_rows = 64;
    int rewrite_workers = 4;
};

struct PreservedPostingAssignments {
    bool available = false;
    std::vector<std::string> paths;
    std::vector<size_t> cells_per_subspace;
};

inline void disk_reorder_read_all(int fd, void* data, size_t bytes, const std::string& path) {
    char* ptr = static_cast<char*>(data);
    size_t done = 0;
    while (done < bytes) {
        ssize_t rc = ::read(fd, ptr + done, bytes - done);
        if (rc < 0 && errno == EINTR) continue;
        if (rc <= 0) throw std::runtime_error("failed reading reorder file: " + path);
        done += (size_t)rc;
    }
}

inline void disk_reorder_write_all(int fd, const void* data, size_t bytes, const std::string& path) {
    const char* ptr = static_cast<const char*>(data);
    size_t done = 0;
    while (done < bytes) {
        ssize_t rc = ::write(fd, ptr + done, bytes - done);
        if (rc < 0 && errno == EINTR) continue;
        if (rc <= 0) throw std::runtime_error("failed writing reorder file: " + path);
        done += (size_t)rc;
    }
}

inline void disk_reorder_pwrite_all(int fd,
                                    const void* data,
                                    size_t bytes,
                                    off_t offset,
                                    const std::string& path) {
    const char* ptr = static_cast<const char*>(data);
    size_t done = 0;
    while (done < bytes) {
        ssize_t rc = ::pwrite(fd, ptr + done, bytes - done, offset + (off_t)done);
        if (rc < 0 && errno == EINTR) continue;
        if (rc <= 0) throw std::runtime_error("failed positioned writing reorder file: " + path);
        done += (size_t)rc;
    }
}

inline void disk_reorder_pread_all(int fd,
                                   void* data,
                                   size_t bytes,
                                   off_t offset,
                                   const std::string& path) {
    char* ptr = static_cast<char*>(data);
    size_t done = 0;
    while (done < bytes) {
        ssize_t rc = ::pread(fd, ptr + done, bytes - done, offset + (off_t)done);
        if (rc < 0 && errno == EINTR) continue;
        if (rc <= 0) throw std::runtime_error("failed positioned reading reorder file: " + path);
        done += (size_t)rc;
    }
}

inline off_t disk_reorder_file_size(const std::string& path) {
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) {
        throw std::runtime_error("cannot stat reorder file: " + path);
    }
    return st.st_size;
}

inline void disk_reorder_expect_file_size(const std::string& path, off_t expected_bytes) {
    off_t actual = disk_reorder_file_size(path);
    if (actual != expected_bytes) {
        throw std::runtime_error("unexpected reorder file size: " + path);
    }
}

inline int disk_reorder_open_read(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) throw std::runtime_error("cannot open reorder input: " + path);
    return fd;
}

inline int disk_reorder_open_write_trunc(const std::string& path) {
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) throw std::runtime_error("cannot open reorder output: " + path);
    return fd;
}

inline int disk_reorder_open_write_exclusive(const std::string& path) {
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0) throw std::runtime_error("cannot create reorder output: " + path);
    return fd;
}

inline int disk_reorder_open_write_replace_exclusive(const std::string& path) {
    if (::unlink(path.c_str()) != 0 && errno != ENOENT) {
        throw std::runtime_error("cannot replace reorder output: " + path);
    }
    return disk_reorder_open_write_exclusive(path);
}

inline std::string disk_reorder_final_marker_temp_path(const std::string& final_path, int attempt) {
    return final_path + ".tmp.finalizing." + std::to_string((long long)getpid()) + "." + std::to_string(attempt);
}

inline void disk_reorder_publish_final_marker_temp(const std::string& temp_path,
                                                   const std::string& final_path) {
    if (::link(temp_path.c_str(), final_path.c_str()) != 0) {
        throw std::runtime_error("cannot publish final reorder marker: " + final_path);
    }
    (void)::unlink(temp_path.c_str());
}

inline void disk_reorder_close_checked(int fd, const std::string& path) {
    if (fd >= 0 && ::close(fd) != 0) {
        throw std::runtime_error("failed closing reorder file: " + path);
    }
}

inline bool disk_reorder_path_exists(const std::string& path) {
    return ::access(path.c_str(), F_OK) == 0;
}

inline std::string preserved_posthoc_assignment_manifest_path(const std::string& index_dir) {
    return index_dir + "/posting_assignments.manifest";
}

inline std::string preserved_posthoc_assignment_path(const std::string& index_dir, int m) {
    return index_dir + "/posting_assign_subspace_" + std::to_string(m) + ".assign.bin";
}

inline constexpr size_t posting_assignment_cell_bytes() {
    return sizeof(uint16_t);
}

inline int read_and_validate_posting_header_k(const std::string& posting_path,
                                              int n,
                                              int expected_K) {
    int fd = disk_reorder_open_read(posting_path);
    try {
        int K = 0, total = 0;
        disk_reorder_read_all(fd, &K, sizeof(int), posting_path);
        disk_reorder_read_all(fd, &total, sizeof(int), posting_path);
        if (K <= 0 || total != n || K != expected_K) {
            throw std::runtime_error("preserved assignment posting header mismatch: " + posting_path);
        }
        std::vector<int> offsets((size_t)K + 1);
        disk_reorder_read_all(fd, offsets.data(), offsets.size() * sizeof(int), posting_path);
        if (offsets.front() != 0 || offsets.back() != n) {
            throw std::runtime_error("preserved assignment posting offsets mismatch: " + posting_path);
        }
        for (int k = 0; k < K; ++k) {
            if (offsets[(size_t)k] > offsets[(size_t)k + 1]) {
                throw std::runtime_error("preserved assignment posting offsets are non-monotonic: " + posting_path);
            }
        }
        off_t expected_size = (off_t)(2 * sizeof(int) + ((size_t)K + 1) * sizeof(int) + (size_t)n * sizeof(uint32_t));
        disk_reorder_expect_file_size(posting_path, expected_size);
        disk_reorder_close_checked(fd, posting_path);
        return K;
    } catch (...) {
        ::close(fd);
        throw;
    }
}

inline PreservedPostingAssignments load_preserved_posting_assignments(const std::string& source_index_dir,
                                                                      int n,
                                                                      int M) {
    PreservedPostingAssignments preserved;
    std::string manifest_path = preserved_posthoc_assignment_manifest_path(source_index_dir);
    if (!disk_reorder_path_exists(manifest_path)) return preserved;

    std::ifstream in(manifest_path);
    if (!in.is_open()) throw std::runtime_error("cannot open preserved assignment manifest: " + manifest_path);
    std::string magic, label;
    int manifest_n = -1, manifest_M = -1;
    in >> magic;
    if (magic != "SCORE_POSTING_ASSIGNMENTS_V2") {
        throw std::runtime_error("invalid preserved assignment manifest magic: " + manifest_path);
    }
    in >> label >> manifest_n;
    if (!in || label != "n" || manifest_n != n) {
        throw std::runtime_error("preserved assignment manifest n mismatch: " + manifest_path);
    }
    in >> label >> manifest_M;
    if (!in || label != "M" || manifest_M != M) {
        throw std::runtime_error("preserved assignment manifest M mismatch: " + manifest_path);
    }

    preserved.available = true;
    preserved.paths.resize((size_t)M);
    preserved.cells_per_subspace.assign((size_t)M, 0);
    std::vector<bool> seen((size_t)M, false);
    for (int line = 0; line < M; ++line) {
        std::string k_label, bytes_label;
        int m = -1, K = -1;
        size_t bytes = 0;
        in >> label >> m >> k_label >> K >> bytes_label >> bytes;
        if (!in || label != "subspace" || k_label != "K" || bytes_label != "bytes" ||
            m < 0 || m >= M || seen[(size_t)m] || K <= 0 ||
            K > (int)std::numeric_limits<uint16_t>::max() ||
            bytes != (size_t)n * posting_assignment_cell_bytes()) {
            throw std::runtime_error("invalid preserved assignment manifest entry: " + manifest_path);
        }
        std::string assignment_path = preserved_posthoc_assignment_path(source_index_dir, m);
        disk_reorder_expect_file_size(assignment_path, (off_t)n * (off_t)posting_assignment_cell_bytes());
        int posting_K = read_and_validate_posting_header_k(
            source_index_dir + "/posting_subspace_" + std::to_string(m) + ".bin", n, K);
        preserved.paths[(size_t)m] = assignment_path;
        preserved.cells_per_subspace[(size_t)m] = (size_t)posting_K;
        seen[(size_t)m] = true;
    }
    for (int m = 0; m < M; ++m) {
        if (!seen[(size_t)m]) {
            throw std::runtime_error("missing preserved assignment manifest entry: " + manifest_path);
        }
    }
    std::string trailing;
    if (in >> trailing) {
        throw std::runtime_error("trailing data in preserved assignment manifest: " + manifest_path);
    }
    return preserved;
}

inline void copy_file_exclusive(const std::string& src_path,
                                const std::string& dst_path,
                                size_t io_bytes = 1u << 20) {
    int in_fd = disk_reorder_open_read(src_path);
    int out_fd = disk_reorder_open_write_exclusive(dst_path);
    std::vector<char> buffer(std::max<size_t>(1, io_bytes));
    try {
        while (true) {
            ssize_t rc = ::read(in_fd, buffer.data(), buffer.size());
            if (rc < 0 && errno == EINTR) continue;
            if (rc < 0) throw std::runtime_error("failed reading file copy input: " + src_path);
            if (rc == 0) break;
            disk_reorder_write_all(out_fd, buffer.data(), (size_t)rc, dst_path);
        }
    } catch (...) {
        ::close(in_fd);
        ::close(out_fd);
        ::unlink(dst_path.c_str());
        throw;
    }
    disk_reorder_close_checked(in_fd, src_path);
    disk_reorder_close_checked(out_fd, dst_path);
}

inline int read_headered_uint32_count(const std::string& path) {
    disk_reorder_expect_file_size(path, disk_reorder_file_size(path));
    int fd = disk_reorder_open_read(path);
    int n = 0;
    try {
        disk_reorder_read_all(fd, &n, sizeof(int), path);
    } catch (...) {
        ::close(fd);
        throw;
    }
    disk_reorder_close_checked(fd, path);
    if (n < 0) throw std::runtime_error("negative count in uint32 artifact: " + path);
    return n;
}

inline std::vector<uint32_t> load_reorder_assignment_key_column(
    const std::string& path,
    int n,
    size_t assignment_key_range) {
    if (assignment_key_range == 0) {
        throw std::runtime_error("assignment key range must be positive for disk reorder");
    }
    disk_reorder_expect_file_size(path, (off_t)sizeof(int) + (off_t)n * (off_t)sizeof(uint32_t));
    int fd = disk_reorder_open_read(path);
    std::vector<uint32_t> keys((size_t)n);
    try {
        int file_n = 0;
        disk_reorder_read_all(fd, &file_n, sizeof(int), path);
        if (file_n != n) throw std::runtime_error("rabitq assignment count mismatch for disk reorder");
        disk_reorder_read_all(fd, keys.data(), (size_t)n * sizeof(uint32_t), path);
    } catch (...) {
        ::close(fd);
        throw;
    }
    disk_reorder_close_checked(fd, path);
    for (uint32_t key : keys) {
        if ((size_t)key >= assignment_key_range) {
            throw std::runtime_error("rabitq assignment out of range for disk reorder");
        }
    }
    return keys;
}

inline std::vector<uint32_t> load_reorder_cell_key_column(
    const std::string& path,
    int n,
    size_t cell_key_range) {
    if (cell_key_range == 0) {
        throw std::runtime_error("cell key range must be positive for disk reorder");
    }
    if (cell_key_range > (size_t)std::numeric_limits<uint16_t>::max()) {
        throw std::runtime_error("cell key range exceeds uint16 assignment limit for disk reorder");
    }
    disk_reorder_expect_file_size(path, (off_t)n * (off_t)posting_assignment_cell_bytes());
    int fd = disk_reorder_open_read(path);
    std::vector<uint32_t> keys((size_t)n);
    const size_t chunk_rows = 65536;
    std::vector<uint16_t> raw(chunk_rows);
    try {
        for (int start = 0; start < n; start += (int)chunk_rows) {
            int rows = std::min<int>((int)chunk_rows, n - start);
            disk_reorder_read_all(fd, raw.data(), (size_t)rows * sizeof(uint16_t), path);
            for (int i = 0; i < rows; ++i) {
                uint16_t key = raw[(size_t)i];
                if ((size_t)key >= cell_key_range) {
                    throw std::runtime_error("cell assignment out of range for disk reorder");
                }
                keys[(size_t)start + (size_t)i] = (uint32_t)key;
            }
        }
    } catch (...) {
        ::close(fd);
        throw;
    }
    disk_reorder_close_checked(fd, path);
    return keys;
}

inline void initialize_identity_perm_file(const std::string& path, int n, size_t io_rows) {
    int fd = disk_reorder_open_write_exclusive(path);
    std::vector<uint32_t> chunk(std::max<size_t>(1, io_rows));
    try {
        for (int start = 0; start < n; start += (int)chunk.size()) {
            int rows = std::min<int>((int)chunk.size(), n - start);
            for (int i = 0; i < rows; ++i) chunk[(size_t)i] = (uint32_t)(start + i);
            disk_reorder_write_all(fd, chunk.data(), (size_t)rows * sizeof(uint32_t), path);
        }
    } catch (...) {
        ::close(fd);
        ::unlink(path.c_str());
        throw;
    }
    disk_reorder_close_checked(fd, path);
}

inline void copy_reorder_file(const std::string& src_path,
                              const std::string& dst_path,
                              int n,
                              size_t io_rows) {
    disk_reorder_expect_file_size(src_path, (off_t)n * (off_t)sizeof(uint32_t));
    int src_fd = disk_reorder_open_read(src_path);
    int dst_fd = disk_reorder_open_write_replace_exclusive(dst_path);
    std::vector<uint32_t> chunk(std::max<size_t>(1, io_rows));
    try {
        for (int start = 0; start < n; start += (int)chunk.size()) {
            int rows = std::min<int>((int)chunk.size(), n - start);
            disk_reorder_read_all(src_fd, chunk.data(), (size_t)rows * sizeof(uint32_t), src_path);
            disk_reorder_write_all(dst_fd, chunk.data(), (size_t)rows * sizeof(uint32_t), dst_path);
        }
    } catch (...) {
        ::close(src_fd);
        ::close(dst_fd);
        throw;
    }
    disk_reorder_close_checked(src_fd, src_path);
    disk_reorder_close_checked(dst_fd, dst_path);
}

inline void stable_counting_pass_disk(const std::string& src_perm_path,
                                      const std::string& dst_perm_path,
                                      int n,
                                      const std::vector<uint32_t>& keys_by_orig,
                                      size_t key_range,
                                      size_t io_rows,
                                      size_t bucket_buffer_rows) {
    if (key_range == 0) throw std::runtime_error("disk reorder key range must be positive");
    if ((int)keys_by_orig.size() != n) throw std::runtime_error("disk reorder key column length mismatch");
    disk_reorder_expect_file_size(src_perm_path, (off_t)n * (off_t)sizeof(uint32_t));

    std::vector<uint32_t> offsets(key_range, 0);
    std::vector<uint32_t> chunk(std::max<size_t>(1, io_rows));
    int src_fd = disk_reorder_open_read(src_perm_path);
    try {
        for (int start = 0; start < n; start += (int)chunk.size()) {
            int rows = std::min<int>((int)chunk.size(), n - start);
            disk_reorder_read_all(src_fd, chunk.data(), (size_t)rows * sizeof(uint32_t), src_perm_path);
            for (int i = 0; i < rows; ++i) {
                uint32_t pid = chunk[(size_t)i];
                if (pid >= (uint32_t)n) throw std::runtime_error("disk reorder permutation pid out of range");
                uint32_t key = keys_by_orig[(size_t)pid];
                if ((size_t)key >= key_range) throw std::runtime_error("disk reorder key out of range");
                offsets[(size_t)key]++;
            }
        }
    } catch (...) {
        ::close(src_fd);
        throw;
    }
    disk_reorder_close_checked(src_fd, src_perm_path);

    uint32_t sum = 0;
    for (size_t k = 0; k < key_range; ++k) {
        uint32_t count = offsets[k];
        offsets[k] = sum;
        sum += count;
    }
    if (sum != (uint32_t)n) throw std::runtime_error("disk reorder count sum mismatch");

    src_fd = disk_reorder_open_read(src_perm_path);
    int dst_fd = disk_reorder_open_write_replace_exclusive(dst_perm_path);
    std::vector<uint32_t> write_pos = offsets;
    std::vector<std::vector<uint32_t>> bucket_buffers(key_range);
    size_t buffer_rows = std::max<size_t>(1, bucket_buffer_rows);
    for (auto& buffer : bucket_buffers) buffer.reserve(buffer_rows);
    auto flush_bucket = [&](size_t key) {
        std::vector<uint32_t>& buffer = bucket_buffers[key];
        if (buffer.empty()) return;
        uint32_t dst = write_pos[key];
        disk_reorder_pwrite_all(dst_fd, buffer.data(), buffer.size() * sizeof(uint32_t),
                                (off_t)dst * (off_t)sizeof(uint32_t), dst_perm_path);
        write_pos[key] += (uint32_t)buffer.size();
        buffer.clear();
    };
    try {
        for (int start = 0; start < n; start += (int)chunk.size()) {
            int rows = std::min<int>((int)chunk.size(), n - start);
            disk_reorder_read_all(src_fd, chunk.data(), (size_t)rows * sizeof(uint32_t), src_perm_path);
            for (int i = 0; i < rows; ++i) {
                uint32_t pid = chunk[(size_t)i];
                uint32_t key = keys_by_orig[(size_t)pid];
                std::vector<uint32_t>& buffer = bucket_buffers[(size_t)key];
                buffer.push_back(pid);
                if (buffer.size() == buffer_rows) flush_bucket((size_t)key);
            }
        }
        for (size_t key = 0; key < key_range; ++key) flush_bucket(key);
    } catch (...) {
        ::close(src_fd);
        ::close(dst_fd);
        throw;
    }
    disk_reorder_close_checked(src_fd, src_perm_path);
    disk_reorder_close_checked(dst_fd, dst_perm_path);
    disk_reorder_expect_file_size(dst_perm_path, (off_t)n * (off_t)sizeof(uint32_t));
}

inline void write_orig_to_new_from_perm(const std::string& perm_to_orig_path,
                                        const std::string& orig_to_new_path,
                                        int n,
                                        size_t io_rows) {
    disk_reorder_expect_file_size(perm_to_orig_path, (off_t)n * (off_t)sizeof(uint32_t));
    int src_fd = disk_reorder_open_read(perm_to_orig_path);
    int dst_fd = disk_reorder_open_write_exclusive(orig_to_new_path);
    std::vector<uint32_t> chunk(std::max<size_t>(1, io_rows));
    try {
        for (int start = 0; start < n; start += (int)chunk.size()) {
            int rows = std::min<int>((int)chunk.size(), n - start);
            disk_reorder_read_all(src_fd, chunk.data(), (size_t)rows * sizeof(uint32_t), perm_to_orig_path);
            for (int i = 0; i < rows; ++i) {
                uint32_t orig = chunk[(size_t)i];
                if (orig >= (uint32_t)n) throw std::runtime_error("perm_to_orig pid out of range while writing inverse");
                uint32_t new_pid = (uint32_t)(start + i);
                disk_reorder_pwrite_all(dst_fd, &new_pid, sizeof(uint32_t),
                                        (off_t)orig * (off_t)sizeof(uint32_t), orig_to_new_path);
            }
        }
    } catch (...) {
        ::close(src_fd);
        ::close(dst_fd);
        ::unlink(orig_to_new_path.c_str());
        throw;
    }
    disk_reorder_close_checked(src_fd, perm_to_orig_path);
    disk_reorder_close_checked(dst_fd, orig_to_new_path);
    disk_reorder_expect_file_size(orig_to_new_path, (off_t)n * (off_t)sizeof(uint32_t));
}

inline void initialize_uint32_file(const std::string& path, int n, uint32_t value, size_t io_rows) {
    int fd = disk_reorder_open_write_exclusive(path);
    std::vector<uint32_t> chunk(std::max<size_t>(1, io_rows), value);
    try {
        for (int start = 0; start < n; start += (int)chunk.size()) {
            int rows = std::min<int>((int)chunk.size(), n - start);
            disk_reorder_write_all(fd, chunk.data(), (size_t)rows * sizeof(uint32_t), path);
        }
    } catch (...) {
        ::close(fd);
        ::unlink(path.c_str());
        throw;
    }
    disk_reorder_close_checked(fd, path);
}

inline bool rewrite_fadvise_enabled() {
    const char* env = std::getenv("SCORE_REWRITE_FADVISE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

inline bool rewrite_io_uring_enabled() {
#if SCORE_USE_LIBURING
    const char* env = std::getenv("SCORE_REWRITE_IO_URING");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
#else
    return false;
#endif
}

inline unsigned rewrite_io_uring_queue_depth() {
    constexpr unsigned kDefaultDepth = 64;
    constexpr unsigned kMinDepth = 1;
    constexpr unsigned kMaxDepth = 1024;
    const char* env = std::getenv("SCORE_REWRITE_IO_URING_QD");
    if (env == nullptr || env[0] == '\0') return kDefaultDepth;
    char* end = nullptr;
    unsigned long value = std::strtoul(env, &end, 10);
    if (end == env || (end != nullptr && *end != '\0')) return kDefaultDepth;
    return (unsigned)std::max<unsigned long>(kMinDepth, std::min<unsigned long>(value, kMaxDepth));
}

inline unsigned rewrite_io_uring_cqe_batch() {
    constexpr unsigned kDefaultBatch = 32;
    constexpr unsigned kMinBatch = 1;
    constexpr unsigned kMaxBatch = 256;
    const char* env = std::getenv("SCORE_REWRITE_IO_URING_CQE_BATCH");
    if (env == nullptr || env[0] == '\0') return kDefaultBatch;
    char* end = nullptr;
    unsigned long value = std::strtoul(env, &end, 10);
    if (end == env || (end != nullptr && *end != '\0')) return kDefaultBatch;
    return (unsigned)std::max<unsigned long>(kMinBatch, std::min<unsigned long>(value, kMaxBatch));
}

inline bool rewrite_io_uring_registered_files_enabled() {
#if SCORE_USE_LIBURING
    const char* env = std::getenv("SCORE_REWRITE_IO_URING_FIXED_FILE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
#else
    return false;
#endif
}

inline bool rewrite_io_uring_taskrun_flags_enabled() {
#if SCORE_USE_LIBURING
    const char* env = std::getenv("SCORE_REWRITE_IO_URING_TASKRUN");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
#else
    return false;
#endif
}

inline bool rewrite_io_trace_enabled() {
    const char* env = std::getenv("SCORE_REWRITE_IO_TRACE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

inline size_t posting_materialize_ids_buffer_capacity(int K) {
    if (K <= 0) return 0;
    constexpr size_t kTargetBufferBytes = 4ull * 1024ull * 1024ull;
    constexpr size_t kMinIdsPerCell = 16;
    constexpr size_t kMaxIdsPerCell = 1024;
    size_t by_budget = kTargetBufferBytes / ((size_t)K * sizeof(uint32_t));
    return std::max(kMinIdsPerCell, std::min(kMaxIdsPerCell, by_budget));
}

struct FinalizePhaseTraceSample {
    Timer wall;
    struct rusage usage;
};

inline double timeval_ms(const struct timeval& value) {
    return (double)value.tv_sec * 1000.0 + (double)value.tv_usec / 1000.0;
}

inline FinalizePhaseTraceSample begin_finalize_phase_trace() {
    FinalizePhaseTraceSample sample;
    sample.wall.reset();
    getrusage(RUSAGE_SELF, &sample.usage);
    return sample;
}

inline void trace_finalize_phase(const char* phase,
                                 const FinalizePhaseTraceSample& sample) noexcept {
    if (!rewrite_io_trace_enabled()) return;
    try {
        struct rusage end_usage;
        getrusage(RUSAGE_SELF, &end_usage);
        double wall_ms = sample.wall.elapsed_ms();
        double user_ms = timeval_ms(end_usage.ru_utime) - timeval_ms(sample.usage.ru_utime);
        double sys_ms = timeval_ms(end_usage.ru_stime) - timeval_ms(sample.usage.ru_stime);
        double cpu_util = wall_ms > 0.0 ? (user_ms + sys_ms) / wall_ms : 0.0;
        std::cerr << "[FinalizePhase] phase=" << phase
                  << " wall_ms=" << wall_ms
                  << " user_ms=" << user_ms
                  << " sys_ms=" << sys_ms
                  << " cpu_util=" << cpu_util << std::endl;
    } catch (...) {
        std::cerr.clear();
    }
}

inline void trace_finalize_phase(const std::string& phase,
                                 const FinalizePhaseTraceSample& sample) noexcept {
    trace_finalize_phase(phase.c_str(), sample);
}

inline size_t rewrite_parallel_buffer_budget_bytes() {
    constexpr size_t kDefaultMiB = 256;
    constexpr size_t kMinMiB = 16;
    constexpr size_t kMaxMiB = 4096;
    const char* env = std::getenv("SCORE_REWRITE_BUFFER_MB");
    if (env == nullptr || env[0] == '\0') return kDefaultMiB * 1024 * 1024;
    char* end = nullptr;
    unsigned long long value = std::strtoull(env, &end, 10);
    if (end == env || (end != nullptr && *end != '\0')) return kDefaultMiB * 1024 * 1024;
    size_t mib = (size_t)std::max<unsigned long long>(kMinMiB,
        std::min<unsigned long long>(value, kMaxMiB));
    return mib * 1024 * 1024;
}

inline void disk_reorder_fadvise_ignore(int fd, off_t offset, off_t len, int advice) {
#if defined(_POSIX_ADVISORY_INFO) || defined(POSIX_FADV_RANDOM)
    (void)::posix_fadvise(fd, offset, len, advice);
#else
    (void)fd;
    (void)offset;
    (void)len;
    (void)advice;
#endif
}

inline void disk_reorder_preallocate_output(int fd,
                                            const std::string& path,
                                            off_t bytes) {
    if (bytes <= 0) return;
#if defined(__linux__)
    int rc = 0;
    do {
        rc = ::posix_fallocate(fd, 0, bytes);
    } while (rc == EINTR);
    if (rewrite_io_trace_enabled()) {
        std::cerr << "[RewriteIO] preallocate output=" << path
                  << " bytes=" << bytes
                  << " rc=" << rc << std::endl;
    }
    if (rc == 0 || rc == EOPNOTSUPP || rc == ENOSYS || rc == EINVAL) return;
    throw std::runtime_error("failed preallocating reorder output: " + path +
                             " rc=" + std::to_string(rc));
#else
    (void)fd;
    (void)path;
    (void)bytes;
#endif
}

inline void build_pid_reorder_perm_disk_staging(const DiskPidReorderOptions& options);

inline void rewrite_fixed_records_by_perm_parallel_large(const std::string& input_path,
                                                         const std::string& output_path,
                                                         const std::string& raw_perm_path,
                                                         int n,
                                                         off_t input_header_bytes,
                                                         size_t output_header_bytes,
                                                         size_t record_bytes,
                                                         size_t io_rows,
                                                         int rewrite_workers,
                                                         size_t parallel_buffer_budget_bytes,
                                                         int perm_fd,
                                                         int in_fd,
                                                         int out_fd);

inline void rewrite_fixed_records_by_perm(const std::string& input_path,
                                          const std::string& output_path,
                                          const std::string& raw_perm_path,
                                          int n,
                                          off_t input_header_bytes,
                                          const void* output_header,
                                          size_t output_header_bytes,
                                          size_t record_bytes,
                                          size_t io_rows,
                                          int rewrite_workers = 4,
                                          size_t max_cached_record_bytes = (size_t)512 * 1024 * 1024,
                                          size_t parallel_buffer_budget_bytes = 0) {
    disk_reorder_expect_file_size(raw_perm_path, (off_t)n * (off_t)sizeof(uint32_t));
    disk_reorder_expect_file_size(input_path, input_header_bytes + (off_t)n * (off_t)record_bytes);
    int perm_fd = disk_reorder_open_read(raw_perm_path);
    int in_fd = disk_reorder_open_read(input_path);
    int out_fd = disk_reorder_open_write_exclusive(output_path);
    try {
        disk_reorder_preallocate_output(out_fd, output_path,
            (off_t)output_header_bytes + (off_t)n * (off_t)record_bytes);
        if (output_header_bytes > 0) {
            disk_reorder_write_all(out_fd, output_header, output_header_bytes, output_path);
        }
        size_t total_record_bytes = (size_t)n * record_bytes;
        if (record_bytes > 0 && total_record_bytes / record_bytes == (size_t)n &&
            max_cached_record_bytes > 0 &&
            total_record_bytes <= max_cached_record_bytes) {
            std::vector<uint32_t> perm_chunk(std::max<size_t>(1, io_rows));
            std::vector<char> record_chunk(perm_chunk.size() * record_bytes);
            std::vector<char> input_records(total_record_bytes);
            if (::lseek(in_fd, input_header_bytes, SEEK_SET) < 0) {
                throw std::runtime_error("failed seeking input records while rewriting: " + input_path);
            }
            disk_reorder_read_all(in_fd, input_records.data(), total_record_bytes, input_path);
            for (int start = 0; start < n; start += (int)perm_chunk.size()) {
                int rows = std::min<int>((int)perm_chunk.size(), n - start);
                disk_reorder_read_all(perm_fd, perm_chunk.data(), (size_t)rows * sizeof(uint32_t), raw_perm_path);
                if (record_chunk.size() < (size_t)rows * record_bytes) record_chunk.resize((size_t)rows * record_bytes);
                for (int j = 0; j < rows; ++j) {
                    uint32_t orig = perm_chunk[(size_t)j];
                    if (orig >= (uint32_t)n) throw std::runtime_error("perm pid out of range while rewriting records");
                    std::memcpy(record_chunk.data() + (size_t)j * record_bytes,
                                input_records.data() + (size_t)orig * record_bytes,
                                record_bytes);
                }
                disk_reorder_write_all(out_fd, record_chunk.data(), (size_t)rows * record_bytes, output_path);
            }
        } else {
            rewrite_fixed_records_by_perm_parallel_large(input_path, output_path, raw_perm_path,
                                                         n, input_header_bytes, output_header_bytes,
                                                         record_bytes, io_rows, rewrite_workers,
                                                         parallel_buffer_budget_bytes,
                                                         perm_fd, in_fd, out_fd);
        }
    } catch (...) {
        ::close(perm_fd);
        ::close(in_fd);
        ::close(out_fd);
        ::unlink(output_path.c_str());
        throw;
    }
    disk_reorder_close_checked(perm_fd, raw_perm_path);
    disk_reorder_close_checked(in_fd, input_path);
    disk_reorder_close_checked(out_fd, output_path);
}

inline void rewrite_fixed_records_by_perm_parallel_large(const std::string& input_path,
                                                         const std::string& output_path,
                                                         const std::string& raw_perm_path,
                                                         int n,
                                                         off_t input_header_bytes,
                                                         size_t output_header_bytes,
                                                         size_t record_bytes,
                                                         size_t io_rows,
                                                         int rewrite_workers,
                                                         size_t parallel_buffer_budget_bytes,
                                                         int perm_fd,
                                                          int in_fd,
                                                          int out_fd) {
    if (n <= 0 || record_bytes == 0) return;
    struct RewriteIoTraceStats {
        uint64_t chunks = 0;
        uint64_t rows = 0;
        uint64_t source_read_ops = 0;
        uint64_t io_uring_submits = 0;
        uint64_t io_uring_cqe_batches = 0;
        uint64_t io_uring_cqes = 0;
        uint64_t short_reads = 0;
        uint64_t fallback_chunks = 0;
        uint64_t output_writes = 0;
        double perm_read_ms = 0.0;
        double pair_build_ms = 0.0;
        double sort_ms = 0.0;
        double source_read_ms = 0.0;
        double output_write_ms = 0.0;
    };
    auto add_trace_stats = [](RewriteIoTraceStats& dst, const RewriteIoTraceStats& src) {
        dst.chunks += src.chunks;
        dst.rows += src.rows;
        dst.source_read_ops += src.source_read_ops;
        dst.io_uring_submits += src.io_uring_submits;
        dst.io_uring_cqe_batches += src.io_uring_cqe_batches;
        dst.io_uring_cqes += src.io_uring_cqes;
        dst.short_reads += src.short_reads;
        dst.fallback_chunks += src.fallback_chunks;
        dst.output_writes += src.output_writes;
        dst.perm_read_ms += src.perm_read_ms;
        dst.pair_build_ms += src.pair_build_ms;
        dst.sort_ms += src.sort_ms;
        dst.source_read_ms += src.source_read_ms;
        dst.output_write_ms += src.output_write_ms;
    };
    auto run_timed_ms = [](auto&& fn) -> double {
        Timer timer;
        timer.reset();
        fn();
        return timer.elapsed_ms();
    };
    bool trace_io = rewrite_io_trace_enabled();
    Timer rewrite_trace_timer;
    if (trace_io) rewrite_trace_timer.reset();
    if (rewrite_fadvise_enabled()) {
        disk_reorder_fadvise_ignore(perm_fd, 0, (off_t)n * (off_t)sizeof(uint32_t), POSIX_FADV_SEQUENTIAL);
        disk_reorder_fadvise_ignore(in_fd, input_header_bytes, (off_t)n * (off_t)record_bytes, POSIX_FADV_RANDOM);
        disk_reorder_fadvise_ignore(out_fd, (off_t)output_header_bytes,
                                    (off_t)n * (off_t)record_bytes, POSIX_FADV_SEQUENTIAL);
#ifdef POSIX_FADV_NOREUSE
        disk_reorder_fadvise_ignore(in_fd, input_header_bytes, (off_t)n * (off_t)record_bytes, POSIX_FADV_NOREUSE);
        disk_reorder_fadvise_ignore(out_fd, (off_t)output_header_bytes,
                                    (off_t)n * (off_t)record_bytes, POSIX_FADV_NOREUSE);
#endif
    }
    struct RewriteEntry {
        uint32_t orig;
        uint32_t local_row;
    };
    auto fill_records_sorted_by_orig = [&](const uint32_t* perm,
                                           int rows,
                                           char* records,
                                            std::vector<RewriteEntry>& entries,
                                            std::vector<char>& read_buffer,
                                            RewriteIoTraceStats* stats = nullptr) {
        auto build_pairs = [&]() {
            entries.resize((size_t)rows);
            for (int j = 0; j < rows; ++j) {
                uint32_t orig = perm[(size_t)j];
                if (orig >= (uint32_t)n) throw std::runtime_error("perm pid out of range while rewriting records");
                entries[(size_t)j] = RewriteEntry{orig, (uint32_t)j};
            }
        };
        if (stats) {
            stats->pair_build_ms += run_timed_ms(build_pairs);
        } else {
            build_pairs();
        }
        auto sort_pairs = [&]() {
            std::sort(entries.begin(), entries.end(), [](const RewriteEntry& a, const RewriteEntry& b) {
                return a.orig < b.orig;
            });
        };
        if (stats) {
            stats->sort_ms += run_timed_ms(sort_pairs);
        } else {
            sort_pairs();
        }
        auto read_sources = [&]() {
            int pos = 0;
            while (pos < rows) {
                uint32_t run_orig = entries[(size_t)pos].orig;
                int run_len = 1;
                while (pos + run_len < rows &&
                       entries[(size_t)pos + (size_t)run_len].orig == run_orig + (uint32_t)run_len) {
                    ++run_len;
                }
                if (run_len == 1) {
                    if (stats) ++stats->source_read_ops;
                    uint32_t local_row = entries[(size_t)pos].local_row;
                    disk_reorder_pread_all(in_fd, records + (size_t)local_row * record_bytes, record_bytes,
                                           input_header_bytes + (off_t)run_orig * (off_t)record_bytes,
                                           input_path);
                } else {
                    if (stats) ++stats->source_read_ops;
                    if (read_buffer.size() < (size_t)run_len * record_bytes) {
                        read_buffer.resize((size_t)run_len * record_bytes);
                    }
                    disk_reorder_pread_all(in_fd, read_buffer.data(), (size_t)run_len * record_bytes,
                                           input_header_bytes + (off_t)run_orig * (off_t)record_bytes,
                                           input_path);
                    for (int j = 0; j < run_len; ++j) {
                        uint32_t local_row = entries[(size_t)pos + (size_t)j].local_row;
                        std::memcpy(records + (size_t)local_row * record_bytes,
                                    read_buffer.data() + (size_t)j * record_bytes,
                                    record_bytes);
                    }
                }
                pos += run_len;
            }
        };
        if (stats) {
            stats->source_read_ms += run_timed_ms(read_sources);
        } else {
            read_sources();
        }
    };
#if SCORE_USE_LIBURING
    struct IoUringReadRequest {
        uint32_t orig;
        uint32_t local_row;
    };
    auto init_rewrite_io_uring = [&](io_uring& ring, unsigned queue_depth) -> bool {
        if (queue_depth == 0) return false;
        if (rewrite_io_uring_taskrun_flags_enabled()) {
            io_uring_params params{};
#ifdef IORING_SETUP_COOP_TASKRUN
            params.flags |= IORING_SETUP_COOP_TASKRUN;
#endif
#ifdef IORING_SETUP_SINGLE_ISSUER
            params.flags |= IORING_SETUP_SINGLE_ISSUER;
#endif
#ifdef IORING_SETUP_DEFER_TASKRUN
            params.flags |= IORING_SETUP_DEFER_TASKRUN;
#endif
#ifdef IORING_SETUP_CQSIZE
            params.flags |= IORING_SETUP_CQSIZE;
            params.cq_entries = std::max<unsigned>(queue_depth * 2, queue_depth + 32);
#endif
            if (io_uring_queue_init_params(queue_depth, &ring, &params) == 0) return true;
        }
        return io_uring_queue_init(queue_depth, &ring, 0) == 0;
    };

    auto register_rewrite_io_uring_file = [&](io_uring& ring) -> bool {
        if (!rewrite_io_uring_registered_files_enabled()) return false;
        int fds[1] = {in_fd};
        return io_uring_register_files(&ring, fds, 1) == 0;
    };

    auto cleanup_rewrite_io_uring = [&](io_uring& ring, bool files_registered) {
        if (files_registered) (void)io_uring_unregister_files(&ring);
        io_uring_queue_exit(&ring);
    };

    auto fill_records_io_uring_sorted_by_orig = [&](io_uring& ring,
                                                    unsigned queue_depth,
                                                    unsigned cqe_batch,
                                                    bool fixed_file,
                                                    const uint32_t* perm,
                                                    int rows,
                                                    char* records,
                                                    std::vector<RewriteEntry>& entries,
                                                     std::vector<IoUringReadRequest>& requests,
                                                     std::vector<io_uring_cqe*>& cqes,
                                                     RewriteIoTraceStats* stats = nullptr) {
        if (rows <= 0 || record_bytes > (size_t)std::numeric_limits<unsigned>::max()) return;
        auto build_pairs = [&]() {
            entries.resize((size_t)rows);
            requests.resize((size_t)rows);
            cqes.resize(std::max<unsigned>(1, std::min<unsigned>(cqe_batch, queue_depth)));
            for (int j = 0; j < rows; ++j) {
                uint32_t orig = perm[(size_t)j];
                if (orig >= (uint32_t)n) throw std::runtime_error("perm pid out of range while rewriting records");
                entries[(size_t)j] = RewriteEntry{orig, (uint32_t)j};
            }
        };
        if (stats) {
            stats->pair_build_ms += run_timed_ms(build_pairs);
        } else {
            build_pairs();
        }
        auto sort_pairs = [&]() {
            std::sort(entries.begin(), entries.end(), [](const RewriteEntry& a, const RewriteEntry& b) {
                return a.orig < b.orig;
            });
        };
        if (stats) {
            stats->sort_ms += run_timed_ms(sort_pairs);
        } else {
            sort_pairs();
        }
        auto read_sources = [&]() {
            size_t next = 0;
            size_t in_flight = 0;
            while (next < (size_t)rows || in_flight > 0) {
                while (next < (size_t)rows && in_flight < (size_t)queue_depth) {
                    io_uring_sqe* sqe = io_uring_get_sqe(&ring);
                    if (sqe == nullptr) break;
                    IoUringReadRequest& req = requests[next];
                    req.orig = entries[next].orig;
                    req.local_row = entries[next].local_row;
                    char* dst = records + (size_t)req.local_row * record_bytes;
                    off_t offset = input_header_bytes + (off_t)req.orig * (off_t)record_bytes;
                    io_uring_prep_read(sqe, fixed_file ? 0 : in_fd, dst, (unsigned)record_bytes, (uint64_t)offset);
                    if (fixed_file) sqe->flags |= IOSQE_FIXED_FILE;
                    io_uring_sqe_set_data(sqe, &req);
                    ++next;
                    ++in_flight;
                }
                int submit_rc = io_uring_submit_and_wait(&ring, in_flight > 0 ? 1 : 0);
                if (submit_rc < 0) throw std::runtime_error("io_uring_submit failed while rewriting records");
                if (stats && submit_rc > 0) stats->io_uring_submits += 1;
                if (in_flight == 0) continue;

                unsigned got = io_uring_peek_batch_cqe(&ring, cqes.data(), (unsigned)cqes.size());
                if (got == 0) {
                    io_uring_cqe* cqe = nullptr;
                    int wait_rc = io_uring_wait_cqe(&ring, &cqe);
                    if (wait_rc < 0) throw std::runtime_error("io_uring_wait_cqe failed while rewriting records");
                    cqes[0] = cqe;
                    got = 1;
                }
                if (stats) {
                    ++stats->io_uring_cqe_batches;
                    stats->io_uring_cqes += got;
                    stats->source_read_ops += got;
                }
                for (unsigned c = 0; c < got; ++c) {
                    io_uring_cqe* cqe = cqes[c];
                    IoUringReadRequest* req = static_cast<IoUringReadRequest*>(io_uring_cqe_get_data(cqe));
                    int res = cqe->res;
                    if (res < 0) throw std::runtime_error("io_uring read failed while rewriting records");
                    if ((size_t)res > record_bytes) {
                        throw std::runtime_error("io_uring read returned too many bytes while rewriting records");
                    }
                    if ((size_t)res < record_bytes) {
                        if (stats) ++stats->short_reads;
                        char* dst = records + (size_t)req->local_row * record_bytes + (size_t)res;
                        off_t offset = input_header_bytes + (off_t)req->orig * (off_t)record_bytes + (off_t)res;
                        disk_reorder_pread_all(in_fd, dst, record_bytes - (size_t)res, offset, input_path);
                    }
                    --in_flight;
                }
                io_uring_cq_advance(&ring, got);
            }
        };
        if (stats) {
            stats->source_read_ms += run_timed_ms(read_sources);
        } else {
            read_sources();
        }
    };
#endif
    bool use_sorted_reads = record_bytes >= 64;
    bool use_io_uring_reads = rewrite_io_uring_enabled() && use_sorted_reads;
    unsigned hw = std::thread::hardware_concurrency();
    int hardware_workers = (int)std::max(1u, hw == 0 ? 4u : hw);
    int max_workers = rewrite_workers > 0 ? rewrite_workers : 1;
    max_workers = std::min(max_workers, hardware_workers);
    max_workers = std::min(max_workers, n);
    size_t parallel_buffer_budget = parallel_buffer_budget_bytes != 0
        ? parallel_buffer_budget_bytes
        : rewrite_parallel_buffer_budget_bytes();
    size_t rows_per_worker = std::max<size_t>(1, std::min<size_t>(std::max<size_t>(1, io_rows),
        parallel_buffer_budget / (size_t)max_workers / std::max<size_t>(record_bytes + sizeof(uint32_t), 1)));
    if (max_workers <= 1 || n < 4096) {
        std::vector<uint32_t> perm_chunk(std::max<size_t>(1, io_rows));
        std::vector<char> record_chunk(perm_chunk.size() * record_bytes);
        std::vector<RewriteEntry> entries;
        std::vector<char> read_buffer;
        RewriteIoTraceStats serial_stats;
#if SCORE_USE_LIBURING
        io_uring ring;
        unsigned queue_depth = std::min<unsigned>(rewrite_io_uring_queue_depth(), (unsigned)std::max<size_t>(1, io_rows));
        unsigned cqe_batch = std::min<unsigned>(rewrite_io_uring_cqe_batch(), queue_depth);
        bool ring_ready = use_io_uring_reads &&
                          record_bytes <= (size_t)std::numeric_limits<unsigned>::max() &&
                          init_rewrite_io_uring(ring, queue_depth);
        bool fixed_file = ring_ready && register_rewrite_io_uring_file(ring);
        std::vector<IoUringReadRequest> requests;
        std::vector<io_uring_cqe*> cqes;
#endif
        try {
        for (int start = 0; start < n; start += (int)perm_chunk.size()) {
            int rows = std::min<int>((int)perm_chunk.size(), n - start);
            auto read_perm = [&]() {
                disk_reorder_pread_all(perm_fd, perm_chunk.data(), (size_t)rows * sizeof(uint32_t),
                                       (off_t)start * (off_t)sizeof(uint32_t), raw_perm_path);
            };
            if (trace_io) {
                serial_stats.perm_read_ms += run_timed_ms(read_perm);
            } else {
                read_perm();
            }
            if (record_chunk.size() < (size_t)rows * record_bytes) record_chunk.resize((size_t)rows * record_bytes);
            if (trace_io) {
                ++serial_stats.chunks;
                serial_stats.rows += (uint64_t)rows;
            }
            if (use_io_uring_reads) {
#if SCORE_USE_LIBURING
                if (ring_ready) {
                    fill_records_io_uring_sorted_by_orig(ring, queue_depth, cqe_batch, fixed_file,
                                                         perm_chunk.data(), rows, record_chunk.data(),
                                                         entries, requests, cqes, trace_io ? &serial_stats : nullptr);
                } else {
                    if (trace_io) ++serial_stats.fallback_chunks;
                    fill_records_sorted_by_orig(perm_chunk.data(), rows, record_chunk.data(), entries, read_buffer,
                                                trace_io ? &serial_stats : nullptr);
                }
#else
                if (trace_io) ++serial_stats.fallback_chunks;
                fill_records_sorted_by_orig(perm_chunk.data(), rows, record_chunk.data(), entries, read_buffer,
                                            trace_io ? &serial_stats : nullptr);
#endif
            } else if (use_sorted_reads) {
                fill_records_sorted_by_orig(perm_chunk.data(), rows, record_chunk.data(), entries, read_buffer,
                                            trace_io ? &serial_stats : nullptr);
            } else {
                auto read_sources = [&]() {
                    for (int j = 0; j < rows; ++j) {
                        if (trace_io) ++serial_stats.source_read_ops;
                        uint32_t orig = perm_chunk[(size_t)j];
                        if (orig >= (uint32_t)n) throw std::runtime_error("perm pid out of range while rewriting records");
                        disk_reorder_pread_all(in_fd, record_chunk.data() + (size_t)j * record_bytes, record_bytes,
                                               input_header_bytes + (off_t)orig * (off_t)record_bytes, input_path);
                    }
                };
                if (trace_io) {
                    serial_stats.source_read_ms += run_timed_ms(read_sources);
                } else {
                    read_sources();
                }
            }
            auto write_output = [&]() {
                disk_reorder_pwrite_all(out_fd, record_chunk.data(), (size_t)rows * record_bytes,
                                        (off_t)output_header_bytes + (off_t)start * (off_t)record_bytes, output_path);
            };
            if (trace_io) {
                serial_stats.output_write_ms += run_timed_ms(write_output);
            } else {
                write_output();
            }
            if (trace_io) ++serial_stats.output_writes;
        }
#if SCORE_USE_LIBURING
        if (ring_ready) cleanup_rewrite_io_uring(ring, fixed_file);
#endif
        } catch (...) {
#if SCORE_USE_LIBURING
            if (ring_ready) cleanup_rewrite_io_uring(ring, fixed_file);
#endif
            throw;
        }
        if (trace_io) {
            double elapsed_ms = rewrite_trace_timer.elapsed_ms();
            double seconds = elapsed_ms / 1000.0;
            double input_mib = (double)((long double)serial_stats.rows * (long double)record_bytes / (1024.0L * 1024.0L));
            double output_mib = (double)(((long double)serial_stats.rows * (long double)record_bytes +
                                         (long double)output_header_bytes) / (1024.0L * 1024.0L));
            std::cerr << "[RewriteIO] output=" << output_path
                      << " backend=" << (use_io_uring_reads ? "io_uring" : (use_sorted_reads ? "sorted-pread" : "pread"))
                      << " workers=1 rows_per_worker=" << perm_chunk.size()
                      << " record_bytes=" << record_bytes
                      << " chunks=" << serial_stats.chunks
                      << " rows=" << serial_stats.rows
                      << " input_mib=" << input_mib
                      << " output_mib=" << output_mib
                      << " input_gib_s=" << (seconds > 0.0 ? input_mib / 1024.0 / seconds : 0.0)
                      << " output_gib_s=" << (seconds > 0.0 ? output_mib / 1024.0 / seconds : 0.0)
                      << " source_read_ops=" << serial_stats.source_read_ops
                      << " uring_submits=" << serial_stats.io_uring_submits
                      << " uring_cqe_batches=" << serial_stats.io_uring_cqe_batches
                      << " uring_cqes=" << serial_stats.io_uring_cqes
                      << " short_reads=" << serial_stats.short_reads
                      << " fallback_chunks=" << serial_stats.fallback_chunks
                      << " output_writes=" << serial_stats.output_writes
                      << " perm_read_ms=" << serial_stats.perm_read_ms
                      << " pair_build_ms=" << serial_stats.pair_build_ms
                      << " sort_ms=" << serial_stats.sort_ms
                      << " source_read_ms=" << serial_stats.source_read_ms
                      << " output_write_ms=" << serial_stats.output_write_ms
                      << " elapsed_ms=" << elapsed_ms << std::endl;
        }
        return;
    }

    std::vector<std::thread> threads;
    threads.reserve((size_t)max_workers);
    std::vector<std::exception_ptr> exceptions((size_t)max_workers);
    std::vector<RewriteIoTraceStats> worker_stats((size_t)max_workers);
    int base_rows = n / max_workers;
    int remainder = n % max_workers;
    try {
        for (int worker = 0; worker < max_workers; ++worker) {
            int begin = worker * base_rows + std::min(worker, remainder);
            int end = begin + base_rows + (worker < remainder ? 1 : 0);
            threads.emplace_back([&, worker, begin, end]() {
                try {
                    std::vector<uint32_t> local_perm(rows_per_worker);
                    std::vector<char> local_records(rows_per_worker * record_bytes);
                    std::vector<RewriteEntry> local_entries;
                    std::vector<char> local_read_buffer;
                    RewriteIoTraceStats& stats = worker_stats[(size_t)worker];
#if SCORE_USE_LIBURING
                    io_uring local_ring;
                    unsigned local_queue_depth = std::min<unsigned>(rewrite_io_uring_queue_depth(),
                                                                    (unsigned)std::max<size_t>(1, rows_per_worker));
                    unsigned local_cqe_batch = std::min<unsigned>(rewrite_io_uring_cqe_batch(), local_queue_depth);
                    bool local_ring_ready = use_io_uring_reads &&
                                            record_bytes <= (size_t)std::numeric_limits<unsigned>::max() &&
                                            init_rewrite_io_uring(local_ring, local_queue_depth);
                    bool local_fixed_file = local_ring_ready && register_rewrite_io_uring_file(local_ring);
                    std::vector<IoUringReadRequest> local_requests;
                    std::vector<io_uring_cqe*> local_cqes;
#endif
                    try {
                        for (int start = begin; start < end; start += (int)rows_per_worker) {
                            int rows = std::min<int>((int)rows_per_worker, end - start);
                            auto read_perm = [&]() {
                                disk_reorder_pread_all(perm_fd, local_perm.data(), (size_t)rows * sizeof(uint32_t),
                                                       (off_t)start * (off_t)sizeof(uint32_t), raw_perm_path);
                            };
                            if (trace_io) {
                                stats.perm_read_ms += run_timed_ms(read_perm);
                            } else {
                                read_perm();
                            }
                            if (local_records.size() < (size_t)rows * record_bytes) {
                                local_records.resize((size_t)rows * record_bytes);
                            }
                            if (trace_io) {
                                ++stats.chunks;
                                stats.rows += (uint64_t)rows;
                            }
                            if (use_io_uring_reads) {
#if SCORE_USE_LIBURING
                                if (local_ring_ready) {
                                    fill_records_io_uring_sorted_by_orig(local_ring, local_queue_depth, local_cqe_batch,
                                                                         local_fixed_file, local_perm.data(), rows,
                                                                         local_records.data(), local_entries,
                                                                         local_requests, local_cqes,
                                                                         trace_io ? &stats : nullptr);
                                } else {
                                    if (trace_io) ++stats.fallback_chunks;
                                    fill_records_sorted_by_orig(local_perm.data(), rows, local_records.data(),
                                                                local_entries, local_read_buffer,
                                                                trace_io ? &stats : nullptr);
                                }
#else
                                if (trace_io) ++stats.fallback_chunks;
                                fill_records_sorted_by_orig(local_perm.data(), rows, local_records.data(),
                                                            local_entries, local_read_buffer,
                                                            trace_io ? &stats : nullptr);
#endif
                            } else if (use_sorted_reads) {
                                fill_records_sorted_by_orig(local_perm.data(), rows, local_records.data(),
                                                            local_entries, local_read_buffer,
                                                            trace_io ? &stats : nullptr);
                            } else {
                                auto read_sources = [&]() {
                                    for (int j = 0; j < rows; ++j) {
                                        if (trace_io) ++stats.source_read_ops;
                                        uint32_t orig = local_perm[(size_t)j];
                                        if (orig >= (uint32_t)n) throw std::runtime_error("perm pid out of range while rewriting records");
                                        disk_reorder_pread_all(in_fd, local_records.data() + (size_t)j * record_bytes,
                                                               record_bytes,
                                                               input_header_bytes + (off_t)orig * (off_t)record_bytes,
                                                               input_path);
                                    }
                                };
                                if (trace_io) {
                                    stats.source_read_ms += run_timed_ms(read_sources);
                                } else {
                                    read_sources();
                                }
                            }
                            auto write_output = [&]() {
                                disk_reorder_pwrite_all(out_fd, local_records.data(), (size_t)rows * record_bytes,
                                                        (off_t)output_header_bytes + (off_t)start * (off_t)record_bytes,
                                                        output_path);
                            };
                            if (trace_io) {
                                stats.output_write_ms += run_timed_ms(write_output);
                            } else {
                                write_output();
                            }
                            if (trace_io) ++stats.output_writes;
                        }
                    } catch (...) {
#if SCORE_USE_LIBURING
                        if (local_ring_ready) cleanup_rewrite_io_uring(local_ring, local_fixed_file);
#endif
                        throw;
                    }
#if SCORE_USE_LIBURING
                    if (local_ring_ready) cleanup_rewrite_io_uring(local_ring, local_fixed_file);
#endif
                } catch (...) {
                    exceptions[(size_t)worker] = std::current_exception();
                }
            });
        }
    } catch (...) {
        for (std::thread& thread : threads) {
            if (thread.joinable()) thread.join();
        }
        throw;
    }
    for (std::thread& thread : threads) {
        if (thread.joinable()) thread.join();
    }
    for (const std::exception_ptr& ex : exceptions) {
        if (ex) std::rethrow_exception(ex);
    }
    if (trace_io) {
        RewriteIoTraceStats total_stats;
        RewriteIoTraceStats max_stats;
        for (const RewriteIoTraceStats& stats : worker_stats) {
            add_trace_stats(total_stats, stats);
            max_stats.perm_read_ms = std::max(max_stats.perm_read_ms, stats.perm_read_ms);
            max_stats.pair_build_ms = std::max(max_stats.pair_build_ms, stats.pair_build_ms);
            max_stats.sort_ms = std::max(max_stats.sort_ms, stats.sort_ms);
            max_stats.source_read_ms = std::max(max_stats.source_read_ms, stats.source_read_ms);
            max_stats.output_write_ms = std::max(max_stats.output_write_ms, stats.output_write_ms);
        }
        double elapsed_ms = rewrite_trace_timer.elapsed_ms();
        double seconds = elapsed_ms / 1000.0;
        double input_mib = (double)((long double)total_stats.rows * (long double)record_bytes / (1024.0L * 1024.0L));
        double output_mib = (double)(((long double)total_stats.rows * (long double)record_bytes +
                                     (long double)output_header_bytes) / (1024.0L * 1024.0L));
        std::cerr << "[RewriteIO] output=" << output_path
                  << " backend=" << (use_io_uring_reads ? "io_uring" : (use_sorted_reads ? "sorted-pread" : "pread"))
                  << " workers=" << max_workers
                  << " rows_per_worker=" << rows_per_worker
                  << " record_bytes=" << record_bytes
                  << " chunks=" << total_stats.chunks
                  << " rows=" << total_stats.rows
                  << " input_mib=" << input_mib
                  << " output_mib=" << output_mib
                  << " input_gib_s=" << (seconds > 0.0 ? input_mib / 1024.0 / seconds : 0.0)
                  << " output_gib_s=" << (seconds > 0.0 ? output_mib / 1024.0 / seconds : 0.0)
                  << " source_read_ops=" << total_stats.source_read_ops
                  << " uring_submits=" << total_stats.io_uring_submits
                  << " uring_cqe_batches=" << total_stats.io_uring_cqe_batches
                  << " uring_cqes=" << total_stats.io_uring_cqes
                  << " short_reads=" << total_stats.short_reads
                  << " fallback_chunks=" << total_stats.fallback_chunks
                  << " output_writes=" << total_stats.output_writes
                  << " perm_read_ms=" << total_stats.perm_read_ms
                  << " pair_build_ms=" << total_stats.pair_build_ms
                  << " sort_ms=" << total_stats.sort_ms
                  << " source_read_ms=" << total_stats.source_read_ms
                  << " output_write_ms=" << total_stats.output_write_ms
                  << " max_perm_read_ms=" << max_stats.perm_read_ms
                  << " max_pair_build_ms=" << max_stats.pair_build_ms
                  << " max_sort_ms=" << max_stats.sort_ms
                  << " max_source_read_ms=" << max_stats.source_read_ms
                  << " max_output_write_ms=" << max_stats.output_write_ms
                  << " elapsed_ms=" << elapsed_ms << std::endl;
    }
}

inline void write_final_perm_to_orig_with_header(const std::string& raw_perm_path,
                                                 const std::string& final_perm_path,
                                                 int n,
                                                 size_t io_rows) {
    disk_reorder_expect_file_size(raw_perm_path, (off_t)n * (off_t)sizeof(uint32_t));
    int in_fd = disk_reorder_open_read(raw_perm_path);
    int out_fd = -1;
    std::string temp_perm_path;
    for (int attempt = 0; attempt < 1024; ++attempt) {
        temp_perm_path = disk_reorder_final_marker_temp_path(final_perm_path, attempt);
        out_fd = ::open(temp_perm_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
        if (out_fd >= 0) break;
        if (errno == EEXIST) continue;
        ::close(in_fd);
        throw std::runtime_error("cannot create temporary reorder marker: " + temp_perm_path);
    }
    if (out_fd < 0) {
        ::close(in_fd);
        throw std::runtime_error("cannot create unique temporary reorder marker for: " + final_perm_path);
    }
    std::vector<uint32_t> chunk(std::max<size_t>(1, io_rows));
    std::vector<uint64_t> seen(((size_t)n + 63) / 64, 0);
    size_t seen_count = 0;
    try {
        disk_reorder_write_all(out_fd, &n, sizeof(int), temp_perm_path);
        for (int start = 0; start < n; start += (int)chunk.size()) {
            int rows = std::min<int>((int)chunk.size(), n - start);
            disk_reorder_read_all(in_fd, chunk.data(), (size_t)rows * sizeof(uint32_t), raw_perm_path);
            for (int i = 0; i < rows; ++i) {
                uint32_t orig = chunk[(size_t)i];
                if (orig >= (uint32_t)n) {
                    throw std::runtime_error("raw perm_to_orig pid out of range while writing final marker");
                }
                size_t word = (size_t)orig >> 6;
                uint64_t bit = uint64_t{1} << (orig & 63u);
                if ((seen[word] & bit) != 0) {
                    throw std::runtime_error("duplicate pid in raw perm_to_orig while writing final marker");
                }
                seen[word] |= bit;
                ++seen_count;
            }
            disk_reorder_write_all(out_fd, chunk.data(), (size_t)rows * sizeof(uint32_t), temp_perm_path);
        }
        if (seen_count != (size_t)n) {
            throw std::runtime_error("missing pid in raw perm_to_orig while writing final marker");
        }
        disk_reorder_close_checked(out_fd, temp_perm_path);
        out_fd = -1;
        disk_reorder_close_checked(in_fd, raw_perm_path);
        in_fd = -1;
        disk_reorder_publish_final_marker_temp(temp_perm_path, final_perm_path);
    } catch (...) {
        if (in_fd >= 0) ::close(in_fd);
        if (out_fd >= 0) ::close(out_fd);
        if (!temp_perm_path.empty()) ::unlink(temp_perm_path.c_str());
        throw;
    }
}

inline void write_final_perm_to_orig_vector_atomic(const std::vector<uint32_t>& perm_to_orig,
                                                   const std::string& final_perm_path,
                                                   int n) {
    if (n < 0 || perm_to_orig.size() != (size_t)n) {
        throw std::runtime_error("invalid in-memory perm_to_orig size while writing final marker");
    }
    std::vector<uint64_t> seen(((size_t)n + 63) / 64, 0);
    for (uint32_t orig : perm_to_orig) {
        if (orig >= (uint32_t)n) {
            throw std::runtime_error("in-memory perm_to_orig pid out of range while writing final marker");
        }
        size_t word = (size_t)orig >> 6;
        uint64_t bit = uint64_t{1} << (orig & 63u);
        if ((seen[word] & bit) != 0) {
            throw std::runtime_error("duplicate pid in in-memory perm_to_orig while writing final marker");
        }
        seen[word] |= bit;
    }

    int out_fd = -1;
    std::string temp_perm_path;
    for (int attempt = 0; attempt < 1024; ++attempt) {
        temp_perm_path = disk_reorder_final_marker_temp_path(final_perm_path, attempt);
        out_fd = ::open(temp_perm_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
        if (out_fd >= 0) break;
        if (errno == EEXIST) continue;
        throw std::runtime_error("cannot create temporary reorder marker: " + temp_perm_path);
    }
    if (out_fd < 0) {
        throw std::runtime_error("cannot create unique temporary reorder marker for: " + final_perm_path);
    }
    try {
        disk_reorder_write_all(out_fd, &n, sizeof(int), temp_perm_path);
        disk_reorder_write_all(out_fd, perm_to_orig.data(), (size_t)n * sizeof(uint32_t), temp_perm_path);
        disk_reorder_close_checked(out_fd, temp_perm_path);
        out_fd = -1;
        disk_reorder_publish_final_marker_temp(temp_perm_path, final_perm_path);
    } catch (...) {
        if (out_fd >= 0) ::close(out_fd);
        if (!temp_perm_path.empty()) ::unlink(temp_perm_path.c_str());
        throw;
    }
}

inline void materialize_reordered_posting_from_temp(const std::string& temp_assign_path,
                                                    const std::string& output_path,
                                                    const std::string& raw_perm_path,
                                                    int n,
                                                    int K,
                                                    size_t io_rows) {
    if (K <= 0) throw std::runtime_error("posting K must be positive during reorder finalize");
    if (K > (int)std::numeric_limits<uint16_t>::max()) {
        throw std::runtime_error("posting K exceeds uint16 scratch limit during reorder finalize");
    }
    disk_reorder_expect_file_size(temp_assign_path, (off_t)n * (off_t)posting_assignment_cell_bytes());
    disk_reorder_expect_file_size(raw_perm_path, (off_t)n * (off_t)sizeof(uint32_t));
    int temp_fd = -1;
    int perm_fd = -1;
    int out_fd = -1;
    bool output_created = false;
    try {
        temp_fd = disk_reorder_open_read(temp_assign_path);
        perm_fd = disk_reorder_open_read(raw_perm_path);
        out_fd = disk_reorder_open_write_exclusive(output_path);
        output_created = true;
        std::vector<uint16_t> assignments((size_t)n);
        std::vector<int> sizes((size_t)K, 0);
        std::vector<uint32_t> perm_chunk(std::max<size_t>(1, io_rows));
        std::vector<uint16_t> assignment_chunk(perm_chunk.size());
        for (int start = 0; start < n; start += (int)perm_chunk.size()) {
            int rows = std::min<int>((int)perm_chunk.size(), n - start);
            disk_reorder_read_all(temp_fd, assignment_chunk.data(), (size_t)rows * sizeof(uint16_t), temp_assign_path);
            for (int i = 0; i < rows; ++i) {
                int cell = (int)assignment_chunk[(size_t)i];
                if (cell < 0 || cell >= K) throw std::runtime_error("temp assignment out of range while loading postings");
                assignments[(size_t)start + (size_t)i] = (uint16_t)cell;
                ++sizes[(size_t)cell];
            }
        }
        std::vector<int> offsets((size_t)K + 1, 0);
        for (int k = 0; k < K; ++k) offsets[(size_t)k + 1] = offsets[(size_t)k] + sizes[(size_t)k];
        if (offsets[(size_t)K] != n) throw std::runtime_error("posting reorder count mismatch");
        int total = n;
        off_t expected_output_size = (off_t)(2 * sizeof(int) + offsets.size() * sizeof(int) + (size_t)n * sizeof(uint32_t));
        disk_reorder_preallocate_output(out_fd, output_path, expected_output_size);
        disk_reorder_write_all(out_fd, &K, sizeof(int), output_path);
        disk_reorder_write_all(out_fd, &total, sizeof(int), output_path);
        disk_reorder_write_all(out_fd, offsets.data(), offsets.size() * sizeof(int), output_path);
        size_t ids_buffer_capacity = posting_materialize_ids_buffer_capacity(K);
        off_t ids_base = (off_t)(2 * sizeof(int) + offsets.size() * sizeof(int));
        if (::lseek(perm_fd, 0, SEEK_SET) < 0) throw std::runtime_error("failed seeking perm while filling postings");
        size_t bounded_ids_scratch = (size_t)K * ids_buffer_capacity * sizeof(uint32_t);
        size_t contiguous_ids_scratch = (size_t)n * sizeof(uint32_t);
        if (contiguous_ids_scratch <= bounded_ids_scratch) {
            std::vector<int> write_pos = offsets;
            std::vector<uint32_t> ids_by_cell((size_t)n);
            for (int start = 0; start < n; start += (int)perm_chunk.size()) {
                int rows = std::min<int>((int)perm_chunk.size(), n - start);
                disk_reorder_read_all(perm_fd, perm_chunk.data(), (size_t)rows * sizeof(uint32_t), raw_perm_path);
                for (int i = 0; i < rows; ++i) {
                    uint32_t orig = perm_chunk[(size_t)i];
                    if (orig >= (uint32_t)n) throw std::runtime_error("perm pid out of range while filling postings");
                    int cell = (int)assignments[(size_t)orig];
                    uint32_t new_pid = (uint32_t)(start + i);
                    int dst = write_pos[(size_t)cell]++;
                    ids_by_cell[(size_t)dst] = new_pid;
                }
            }
            for (int k = 0; k < K; ++k) {
                if (write_pos[(size_t)k] != offsets[(size_t)k + 1]) {
                    throw std::runtime_error("posting reorder cursor mismatch");
                }
            }
            disk_reorder_write_all(out_fd, ids_by_cell.data(), (size_t)n * sizeof(uint32_t), output_path);
        } else {
            std::vector<int> write_pos = offsets;
            std::vector<uint32_t> ids_buffers((size_t)K * ids_buffer_capacity);
            std::vector<uint16_t> ids_buffer_counts((size_t)K, 0);
            auto flush_cell_ids = [&](int cell) {
                uint16_t count = ids_buffer_counts[(size_t)cell];
                if (count == 0) return;
                int dst = write_pos[(size_t)cell];
                disk_reorder_pwrite_all(out_fd,
                                        ids_buffers.data() + (size_t)cell * ids_buffer_capacity,
                                        (size_t)count * sizeof(uint32_t),
                                        ids_base + (off_t)dst * (off_t)sizeof(uint32_t),
                                        output_path);
                write_pos[(size_t)cell] += (int)count;
                ids_buffer_counts[(size_t)cell] = 0;
            };
            for (int start = 0; start < n; start += (int)perm_chunk.size()) {
                int rows = std::min<int>((int)perm_chunk.size(), n - start);
                disk_reorder_read_all(perm_fd, perm_chunk.data(), (size_t)rows * sizeof(uint32_t), raw_perm_path);
                for (int i = 0; i < rows; ++i) {
                    uint32_t orig = perm_chunk[(size_t)i];
                    if (orig >= (uint32_t)n) throw std::runtime_error("perm pid out of range while filling postings");
                    int cell = (int)assignments[(size_t)orig];
                    uint32_t new_pid = (uint32_t)(start + i);
                    uint16_t& count = ids_buffer_counts[(size_t)cell];
                    ids_buffers[(size_t)cell * ids_buffer_capacity + (size_t)count] = new_pid;
                    ++count;
                    if ((size_t)count == ids_buffer_capacity) flush_cell_ids(cell);
                }
            }
            for (int k = 0; k < K; ++k) {
                flush_cell_ids(k);
                if (write_pos[(size_t)k] != offsets[(size_t)k + 1]) {
                    throw std::runtime_error("posting reorder cursor mismatch");
                }
            }
        }
        disk_reorder_close_checked(temp_fd, temp_assign_path);
        temp_fd = -1;
        disk_reorder_close_checked(perm_fd, raw_perm_path);
        perm_fd = -1;
        disk_reorder_close_checked(out_fd, output_path);
        out_fd = -1;
    } catch (...) {
        if (temp_fd >= 0) ::close(temp_fd);
        if (perm_fd >= 0) ::close(perm_fd);
        if (out_fd >= 0) ::close(out_fd);
        if (output_created) ::unlink(output_path.c_str());
        throw;
    }
}

inline int reconstruct_temp_assignments_from_posting_file(const std::string& posting_path,
                                                          const std::string& temp_assign_path,
                                                          int n,
                                                          size_t io_rows) {
    if (n < 0) throw std::runtime_error("invalid n while reconstructing temp assignments");
    int posting_fd = disk_reorder_open_read(posting_path);
    int temp_fd = disk_reorder_open_write_exclusive(temp_assign_path);
    if (temp_fd < 0) throw std::runtime_error("cannot create temp assignment output: " + temp_assign_path);
    constexpr uint16_t kMissingAssignment = std::numeric_limits<uint16_t>::max();
    std::vector<uint16_t> assignments((size_t)n, kMissingAssignment);
    try {
        int K = 0, total = 0;
        disk_reorder_read_all(posting_fd, &K, sizeof(int), posting_path);
        disk_reorder_read_all(posting_fd, &total, sizeof(int), posting_path);
        if (K <= 0 || total != n) throw std::runtime_error("invalid posting header while reconstructing assignments");
        if (K > (int)kMissingAssignment) {
            throw std::runtime_error("posting K exceeds uint16 assignment limit while reconstructing assignments");
        }
        std::vector<int> offsets((size_t)K + 1);
        disk_reorder_read_all(posting_fd, offsets.data(), offsets.size() * sizeof(int), posting_path);
        if (offsets.front() != 0 || offsets.back() != n) {
            throw std::runtime_error("invalid posting offsets while reconstructing assignments");
        }
        for (int k = 0; k < K; ++k) {
            if (offsets[(size_t)k] > offsets[(size_t)k + 1]) {
                throw std::runtime_error("non-monotonic posting offsets while reconstructing assignments");
            }
        }

        std::vector<uint32_t> ids(std::max<size_t>(1, io_rows));
        off_t ids_base = (off_t)(2 * sizeof(int) + ((size_t)K + 1) * sizeof(int));
        for (int cell = 0; cell < K; ++cell) {
            int begin = offsets[(size_t)cell];
            int end = offsets[(size_t)cell + 1];
            for (int pos = begin; pos < end; pos += (int)ids.size()) {
                int count = std::min<int>((int)ids.size(), end - pos);
                disk_reorder_pread_all(posting_fd, ids.data(), (size_t)count * sizeof(uint32_t),
                                       ids_base + (off_t)pos * (off_t)sizeof(uint32_t), posting_path);
                for (int i = 0; i < count; ++i) {
                    uint32_t orig = ids[(size_t)i];
                    if (orig >= (uint32_t)n) {
                        throw std::runtime_error("posting id out of range while reconstructing assignments");
                    }
                    if (assignments[(size_t)orig] != kMissingAssignment) {
                        throw std::runtime_error("duplicate posting id while reconstructing assignments");
                    }
                    assignments[(size_t)orig] = (uint16_t)cell;
                }
            }
        }

        std::vector<uint16_t> assignment_chunk(ids.size());
        for (int start = 0; start < n; start += (int)ids.size()) {
            int rows = std::min<int>((int)ids.size(), n - start);
            for (int i = 0; i < rows; ++i) {
                uint16_t assignment = assignments[(size_t)start + (size_t)i];
                if (assignment == kMissingAssignment || (int)assignment >= K) {
                    throw std::runtime_error("missing posting id while reconstructing assignments");
                }
                assignment_chunk[(size_t)i] = assignment;
            }
            disk_reorder_write_all(temp_fd, assignment_chunk.data(), (size_t)rows * sizeof(uint16_t), temp_assign_path);
        }

        disk_reorder_close_checked(posting_fd, posting_path);
        disk_reorder_close_checked(temp_fd, temp_assign_path);
        return K;
    } catch (...) {
        ::close(posting_fd);
        ::close(temp_fd);
        ::unlink(temp_assign_path.c_str());
        throw;
    }
}

inline void finalize_reordered_artifacts_from_staging(const ReorderedArtifactFinalizeOptions& options) {
    if (options.n < 0 || options.M <= 0) throw std::runtime_error("invalid reorder finalize dimensions");
    if ((int)options.cells_per_subspace.size() != options.M ||
        (int)options.subspace_assignment_paths.size() != options.M ||
        (int)options.posting_output_paths.size() != options.M) {
        throw std::runtime_error("reorder finalize metadata size mismatch");
    }
    if (options.final_perm_to_orig_path.empty()) throw std::runtime_error("final perm path is required");
    if (::access(options.final_perm_to_orig_path.c_str(), F_OK) == 0) {
        throw std::runtime_error("final perm output already exists");
    }
    std::vector<std::string> created;
    std::vector<uint8_t> posting_outputs_created((size_t)options.M, 0);
    bool completed = false;
    auto cleanup_outputs = [&]() noexcept {
        if (completed) return;
        for (const std::string& path : created) ::unlink(path.c_str());
        for (int m = 0; m < options.M; ++m) {
            if (posting_outputs_created[(size_t)m] != 0) {
                ::unlink(options.posting_output_paths[(size_t)m].c_str());
            }
        }
    };
    try {
        int verify_n = 0, verify_dim = 0, reserved = 0;
        {
            int fd = disk_reorder_open_read(options.verify_input_path);
            disk_reorder_read_all(fd, &verify_n, sizeof(int), options.verify_input_path);
            disk_reorder_read_all(fd, &verify_dim, sizeof(int), options.verify_input_path);
            disk_reorder_read_all(fd, &reserved, sizeof(int), options.verify_input_path);
            disk_reorder_close_checked(fd, options.verify_input_path);
        }
        if (verify_n != options.n || verify_dim <= 0 || reserved != 0) {
            throw std::runtime_error("invalid verify_vectors.fbin header for reorder finalize");
        }
        int verify_header[3] = {options.n, verify_dim, 0};
        size_t verify_record_bytes = (size_t)verify_dim * sizeof(float);
        auto verify_trace = begin_finalize_phase_trace();
        rewrite_fixed_records_by_perm(options.verify_input_path, options.verify_output_path,
                                      options.raw_perm_to_orig_path, options.n, 3 * (off_t)sizeof(int),
                                      verify_header, sizeof(verify_header),
                                      verify_record_bytes, options.io_rows,
                                      options.rewrite_workers,
                                      options.max_cached_record_bytes,
                                      options.parallel_buffer_budget_bytes);
        trace_finalize_phase("verify_rewrite", verify_trace);
        created.push_back(options.verify_output_path);

        disk_reorder_expect_file_size(options.assignments_input_path,
                                      (off_t)sizeof(int) + (off_t)options.n * (off_t)sizeof(uint32_t));
        int assign_header = options.n;
        auto assignments_trace = begin_finalize_phase_trace();
        rewrite_fixed_records_by_perm(options.assignments_input_path, options.assignments_output_path,
                                      options.raw_perm_to_orig_path, options.n, sizeof(int),
                                      &assign_header, sizeof(assign_header),
                                      sizeof(uint32_t), options.io_rows,
                                      options.rewrite_workers,
                                      options.max_cached_record_bytes,
                                      options.parallel_buffer_budget_bytes);
        trace_finalize_phase("assignments_rewrite", assignments_trace);
        created.push_back(options.assignments_output_path);

        int bin_n = 0, bin_bytes = 0;
        {
            int fd = disk_reorder_open_read(options.bin_data_input_path);
            disk_reorder_read_all(fd, &bin_n, sizeof(int), options.bin_data_input_path);
            disk_reorder_read_all(fd, &bin_bytes, sizeof(int), options.bin_data_input_path);
            disk_reorder_close_checked(fd, options.bin_data_input_path);
        }
        if (bin_n != options.n || bin_bytes <= 0) {
            throw std::runtime_error("invalid rabitq_bin_data.bin header for reorder finalize");
        }
        int bin_header[2] = {options.n, bin_bytes};
        auto bin_trace = begin_finalize_phase_trace();
        rewrite_fixed_records_by_perm(options.bin_data_input_path, options.bin_data_output_path,
                                      options.raw_perm_to_orig_path, options.n, 2 * (off_t)sizeof(int),
                                      bin_header, sizeof(bin_header),
                                      (size_t)bin_bytes, options.io_rows,
                                      options.rewrite_workers,
                                      options.max_cached_record_bytes,
                                      options.parallel_buffer_budget_bytes);
        trace_finalize_phase("bin_data_rewrite", bin_trace);
        created.push_back(options.bin_data_output_path);

        auto postings_trace = begin_finalize_phase_trace();
        for (int m = 0; m < options.M; ++m) {
            if (options.cells_per_subspace[(size_t)m] > (size_t)std::numeric_limits<uint16_t>::max()) {
                throw std::runtime_error("posting K exceeds uint16 assignment format during reorder finalize");
            }
            for (int j = m + 1; j < options.M; ++j) {
                if (options.posting_output_paths[(size_t)m] == options.posting_output_paths[(size_t)j]) {
                    throw std::runtime_error("duplicate posting output path during reorder finalize");
                }
            }
        }
        int posting_workers = std::max(1, options.M);
        if (posting_workers <= 1) {
            for (int m = 0; m < options.M; ++m) {
                auto posting_trace = begin_finalize_phase_trace();
                materialize_reordered_posting_from_temp(options.subspace_assignment_paths[(size_t)m],
                                                        options.posting_output_paths[(size_t)m],
                                                        options.raw_perm_to_orig_path,
                                                        options.n,
                                                        (int)options.cells_per_subspace[(size_t)m],
                                                        options.io_rows);
                posting_outputs_created[(size_t)m] = 1;
                trace_finalize_phase("posting_materialize_" + std::to_string(m), posting_trace);
            }
        } else {
            std::atomic<int> next_posting{0};
            std::vector<std::thread> posting_threads;
            std::vector<std::exception_ptr> posting_errors((size_t)posting_workers);
            posting_threads.reserve((size_t)posting_workers);
            try {
                for (int worker = 0; worker < posting_workers; ++worker) {
                    posting_threads.emplace_back([&, worker]() {
                        try {
                            while (true) {
                                int m = next_posting.fetch_add(1);
                                if (m >= options.M) break;
                                materialize_reordered_posting_from_temp(options.subspace_assignment_paths[(size_t)m],
                                                                        options.posting_output_paths[(size_t)m],
                                                                        options.raw_perm_to_orig_path,
                                                                        options.n,
                                                                        (int)options.cells_per_subspace[(size_t)m],
                                                                        options.io_rows);
                                posting_outputs_created[(size_t)m] = 1;
                            }
                        } catch (...) {
                            posting_errors[(size_t)worker] = std::current_exception();
                        }
                    });
                }
            } catch (...) {
                for (std::thread& thread : posting_threads) {
                    if (thread.joinable()) thread.join();
                }
                throw;
            }
            for (std::thread& thread : posting_threads) {
                if (thread.joinable()) thread.join();
            }
            for (const std::exception_ptr& error : posting_errors) {
                if (error) std::rethrow_exception(error);
            }
        }
        trace_finalize_phase("postings_materialize", postings_trace);

        auto final_perm_trace = begin_finalize_phase_trace();
        write_final_perm_to_orig_with_header(options.raw_perm_to_orig_path,
                                             options.final_perm_to_orig_path,
                                             options.n,
                                             options.io_rows);
        completed = true;
        trace_finalize_phase("final_perm_write", final_perm_trace);
    } catch (...) {
        cleanup_outputs();
        throw;
    }
}

inline void posthoc_reorder_no_reorder_index(const PosthocReorderIndexOptions& options) {
    if (options.n <= 0 || options.M <= 0) throw std::runtime_error("invalid posthoc reorder dimensions");
    if (options.source_index_dir.empty() || options.output_index_dir.empty() ||
        options.source_index_dir == options.output_index_dir) {
        throw std::runtime_error("posthoc reorder source/output dirs must be distinct");
    }
    if (disk_reorder_path_exists(options.source_index_dir + "/perm_to_orig.bin")) {
        throw std::runtime_error("posthoc reorder source index already has perm_to_orig.bin");
    }
    if (disk_reorder_path_exists(options.source_index_dir + "/ivf_subspace_" + std::to_string(options.M) + ".bin") ||
        disk_reorder_path_exists(options.source_index_dir + "/posting_subspace_" + std::to_string(options.M) + ".bin")) {
        throw std::runtime_error("posthoc reorder M is smaller than source index subspace count");
    }
    std::vector<std::string> created;
    bool completed = false;
    auto cleanup = [&]() noexcept {
        if (completed) return;
        for (const std::string& path : created) ::unlink(path.c_str());
    };
    auto copy_artifact = [&](const std::string& name) {
        std::string src = options.source_index_dir + "/" + name;
        std::string dst = options.output_index_dir + "/" + name;
        copy_file_exclusive(src, dst);
        created.push_back(dst);
    };

    try {
        copy_artifact("rotation.bin");
        if (disk_reorder_path_exists(options.source_index_dir + "/rotate_subspace.flag")) {
            copy_artifact("rotate_subspace.flag");
            if (disk_reorder_path_exists(options.source_index_dir + "/subspace_rotation.bin")) {
                copy_artifact("subspace_rotation.bin");
            }
        }
        copy_artifact("rabitq_centroids.bin");
        for (int m = 0; m < options.M; ++m) {
            copy_artifact("ivf_subspace_" + std::to_string(m) + ".bin");
        }

        PreservedPostingAssignments preserved_assignments =
            load_preserved_posting_assignments(options.source_index_dir, options.n, options.M);
        std::vector<std::string> temp_assignment_paths((size_t)options.M);
        std::vector<size_t> cells_per_subspace((size_t)options.M);
        std::vector<std::string> output_temp_assignment_paths;
        if (preserved_assignments.available) {
            temp_assignment_paths = preserved_assignments.paths;
            cells_per_subspace = preserved_assignments.cells_per_subspace;
        } else {
            for (int m = 0; m < options.M; ++m) {
                temp_assignment_paths[(size_t)m] = options.output_index_dir + "/posting_assign_subspace_" +
                                                   std::to_string(m) + ".reorder.tmp";
                int K = reconstruct_temp_assignments_from_posting_file(
                    options.source_index_dir + "/posting_subspace_" + std::to_string(m) + ".bin",
                    temp_assignment_paths[(size_t)m], options.n, options.io_rows);
                cells_per_subspace[(size_t)m] = (size_t)K;
                created.push_back(temp_assignment_paths[(size_t)m]);
                output_temp_assignment_paths.push_back(temp_assignment_paths[(size_t)m]);
            }
        }

        const std::string raw_perm_path = options.output_index_dir + "/perm_to_orig.raw.tmp";
        const std::string scratch_perm_path = options.output_index_dir + "/perm_to_orig.scratch.tmp";
        impl::DiskPidReorderOptions reorder_options;
        reorder_options.n = options.n;
        reorder_options.M = options.M;
        reorder_options.assignment_key_range = (size_t)std::max(1, options.config.rabitq_k);
        reorder_options.cells_per_subspace = cells_per_subspace;
        reorder_options.subspace_assignment_paths = temp_assignment_paths;
        reorder_options.rabitq_assignments_path = options.source_index_dir + "/rabitq_assignments.bin";
        reorder_options.perm_to_orig_path = raw_perm_path;
        reorder_options.scratch_perm_path = scratch_perm_path;
        reorder_options.config = options.config;
        reorder_options.io_rows = options.io_rows;
        reorder_options.bucket_buffer_rows = options.bucket_buffer_rows;
        build_pid_reorder_perm_disk_staging(reorder_options);
        created.push_back(raw_perm_path);
        created.push_back(scratch_perm_path);

        ReorderedArtifactFinalizeOptions finalize_options;
        finalize_options.n = options.n;
        finalize_options.M = options.M;
        finalize_options.cells_per_subspace = cells_per_subspace;
        finalize_options.subspace_assignment_paths = temp_assignment_paths;
        finalize_options.raw_perm_to_orig_path = raw_perm_path;
        finalize_options.verify_input_path = options.source_index_dir + "/verify_vectors.fbin";
        finalize_options.assignments_input_path = options.source_index_dir + "/rabitq_assignments.bin";
        finalize_options.bin_data_input_path = options.source_index_dir + "/rabitq_bin_data.bin";
        finalize_options.verify_output_path = options.output_index_dir + "/verify_vectors.fbin";
        finalize_options.assignments_output_path = options.output_index_dir + "/rabitq_assignments.bin";
        finalize_options.bin_data_output_path = options.output_index_dir + "/rabitq_bin_data.bin";
        for (int m = 0; m < options.M; ++m) {
            finalize_options.posting_output_paths.push_back(options.output_index_dir + "/posting_subspace_" +
                                                            std::to_string(m) + ".bin");
        }
        finalize_options.final_perm_to_orig_path = options.output_index_dir + "/perm_to_orig.bin";
        finalize_options.io_rows = options.io_rows;
        finalize_options.rewrite_workers = options.rewrite_workers;
        finalize_reordered_artifacts_from_staging(finalize_options);
        completed = true;

        for (const std::string& path : output_temp_assignment_paths) ::unlink(path.c_str());
        ::unlink(raw_perm_path.c_str());
        ::unlink(scratch_perm_path.c_str());
    } catch (...) {
        cleanup();
        throw;
    }
}

inline void build_pid_reorder_perm_disk_staging(const DiskPidReorderOptions& options) {
    if (options.n < 0 || options.M <= 0) throw std::runtime_error("invalid disk reorder dimensions");
    if ((int)options.cells_per_subspace.size() != options.M ||
        (int)options.subspace_assignment_paths.size() != options.M) {
        throw std::runtime_error("disk reorder subspace metadata size mismatch");
    }
    if (options.perm_to_orig_path.empty() || options.scratch_perm_path.empty() ||
        options.perm_to_orig_path == options.scratch_perm_path) {
        throw std::runtime_error("disk reorder output and scratch paths must be distinct");
    }
    if (!options.orig_to_new_path.empty() &&
        (options.orig_to_new_path == options.perm_to_orig_path ||
         options.orig_to_new_path == options.scratch_perm_path)) {
        throw std::runtime_error("disk reorder inverse path must be distinct");
    }
    if (options.assignment_key_range == 0) {
        throw std::runtime_error("disk reorder assignment key range must be positive");
    }
    for (size_t range : options.cells_per_subspace) {
        if (range == 0) throw std::runtime_error("disk reorder cell key ranges must be positive");
    }
    if (::access(options.perm_to_orig_path.c_str(), F_OK) == 0 ||
        ::access(options.scratch_perm_path.c_str(), F_OK) == 0 ||
        (!options.orig_to_new_path.empty() && ::access(options.orig_to_new_path.c_str(), F_OK) == 0)) {
        throw std::runtime_error("disk reorder staging output already exists");
    }

    bool completed = false;
    auto cleanup_outputs = [&]() noexcept {
        if (completed) return;
        ::unlink(options.perm_to_orig_path.c_str());
        ::unlink(options.scratch_perm_path.c_str());
        if (!options.orig_to_new_path.empty()) ::unlink(options.orig_to_new_path.c_str());
    };

    try {
        std::vector<int> key_subspaces = normalize_pid_reorder_subspaces(options.config, options.M);
        int key_depth = pid_reorder_key_depth(options.config, key_subspaces);
        for (int i = 0; i < key_depth; ++i) {
            int subspace = key_subspaces[(size_t)i];
            if (subspace < 0 || subspace >= options.M) {
                throw std::runtime_error("pid reorder subspace out of range for disk staging");
            }
        }

        initialize_identity_perm_file(options.perm_to_orig_path, options.n, options.io_rows);
        bool current_is_output = true;
        auto run_assignment_pass = [&]() {
            std::vector<uint32_t> keys = load_reorder_assignment_key_column(
                options.rabitq_assignments_path, options.n, options.assignment_key_range);
            const std::string& src = current_is_output ? options.perm_to_orig_path : options.scratch_perm_path;
            const std::string& dst = current_is_output ? options.scratch_perm_path : options.perm_to_orig_path;
            stable_counting_pass_disk(src, dst, options.n, keys, options.assignment_key_range,
                                      options.io_rows, options.bucket_buffer_rows);
            current_is_output = !current_is_output;
        };
        auto run_cell_pass = [&](int subspace) {
            std::vector<uint32_t> keys = load_reorder_cell_key_column(
                options.subspace_assignment_paths[(size_t)subspace], options.n,
                options.cells_per_subspace[(size_t)subspace]);
            const std::string& src = current_is_output ? options.perm_to_orig_path : options.scratch_perm_path;
            const std::string& dst = current_is_output ? options.scratch_perm_path : options.perm_to_orig_path;
            stable_counting_pass_disk(src, dst, options.n, keys,
                                      options.cells_per_subspace[(size_t)subspace],
                                      options.io_rows, options.bucket_buffer_rows);
            current_is_output = !current_is_output;
        };

        if (options.config.pid_reorder_key_mode == 0) {
            for (int i = key_depth - 1; i >= 0; --i) run_cell_pass(key_subspaces[(size_t)i]);
            run_assignment_pass();
        } else if (options.config.pid_reorder_key_mode == 1) {
            run_assignment_pass();
            for (int i = key_depth - 1; i >= 0; --i) run_cell_pass(key_subspaces[(size_t)i]);
        } else if (key_depth > 0) {
            for (int i = key_depth - 1; i >= 0; --i) {
                run_assignment_pass();
                run_cell_pass(key_subspaces[(size_t)i]);
            }
        }

        if (!current_is_output) {
            copy_reorder_file(options.scratch_perm_path, options.perm_to_orig_path, options.n, options.io_rows);
        }
        if (!options.orig_to_new_path.empty()) {
            write_orig_to_new_from_perm(options.perm_to_orig_path, options.orig_to_new_path,
                                        options.n, options.io_rows);
        }
        completed = true;
    } catch (...) {
        cleanup_outputs();
        throw;
    }
}
} // namespace impl

// Index builder: offline construction pipeline
//
// Two modes:
// - Legacy (rabitq_k == 0): global centroid + Hadamard rotation + per-vector binary codes
// - RaBitQ-clustered (rabitq_k > 0): SuperKMeans clustering + FhtKac rotation + per-cluster residual quantization
class IndexBuilder {
public:
    void build(std::vector<float>& data, int n, int dim, const IndexConfig& config) {
        Timer total_timer;
        total_timer.reset();

        std::string path = config.index_path;
        mkdir(path.c_str(), 0755);

        int M = config.num_subspaces;

        if (config.rabitq_k <= 0) {
            std::cerr << "[Build] Error: -rabitq_k is required (e.g. -rabitq_k 200)" << std::endl;
            return;
        }

        build_rabitq_clustered(data, n, dim, config);

        std::cout << "\n[Build] Complete! Total: " << total_timer.elapsed_ms() / 1000.0 << " s"
                  << "\n  Index: " << path
                  << "\n  N=" << n << " D=" << dim << " M=" << M
                  << std::endl;
    }

private:
    // ========== RaBitQ-clustered mode: SuperKMeans + FhtKac + per-cluster residual ==========
    void build_rabitq_clustered(std::vector<float>& data, int n, int dim, const IndexConfig& config) {
        Timer step_timer;
        std::string path = config.index_path;
        int K = config.rabitq_k;

        std::cout << "[Build] RaBitQ-clustered: K=" << K
                  << " (FhtKac + HSKM in rotated space)" << std::endl;

        // Step 1: Init FhtKac rotation (shared by clustering + RaBitQ + IVF subspace)
        // One rotation serves three purposes ("一石三鸟"):
        //   1. K-means clustering quality (decorrelated dimensions)
        //   2. RaBitQ residual quantization (binary codes)
        //   3. Subspace IVF partitioning (balanced subspaces)
        std::cout << "[Build] Step 1: FhtKacRotator..." << std::endl;
        step_timer.reset();

        int padded_dim = RandomRotation::padding_requirement(dim, RotationType::FhtKac);
        RandomRotation rotation;
        rotation.init(dim, 42, RotationType::FhtKac);
        rotation.save(path + "/rotation.bin");

        std::cout << "  Rotation init (padded=" << padded_dim << ") in " << step_timer.elapsed_ms() << " ms" << std::endl;

        // Step 2: Rotate ALL data once (shared by clustering + RaBitQ + IVF subspace)
        // Produces two views: padded-stride for RaBitQ, dim-stride for clustering + IVF.
        std::cout << "[Build] Step 2: Rotate all vectors..." << std::endl;
        step_timer.reset();

        // Barrier A owner/view split:
        // - `rot_padded` is the full padded rotated backing produced by Step 2.
        // - `dim_stride_rows` is only the dim-stride consumer view used by clustering / IVF.
        // - when padded_dim != dim, that dim-stride view is backed by `data` storage, which
        //   means `data` no longer carries raw-input semantics after Step 2.
        impl::SharedRotationViews rotation_views;
        rotation_views.rot_padded.resize((size_t)n * padded_dim);
        if (padded_dim == dim) {
            rotation_views.dim_stride_rows = rotation_views.rot_padded.data();
            rotation_views.dim_stride_backing = impl::RotatedDimStrideBacking::RotPaddedAlias;
        } else {
            rotation_views.dim_stride_rows = data.data();
            rotation_views.dim_stride_backing = impl::RotatedDimStrideBacking::ReusedDataStorage;
        }

        #pragma omp parallel
        {
            std::vector<float> tmp((size_t)rotation.padded_dim(), 0.0f);
            #pragma omp for schedule(dynamic, 256)
            for (int i = 0; i < n; i++) {
                rotation.apply(data.data() + (size_t)i * dim, tmp.data());
                // Copy full padded result for quantization / verify-facing consumers.
                for (int d = 0; d < padded_dim; d++)
                    rotation_views.rot_padded[(size_t)i * padded_dim + d] = tmp[d];
                if (!rotation_views.dim_stride_aliases_rot_padded()) {
                    // Copy first dim elements for clustering / IVF-facing dim-stride consumers.
                    // On this path, `data` storage stops carrying raw-input semantics and becomes
                    // the dim-stride rotated backing.
                    for (int d = 0; d < dim; d++)
                        rotation_views.dim_stride_rows[(size_t)i * dim + d] = tmp[d];
                }
            }
        }
        std::cout << "  Full rotation done in " << step_timer.elapsed_ms() << " ms" << std::endl;

        // Step 3: K-means on ROTATED data
        // Clustering in rotated space: decorrelated dimensions improve convergence.
        // Centroids are produced directly in rotated space (no separate rotation needed).
        std::cout << "[Build] Step 3: Clustering K=" << K
                  << " seed=" << config.rabitq_seed << " (rotated space)..." << std::endl;
        step_timer.reset();

        std::vector<float> centroids((size_t)K * dim);
        std::vector<uint32_t> assignments(n);

        {
            IVFIndex temp_ivf;
            temp_ivf.init(dim, K, 0);
            // train() → HSKM: hierarchical meso+fine clustering, better centroid quality
            // than Forgy-init train_blas(). data_already_rotated=true set inside train_hskm().
            temp_ivf.train(rotation_views.dim_stride_rows, n, 10, (uint64_t)config.rabitq_seed);
            centroids.assign(temp_ivf.centroids().begin(), temp_ivf.centroids().end());

            // Assign all points to nearest centroid
            std::vector<int> int_assignments(n);
            temp_ivf.assign_batch(rotation_views.dim_stride_rows, int_assignments.data(), n);
            for (int i = 0; i < n; i++)
                assignments[i] = (uint32_t)int_assignments[i];
        }
        std::cout << "  Clustering done in " << step_timer.elapsed_ms() / 1000.0 << " s" << std::endl;

        // Step 4: Zero-pad rotated centroids to padded_dim
        // Centroids from Step 3 are already in rotated space, just need padding.
        std::vector<float> rot_centroids((size_t)K * padded_dim, 0);
        for (int k = 0; k < K; k++)
            for (int d = 0; d < dim; d++)
                rot_centroids[(size_t)k * padded_dim + d] = centroids[(size_t)k * dim + d];
        std::vector<float>().swap(centroids);

        // Step 5: Per-point residual quantization using library's quantize_compact_one_bit
        // This guarantees 100% compatibility with split_single_estdist at search time.
        std::cout << "[Build] Step 5: Residual quantization (" << n << " points)..." << std::endl;
        step_timer.reset();

        size_t bin_bytes = rabitqlib::BinDataMap<float>::data_bytes(padded_dim);
        bool will_pid_reorder = config.pid_reorder;

        if (!will_pid_reorder) {
            std::ofstream out(path + "/rabitq_bin_data.bin", std::ios::binary);
            int nn = n;
            int bb = (int)(sizeof(uint32_t) + bin_bytes);
            out.write(reinterpret_cast<const char*>(&nn), sizeof(int));
            out.write(reinterpret_cast<const char*>(&bb), sizeof(int));

            constexpr int kQuantChunk = 1 << 14;
            std::vector<char> chunk_data((size_t)kQuantChunk * (size_t)bb);
            for (int start = 0; start < n; start += kQuantChunk) {
                int chunk_n = std::min(kQuantChunk, n - start);
                #pragma omp parallel for schedule(dynamic, 256)
                for (int j = 0; j < chunk_n; j++) {
                    int i = start + j;
                    uint32_t cid = assignments[i];
                    const float* rot_vec = rotation_views.rot_padded.data() + (size_t)i * padded_dim;
                    const float* c_rot = rot_centroids.data() + (size_t)cid * padded_dim;

                    char* out_ptr = chunk_data.data() + (size_t)j * bb;
                    std::memcpy(out_ptr, &cid, sizeof(uint32_t));
                    rabitqlib::quant::quantize_compact_one_bit<float>(
                        rot_vec, c_rot, padded_dim,
                        out_ptr + sizeof(uint32_t),
                        rabitqlib::METRIC_L2);
                }
                out.write(chunk_data.data(), (size_t)chunk_n * (size_t)bb);
            }
        }
        if (will_pid_reorder) {
            std::cout << "  Quantization deferred to PID reorder finalize ("
                      << (sizeof(uint32_t) + bin_bytes) << " bytes/pt embedded)" << std::endl;
        } else {
            std::cout << "  Quantization done (" << (sizeof(uint32_t) + bin_bytes)
                      << " bytes/pt embedded) in "
                      << step_timer.elapsed_ms() << " ms" << std::endl;
        }

        // Save rotated centroids
        {
            std::ofstream out(path + "/rabitq_centroids.bin", std::ios::binary);
            int kk = K, dd = dim, pd = padded_dim;
            out.write(reinterpret_cast<const char*>(&kk), sizeof(int));
            out.write(reinterpret_cast<const char*>(&dd), sizeof(int));
            out.write(reinterpret_cast<const char*>(&pd), sizeof(int));
            out.write(reinterpret_cast<const char*>(rot_centroids.data()),
                      (size_t)K * padded_dim * sizeof(float));
        }

        // Save cluster assignments on non-reorder builds only.
        // Reorder builds emit the final ordered file later, after perm_to_orig is known.
        if (!will_pid_reorder) {
            std::ofstream out(path + "/rabitq_assignments.bin", std::ios::binary);
            out.write(reinterpret_cast<const char*>(&n), sizeof(int));
            out.write(reinterpret_cast<const char*>(assignments.data()),
                      (size_t)n * sizeof(uint32_t));
        }

        std::cout << "\n[Build] RaBitQ-clustered files:"
                  << "\n  rabitq_centroids.bin: " << (double)K * padded_dim * sizeof(float) / 1e6 << " MB"
                  << "\n  rabitq_assignments.bin: " << (double)n * sizeof(uint32_t) / 1e6 << " MB"
                  << "\n  rabitq_bin_data.bin: " << (double)n * (sizeof(uint32_t) + bin_bytes) / 1e6 << " MB (embedded)"
                  << std::endl;

        // Step 6: IVF subspace index + posting lists (reuse the Step-2 dim-stride rotated view)
        build_subspace_and_postings(data, n, dim, config.num_subspaces, dim / config.num_subspaces,
                                  config, path, assignments,
                                    bin_bytes, rotation_views.dim_stride_rows, &rotation_views.rot_padded,
                                    rot_centroids.data(), padded_dim);

        std::vector<float>().swap(rot_centroids);

        if (!rotation_views.dim_stride_aliases_rot_padded())
            std::vector<float>().swap(rotation_views.rot_padded);
    }

    // ========== IVF subspace index + posting lists ==========
    // If dim_stride_rotated_view != nullptr, reuses that Step-2 dim-stride rotated consumer view
    // instead of recomputing R*x.
    void build_subspace_and_postings(std::vector<float>& data, int n, int dim, int M, int sub_dim,
                                   const IndexConfig& config, const std::string& path,
                                   const std::vector<uint32_t>& assignments,
                                   size_t bin_bytes,
                                       const float* dim_stride_rotated_view = nullptr,
                                       std::vector<float>* verify_rotated_padded_owner = nullptr,
                                       const float* quant_rot_centroids = nullptr,
                                      int verify_rotated_dim = 0) {
        Timer step_timer;
        const float* verify_rotated_padded =
            (verify_rotated_padded_owner != nullptr && !verify_rotated_padded_owner->empty())
                ? verify_rotated_padded_owner->data()
                : nullptr;

        // Optionally apply FhtKac rotation before subspace partitioning.
        // This decorrelates dimensions so each subspace captures independent signal,
        // improving collision quality on datasets with correlated features (e.g. SIFT).
        //
        // IMPORTANT: When dim_stride_rotated_view is provided (from build_rabitq_clustered Step 2),
        // this reuses that data instead of recomputing R*x. One rotation serves all:
        //   Clustering:     K-means on R * x  (Step 3 in build_rabitq_clustered)
        //   RaBitQ residual: R * (x - c_k)    (Step 5 in build_rabitq_clustered)
        //   Subspace data:  R * x dim-stride  (reused here from Step 2 consumer view)
        //   Query:          R * q             (Phase 1 in search)
        // All use identical R, so subspace matching remains consistent.
        const float* subspace_data = data.data();
        std::vector<float> subspace_rotated_storage;

        if (config.subspace_force_raw) {
            std::cout << "[Build] Forcing raw vectors for subspace partitioning..." << std::endl;
            subspace_data = data.data();
        } else if (config.rotate_subspace) {
            // Check for externally-provided subspace rotation (e.g. QR from arma)
            std::string sub_rot_path = path + "/subspace_rotation.bin";
            std::ifstream sub_rot_check(sub_rot_path, std::ios::binary);

            if (sub_rot_check.good()) {
                // Use externally-provided rotation (e.g. arma::qr, same as CRISP)
                sub_rot_check.close();
                std::cout << "[Build] Loading external subspace rotation from subspace_rotation.bin..." << std::endl;
                step_timer.reset();

                RandomRotation sub_rotation;
                sub_rotation.load(sub_rot_path);

                    subspace_rotated_storage.resize((size_t)n * dim);
                    #pragma omp parallel
                    {
                        std::vector<float> tmp((size_t)sub_rotation.padded_dim(), 0.0f);
                        #pragma omp for schedule(dynamic, 256)
                        for (int i = 0; i < n; i++) {
                            sub_rotation.apply(data.data() + (size_t)i * dim, tmp.data());
                            for (int d = 0; d < dim; d++)
                                subspace_rotated_storage[(size_t)i * dim + d] = tmp[d];
                        }
                    }
                subspace_data = subspace_rotated_storage.data();
                std::cout << "  Subspace rotation done in " << step_timer.elapsed_ms() << " ms" << std::endl;
            } else if (dim_stride_rotated_view != nullptr) {
                // Optimization: reuse the Step-2 dim-stride rotated view directly.
                std::cout << "[Build] Reusing Step-2 dim-stride rotated view for subspace partitioning (no extra R*x)..." << std::endl;
                subspace_data = dim_stride_rotated_view;
            } else {
                // Fallback: rotate data with FhtKac
                std::cout << "[Build] Rotating data for subspace partitioning (FhtKac)..." << std::endl;
                step_timer.reset();

                RandomRotation sub_rotation;
                std::string rot_path = path + "/rotation.bin";
                std::ifstream rot_check(rot_path, std::ios::binary);
                if (rot_check.good()) {
                    rot_check.close();
                    sub_rotation.load(rot_path);
                } else {
                    sub_rotation.init(dim, 42, RotationType::FhtKac);
                }

                subspace_rotated_storage.resize((size_t)n * dim);
                #pragma omp parallel
                {
                    std::vector<float> tmp((size_t)sub_rotation.padded_dim(), 0.0f);
                    #pragma omp for schedule(dynamic, 256)
                    for (int i = 0; i < n; i++) {
                        sub_rotation.apply(data.data() + (size_t)i * dim, tmp.data());
                        for (int d = 0; d < dim; d++)
                            subspace_rotated_storage[(size_t)i * dim + d] = tmp[d];
                    }
                }
                subspace_data = subspace_rotated_storage.data();
                std::cout << "  Subspace rotation done in " << step_timer.elapsed_ms() << " ms" << std::endl;
            }

            // Save a flag so query engine knows to rotate queries for subspace matching
            {
                std::ofstream flag(path + "/rotate_subspace.flag");
                flag << "1" << std::endl;
            }
        }

            // Pre-extract all subspace data as contiguous N x sub_dim slabs.
            // If the source is not `data` itself, recycle `data`'s storage as the destination,
            // since total size matches N * dim.
            // Special case: on the shared FhtKac path with padded_dim > dim, use rot_padded
            // as the repack source so `data` can be reused as the slab destination instead of
            // allocating a separate full all_sub_data_storage copy.
            std::vector<std::vector<float>> all_sub_data_storage;
            std::vector<float*> all_sub_data_ptrs(M, nullptr);
            const float* repack_source = subspace_data;
            int repack_row_stride = dim;
            bool use_rot_padded_for_repack = !config.subspace_force_raw && impl::should_repack_ivf_from_rot_padded(
                config.rotate_subspace, subspace_data, dim_stride_rotated_view, data.data(),
                verify_rotated_padded, verify_rotated_dim, dim);
            if (use_rot_padded_for_repack) {
                repack_source = verify_rotated_padded;
                repack_row_stride = verify_rotated_dim;
            }

            bool reuse_data_buffer_for_all_sub_data = (repack_source != data.data());
            if (reuse_data_buffer_for_all_sub_data) {
                // Barrier A explicit storage rebinding point:
                // when `data` previously backed the dim-stride rotated view, this branch is the
                // point where that semantic role dies and `data` becomes Step-6 slab destination
                // storage instead.
                for (int m = 0; m < M; m++) {
                    all_sub_data_ptrs[m] = data.data() + (size_t)m * n * sub_dim;
                }
            } else {
                all_sub_data_storage.resize(M);
            }

            if (!reuse_data_buffer_for_all_sub_data) {
                for (int m = 0; m < M; m++) {
                    all_sub_data_storage[m].resize((size_t)n * sub_dim);
                    all_sub_data_ptrs[m] = all_sub_data_storage[m].data();
                }
            }

            auto copy_subspace = [&](float* dst, const float* src) {
#if HAS_AVX512
                if (sub_dim == 16) {
                    __m512 v = _mm512_loadu_ps(src);
                    _mm512_storeu_ps(dst, v);
                    return;
                }
                if (sub_dim == 32) {
                    __m512 v0 = _mm512_loadu_ps(src);
                    __m512 v1 = _mm512_loadu_ps(src + 16);
                    _mm512_storeu_ps(dst, v0);
                    _mm512_storeu_ps(dst + 16, v1);
                    return;
                }
#endif
#if HAS_AVX2
                if (sub_dim == 16) {
                    __m256 v0 = _mm256_loadu_ps(src);
                    __m256 v1 = _mm256_loadu_ps(src + 8);
                    _mm256_storeu_ps(dst, v0);
                    _mm256_storeu_ps(dst + 8, v1);
                    return;
                }
                if (sub_dim == 32) {
                    __m256 v0 = _mm256_loadu_ps(src);
                    __m256 v1 = _mm256_loadu_ps(src + 8);
                    __m256 v2 = _mm256_loadu_ps(src + 16);
                    __m256 v3 = _mm256_loadu_ps(src + 24);
                    _mm256_storeu_ps(dst, v0);
                    _mm256_storeu_ps(dst + 8, v1);
                    _mm256_storeu_ps(dst + 16, v2);
                    _mm256_storeu_ps(dst + 24, v3);
                    return;
                }
#endif
                std::memcpy(dst, src, (size_t)sub_dim * sizeof(float));
            };

            constexpr int kPointTile = 64;
            constexpr int kSubspaceTile = 4;
            #pragma omp parallel for schedule(static)
            for (int ib = 0; ib < n; ib += kPointTile) {
                int i_end = std::min(n, ib + kPointTile);
                for (int i = ib; i < i_end; ++i) {
                    const float* src = repack_source + (size_t)i * repack_row_stride;
                    for (int mb = 0; mb < M; mb += kSubspaceTile) {
                        int m_end = std::min(M, mb + kSubspaceTile);
                        for (int m = mb; m < m_end; ++m) {
                            copy_subspace(
                                all_sub_data_ptrs[m] + (size_t)i * sub_dim,
                                src + (size_t)m * sub_dim);
                        }
                    }
                }
            }
            log_build_rss_checkpoint("step6_after_slab_repack");

            if (use_rot_padded_for_repack) {
                std::cout << "  Repacked IVF slabs from rot_padded with row stride "
                          << verify_rotated_dim << " (reused data as slab destination)" << std::endl;
            }

        impl::LateRotatedSource late_rotated_source = impl::LateRotatedSource::VerifyRotPadded;
        if (impl::can_finalize_rotated_outputs_from_ivf_slabs(
                config.rotate_subspace,
                config.subspace_force_raw,
                subspace_data,
                dim_stride_rotated_view,
                verify_rotated_padded,
                verify_rotated_dim,
                dim,
                reuse_data_buffer_for_all_sub_data)) {
            late_rotated_source = impl::LateRotatedSource::IvfSlabs;
            if (verify_rotated_padded_owner != nullptr) {
                std::vector<float>().swap(*verify_rotated_padded_owner);
                verify_rotated_padded = nullptr;
            }
            std::cout << "  Late rotated finalize will reuse IVF slabs; released rot_padded before IVF workers"
                      << std::endl;
        }

        auto gather_rotated_chunk = [&](float* dst,
                                        const uint32_t* orig_ids,
                                        int count) {
            impl::gather_rotated_rows_from_ivf_slabs(
                dst, all_sub_data_ptrs, orig_ids, count, M, sub_dim);
        };

        bool slab_backing_live = true;
        auto release_slab_backing = [&]() {
            if (!slab_backing_live) return;
            slab_backing_live = false;

            for (auto& storage : all_sub_data_storage) {
                std::vector<float>().swap(storage);
            }
            std::vector<std::vector<float>>().swap(all_sub_data_storage);
            all_sub_data_ptrs.clear();

            if (reuse_data_buffer_for_all_sub_data) {
                // Barrier C/D explicit storage death point: when `data` was rebound to Step-6
                // slab backing, later phases must not treat that storage as still live once the
                // last slab-backed consumer has completed.
                std::vector<float>().swap(data);
            }
        };

        auto release_rotated_source_before_postings = [&]() {
            if (verify_rotated_padded_owner == nullptr) return;
            if (verify_rotated_padded_owner->empty()) return;
            std::vector<float>().swap(*verify_rotated_padded_owner);
            verify_rotated_padded = nullptr;
        };

        std::vector<int> submajor_cell_ids((size_t)n * M);
        std::vector<size_t> cells_per_subspace(M);
        int finalize_thread_count = std::max(1, omp_get_max_threads());

        // IVF subspace index: flat K-means per subspace (parallel training)
        {
            int K = config.ivf_k;
            std::cout << "[Build] IVF K-means (M=" << M << ", K=" << K
                      << ", nprobe=1, parallel)..." << std::endl;
            step_timer.reset();

            std::vector<IVFIndex> ivf_indices(M);

            int total_cores = omp_get_max_threads();

            // All M subspaces train concurrently via HSKM (best centroid quality).
            // Key: set BLIS threads = total_cores/M so 16 concurrent SGEMM calls
            // don't over-subscribe. HSKM runs with n_threads=1 (no OMP barriers),
            // BLIS handles parallelism via its own pthread pool.
                int concurrent_subspaces = M;
                int blis_threads = std::max(1, total_cores / concurrent_subspaces);
            std::cout << "  BLIS threads per subspace: " << blis_threads
                      << " (" << total_cores << " total, " << concurrent_subspaces
                      << " concurrent subspaces)" << std::endl;

            // Set BLIS thread count for concurrent SGEMM calls
#if HAS_AOCL_BLAS
            int saved_blis_threads = bli_thread_get_num_threads();
            bli_thread_set_num_threads(blis_threads);
#endif
            int threads_per_sub = blis_threads;
            log_build_rss_checkpoint("step6_before_ivf_threads");

            std::vector<std::thread> threads;
            for (int m = 0; m < M; m++) {
                threads.emplace_back([&, m, threads_per_sub, blis_threads]() {
                    omp_set_num_threads(threads_per_sub);
#if HAS_AOCL_BLAS
                    bli_thread_set_num_threads(blis_threads);
#endif

                    const float* sub_data = all_sub_data_ptrs[m];
                    ivf_indices[m].init(sub_dim, K, m);
                    ivf_indices[m].train(sub_data, n, 10, 42 + m);
                    // Note: prepare_rabitq_centroids() is called at query time
                    // when config.rabitq_centroid is enabled

                    ivf_indices[m].assign_batch(
                        sub_data,
                        submajor_cell_ids.data() + (size_t)m * n, n);
                    ivf_indices[m].save(path + "/ivf_subspace_" + std::to_string(m) + ".bin");
                });
            }
            for (auto& t : threads) t.join();
            log_build_rss_checkpoint("step6_after_ivf_join");
            omp_set_num_threads(total_cores);
#if HAS_AOCL_BLAS
            bli_thread_set_num_threads(saved_blis_threads);
#endif

            #pragma omp parallel for schedule(static)
            for (int m = 0; m < M; m++) {
                ivf_indices[m].compute_cell_sizes(submajor_cell_ids.data() + (size_t)m * n, n);
                cells_per_subspace[m] = ivf_indices[m].k_total();
            }
            log_build_rss_checkpoint("step6_after_cell_sizes");

            // Barrier C: once IVF cell IDs and per-subspace totals are fixed, slab backing is
            // dead unless it is still the explicit late rotated-source carrier for verify.
            if (late_rotated_source != impl::LateRotatedSource::IvfSlabs) {
                release_slab_backing();
                log_build_rss_checkpoint("step6_after_barrier_c_release");
            }

            std::cout << "  IVF done in " << step_timer.elapsed_ms() / 1000.0 << " s" << std::endl;
        }

        std::vector<uint32_t> perm_to_orig;
        if (config.pid_reorder) {
            std::cout << "[Build] Reordering point IDs for locality..." << std::endl;
            step_timer.reset();
            perm_to_orig = impl::build_pid_reorder_perm_counting(
                assignments, submajor_cell_ids.data(), n, M, cells_per_subspace, config);
            std::cout << "  PID reorder done in " << step_timer.elapsed_ms() << " ms" << std::endl;
            log_build_rss_checkpoint("step6_after_pid_reorder");
        }

        std::cout << "[Build] Raw vectors.fbin omitted; rotated verify vectors are the main exact-verify artifact" << std::endl;

        if ((verify_rotated_padded != nullptr || late_rotated_source == impl::LateRotatedSource::IvfSlabs) &&
            verify_rotated_dim > 0) {
            std::cout << "[Build] Writing verify_vectors.fbin..." << std::endl;
            Timer rot_vec_timer;
            rot_vec_timer.reset();
            std::ofstream out(path + "/verify_vectors.fbin", std::ios::binary);
            int reserved = 0;
            out.write(reinterpret_cast<const char*>(&n), sizeof(int));
            out.write(reinterpret_cast<const char*>(&verify_rotated_dim), sizeof(int));
            out.write(reinterpret_cast<const char*>(&reserved), sizeof(int));
            if (late_rotated_source == impl::LateRotatedSource::VerifyRotPadded && perm_to_orig.empty()) {
                out.write(reinterpret_cast<const char*>(verify_rotated_padded),
                          (size_t)n * verify_rotated_dim * sizeof(float));
            } else {
                size_t record_bytes = (size_t)verify_rotated_dim * sizeof(float);
                size_t records_per_chunk =
                    impl::finalize_rows_per_chunk(record_bytes, finalize_thread_count);
                std::vector<char> chunk(records_per_chunk * record_bytes);
                std::vector<float> rotated_chunk;
                std::vector<uint32_t> chunk_orig(records_per_chunk);
                if (late_rotated_source == impl::LateRotatedSource::IvfSlabs) {
                    rotated_chunk.resize(records_per_chunk * (size_t)verify_rotated_dim);
                }
                for (int start = 0; start < n; start += (int)records_per_chunk) {
                    int count = std::min<int>((int)records_per_chunk, n - start);
                    for (int j = 0; j < count; j++) {
                        chunk_orig[(size_t)j] = perm_to_orig.empty()
                            ? (uint32_t)(start + j)
                            : perm_to_orig[(size_t)(start + j)];
                    }
                    if (late_rotated_source == impl::LateRotatedSource::IvfSlabs) {
                        gather_rotated_chunk(rotated_chunk.data(), chunk_orig.data(), count);
                        out.write(reinterpret_cast<const char*>(rotated_chunk.data()),
                                  (size_t)count * record_bytes);
                    } else {
                        char* dst = chunk.data();
                        for (int j = 0; j < count; j++) {
                            uint32_t orig = chunk_orig[(size_t)j];
                            std::memcpy(dst + (size_t)j * record_bytes,
                                        verify_rotated_padded + (size_t)orig * verify_rotated_dim,
                                        record_bytes);
                        }
                        out.write(chunk.data(), (size_t)count * record_bytes);
                    }
                }
            }
            std::cout << "  verify_vectors.fbin done in " << rot_vec_timer.elapsed_ms() << " ms" << std::endl;
            log_build_rss_checkpoint("step6_after_verify_finalize");

            if (!perm_to_orig.empty()) {
                if (quant_rot_centroids == nullptr) {
                    throw std::runtime_error("reorder bin-code finalize requires rotated centroids");
                }

                std::ofstream out(path + "/rabitq_bin_data.bin", std::ios::binary);
                int bb = (int)(sizeof(uint32_t) + bin_bytes);
                out.write(reinterpret_cast<const char*>(&n), sizeof(int));
                out.write(reinterpret_cast<const char*>(&bb), sizeof(int));

                size_t record_bytes = (size_t)bb;
                size_t records_per_chunk =
                    impl::finalize_rows_per_chunk(record_bytes, finalize_thread_count);
                std::vector<char> chunk(records_per_chunk * record_bytes);
                std::vector<float> rotated_chunk;
                std::vector<uint32_t> chunk_orig(records_per_chunk);
                if (late_rotated_source == impl::LateRotatedSource::IvfSlabs) {
                    rotated_chunk.resize(records_per_chunk * (size_t)verify_rotated_dim);
                }

                for (int start = 0; start < n; start += (int)records_per_chunk) {
                    int count = std::min<int>((int)records_per_chunk, n - start);
                    for (int j = 0; j < count; j++) {
                        chunk_orig[(size_t)j] = perm_to_orig[(size_t)(start + j)];
                    }
                    if (late_rotated_source == impl::LateRotatedSource::IvfSlabs) {
                        gather_rotated_chunk(rotated_chunk.data(), chunk_orig.data(), count);
                    }

                    #pragma omp parallel for schedule(dynamic, 256)
                    for (int j = 0; j < count; j++) {
                        uint32_t orig = chunk_orig[(size_t)j];
                        uint32_t cid = assignments[(size_t)orig];
                        const float* rot_vec =
                            late_rotated_source == impl::LateRotatedSource::IvfSlabs
                                ? rotated_chunk.data() + (size_t)j * verify_rotated_dim
                                : verify_rotated_padded + (size_t)orig * verify_rotated_dim;
                        const float* c_rot = quant_rot_centroids + (size_t)cid * verify_rotated_dim;

                        char* out_ptr = chunk.data() + (size_t)j * record_bytes;
                        std::memcpy(out_ptr, &cid, sizeof(uint32_t));
                        rabitqlib::quant::quantize_compact_one_bit<float>(
                            rot_vec, c_rot, verify_rotated_dim,
                            out_ptr + sizeof(uint32_t),
                            rabitqlib::METRIC_L2);
                    }
                    out.write(chunk.data(), (size_t)count * record_bytes);
                }
            }

            // Barrier D: verify/reordered-output writers are the final sanctioned rotated-source
            // consumers. Posting build must start with only bridge objects (`submajor_cell_ids`,
            // `cells_per_subspace`, optional `perm_to_orig`) and no hidden slab/rotated backing.
            if (late_rotated_source == impl::LateRotatedSource::IvfSlabs) {
                release_slab_backing();
                log_build_rss_checkpoint("step6_after_barrier_d_release");
            }
            release_rotated_source_before_postings();
        }

        if (!perm_to_orig.empty()) {
            {
                std::ofstream out(path + "/rabitq_assignments.bin", std::ios::binary);
                out.write(reinterpret_cast<const char*>(&n), sizeof(int));
                constexpr size_t record_bytes = sizeof(uint32_t);
                size_t chunk_bytes = 256 * PAGE_SIZE;
                size_t records_per_chunk = std::max<size_t>(1, chunk_bytes / record_bytes);
                std::vector<char> chunk(records_per_chunk * record_bytes);
                for (int start = 0; start < n; start += (int)records_per_chunk) {
                    int count = std::min<int>((int)records_per_chunk, n - start);
                    char* dst = chunk.data();
                    for (int j = 0; j < count; j++) {
                        uint32_t orig = perm_to_orig[(size_t)(start + j)];
                        std::memcpy(dst + (size_t)j * record_bytes,
                                    &assignments[(size_t)orig], record_bytes);
                    }
                    out.write(chunk.data(), (size_t)count * record_bytes);
                }
            }
        }

        // Posting lists
        std::cout << "[Build] Posting lists..." << std::endl;
        step_timer.reset();
        PostingLists posting_lists;
        posting_lists.build_submajor(cells_per_subspace, submajor_cell_ids.data(), n, M, path,
                                     perm_to_orig.empty() ? nullptr : &perm_to_orig);
        std::cout << "  Posting lists done in " << step_timer.elapsed_ms() << " ms" << std::endl;
        log_build_rss_checkpoint("step6_after_postings");

        if (!perm_to_orig.empty()) {
            std::string final_perm_path = path + "/perm_to_orig.bin";
            impl::write_final_perm_to_orig_vector_atomic(perm_to_orig, final_perm_path, n);
        }
    }
};

} // namespace score
