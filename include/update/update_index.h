#pragma once

#include "core/common.h"
#include "query/query_engine.h"
#include "core/rotation.h"
#include "update/update_manifest.h"
#include "update/update_state.h"
#include "update/update_wal.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <cctype>
#include <exception>
#include <fcntl.h>
#include <functional>
#include <iterator>
#include <sys/file.h>
#include <thread>
#include <vector>

#if HAS_AOCL_BLAS
extern "C" void bli_thread_set_num_threads(int);
extern "C" int bli_thread_get_num_threads(void);
#endif

namespace score {

namespace impl {

constexpr size_t kUpdateMicroVectorWork = 32ull * 1024ull;
constexpr size_t kUpdateMicroDotWork = 4ull * 1024ull * 1024ull;
constexpr size_t kUpdateMicroIvfDotWork = 1ull * 1024ull * 1024ull;
constexpr size_t kUpdateDirectAssignWork = 64ull * 1024ull * 1024ull;
// Calibrated to keep known-small GIST1M batches on the lightweight SIMD path,
// while scaling the cutoff by actual IVF assignment work for other dimensions.
constexpr size_t kUpdateIvfDirectAssignWork = 1536ull * 1024ull * 1024ull;
constexpr size_t kUpdateIvfSubspaceParallelMinWork = 8ull * 1024ull * 1024ull;
constexpr size_t kUpdateIvfSubspaceParallelMaxWork = 64ull * 1024ull * 1024ull;
constexpr size_t kUpdateLastWinsSmallScanRows = 64;

inline bool update_work_leq(size_t a, size_t b, size_t c, size_t limit) {
    if (a == 0 || b == 0 || c == 0) return true;
    if (a > limit / b) return false;
    size_t ab = a * b;
    if (ab > limit / c) return false;
    return ab * c <= limit;
}

inline bool update_vector_work_is_micro(int rows, int dim) {
    if (rows <= 0 || dim <= 0) return true;
    return update_work_leq((size_t)rows, (size_t)dim, 1, kUpdateMicroVectorWork);
}

inline bool update_dot_work_is_micro(int rows, int k, int dim) {
    if (rows <= 0 || k <= 0 || dim <= 0) return true;
    return update_work_leq((size_t)rows, (size_t)k, (size_t)dim, kUpdateMicroDotWork);
}

inline bool update_ivf_dot_work_is_micro(int rows, int k, int dim) {
    if (rows <= 0 || k <= 0 || dim <= 0) return true;
    return update_work_leq((size_t)rows, (size_t)k, (size_t)dim, kUpdateMicroIvfDotWork);
}

inline bool update_direct_assign_work_is_small(int rows, int k, int dim) {
    if (rows <= 0 || k <= 0 || dim <= 0) return true;
    return update_work_leq((size_t)rows, (size_t)k, (size_t)dim, kUpdateDirectAssignWork);
}

inline bool update_ivf_direct_assign_work_is_small(int rows, const BaseSegment& base) {
    if (rows <= 0) return true;
    size_t remaining = kUpdateIvfDirectAssignWork;
    for (const IVFIndex& index : base.ivf_indices) {
        int k = index.k_total();
        int sub_dim = index.subspace_dim();
        if (k <= 0 || sub_dim <= 0) continue;
        if (!update_work_leq((size_t)rows, (size_t)k, (size_t)sub_dim, remaining)) {
            return false;
        }
        remaining -= (size_t)rows * (size_t)k * (size_t)sub_dim;
    }
    return true;
}

inline size_t update_ivf_assign_work_capped(int rows, const BaseSegment& base, size_t cap) {
    if (rows <= 0) return 0;
    size_t total = 0;
    for (const IVFIndex& index : base.ivf_indices) {
        int k = index.k_total();
        int sub_dim = index.subspace_dim();
        if (k <= 0 || sub_dim <= 0) continue;
        if (!update_work_leq((size_t)rows, (size_t)k, (size_t)sub_dim, cap - total)) {
            return cap;
        }
        total += (size_t)rows * (size_t)k * (size_t)sub_dim;
        if (total >= cap) return cap;
    }
    return total;
}

inline size_t add_saturated_size(size_t a, size_t b) {
    if (a > std::numeric_limits<size_t>::max() - b) {
        return std::numeric_limits<size_t>::max();
    }
    return a + b;
}

inline size_t mul_saturated_size(size_t a, size_t b) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return std::numeric_limits<size_t>::max();
    }
    return a * b;
}

inline size_t estimate_update_encode_work_per_row(const BaseSegment& base) {
    size_t work = 0;
    if (base.rabitq_k > 0 && base.rabitq_padded_dim > 0) {
        work = add_saturated_size(
            work,
            mul_saturated_size((size_t)base.rabitq_k, (size_t)base.rabitq_padded_dim));
    }
    for (const IVFIndex& index : base.ivf_indices) {
        int k = index.k_total();
        int sub_dim = index.subspace_dim();
        if (k <= 0 || sub_dim <= 0) continue;
        work = add_saturated_size(
            work,
            mul_saturated_size((size_t)k, (size_t)sub_dim));
    }
    if (base.rabitq_padded_dim > 0) {
        work = add_saturated_size(work, (size_t)base.rabitq_padded_dim * 4u);
    }
    return std::max<size_t>(1, work);
}

inline int adaptive_update_threads_for_work(size_t total_work, int max_threads) {
    if (max_threads <= 1) return std::max(1, max_threads);
    int target = max_threads;
    if (total_work < 1ull * 1000ull * 1000ull) {
        target = 1;
    } else if (total_work < 4ull * 1000ull * 1000ull) {
        target = 4;
    } else if (total_work < 64ull * 1000ull * 1000ull) {
        target = 8;
    } else if (total_work < 512ull * 1000ull * 1000ull) {
        target = 16;
    } else if (total_work < 4ull * 1000ull * 1000ull * 1000ull) {
        target = 32;
    }
    return std::max(1, std::min(max_threads, target));
}

inline bool path_exists(const std::string& path) {
    if (::access(path.c_str(), F_OK) == 0) return true;
    if (errno == ENOENT) return false;
    throw std::runtime_error("cannot probe path: " + path);
}

inline std::string join_index_path(const std::string& index_path, const std::string& relative_or_absolute) {
    if (relative_or_absolute.empty()) return relative_or_absolute;
    if (relative_or_absolute[0] == '/') return relative_or_absolute;
    return index_path + "/" + relative_or_absolute;
}

inline void ensure_dir(const std::string& path) {
    if (::mkdir(path.c_str(), 0755) == 0) return;
    if (errno == EEXIST) return;
    throw std::runtime_error("cannot create directory: " + path + " (" + std::strerror(errno) + ")");
}

inline void fsync_file_if_requested(const std::string& path, bool do_fsync) {
    update_io::fsync_file_if_requested(path, do_fsync);
}

inline void fsync_directory_if_requested(const std::string& path, bool do_fsync) {
    update_io::fsync_directory_if_requested(path, do_fsync);
}

inline void fsync_parent_dir_if_requested(const std::string& path, bool do_fsync) {
    update_io::fsync_parent_dir_if_requested(path, do_fsync);
}

using UpdateWriteLock = update_io::UpdateWriterLock;
using UpdatePublishWriteLock = update_io::UpdatePublishWriteLock;

class ScopedUpdateThreadLimit {
public:
    explicit ScopedUpdateThreadLimit(int requested_threads)
        : active_(requested_threads != 0) {
        if (!active_) return;
        if (requested_threads < 1) {
            throw std::invalid_argument("update thread count must be positive");
        }
        saved_omp_threads_ = omp_get_max_threads();
        omp_set_num_threads(requested_threads);
#if HAS_AOCL_BLAS
        saved_blis_threads_ = bli_thread_get_num_threads();
        bli_thread_set_num_threads(requested_threads);
#endif
    }

    ~ScopedUpdateThreadLimit() {
        if (!active_) return;
        omp_set_num_threads(saved_omp_threads_);
#if HAS_AOCL_BLAS
        bli_thread_set_num_threads(saved_blis_threads_);
#endif
    }

    ScopedUpdateThreadLimit(const ScopedUpdateThreadLimit&) = delete;
    ScopedUpdateThreadLimit& operator=(const ScopedUpdateThreadLimit&) = delete;

private:
    bool active_ = false;
    int saved_omp_threads_ = 1;
#if HAS_AOCL_BLAS
    int saved_blis_threads_ = 1;
#endif
};

inline std::vector<uint32_t> load_update_ids(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("cannot open ids file: " + path);
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    bool text = true;
    for (char ch : bytes) {
        unsigned char c = (unsigned char)ch;
        if (!(std::isdigit(c) || std::isspace(c) || ch == ',' || ch == '+')) {
            text = false;
            break;
        }
    }
    std::vector<uint32_t> ids;
    if (text) {
        std::string content(bytes.begin(), bytes.end());
        std::replace(content.begin(), content.end(), ',', ' ');
        std::istringstream iss(content);
        uint64_t value = 0;
        while (iss >> value) {
            if (value > (uint64_t)std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("ids file contains value above uint32 range: " + path);
            }
            ids.push_back((uint32_t)value);
        }
        return ids;
    }
    if (bytes.size() % sizeof(uint32_t) != 0) {
        throw std::runtime_error("binary ids file size is not a multiple of uint32: " + path);
    }
    ids.resize(bytes.size() / sizeof(uint32_t));
    if (!ids.empty()) std::memcpy(ids.data(), bytes.data(), bytes.size());
    return ids;
}

struct UpdateFbinMetadata {
    int n = 0;
    int dim = 0;
};

inline UpdateFbinMetadata read_update_fbin_metadata(const std::string& path) {
    auto [tag, detected_n, detected_dim] = detect_vector_format(path);
    if (tag != 0) throw std::runtime_error("update vector input must be .fbin: " + path);
    UpdateFbinMetadata meta;
    meta.n = detected_n;
    meta.dim = detected_dim;
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("cannot open update fbin: " + path);
    int header_n = 0, header_dim = 0, reserved = 0;
    in.read(reinterpret_cast<char*>(&header_n), sizeof(int));
    in.read(reinterpret_cast<char*>(&header_dim), sizeof(int));
    in.read(reinterpret_cast<char*>(&reserved), sizeof(int));
    if (header_n != meta.n || header_dim != meta.dim || reserved != 0) {
        throw std::runtime_error("invalid update fbin header: " + path);
    }
    return meta;
}

inline std::vector<float> load_update_fbin_rows(const std::string& path,
                                                int total_n,
                                                int dim,
                                                size_t row_offset,
                                                size_t row_count) {
    if (total_n < 0 || dim <= 0) {
        throw std::runtime_error("invalid update fbin shape: " + path);
    }
    if (row_offset > (size_t)total_n || row_count > (size_t)total_n - row_offset) {
        throw std::runtime_error("update fbin row slice is out of range: " + path);
    }
    if (row_count > std::numeric_limits<size_t>::max() / (size_t)dim) {
        throw std::runtime_error("update fbin row slice size overflows: " + path);
    }
    size_t float_count = row_count * (size_t)dim;
    uint64_t payload_offset = (uint64_t)sizeof(int) * 3u +
        (uint64_t)row_offset * (uint64_t)dim * (uint64_t)sizeof(float);
    if (payload_offset > (uint64_t)std::numeric_limits<std::streamoff>::max()) {
        throw std::runtime_error("update fbin row offset exceeds stream range: " + path);
    }

    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("cannot open update fbin: " + path);
    in.seekg((std::streamoff)payload_offset, std::ios::beg);
    if (!in) throw std::runtime_error("cannot seek update fbin: " + path);
    std::vector<float> data(float_count);
    in.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    if (!in) throw std::runtime_error("short update fbin row slice: " + path);
    return data;
}

class SequentialUpdateFbinReader {
public:
    SequentialUpdateFbinReader(const std::string& path, int total_n, int dim)
        : path_(path), total_n_(total_n), dim_(dim) {
        if (total_n_ < 0 || dim_ <= 0) {
            throw std::runtime_error("invalid update fbin shape: " + path_);
        }
        in_.open(path_, std::ios::binary);
        if (!in_.is_open()) throw std::runtime_error("cannot open update fbin: " + path_);
        in_.seekg((std::streamoff)(sizeof(int) * 3), std::ios::beg);
        if (!in_) throw std::runtime_error("cannot seek update fbin payload: " + path_);
    }

    void read_rows(size_t row_offset, size_t row_count, std::vector<float>& data) {
        if (row_offset > (size_t)total_n_ || row_count > (size_t)total_n_ - row_offset) {
            throw std::runtime_error("update fbin row slice is out of range: " + path_);
        }
        if (row_count > std::numeric_limits<size_t>::max() / (size_t)dim_) {
            throw std::runtime_error("update fbin row slice size overflows: " + path_);
        }
        if (row_offset != next_row_) {
            uint64_t payload_offset = (uint64_t)sizeof(int) * 3u +
                (uint64_t)row_offset * (uint64_t)dim_ * (uint64_t)sizeof(float);
            if (payload_offset > (uint64_t)std::numeric_limits<std::streamoff>::max()) {
                throw std::runtime_error("update fbin row offset exceeds stream range: " + path_);
            }
            in_.clear();
            in_.seekg((std::streamoff)payload_offset, std::ios::beg);
            if (!in_) throw std::runtime_error("cannot seek update fbin: " + path_);
        }

        size_t float_count = row_count * (size_t)dim_;
        data.resize(float_count);
        if (float_count != 0) {
            in_.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
            if (!in_) throw std::runtime_error("short update fbin row slice: " + path_);
        }
        next_row_ = row_offset + row_count;
    }

private:
    std::string path_;
    int total_n_ = 0;
    int dim_ = 0;
    size_t next_row_ = 0;
    std::ifstream in_;
};

inline std::vector<float> load_update_fbin(const std::string& path, int& n, int& dim) {
    UpdateFbinMetadata meta = read_update_fbin_metadata(path);
    n = meta.n;
    dim = meta.dim;
    return load_update_fbin_rows(path, n, dim, 0, (size_t)n);
}

inline std::vector<size_t> cells_per_subspace_from_base(const BaseSegment& base) {
    std::vector<size_t> cells;
    cells.reserve(base.ivf_indices.size());
    for (size_t m = 0; m < base.ivf_indices.size(); ++m) {
        if (base.posting_lists) {
            cells.push_back((size_t)base.posting_lists->K((int)m));
        } else {
            cells.push_back(base.ivf_indices[m].k_total());
        }
    }
    return cells;
}

inline void validate_update_row_cell_width(const BaseSegment& base) {
    for (size_t m = 0; m < base.ivf_indices.size(); ++m) {
        size_t K = base.posting_lists ? (size_t)base.posting_lists->K((int)m)
                                      : base.ivf_indices[m].k_total();
        if (K > (size_t)std::numeric_limits<uint16_t>::max()) {
            throw std::runtime_error("update subspace cell count exceeds uint16 row-cell range");
        }
    }
}

inline std::vector<float> compute_rabitq_centroid_bias(const BaseSegment& base) {
    if (base.rabitq_k <= 0 || base.rabitq_padded_dim <= 0 ||
        base.rabitq_rot_centroids.size() <
            (size_t)base.rabitq_k * (size_t)base.rabitq_padded_dim) {
        return {};
    }
    std::vector<float> computed_bias((size_t)base.rabitq_k);
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < base.rabitq_k; ++c) {
        float nsq = 0.0f;
        const float* centroid = base.rabitq_rot_centroids.data() +
            (size_t)c * (size_t)base.rabitq_padded_dim;
        for (int d = 0; d < base.rabitq_padded_dim; ++d) nsq += centroid[d] * centroid[d];
        computed_bias[(size_t)c] = 0.5f * nsq;
    }
    return computed_bias;
}

inline BaseSegment load_update_base_segment_lightweight(const std::string& index_path) {
    BaseSegment base;

    RandomRotation::Metadata rotation =
        RandomRotation::load_metadata(index_path + "/rotation.bin");
    base.dim = rotation.dim;
    base.rabitq_padded_dim = rotation.padded_dim;
    base.verify_vectors_dim = rotation.padded_dim;

    {
        std::ifstream cent_in(index_path + "/rabitq_centroids.bin", std::ios::binary);
        if (!cent_in.is_open()) {
            throw std::runtime_error("cannot open rabitq_centroids.bin for update");
        }
        int K = 0, dim = 0, padded_dim = 0;
        cent_in.read(reinterpret_cast<char*>(&K), sizeof(int));
        cent_in.read(reinterpret_cast<char*>(&dim), sizeof(int));
        cent_in.read(reinterpret_cast<char*>(&padded_dim), sizeof(int));
        if (!cent_in || K <= 0 || dim != base.dim ||
            padded_dim != base.rabitq_padded_dim) {
            throw std::runtime_error("invalid rabitq_centroids.bin header for update");
        }
        if ((size_t)K > std::numeric_limits<size_t>::max() / (size_t)padded_dim) {
            throw std::runtime_error("rabitq centroid payload size overflows for update");
        }
        base.rabitq_k = K;
        base.rabitq_rot_centroids.resize((size_t)K * (size_t)padded_dim);
        cent_in.read(reinterpret_cast<char*>(base.rabitq_rot_centroids.data()),
                     base.rabitq_rot_centroids.size() * sizeof(float));
        if (!cent_in) throw std::runtime_error("short rabitq_centroids.bin payload for update");
        base.rabitq_centroid_bias = compute_rabitq_centroid_bias(base);
    }

    {
        std::ifstream bindata_in(index_path + "/rabitq_bin_data.bin", std::ios::binary);
        if (!bindata_in.is_open()) {
            throw std::runtime_error("cannot open rabitq_bin_data.bin for update");
        }
        int n = 0, record_bytes = 0;
        bindata_in.read(reinterpret_cast<char*>(&n), sizeof(int));
        bindata_in.read(reinterpret_cast<char*>(&record_bytes), sizeof(int));
        const size_t expected_payload_bytes =
            rabitqlib::BinDataMap<float>::data_bytes(base.rabitq_padded_dim);
        const size_t expected_record_bytes = sizeof(uint32_t) + expected_payload_bytes;
        if (!bindata_in || n <= 0 || record_bytes != (int)expected_record_bytes) {
            throw std::runtime_error("invalid rabitq_bin_data.bin header for update");
        }
        base.n = n;
        base.rabitq_bin_bytes = expected_payload_bytes;
        base.rabitq_bin_with_assign_record_bytes_ = expected_record_bytes;
    }

    {
        auto [tag, verify_n, verify_dim] = detect_vector_format(index_path + "/verify_vectors.fbin");
        if (tag != 0 || verify_n != base.n || verify_dim != base.rabitq_padded_dim) {
            throw std::runtime_error("verify_vectors.fbin header does not match update base metadata");
        }
        base.verify_vectors_dim = verify_dim;
    }

    int detected_M = 0;
    for (int m = 0; ; ++m) {
        std::ifstream probe(index_path + "/ivf_subspace_" + std::to_string(m) + ".bin",
                            std::ios::binary);
        if (!probe.is_open()) break;
        ++detected_M;
    }
    if (detected_M <= 0) throw std::runtime_error("update requires IVF subspace indices");
    base.ivf_indices.resize((size_t)detected_M);
    for (int m = 0; m < detected_M; ++m) {
        base.ivf_indices[(size_t)m].load(
            index_path + "/ivf_subspace_" + std::to_string(m) + ".bin");
        base.ivf_indices[(size_t)m].prepare_simd_centroids();
    }

    {
        std::ifstream perm_in(index_path + "/perm_to_orig.bin", std::ios::binary);
        if (perm_in.good()) {
            int n_perm = 0;
            perm_in.read(reinterpret_cast<char*>(&n_perm), sizeof(int));
            if (!perm_in || n_perm != base.n) {
                throw std::runtime_error("perm_to_orig.bin length does not match update base");
            }
            base.perm_to_orig.resize((size_t)n_perm);
            perm_in.read(reinterpret_cast<char*>(base.perm_to_orig.data()),
                         base.perm_to_orig.size() * sizeof(uint32_t));
            if (!perm_in) throw std::runtime_error("short perm_to_orig.bin payload for update");
        }
    }

    return base;
}

inline void assign_l0_rabitq_centroids(const BaseSegment& base,
                                       const float* rotated,
                                       int n,
                                       std::vector<uint32_t>& assignments) {
    if (n < 0) throw std::invalid_argument("negative update batch size");
    assignments.assign((size_t)n, 0);
    if (n == 0) return;
    if (base.rabitq_k <= 0 || base.rabitq_padded_dim <= 0) {
        throw std::runtime_error("update insert requires RaBitQ centroids");
    }
    if (base.rabitq_rot_centroids.size() <
        (size_t)base.rabitq_k * (size_t)base.rabitq_padded_dim) {
        throw std::runtime_error("loaded RaBitQ centroid payload is too short");
    }

    std::vector<float> fallback_bias;
    const float* centroid_bias = base.rabitq_centroid_bias.data();
    if (base.rabitq_centroid_bias.size() != (size_t)base.rabitq_k) {
        fallback_bias = compute_rabitq_centroid_bias(base);
        if (fallback_bias.size() != (size_t)base.rabitq_k) {
            throw std::runtime_error("could not compute RaBitQ centroid bias for update");
        }
        centroid_bias = fallback_bias.data();
    }

    if (update_direct_assign_work_is_small(n, base.rabitq_k, base.rabitq_padded_dim)) {
        auto assign_one_row = [&](int i) {
            const float* row = rotated + (size_t)i * (size_t)base.rabitq_padded_dim;
            float best_score = std::numeric_limits<float>::max();
            uint32_t best = 0;
            for (int c = 0; c < base.rabitq_k; ++c) {
                const float* centroid = base.rabitq_rot_centroids.data() +
                    (size_t)c * (size_t)base.rabitq_padded_dim;
                float dot = 0.0f;
                for (int d = 0; d < base.rabitq_padded_dim; ++d) dot += row[d] * centroid[d];
                float score = centroid_bias[(size_t)c] - dot;
                if (score < best_score) {
                    best_score = score;
                    best = (uint32_t)c;
                }
            }
            assignments[(size_t)i] = best;
        };
        if (update_dot_work_is_micro(n, base.rabitq_k, base.rabitq_padded_dim)) {
            for (int i = 0; i < n; ++i) assign_one_row(i);
        } else {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                assign_one_row(i);
            }
        }
        return;
    }

#if HAS_AOCL_BLAS
    const int block_size = score::impl::capped_assign_block_size(
        base.rabitq_k, 128ull * 1024ull * 1024ull);
    std::vector<float> ip_block((size_t)block_size * (size_t)base.rabitq_k);
    for (int start = 0; start < n; start += block_size) {
        int block_n = std::min(block_size, n - start);
        const float* block_data = rotated + (size_t)start * (size_t)base.rabitq_padded_dim;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    block_n, base.rabitq_k, base.rabitq_padded_dim, 1.0f,
                    block_data, base.rabitq_padded_dim,
                    base.rabitq_rot_centroids.data(), base.rabitq_padded_dim,
                    0.0f, ip_block.data(), base.rabitq_k);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < block_n; ++i) {
            const float* ip_row = ip_block.data() + (size_t)i * (size_t)base.rabitq_k;
            float best_score = std::numeric_limits<float>::max();
            uint32_t best = 0;
            for (int c = 0; c < base.rabitq_k; ++c) {
                float score = centroid_bias[(size_t)c] - ip_row[c];
                if (score < best_score) {
                    best_score = score;
                    best = (uint32_t)c;
                }
            }
            assignments[(size_t)start + (size_t)i] = best;
        }
    }
#else
    #pragma omp parallel for schedule(dynamic, 128)
    for (int i = 0; i < n; ++i) {
        const float* row = rotated + (size_t)i * (size_t)base.rabitq_padded_dim;
        float best_score = std::numeric_limits<float>::max();
        uint32_t best = 0;
        for (int c = 0; c < base.rabitq_k; ++c) {
            const float* centroid = base.rabitq_rot_centroids.data() +
                (size_t)c * (size_t)base.rabitq_padded_dim;
            float dot = 0.0f;
            for (int d = 0; d < base.rabitq_padded_dim; ++d) dot += row[d] * centroid[d];
            float score = centroid_bias[(size_t)c] - dot;
            if (score < best_score) {
                best_score = score;
                best = (uint32_t)c;
            }
        }
        assignments[(size_t)i] = best;
    }
#endif
}

inline void assign_l0_subspace_cells(const BaseSegment& base,
                                     const float* subspace_source,
                                     int subspace_stride,
                                     int n,
                                     std::vector<int>& submajor_cell_ids) {
    const size_t M = base.ivf_indices.size();
    if (M == 0) throw std::runtime_error("update insert requires IVF subspace indices");
    if (n > 0 && M > std::numeric_limits<size_t>::max() / (size_t)n) {
        throw std::runtime_error("update subspace assignment buffer size overflows");
    }
    submajor_cell_ids.assign(M * (size_t)n, 0);
    if (n <= 0) return;

    auto assign_one_subspace_simd = [&](size_t m, bool parallel_rows) {
        int sub_dim = base.ivf_indices[m].subspace_dim();
        const float* subspace_data = subspace_source + m * (size_t)sub_dim;
        base.ivf_indices[m].assign_batch_strided_simd_soa(
            subspace_data,
            subspace_stride,
            submajor_cell_ids.data() + m * (size_t)n,
            n,
            parallel_rows);
    };

    if (update_ivf_direct_assign_work_is_small(n, base)) {
        size_t total_work = update_ivf_assign_work_capped(
            n, base, kUpdateIvfSubspaceParallelMaxWork + 1);
        bool parallel_subspaces =
            M > 1 &&
            omp_get_max_threads() > 1 &&
            total_work >= kUpdateIvfSubspaceParallelMinWork &&
            total_work <= kUpdateIvfSubspaceParallelMaxWork;
        if (parallel_subspaces) {
            #pragma omp parallel for schedule(static)
            for (int mi = 0; mi < (int)M; ++mi) {
                assign_one_subspace_simd((size_t)mi, false);
            }
            return;
        }
        for (size_t m = 0; m < M; ++m) {
            int sub_dim = base.ivf_indices[m].subspace_dim();
            bool parallel = !update_ivf_dot_work_is_micro(
                n,
                (int)base.ivf_indices[m].k_total(),
                sub_dim);
            assign_one_subspace_simd(m, parallel);
        }
        return;
    }

    const int total_threads = std::max(1, omp_get_max_threads());
    const int worker_count = std::max(1, std::min<int>((int)M, total_threads));
    const int threads_per_worker = std::max(1, total_threads / worker_count);
    const size_t scratch_cap = std::max<size_t>(
        8ull * 1024ull * 1024ull,
        (128ull * 1024ull * 1024ull) / (size_t)worker_count);

    auto assign_one_subspace = [&](size_t m) {
        int sub_dim = base.ivf_indices[m].subspace_dim();
        const float* subspace_data = subspace_source + m * (size_t)sub_dim;
        base.ivf_indices[m].assign_batch_strided_with_scratch_cap(
            subspace_data,
            subspace_stride,
            submajor_cell_ids.data() + m * (size_t)n,
            n,
            scratch_cap);
    };

    if (worker_count == 1) {
        for (size_t m = 0; m < M; ++m) assign_one_subspace(m);
        return;
    }

#if HAS_AOCL_BLAS
    int saved_blis_threads = bli_thread_get_num_threads();
    bli_thread_set_num_threads(threads_per_worker);
#endif

    std::atomic<size_t> next_subspace{0};
    std::vector<std::thread> workers;
    std::vector<std::exception_ptr> worker_errors((size_t)worker_count);
    workers.reserve((size_t)worker_count);
    for (int worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&, worker]() {
            try {
                omp_set_num_threads(threads_per_worker);
#if HAS_AOCL_BLAS
                bli_thread_set_num_threads(threads_per_worker);
#endif
                while (true) {
                    size_t m = next_subspace.fetch_add(1, std::memory_order_relaxed);
                    if (m >= M) break;
                    assign_one_subspace(m);
                }
            } catch (...) {
                worker_errors[(size_t)worker] = std::current_exception();
            }
        });
    }
    for (std::thread& worker : workers) worker.join();
    omp_set_num_threads(total_threads);
#if HAS_AOCL_BLAS
    bli_thread_set_num_threads(saved_blis_threads);
#endif
    for (const std::exception_ptr& error : worker_errors) {
        if (error) std::rethrow_exception(error);
    }
}

struct EncodedL0InsertBatch {
    uint32_t pid_begin = 0;
    uint32_t row_count = 0;
    uint32_t verify_dim = 0;
    uint32_t rabitq_record_bytes = 0;
    uint32_t num_subspaces = 0;
    std::vector<uint32_t> orig_ids;
    std::vector<float> verify_rotated;
    std::vector<uint8_t> rabitq_records;
    std::vector<uint16_t> subspace_cells;

    size_t size() const { return row_count; }

    uint32_t global_pid(size_t row) const {
        return pid_begin + (uint32_t)row;
    }

    const float* verify_ptr(size_t row) const {
        return verify_rotated.data() + row * (size_t)verify_dim;
    }

    const uint8_t* rabitq_record_ptr(size_t row) const {
        return rabitq_records.data() + row * (size_t)rabitq_record_bytes;
    }

    const uint16_t* subspace_cells_ptr(size_t row) const {
        return subspace_cells.data() + row * (size_t)num_subspaces;
    }
};

struct L0EncodeScratch {
    std::vector<float> subspace_rotated;
    std::vector<uint32_t> rabitq_assignments;
    std::vector<int> submajor_cell_ids;
};

inline std::shared_ptr<const L0Block> build_l0_block_from_encoded_batches(
    uint64_t commit_lsn,
    const std::vector<size_t>& cells_per_subspace,
    const EncodedL0InsertBatch& first,
    const EncodedL0InsertBatch& second) {
    size_t total_rows = first.size() + second.size();
    if (total_rows == 0) return nullptr;
    uint32_t verify_dim = first.size() != 0 ? first.verify_dim : second.verify_dim;
    uint32_t rabitq_record_bytes =
        first.size() != 0 ? first.rabitq_record_bytes : second.rabitq_record_bytes;
    if (verify_dim == 0 || rabitq_record_bytes == 0) {
        throw std::runtime_error("encoded L0 batch payload dimensions are empty");
    }
    L0Block::Builder builder(cells_per_subspace, verify_dim, rabitq_record_bytes, false);
    builder.reserve_rows(total_rows);
    auto append_batch = [&](const EncodedL0InsertBatch& batch) {
        if (batch.size() == 0) return;
        if (batch.verify_dim != verify_dim ||
            batch.rabitq_record_bytes != rabitq_record_bytes ||
            batch.num_subspaces != cells_per_subspace.size()) {
            throw std::runtime_error("encoded L0 batch dimensions do not match reader cells");
        }
        builder.append_contiguous_rows(
            batch.pid_begin,
            batch.size(),
            batch.orig_ids.data(),
            batch.verify_rotated.data(),
            batch.rabitq_records.data(),
            batch.subspace_cells.data());
    };
    append_batch(first);
    append_batch(second);
    return builder.finish(commit_lsn);
}

inline void encode_l0_inserts_contiguous_into(
    const BaseSegment& base,
    const RandomRotation& rotation,
    bool rotate_subspace,
    const RandomRotation* subspace_rotation,
    const float* rows,
    int n,
    int dim,
    const std::vector<uint32_t>& orig_ids,
    uint32_t pid_begin,
    EncodedL0InsertBatch& batch,
    L0EncodeScratch* scratch = nullptr) {
    if (n < 0 || dim <= 0 || (int)orig_ids.size() != n) {
        throw std::invalid_argument("insert vector/id count mismatch");
    }
    if (n > 0 && rows == nullptr) {
        throw std::invalid_argument("insert row pointer is null");
    }
    if (dim != base.dim) {
        throw std::runtime_error("insert vector dimension does not match index dimension");
    }
    if ((uint64_t)pid_begin + (uint64_t)n > (uint64_t)std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("insert batch exceeds uint32 PID range");
    }
    if (rotation.dim() != dim || rotation.padded_dim() != base.verify_vectors_dim ||
        rotation.padded_dim() != base.rabitq_padded_dim) {
        throw std::runtime_error("rotation dimensions do not match loaded index");
    }
    validate_update_row_cell_width(base);

    bool use_external_subspace_rotation = rotate_subspace && subspace_rotation != nullptr;
    if (use_external_subspace_rotation && subspace_rotation->dim() != dim) {
        throw std::runtime_error("subspace_rotation.bin dimension does not match update vectors");
    }

    batch.orig_ids.clear();
    batch.verify_rotated.clear();
    batch.rabitq_records.clear();
    batch.subspace_cells.clear();
    batch.pid_begin = pid_begin;
    batch.row_count = (uint32_t)n;
    batch.verify_dim = (uint32_t)base.verify_vectors_dim;
    batch.rabitq_record_bytes = (uint32_t)base.rabitq_bin_with_assign_record_bytes_;
    batch.num_subspaces = (uint32_t)base.ivf_indices.size();
    batch.orig_ids.assign(orig_ids.begin(), orig_ids.end());
    if (n == 0) return;

    batch.verify_rotated.resize((size_t)n * (size_t)base.verify_vectors_dim);
    std::vector<float>& rotated = batch.verify_rotated;
    if (update_vector_work_is_micro(n, base.rabitq_padded_dim)) {
        for (int i = 0; i < n; ++i) {
            rotation.apply(rows + (size_t)i * (size_t)dim,
                           rotated.data() + (size_t)i * (size_t)base.rabitq_padded_dim);
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            rotation.apply(rows + (size_t)i * (size_t)dim,
                           rotated.data() + (size_t)i * (size_t)base.rabitq_padded_dim);
        }
    }

    std::vector<float> local_subspace_rotated;
    std::vector<float>& subspace_rotated =
        scratch ? scratch->subspace_rotated : local_subspace_rotated;
    subspace_rotated.clear();
    const float* subspace_source = rows;
    int subspace_stride = dim;
    if (rotate_subspace) {
        if (use_external_subspace_rotation) {
            int padded = subspace_rotation->padded_dim();
            subspace_rotated.resize((size_t)n * (size_t)padded);
            if (update_vector_work_is_micro(n, padded)) {
                for (int i = 0; i < n; ++i) {
                    subspace_rotation->apply(
                        rows + (size_t)i * (size_t)dim,
                        subspace_rotated.data() + (size_t)i * (size_t)padded);
                }
            } else {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < n; ++i) {
                    subspace_rotation->apply(
                        rows + (size_t)i * (size_t)dim,
                        subspace_rotated.data() + (size_t)i * (size_t)padded);
                }
            }
            subspace_source = subspace_rotated.data();
            subspace_stride = padded;
        } else {
            subspace_source = rotated.data();
            subspace_stride = base.rabitq_padded_dim;
        }
    }

    std::vector<uint32_t> local_rabitq_assignments;
    std::vector<uint32_t>& rabitq_assignments =
        scratch ? scratch->rabitq_assignments : local_rabitq_assignments;
    assign_l0_rabitq_centroids(base, rotated.data(), n, rabitq_assignments);

    std::vector<int> local_submajor_cell_ids;
    std::vector<int>& submajor_cell_ids =
        scratch ? scratch->submajor_cell_ids : local_submajor_cell_ids;
    assign_l0_subspace_cells(base, subspace_source, subspace_stride, n, submajor_cell_ids);

    const size_t M = base.ivf_indices.size();
    batch.rabitq_records.assign(
        (size_t)n * (size_t)base.rabitq_bin_with_assign_record_bytes_, (uint8_t)0);
    batch.subspace_cells.resize((size_t)n * M);

    auto materialize_row = [&](int i) {
        const float* rot = rotated.data() + (size_t)i * (size_t)base.rabitq_padded_dim;

        uint32_t cid = rabitq_assignments[(size_t)i];
        uint8_t* record = batch.rabitq_records.data() +
            (size_t)i * (size_t)base.rabitq_bin_with_assign_record_bytes_;
        std::memcpy(record, &cid, sizeof(uint32_t));
        const float* c_rot = base.rabitq_rot_centroids.data() +
            (size_t)cid * (size_t)base.rabitq_padded_dim;
        rabitqlib::quant::quantize_compact_one_bit<float>(
            rot,
            c_rot,
            base.rabitq_padded_dim,
            reinterpret_cast<char*>(record + sizeof(uint32_t)),
            rabitqlib::METRIC_L2);

        uint16_t* cells = batch.subspace_cells.data() + (size_t)i * M;
        for (size_t m = 0; m < M; ++m) {
            int cell = submajor_cell_ids[m * (size_t)n + (size_t)i];
            if (cell < 0) {
                throw std::runtime_error("update subspace assignment produced negative cell");
            }
            if ((uint32_t)cell >= (uint32_t)std::numeric_limits<uint16_t>::max()) {
                throw std::runtime_error("update subspace assignment exceeds uint16 row-cell range");
            }
            cells[m] = (uint16_t)cell;
        }
    };
    size_t materialize_dim =
        (size_t)base.rabitq_padded_dim + (size_t)base.verify_vectors_dim + M;
    if (update_work_leq((size_t)n, materialize_dim, 1, kUpdateMicroVectorWork)) {
        for (int i = 0; i < n; ++i) materialize_row(i);
    } else {
        #pragma omp parallel for schedule(dynamic, 128)
        for (int i = 0; i < n; ++i) {
            materialize_row(i);
        }
    }
}

inline EncodedL0InsertBatch encode_l0_inserts_contiguous(
    const BaseSegment& base,
    const RandomRotation& rotation,
    bool rotate_subspace,
    const RandomRotation* subspace_rotation,
    const float* rows,
    int n,
    int dim,
    const std::vector<uint32_t>& orig_ids,
    uint32_t pid_begin) {
    EncodedL0InsertBatch batch;
    encode_l0_inserts_contiguous_into(
        base,
        rotation,
        rotate_subspace,
        subspace_rotation,
        rows,
        n,
        dim,
        orig_ids,
        pid_begin,
        batch);
    return batch;
}

inline std::vector<UpdateWalInsert> encode_l0_inserts(
    const BaseSegment& base,
    const RandomRotation& rotation,
    bool rotate_subspace,
    const RandomRotation* subspace_rotation,
    const float* rows,
    int n,
    int dim,
    const std::vector<uint32_t>& orig_ids,
    uint32_t pid_begin) {
    if (n < 0 || dim <= 0 || (int)orig_ids.size() != n) {
        throw std::invalid_argument("insert vector/id count mismatch");
    }
    if (n > 0 && rows == nullptr) {
        throw std::invalid_argument("insert row pointer is null");
    }
    if (dim != base.dim) {
        throw std::runtime_error("insert vector dimension does not match index dimension");
    }
    if ((uint64_t)pid_begin + (uint64_t)n > (uint64_t)std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("insert batch exceeds uint32 PID range");
    }
    if (rotation.dim() != dim || rotation.padded_dim() != base.verify_vectors_dim ||
        rotation.padded_dim() != base.rabitq_padded_dim) {
        throw std::runtime_error("rotation dimensions do not match loaded index");
    }
    validate_update_row_cell_width(base);

    bool use_external_subspace_rotation = rotate_subspace && subspace_rotation != nullptr;
    if (use_external_subspace_rotation) {
        if (subspace_rotation->dim() != dim) {
            throw std::runtime_error("subspace_rotation.bin dimension does not match update vectors");
        }
    }

    std::vector<float> rotated((size_t)n * (size_t)base.rabitq_padded_dim);
    if (update_vector_work_is_micro(n, base.rabitq_padded_dim)) {
        for (int i = 0; i < n; ++i) {
            rotation.apply(rows + (size_t)i * (size_t)dim,
                           rotated.data() + (size_t)i * (size_t)base.rabitq_padded_dim);
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            rotation.apply(rows + (size_t)i * (size_t)dim,
                           rotated.data() + (size_t)i * (size_t)base.rabitq_padded_dim);
        }
    }

    std::vector<float> subspace_rotated;
    const float* subspace_source = rows;
    int subspace_stride = dim;
    if (rotate_subspace) {
        if (use_external_subspace_rotation) {
            int padded = subspace_rotation->padded_dim();
            subspace_rotated.resize((size_t)n * (size_t)padded);
            if (update_vector_work_is_micro(n, padded)) {
                for (int i = 0; i < n; ++i) {
                    subspace_rotation->apply(
                        rows + (size_t)i * (size_t)dim,
                        subspace_rotated.data() + (size_t)i * (size_t)padded);
                }
            } else {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < n; ++i) {
                    subspace_rotation->apply(
                        rows + (size_t)i * (size_t)dim,
                        subspace_rotated.data() + (size_t)i * (size_t)padded);
                }
            }
            subspace_source = subspace_rotated.data();
            subspace_stride = padded;
        } else {
            subspace_source = rotated.data();
            subspace_stride = base.rabitq_padded_dim;
        }
    }

    const size_t M = base.ivf_indices.size();
    std::vector<uint32_t> rabitq_assignments;
    assign_l0_rabitq_centroids(base, rotated.data(), n, rabitq_assignments);

    std::vector<int> submajor_cell_ids;
    assign_l0_subspace_cells(base, subspace_source, subspace_stride, n, submajor_cell_ids);

    std::vector<UpdateWalInsert> inserts((size_t)n);

    auto materialize_insert = [&](int i) {
        const float* rot = rotated.data() + (size_t)i * (size_t)base.rabitq_padded_dim;
        uint32_t cid = rabitq_assignments[(size_t)i];

        UpdateWalInsert insert;
        insert.orig_id = orig_ids[(size_t)i];
        insert.global_pid = pid_begin + (uint32_t)i;
        insert.verify_rotated.assign(rot, rot + (size_t)base.verify_vectors_dim);
        insert.rabitq_record.assign(
            (size_t)base.rabitq_bin_with_assign_record_bytes_, (uint8_t)0);
        uint8_t* record = insert.rabitq_record.data();
        std::memcpy(record, &cid, sizeof(uint32_t));
        const float* c_rot = base.rabitq_rot_centroids.data() +
            (size_t)cid * (size_t)base.rabitq_padded_dim;
        rabitqlib::quant::quantize_compact_one_bit<float>(
            rot,
            c_rot,
            base.rabitq_padded_dim,
            reinterpret_cast<char*>(record + sizeof(uint32_t)),
            rabitqlib::METRIC_L2);

        insert.subspace_cells.resize(M);
        for (size_t m = 0; m < M; ++m) {
            int cell = submajor_cell_ids[m * (size_t)n + (size_t)i];
            if (cell < 0) {
                throw std::runtime_error("update subspace assignment produced negative cell");
            }
            if ((uint32_t)cell >= (uint32_t)std::numeric_limits<uint16_t>::max()) {
                throw std::runtime_error("update subspace assignment exceeds uint16 row-cell range");
            }
            insert.subspace_cells[m] = (uint16_t)cell;
        }
        inserts[(size_t)i] = std::move(insert);
    };
    size_t materialize_dim =
        (size_t)base.rabitq_padded_dim + (size_t)base.verify_vectors_dim + M;
    if (update_work_leq((size_t)n, materialize_dim, 1, kUpdateMicroVectorWork)) {
        for (int i = 0; i < n; ++i) materialize_insert(i);
    } else {
        #pragma omp parallel for schedule(dynamic, 128)
        for (int i = 0; i < n; ++i) {
            materialize_insert(i);
        }
    }
    return inserts;
}

inline EncodedL0InsertBatch encode_l0_inserts_contiguous(
    const std::string& index_path,
    const BaseSegment& base,
    const float* rows,
    int n,
    int dim,
    const std::vector<uint32_t>& orig_ids,
    uint32_t pid_begin) {
    RandomRotation rotation;
    rotation.load(index_path + "/rotation.bin");

    bool rotate_subspace = path_exists(index_path + "/rotate_subspace.flag");
    std::unique_ptr<RandomRotation> subspace_rotation;
    if (rotate_subspace && path_exists(index_path + "/subspace_rotation.bin")) {
        subspace_rotation = std::make_unique<RandomRotation>();
        subspace_rotation->load(index_path + "/subspace_rotation.bin");
    }

    return encode_l0_inserts_contiguous(
        base,
        rotation,
        rotate_subspace,
        subspace_rotation.get(),
        rows,
        n,
        dim,
        orig_ids,
        pid_begin);
}

inline std::vector<UpdateWalInsert> encode_l0_inserts(
    const std::string& index_path,
    const BaseSegment& base,
    const float* rows,
    int n,
    int dim,
    const std::vector<uint32_t>& orig_ids,
    uint32_t pid_begin) {
    RandomRotation rotation;
    rotation.load(index_path + "/rotation.bin");

    bool rotate_subspace = path_exists(index_path + "/rotate_subspace.flag");
    std::unique_ptr<RandomRotation> subspace_rotation;
    if (rotate_subspace && path_exists(index_path + "/subspace_rotation.bin")) {
        subspace_rotation = std::make_unique<RandomRotation>();
        subspace_rotation->load(index_path + "/subspace_rotation.bin");
    }

    return encode_l0_inserts(
        base,
        rotation,
        rotate_subspace,
        subspace_rotation.get(),
        rows,
        n,
        dim,
        orig_ids,
        pid_begin);
}

inline std::vector<UpdateWalInsert> encode_l0_inserts(
    const std::string& index_path,
    const BaseSegment& base,
    const std::vector<float>& rows,
    int n,
    int dim,
    const std::vector<uint32_t>& orig_ids,
    uint32_t pid_begin) {
    if (n < 0 || dim <= 0 || (int)orig_ids.size() != n) {
        throw std::invalid_argument("insert vector/id count mismatch");
    }
    if (rows.size() < (size_t)n * (size_t)dim) {
        throw std::invalid_argument("insert vector payload is shorter than n*dim");
    }
    return encode_l0_inserts(index_path, base, rows.data(), n, dim, orig_ids, pid_begin);
}

inline DirectMap build_direct_map_from_base_and_wal(const BaseSegment& base,
                                                    const UpdateManifest& manifest,
                                                    const std::string& index_path) {
    if (!base.perm_to_orig.empty() && base.perm_to_orig.size() < manifest.base_n) {
        throw std::runtime_error("perm_to_orig is shorter than manifest base_n");
    }

    std::shared_ptr<const LiveTable> checkpoint_live;
    bool checkpoint_exists = false;
    if (!manifest.checkpoint_path.empty()) {
        std::ifstream checkpoint_probe(join_index_path(index_path, manifest.checkpoint_path),
                                       std::ios::binary);
        checkpoint_exists = checkpoint_probe.good();
    }
    if (!manifest.sealed_deltas.empty() && !checkpoint_exists) {
        throw std::runtime_error("cannot rebuild DirectMap over sealed deltas without live checkpoint");
    }
    if (checkpoint_exists) {
        checkpoint_live = LiveTable::load_binary(join_index_path(index_path, manifest.checkpoint_path));
        if (checkpoint_live->max_pid() > manifest.max_pid) {
            throw std::runtime_error("cannot rebuild DirectMap: live checkpoint is ahead of manifest");
        }
    }

    DirectMap map = manifest.direct_map_mode == "sparse"
        ? DirectMap::sparse()
        : DirectMap::dense();
    map.reserve_entries((size_t)manifest.live_count);
    for (uint32_t pid = 0; pid < manifest.base_n; ++pid) {
        if (checkpoint_live && !checkpoint_live->is_live(pid)) continue;
        uint32_t orig = base.perm_to_orig.empty() ? pid : base.perm_to_orig[(size_t)pid];
        map.set(orig, pid);
    }

    if (!manifest.sealed_deltas.empty()) {
        std::vector<size_t> cells_per_subspace = cells_per_subspace_from_base(base);
        for (const UpdateManifestDelta& delta : manifest.sealed_deltas) {
            auto segment = SealedDeltaSegment::load(
                delta.segment_id,
                join_index_path(index_path, delta.path),
                cells_per_subspace,
                (uint32_t)base.verify_vectors_dim,
                (uint32_t)base.rabitq_bin_with_assign_record_bytes_,
                delta.pid_begin,
                delta.pid_count,
                delta.live_count);
            for (uint32_t row = 0; row < segment->row_count(); ++row) {
                uint32_t pid = segment->pid(row);
                if (!checkpoint_live->is_live(pid)) continue;
                map.set(segment->orig_id(row), pid);
            }
        }
    }

    if (manifest.last_committed_lsn == 0) {
        if (map.live_size() != manifest.live_count) {
            throw std::runtime_error("rebuilt DirectMap live size does not match manifest");
        }
        return map;
    }

    UpdateWalReplayResult replay =
        replay_update_wal_metadata(join_index_path(index_path, manifest.wal_path));
    if (replay.header.generation != manifest.generation ||
        replay.header.base_n != manifest.base_n ||
        replay.header.num_subspaces != (uint32_t)base.ivf_indices.size() ||
        replay.header.verify_dim != (uint32_t)base.verify_vectors_dim ||
        replay.header.rabitq_record_bytes != (uint32_t)base.rabitq_bin_with_assign_record_bytes_) {
        throw std::runtime_error("cannot rebuild DirectMap: WAL header does not match manifest");
    }
    if (replay.last_committed_lsn < manifest.last_committed_lsn) {
        throw std::runtime_error("cannot rebuild DirectMap: WAL committed LSN does not match manifest");
    }
    for (const UpdateWalBatch& batch : replay.committed_batches) {
        for (const UpdateWalRecord& record : batch.records) {
            if (record.type == UpdateWalRecord::Type::Delete) {
                map.erase(record.tombstone.orig_id);
            } else {
                map.set(record.insert.orig_id, record.insert.global_pid);
            }
        }
    }
    uint32_t effective_live_count =
        replay.last_committed_lsn != 0 ? replay.live_count : manifest.live_count;
    if (map.live_size() != effective_live_count) {
        throw std::runtime_error("rebuilt DirectMap live size does not match manifest");
    }
    return map;
}

inline std::shared_ptr<const LiveTable> load_update_live_table_from_manifest(
    const BaseSegment& base,
    UpdateManifest& manifest,
    const std::string& index_path) {
    bool checkpoint_exists = false;
    if (!manifest.checkpoint_path.empty()) {
        std::ifstream checkpoint_probe(join_index_path(index_path, manifest.checkpoint_path),
                                       std::ios::binary);
        checkpoint_exists = checkpoint_probe.good();
    }
    bool use_checkpoint = !manifest.sealed_deltas.empty() || checkpoint_exists;
    std::shared_ptr<const LiveTable> live;
    if (use_checkpoint) {
        if (manifest.checkpoint_path.empty()) {
            throw std::runtime_error("update manifest requires live checkpoint");
        }
        live = LiveTable::load_binary(join_index_path(index_path, manifest.checkpoint_path));
        if (live->max_pid() > manifest.max_pid) {
            throw std::runtime_error("live checkpoint is ahead of update manifest");
        }
    } else {
        live = LiveTable::all_live(manifest.base_n);
    }

    UpdateWalReplayResult replay =
        replay_update_wal_metadata(join_index_path(index_path, manifest.wal_path));
    if (replay.header.generation != manifest.generation ||
        replay.header.base_n != manifest.base_n ||
        replay.header.num_subspaces != (uint32_t)base.ivf_indices.size() ||
        replay.header.verify_dim != (uint32_t)base.verify_vectors_dim ||
        replay.header.rabitq_record_bytes !=
            (uint32_t)base.rabitq_bin_with_assign_record_bytes_) {
        throw std::runtime_error("update WAL header does not match lightweight update base");
    }
    if (replay.last_committed_lsn < manifest.last_committed_lsn) {
        throw std::runtime_error("update WAL committed LSN does not match manifest");
    }

    for (const UpdateWalBatch& batch : replay.committed_batches) {
        std::vector<uint32_t> set_live;
        std::vector<uint32_t> set_dead;
        set_live.reserve(batch.records.size());
        set_dead.reserve(batch.records.size());
        for (const UpdateWalRecord& record : batch.records) {
            if (record.type == UpdateWalRecord::Type::Insert) {
                if (record.insert.global_pid < manifest.base_n) {
                    throw std::runtime_error("update WAL insert PID overlaps base segment");
                }
                set_live.push_back(record.insert.global_pid);
            } else {
                set_dead.push_back(record.tombstone.old_global_pid);
            }
        }
        live = live->with_updates(batch.commit.max_pid, set_live, set_dead);
        if (live->live_count() != batch.commit.live_count) {
            throw std::runtime_error("update WAL live_count does not match replayed live table");
        }
    }

    if (replay.last_committed_lsn != 0 &&
        replay.last_committed_lsn == manifest.last_committed_lsn &&
        !use_checkpoint &&
        (replay.max_pid != manifest.max_pid || replay.live_count != manifest.live_count)) {
        throw std::runtime_error("update WAL replay state does not match manifest PID/live counters");
    }
    if (replay.last_committed_lsn == manifest.last_committed_lsn &&
        (live->max_pid() != manifest.max_pid || live->live_count() != manifest.live_count)) {
        throw std::runtime_error("update WAL replay state does not match manifest PID/live counters");
    }
    if (replay.last_committed_lsn > manifest.last_committed_lsn) {
        manifest.last_committed_lsn = replay.last_committed_lsn;
        manifest.max_pid = live->max_pid();
        manifest.live_count = live->live_count();
    }
    return live;
}

inline bool direct_map_matches_live_table(const DirectMap& map,
                                          const UpdateManifest& manifest,
                                          const LiveTable& live) {
    if ((manifest.direct_map_mode == "dense" && map.mode() != DirectMap::Mode::Dense) ||
        (manifest.direct_map_mode == "sparse" && map.mode() != DirectMap::Mode::Sparse) ||
        map.live_size() != manifest.live_count ||
        live.max_pid() != manifest.max_pid ||
        live.live_count() != manifest.live_count) {
        return false;
    }
    std::vector<uint8_t> seen((size_t)manifest.max_pid, 0);
    size_t visited = 0;
    bool valid = true;
    map.for_each_live_entry([&](uint32_t /*orig_id*/, uint32_t pid) {
        if (!valid) return;
        if (pid >= manifest.max_pid || !live.is_live(pid) || seen[(size_t)pid] != 0) {
            valid = false;
            return;
        }
        seen[(size_t)pid] = 1;
        ++visited;
    });
    return valid && visited == (size_t)manifest.live_count;
}

inline bool direct_map_matches_reader_view(const DirectMap& map,
                                           const UpdateManifest& manifest,
                                           const ReaderView& view) {
    if ((manifest.direct_map_mode == "dense" && map.mode() != DirectMap::Mode::Dense) ||
        (manifest.direct_map_mode == "sparse" && map.mode() != DirectMap::Mode::Sparse) ||
        map.live_size() != manifest.live_count ||
        view.max_pid != manifest.max_pid ||
        view.live_count != manifest.live_count ||
        !view.live) {
        return false;
    }

    std::vector<uint8_t> seen((size_t)view.max_pid, 0);
    size_t visited = 0;
    bool valid = true;
    map.for_each_live_entry([&](uint32_t /*orig_id*/, uint32_t pid) {
        if (!valid) return;
        if (pid >= view.max_pid || !view.live->is_live(pid) || seen[(size_t)pid] != 0) {
            valid = false;
            return;
        }
        seen[(size_t)pid] = 1;
        ++visited;
    });
    return valid && visited == (size_t)manifest.live_count;
}

inline std::string choose_direct_map_mode(const BaseSegment& base,
                                          const std::vector<uint32_t>& incoming_ids) {
    uint64_t max_orig = base.n > 0 ? (uint64_t)base.n - 1 : 0;
    for (uint32_t orig : base.perm_to_orig) max_orig = std::max(max_orig, (uint64_t)orig);
    for (uint32_t orig : incoming_ids) max_orig = std::max(max_orig, (uint64_t)orig);
    uint64_t dense_limit = std::max<uint64_t>((uint64_t)std::max(1, base.n) * 4u, 1u << 20);
    return max_orig <= dense_limit ? "dense" : "sparse";
}

inline bool build_last_wins_keep_mask(const std::vector<uint32_t>& ids,
                                      std::vector<uint8_t>& keep,
                                      std::vector<uint32_t>& kept_ids) {
    keep.clear();
    kept_ids.clear();
    if (ids.size() <= 1) return false;

    bool strictly_increasing = true;
    for (size_t i = 1; i < ids.size(); ++i) {
        if (ids[i] <= ids[i - 1]) {
            strictly_increasing = false;
            break;
        }
    }
    if (strictly_increasing) return false;

    if (ids.size() <= kUpdateLastWinsSmallScanRows) {
        bool has_duplicate = false;
        keep.assign(ids.size(), 1);
        for (size_t i = 0; i < ids.size(); ++i) {
            for (size_t j = i + 1; j < ids.size(); ++j) {
                if (ids[i] != ids[j]) continue;
                keep[i] = 0;
                has_duplicate = true;
                break;
            }
        }
        if (!has_duplicate) return false;
        kept_ids.reserve(ids.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            if (keep[i] == 0) continue;
            kept_ids.push_back(ids[i]);
        }
        return true;
    }

    if (ids.size() > (size_t)std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("update batch is too large for v1 last-wins coalescing");
    }
    SparseDirectMapTable last_index;
    last_index.reserve(ids.size());
    bool has_duplicate = false;
    for (size_t i = 0; i < ids.size(); ++i) {
        uint32_t current_index = (uint32_t)i;
        if (!last_index.set(ids[i], current_index)) has_duplicate = true;
    }
    if (!has_duplicate) return false;

    keep.assign(ids.size(), 0);
    kept_ids.reserve(last_index.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        uint32_t last_seen = 0;
        if (!last_index.lookup(ids[i], last_seen) || last_seen != (uint32_t)i) continue;
        keep[i] = 1;
        kept_ids.push_back(ids[i]);
    }
    return true;
}

inline bool coalesce_memory_rows_last_wins(const float* rows,
                                           int n,
                                           int dim,
                                           const std::vector<uint32_t>& ids,
                                           std::vector<uint32_t>& kept_ids,
                                           std::vector<float>& kept_rows) {
    std::vector<uint8_t> keep;
    if (!build_last_wins_keep_mask(ids, keep, kept_ids)) {
        kept_rows.clear();
        return false;
    }
    if (rows == nullptr || n < 0 || dim <= 0 || (size_t)n != ids.size()) {
        throw std::runtime_error("cannot coalesce inconsistent update rows");
    }
    kept_rows.resize(kept_ids.size() * (size_t)dim);
    size_t out_row = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (keep[i] == 0) continue;
        const float* src = rows + i * (size_t)dim;
        float* dst = kept_rows.data() + out_row * (size_t)dim;
        std::memcpy(dst, src, (size_t)dim * sizeof(float));
        ++out_row;
    }
    return true;
}

inline void validate_no_insert_update_id_conflicts(
    const std::vector<uint32_t>& insert_ids,
    const std::vector<uint32_t>& upsert_ids) {
    if (insert_ids.empty()) return;
    if (insert_ids.size() <= kUpdateLastWinsSmallScanRows) {
        for (size_t i = 0; i < insert_ids.size(); ++i) {
            uint32_t orig_id = insert_ids[i];
            for (size_t j = i + 1; j < insert_ids.size(); ++j) {
                if (orig_id == insert_ids[j]) {
                    throw std::runtime_error("duplicate insert orig_id in one batch: " +
                                             std::to_string(orig_id));
                }
            }
            for (uint32_t upsert_orig_id : upsert_ids) {
                if (orig_id == upsert_orig_id) {
                    throw std::runtime_error("insert/upsert orig_id overlap in one batch: " +
                                             std::to_string(orig_id));
                }
            }
        }
        return;
    }

    SparseDirectMapTable seen_insert_ids;
    seen_insert_ids.reserve(insert_ids.size());
    for (uint32_t orig_id : insert_ids) {
        if (!seen_insert_ids.insert_new(orig_id, 1)) {
            throw std::runtime_error("duplicate insert orig_id in one batch: " +
                                     std::to_string(orig_id));
        }
    }
    uint32_t ignored = 0;
    for (uint32_t orig_id : upsert_ids) {
        if (seen_insert_ids.lookup(orig_id, ignored)) {
            throw std::runtime_error("insert/upsert orig_id overlap in one batch: " +
                                     std::to_string(orig_id));
        }
    }
}

} // namespace impl

struct UpdateIndexBatch {
    std::string insert_fbin;
    std::string insert_ids;
    std::string delete_ids;
    std::string upsert_fbin;
    std::string upsert_ids;
    std::string durability_mode = "buffered";
    int threads = 0;
    uint32_t batch_rows = 0;
};

struct UpdateIndexMemoryBatch {
    const float* insert_rows = nullptr;
    int insert_n = 0;
    int insert_dim = 0;
    std::vector<uint32_t> insert_orig_ids;

    const float* upsert_rows = nullptr;
    int upsert_n = 0;
    int upsert_dim = 0;
    std::vector<uint32_t> upsert_orig_ids;

    std::vector<uint32_t> delete_orig_ids;
};

struct UpdateCommitTiming {
    double total_ms = 0.0;
    double prepare_ms = 0.0;
    double encode_ms = 0.0;
    double wal_commit_ms = 0.0;
    double publish_ms = 0.0;
    bool published = false;
    bool direct_publish = false;
    uint32_t insert_rows = 0;
    uint32_t upsert_rows = 0;
    uint32_t delete_rows = 0;
    int encode_threads = 1;
};

namespace impl {

inline int adaptive_update_threads_for_batch(const BaseSegment& base,
                                             int configured_threads,
                                             const UpdateIndexMemoryBatch& batch) {
    int max_threads = configured_threads > 0 ? configured_threads : omp_get_max_threads();
    max_threads = std::max(1, max_threads);
    size_t rows = 0;
    if (batch.insert_n > 0) rows = add_saturated_size(rows, (size_t)batch.insert_n);
    if (batch.upsert_n > 0) rows = add_saturated_size(rows, (size_t)batch.upsert_n);
    if (rows == 0) return 1;
    size_t work_per_row = estimate_update_encode_work_per_row(base);
    size_t total_work = mul_saturated_size(rows, work_per_row);
    return adaptive_update_threads_for_work(total_work, max_threads);
}

} // namespace impl

class UpdateIndexWriter {
public:
    UpdateIndexWriter(const std::string& index_path,
                      std::string durability_mode = "buffered",
                      int threads = 0,
                      const std::vector<uint32_t>& incoming_ids_hint = {},
                      QueryIndexHandle* publish_handle = nullptr)
        : index_path_(index_path),
          durability_mode_(std::move(durability_mode)),
          configured_threads_(threads),
          thread_limit_(threads),
          publish_handle_(publish_handle),
          writer_lock_(index_path) {
        {
            impl::UpdatePublishWriteLock publish_lock(index_path_);
        }
        if (!is_valid_update_durability_mode(durability_mode_)) {
            throw std::runtime_error("invalid update durability mode: " + durability_mode_);
        }
        base_ = impl::load_update_base_segment_lightweight(index_path_);
        if (base_.n < 0 ||
            (uint64_t)base_.n > (uint64_t)std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("update base segment size exceeds v1 uint32 PID range");
        }
        loaded_base_n_ = (uint32_t)base_.n;
        rotation_.load(index_path_ + "/rotation.bin");
        if (rotation_.dim() != base_.dim ||
            rotation_.padded_dim() != base_.verify_vectors_dim ||
            rotation_.padded_dim() != base_.rabitq_padded_dim) {
            throw std::runtime_error("rotation dimensions do not match loaded update base");
        }
        rotate_subspace_ = impl::path_exists(index_path_ + "/rotate_subspace.flag");
        if (rotate_subspace_ &&
            impl::path_exists(index_path_ + "/subspace_rotation.bin")) {
            subspace_rotation_ = std::make_unique<RandomRotation>();
            subspace_rotation_->load(index_path_ + "/subspace_rotation.bin");
            if (subspace_rotation_->dim() != base_.dim) {
                throw std::runtime_error("subspace_rotation.bin dimension does not match update base");
            }
        }

        std::shared_ptr<const LiveTable> live = LiveTable::all_live(loaded_base_n_);
        has_manifest_ = update_manifest_exists(index_path_);
        if (has_manifest_) {
            manifest_ = load_update_manifest_json_for_index(index_path_);
            if (manifest_.base_path != "." || manifest_.base_n != loaded_base_n_) {
                throw std::runtime_error("loaded lightweight update base does not match update manifest");
            }
            live = impl::load_update_live_table_from_manifest(base_, manifest_, index_path_);
            manifest_.durability_mode = durability_mode_;
        } else {
            manifest_.format = "score_update_manifest_v1";
            manifest_.generation = 1;
            manifest_.base_path = ".";
            manifest_.base_n = loaded_base_n_;
            manifest_.max_pid = loaded_base_n_;
            manifest_.live_count = loaded_base_n_;
            manifest_.wal_path = "updates/l0.wal";
            manifest_.checkpoint_path = "updates/l0.checkpoint";
            manifest_.direct_map_mode =
                impl::choose_direct_map_mode(base_, incoming_ids_hint);
            manifest_.direct_map_path = "updates/direct_map.bin";
            manifest_.direct_map_optional_for_search = true;
            manifest_.durability_mode = durability_mode_;
            manifest_.last_committed_lsn = 0;
            manifest_.last_fsynced_lsn = 0;
        }

        fsync_each_commit_ = update_durability_fsync_each_commit(durability_mode_);
        final_buffered_fsync_ = update_durability_flush_at_buffer_boundary(durability_mode_);
        updates_dir_ = index_path_ + "/updates";
        impl::ensure_dir(updates_dir_);
        impl::fsync_parent_dir_if_requested(
            updates_dir_, update_durability_fsync_artifacts(durability_mode_));

        direct_map_path_ =
            impl::join_index_path(index_path_, manifest_.direct_map_path);
        direct_map_ = manifest_.direct_map_mode == "sparse"
            ? DirectMap::sparse()
            : DirectMap::dense();
        if (impl::path_exists(direct_map_path_)) {
            try {
                direct_map_ = DirectMap::load_binary(direct_map_path_);
            } catch (...) {
                direct_map_ =
                    impl::build_direct_map_from_base_and_wal(base_, manifest_, index_path_);
            }
            if (!impl::direct_map_matches_live_table(direct_map_, manifest_, *live)) {
                direct_map_ =
                    impl::build_direct_map_from_base_and_wal(base_, manifest_, index_path_);
            }
        } else {
            direct_map_ =
                impl::build_direct_map_from_base_and_wal(base_, manifest_, index_path_);
        }
        if (!impl::direct_map_matches_live_table(direct_map_, manifest_, *live)) {
            throw std::runtime_error("rebuilt DirectMap does not match current live table");
        }
        direct_map_.reserve_entries(direct_map_.live_size() + incoming_ids_hint.size());

        current_max_pid_ = manifest_.max_pid;
        current_live_count_ = manifest_.live_count;
        next_lsn_ = manifest_.last_committed_lsn + 1;

        header_.generation = manifest_.generation;
        header_.base_n = manifest_.base_n;
        header_.num_subspaces = (uint32_t)base_.ivf_indices.size();
        header_.verify_dim = (uint32_t)base_.verify_vectors_dim;
        header_.rabitq_record_bytes =
            (uint32_t)base_.rabitq_bin_with_assign_record_bytes_;

        wal_path_ = impl::join_index_path(index_path_, manifest_.wal_path);
        if (has_manifest_ && !impl::path_exists(wal_path_)) {
            throw std::runtime_error("update manifest references missing WAL: " + wal_path_);
        }
    }

    UpdateIndexWriter(const UpdateIndexWriter&) = delete;
    UpdateIndexWriter& operator=(const UpdateIndexWriter&) = delete;

    ~UpdateIndexWriter() = default;

    void set_publish_handle(QueryIndexHandle* publish_handle) {
        publish_handle_ = publish_handle;
        publish_callback_ = nullptr;
    }

    void set_publish_callback(std::function<void()> publish_callback) {
        publish_callback_ = std::move(publish_callback);
        publish_handle_ = nullptr;
    }

    void commit_batch(const UpdateIndexMemoryBatch& batch) {
        if (collect_commit_timing_) {
            UpdateCommitTiming timing;
            commit_batch_impl<true>(batch, &timing);
            last_commit_timing_ = timing;
        } else {
            commit_batch_impl<false>(batch, nullptr);
        }
    }

    void set_collect_commit_timing(bool enabled) {
        collect_commit_timing_ = enabled;
        if (!enabled) last_commit_timing_ = {};
    }

    bool collect_commit_timing() const {
        return collect_commit_timing_;
    }

    const UpdateCommitTiming& last_commit_timing() const {
        return last_commit_timing_;
    }

    const CommittedUpdateDelta& last_commit_delta() const {
        return last_commit_delta_;
    }

    void flush_buffered_commits() {
        if (!published_any_records_) return;
        impl::UpdatePublishWriteLock publish_lock(index_path_);
        flush_buffered_commits_unlocked();
    }

    const UpdateManifest& manifest() const { return manifest_; }

    bool requires_reopen_after_compaction() const {
        return requires_reopen_after_compaction_;
    }

    template <typename Fn>
    auto run_locked_maintenance(Fn&& fn) -> decltype(fn()) {
        if (requires_reopen_after_compaction_) {
            throw std::runtime_error("UpdateIndexWriter must be reopened after compaction");
        }
        auto decision = [&]() {
            impl::UpdatePublishWriteLock publish_lock(index_path_);
            flush_buffered_commits_unlocked();
            wal_writer_.reset();
            auto locked_decision = fn();
            if (locked_decision.compacted) {
                requires_reopen_after_compaction_ = true;
                return locked_decision;
            }
            if (locked_decision.changed()) {
                reload_update_state_after_locked_maintenance();
            } else {
                direct_map_dirty_ = false;
                published_any_records_ = false;
            }
            return locked_decision;
        }();
        if (decision.changed() && !decision.compacted) {
            publish_to_query_engine();
        }
        return decision;
    }

private:
    template <bool CollectTiming>
    void commit_batch_impl(const UpdateIndexMemoryBatch& batch, UpdateCommitTiming* timing) {
        (void)timing;
        last_commit_delta_ = {};
        using Clock = std::chrono::steady_clock;
        [[maybe_unused]] Clock::time_point total_start;
        [[maybe_unused]] Clock::time_point encode_start;
        [[maybe_unused]] Clock::time_point wal_start;
        [[maybe_unused]] Clock::time_point publish_start;
        if constexpr (CollectTiming) {
            total_start = Clock::now();
            timing->insert_rows = (uint32_t)batch.insert_orig_ids.size();
            timing->upsert_rows = (uint32_t)batch.upsert_orig_ids.size();
            timing->delete_rows = (uint32_t)batch.delete_orig_ids.size();
        }

        validate_memory_batch(batch);
        int adaptive_threads =
            impl::adaptive_update_threads_for_batch(base_, configured_threads_, batch);
        if constexpr (CollectTiming) {
            timing->encode_threads = adaptive_threads > 0 ? adaptive_threads
                                                          : std::max(1, omp_get_max_threads());
        }
        int current_threads = std::max(1, omp_get_max_threads());
        impl::ScopedUpdateThreadLimit adaptive_thread_limit(
            adaptive_threads < current_threads ? adaptive_threads : 0);

        std::vector<uint32_t> coalesced_upsert_ids;
        std::vector<float> coalesced_upsert_rows;
        const std::vector<uint32_t>* effective_upsert_ids = &batch.upsert_orig_ids;
        const float* effective_upsert_rows = batch.upsert_rows;
        int effective_upsert_n = batch.upsert_n;
        if (!batch.upsert_orig_ids.empty() &&
            impl::coalesce_memory_rows_last_wins(
                batch.upsert_rows,
                batch.upsert_n,
                batch.upsert_dim,
                batch.upsert_orig_ids,
                coalesced_upsert_ids,
                coalesced_upsert_rows)) {
            effective_upsert_ids = &coalesced_upsert_ids;
            effective_upsert_rows = coalesced_upsert_rows.data();
            effective_upsert_n = (int)coalesced_upsert_ids.size();
        }

        records_scratch_.clear();
        records_scratch_.reserve(batch.delete_orig_ids.size() + effective_upsert_ids->size());
        std::vector<UpdateWalRecord>& records = records_scratch_;

        for (uint32_t orig_id : batch.delete_orig_ids) append_delete(orig_id, records);

        if constexpr (CollectTiming) {
            encode_start = Clock::now();
            timing->prepare_ms =
                std::chrono::duration<double, std::milli>(encode_start - total_start).count();
        }

        insert_batch_scratch_.row_count = 0;
        upsert_batch_scratch_.row_count = 0;
        if (!batch.insert_orig_ids.empty()) {
            for (uint32_t orig_id : batch.insert_orig_ids) {
                if (direct_map_.contains(orig_id)) {
                    throw std::runtime_error("insert ID already exists; use upsert for orig_id " +
                                             std::to_string(orig_id));
                }
            }
            impl::encode_l0_inserts_contiguous_into(
                base_,
                rotation_,
                rotate_subspace_,
                subspace_rotation_.get(),
                batch.insert_rows,
                batch.insert_n,
                batch.insert_dim,
                batch.insert_orig_ids,
                current_max_pid_,
                insert_batch_scratch_,
                &insert_encode_scratch_);
            current_max_pid_ += (uint32_t)insert_batch_scratch_.size();
            stage_encoded_inserts(insert_batch_scratch_);
        }

        if (!effective_upsert_ids->empty()) {
            impl::encode_l0_inserts_contiguous_into(
                base_,
                rotation_,
                rotate_subspace_,
                subspace_rotation_.get(),
                effective_upsert_rows,
                effective_upsert_n,
                batch.upsert_dim,
                *effective_upsert_ids,
                current_max_pid_,
                upsert_batch_scratch_,
                &upsert_encode_scratch_);
            current_max_pid_ += (uint32_t)upsert_batch_scratch_.size();
            stage_encoded_upserts(upsert_batch_scratch_, records);
        }

        if constexpr (CollectTiming) {
            wal_start = Clock::now();
            timing->encode_ms =
                std::chrono::duration<double, std::milli>(wal_start - encode_start).count();
        }

        bool committed = false;
        if (publish_records_and_encoded_inserts(
                records, insert_batch_scratch_, upsert_batch_scratch_)) {
            committed = true;
            if constexpr (CollectTiming) {
                publish_start = Clock::now();
                timing->wal_commit_ms =
                    std::chrono::duration<double, std::milli>(publish_start - wal_start).count();
            }
            bool direct_publish = publish_to_query_engine();
            if constexpr (CollectTiming) timing->direct_publish = direct_publish;
        }
        if constexpr (CollectTiming) {
            auto total_end = Clock::now();
            if (!committed) {
                timing->wal_commit_ms =
                    std::chrono::duration<double, std::milli>(total_end - wal_start).count();
            } else {
                timing->published = true;
                timing->publish_ms =
                    std::chrono::duration<double, std::milli>(total_end - publish_start).count();
            }
            timing->total_ms =
                std::chrono::duration<double, std::milli>(total_end - total_start).count();
        }
    }

    bool publish_to_query_engine() {
        if (publish_callback_) {
            publish_callback_();
            return false;
        } else if (publish_handle_ != nullptr) {
            return publish_handle_->publish_committed_update_delta(manifest_, last_commit_delta_);
        }
        return false;
    }

    void flush_buffered_commits_unlocked() {
        if (!published_any_records_) return;
        if (final_buffered_fsync_) {
            impl::fsync_file_if_requested(wal_path_, true);
            impl::fsync_parent_dir_if_requested(wal_path_, true);
        }
        if (direct_map_dirty_) {
            direct_map_.save_binary(direct_map_path_, final_buffered_fsync_);
            direct_map_dirty_ = false;
        }
        if (final_buffered_fsync_) manifest_.last_fsynced_lsn = manifest_.last_committed_lsn;
        if (final_buffered_fsync_ || durability_mode_ == "none") {
            write_update_manifest_json_for_index(manifest_, index_path_, final_buffered_fsync_);
        }
        published_any_records_ = false;
    }

    void reload_update_state_after_locked_maintenance() {
        if (!update_manifest_exists(index_path_)) {
            throw std::runtime_error("non-compaction maintenance unexpectedly removed update manifest");
        }
        manifest_ = load_update_manifest_json_for_index(index_path_);
        if (manifest_.base_path != "." || manifest_.base_n != loaded_base_n_) {
            throw std::runtime_error("maintenance update manifest no longer matches writer base");
        }
        manifest_.durability_mode = durability_mode_;
        has_manifest_ = true;
        current_max_pid_ = manifest_.max_pid;
        current_live_count_ = manifest_.live_count;
        next_lsn_ = manifest_.last_committed_lsn + 1;

        direct_map_path_ =
            impl::join_index_path(index_path_, manifest_.direct_map_path);
        wal_path_ = impl::join_index_path(index_path_, manifest_.wal_path);
        if (!impl::path_exists(wal_path_)) {
            throw std::runtime_error("maintenance update manifest references missing WAL: " + wal_path_);
        }

        header_.generation = manifest_.generation;
        header_.base_n = manifest_.base_n;
        header_.num_subspaces = (uint32_t)base_.ivf_indices.size();
        header_.verify_dim = (uint32_t)base_.verify_vectors_dim;
        header_.rabitq_record_bytes =
            (uint32_t)base_.rabitq_bin_with_assign_record_bytes_;
        direct_map_dirty_ = false;
        published_any_records_ = false;
    }

    void validate_memory_batch(const UpdateIndexMemoryBatch& batch) const {
        bool has_insert = !batch.insert_orig_ids.empty();
        bool has_upsert = !batch.upsert_orig_ids.empty();
        bool has_delete = !batch.delete_orig_ids.empty();
        if (!has_insert && !has_upsert && !has_delete) {
            throw std::runtime_error("no update records requested");
        }
        if (has_insert) {
            if (batch.insert_rows == nullptr || batch.insert_n <= 0 || batch.insert_dim <= 0 ||
                (size_t)batch.insert_n != batch.insert_orig_ids.size()) {
                throw std::runtime_error("insert memory batch has inconsistent rows/IDs");
            }
        }
        if (has_upsert) {
            if (batch.upsert_rows == nullptr || batch.upsert_n <= 0 || batch.upsert_dim <= 0 ||
                (size_t)batch.upsert_n != batch.upsert_orig_ids.size()) {
                throw std::runtime_error("upsert memory batch has inconsistent rows/IDs");
            }
        }
        impl::validate_no_insert_update_id_conflicts(
            batch.insert_orig_ids, batch.upsert_orig_ids);
    }

    void decrement_sealed_delta_live_count(uint32_t old_pid) {
        if (old_pid < manifest_.base_n) return;
        for (UpdateManifestDelta& delta : manifest_.sealed_deltas) {
            uint64_t begin = delta.pid_begin;
            uint64_t end = begin + (uint64_t)delta.pid_count;
            if ((uint64_t)old_pid < begin || (uint64_t)old_pid >= end) continue;
            if (delta.live_count == 0) {
                throw std::runtime_error("sealed delta live_count underflow while deleting PID " +
                                         std::to_string(old_pid));
            }
            --delta.live_count;
            return;
        }
        if (old_pid < current_max_pid_) return;
        throw std::runtime_error("delta PID is not covered by any sealed delta manifest entry");
    }

    void append_delete(uint32_t orig_id, std::vector<UpdateWalRecord>& records) {
        uint32_t old_pid = 0;
        if (!direct_map_.erase(orig_id, &old_pid)) return;
        UpdateWalRecord record;
        record.type = UpdateWalRecord::Type::Delete;
        record.lsn = next_lsn_++;
        record.tombstone.orig_id = orig_id;
        record.tombstone.old_global_pid = old_pid;
        records.push_back(record);
        direct_map_dirty_ = true;
        decrement_sealed_delta_live_count(old_pid);
        --current_live_count_;
    }

    void append_tombstone_record(uint32_t orig_id,
                                 uint32_t old_pid,
                                 std::vector<UpdateWalRecord>& records) {
        UpdateWalRecord record;
        record.type = UpdateWalRecord::Type::Delete;
        record.lsn = next_lsn_++;
        record.tombstone.orig_id = orig_id;
        record.tombstone.old_global_pid = old_pid;
        records.push_back(record);
        direct_map_dirty_ = true;
        decrement_sealed_delta_live_count(old_pid);
        --current_live_count_;
    }

    void append_inserts(std::vector<UpdateWalInsert>& inserts,
                        std::vector<UpdateWalRecord>& records) {
        for (UpdateWalInsert& insert : inserts) {
            UpdateWalRecord record;
            record.type = UpdateWalRecord::Type::Insert;
            record.lsn = next_lsn_++;
            record.insert = std::move(insert);
            direct_map_.set(record.insert.orig_id, record.insert.global_pid);
            direct_map_dirty_ = true;
            ++current_live_count_;
            records.push_back(std::move(record));
        }
    }

    void stage_encoded_inserts(const impl::EncodedL0InsertBatch& inserts) {
        for (size_t row = 0; row < inserts.size(); ++row) {
            direct_map_.set(inserts.orig_ids[row], inserts.global_pid(row));
            direct_map_dirty_ = true;
            ++current_live_count_;
        }
    }

    void stage_encoded_upserts(const impl::EncodedL0InsertBatch& inserts,
                               std::vector<UpdateWalRecord>& records) {
        for (size_t row = 0; row < inserts.size(); ++row) {
            uint32_t old_pid = 0;
            bool had_old = direct_map_.set_with_old_pid(
                inserts.orig_ids[row], inserts.global_pid(row), &old_pid);
            direct_map_dirty_ = true;
            if (had_old) append_tombstone_record(inserts.orig_ids[row], old_pid, records);
            ++current_live_count_;
        }
    }

    void populate_commit_delta_from_records(CommittedUpdateDelta& delta,
                                            const std::vector<UpdateWalRecord>& records,
                                            uint32_t previous_max_pid) const {
        for (const UpdateWalRecord& record : records) {
            if (record.type == UpdateWalRecord::Type::Delete) {
                uint32_t pid = record.tombstone.old_global_pid;
                delta.committed_delete = true;
                delta.committed_dead_pids.push_back(pid);
                if (pid < manifest_.base_n) {
                    delta.committed_base_live_change = true;
                    ++delta.committed_base_live_deletes;
                }
            } else {
                uint32_t pid = record.insert.global_pid;
                delta.committed_insert = true;
                delta.set_live_pids.push_back(pid);
                if (pid < previous_max_pid) delta.committed_insert_reused_pid = true;
            }
        }
    }

    void populate_commit_delta_from_encoded_batch(
        CommittedUpdateDelta& delta,
        const impl::EncodedL0InsertBatch& batch,
        uint32_t previous_max_pid) const {
        for (size_t row = 0; row < batch.size(); ++row) {
            uint32_t pid = batch.global_pid(row);
            delta.committed_insert = true;
            delta.set_live_pids.push_back(pid);
            if (pid < previous_max_pid) delta.committed_insert_reused_pid = true;
        }
    }

    CommittedUpdateDelta make_commit_delta_header(const UpdateWalCommit& commit,
                                                  uint64_t wal_offset_after_commit) const {
        CommittedUpdateDelta delta;
        delta.header = header_;
        delta.commit_lsn = commit.lsn;
        delta.wal_offset_after_commit = wal_offset_after_commit;
        delta.max_pid = commit.max_pid;
        delta.live_count = commit.live_count;
        return delta;
    }

    bool publish_records(const std::vector<UpdateWalRecord>& records) {
        if (records.empty()) return false;
        impl::UpdatePublishWriteLock publish_lock(index_path_);
        uint32_t previous_max_pid = manifest_.max_pid;
        uint64_t commit_lsn = next_lsn_++;
        UpdateWalWriter& writer = ensure_wal_writer();
        for (const UpdateWalRecord& record : records) {
            if (record.type == UpdateWalRecord::Type::Delete) {
                writer.append_delete(record.lsn, record.tombstone);
            } else {
                writer.append_insert(record.lsn, record.insert);
            }
        }
        UpdateWalCommit commit;
        commit.lsn = commit_lsn;
        commit.batch_record_count = (uint32_t)records.size();
        commit.max_pid = current_max_pid_;
        commit.live_count = current_live_count_;
        writer.append_commit(commit);
        writer.flush();
        uint64_t wal_offset_after_commit = writer.current_offset();

        impl::fsync_file_if_requested(wal_path_, fsync_each_commit_);
        impl::fsync_parent_dir_if_requested(wal_path_, fsync_each_commit_);
        if (fsync_each_commit_ && direct_map_dirty_) {
            direct_map_.save_binary(direct_map_path_, true);
            direct_map_dirty_ = false;
        }

        manifest_.max_pid = current_max_pid_;
        manifest_.live_count = current_live_count_;
        manifest_.last_committed_lsn = commit_lsn;
        if (fsync_each_commit_) manifest_.last_fsynced_lsn = commit_lsn;
        bool publish_manifest_now = !has_manifest_ || fsync_each_commit_;
        if (publish_manifest_now) {
            write_update_manifest_json_for_index(manifest_, index_path_, fsync_each_commit_);
        }
        has_manifest_ = true;
        published_any_records_ = true;
        last_commit_delta_ = make_commit_delta_header(commit, wal_offset_after_commit);
        populate_commit_delta_from_records(
            last_commit_delta_, records, previous_max_pid);
        if (publish_handle_ != nullptr && last_commit_delta_.committed_insert) {
            std::vector<UpdateWalInsert> inserts;
            inserts.reserve(records.size());
            for (const UpdateWalRecord& record : records) {
                if (record.type == UpdateWalRecord::Type::Insert) {
                    inserts.push_back(record.insert);
                }
            }
            if (!inserts.empty()) {
                last_commit_delta_.committed_blocks.push_back(
                    L0Block::from_inserts(
                        commit.lsn,
                        inserts,
                        impl::cells_per_subspace_from_base(base_),
                        header_.verify_dim,
                        header_.rabitq_record_bytes));
            }
        }
        return true;
    }

    bool publish_records_and_encoded_inserts(
        const std::vector<UpdateWalRecord>& records,
        const impl::EncodedL0InsertBatch& insert_batch,
        const impl::EncodedL0InsertBatch& upsert_batch) {
        size_t record_count = records.size() + insert_batch.size() + upsert_batch.size();
        if (record_count == 0) return false;
        if (record_count > (size_t)std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("update batch record count exceeds uint32 range");
        }

        impl::UpdatePublishWriteLock publish_lock(index_path_);
        uint32_t previous_max_pid = manifest_.max_pid;
        UpdateWalWriter& writer = ensure_wal_writer();
        for (const UpdateWalRecord& record : records) {
            if (record.type == UpdateWalRecord::Type::Delete) {
                writer.append_delete(record.lsn, record.tombstone);
            } else {
                writer.append_insert(record.lsn, record.insert);
            }
        }

        auto append_encoded_batch = [&](const impl::EncodedL0InsertBatch& batch) {
            if (batch.size() == 0) return;
            writer.append_insert_rows_batch(
                next_lsn_,
                batch.size(),
                batch.orig_ids.data(),
                batch.pid_begin,
                batch.verify_rotated.data(),
                batch.rabitq_records.data(),
                batch.subspace_cells.data());
            next_lsn_ += batch.size();
        };
        append_encoded_batch(insert_batch);
        append_encoded_batch(upsert_batch);

        uint64_t commit_lsn = next_lsn_++;
        UpdateWalCommit commit;
        commit.lsn = commit_lsn;
        commit.batch_record_count = (uint32_t)record_count;
        commit.max_pid = current_max_pid_;
        commit.live_count = current_live_count_;
        writer.append_commit(commit);
        writer.flush();
        uint64_t wal_offset_after_commit = writer.current_offset();

        impl::fsync_file_if_requested(wal_path_, fsync_each_commit_);
        impl::fsync_parent_dir_if_requested(wal_path_, fsync_each_commit_);
        if (fsync_each_commit_ && direct_map_dirty_) {
            direct_map_.save_binary(direct_map_path_, true);
            direct_map_dirty_ = false;
        }

        manifest_.max_pid = current_max_pid_;
        manifest_.live_count = current_live_count_;
        manifest_.last_committed_lsn = commit_lsn;
        if (fsync_each_commit_) manifest_.last_fsynced_lsn = commit_lsn;
        bool publish_manifest_now = !has_manifest_ || fsync_each_commit_;
        if (publish_manifest_now) {
            write_update_manifest_json_for_index(manifest_, index_path_, fsync_each_commit_);
        }
        has_manifest_ = true;
        published_any_records_ = true;
        last_commit_delta_ = make_commit_delta_header(commit, wal_offset_after_commit);
        populate_commit_delta_from_records(
            last_commit_delta_, records, previous_max_pid);
        populate_commit_delta_from_encoded_batch(
            last_commit_delta_, insert_batch, previous_max_pid);
        populate_commit_delta_from_encoded_batch(
            last_commit_delta_, upsert_batch, previous_max_pid);
        if (publish_handle_ != nullptr && last_commit_delta_.committed_insert) {
            std::shared_ptr<const L0Block> block =
                impl::build_l0_block_from_encoded_batches(
                    commit.lsn,
                    impl::cells_per_subspace_from_base(base_),
                    insert_batch,
                    upsert_batch);
            if (block) last_commit_delta_.committed_blocks.push_back(std::move(block));
        }
        return true;
    }

    UpdateWalWriter& ensure_wal_writer() {
        if (!wal_writer_) {
            wal_writer_ = std::make_unique<UpdateWalWriter>(
                wal_path_, header_, has_manifest_);
        }
        return *wal_writer_;
    }

    std::string index_path_;
    std::string durability_mode_;
    int configured_threads_ = 0;
    impl::ScopedUpdateThreadLimit thread_limit_;
    QueryIndexHandle* publish_handle_ = nullptr;
    std::function<void()> publish_callback_;
    impl::UpdateWriteLock writer_lock_;
    BaseSegment base_;
    uint32_t loaded_base_n_ = 0;
    RandomRotation rotation_;
    bool rotate_subspace_ = false;
    std::unique_ptr<RandomRotation> subspace_rotation_;
    UpdateManifest manifest_;
    bool has_manifest_ = false;
    bool fsync_each_commit_ = false;
    bool final_buffered_fsync_ = false;
    std::string updates_dir_;
    std::string direct_map_path_;
    std::string wal_path_;
    std::unique_ptr<UpdateWalWriter> wal_writer_;
    DirectMap direct_map_ = DirectMap::dense();
    uint32_t current_max_pid_ = 0;
    uint32_t current_live_count_ = 0;
    uint64_t next_lsn_ = 0;
    bool published_any_records_ = false;
    bool direct_map_dirty_ = false;
    bool requires_reopen_after_compaction_ = false;
    UpdateWalHeader header_;
    std::vector<UpdateWalRecord> records_scratch_;
    impl::EncodedL0InsertBatch insert_batch_scratch_;
    impl::EncodedL0InsertBatch upsert_batch_scratch_;
    impl::L0EncodeScratch insert_encode_scratch_;
    impl::L0EncodeScratch upsert_encode_scratch_;
    bool collect_commit_timing_ = false;
    UpdateCommitTiming last_commit_timing_;
    CommittedUpdateDelta last_commit_delta_;
};

inline void commit_update_batch_chunked_with_writer(const std::string& index_path,
                                                    const UpdateIndexBatch& batch) {
    std::vector<uint32_t> insert_orig_ids;
    int insert_n = 0, insert_dim = 0;
    std::unique_ptr<impl::SequentialUpdateFbinReader> insert_reader;
    if (!batch.insert_fbin.empty() || !batch.insert_ids.empty()) {
        if (batch.insert_fbin.empty() || batch.insert_ids.empty()) {
            throw std::runtime_error("insert requires both -insert_fbin and -insert_ids");
        }
        insert_orig_ids = impl::load_update_ids(batch.insert_ids);
        impl::UpdateFbinMetadata meta =
            impl::read_update_fbin_metadata(batch.insert_fbin);
        insert_n = meta.n;
        insert_dim = meta.dim;
        if ((int)insert_orig_ids.size() != insert_n) {
            throw std::runtime_error("insert vector count does not match insert ID count");
        }
        insert_reader = std::make_unique<impl::SequentialUpdateFbinReader>(
            batch.insert_fbin, insert_n, insert_dim);
    }

    std::vector<uint32_t> upsert_orig_ids;
    int upsert_n = 0, upsert_dim = 0;
    std::unique_ptr<impl::SequentialUpdateFbinReader> upsert_reader;
    if (!batch.upsert_fbin.empty() || !batch.upsert_ids.empty()) {
        if (batch.upsert_fbin.empty() || batch.upsert_ids.empty()) {
            throw std::runtime_error("upsert requires both -upsert_fbin and -upsert_ids");
        }
        upsert_orig_ids = impl::load_update_ids(batch.upsert_ids);
        impl::UpdateFbinMetadata meta =
            impl::read_update_fbin_metadata(batch.upsert_fbin);
        upsert_n = meta.n;
        upsert_dim = meta.dim;
        if ((int)upsert_orig_ids.size() != upsert_n) {
            throw std::runtime_error("upsert vector count does not match upsert ID count");
        }
        upsert_reader = std::make_unique<impl::SequentialUpdateFbinReader>(
            batch.upsert_fbin, upsert_n, upsert_dim);
    }

    std::vector<uint32_t> delete_orig_ids;
    if (!batch.delete_ids.empty()) {
        delete_orig_ids = impl::load_update_ids(batch.delete_ids);
    }
    if (insert_orig_ids.empty() && upsert_orig_ids.empty() && delete_orig_ids.empty()) {
        throw std::runtime_error("no update records requested");
    }

    impl::validate_no_insert_update_id_conflicts(insert_orig_ids, upsert_orig_ids);

    std::vector<uint32_t> incoming_ids = insert_orig_ids;
    incoming_ids.insert(incoming_ids.end(), upsert_orig_ids.begin(), upsert_orig_ids.end());
    UpdateIndexWriter writer(index_path, batch.durability_mode, batch.threads, incoming_ids);

    size_t delete_offset = 0;
    size_t upsert_offset = 0;
    size_t insert_offset = 0;
    const size_t rows_per_commit = (size_t)batch.batch_rows;
    UpdateIndexMemoryBatch memory_batch;
    std::vector<float> insert_rows_chunk;
    std::vector<float> upsert_rows_chunk;
    while (delete_offset < delete_orig_ids.size() ||
           upsert_offset < upsert_orig_ids.size() ||
           insert_offset < insert_orig_ids.size()) {
        size_t budget = rows_per_commit;
        size_t delete_count = std::min(budget, delete_orig_ids.size() - delete_offset);
        budget -= delete_count;
        size_t upsert_count = std::min(budget, upsert_orig_ids.size() - upsert_offset);
        budget -= upsert_count;
        size_t insert_count = std::min(budget, insert_orig_ids.size() - insert_offset);
        if (delete_count + upsert_count + insert_count == 0) {
            throw std::runtime_error("internal error while chunking update batch");
        }

        memory_batch.delete_orig_ids.clear();
        memory_batch.upsert_orig_ids.clear();
        memory_batch.insert_orig_ids.clear();
        memory_batch.insert_rows = nullptr;
        memory_batch.insert_n = 0;
        memory_batch.insert_dim = 0;
        memory_batch.upsert_rows = nullptr;
        memory_batch.upsert_n = 0;
        memory_batch.upsert_dim = 0;
        if (delete_count != 0) {
            memory_batch.delete_orig_ids.assign(
                delete_orig_ids.begin() + (std::ptrdiff_t)delete_offset,
                delete_orig_ids.begin() + (std::ptrdiff_t)(delete_offset + delete_count));
        }
        if (upsert_count != 0) {
            memory_batch.upsert_orig_ids.assign(
                upsert_orig_ids.begin() + (std::ptrdiff_t)upsert_offset,
                upsert_orig_ids.begin() + (std::ptrdiff_t)(upsert_offset + upsert_count));
            upsert_reader->read_rows(upsert_offset, upsert_count, upsert_rows_chunk);
            memory_batch.upsert_rows = upsert_rows_chunk.data();
            memory_batch.upsert_n = (int)upsert_count;
            memory_batch.upsert_dim = upsert_dim;
        }
        if (insert_count != 0) {
            memory_batch.insert_orig_ids.assign(
                insert_orig_ids.begin() + (std::ptrdiff_t)insert_offset,
                insert_orig_ids.begin() + (std::ptrdiff_t)(insert_offset + insert_count));
            insert_reader->read_rows(insert_offset, insert_count, insert_rows_chunk);
            memory_batch.insert_rows = insert_rows_chunk.data();
            memory_batch.insert_n = (int)insert_count;
            memory_batch.insert_dim = insert_dim;
        }

        writer.commit_batch(memory_batch);
        delete_offset += delete_count;
        upsert_offset += upsert_count;
        insert_offset += insert_count;
    }
    writer.flush_buffered_commits();
}

inline void commit_update_batch(const std::string& index_path, const UpdateIndexBatch& batch) {
    if (!is_valid_update_durability_mode(batch.durability_mode)) {
        throw std::runtime_error("invalid update durability mode: " + batch.durability_mode);
    }
    if (batch.threads < 0) {
        throw std::runtime_error("update thread count must be non-negative");
    }
    if (batch.batch_rows != 0) {
        commit_update_batch_chunked_with_writer(index_path, batch);
        return;
    }

    impl::ScopedUpdateThreadLimit thread_limit(batch.threads);
    impl::UpdateWriteLock writer_lock(index_path);
    {
        impl::UpdatePublishWriteLock publish_lock(index_path);
    }

    BaseSegment base = impl::load_update_base_segment_lightweight(index_path);
    if (base.n < 0 ||
        (uint64_t)base.n > (uint64_t)std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("update base segment size exceeds v1 uint32 PID range");
    }
    uint32_t loaded_base_n = (uint32_t)base.n;

    std::vector<uint32_t> insert_orig_ids;
    std::vector<float> insert_rows;
    int insert_n = 0, insert_dim = 0;
    bool stream_insert_rows = false;
    if (!batch.insert_fbin.empty() || !batch.insert_ids.empty()) {
        if (batch.insert_fbin.empty() || batch.insert_ids.empty()) {
            throw std::runtime_error("insert requires both -insert_fbin and -insert_ids");
        }
        insert_orig_ids = impl::load_update_ids(batch.insert_ids);
        impl::UpdateFbinMetadata meta =
            impl::read_update_fbin_metadata(batch.insert_fbin);
        insert_n = meta.n;
        insert_dim = meta.dim;
        stream_insert_rows = true;
        if ((int)insert_orig_ids.size() != insert_n) {
            throw std::runtime_error("insert vector count does not match insert ID count");
        }
    }

    std::vector<uint32_t> upsert_orig_ids;
    std::vector<float> upsert_rows;
    int upsert_n = 0, upsert_dim = 0;
    bool stream_upsert_rows = false;
    if (!batch.upsert_fbin.empty() || !batch.upsert_ids.empty()) {
        if (batch.upsert_fbin.empty() || batch.upsert_ids.empty()) {
            throw std::runtime_error("upsert requires both -upsert_fbin and -upsert_ids");
        }
        upsert_orig_ids = impl::load_update_ids(batch.upsert_ids);
        impl::UpdateFbinMetadata meta =
            impl::read_update_fbin_metadata(batch.upsert_fbin);
        upsert_n = meta.n;
        upsert_dim = meta.dim;
        stream_upsert_rows = true;
        if ((int)upsert_orig_ids.size() != upsert_n) {
            throw std::runtime_error("upsert vector count does not match upsert ID count");
        }
    }

    std::vector<uint32_t> delete_orig_ids;
    if (!batch.delete_ids.empty()) {
        delete_orig_ids = impl::load_update_ids(batch.delete_ids);
    }
    if (insert_orig_ids.empty() && upsert_orig_ids.empty() && delete_orig_ids.empty()) {
        throw std::runtime_error("no update records requested");
    }
    impl::validate_no_insert_update_id_conflicts(insert_orig_ids, upsert_orig_ids);

    UpdateManifest manifest;
    bool has_manifest = update_manifest_exists(index_path);
    std::shared_ptr<const LiveTable> live =
        LiveTable::all_live(loaded_base_n);
    if (has_manifest) {
        manifest = load_update_manifest_json_for_index(index_path);
        if (manifest.base_path != "." || manifest.base_n != loaded_base_n) {
            throw std::runtime_error("loaded lightweight update base does not match update manifest");
        }
        live = impl::load_update_live_table_from_manifest(base, manifest, index_path);
        manifest.durability_mode = batch.durability_mode;
    } else {
        std::vector<uint32_t> incoming_ids = insert_orig_ids;
        incoming_ids.insert(incoming_ids.end(), upsert_orig_ids.begin(), upsert_orig_ids.end());
        manifest.format = "score_update_manifest_v1";
        manifest.generation = 1;
        manifest.base_path = ".";
        manifest.base_n = loaded_base_n;
        manifest.max_pid = loaded_base_n;
        manifest.live_count = loaded_base_n;
        manifest.wal_path = "updates/l0.wal";
        manifest.checkpoint_path = "updates/l0.checkpoint";
        manifest.direct_map_mode = impl::choose_direct_map_mode(base, incoming_ids);
        manifest.direct_map_path = "updates/direct_map.bin";
        manifest.direct_map_optional_for_search = true;
        manifest.durability_mode = batch.durability_mode;
        manifest.last_committed_lsn = 0;
        manifest.last_fsynced_lsn = 0;
    }

    const bool fsync_each_commit = update_durability_fsync_each_commit(batch.durability_mode);
    const bool final_buffered_fsync =
        update_durability_flush_at_buffer_boundary(batch.durability_mode);
    const std::string updates_dir = index_path + "/updates";
    impl::ensure_dir(updates_dir);
    impl::fsync_parent_dir_if_requested(
        updates_dir, update_durability_fsync_artifacts(batch.durability_mode));
    std::string direct_map_path = impl::join_index_path(index_path, manifest.direct_map_path);
    DirectMap direct_map = manifest.direct_map_mode == "sparse"
        ? DirectMap::sparse()
        : DirectMap::dense();
    if (impl::path_exists(direct_map_path)) {
        try {
            direct_map = DirectMap::load_binary(direct_map_path);
        } catch (...) {
            direct_map = impl::build_direct_map_from_base_and_wal(base, manifest, index_path);
        }
        if (!impl::direct_map_matches_live_table(direct_map, manifest, *live)) {
            direct_map = impl::build_direct_map_from_base_and_wal(base, manifest, index_path);
        }
    } else {
        direct_map = impl::build_direct_map_from_base_and_wal(base, manifest, index_path);
    }
    if (!impl::direct_map_matches_live_table(direct_map, manifest, *live)) {
        throw std::runtime_error("rebuilt DirectMap does not match current live reader view");
    }
    direct_map.reserve_entries(direct_map.live_size() +
                               insert_orig_ids.size() +
                               upsert_orig_ids.size());

    uint32_t current_max_pid = manifest.max_pid;
    uint32_t current_live_count = manifest.live_count;
    uint64_t next_lsn = manifest.last_committed_lsn + 1;
    bool published_any_records = false;

    auto decrement_sealed_delta_live_count = [&](uint32_t old_pid) {
        if (old_pid < manifest.base_n) return;
        for (UpdateManifestDelta& delta : manifest.sealed_deltas) {
            uint64_t begin = delta.pid_begin;
            uint64_t end = begin + (uint64_t)delta.pid_count;
            if ((uint64_t)old_pid < begin || (uint64_t)old_pid >= end) continue;
            if (delta.live_count == 0) {
                throw std::runtime_error("sealed delta live_count underflow while deleting PID " +
                                         std::to_string(old_pid));
            }
            --delta.live_count;
            return;
        }
        if (old_pid < current_max_pid) return;
        throw std::runtime_error("delta PID is not covered by any sealed delta manifest entry");
    };

    auto append_delete = [&](uint32_t orig_id, std::vector<UpdateWalRecord>& records) {
        uint32_t old_pid = 0;
        if (!direct_map.erase(orig_id, &old_pid)) return;
        UpdateWalRecord record;
        record.type = UpdateWalRecord::Type::Delete;
        record.lsn = next_lsn++;
        record.tombstone.orig_id = orig_id;
        record.tombstone.old_global_pid = old_pid;
        records.push_back(record);
        decrement_sealed_delta_live_count(old_pid);
        --current_live_count;
    };

    auto append_tombstone_record =
        [&](uint32_t orig_id, uint32_t old_pid, std::vector<UpdateWalRecord>& records) {
        UpdateWalRecord record;
        record.type = UpdateWalRecord::Type::Delete;
        record.lsn = next_lsn++;
        record.tombstone.orig_id = orig_id;
        record.tombstone.old_global_pid = old_pid;
        records.push_back(record);
        decrement_sealed_delta_live_count(old_pid);
        --current_live_count;
    };

    UpdateWalHeader header;
    header.generation = manifest.generation;
    header.base_n = manifest.base_n;
    header.num_subspaces = (uint32_t)base.ivf_indices.size();
    header.verify_dim = (uint32_t)base.verify_vectors_dim;
    header.rabitq_record_bytes = (uint32_t)base.rabitq_bin_with_assign_record_bytes_;

    std::string wal_path = impl::join_index_path(index_path, manifest.wal_path);
    if (has_manifest && !impl::path_exists(wal_path)) {
        throw std::runtime_error("update manifest references missing WAL: " + wal_path);
    }

    auto id_slice = [](const std::vector<uint32_t>& ids,
                       size_t offset,
                       size_t count) {
        return std::vector<uint32_t>(
            ids.begin() + (std::ptrdiff_t)offset,
            ids.begin() + (std::ptrdiff_t)(offset + count));
    };

    auto append_inserts = [&](std::vector<UpdateWalInsert>& inserts,
                              std::vector<UpdateWalRecord>& records) {
        for (UpdateWalInsert& insert : inserts) {
            UpdateWalRecord record;
            record.type = UpdateWalRecord::Type::Insert;
            record.lsn = next_lsn++;
            record.insert = std::move(insert);
            direct_map.set(record.insert.orig_id, record.insert.global_pid);
            ++current_live_count;
            records.push_back(std::move(record));
        }
    };

    auto append_upserts = [&](std::vector<UpdateWalInsert>& inserts,
                              std::vector<UpdateWalRecord>& records) {
        for (UpdateWalInsert& insert : inserts) {
            uint32_t old_pid = 0;
            bool had_old =
                direct_map.set_with_old_pid(insert.orig_id, insert.global_pid, &old_pid);
            if (had_old) append_tombstone_record(insert.orig_id, old_pid, records);

            UpdateWalRecord record;
            record.type = UpdateWalRecord::Type::Insert;
            record.lsn = next_lsn++;
            record.insert = std::move(insert);
            ++current_live_count;
            records.push_back(std::move(record));
        }
    };

    auto publish_records = [&](std::vector<UpdateWalRecord>& records) {
        if (records.empty()) return false;
        impl::UpdatePublishWriteLock publish_lock(index_path);
        uint64_t commit_lsn = next_lsn++;
        UpdateWalWriter writer(wal_path, header, has_manifest);
        for (const UpdateWalRecord& record : records) {
            if (record.type == UpdateWalRecord::Type::Delete) {
                writer.append_delete(record.lsn, record.tombstone);
            } else {
                writer.append_insert(record.lsn, record.insert);
            }
        }
        UpdateWalCommit commit;
        commit.lsn = commit_lsn;
        commit.batch_record_count = (uint32_t)records.size();
        commit.max_pid = current_max_pid;
        commit.live_count = current_live_count;
        writer.append_commit(commit);
        writer.flush();

        impl::fsync_file_if_requested(wal_path, fsync_each_commit);
        impl::fsync_parent_dir_if_requested(wal_path, fsync_each_commit);

        direct_map.save_binary(direct_map_path, fsync_each_commit);

        manifest.max_pid = current_max_pid;
        manifest.live_count = current_live_count;
        manifest.last_committed_lsn = commit_lsn;
        if (fsync_each_commit) manifest.last_fsynced_lsn = commit_lsn;
        write_update_manifest_json_for_index(manifest, index_path, fsync_each_commit);
        has_manifest = true;
        published_any_records = true;
        return true;
    };

    auto flush_buffered_commits = [&]() {
        if (!final_buffered_fsync || !published_any_records) return;
        impl::UpdatePublishWriteLock publish_lock(index_path);
        impl::fsync_file_if_requested(wal_path, true);
        impl::fsync_parent_dir_if_requested(wal_path, true);
        direct_map.save_binary(direct_map_path, true);
        manifest.last_fsynced_lsn = manifest.last_committed_lsn;
        write_update_manifest_json_for_index(manifest, index_path, true);
    };

    auto commit_chunk = [&](size_t delete_offset,
                            size_t delete_count,
                            size_t upsert_offset,
                            size_t upsert_count,
                            size_t insert_offset,
                            size_t insert_count) {
        std::vector<uint32_t> upsert_ids_slice;
        std::vector<float> upsert_rows_chunk;
        std::vector<uint32_t> coalesced_upsert_ids;
        std::vector<float> coalesced_upsert_rows;
        const std::vector<uint32_t>* effective_upsert_ids = &upsert_ids_slice;
        const float* effective_upsert_rows = nullptr;
        int effective_upsert_n = (int)upsert_count;
        if (upsert_count != 0) {
            upsert_ids_slice = id_slice(upsert_orig_ids, upsert_offset, upsert_count);
            if (stream_upsert_rows) {
                upsert_rows_chunk = impl::load_update_fbin_rows(
                    batch.upsert_fbin, upsert_n, upsert_dim, upsert_offset, upsert_count);
                effective_upsert_rows = upsert_rows_chunk.data();
            } else {
                effective_upsert_rows =
                    upsert_rows.data() + upsert_offset * (size_t)upsert_dim;
            }
            if (impl::coalesce_memory_rows_last_wins(
                    effective_upsert_rows,
                    (int)upsert_count,
                    upsert_dim,
                    upsert_ids_slice,
                    coalesced_upsert_ids,
                    coalesced_upsert_rows)) {
                effective_upsert_ids = &coalesced_upsert_ids;
                effective_upsert_rows = coalesced_upsert_rows.data();
                effective_upsert_n = (int)coalesced_upsert_ids.size();
            }
        }

        std::vector<UpdateWalRecord> records;
        records.reserve(delete_count + effective_upsert_ids->size() * 2 + insert_count);

        for (size_t i = 0; i < delete_count; ++i) {
            append_delete(delete_orig_ids[delete_offset + i], records);
        }

        if (insert_count != 0) {
            for (size_t i = 0; i < insert_count; ++i) {
                uint32_t orig_id = insert_orig_ids[insert_offset + i];
                if (direct_map.contains(orig_id)) {
                    throw std::runtime_error("insert ID already exists; use upsert for orig_id " +
                                             std::to_string(orig_id));
                }
            }
            std::vector<uint32_t> insert_ids_slice =
                id_slice(insert_orig_ids, insert_offset, insert_count);
            std::vector<float> insert_rows_chunk;
            const float* insert_rows_ptr = nullptr;
            if (stream_insert_rows) {
                insert_rows_chunk = impl::load_update_fbin_rows(
                    batch.insert_fbin, insert_n, insert_dim, insert_offset, insert_count);
                insert_rows_ptr = insert_rows_chunk.data();
            } else {
                insert_rows_ptr = insert_rows.data() + insert_offset * (size_t)insert_dim;
            }
            std::vector<UpdateWalInsert> encoded = impl::encode_l0_inserts(
                index_path,
                base,
                insert_rows_ptr,
                (int)insert_count,
                insert_dim,
                insert_ids_slice,
                current_max_pid);
            current_max_pid += (uint32_t)encoded.size();
            append_inserts(encoded, records);
        }

        if (!effective_upsert_ids->empty()) {
            std::vector<UpdateWalInsert> encoded = impl::encode_l0_inserts(
                index_path,
                base,
                effective_upsert_rows,
                effective_upsert_n,
                upsert_dim,
                *effective_upsert_ids,
                current_max_pid);
            current_max_pid += (uint32_t)encoded.size();
            append_upserts(encoded, records);
        }

        return publish_records(records);
    };

    if (batch.batch_rows == 0) {
        constexpr size_t kDefaultStreamRawBytes = 64ull << 20;
        auto stream_rows_for_dim = [](int dim, size_t remaining) {
            if (remaining == 0) return (size_t)0;
            if (dim <= 0) return remaining;
            size_t floats = kDefaultStreamRawBytes / sizeof(float);
            size_t rows = std::max<size_t>(1, floats / (size_t)dim);
            return std::min(rows, remaining);
        };

        RandomRotation stream_rotation;
        stream_rotation.load(index_path + "/rotation.bin");
        if (stream_rotation.dim() != base.dim ||
            stream_rotation.padded_dim() != base.verify_vectors_dim ||
            stream_rotation.padded_dim() != base.rabitq_padded_dim) {
            throw std::runtime_error("rotation dimensions do not match loaded update base");
        }
        bool stream_rotate_subspace =
            impl::path_exists(index_path + "/rotate_subspace.flag");
        std::unique_ptr<RandomRotation> stream_subspace_rotation;
        if (stream_rotate_subspace &&
            impl::path_exists(index_path + "/subspace_rotation.bin")) {
            stream_subspace_rotation = std::make_unique<RandomRotation>();
            stream_subspace_rotation->load(index_path + "/subspace_rotation.bin");
            if (stream_subspace_rotation->dim() != base.dim) {
                throw std::runtime_error("subspace_rotation.bin dimension does not match update base");
            }
        }

        std::unique_ptr<UpdateWalWriter> writer;
        uint32_t batch_record_count = 0;
        auto ensure_writer = [&]() -> UpdateWalWriter& {
            if (!writer) {
                writer = std::make_unique<UpdateWalWriter>(wal_path, header, has_manifest);
            }
            return *writer;
        };
        auto append_delete_stream = [&](uint32_t orig_id) {
            uint32_t old_pid = 0;
            if (!direct_map.erase(orig_id, &old_pid)) return;
            UpdateWalDelete tombstone;
            tombstone.orig_id = orig_id;
            tombstone.old_global_pid = old_pid;
            ensure_writer().append_delete(next_lsn++, tombstone);
            decrement_sealed_delta_live_count(old_pid);
            --current_live_count;
            ++batch_record_count;
        };
        auto append_tombstone_stream = [&](uint32_t orig_id, uint32_t old_pid) {
            UpdateWalDelete tombstone;
            tombstone.orig_id = orig_id;
            tombstone.old_global_pid = old_pid;
            ensure_writer().append_delete(next_lsn++, tombstone);
            decrement_sealed_delta_live_count(old_pid);
            --current_live_count;
            ++batch_record_count;
        };
        auto append_encoded_inserts_stream = [&](const impl::EncodedL0InsertBatch& encoded) {
            if (encoded.size() == 0) return;
            ensure_writer().append_insert_rows_batch(
                next_lsn,
                encoded.size(),
                encoded.orig_ids.data(),
                encoded.pid_begin,
                encoded.verify_rotated.data(),
                encoded.rabitq_records.data(),
                encoded.subspace_cells.data());
            next_lsn += encoded.size();
            for (size_t row = 0; row < encoded.size(); ++row) {
                uint32_t orig_id = encoded.orig_ids[row];
                uint32_t global_pid = encoded.global_pid(row);
                direct_map.set(orig_id, global_pid);
                ++current_live_count;
                ++batch_record_count;
            }
        };
        auto append_encoded_upserts_stream = [&](const impl::EncodedL0InsertBatch& encoded) {
            if (encoded.size() == 0) return;
            for (size_t row = 0; row < encoded.size(); ++row) {
                uint32_t orig_id = encoded.orig_ids[row];
                uint32_t global_pid = encoded.global_pid(row);
                uint32_t old_pid = 0;
                bool had_old = direct_map.set_with_old_pid(orig_id, global_pid, &old_pid);
                if (had_old) append_tombstone_stream(orig_id, old_pid);
                ++current_live_count;
                ++batch_record_count;
            }
            ensure_writer().append_insert_rows_batch(
                next_lsn,
                encoded.size(),
                encoded.orig_ids.data(),
                encoded.pid_begin,
                encoded.verify_rotated.data(),
                encoded.rabitq_records.data(),
                encoded.subspace_cells.data());
            next_lsn += encoded.size();
        };

        std::vector<uint8_t> upsert_keep_mask;
        std::vector<uint32_t> coalesced_stream_upsert_ids;
        bool coalesced_stream_upserts = impl::build_last_wins_keep_mask(
            upsert_orig_ids, upsert_keep_mask, coalesced_stream_upsert_ids);

        for (uint32_t orig_id : delete_orig_ids) append_delete_stream(orig_id);

        if (!insert_orig_ids.empty()) {
            for (uint32_t orig_id : insert_orig_ids) {
                if (direct_map.contains(orig_id)) {
                    throw std::runtime_error("insert ID already exists; use upsert for orig_id " +
                                             std::to_string(orig_id));
                }
            }
            size_t offset = 0;
            while (offset < insert_orig_ids.size()) {
                size_t count = stream_rows_for_dim(insert_dim, insert_orig_ids.size() - offset);
                std::vector<uint32_t> ids_slice =
                    id_slice(insert_orig_ids, offset, count);
                std::vector<float> rows_chunk = impl::load_update_fbin_rows(
                    batch.insert_fbin, insert_n, insert_dim, offset, count);
                impl::EncodedL0InsertBatch encoded =
                    impl::encode_l0_inserts_contiguous(
                        base,
                        stream_rotation,
                        stream_rotate_subspace,
                        stream_subspace_rotation.get(),
                        rows_chunk.data(),
                        (int)count,
                        insert_dim,
                        ids_slice,
                        current_max_pid);
                current_max_pid += (uint32_t)encoded.size();
                append_encoded_inserts_stream(encoded);
                offset += count;
            }
        }

        if (!upsert_orig_ids.empty()) {
            size_t offset = 0;
            while (offset < upsert_orig_ids.size()) {
                size_t count = stream_rows_for_dim(upsert_dim, upsert_orig_ids.size() - offset);
                std::vector<float> rows_chunk = impl::load_update_fbin_rows(
                    batch.upsert_fbin, upsert_n, upsert_dim, offset, count);
                std::vector<uint32_t> ids_slice;
                const float* rows_ptr = rows_chunk.data();
                int rows_n = (int)count;
                if (coalesced_stream_upserts) {
                    std::vector<float> kept_rows;
                    kept_rows.reserve(count * (size_t)upsert_dim);
                    ids_slice.reserve(count);
                    for (size_t i = 0; i < count; ++i) {
                        if (upsert_keep_mask[offset + i] == 0) continue;
                        ids_slice.push_back(upsert_orig_ids[offset + i]);
                        const float* src = rows_chunk.data() + i * (size_t)upsert_dim;
                        kept_rows.insert(kept_rows.end(), src, src + upsert_dim);
                    }
                    if (ids_slice.empty()) {
                        offset += count;
                        continue;
                    }
                    rows_chunk.swap(kept_rows);
                    rows_ptr = rows_chunk.data();
                    rows_n = (int)ids_slice.size();
                } else {
                    ids_slice = id_slice(upsert_orig_ids, offset, count);
                }
                impl::EncodedL0InsertBatch encoded =
                    impl::encode_l0_inserts_contiguous(
                        base,
                        stream_rotation,
                        stream_rotate_subspace,
                        stream_subspace_rotation.get(),
                        rows_ptr,
                        rows_n,
                        upsert_dim,
                        ids_slice,
                        current_max_pid);
                current_max_pid += (uint32_t)encoded.size();
                append_encoded_upserts_stream(encoded);
                offset += count;
            }
        }

        if (batch_record_count == 0) {
            flush_buffered_commits();
            return;
        }

        {
            impl::UpdatePublishWriteLock publish_lock(index_path);
            uint64_t commit_lsn = next_lsn++;
            UpdateWalCommit commit;
            commit.lsn = commit_lsn;
            commit.batch_record_count = batch_record_count;
            commit.max_pid = current_max_pid;
            commit.live_count = current_live_count;
            ensure_writer().append_commit(commit);
            writer->flush();

            impl::fsync_file_if_requested(wal_path, fsync_each_commit);
            impl::fsync_parent_dir_if_requested(wal_path, fsync_each_commit);

            direct_map.save_binary(direct_map_path, fsync_each_commit);

            manifest.max_pid = current_max_pid;
            manifest.live_count = current_live_count;
            manifest.last_committed_lsn = commit_lsn;
            if (fsync_each_commit) manifest.last_fsynced_lsn = commit_lsn;
            write_update_manifest_json_for_index(manifest, index_path, fsync_each_commit);
            has_manifest = true;
            published_any_records = true;
        }
        flush_buffered_commits();
        return;
    }

    size_t delete_offset = 0;
    size_t upsert_offset = 0;
    size_t insert_offset = 0;
    const size_t rows_per_commit = (size_t)batch.batch_rows;
    while (delete_offset < delete_orig_ids.size() ||
           upsert_offset < upsert_orig_ids.size() ||
           insert_offset < insert_orig_ids.size()) {
        size_t budget = rows_per_commit;
        size_t delete_count = std::min(budget, delete_orig_ids.size() - delete_offset);
        budget -= delete_count;
        size_t upsert_count = std::min(budget, upsert_orig_ids.size() - upsert_offset);
        budget -= upsert_count;
        size_t insert_count = std::min(budget, insert_orig_ids.size() - insert_offset);
        if (delete_count + upsert_count + insert_count == 0) {
            throw std::runtime_error("internal error while chunking update batch");
        }
        (void)commit_chunk(
            delete_offset, delete_count,
            upsert_offset, upsert_count,
            insert_offset, insert_count);
        delete_offset += delete_count;
        upsert_offset += upsert_count;
        insert_offset += insert_count;
    }
    flush_buffered_commits();
}

} // namespace score
