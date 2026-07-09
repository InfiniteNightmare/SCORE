#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <limits>
#include <stdexcept>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <omp.h>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

// SuperKMeans integration for RaBitQ clustering
#if HAS_SUPERKMEANS
#include "superkmeans/superkmeans.h"
#include "superkmeans/hierarchical_superkmeans.h"
#endif

namespace score {

inline void parse_int_csv(const std::string& csv, std::vector<int>& out) {
    out.clear();
    size_t start = 0;
    while (start < csv.size()) {
        size_t comma = csv.find(',', start);
        std::string token = csv.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!token.empty()) out.push_back(std::stoi(token));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
}

struct ProcRssSnapshot {
    long vmrss_kb = -1;
    long vmhwm_kb = -1;
};

inline bool build_rss_trace_enabled() {
    const char* env = std::getenv("SCORE_BUILD_RSS_TRACE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

inline ProcRssSnapshot read_proc_rss_snapshot() {
    ProcRssSnapshot snapshot;
    std::ifstream in("/proc/self/status");
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::istringstream iss(line.substr(6));
            iss >> snapshot.vmrss_kb;
        } else if (line.rfind("VmHWM:", 0) == 0) {
            std::istringstream iss(line.substr(6));
            iss >> snapshot.vmhwm_kb;
        }
    }
    return snapshot;
}

inline void log_build_rss_checkpoint(const std::string& label) {
    if (!build_rss_trace_enabled()) return;
    ProcRssSnapshot snapshot = read_proc_rss_snapshot();
    std::cout << "[Build][RSS] " << label
              << " VmRSS_KB=" << snapshot.vmrss_kb
              << " VmHWM_KB=" << snapshot.vmhwm_kb << std::endl;
}

// Page size for SSD alignment
static constexpr size_t PAGE_SIZE = 4096;
static constexpr size_t SECTOR_LEN = PAGE_SIZE;
static constexpr size_t MAX_IO_DEPTH = 128;

// Default index parameters
struct IndexConfig {
    // Subspace collision
    int num_subspaces = 16;        // M: number of subspaces
    float collision_ratio = 0.05f; // alpha: fraction of points as collision candidates per subspace
    float candidate_ratio = 0.001f;// beta: fraction of points for re-ranking

    // RaBitQ with per-cluster clustering (SuperKMeans + FhtKacRotator)
    int rabitq_k = 200;            // K clusters for RaBitQ (required, default 200)
    int rabitq_seed = 42;          // Seed for top-level K-means clustering

    // Query
    int topk = 10;

    // Index path (build output / search input)
    std::string index_path;

    // IVF subspace index
    int ivf_k = 2500;              // K centroids per subspace for IVF
    bool rotate_subspace = false;  // Apply FhtKac rotation before subspace partitioning
    bool subspace_force_raw = false; // Force raw vectors for subspace stage only
    int num_threads = 0;           // 0 = use OMP default, omp_get_max_threads()
    bool pid_reorder = true;
    int pid_reorder_cell_key_depth = 2;
    int pid_reorder_key_mode = 0; // 0=cluster-first, 1=cell-first, 2=interleave
    std::vector<int> pid_reorder_subspaces;

    // Phase 4/5 weighted scoring: tier weights w∈{1..max_w}, W=3 default (proven optimal)
    int max_w = 3;                 // Max tier weight (see docs/weighted_sc_theory.md)
    int phase4_score_prefetch = 32; // Internal score-array prefetch distance in posting traversal
    bool query_profile = false;    // Print per-phase query timings when enabled by CLI
};

// Page-aligned memory allocation (4KB) for madvise and pread buffers
template<typename T>
struct AlignedAllocator {
    using value_type = T;
    AlignedAllocator() = default;
    template<typename U> constexpr AlignedAllocator(const AlignedAllocator<U>&) noexcept {}
    T* allocate(size_t n) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, PAGE_SIZE, n * sizeof(T)) != 0)
            throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }
    void deallocate(T* p, size_t) noexcept { free(p); }
};
using aligned_vector_float = std::vector<float, AlignedAllocator<float>>;
using aligned_vector_int = std::vector<int, AlignedAllocator<int>>;

static constexpr size_t HUGE_PAGE_SIZE = 2 * 1024 * 1024;

#if defined(__linux__) && !defined(MADV_COLLAPSE)
#define MADV_COLLAPSE 25
#endif

// Anonymous mmap-backed byte buffer for arrays where THP can reduce TLB pressure.
class HugePageByteBuffer {
public:
    HugePageByteBuffer() = default;
    ~HugePageByteBuffer() { release(); }
    HugePageByteBuffer(const HugePageByteBuffer&) = delete;
    HugePageByteBuffer& operator=(const HugePageByteBuffer&) = delete;
    HugePageByteBuffer(HugePageByteBuffer&& other) noexcept { move_from(other); }
    HugePageByteBuffer& operator=(HugePageByteBuffer&& other) noexcept {
        if (this != &other) {
            release();
            move_from(other);
        }
        return *this;
    }

    void resize(size_t n, uint8_t value = 0) {
        if (n == 0) {
            release();
            return;
        }
        if (data_ != nullptr && n <= mapped_size_) {
            size_ = n;
            std::memset(data_, value, size_);
            return;
        }
        release();
        size_ = n;
        mapped_size_ = round_up(n, HUGE_PAGE_SIZE);
        void* ptr = mmap(nullptr, mapped_size_, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ptr == MAP_FAILED) {
            data_ = nullptr;
            mapped_size_ = 0;
            size_ = 0;
            throw std::bad_alloc();
        }
        data_ = static_cast<uint8_t*>(ptr);
        advise_hugepage();
        std::memset(data_, value, size_);
        advise_hugepage();
    }

    void resize_uninitialized(size_t n) {
        if (n == 0) {
            release();
            return;
        }
        if (data_ != nullptr && n <= mapped_size_) {
            size_ = n;
            return;
        }
        release();
        size_ = n;
        mapped_size_ = round_up(n, HUGE_PAGE_SIZE);
        void* ptr = mmap(nullptr, mapped_size_, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ptr == MAP_FAILED) {
            data_ = nullptr;
            mapped_size_ = 0;
            size_ = 0;
            throw std::bad_alloc();
        }
        data_ = static_cast<uint8_t*>(ptr);
        (void)madvise(data_, mapped_size_, MADV_HUGEPAGE);
    }

    uint8_t* data() { return data_; }
    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    uint8_t& operator[](size_t idx) { return data_[idx]; }
    const uint8_t& operator[](size_t idx) const { return data_[idx]; }
    uint8_t* begin() { return data_; }
    uint8_t* end() { return data_ + size_; }
    const uint8_t* begin() const { return data_; }
    const uint8_t* end() const { return data_ + size_; }

    void advise_hugepage() const {
        if (!data_ || mapped_size_ == 0) return;
        (void)madvise(data_, mapped_size_, MADV_HUGEPAGE);
        (void)madvise(data_, mapped_size_, MADV_COLLAPSE);
    }

private:
    static size_t round_up(size_t n, size_t alignment) {
        return ((n + alignment - 1) / alignment) * alignment;
    }

    void release() {
        if (data_) {
            munmap(data_, mapped_size_);
            data_ = nullptr;
        }
        size_ = 0;
        mapped_size_ = 0;
    }

    void move_from(HugePageByteBuffer& other) noexcept {
        data_ = other.data_;
        size_ = other.size_;
        mapped_size_ = other.mapped_size_;
        other.data_ = nullptr;
        other.size_ = 0;
        other.mapped_size_ = 0;
    }

    uint8_t* data_ = nullptr;
    size_t size_ = 0;
    size_t mapped_size_ = 0;
};
using hugepage_vector_uint8 = HugePageByteBuffer;

inline bool enable_transparent_huge_pages_for_process() {
#if defined(__linux__) && defined(PR_SET_THP_DISABLE)
    return prctl(PR_SET_THP_DISABLE, 0, 0, 0, 0) == 0;
#else
    return false;
#endif
}

// Parallel file read into a pre-allocated buffer.
// N threads read disjoint page-aligned chunks via pread().
inline void parallel_read(int fd, void* buffer, size_t bytes, size_t skip = 0) {
    if (bytes == 0) return;
    int nt = std::min(omp_get_max_threads(),
                      std::max(1, (int)((bytes + (256ull << 20) - 1) / (256ull << 20))));
    size_t chunk = ((bytes + nt - 1) / nt + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
    std::atomic<bool> failed{false};
    #pragma omp parallel for num_threads(nt) schedule(static, 1)
    for (int t = 0; t < nt; t++) {
        size_t off = (size_t)t * chunk;
        size_t end = std::min(off + chunk, bytes);
        if (off >= bytes) continue;
        char* ptr = static_cast<char*>(buffer) + off;
        size_t rem = end - off;
        off_t file_off = skip + off;
        while (rem > 0) {
            ssize_t r = pread(fd, ptr, rem, file_off);
            if (r <= 0) {
                if (errno == EINTR) continue;
                failed.store(true, std::memory_order_relaxed);
                break;
            }
            ptr += r; file_off += r; rem -= r;
        }
        if (rem != 0) failed.store(true, std::memory_order_relaxed);
    }
    if (failed.load(std::memory_order_relaxed)) {
        throw std::runtime_error("parallel_read failed or returned a short read");
    }
}

// Dataset format (compatible with ann-benchmarks: .fvecs, .ivecs, .bvecs)
inline std::vector<float> load_fvecs(const std::string& path, int& n, int& dim) {
    // Bulk read: single read() + in-memory parse.
    // Old approach read per-vector (2M syscalls for 1M vectors).
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("Cannot open: " + path);

    struct stat st;
    fstat(fd, &st);
    size_t fsize = st.st_size;

    // Read dim from first 4 bytes
    ::read(fd, &dim, sizeof(int));
    size_t vec_bytes = sizeof(int) + (size_t)dim * sizeof(float);
    n = fsize / vec_bytes;

    // mmap the entire file for zero-copy access
    void* mapped = mmap(NULL, fsize, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    if (mapped == MAP_FAILED) {
        // Fallback: bulk read into buffer
        ::lseek(fd, 0, SEEK_SET);
        std::vector<char> raw(fsize);
        size_t total_read = 0;
        while (total_read < fsize) {
            ssize_t r = ::read(fd, raw.data() + total_read, fsize - total_read);
            if (r <= 0) break;
            total_read += r;
        }
        ::close(fd);

        std::vector<float> data((size_t)n * dim);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            const char* src = raw.data() + (size_t)i * vec_bytes + sizeof(int);
            memcpy(&data[(size_t)i * dim], src, dim * sizeof(float));
        }
        return data;
    }

    // mmap succeeded: parallel strip headers
    madvise(mapped, fsize, MADV_SEQUENTIAL);
    const char* base = static_cast<const char*>(mapped);
    std::vector<float> data((size_t)n * dim);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        const char* src = base + (size_t)i * vec_bytes + sizeof(int);
        memcpy(&data[(size_t)i * dim], src, dim * sizeof(float));
    }
    munmap(mapped, fsize);
    ::close(fd);
    return data;
}

// Detect vector file format: returns {format_tag, n, dim}
// format_tag: 0=.fbin, 1=.fvecs, -1=invalid
// .fbin: [n:int32][dim:int32][reserved=0:int32][data:n*dim*float]
// .fvecs: [dim:int32][data:dim*float]... per-vector header, no n
inline std::tuple<int, int, int> detect_vector_format(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return {-1, 0, 0};
    int first4 = 0;
    f.read(reinterpret_cast<char*>(&first4), sizeof(int));
    f.seekg(0, std::ios::end);
    size_t file_size = f.tellg();

    auto has_suffix = [&](const std::string& suffix) {
        return path.size() >= suffix.size() &&
               path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0;
    };

    auto try_fvecs = [&]() -> std::tuple<int, int, int> {
        int dim = first4;
        if (dim < 4 || dim > 16384) return {-1, 0, 0};
        size_t stride = sizeof(int) + (size_t)dim * sizeof(float);
        if (file_size % stride == 0) {
            size_t n_size = file_size / stride;
            if (n_size > (size_t)std::numeric_limits<int>::max()) {
                return {-1, 0, 0};
            }
            int n = (int)n_size;
            return {1, n, dim};  // .fvecs
        }
        return {-1, 0, 0};
    };

    auto try_fbin = [&]() -> std::tuple<int, int, int> {
        // Try .fbin: [n][dim][reserved=0][data]
        f.seekg(0);
        int n_r = 0, dim_r = 0, reserved = 0;
        f.read(reinterpret_cast<char*>(&n_r), sizeof(int));
        f.read(reinterpret_cast<char*>(&dim_r), sizeof(int));
        f.read(reinterpret_cast<char*>(&reserved), sizeof(int));
        if (n_r <= 0 || dim_r < 4 || dim_r > 16384 || reserved != 0) {
            return {-1, 0, 0};
        }
        if ((size_t)n_r > std::numeric_limits<size_t>::max() / (size_t)dim_r) {
            return {-1, 0, 0};
        }
        size_t elem_count = (size_t)n_r * (size_t)dim_r;
        if (elem_count > (std::numeric_limits<size_t>::max() - 12) / sizeof(float)) {
            return {-1, 0, 0};
        }
        size_t expected = 12 + elem_count * sizeof(float);
        if (file_size == expected) {
            return {0, n_r, dim_r};  // .fbin
        }
        return {-1, 0, 0};
    };

    // `.fbin` and tiny `.fvecs` can be byte-size ambiguous. Respect the
    // extension first so files like [n=4, dim=8, reserved=0] are not mistaken
    // for `.fvecs` with dim=4.
    if (has_suffix(".fbin")) {
        auto fbin = try_fbin();
        if (std::get<0>(fbin) >= 0) return fbin;
        return try_fvecs();
    }
    if (has_suffix(".fvecs")) {
        auto fvecs = try_fvecs();
        if (std::get<0>(fvecs) >= 0) return fvecs;
        return try_fbin();
    }

    auto fbin = try_fbin();
    if (std::get<0>(fbin) >= 0) return fbin;
    auto fvecs = try_fvecs();
    if (std::get<0>(fvecs) >= 0) return fvecs;

    return {-1, 0, 0};
}

inline std::vector<int> load_ivecs(const std::string& path, int& n, int& dim) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("Cannot open: " + path);
    in.read(reinterpret_cast<char*>(&n), sizeof(int));   // read first int as n
    in.read(reinterpret_cast<char*>(&dim), sizeof(int));  // read second int as dim
    size_t flat_bytes = (size_t)n * dim * sizeof(int);
    size_t flat_total = 2 * sizeof(int) + flat_bytes;
    in.seekg(0, std::ios::end);
    size_t fsize = in.tellg();
    in.seekg(0);
    if (flat_total == fsize) {
        // Flat format: [n][dim] + n*dim*4 bytes (no per-entry headers)
        std::vector<int> data((size_t)n * dim);
        in.seekg(2 * sizeof(int));  // skip [n][dim] header
        in.read(reinterpret_cast<char*>(data.data()), flat_bytes);
        return data;
    }
    // Standard ivecs: each entry is [dim][dim ints], no global header
    // Re-read dim from first entry (n/dim from flat check are stale)
    in.seekg(0);
    int d0; in.read(reinterpret_cast<char*>(&d0), sizeof(int));
    dim = d0;
    size_t rec_size = sizeof(int) + (size_t)dim * sizeof(int);
    n = fsize / rec_size;
    std::vector<int> data((size_t)n * dim);
    in.seekg(0);
    for (int i = 0; i < n; i++) {
        int d; in.read(reinterpret_cast<char*>(&d), sizeof(int));
        assert(d == dim);
        in.read(reinterpret_cast<char*>(&data[(size_t)i * dim]), dim * sizeof(int));
    }
    return data;
}

// Timer
struct Timer {
    struct timespec start;
    void reset() { clock_gettime(CLOCK_MONOTONIC, &start); }
    double elapsed_ms() const {
        struct timespec end;
        clock_gettime(CLOCK_MONOTONIC, &end);
        return (end.tv_sec - start.tv_sec) * 1000.0 +
               (end.tv_nsec - start.tv_nsec) / 1e6;
    }
    double elapsed_us() const { return elapsed_ms() * 1000.0; }
};

// SIMD-optimized L2 squared distance
#if defined(__AVX512F__)
#include <immintrin.h>
inline float l2_distance_sq_avx512(const float* a, const float* b, int dim) {
    // Two accumulators hide the 4-cycle fmadd latency (matches AVX2 path style).
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 31 < dim; i += 32) {
        __m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(a + i),      _mm512_loadu_ps(b + i));
        __m512 d1 = _mm512_sub_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16));
        sum0 = _mm512_fmadd_ps(d0, d0, sum0);
        sum1 = _mm512_fmadd_ps(d1, d1, sum1);
    }
    for (; i + 15 < dim; i += 16) {
        __m512 d = _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
        sum0 = _mm512_fmadd_ps(d, d, sum0);
    }
    float dist = _mm512_reduce_add_ps(_mm512_add_ps(sum0, sum1));
    for (; i < dim; i++) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}
#endif

#if defined(__AVX2__)
#include <immintrin.h>
inline float l2_distance_sq_avx2(const float* a, const float* b, int dim) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 15 < dim; i += 16) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        __m256 diff0 = _mm256_sub_ps(va0, vb0);
        sum0 = _mm256_fmadd_ps(diff0, diff0, sum0);
        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        __m256 diff1 = _mm256_sub_ps(va1, vb1);
        sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);
    }
    __m256 sum = _mm256_add_ps(sum0, sum1);
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 s = _mm_add_ps(hi, lo);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float dist;
    _mm_store_ss(&dist, s);
    for (; i < dim; i++) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}
#endif

// Scalar fallback
inline float l2_distance_sq_scalar(const float* a, const float* b, int dim) {
    float dist = 0;
    for (int d = 0; d < dim; d++) {
        float diff = a[d] - b[d];
        dist += diff * diff;
    }
    return dist;
}

// Auto-dispatch based on compile-time flags
inline float l2_distance_sq(const float* a, const float* b, int dim) {
#if defined(__AVX512F__)
    return l2_distance_sq_avx512(a, b, dim);
#elif defined(__AVX2__)
    return l2_distance_sq_avx2(a, b, dim);
#else
    return l2_distance_sq_scalar(a, b, dim);
#endif
}

// SIMD-optimized squared norm ||a||² = dot(a, a)
#if defined(__AVX512F__)
inline float dot_self_avx512(const float* a, int dim) {
    __m512 sum = _mm512_setzero_ps();
    int i = 0;
    for (; i + 15 < dim; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        sum = _mm512_fmadd_ps(va, va, sum);
    }
    float result = _mm512_reduce_add_ps(sum);
    for (; i < dim; i++) result += a[i] * a[i];
    return result;
}
#endif

#if defined(__AVX2__)
inline float dot_self_avx2(const float* a, int dim) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 15 < dim; i += 16) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        sum0 = _mm256_fmadd_ps(va0, va0, sum0);
        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        sum1 = _mm256_fmadd_ps(va1, va1, sum1);
    }
    __m256 sum = _mm256_add_ps(sum0, sum1);
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 s = _mm_add_ps(hi, lo);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float result;
    _mm_store_ss(&result, s);
    for (; i < dim; i++) result += a[i] * a[i];
    return result;
}
#endif

inline float dot_self(const float* a, int dim) {
#if defined(__AVX512F__)
    return dot_self_avx512(a, dim);
#elif defined(__AVX2__)
    return dot_self_avx2(a, dim);
#else
    float s = 0;
    for (int i = 0; i < dim; i++) s += a[i] * a[i];
    return s;
#endif
}

} // namespace score
