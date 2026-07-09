#pragma once
#include <vector>
#include <algorithm>
#include <cstdint>
#include <string>
#include <fstream>
#include <cerrno>
#include <limits>
#include <stdexcept>
#include <utility>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace score {

// In-memory CSR posting lists. Each subspace has one flat binary file:
//
//   [K: int32][total: int32][offsets: int32[K+1]][ids: uint32[total]]
//
// Loaded into anonymous THP-eligible memory for stable query-time access. Build writes this format directly.
class PostingLists {
public:
    struct CellSpan {
        const uint32_t* data = nullptr;
        size_t count = 0;
    };

    PostingLists() = default;
    PostingLists(const PostingLists&) = delete;
    PostingLists& operator=(const PostingLists&) = delete;
    PostingLists(PostingLists&& other) noexcept { move_from(std::move(other)); }
    PostingLists& operator=(PostingLists&& other) noexcept {
        if (this != &other) {
            release();
            move_from(std::move(other));
        }
        return *this;
    }
    ~PostingLists() {
        release();
    }

    // Build from single-assignment cell IDs (row-major: all_cell_ids[i*M + m])
    void build(const std::vector<size_t>& cells_per_subspace,
               const std::vector<int>& all_cell_ids,
               int n, int M, const std::string& path) {
        for (int m = 0; m < M; m++) {
            int K = (int)cells_per_subspace[m];
            // Count cell sizes
            std::vector<int> sizes(K, 0);
            for (int i = 0; i < n; i++) sizes[all_cell_ids[(size_t)i * M + m]]++;

            // CSR offsets
            std::vector<int> offsets(K + 1, 0);
            for (int k = 0; k < K; k++) offsets[k + 1] = offsets[k] + sizes[k];
            int total = offsets[K];

            // Fill IDs
            std::vector<uint32_t> ids(total);
            std::vector<int> cur(K, 0);
            for (int i = 0; i < n; i++) {
                int c = all_cell_ids[(size_t)i * M + m];
                ids[(size_t)offsets[c] + (size_t)cur[c]++] = (uint32_t)i;
            }

            write_subspace(path, m, K, total, offsets, ids);
        }
    }

    // Build from single-assignment cell IDs in submajor layout: [M][n].
    // If perm_to_orig is provided, posting IDs are emitted in reordered PID space,
    // where new_pid -> original pid is given by perm_to_orig[new_pid].
    void build_submajor(const std::vector<size_t>& cells_per_subspace,
                        const int* submajor_cell_ids,
                        int n, int M, const std::string& path,
                        const std::vector<uint32_t>* perm_to_orig = nullptr) {
        #pragma omp parallel for schedule(dynamic, 1)
        for (int m = 0; m < M; m++) {
            int K = (int)cells_per_subspace[m];
            const int* col = submajor_cell_ids + (size_t)m * n;

            std::vector<int> sizes(K, 0);
            if (perm_to_orig == nullptr) {
                for (int i = 0; i < n; i++) sizes[col[i]]++;
            } else {
                for (int new_pid = 0; new_pid < n; new_pid++) {
                    uint32_t orig = (*perm_to_orig)[(size_t)new_pid];
                    sizes[col[orig]]++;
                }
            }

            std::vector<int> offsets(K + 1, 0);
            for (int k = 0; k < K; k++) offsets[k + 1] = offsets[k] + sizes[k];
            int total = offsets[K];

            std::vector<uint32_t> ids(total);
            std::vector<int> cur(K, 0);
            if (perm_to_orig == nullptr) {
                for (int i = 0; i < n; i++) {
                    int c = col[i];
                    ids[(size_t)offsets[c] + (size_t)cur[c]++] = (uint32_t)i;
                }
            } else {
                for (int new_pid = 0; new_pid < n; new_pid++) {
                    uint32_t orig = (*perm_to_orig)[(size_t)new_pid];
                    int c = col[orig];
                    ids[(size_t)offsets[c] + (size_t)cur[c]++] = (uint32_t)new_pid;
                }
            }

            write_subspace(path, m, K, total, offsets, ids);
        }
    }

    // Load all subspace files into anonymous THP-eligible memory.
    void load(const std::string& path, int M) {
        maps_.resize(M, nullptr);
        map_sizes_.resize(M, 0);
        K_.resize(M, 0);
        offsets_.resize(M, nullptr);
        ids_.resize(M, nullptr);

        for (int m = 0; m < M; m++) {
            std::string fpath = path + "/posting_subspace_" + std::to_string(m) + ".bin";
            int fd = ::open(fpath.c_str(), O_RDONLY);
            if (fd < 0) {
                throw std::runtime_error("Cannot open posting list: " + fpath);
            }

            struct stat st;
            if (fstat(fd, &st) != 0) {
                ::close(fd);
                throw std::runtime_error("Cannot stat posting list: " + fpath);
            }
            if (st.st_size < static_cast<off_t>(2 * sizeof(int))) {
                ::close(fd);
                throw std::runtime_error("Posting list too small: " + fpath);
            }
            if (static_cast<uintmax_t>(st.st_size) > static_cast<uintmax_t>(std::numeric_limits<size_t>::max())) {
                ::close(fd);
                throw std::runtime_error("Posting list too large: " + fpath);
            }
            size_t sz = st.st_size;

            void* ptr = mmap(nullptr, sz, PROT_READ | PROT_WRITE,
                             MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
            if (ptr == MAP_FAILED) {
                ::close(fd);
                throw std::runtime_error("mmap anonymous failed for posting list: " + fpath);
            }
            if (!read_full(fd, ptr, sz, fpath)) {
                ::close(fd);
                munmap(ptr, sz);
                throw std::runtime_error("pread failed for posting list: " + fpath);
            }
            ::close(fd);
            madvise(ptr, sz, MADV_HUGEPAGE);
            madvise(ptr, sz, MADV_WILLNEED);

            const int* base = reinterpret_cast<const int*>(ptr);
            int K = base[0];
            int total = base[1];
            size_t words = 0;
            if (K <= 0 || total < 0 ||
                !checked_int_words(static_cast<size_t>(K), static_cast<size_t>(total), words) ||
                words > std::numeric_limits<size_t>::max() / sizeof(int) ||
                words * sizeof(int) != sz) {
                munmap(ptr, sz);
                throw std::runtime_error("Bad posting list header or size: " + fpath);
            }

            const int* offsets = base + 2;
            if (offsets[0] != 0 || offsets[K] != total) {
                munmap(ptr, sz);
                throw std::runtime_error("Bad posting list offsets: " + fpath);
            }
            for (int c = 0; c < K; ++c) {
                if (offsets[c] > offsets[c + 1]) {
                    munmap(ptr, sz);
                    throw std::runtime_error("Non-monotonic posting list offsets: " + fpath);
                }
            }

            maps_[m] = ptr;
            map_sizes_[m] = sz;
            K_[m] = K;
            offsets_[m] = base + 2;           // K+1 offsets
            ids_[m] = reinterpret_cast<const uint32_t*>(base + 2 + K_[m] + 1);  // point IDs
        }
    }

    const uint32_t* cell_data(int m, int cell) const { return ids_[m] + offsets_[m][cell]; }
    size_t     cell_size(int m, int cell) const { return (size_t)(offsets_[m][cell + 1] - offsets_[m][cell]); }
    CellSpan   cell_span(int m, int cell) const {
        const int begin = offsets_[m][cell];
        const int end = offsets_[m][cell + 1];
        return {ids_[m] + begin, (size_t)(end - begin)};
    }
    int        K(int m) const { return K_[m]; }

    void validate_point_ids(int n) const {
        if (n <= 0) throw std::runtime_error("Invalid point count for posting-list validation");
        for (int m = 0; m < static_cast<int>(K_.size()); ++m) {
            int total = offsets_[m][K_[m]];
            const uint32_t* ids = ids_[m];
            for (int i = 0; i < total; ++i) {
                if (ids[i] >= (uint32_t)n) {
                    throw std::runtime_error("Posting list point id out of range at subspace " + std::to_string(m));
                }
            }
        }
    }

private:
    static void write_subspace(const std::string& path, int m,
                                  int K, int total,
                                  const std::vector<int>& offsets,
                                  const std::vector<uint32_t>& ids) {
        std::string fpath = path + "/posting_subspace_" + std::to_string(m) + ".bin";
        std::ofstream out(fpath, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&K),      sizeof(int));
        out.write(reinterpret_cast<const char*>(&total),  sizeof(int));
        out.write(reinterpret_cast<const char*>(offsets.data()), (K + 1) * sizeof(int));
        out.write(reinterpret_cast<const char*>(ids.data()),     (size_t)total * sizeof(uint32_t));
    }

    std::vector<void*>        maps_;
    std::vector<size_t>       map_sizes_;
    std::vector<int>          K_;
    std::vector<const int*>   offsets_;
    std::vector<const uint32_t*> ids_;

    void release() noexcept {
        for (int m = 0; m < static_cast<int>(maps_.size()); m++) {
            if (maps_[m] && map_sizes_[m] > 0) {
                munmap(maps_[m], map_sizes_[m]);
            }
        }
        maps_.clear();
        map_sizes_.clear();
        K_.clear();
        offsets_.clear();
        ids_.clear();
    }

    void move_from(PostingLists&& other) noexcept {
        maps_ = std::move(other.maps_);
        map_sizes_ = std::move(other.map_sizes_);
        K_ = std::move(other.K_);
        offsets_ = std::move(other.offsets_);
        ids_ = std::move(other.ids_);
        other.maps_.clear();
        other.map_sizes_.clear();
        other.K_.clear();
        other.offsets_.clear();
        other.ids_.clear();
    }

    static bool checked_int_words(size_t K, size_t total, size_t& words) {
        if (K > std::numeric_limits<size_t>::max() - 3) return false;
        size_t header_and_offsets = K + 3; // K + 1 offsets plus K/total header words
        if (total > std::numeric_limits<size_t>::max() - header_and_offsets) return false;
        words = header_and_offsets + total;
        return true;
    }

    static bool read_full(int fd, void* dst, size_t sz, const std::string& fpath) {
        char* out = static_cast<char*>(dst);
        size_t done = 0;
        while (done < sz) {
            size_t remaining = sz - done;
            size_t chunk = std::min(remaining, static_cast<size_t>(std::numeric_limits<ssize_t>::max()));
            ssize_t r = ::pread(fd, out + done, chunk, static_cast<off_t>(done));
            if (r < 0) {
                if (errno == EINTR) continue;
                return false;
            }
            if (r == 0) return false;
            done += static_cast<size_t>(r);
        }
        (void)fpath;
        return true;
    }
};

} // namespace score
