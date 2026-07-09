#pragma once

#include "core/ivf.h"
#include "core/posting_lists.h"

#include <sys/mman.h>

#include "rabitqlib/quantization/rabitq.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace score {

// Immutable base index segment shared across per-thread query executors: base
// vectors, RaBitQ config/centroids, per-subspace IVF indices, posting lists,
// and the reordered-PID permutation. Loaded once and shared by value via
// std::shared_ptr<BaseSegment>.
struct BaseSegment {
    ~BaseSegment() {
        if (rabitq_bin_with_assign_) {
            munmap(rabitq_bin_with_assign_, rabitq_bin_with_assign_alloc_bytes_);
            rabitq_bin_with_assign_ = nullptr;
        }
        if (verify_vectors_) {
            munmap(verify_vectors_, verify_vectors_bytes_);
            verify_vectors_ = nullptr;
        }
    }

    int dim = 0;
    int n = 0;
    int rabitq_k = 0;
    int rabitq_padded_dim = 0;
    size_t rabitq_bin_bytes = 0;
    rabitqlib::quant::RabitqConfig query_config;
    std::vector<IVFIndex> ivf_indices;
    std::shared_ptr<PostingLists> posting_lists;
    float* verify_vectors_ = nullptr;
    size_t verify_vectors_bytes_ = 0;
    int verify_vectors_dim = 0;
    std::vector<float> rabitq_rot_centroids;
    std::vector<float> rabitq_centroid_bias;
    char* rabitq_bin_with_assign_ = nullptr;
    size_t rabitq_bin_with_assign_alloc_bytes_ = 0;
    size_t rabitq_bin_with_assign_record_bytes_ = 0;
    std::vector<uint32_t> perm_to_orig;
    std::vector<size_t> cell_size_q25_entries;
};

} // namespace score
