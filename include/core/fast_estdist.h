#pragma once
//
// Query-side RaBitQ helper layout.

#include <cstdint>
#include <cstddef>

namespace score {

struct RabitqFactorTriplet {
    float f_add;
    float f_rescale;
    float f_error;
};
static_assert(sizeof(RabitqFactorTriplet) == 12, "RaBitQ factor layout must stay packed");

// Transpose query binary codes from block-major to bit-plane-major layout.
//
// Block-major (rabitqlib):  [blk0_b0, blk0_b1, blk0_b2, blk0_b3, blk1_b0, ...]
// Bit-plane-major (ours):   [blk0_b0, blk1_b0, ..., blkN_b0, blk0_b1, ...]
//
// After transpose, each bit plane is contiguous → sequential AVX-512 load.
// Cost: num_blk*4 assignments = ~60 ops for D=960. Negligible.
static inline void transpose_query_to_bitplane(
    const uint64_t* block_major,   // size: num_blk * B_QUERY
    uint64_t* bitplane_major,      // size: num_blk * B_QUERY (same total)
    size_t num_blk,
    size_t b_query = 4)
{
    for (size_t j = 0; j < b_query; j++) {
        for (size_t i = 0; i < num_blk; i++) {
            bitplane_major[j * num_blk + i] = block_major[i * b_query + j];
        }
    }
}

} // namespace score
