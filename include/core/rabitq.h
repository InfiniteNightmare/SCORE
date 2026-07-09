#pragma once
#include "core/common.h"
#include <immintrin.h>

namespace score {

// RaBitQ 1-bit quantization
class RaBitQuantizer {
public:
    RaBitQuantizer() = default;

    // Corrective factors per vector (1-bit RaBitQ)
    struct Factor {
        float sqr_res;     // ||d - c||² (residual norm squared)
        float error;       // Theoretical error bound
        float factor_ppc;  // f_add = ||res||² + 2*||res||²*<cent,xu_cb>/<res,xu_cb>
        float factor_ip;   // f_rescale = -2*||res||² / <res, xu_cb>
    };

    void init(int dim) {
        dim_ = dim;
        dim_padded_ = ((dim + 63) / 64) * 64;
        words_per_vec_ = dim_padded_ / 64;
    }

    // ---- 1-bit quantization (RaBitQ) ----

    // Quantize a rotated vector to binary: sign(x'_i) → 1 bit
    // Returns packed uint64 array
    void quantize_binary(const float* x_rot, uint64_t* code) const {
        for (int w = 0; w < words_per_vec_; w++) {
            uint64_t bits = 0;
            for (int b = 0; b < 64; b++) {
                int idx = w * 64 + b;
                if (idx < dim_ && x_rot[idx] >= 0)
                    bits |= (1ULL << b);
            }
            code[w] = bits;
        }
    }

    // Batch quantize
    void quantize_binary_batch(const float* data_rot, uint64_t* codes, int n) const {
        #pragma omp parallel for schedule(dynamic, 256)
        for (int i = 0; i < n; i++) {
            quantize_binary(data_rot + (size_t)i * dim_, codes + (size_t)i * words_per_vec_);
        }
    }

    // ---- Hamming distance (SIMD-optimized) ----

    // Compute Hamming distance between two binary codes
    int hamming_distance(const uint64_t* a, const uint64_t* b) const {
        int dist = 0;
        for (int w = 0; w < words_per_vec_; w++) {
            dist += __builtin_popcountll(a[w] ^ b[w]);
        }
        return dist;
    }

    // ---- Distance estimation from binary codes (RaBitQ) ----

    // Compute corrective factors for a vector (1-bit RaBitQ)
    // Formula from RaBitQ-Library (rabitq_impl.hpp one_bit_code_with_factor):
    //   residual = x_rot - centroid (for global centroid, centroid = 0)
    //   xu_cb = sign(residual) - 0.5
    //   ip_resi_xucb = <residual, xu_cb>
    //   f_add = ||residual||² + 2*||residual||²*<centroid,xu_cb>/<residual,xu_cb>
    //   f_rescale = -2*||residual||² / <residual, xu_cb>
    Factor compute_factor(const float* x_rot, const float* centroid,
                          const uint64_t* binary_code, float norm_x) const {
        Factor f;
        (void)norm_x;

        float l2_sqr = 0;
        float ip_resi_xucb = 0;  // <residual, sign(residual) - 0.5>
        float ip_cent_xucb = 0;  // <centroid, sign(residual) - 0.5>

        for (int d = 0; d < dim_; d++) {
            float c = centroid ? centroid[d] : 0.0f;
            float res = x_rot[d] - c;
            l2_sqr += res * res;

            int bit = (binary_code[d / 64] >> (d % 64)) & 1;
            float xu_cb_d = (float)bit - 0.5f;

            ip_resi_xucb += res * xu_cb_d;
            ip_cent_xucb += c * xu_cb_d;
        }

        f.sqr_res = l2_sqr;

        if (ip_resi_xucb == 0) ip_resi_xucb = 1e-10f;

        // f_rescale (stored in factor_ip): -2 * ||residual||² / <residual, xu_cb>
        f.factor_ip = -2.0f * l2_sqr / ip_resi_xucb;

        // f_add (stored in factor_ppc): ||residual||² + 2*||residual||²*<cent,xu_cb>/<res,xu_cb>
        f.factor_ppc = l2_sqr + 2.0f * l2_sqr * ip_cent_xucb / ip_resi_xucb;

        // Error bound
        float l2_norm = std::sqrt(l2_sqr);
        float xu_cb_norm_sqr = (float)dim_ * 0.25f;  // ||xu_cb||² = D * 0.25
        float tmp_error = l2_norm * 1.9f *
            std::sqrt(std::max(0.0f, (l2_sqr * xu_cb_norm_sqr / (ip_resi_xucb * ip_resi_xucb) - 1.0f)) / (float)(dim_ - 1));
        f.error = 2.0f * tmp_error;

        return f;
    }

    // Estimate squared distance using binary codes and factors (1-bit RaBitQ)
    //
    // Formula from RaBitQ-Library (estimator.hpp line 67-68):
    //   est = f_add + g_add + f_rescale * (ip_x0_qr + k1xsumq)
    //
    // Where:
    //   f_add = fac.factor_ppc
    //   g_add = sqr_q_norm = ||H(q_r)||²
    //   f_rescale = fac.factor_ip
    //   k1xsumq = sum(q_rot) * cb = sum(q_rot) * (-0.5)
    //   ip_x0_qr = popcount(q_code AND db_code) - 0.5 * popcount(db_code)
    float estimate_distance(const float* q_rot, const uint64_t* q_code,
                            const uint64_t* db_code, const Factor& fac,
                            float sqr_q_norm) const {
        // popcount(q_code AND db_code)
        int ip_popcount = 0;
        for (int w = 0; w < words_per_vec_; w++) {
            ip_popcount += __builtin_popcountll(q_code[w] & db_code[w]);
        }

        // k1xsumq = sum(q_rot) * (-0.5)
        float sum_q = 0;
        for (int d = 0; d < dim_; d++) sum_q += q_rot[d];
        float k1xsumq = sum_q * (-0.5f);

        // ip_x0_qr = popcount(q&db) - 0.5 * popcount(db)
        // For db_code, popcount(db_code) = sum of all sign bits
        float ip_x0_qr = (float)ip_popcount - 0.5f * (float)dim_;

        // est = f_add + g_add + f_rescale * (ip_x0_qr + k1xsumq)
        float est_dist = fac.factor_ppc + sqr_q_norm + fac.factor_ip * (ip_x0_qr + k1xsumq);
        return std::max(0.0f, est_dist);
    }

    // ---- Accessors ----
    int dim() const { return dim_; }
    int words_per_vector() const { return words_per_vec_; }
    int code_bytes_per_vector() const { return words_per_vec_ * sizeof(uint64_t); }

private:
    int dim_ = 0;
    int dim_padded_ = 0;
    int words_per_vec_ = 0;
};

} // namespace score
