#pragma once
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>

// RaBitQ-Library's FhtKacRotator for bit-exact rotation
#include "rabitqlib/utils/rotator.hpp"

namespace score {

// Rotation types for RaBitQ
enum class RotationType {
    Hadamard,   // Fast Walsh-Hadamard: O(n·log d), deterministic
    QR,         // Random QR: O(n·d²), better decorrelation
    FhtKac,     // 4x FHT + random sign flips (RaBitQ-Library FhtKacRotator)
};

// Random rotation via either Hadamard or QR decomposition
class RandomRotation {
public:
    struct Metadata {
        int dim = 0;
        uint64_t seed = 0;
        int padded_dim = 0;
        RotationType type = RotationType::Hadamard;
    };

    RandomRotation() = default;

    void init(int dim, uint64_t seed = 42, RotationType type = RotationType::Hadamard) {
        dim_ = dim;
        seed_ = seed;
        type_ = type;

        if (type_ == RotationType::Hadamard) {
            // Pad to power of 2 for WHT
            padded_dim_ = 1;
            while (padded_dim_ < dim_) padded_dim_ <<= 1;
            hadamard_tmp_.resize(padded_dim_);
        } else if (type_ == RotationType::FhtKac) {
            // Use RaBitQ-Library's FhtKacRotator (generates random signs via std::random_device)
            padded_dim_ = (int)rabitqlib::rotator_impl::padding_requirement((size_t)dim_, rabitqlib::RotatorType::FhtKacRotator);
            fhtkac_rotator_.reset(new rabitqlib::rotator_impl::FhtKacRotator((size_t)dim_, (size_t)padded_dim_));
            fhtkac_loaded_ = false;
        } else {
            padded_dim_ = dim;
            // Generate random orthogonal matrix via QR decomposition
            generate_qr_matrix(seed);
        }
    }

    // Apply rotation to a single vector
    void apply(const float* x, float* x_rot) const {
        if (type_ == RotationType::Hadamard) {
            apply_hadamard(x, x_rot);
        } else if (type_ == RotationType::FhtKac) {
            apply_fhtkac(x, x_rot);
        } else {
            apply_qr(x, x_rot);
        }
    }

    // Batch apply
    void apply_batch(const float* data, float* rotated, int n) const {
        if (dim_ <= 1) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; i++) rotated[i] = data[i];
            return;
        }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            apply(data + (size_t)i * dim_, rotated + (size_t)i * dim_);
        }
    }

    int dim() const { return dim_; }
    RotationType type() const { return type_; }
    int padded_dim() const { return padded_dim_; }

    // Get padding requirement for FhtKac: round up to multiple of 64
    static int padding_requirement(int dim, RotationType type) {
        if (type == RotationType::FhtKac) {
            return ((dim + 63) / 64) * 64;
        }
        return dim;  // Hadamard: power of 2, QR: no padding
    }

    void save(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        save(out);
    }

    void save(std::ofstream& out) const {
        int d = dim_;
        int t = static_cast<int>(type_);
        out.write(reinterpret_cast<const char*>(&d), sizeof(int));
        out.write(reinterpret_cast<const char*>(&seed_), sizeof(uint64_t));
        out.write(reinterpret_cast<const char*>(&padded_dim_), sizeof(int));
        out.write(reinterpret_cast<const char*>(&t), sizeof(int));
        if (type_ == RotationType::QR) {
            out.write(reinterpret_cast<const char*>(qr_matrix_.data()),
                      (size_t)dim_ * dim_ * sizeof(float));
        } else if (type_ == RotationType::FhtKac) {
            ensure_fhtkac_rotator();
            fhtkac_rotator_->save(out);
        }
    }

    static Metadata load_metadata(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) {
            throw std::runtime_error("Cannot open rotation file: " + path);
        }
        Metadata meta;
        int type_value = 0;
        in.read(reinterpret_cast<char*>(&meta.dim), sizeof(int));
        in.read(reinterpret_cast<char*>(&meta.seed), sizeof(uint64_t));
        in.read(reinterpret_cast<char*>(&meta.padded_dim), sizeof(int));
        in.read(reinterpret_cast<char*>(&type_value), sizeof(int));
        if (!in) {
            throw std::runtime_error("Short rotation header: " + path);
        }
        if (meta.dim <= 0 || type_value < (int)RotationType::Hadamard ||
            type_value > (int)RotationType::FhtKac) {
            throw std::runtime_error("Invalid rotation header: " + path);
        }
        meta.type = static_cast<RotationType>(type_value);
        if (meta.type == RotationType::Hadamard) {
            int expected = 1;
            while (expected < meta.dim) expected <<= 1;
            meta.padded_dim = expected;
        } else if (meta.type == RotationType::FhtKac) {
            meta.padded_dim = padding_requirement(meta.dim, meta.type);
        } else {
            meta.padded_dim = meta.dim;
        }
        return meta;
    }

    void load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) {
            throw std::runtime_error("Cannot open rotation file: " + path);
        }
        int d, t;
        in.read(reinterpret_cast<char*>(&d), sizeof(int));
        in.read(reinterpret_cast<char*>(&seed_), sizeof(uint64_t));
        in.read(reinterpret_cast<char*>(&padded_dim_), sizeof(int));
        in.read(reinterpret_cast<char*>(&t), sizeof(int));
        if (!in) {
            throw std::runtime_error("Short rotation header: " + path);
        }
        dim_ = d;
        type_ = static_cast<RotationType>(t);

        if (type_ == RotationType::Hadamard) {
            int expected = 1;
            while (expected < dim_) expected <<= 1;
            padded_dim_ = expected;
            hadamard_tmp_.resize(padded_dim_);
        } else if (type_ == RotationType::FhtKac) {
            // Load flip signs from file using the library's format
            padded_dim_ = (int)rabitqlib::rotator_impl::padding_requirement((size_t)dim_, rabitqlib::RotatorType::FhtKacRotator);
            // Create rotator with correct dimensions, then load flip signs from file
            fhtkac_rotator_.reset(new rabitqlib::rotator_impl::FhtKacRotator((size_t)dim_, (size_t)padded_dim_));
            fhtkac_rotator_->load(in);
            if (!in) {
                throw std::runtime_error("Short FhtKac rotation payload: " + path);
            }
            fhtkac_loaded_ = true;
        } else {
            padded_dim_ = dim_;
            qr_matrix_.resize((size_t)dim_ * dim_);
            in.read(reinterpret_cast<char*>(qr_matrix_.data()),
                    (size_t)dim_ * dim_ * sizeof(float));
            if (!in) {
                throw std::runtime_error("Short QR rotation payload: " + path);
            }
        }
    }

private:
    int dim_ = 0;
    uint64_t seed_ = 42;
    int padded_dim_ = 0;
    RotationType type_ = RotationType::Hadamard;
    std::vector<float> qr_matrix_;  // dim x dim, row-major (only for QR)

    // Library's FhtKacRotator for bit-exact FhtKac rotation
    mutable std::unique_ptr<rabitqlib::rotator_impl::FhtKacRotator> fhtkac_rotator_;
    mutable bool fhtkac_loaded_ = false;
    mutable std::vector<float> hadamard_tmp_;  // Pre-allocated WHT workspace

    void ensure_fhtkac_rotator() const {
        if (!fhtkac_rotator_) {
            size_t pd = rabitqlib::rotator_impl::padding_requirement((size_t)dim_, rabitqlib::RotatorType::FhtKacRotator);
            fhtkac_rotator_.reset(new rabitqlib::rotator_impl::FhtKacRotator((size_t)dim_, pd));
        }
    }

    void apply_hadamard(const float* x, float* x_rot) const {
        if (dim_ <= 1) {
            if (dim_ == 1) x_rot[0] = x[0];
            return;
        }

        // Thread-local scratch buffer: safe for OMP parallel apply_batch.
        // hadamard_tmp_ is kept for single-threaded contexts only.
        thread_local std::vector<float> tl_tmp;
        if ((int)tl_tmp.size() < padded_dim_) tl_tmp.resize(padded_dim_);
        float* tmp = tl_tmp.data();
        std::memcpy(tmp, x, dim_ * sizeof(float));
        if (padded_dim_ > dim_)
            std::memset(tmp + dim_, 0, (padded_dim_ - dim_) * sizeof(float));

        for (int stride = 1; stride < padded_dim_; stride <<= 1) {
            for (int i = 0; i + stride < padded_dim_; i += stride << 1) {
                for (int j = 0; j < stride; j++) {
                    float u = tmp[i + j];
                    float v = tmp[i + j + stride];
                    tmp[i + j] = u + v;
                    tmp[i + j + stride] = u - v;
                }
            }
        }

        float inv_sqrt_dim = 1.0f / std::sqrt(static_cast<float>(padded_dim_));
        for (int i = 0; i < dim_; i++) x_rot[i] = tmp[i] * inv_sqrt_dim;
    }

    void apply_qr(const float* x, float* x_rot) const {
        // Matrix-vector multiply: x_rot = Q @ x
        for (int i = 0; i < dim_; i++) {
            float sum = 0;
            const float* row = qr_matrix_.data() + (size_t)i * dim_;
            for (int j = 0; j < dim_; j++) {
                sum += row[j] * x[j];
            }
            x_rot[i] = sum;
        }
    }

    // FhtKac rotation: delegate to RaBitQ-Library's FhtKacRotator (bit-exact AVX implementation)
    void apply_fhtkac(const float* x, float* x_rot) const {
        ensure_fhtkac_rotator();
        fhtkac_rotator_->rotate(x, x_rot);
    }

    // Generate random orthogonal matrix via modified Gram-Schmidt QR
    void generate_qr_matrix(uint64_t seed) {
        int n = dim_;
        qr_matrix_.resize((size_t)n * n);

        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        // Fill with random Gaussian entries
        for (int i = 0; i < n * n; i++) {
            qr_matrix_[i] = dist(rng);
        }

        // Modified Gram-Schmidt: orthogonalize columns
        // qr_matrix_ is stored row-major, so column j has elements at [i*n+j]
        for (int j = 0; j < n; j++) {
            // Compute norm of column j
            float norm = 0;
            for (int i = 0; i < n; i++) {
                float v = qr_matrix_[(size_t)i * n + j];
                norm += v * v;
            }
            norm = std::sqrt(norm);
            if (norm < 1e-10f) continue;
            float inv_norm = 1.0f / norm;

            // Normalize column j
            for (int i = 0; i < n; i++) {
                qr_matrix_[(size_t)i * n + j] *= inv_norm;
            }

            // Remove component along column j from columns j+1..n-1
            for (int k = j + 1; k < n; k++) {
                float dot = 0;
                for (int i = 0; i < n; i++) {
                    dot += qr_matrix_[(size_t)i * n + j] * qr_matrix_[(size_t)i * n + k];
                }
                for (int i = 0; i < n; i++) {
                    qr_matrix_[(size_t)i * n + k] -= dot * qr_matrix_[(size_t)i * n + j];
                }
            }
        }
    }
};

} // namespace score
