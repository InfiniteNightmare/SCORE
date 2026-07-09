#pragma once

#include "update/update_state.h"
#include "update/update_wal.h"

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace score {

// Build per-subspace CSR posting lists (offsets + ids) from per-row cell/pid
// sources: count cells -> prefix-sum offsets -> scatter pids via cursors.
// Shared by L0Block and SealedDeltaSegment construction (and the delta views).
// CellFn: (row, subspace) -> uint16_t cell; PidFn: (row) -> uint32_t pid.
template <typename CellFn, typename PidFn>
inline void build_posting_csr(const std::vector<size_t>& cells_per_subspace,
                              size_t n_rows, CellFn cell_of, PidFn pid_of,
                              std::vector<std::vector<uint32_t>>& posting_offsets,
                              std::vector<std::vector<uint32_t>>& posting_ids) {
    posting_offsets.assign(cells_per_subspace.size(), {});
    posting_ids.assign(cells_per_subspace.size(), {});
    for (size_t m = 0; m < cells_per_subspace.size(); ++m) {
        const size_t K = cells_per_subspace[m];
        if (K > (size_t)std::numeric_limits<uint32_t>::max()) {
            throw std::invalid_argument("delta cell count exceeds uint32 range");
        }
        std::vector<uint32_t> sizes(K, 0);
        for (size_t row = 0; row < n_rows; ++row) sizes[(size_t)cell_of(row, m)]++;
        std::vector<uint32_t>& offsets = posting_offsets[m];
        offsets.assign(K + 1, 0);
        for (size_t cell = 0; cell < K; ++cell) offsets[cell + 1] = offsets[cell] + sizes[cell];
        posting_ids[m].assign(n_rows, 0);
        std::vector<uint32_t> cursors(K, 0);
        for (size_t row = 0; row < n_rows; ++row) {
            const uint16_t cell = cell_of(row, m);
            const uint32_t offset = offsets[(size_t)cell] + cursors[(size_t)cell]++;
            posting_ids[m][(size_t)offset] = pid_of(row);
        }
    }
}

class L0Block {
public:
    class Builder {
    public:
        Builder(const std::vector<size_t>& cells_per_subspace,
                uint32_t verify_dim,
                uint32_t rabitq_record_bytes,
                bool build_posting_index = true)
            : block_(std::make_shared<L0Block>()),
              cells_per_subspace_(cells_per_subspace),
              build_posting_index_(build_posting_index) {
            if (cells_per_subspace.empty()) {
                throw std::invalid_argument("L0Block requires at least one subspace");
            }
            validate_row_cell_width(cells_per_subspace);
            if (verify_dim == 0 || rabitq_record_bytes == 0) {
                throw std::invalid_argument("L0Block payload dimensions must be positive");
            }
            block_->verify_dim_ = verify_dim;
            block_->rabitq_record_bytes_ = rabitq_record_bytes;
            block_->num_subspaces_ = (uint32_t)cells_per_subspace.size();
        }

        void reserve_rows(size_t rows) {
            if (rows > (size_t)std::numeric_limits<uint32_t>::max()) {
                throw std::invalid_argument("L0Block insert count exceeds uint32 range");
            }
            block_->orig_ids_.reserve(rows);
            block_->pids_.reserve(rows);
            block_->verify_rotated_.reserve(rows * (size_t)block_->verify_dim_);
            block_->rabitq_records_.reserve(rows * (size_t)block_->rabitq_record_bytes_);
            block_->row_cells_.reserve(rows * cells_per_subspace_.size());
        }

        size_t row_count() const { return block_->orig_ids_.size(); }

        uint32_t append_insert_payload(const UpdateWalHeader& header,
                                       const std::vector<uint8_t>& payload) {
            if (header.num_subspaces != block_->num_subspaces_ ||
                header.verify_dim != block_->verify_dim_ ||
                header.rabitq_record_bytes != block_->rabitq_record_bytes_) {
                throw std::invalid_argument("L0Block insert WAL header dimension mismatch");
            }

            size_t offset = 0;
            uint32_t orig_id =
                impl::read_payload_pod<uint32_t>(payload, offset, "insert orig_id");
            uint32_t global_pid =
                impl::read_payload_pod<uint32_t>(payload, offset, "insert global_pid");

            size_t verify_bytes = (size_t)block_->verify_dim_ * sizeof(float);
            size_t rabitq_bytes = (size_t)block_->rabitq_record_bytes_;
            if (payload.size() - offset <
                verify_bytes + rabitq_bytes + cells_per_subspace_.size() * sizeof(uint16_t)) {
                throw std::runtime_error("short update WAL payload while reading insert payload");
            }
            const uint8_t* verify_bytes_ptr = payload.data() + offset;
            offset += verify_bytes;
            const uint8_t* rabitq_bytes_ptr = payload.data() + offset;
            offset += rabitq_bytes;

            size_t cell_offset = offset;
            for (size_t m = 0; m < cells_per_subspace_.size(); ++m) {
                uint16_t cell =
                    impl::read_payload_pod<uint16_t>(payload, cell_offset, "insert cell");
                if (cell >= cells_per_subspace_[m]) {
                    throw std::invalid_argument("L0Block insert subspace cell is out of range");
                }
            }

            append_insert_parts(orig_id, global_pid, verify_bytes_ptr, rabitq_bytes_ptr);
            for (size_t m = 0; m < cells_per_subspace_.size(); ++m) {
                uint16_t cell =
                    impl::read_payload_pod<uint16_t>(payload, offset, "insert cell");
                block_->row_cells_.push_back(cell);
            }
            return global_pid;
        }

        void append_insert_parts(uint32_t orig_id,
                                 uint32_t global_pid,
                                 const float* verify_rotated,
                                 const uint8_t* rabitq_record,
                                 const uint16_t* subspace_cells) {
            if (verify_rotated == nullptr || rabitq_record == nullptr ||
                subspace_cells == nullptr) {
                throw std::invalid_argument("L0Block insert raw payload pointer is null");
            }
            for (size_t m = 0; m < cells_per_subspace_.size(); ++m) {
                uint16_t cell = subspace_cells[m];
                if ((size_t)cell >= cells_per_subspace_[m]) {
                    throw std::invalid_argument("L0Block insert subspace cell is out of range");
                }
            }
            append_insert_parts(orig_id,
                                global_pid,
                                reinterpret_cast<const uint8_t*>(verify_rotated),
                                rabitq_record);
            for (size_t m = 0; m < cells_per_subspace_.size(); ++m) {
                block_->row_cells_.push_back(subspace_cells[m]);
            }
        }

        // Bulk-append already-encoded contiguous SoA rows, avoiding the per-row
        // resize+memcpy of append_insert_parts. The caller (encode path) must
        // guarantee: PIDs form a contiguous ascending range starting at first_pid
        // (EncodedL0InsertBatch.global_pid(row)=pid_begin+row), and verify/rabitq/
        // cells are row-major with strides matching verify_dim_/rabitq_record_bytes_/
        // num_subspaces_. Cell range is still validated (cheap) for safety.
        void append_contiguous_rows(uint32_t first_pid,
                                    size_t row_count_in,
                                    const uint32_t* orig_ids_in,
                                    const float* verify_rotated_in,
                                    const uint8_t* rabitq_records_in,
                                    const uint16_t* subspace_cells_in) {
            if (row_count_in == 0) return;
            if (orig_ids_in == nullptr || verify_rotated_in == nullptr ||
                rabitq_records_in == nullptr || subspace_cells_in == nullptr) {
                throw std::invalid_argument("L0Block contiguous append raw pointer is null");
            }
            if (block_->orig_ids_.size() + row_count_in >
                (size_t)std::numeric_limits<uint32_t>::max()) {
                throw std::invalid_argument("L0Block insert count exceeds uint32 range");
            }
            size_t existing = block_->orig_ids_.size();
            uint32_t expected_first =
                existing == 0 ? first_pid : block_->pid_begin_ + (uint32_t)existing;
            if (first_pid != expected_first) {
                throw std::invalid_argument(
                    "L0Block contiguous append PID range is not contiguous with prior rows");
            }
            const size_t M = cells_per_subspace_.size();
            for (size_t r = 0; r < row_count_in; ++r) {
                for (size_t m = 0; m < M; ++m) {
                    uint16_t cell = subspace_cells_in[r * M + m];
                    if ((size_t)cell >= cells_per_subspace_[m]) {
                        throw std::invalid_argument(
                            "L0Block insert subspace cell is out of range");
                    }
                }
            }
            if (existing == 0) block_->pid_begin_ = first_pid;
            block_->orig_ids_.insert(block_->orig_ids_.end(),
                                     orig_ids_in, orig_ids_in + row_count_in);
            size_t old_pids = block_->pids_.size();
            block_->pids_.resize(old_pids + row_count_in);
            for (size_t i = 0; i < row_count_in; ++i) {
                block_->pids_[old_pids + i] = first_pid + (uint32_t)i;
            }
            block_->pid_count_ = (uint32_t)(existing + row_count_in);
            size_t verify_pos = block_->verify_rotated_.size();
            block_->verify_rotated_.resize(
                verify_pos + row_count_in * (size_t)block_->verify_dim_);
            std::memcpy(block_->verify_rotated_.data() + verify_pos,
                        verify_rotated_in,
                        row_count_in * (size_t)block_->verify_dim_ * sizeof(float));
            size_t rabitq_pos = block_->rabitq_records_.size();
            block_->rabitq_records_.resize(
                rabitq_pos + row_count_in * (size_t)block_->rabitq_record_bytes_);
            std::memcpy(block_->rabitq_records_.data() + rabitq_pos,
                        rabitq_records_in,
                        row_count_in * (size_t)block_->rabitq_record_bytes_);
            block_->row_cells_.insert(block_->row_cells_.end(),
                                      subspace_cells_in,
                                      subspace_cells_in + row_count_in * M);
        }

        std::shared_ptr<const L0Block> finish(uint64_t commit_lsn) {
            if (row_count() == 0) return nullptr;
            block_->commit_lsn_ = commit_lsn;
            block_->dense_pid_order_ = true;
            if (build_posting_index_) {
                build_postings();
            }
            block_->orig_ids_.shrink_to_fit();
            block_->pids_.shrink_to_fit();
            block_->verify_rotated_.shrink_to_fit();
            block_->rabitq_records_.shrink_to_fit();
            block_->row_cells_.shrink_to_fit();
            for (std::vector<uint32_t>& offsets : block_->posting_offsets_) offsets.shrink_to_fit();
            for (std::vector<uint32_t>& ids : block_->posting_ids_) ids.shrink_to_fit();
            return block_;
        }

    private:
        void append_insert_parts(uint32_t orig_id,
                                 uint32_t global_pid,
                                 const uint8_t* verify_bytes,
                                 const uint8_t* rabitq_bytes) {
            if (block_->orig_ids_.size() > (size_t)std::numeric_limits<uint32_t>::max()) {
                throw std::invalid_argument("L0Block insert count exceeds uint32 range");
            }
            if (block_->orig_ids_.empty()) {
                block_->pid_begin_ = global_pid;
            } else {
                uint32_t expected_pid = block_->pid_begin_ + (uint32_t)block_->orig_ids_.size();
                if (global_pid != expected_pid) {
                    throw std::invalid_argument("L0Block inserts must have a contiguous PID range");
                }
            }

            block_->orig_ids_.push_back(orig_id);
            block_->pids_.push_back(global_pid);
            block_->pid_count_ = (uint32_t)block_->orig_ids_.size();

            size_t verify_pos = block_->verify_rotated_.size();
            block_->verify_rotated_.resize(verify_pos + (size_t)block_->verify_dim_);
            std::memcpy(block_->verify_rotated_.data() + verify_pos,
                        verify_bytes,
                        (size_t)block_->verify_dim_ * sizeof(float));

            size_t rabitq_pos = block_->rabitq_records_.size();
            block_->rabitq_records_.resize(rabitq_pos + (size_t)block_->rabitq_record_bytes_);
            std::memcpy(block_->rabitq_records_.data() + rabitq_pos,
                        rabitq_bytes,
                        (size_t)block_->rabitq_record_bytes_);
        }

        void build_postings() {
            const size_t M = cells_per_subspace_.size();
            build_posting_csr(
                cells_per_subspace_, block_->orig_ids_.size(),
                [&](size_t row, size_t m) {
                    return block_->row_cells_[row * M + m];
                },
                [&](size_t row) { return block_->pids_[row]; },
                block_->posting_offsets_, block_->posting_ids_);
        }

        std::shared_ptr<L0Block> block_;
        std::vector<size_t> cells_per_subspace_;
        bool build_posting_index_ = true;
    };

    static std::shared_ptr<const L0Block> from_inserts(
        uint64_t commit_lsn,
        const std::vector<UpdateWalInsert>& inserts,
        const std::vector<size_t>& cells_per_subspace,
        uint32_t verify_dim,
        uint32_t rabitq_record_bytes) {
        if (inserts.empty()) {
            throw std::invalid_argument("L0Block requires at least one insert");
        }
        if (cells_per_subspace.empty()) {
            throw std::invalid_argument("L0Block requires at least one subspace");
        }
        validate_row_cell_width(cells_per_subspace);
        if (verify_dim == 0 || rabitq_record_bytes == 0) {
            throw std::invalid_argument("L0Block payload dimensions must be positive");
        }
        if (inserts.size() > (size_t)std::numeric_limits<uint32_t>::max()) {
            throw std::invalid_argument("L0Block insert count exceeds uint32 range");
        }

        auto block = std::make_shared<L0Block>();
        block->commit_lsn_ = commit_lsn;
        uint32_t min_pid = std::numeric_limits<uint32_t>::max();
        uint32_t max_pid = 0;
        for (const UpdateWalInsert& insert : inserts) {
            min_pid = std::min(min_pid, insert.global_pid);
            max_pid = std::max(max_pid, insert.global_pid);
        }
        uint64_t pid_range = (uint64_t)max_pid - (uint64_t)min_pid + 1ull;
        if (pid_range > (uint64_t)std::numeric_limits<uint32_t>::max()) {
            throw std::invalid_argument("L0Block PID range exceeds uint32 range");
        }
        block->pid_begin_ = min_pid;
        block->pid_count_ = (uint32_t)pid_range;
        block->verify_dim_ = verify_dim;
        block->rabitq_record_bytes_ = rabitq_record_bytes;
        block->num_subspaces_ = (uint32_t)cells_per_subspace.size();

        block->orig_ids_.resize(inserts.size());
        block->pids_.resize(inserts.size());
        block->pid_to_row_.assign((size_t)block->pid_count_,
                                  std::numeric_limits<uint32_t>::max());
        block->verify_rotated_.resize(inserts.size() * (size_t)verify_dim);
        block->rabitq_records_.resize(inserts.size() * (size_t)rabitq_record_bytes);
        block->row_cells_.resize(inserts.size() * cells_per_subspace.size());

        for (size_t row = 0; row < inserts.size(); ++row) {
            const UpdateWalInsert& insert = inserts[row];
            uint32_t pid_offset = insert.global_pid - block->pid_begin_;
            if (block->pid_to_row_[(size_t)pid_offset] !=
                std::numeric_limits<uint32_t>::max()) {
                throw std::invalid_argument("L0Block inserts contain duplicate PID");
            }
            if (insert.verify_rotated.size() != (size_t)verify_dim) {
                throw std::invalid_argument("L0Block insert verify dimension mismatch");
            }
            if (insert.rabitq_record.size() != (size_t)rabitq_record_bytes) {
                throw std::invalid_argument("L0Block insert RaBitQ record size mismatch");
            }
            if (insert.subspace_cells.size() != cells_per_subspace.size()) {
                throw std::invalid_argument("L0Block insert subspace cell count mismatch");
            }
            for (size_t m = 0; m < cells_per_subspace.size(); ++m) {
                if (insert.subspace_cells[m] >= cells_per_subspace[m]) {
                    throw std::invalid_argument("L0Block insert subspace cell is out of range");
                }
            }

            block->orig_ids_[row] = insert.orig_id;
            block->pids_[row] = insert.global_pid;
            block->pid_to_row_[(size_t)pid_offset] = (uint32_t)row;
            std::copy(insert.verify_rotated.begin(), insert.verify_rotated.end(),
                      block->verify_rotated_.begin() + (std::ptrdiff_t)(row * (size_t)verify_dim));
            std::copy(insert.rabitq_record.begin(), insert.rabitq_record.end(),
                      block->rabitq_records_.begin() +
                          (std::ptrdiff_t)(row * (size_t)rabitq_record_bytes));
            auto row_cells_begin =
                block->row_cells_.begin() +
                (std::ptrdiff_t)(row * cells_per_subspace.size());
            for (size_t m = 0; m < insert.subspace_cells.size(); ++m) {
                row_cells_begin[(std::ptrdiff_t)m] = (uint16_t)insert.subspace_cells[m];
            }
        }

        build_posting_csr(
            cells_per_subspace, inserts.size(),
            [&](size_t row, size_t m) { return inserts[row].subspace_cells[m]; },
            [&](size_t row) { return inserts[row].global_pid; },
            block->posting_offsets_, block->posting_ids_);
        block->dense_pid_order_ = block->compute_dense_pid_order();
        if (block->dense_pid_order_) {
            block->pid_to_row_.clear();
        }

        return block;
    }

    uint64_t commit_lsn() const { return commit_lsn_; }
    uint32_t pid_begin() const { return pid_begin_; }
    uint32_t pid_count() const { return pid_count_; }
    uint32_t row_count() const { return (uint32_t)orig_ids_.size(); }
    uint32_t pid_end() const { return pid_begin_ + pid_count_; }
    bool dense_pid_order() const { return dense_pid_order_; }
    bool contains_pid(uint32_t pid) const {
        if (pid < pid_begin_) return false;
        uint64_t offset = (uint64_t)pid - (uint64_t)pid_begin_;
        if (offset >= (uint64_t)pid_count_) return false;
        if (dense_pid_order_) return true;
        return !pid_to_row_.empty() &&
               pid_to_row_[(size_t)offset] != std::numeric_limits<uint32_t>::max();
    }
    uint32_t local_row_for_pid(uint32_t pid) const {
        if (!contains_pid(pid)) throw std::out_of_range("L0Block PID out of range");
        if (dense_pid_order_) return pid - pid_begin_;
        uint32_t row = pid_to_row_[(size_t)(pid - pid_begin_)];
        if (row == std::numeric_limits<uint32_t>::max()) {
            throw std::out_of_range("L0Block PID out of range");
        }
        return row;
    }
    uint32_t num_subspaces() const { return num_subspaces_; }
    uint32_t verify_dim() const { return verify_dim_; }
    uint32_t rabitq_record_bytes() const { return rabitq_record_bytes_; }

    uint32_t orig_id(uint32_t local_row) const {
        if ((size_t)local_row >= orig_ids_.size()) throw std::out_of_range("L0Block local row out of range");
        return orig_ids_[(size_t)local_row];
    }

    uint32_t pid(uint32_t local_row) const {
        if ((size_t)local_row >= pids_.size()) throw std::out_of_range("L0Block local row out of range");
        return pids_[(size_t)local_row];
    }

    const float* verify_vector(uint32_t local_row) const {
        if ((size_t)local_row >= orig_ids_.size()) throw std::out_of_range("L0Block local row out of range");
        return verify_rotated_.data() + (size_t)local_row * verify_dim_;
    }

    const uint8_t* rabitq_record(uint32_t local_row) const {
        if ((size_t)local_row >= orig_ids_.size()) throw std::out_of_range("L0Block local row out of range");
        return rabitq_records_.data() + (size_t)local_row * rabitq_record_bytes_;
    }

    size_t cell_size(uint32_t subspace, uint32_t cell) const {
        validate_cell(subspace, cell);
        const std::vector<uint32_t>& offsets = posting_offsets_[(size_t)subspace];
        return (size_t)(offsets[(size_t)cell + 1] - offsets[(size_t)cell]);
    }

    const uint32_t* cell_data(uint32_t subspace, uint32_t cell) const {
        validate_cell(subspace, cell);
        const std::vector<uint32_t>& offsets = posting_offsets_[(size_t)subspace];
        return posting_ids_[(size_t)subspace].data() + offsets[(size_t)cell];
    }

    bool has_row_cells() const { return !row_cells_.empty(); }

    uint16_t row_cell(uint32_t local_row, uint32_t subspace) const {
        if ((size_t)local_row >= orig_ids_.size()) throw std::out_of_range("L0Block local row out of range");
        if ((size_t)subspace >= (size_t)num_subspaces_) throw std::out_of_range("L0Block subspace out of range");
        size_t offset = (size_t)local_row * (size_t)num_subspaces_ + (size_t)subspace;
        if (offset >= row_cells_.size()) throw std::out_of_range("L0Block row cell missing");
        return row_cells_[offset];
    }

    void release_posting_index() const {
        std::vector<std::vector<uint32_t>>().swap(posting_offsets_);
        std::vector<std::vector<uint32_t>>().swap(posting_ids_);
    }

    void release_row_cells() const {
        std::vector<uint16_t>().swap(row_cells_);
    }

private:
    static void validate_row_cell_width(const std::vector<size_t>& cells_per_subspace) {
        for (size_t K : cells_per_subspace) {
            if (K > (size_t)std::numeric_limits<uint16_t>::max()) {
                throw std::invalid_argument("L0Block cell count exceeds uint16 row-cell range");
            }
        }
    }

    void validate_cell(uint32_t subspace, uint32_t cell) const {
        if ((size_t)subspace >= posting_offsets_.size()) {
            throw std::out_of_range("L0Block subspace out of range");
        }
        const std::vector<uint32_t>& offsets = posting_offsets_[(size_t)subspace];
        if ((size_t)cell + 1 >= offsets.size()) {
            throw std::out_of_range("L0Block cell out of range");
        }
    }

    bool compute_dense_pid_order() const {
        if (pid_count_ != orig_ids_.size()) return false;
        if (pid_to_row_.empty()) return true;
        for (uint32_t row = 0; row < pid_count_; ++row) {
            if (pid_to_row_[(size_t)row] != row ||
                pids_[(size_t)row] != pid_begin_ + row) {
                return false;
            }
        }
        return true;
    }

    uint64_t commit_lsn_ = 0;
    uint32_t pid_begin_ = 0;
    uint32_t pid_count_ = 0;
    uint32_t num_subspaces_ = 0;
    uint32_t verify_dim_ = 0;
    uint32_t rabitq_record_bytes_ = 0;
    std::vector<uint32_t> orig_ids_;
    std::vector<uint32_t> pids_;
    std::vector<uint32_t> pid_to_row_;
    std::vector<float> verify_rotated_;
    std::vector<uint8_t> rabitq_records_;
    mutable std::vector<uint16_t> row_cells_;
    mutable std::vector<std::vector<uint32_t>> posting_offsets_;
    mutable std::vector<std::vector<uint32_t>> posting_ids_;
    bool dense_pid_order_ = true;
};

struct CommittedUpdateDelta {
    UpdateWalHeader header;
    std::vector<std::shared_ptr<const L0Block>> committed_blocks;
    std::vector<uint32_t> set_live_pids;
    std::vector<uint32_t> committed_dead_pids;
    uint64_t commit_lsn = 0;
    uint64_t wal_offset_after_commit = 0;
    uint32_t max_pid = 0;
    uint32_t live_count = 0;
    bool committed_insert = false;
    bool committed_delete = false;
    bool committed_insert_reused_pid = false;
    bool committed_base_live_change = false;
    uint32_t committed_base_live_deletes = 0;

    bool empty() const {
        return !committed_insert && !committed_delete && committed_blocks.empty() &&
               committed_dead_pids.empty() && set_live_pids.empty();
    }
};

namespace impl {

template <typename T>
inline void write_delta_pod(std::ostream& out, const T& value, const std::string& path) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) throw std::runtime_error("failed writing delta artifact: " + path);
}

template <typename T>
inline void read_delta_pod(std::istream& in, T& value, const std::string& path) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("short delta artifact: " + path);
}

inline void ensure_delta_dir(const std::string& path) {
    if (::mkdir(path.c_str(), 0755) == 0) return;
    if (errno == EEXIST) return;
    throw std::runtime_error("cannot create delta segment directory: " + path);
}

inline std::string delta_path_join(const std::string& dir, const std::string& name) {
    return dir + "/" + name;
}

inline const rapidjson::Value& require_delta_manifest_member(const rapidjson::Value& object,
                                                             const char* name,
                                                             const std::string& path) {
    if (!object.IsObject()) {
        throw std::runtime_error("delta segment manifest expected object while reading " +
                                 std::string(name) + ": " + path);
    }
    auto it = object.FindMember(name);
    if (it == object.MemberEnd()) {
        throw std::runtime_error("delta segment manifest missing field '" +
                                 std::string(name) + "': " + path);
    }
    return it->value;
}

inline std::string require_delta_manifest_string(const rapidjson::Value& object,
                                                 const char* name,
                                                 const std::string& path) {
    const rapidjson::Value& value = require_delta_manifest_member(object, name, path);
    if (!value.IsString()) {
        throw std::runtime_error("delta segment manifest field '" + std::string(name) +
                                 "' must be a string: " + path);
    }
    return value.GetString();
}

inline uint64_t require_delta_manifest_uint64(const rapidjson::Value& object,
                                              const char* name,
                                              const std::string& path) {
    const rapidjson::Value& value = require_delta_manifest_member(object, name, path);
    if (!value.IsUint64()) {
        throw std::runtime_error("delta segment manifest field '" + std::string(name) +
                                 "' must be an unsigned integer: " + path);
    }
    return value.GetUint64();
}

inline uint32_t require_delta_manifest_uint32(const rapidjson::Value& object,
                                              const char* name,
                                              const std::string& path) {
    uint64_t value = require_delta_manifest_uint64(object, name, path);
    if (value > (uint64_t)std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("delta segment manifest field '" + std::string(name) +
                                 "' exceeds uint32 range: " + path);
    }
    return (uint32_t)value;
}

struct DeltaSegmentManifestMetadata {
    std::string segment_id;
    uint32_t pid_begin = 0;
    uint32_t pid_count = 0;
    uint32_t row_count = 0;
    uint32_t num_subspaces = 0;
    uint32_t verify_dim = 0;
    uint32_t rabitq_record_bytes = 0;
    bool has_row_cells = false;
};

inline void write_delta_segment_manifest(const std::string& dir,
                                         const std::string& segment_id,
                                         uint32_t pid_begin,
                                         uint32_t pid_count,
                                         uint32_t row_count,
                                         uint32_t num_subspaces,
                                         uint32_t verify_dim,
                                         uint32_t rabitq_record_bytes,
                                         bool has_row_cells) {
    std::string path = delta_path_join(dir, "segment_manifest.json");
    std::ofstream out(path, std::ios::trunc);
    if (!out.is_open()) throw std::runtime_error("cannot create delta segment manifest: " + path);

    rapidjson::OStreamWrapper stream(out);
    rapidjson::Writer<rapidjson::OStreamWrapper> writer(stream);
    writer.StartObject();
    writer.Key("format");
    writer.String("score_delta_segment_v1");
    writer.Key("segment_id");
    writer.String(segment_id.c_str());

    writer.Key("pid");
    writer.StartObject();
    writer.Key("pid_begin");
    writer.Uint(pid_begin);
    writer.Key("pid_count");
    writer.Uint(pid_count);
    writer.Key("row_count");
    writer.Uint(row_count);
    writer.EndObject();

    writer.Key("layout");
    writer.StartObject();
    writer.Key("num_subspaces");
    writer.Uint(num_subspaces);
    writer.Key("verify_dim");
    writer.Uint(verify_dim);
    writer.Key("rabitq_record_bytes");
    writer.Uint(rabitq_record_bytes);
    writer.EndObject();

    writer.Key("artifacts");
    writer.StartObject();
    writer.Key("orig_ids");
    writer.String("delta_orig_ids.bin");
    writer.Key("pid_to_row");
    writer.String("delta_pid_to_row.bin");
    if (has_row_cells) {
        writer.Key("row_cells");
        writer.String("delta_row_cells.bin");
    }
    writer.Key("verify_vectors");
    writer.String("delta_verify_vectors.fbin");
    writer.Key("rabitq_records");
    writer.String("delta_rabitq_bin_data.bin");
    writer.Key("posting_prefix");
    writer.String("delta_posting_subspace_");
    writer.EndObject();
    writer.EndObject();

    out.flush();
    if (!out) throw std::runtime_error("failed writing delta segment manifest: " + path);
}

inline DeltaSegmentManifestMetadata load_delta_segment_manifest(const std::string& dir) {
    std::string path = delta_path_join(dir, "segment_manifest.json");
    std::ifstream in(path);
    if (!in.is_open()) throw std::runtime_error("cannot open delta segment manifest: " + path);

    rapidjson::IStreamWrapper stream(in);
    rapidjson::Document doc;
    doc.ParseStream(stream);
    if (doc.HasParseError()) {
        throw std::runtime_error(
            "invalid delta segment manifest JSON at byte " +
            std::to_string((uint64_t)doc.GetErrorOffset()) + ": " +
            rapidjson::GetParseError_En(doc.GetParseError()) + ": " + path);
    }
    if (!doc.IsObject()) {
        throw std::runtime_error("delta segment manifest root must be an object: " + path);
    }

    std::string format = require_delta_manifest_string(doc, "format", path);
    if (format != "score_delta_segment_v1") {
        throw std::runtime_error("unsupported delta segment manifest format '" + format + "': " + path);
    }

    DeltaSegmentManifestMetadata manifest;
    manifest.segment_id = require_delta_manifest_string(doc, "segment_id", path);
    const rapidjson::Value& pid = require_delta_manifest_member(doc, "pid", path);
    manifest.pid_begin = require_delta_manifest_uint32(pid, "pid_begin", path);
    manifest.pid_count = require_delta_manifest_uint32(pid, "pid_count", path);
    manifest.row_count = require_delta_manifest_uint32(pid, "row_count", path);
    if (manifest.pid_count == 0) {
        throw std::runtime_error("delta segment manifest pid_count must be positive: " + path);
    }
    if (manifest.row_count > manifest.pid_count) {
        throw std::runtime_error("delta segment manifest row_count exceeds pid_count: " + path);
    }
    const rapidjson::Value& layout = require_delta_manifest_member(doc, "layout", path);
    manifest.num_subspaces = require_delta_manifest_uint32(layout, "num_subspaces", path);
    manifest.verify_dim = require_delta_manifest_uint32(layout, "verify_dim", path);
    manifest.rabitq_record_bytes =
        require_delta_manifest_uint32(layout, "rabitq_record_bytes", path);
    if (manifest.num_subspaces == 0 || manifest.verify_dim == 0 ||
        manifest.rabitq_record_bytes == 0) {
        throw std::runtime_error("delta segment manifest layout fields must be positive: " + path);
    }

    const rapidjson::Value& artifacts = require_delta_manifest_member(doc, "artifacts", path);
    if (require_delta_manifest_string(artifacts, "orig_ids", path) != "delta_orig_ids.bin" ||
        require_delta_manifest_string(artifacts, "pid_to_row", path) != "delta_pid_to_row.bin" ||
        require_delta_manifest_string(artifacts, "verify_vectors", path) != "delta_verify_vectors.fbin" ||
        require_delta_manifest_string(artifacts, "rabitq_records", path) != "delta_rabitq_bin_data.bin" ||
        require_delta_manifest_string(artifacts, "posting_prefix", path) != "delta_posting_subspace_") {
        throw std::runtime_error("delta segment manifest artifact names are unsupported: " + path);
    }
    auto row_cells = artifacts.FindMember("row_cells");
    if (row_cells != artifacts.MemberEnd()) {
        if (!row_cells->value.IsString() ||
            std::string(row_cells->value.GetString()) != "delta_row_cells.bin") {
            throw std::runtime_error("delta segment manifest row_cells artifact name is unsupported: " + path);
        }
        manifest.has_row_cells = true;
    }
    return manifest;
}

} // namespace impl

class SealedDeltaSegment {
public:
    static std::shared_ptr<const SealedDeltaSegment> from_inserts(
        std::string segment_id,
        std::vector<UpdateWalInsert> inserts,
        const std::vector<size_t>& cells_per_subspace,
        uint32_t verify_dim,
        uint32_t rabitq_record_bytes) {
        if (inserts.empty()) {
            throw std::invalid_argument("SealedDeltaSegment requires at least one insert");
        }
        std::sort(inserts.begin(), inserts.end(),
                  [](const UpdateWalInsert& a, const UpdateWalInsert& b) {
                      return a.global_pid < b.global_pid;
                  });
        for (size_t i = 1; i < inserts.size(); ++i) {
            if (inserts[i - 1].global_pid == inserts[i].global_pid) {
                throw std::invalid_argument("SealedDeltaSegment duplicate global PID");
            }
        }
        if (cells_per_subspace.empty()) {
            throw std::invalid_argument("SealedDeltaSegment requires at least one subspace");
        }
        validate_row_cell_width(cells_per_subspace);
        if (verify_dim == 0 || rabitq_record_bytes == 0) {
            throw std::invalid_argument("SealedDeltaSegment payload dimensions must be positive");
        }
        if (inserts.size() > (size_t)std::numeric_limits<uint32_t>::max()) {
            throw std::invalid_argument("SealedDeltaSegment insert count exceeds uint32 range");
        }

        auto segment = std::make_shared<SealedDeltaSegment>();
        segment->segment_id_ = std::move(segment_id);
        segment->pid_begin_ = inserts.front().global_pid;
        uint64_t pid_range = (uint64_t)inserts.back().global_pid -
                             (uint64_t)segment->pid_begin_ + 1;
        if (pid_range > (uint64_t)std::numeric_limits<uint32_t>::max()) {
            throw std::invalid_argument("SealedDeltaSegment PID range exceeds uint32 range");
        }
        segment->pid_count_ = (uint32_t)pid_range;
        segment->live_count_ = (uint32_t)inserts.size();
        segment->verify_dim_ = verify_dim;
        segment->rabitq_record_bytes_ = rabitq_record_bytes;
        segment->num_subspaces_ = (uint32_t)cells_per_subspace.size();

        segment->orig_ids_.resize(inserts.size());
        segment->pids_.resize(inserts.size());
        segment->pid_to_row_.assign((size_t)segment->pid_count_,
                                    std::numeric_limits<uint32_t>::max());
        bool keep_row_cells =
            should_keep_row_cells_artifact(inserts.size(), cells_per_subspace.size());
        if (keep_row_cells) {
            segment->row_cells_.resize(inserts.size() * cells_per_subspace.size());
        }
        segment->verify_rotated_.resize(inserts.size() * (size_t)verify_dim);
        segment->rabitq_records_.resize(inserts.size() * (size_t)rabitq_record_bytes);

        for (size_t row = 0; row < inserts.size(); ++row) {
            const UpdateWalInsert& insert = inserts[row];
            if (insert.verify_rotated.size() != (size_t)verify_dim) {
                throw std::invalid_argument("SealedDeltaSegment insert verify dimension mismatch");
            }
            if (insert.rabitq_record.size() != (size_t)rabitq_record_bytes) {
                throw std::invalid_argument("SealedDeltaSegment insert RaBitQ record size mismatch");
            }
            if (insert.subspace_cells.size() != cells_per_subspace.size()) {
                throw std::invalid_argument("SealedDeltaSegment insert subspace cell count mismatch");
            }
            for (size_t m = 0; m < cells_per_subspace.size(); ++m) {
                if (insert.subspace_cells[m] >= cells_per_subspace[m]) {
                    throw std::invalid_argument("SealedDeltaSegment insert subspace cell is out of range");
                }
            }
            segment->orig_ids_[row] = insert.orig_id;
            segment->pids_[row] = insert.global_pid;
            segment->pid_to_row_[(size_t)(insert.global_pid - segment->pid_begin_)] = (uint32_t)row;
            if (keep_row_cells) {
                std::copy(insert.subspace_cells.begin(), insert.subspace_cells.end(),
                          segment->row_cells_.begin() +
                              (std::ptrdiff_t)(row * cells_per_subspace.size()));
            }
            std::copy(insert.verify_rotated.begin(), insert.verify_rotated.end(),
                      segment->verify_rotated_.begin() +
                          (std::ptrdiff_t)(row * (size_t)verify_dim));
            std::copy(insert.rabitq_record.begin(), insert.rabitq_record.end(),
                      segment->rabitq_records_.begin() +
                          (std::ptrdiff_t)(row * (size_t)rabitq_record_bytes));
        }
        segment->dense_pid_order_ = segment->compute_dense_pid_order();

        build_posting_csr(
            cells_per_subspace, inserts.size(),
            [&](size_t row, size_t m) { return inserts[row].subspace_cells[m]; },
            [&](size_t row) { return inserts[row].global_pid; },
            segment->posting_offsets_, segment->posting_ids_);
        return segment;
    }

    void save(const std::string& dir) const {
        impl::ensure_delta_dir(dir);
        {
            std::string path = impl::delta_path_join(dir, "delta_orig_ids.bin");
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            if (!out.is_open()) throw std::runtime_error("cannot create delta orig IDs: " + path);
            uint32_t n = (uint32_t)orig_ids_.size();
            impl::write_delta_pod(out, n, path);
            if (n != 0) out.write(reinterpret_cast<const char*>(orig_ids_.data()), orig_ids_.size() * sizeof(uint32_t));
            if (!out) throw std::runtime_error("failed writing delta orig IDs: " + path);
        }
        {
            std::string path = impl::delta_path_join(dir, "delta_pid_to_row.bin");
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            if (!out.is_open()) throw std::runtime_error("cannot create delta PID map: " + path);
            uint32_t n = pid_count_;
            impl::write_delta_pod(out, n, path);
            if (n != 0) {
                out.write(reinterpret_cast<const char*>(pid_to_row_.data()),
                          pid_to_row_.size() * sizeof(uint32_t));
            }
            if (!out) throw std::runtime_error("failed writing delta PID map: " + path);
        }
        {
            std::string path = impl::delta_path_join(dir, "delta_row_cells.bin");
            if (should_save_row_cells_artifact()) {
                std::ofstream out(path, std::ios::binary | std::ios::trunc);
                if (!out.is_open()) throw std::runtime_error("cannot create delta row cells: " + path);
                uint32_t n = (uint32_t)orig_ids_.size();
                uint32_t m = num_subspaces_;
                impl::write_delta_pod(out, n, path);
                impl::write_delta_pod(out, m, path);
                if (!row_cells_.empty()) {
                    out.write(reinterpret_cast<const char*>(row_cells_.data()),
                              row_cells_.size() * sizeof(uint16_t));
                }
                if (!out) throw std::runtime_error("failed writing delta row cells: " + path);
            } else {
                ::unlink(path.c_str());
            }
        }
        {
            std::string path = impl::delta_path_join(dir, "delta_verify_vectors.fbin");
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            if (!out.is_open()) throw std::runtime_error("cannot create delta verify vectors: " + path);
            int n = (int)orig_ids_.size();
            int dim = (int)verify_dim_;
            int reserved = 0;
            impl::write_delta_pod(out, n, path);
            impl::write_delta_pod(out, dim, path);
            impl::write_delta_pod(out, reserved, path);
            if (!verify_rotated_.empty()) {
                out.write(reinterpret_cast<const char*>(verify_rotated_.data()),
                          verify_rotated_.size() * sizeof(float));
            }
            if (!out) throw std::runtime_error("failed writing delta verify vectors: " + path);
        }
        {
            std::string path = impl::delta_path_join(dir, "delta_rabitq_bin_data.bin");
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            if (!out.is_open()) throw std::runtime_error("cannot create delta RaBitQ records: " + path);
            int n = (int)orig_ids_.size();
            int record_bytes = (int)rabitq_record_bytes_;
            impl::write_delta_pod(out, n, path);
            impl::write_delta_pod(out, record_bytes, path);
            if (!rabitq_records_.empty()) {
                out.write(reinterpret_cast<const char*>(rabitq_records_.data()), rabitq_records_.size());
            }
            if (!out) throw std::runtime_error("failed writing delta RaBitQ records: " + path);
        }
        for (uint32_t m = 0; m < num_subspaces_; ++m) {
            std::string path = impl::delta_path_join(
                dir, "delta_posting_subspace_" + std::to_string(m) + ".bin");
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            if (!out.is_open()) throw std::runtime_error("cannot create delta postings: " + path);
            int K = (int)posting_offsets_[(size_t)m].size() - 1;
            int total = (int)posting_ids_[(size_t)m].size();
            impl::write_delta_pod(out, K, path);
            impl::write_delta_pod(out, total, path);
            out.write(reinterpret_cast<const char*>(posting_offsets_[(size_t)m].data()),
                      posting_offsets_[(size_t)m].size() * sizeof(uint32_t));
            out.write(reinterpret_cast<const char*>(posting_ids_[(size_t)m].data()),
                      posting_ids_[(size_t)m].size() * sizeof(uint32_t));
            if (!out) throw std::runtime_error("failed writing delta postings: " + path);
        }
        impl::write_delta_segment_manifest(
            dir,
            segment_id_,
            pid_begin_,
            pid_count_,
            row_count(),
            num_subspaces_,
            verify_dim_,
            rabitq_record_bytes_,
            should_save_row_cells_artifact());
    }

    static std::shared_ptr<const SealedDeltaSegment> load(
        const std::string& segment_id,
        const std::string& dir,
        const std::vector<size_t>& cells_per_subspace,
        uint32_t verify_dim,
        uint32_t rabitq_record_bytes,
        uint32_t expected_pid_begin,
        uint32_t expected_pid_count,
        uint32_t expected_live_count,
        bool load_row_cells = false) {
        impl::DeltaSegmentManifestMetadata manifest =
            impl::load_delta_segment_manifest(dir);
        if (manifest.segment_id != segment_id ||
            manifest.pid_begin != expected_pid_begin ||
            manifest.pid_count != expected_pid_count ||
            manifest.num_subspaces != (uint32_t)cells_per_subspace.size() ||
            manifest.verify_dim != verify_dim ||
            manifest.rabitq_record_bytes != rabitq_record_bytes) {
            throw std::runtime_error("delta segment manifest metadata mismatch: " + dir);
        }
        validate_row_cell_width(cells_per_subspace);

        auto segment = std::make_shared<SealedDeltaSegment>();
        segment->segment_id_ = segment_id;
        segment->pid_begin_ = expected_pid_begin;
        segment->pid_count_ = expected_pid_count;
        segment->verify_dim_ = verify_dim;
        segment->rabitq_record_bytes_ = rabitq_record_bytes;
        segment->num_subspaces_ = (uint32_t)cells_per_subspace.size();

        uint32_t n_orig = 0;
        {
            std::string path = impl::delta_path_join(dir, "delta_orig_ids.bin");
            std::ifstream in(path, std::ios::binary);
            if (!in.is_open()) throw std::runtime_error("cannot open delta orig IDs: " + path);
            impl::read_delta_pod(in, n_orig, path);
            if (n_orig != manifest.row_count) {
                throw std::runtime_error("delta orig ID row count differs from segment manifest: " + path);
            }
            segment->orig_ids_.resize(n_orig);
            if (n_orig != 0) {
                in.read(reinterpret_cast<char*>(segment->orig_ids_.data()),
                        segment->orig_ids_.size() * sizeof(uint32_t));
                if (!in) throw std::runtime_error("short delta orig IDs: " + path);
            }
        }
        {
            std::string path = impl::delta_path_join(dir, "delta_pid_to_row.bin");
            std::ifstream in(path, std::ios::binary);
            if (!in.is_open()) throw std::runtime_error("cannot open delta PID map: " + path);
            uint32_t range_count = 0;
            impl::read_delta_pod(in, range_count, path);
            if (range_count != expected_pid_count) {
                throw std::runtime_error("delta PID map range length mismatch: " + path);
            }
            if (range_count < n_orig) {
                throw std::runtime_error("delta PID map range is smaller than row count: " + path);
            }
            segment->pid_to_row_.resize(range_count);
            if (range_count != 0) {
                in.read(reinterpret_cast<char*>(segment->pid_to_row_.data()),
                        segment->pid_to_row_.size() * sizeof(uint32_t));
                if (!in) throw std::runtime_error("short delta PID map: " + path);
            }
            segment->pids_.assign(n_orig, 0);
            std::vector<uint8_t> seen_rows((size_t)n_orig, 0);
            for (uint32_t offset = 0; offset < range_count; ++offset) {
                uint32_t local_row = segment->pid_to_row_[(size_t)offset];
                if (local_row == std::numeric_limits<uint32_t>::max()) continue;
                if (local_row >= n_orig) {
                    throw std::runtime_error("delta PID map local row is out of range: " + path);
                }
                if (seen_rows[(size_t)local_row] != 0) {
                    throw std::runtime_error("delta PID map contains duplicate local row: " + path);
                }
                seen_rows[(size_t)local_row] = 1;
                segment->pids_[(size_t)local_row] = expected_pid_begin + offset;
            }
            for (uint8_t seen : seen_rows) {
                if (seen == 0) {
                    throw std::runtime_error("delta PID map is missing a local row: " + path);
                }
            }
            segment->dense_pid_order_ = segment->compute_dense_pid_order();
        }
        segment->live_count_ = expected_live_count;
        if (expected_live_count > n_orig) {
            throw std::runtime_error("delta manifest live_count exceeds stored row count: " + dir);
        }

        if (load_row_cells && manifest.has_row_cells) {
            std::string path = impl::delta_path_join(dir, "delta_row_cells.bin");
            std::ifstream in(path, std::ios::binary);
            if (!in.is_open()) throw std::runtime_error("cannot open delta row cells: " + path);
            uint32_t n = 0;
            uint32_t num_subspaces = 0;
            impl::read_delta_pod(in, n, path);
            impl::read_delta_pod(in, num_subspaces, path);
            if (n != n_orig || num_subspaces != segment->num_subspaces_) {
                throw std::runtime_error("delta row cells header mismatch: " + path);
            }
            size_t cell_count = (size_t)n * (size_t)num_subspaces;
            segment->row_cells_.resize(cell_count);
            if (cell_count != 0) {
                in.read(reinterpret_cast<char*>(segment->row_cells_.data()),
                        segment->row_cells_.size() * sizeof(uint16_t));
                if (!in) throw std::runtime_error("short delta row cells: " + path);
            }
            segment->validate_row_cells(cells_per_subspace, path);
        }

        {
            std::string path = impl::delta_path_join(dir, "delta_verify_vectors.fbin");
            std::ifstream in(path, std::ios::binary);
            if (!in.is_open()) throw std::runtime_error("cannot open delta verify vectors: " + path);
            int n = 0, dim = 0, reserved = 0;
            impl::read_delta_pod(in, n, path);
            impl::read_delta_pod(in, dim, path);
            impl::read_delta_pod(in, reserved, path);
            if (n < 0 || (uint32_t)n != n_orig || dim != (int)verify_dim || reserved != 0) {
                throw std::runtime_error("delta verify vector header mismatch: " + path);
            }
            segment->verify_rotated_.resize((size_t)n_orig * (size_t)verify_dim);
            if (!segment->verify_rotated_.empty()) {
                in.read(reinterpret_cast<char*>(segment->verify_rotated_.data()),
                        segment->verify_rotated_.size() * sizeof(float));
                if (!in) throw std::runtime_error("short delta verify vectors: " + path);
            }
        }
        {
            std::string path = impl::delta_path_join(dir, "delta_rabitq_bin_data.bin");
            std::ifstream in(path, std::ios::binary);
            if (!in.is_open()) throw std::runtime_error("cannot open delta RaBitQ records: " + path);
            int n = 0, record_bytes = 0;
            impl::read_delta_pod(in, n, path);
            impl::read_delta_pod(in, record_bytes, path);
            if (n < 0 || (uint32_t)n != n_orig || record_bytes != (int)rabitq_record_bytes) {
                throw std::runtime_error("delta RaBitQ record header mismatch: " + path);
            }
            segment->rabitq_records_.resize((size_t)n_orig * (size_t)rabitq_record_bytes);
            if (!segment->rabitq_records_.empty()) {
                in.read(reinterpret_cast<char*>(segment->rabitq_records_.data()),
                        segment->rabitq_records_.size());
                if (!in) throw std::runtime_error("short delta RaBitQ records: " + path);
            }
        }
        segment->posting_offsets_.resize(cells_per_subspace.size());
        segment->posting_ids_.resize(cells_per_subspace.size());
        for (size_t m = 0; m < cells_per_subspace.size(); ++m) {
            std::string path = impl::delta_path_join(
                dir, "delta_posting_subspace_" + std::to_string(m) + ".bin");
            std::ifstream in(path, std::ios::binary);
            if (!in.is_open()) throw std::runtime_error("cannot open delta postings: " + path);
            int K = 0, total = 0;
            impl::read_delta_pod(in, K, path);
            impl::read_delta_pod(in, total, path);
            if (K < 0 || total < 0 || (size_t)K != cells_per_subspace[m]) {
                throw std::runtime_error("delta posting header mismatch: " + path);
            }
            if ((uint32_t)total != n_orig) {
                throw std::runtime_error("delta posting total must match row count: " + path);
            }
            segment->posting_offsets_[m].resize((size_t)K + 1);
            in.read(reinterpret_cast<char*>(segment->posting_offsets_[m].data()),
                    segment->posting_offsets_[m].size() * sizeof(uint32_t));
            if (!in) throw std::runtime_error("short delta posting offsets: " + path);
            if (segment->posting_offsets_[m].empty() || segment->posting_offsets_[m][0] != 0 ||
                segment->posting_offsets_[m].back() != (uint32_t)total) {
                throw std::runtime_error("invalid delta posting offsets: " + path);
            }
            for (size_t cell = 0; cell < (size_t)K; ++cell) {
                if (segment->posting_offsets_[m][cell] > segment->posting_offsets_[m][cell + 1]) {
                    throw std::runtime_error("non-monotonic delta posting offsets: " + path);
                }
            }
            segment->posting_ids_[m].resize((size_t)total);
            if (total != 0) {
                in.read(reinterpret_cast<char*>(segment->posting_ids_[m].data()),
                        segment->posting_ids_[m].size() * sizeof(uint32_t));
                if (!in) throw std::runtime_error("short delta posting IDs: " + path);
            }
            std::vector<uint8_t> seen_rows((size_t)n_orig, 0);
            bool check_row_cells =
                segment->row_cells_.size() ==
                (size_t)n_orig * (size_t)segment->num_subspaces_;
            for (uint32_t cell = 0; (size_t)cell + 1 < segment->posting_offsets_[m].size(); ++cell) {
                uint32_t begin = segment->posting_offsets_[m][(size_t)cell];
                uint32_t end = segment->posting_offsets_[m][(size_t)cell + 1];
                for (uint32_t i = begin; i < end; ++i) {
                    uint32_t pid = segment->posting_ids_[m][(size_t)i];
                    if (!segment->contains_pid(pid)) {
                        throw std::runtime_error("delta posting references PID outside segment: " + path);
                    }
                    size_t row = (size_t)segment->local_row_for_pid(pid);
                    if (seen_rows[row] != 0) {
                        throw std::runtime_error("delta posting references a row more than once: " + path);
                    }
                    if (check_row_cells &&
                        segment->row_cells_[row * (size_t)segment->num_subspaces_ + m] != cell) {
                        throw std::runtime_error("delta row cells do not match postings: " + path);
                    }
                    seen_rows[row] = 1;
                }
            }
            for (uint8_t seen : seen_rows) {
                if (seen == 0) {
                    throw std::runtime_error("delta posting is missing a row: " + path);
                }
            }
        }
        return segment;
    }

    const std::string& segment_id() const { return segment_id_; }
    uint32_t pid_begin() const { return pid_begin_; }
    uint32_t pid_count() const { return pid_count_; }
    uint32_t row_count() const { return (uint32_t)orig_ids_.size(); }
    uint32_t live_count() const { return live_count_; }
    uint32_t num_subspaces() const { return num_subspaces_; }
    uint32_t verify_dim() const { return verify_dim_; }
    uint32_t rabitq_record_bytes() const { return rabitq_record_bytes_; }
    bool dense_pid_order() const { return dense_pid_order_; }

    void build_live_cell_counts(const LiveTable& live) const {
        has_live_cell_counts_ = false;
        live_posting_counts_.clear();
        if (live_count_ == row_count()) return;
        live_posting_counts_.resize(posting_offsets_.size());
        for (size_t m = 0; m < posting_offsets_.size(); ++m) {
            const std::vector<uint32_t>& offsets = posting_offsets_[m];
            if (offsets.empty()) {
                live_posting_counts_[m].clear();
                continue;
            }
            size_t K = offsets.size() - 1;
            std::vector<uint32_t>& counts = live_posting_counts_[m];
            counts.assign(K, 0);
            uint32_t subspace_live = 0;
            for (size_t cell = 0; cell < K; ++cell) {
                const uint32_t begin = offsets[cell];
                const uint32_t end = offsets[cell + 1];
                uint32_t cell_live = 0;
                for (uint32_t i = begin; i < end; ++i) {
                    uint32_t pid = posting_ids_[m][(size_t)i];
                    if (live.is_live(pid)) ++cell_live;
                }
                counts[cell] = cell_live;
                subspace_live += cell_live;
            }
            if (subspace_live != live_count_) {
                throw std::runtime_error("sealed delta live cell counts do not match manifest live_count");
            }
        }
        has_live_cell_counts_ = true;
    }

    bool contains_pid(uint32_t pid) const {
        if (pid < pid_begin_) return false;
        uint64_t offset = (uint64_t)pid - (uint64_t)pid_begin_;
        if (offset >= (uint64_t)pid_to_row_.size()) return false;
        return pid_to_row_[(size_t)offset] != std::numeric_limits<uint32_t>::max();
    }

    uint32_t local_row_for_pid(uint32_t pid) const {
        if (pid < pid_begin_) throw std::out_of_range("sealed delta PID out of range");
        uint64_t offset = (uint64_t)pid - (uint64_t)pid_begin_;
        if (offset >= (uint64_t)pid_to_row_.size()) {
            throw std::out_of_range("sealed delta PID out of range");
        }
        uint32_t row = pid_to_row_[(size_t)offset];
        if (row == std::numeric_limits<uint32_t>::max()) {
            throw std::out_of_range("sealed delta PID out of range");
        }
        return row;
    }

    uint32_t orig_id(uint32_t local_row) const {
        if ((size_t)local_row >= orig_ids_.size()) throw std::out_of_range("sealed delta local row out of range");
        return orig_ids_[(size_t)local_row];
    }

    uint32_t pid(uint32_t local_row) const {
        if ((size_t)local_row >= pids_.size()) throw std::out_of_range("sealed delta local row out of range");
        return pids_[(size_t)local_row];
    }

    const float* verify_vector(uint32_t local_row) const {
        if ((size_t)local_row >= orig_ids_.size()) throw std::out_of_range("sealed delta local row out of range");
        return verify_rotated_.data() + (size_t)local_row * verify_dim_;
    }

    const uint8_t* rabitq_record(uint32_t local_row) const {
        if ((size_t)local_row >= orig_ids_.size()) throw std::out_of_range("sealed delta local row out of range");
        return rabitq_records_.data() + (size_t)local_row * rabitq_record_bytes_;
    }

    size_t cell_size(uint32_t subspace, uint32_t cell) const {
        validate_cell(subspace, cell);
        const std::vector<uint32_t>& offsets = posting_offsets_[(size_t)subspace];
        return (size_t)(offsets[(size_t)cell + 1] - offsets[(size_t)cell]);
    }

    bool has_live_cell_counts() const { return has_live_cell_counts_; }

    size_t live_cell_size(uint32_t subspace, uint32_t cell) const {
        validate_cell(subspace, cell);
        if (!has_live_cell_counts_) return cell_size(subspace, cell);
        return live_posting_counts_[(size_t)subspace][(size_t)cell];
    }

    const uint32_t* cell_data(uint32_t subspace, uint32_t cell) const {
        validate_cell(subspace, cell);
        const std::vector<uint32_t>& offsets = posting_offsets_[(size_t)subspace];
        return posting_ids_[(size_t)subspace].data() + offsets[(size_t)cell];
    }

private:
    static bool should_keep_row_cells_artifact(size_t rows, size_t num_subspaces) {
        constexpr size_t kMinRowCellsArtifactBytes = 1024u * 1024u;
        if (rows == 0 || num_subspaces == 0) return false;
        if (rows > std::numeric_limits<size_t>::max() / num_subspaces) return true;
        size_t cells = rows * num_subspaces;
        if (cells > std::numeric_limits<size_t>::max() / sizeof(uint16_t)) return true;
        return cells * sizeof(uint16_t) >= kMinRowCellsArtifactBytes;
    }

    bool should_save_row_cells_artifact() const {
        return should_keep_row_cells_artifact(orig_ids_.size(), (size_t)num_subspaces_) &&
               row_cells_.size() == orig_ids_.size() * (size_t)num_subspaces_;
    }

    void validate_row_cells(const std::vector<size_t>& cells_per_subspace,
                            const std::string& path) const {
        if (cells_per_subspace.size() != (size_t)num_subspaces_) {
            throw std::runtime_error("delta row cells subspace count mismatch: " + path);
        }
        if (row_cells_.size() != orig_ids_.size() * (size_t)num_subspaces_) {
            throw std::runtime_error("delta row cells payload size mismatch: " + path);
        }
        for (uint32_t row = 0; (size_t)row < orig_ids_.size(); ++row) {
            for (uint32_t m = 0; m < num_subspaces_; ++m) {
                uint16_t cell = row_cells_[(size_t)row * (size_t)num_subspaces_ + (size_t)m];
                if ((size_t)cell >= cells_per_subspace[(size_t)m]) {
                    throw std::runtime_error("delta row cells contain out-of-range cell: " + path);
                }
            }
        }
    }

    static void validate_row_cell_width(const std::vector<size_t>& cells_per_subspace) {
        for (size_t K : cells_per_subspace) {
            if (K > (size_t)std::numeric_limits<uint16_t>::max()) {
                throw std::invalid_argument("SealedDeltaSegment cell count exceeds uint16 row-cell range");
            }
        }
    }

    void validate_cell(uint32_t subspace, uint32_t cell) const {
        if ((size_t)subspace >= posting_offsets_.size()) {
            throw std::out_of_range("sealed delta subspace out of range");
        }
        const std::vector<uint32_t>& offsets = posting_offsets_[(size_t)subspace];
        if ((size_t)cell + 1 >= offsets.size()) {
            throw std::out_of_range("sealed delta cell out of range");
        }
    }

    bool compute_dense_pid_order() const {
        if (pid_count_ != (uint32_t)pids_.size() ||
            pid_to_row_.size() != pids_.size()) {
            return false;
        }
        for (uint32_t row = 0; row < pid_count_; ++row) {
            if (pids_[(size_t)row] != pid_begin_ + row ||
                pid_to_row_[(size_t)row] != row) {
                return false;
            }
        }
        return true;
    }

    std::string segment_id_;
    uint32_t pid_begin_ = 0;
    uint32_t pid_count_ = 0;
    uint32_t live_count_ = 0;
    uint32_t num_subspaces_ = 0;
    uint32_t verify_dim_ = 0;
    uint32_t rabitq_record_bytes_ = 0;
    std::vector<uint32_t> orig_ids_;
    std::vector<uint32_t> pids_;
    std::vector<uint32_t> pid_to_row_;
    std::vector<uint16_t> row_cells_;
    std::vector<float> verify_rotated_;
    std::vector<uint8_t> rabitq_records_;
    std::vector<std::vector<uint32_t>> posting_offsets_;
    std::vector<std::vector<uint32_t>> posting_ids_;
    mutable std::vector<std::vector<uint32_t>> live_posting_counts_;
    mutable bool has_live_cell_counts_ = false;
    bool dense_pid_order_ = false;
};

} // namespace score
