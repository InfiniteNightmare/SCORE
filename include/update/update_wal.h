#pragma once

#include <array>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__SSE4_2__) && \
    (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
#include <nmmintrin.h>
#define SCORE_UPDATE_WAL_HAS_SSE42_CRC32C 1
#else
#define SCORE_UPDATE_WAL_HAS_SSE42_CRC32C 0
#endif

namespace score {

struct UpdateWalHeader {
    uint64_t generation = 0;
    uint32_t base_n = 0;
    uint32_t num_subspaces = 0;
    uint32_t verify_dim = 0;
    uint32_t rabitq_record_bytes = 0;
};

struct UpdateWalInsert {
    uint32_t orig_id = 0;
    uint32_t global_pid = 0;
    std::vector<float> verify_rotated;
    std::vector<uint8_t> rabitq_record;
    std::vector<uint16_t> subspace_cells;
};

struct UpdateWalDelete {
    uint32_t orig_id = 0;
    uint32_t old_global_pid = 0;
};

struct UpdateWalRecord {
    enum class Type : uint8_t {
        Insert = 1,
        Delete = 2
    };

    Type type = Type::Insert;
    uint64_t lsn = 0;
    UpdateWalInsert insert;
    UpdateWalDelete tombstone;
};

struct UpdateWalCommit {
    uint64_t lsn = 0;
    uint32_t batch_record_count = 0;
    uint32_t max_pid = 0;
    uint32_t live_count = 0;
};

struct UpdateWalBatch {
    UpdateWalCommit commit;
    std::vector<UpdateWalRecord> records;
};

struct UpdateWalReplayResult {
    UpdateWalHeader header;
    std::vector<UpdateWalBatch> committed_batches;
    uint64_t last_committed_lsn = 0;
    uint32_t max_pid = 0;
    uint32_t live_count = 0;
    bool ignored_tail = false;
};

namespace impl {

struct ByteSpan {
    const void* data = nullptr;
    size_t size = 0;
};

inline const std::array<uint32_t, 256>& crc32c_table() {
    static const std::array<uint32_t, 256> table = [] {
        std::array<uint32_t, 256> generated{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int bit = 0; bit < 8; ++bit) {
                c = (c & 1) ? (0x82f63b78u ^ (c >> 1)) : (c >> 1);
            }
            generated[i] = c;
        }
        return generated;
    }();
    return table;
}

inline uint32_t crc32c_update_sw(uint32_t crc, const void* data, size_t len) {
    const auto& table = crc32c_table();
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < len; ++i) {
        crc = table[(crc ^ bytes[i]) & 0xffu] ^ (crc >> 8);
    }
    return crc;
}

#if SCORE_UPDATE_WAL_HAS_SSE42_CRC32C
inline uint32_t crc32c_update_hw(uint32_t crc, const void* data, size_t len) {
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
#if defined(__x86_64__) || defined(_M_X64)
    uint64_t crc64 = crc;
    while (len >= sizeof(uint64_t)) {
        uint64_t chunk;
        std::memcpy(&chunk, bytes, sizeof(chunk));
        crc64 = _mm_crc32_u64(crc64, chunk);
        bytes += sizeof(chunk);
        len -= sizeof(chunk);
    }
    crc = (uint32_t)crc64;
#endif
    while (len >= sizeof(uint32_t)) {
        uint32_t chunk;
        std::memcpy(&chunk, bytes, sizeof(chunk));
        crc = _mm_crc32_u32(crc, chunk);
        bytes += sizeof(chunk);
        len -= sizeof(chunk);
    }
    if (len >= sizeof(uint16_t)) {
        uint16_t chunk;
        std::memcpy(&chunk, bytes, sizeof(chunk));
        crc = _mm_crc32_u16(crc, chunk);
        bytes += sizeof(chunk);
        len -= sizeof(chunk);
    }
    if (len != 0) {
        crc = _mm_crc32_u8(crc, *bytes);
    }
    return crc;
}
#endif

inline uint32_t crc32c_update(uint32_t crc, const void* data, size_t len) {
#if SCORE_UPDATE_WAL_HAS_SSE42_CRC32C
    return crc32c_update_hw(crc, data, len);
#else
    return crc32c_update_sw(crc, data, len);
#endif
}

inline uint32_t crc32c_finish(uint32_t crc) {
    return crc ^ 0xffffffffu;
}

template <typename T>
inline void append_pod(std::vector<uint8_t>& out, const T& value) {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&value);
    out.insert(out.end(), bytes, bytes + sizeof(T));
}

template <typename T>
inline void write_pod(std::ostream& out, const T& value, const std::string& path) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) throw std::runtime_error("failed writing update WAL: " + path);
}

template <typename T>
inline bool read_pod_or_tail(std::istream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (in) return true;
    if (in.eof() || in.gcount() > 0) return false;
    return false;
}

template <typename T>
inline T read_payload_pod(const std::vector<uint8_t>& payload,
                          size_t& offset,
                          const std::string& context) {
    if (payload.size() - offset < sizeof(T)) {
        throw std::runtime_error("short update WAL payload while reading " + context);
    }
    T value;
    std::memcpy(&value, payload.data() + offset, sizeof(T));
    offset += sizeof(T);
    return value;
}

inline void append_crc_input(std::vector<uint8_t>& out,
                             uint64_t lsn,
                             uint8_t type,
                             uint32_t payload_bytes,
                             const std::vector<uint8_t>& payload) {
    append_pod(out, lsn);
    append_pod(out, type);
    append_pod(out, payload_bytes);
    out.insert(out.end(), payload.begin(), payload.end());
}

inline uint32_t frame_crc(uint64_t lsn,
                          uint8_t type,
                          uint32_t payload_bytes,
                          const std::vector<uint8_t>& payload) {
    uint32_t crc = crc32c_update(0xffffffffu, &lsn, sizeof(lsn));
    crc = crc32c_update(crc, &type, sizeof(type));
    crc = crc32c_update(crc, &payload_bytes, sizeof(payload_bytes));
    if (!payload.empty()) {
        crc = crc32c_update(crc, payload.data(), payload.size());
    }
    return crc32c_finish(crc);
}

inline uint32_t frame_crc_segments(uint64_t lsn,
                                   uint8_t type,
                                   uint32_t payload_bytes,
                                   const ByteSpan* spans,
                                   size_t span_count) {
    uint32_t crc = crc32c_update(0xffffffffu, &lsn, sizeof(lsn));
    crc = crc32c_update(crc, &type, sizeof(type));
    crc = crc32c_update(crc, &payload_bytes, sizeof(payload_bytes));
    for (size_t i = 0; i < span_count; ++i) {
        if (spans[i].size == 0) continue;
        crc = crc32c_update(crc, spans[i].data, spans[i].size);
    }
    return crc32c_finish(crc);
}

inline void write_bytes(std::ostream& out,
                        const void* data,
                        size_t bytes,
                        const std::string& path) {
    if (bytes == 0) return;
    out.write(reinterpret_cast<const char*>(data), (std::streamsize)bytes);
    if (!out) throw std::runtime_error("failed writing update WAL payload: " + path);
}

inline void write_frame_header(std::ostream& out,
                               uint64_t lsn,
                               uint8_t type,
                               uint32_t payload_bytes,
                               const std::string& path) {
    std::array<uint8_t, sizeof(uint64_t) + sizeof(uint8_t) + sizeof(uint32_t)> header{};
    size_t offset = 0;
    std::memcpy(header.data() + offset, &lsn, sizeof(lsn));
    offset += sizeof(lsn);
    std::memcpy(header.data() + offset, &type, sizeof(type));
    offset += sizeof(type);
    std::memcpy(header.data() + offset, &payload_bytes, sizeof(payload_bytes));
    write_bytes(out, header.data(), header.size(), path);
}

inline bool expected_frame_payload_bytes(const UpdateWalHeader& header,
                                         uint8_t type,
                                         uint32_t& expected_out) {
    if (type == (uint8_t)UpdateWalRecord::Type::Insert) {
        uint64_t expected = sizeof(uint32_t) * 2ull +
            (uint64_t)header.verify_dim * sizeof(float) +
            (uint64_t)header.rabitq_record_bytes +
            (uint64_t)header.num_subspaces * sizeof(uint16_t);
        if (expected > (uint64_t)std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("update WAL insert payload size exceeds uint32 range");
        }
        expected_out = (uint32_t)expected;
        return true;
    }
    if (type == (uint8_t)UpdateWalRecord::Type::Delete) {
        expected_out = sizeof(uint32_t) * 2u;
        return true;
    }
    if (type == 3) {
        expected_out = sizeof(uint32_t) * 3u;
        return true;
    }
    return false;
}

inline void write_wal_header(std::ostream& out,
                             const UpdateWalHeader& header,
                             const std::string& path) {
    constexpr uint32_t magic = 0x314c5753u; // "SWL1" in little-endian native order.
    constexpr uint32_t version = 2;
    write_pod(out, magic, path);
    write_pod(out, version, path);
    write_pod(out, header.generation, path);
    write_pod(out, header.base_n, path);
    write_pod(out, header.num_subspaces, path);
    write_pod(out, header.verify_dim, path);
    write_pod(out, header.rabitq_record_bytes, path);
}

inline UpdateWalHeader read_wal_header(std::istream& in, const std::string& path) {
    constexpr uint32_t expected_magic = 0x314c5753u;
    constexpr uint32_t expected_version = 2;
    uint32_t magic = 0;
    uint32_t version = 0;
    if (!read_pod_or_tail(in, magic) || !read_pod_or_tail(in, version)) {
        throw std::runtime_error("short update WAL header: " + path);
    }
    if (magic != expected_magic || version != expected_version) {
        throw std::runtime_error("invalid update WAL header: " + path);
    }
    UpdateWalHeader header;
    if (!read_pod_or_tail(in, header.generation) ||
        !read_pod_or_tail(in, header.base_n) ||
        !read_pod_or_tail(in, header.num_subspaces) ||
        !read_pod_or_tail(in, header.verify_dim) ||
        !read_pod_or_tail(in, header.rabitq_record_bytes)) {
        throw std::runtime_error("short update WAL header: " + path);
    }
    if (header.num_subspaces == 0 || header.verify_dim == 0 || header.rabitq_record_bytes == 0) {
        throw std::runtime_error("invalid update WAL dimensions: " + path);
    }
    return header;
}

inline bool wal_headers_equal(const UpdateWalHeader& a, const UpdateWalHeader& b) {
    return a.generation == b.generation &&
           a.base_n == b.base_n &&
           a.num_subspaces == b.num_subspaces &&
           a.verify_dim == b.verify_dim &&
           a.rabitq_record_bytes == b.rabitq_record_bytes;
}

inline void encode_insert_payload_into(const UpdateWalHeader& header,
                                       const UpdateWalInsert& record,
                                       std::vector<uint8_t>& payload) {
    if (record.verify_rotated.size() != (size_t)header.verify_dim) {
        throw std::invalid_argument("insert WAL verify vector dimension mismatch");
    }
    if (record.rabitq_record.size() != (size_t)header.rabitq_record_bytes) {
        throw std::invalid_argument("insert WAL RaBitQ record size mismatch");
    }
    if (record.subspace_cells.size() != (size_t)header.num_subspaces) {
        throw std::invalid_argument("insert WAL subspace cell count mismatch");
    }
    payload.clear();
    payload.reserve(sizeof(uint32_t) * 2 +
                    record.verify_rotated.size() * sizeof(float) +
                    record.rabitq_record.size() +
                    record.subspace_cells.size() * sizeof(uint16_t));
    append_pod(payload, record.orig_id);
    append_pod(payload, record.global_pid);
    const uint8_t* verify_bytes = reinterpret_cast<const uint8_t*>(record.verify_rotated.data());
    payload.insert(payload.end(), verify_bytes,
                   verify_bytes + record.verify_rotated.size() * sizeof(float));
    payload.insert(payload.end(), record.rabitq_record.begin(), record.rabitq_record.end());
    const uint8_t* cell_bytes = reinterpret_cast<const uint8_t*>(record.subspace_cells.data());
    payload.insert(payload.end(), cell_bytes,
                   cell_bytes + record.subspace_cells.size() * sizeof(uint16_t));
}

inline void encode_delete_payload_into(const UpdateWalDelete& record,
                                       std::vector<uint8_t>& payload) {
    payload.clear();
    payload.reserve(sizeof(uint32_t) * 2);
    append_pod(payload, record.orig_id);
    append_pod(payload, record.old_global_pid);
}

inline void encode_commit_payload_into(const UpdateWalCommit& commit,
                                       std::vector<uint8_t>& payload) {
    payload.clear();
    payload.reserve(sizeof(uint32_t) * 3);
    append_pod(payload, commit.batch_record_count);
    append_pod(payload, commit.max_pid);
    append_pod(payload, commit.live_count);
}

inline UpdateWalInsert decode_insert_payload(const UpdateWalHeader& header,
                                             const std::vector<uint8_t>& payload) {
    const size_t expected = sizeof(uint32_t) * 2 +
        (size_t)header.verify_dim * sizeof(float) +
        (size_t)header.rabitq_record_bytes +
        (size_t)header.num_subspaces * sizeof(uint16_t);
    if (payload.size() != expected) {
        throw std::runtime_error("insert WAL payload has invalid size");
    }
    size_t offset = 0;
    UpdateWalInsert record;
    record.orig_id = read_payload_pod<uint32_t>(payload, offset, "insert orig_id");
    record.global_pid = read_payload_pod<uint32_t>(payload, offset, "insert global_pid");
    record.verify_rotated.resize(header.verify_dim);
    std::memcpy(record.verify_rotated.data(), payload.data() + offset,
                (size_t)header.verify_dim * sizeof(float));
    offset += (size_t)header.verify_dim * sizeof(float);
    record.rabitq_record.assign(payload.begin() + (std::ptrdiff_t)offset,
                                payload.begin() + (std::ptrdiff_t)offset +
                                    (std::ptrdiff_t)header.rabitq_record_bytes);
    offset += header.rabitq_record_bytes;
    record.subspace_cells.resize(header.num_subspaces);
    std::memcpy(record.subspace_cells.data(), payload.data() + offset,
                (size_t)header.num_subspaces * sizeof(uint16_t));
    return record;
}

inline UpdateWalDelete decode_delete_payload(const std::vector<uint8_t>& payload) {
    if (payload.size() != sizeof(uint32_t) * 2) {
        throw std::runtime_error("delete WAL payload has invalid size");
    }
    size_t offset = 0;
    UpdateWalDelete record;
    record.orig_id = read_payload_pod<uint32_t>(payload, offset, "delete orig_id");
    record.old_global_pid = read_payload_pod<uint32_t>(payload, offset, "delete old_global_pid");
    return record;
}

inline UpdateWalCommit decode_commit_payload(uint64_t lsn,
                                             const std::vector<uint8_t>& payload) {
    if (payload.size() != sizeof(uint32_t) * 3) {
        throw std::runtime_error("commit WAL payload has invalid size");
    }
    size_t offset = 0;
    UpdateWalCommit commit;
    commit.lsn = lsn;
    commit.batch_record_count = read_payload_pod<uint32_t>(payload, offset, "commit record count");
    commit.max_pid = read_payload_pod<uint32_t>(payload, offset, "commit max_pid");
    commit.live_count = read_payload_pod<uint32_t>(payload, offset, "commit live_count");
    return commit;
}

inline void validate_append_only_commit_pids(const std::vector<UpdateWalRecord>& pending,
                                             uint32_t previous_max_pid,
                                             const UpdateWalCommit& commit,
                                             const std::string& path) {
    if (commit.max_pid < previous_max_pid) {
        throw std::runtime_error("update WAL commit max_pid shrinks previous max_pid: " + path);
    }
    uint32_t appended_pid_count = commit.max_pid - previous_max_pid;
    std::vector<uint32_t> insert_pids;
    insert_pids.reserve(pending.size());
    for (const UpdateWalRecord& record : pending) {
        if (record.type != UpdateWalRecord::Type::Insert) continue;
        uint32_t pid = record.insert.global_pid;
        if (pid < previous_max_pid) {
            throw std::runtime_error("update WAL insert reuses an existing PID: " + path);
        }
        if (pid >= commit.max_pid) {
            throw std::runtime_error("update WAL insert PID is outside commit max_pid: " + path);
        }
        insert_pids.push_back(pid);
    }
    if (insert_pids.size() != (size_t)appended_pid_count) {
        throw std::runtime_error("update WAL commit max_pid does not match append insert count: " + path);
    }
    std::sort(insert_pids.begin(), insert_pids.end());
    for (size_t i = 0; i < insert_pids.size(); ++i) {
        uint32_t expected = previous_max_pid + (uint32_t)i;
        if (insert_pids[i] != expected) {
            throw std::runtime_error("update WAL insert PIDs are not a contiguous append range: " + path);
        }
    }
}

} // namespace impl

class UpdateWalWriter {
public:
    UpdateWalWriter(const std::string& path, const UpdateWalHeader& header, bool append = false)
        : path_(path), header_(header) {
        if (append) {
            std::ifstream existing(path_, std::ios::binary);
            if (!existing.is_open()) {
                throw std::runtime_error("cannot open existing update WAL for append: " + path_);
            }
            UpdateWalHeader existing_header = impl::read_wal_header(existing, path_);
            if (!impl::wal_headers_equal(existing_header, header_)) {
                throw std::runtime_error("existing update WAL header does not match append header: " + path_);
            }
        }

        std::ios::openmode mode = std::ios::binary | (append ? std::ios::app : std::ios::trunc);
        out_.open(path_, mode);
        if (!out_.is_open()) throw std::runtime_error("cannot create update WAL: " + path);
        if (!append) impl::write_wal_header(out_, header_, path_);
    }

    void append_insert(uint64_t lsn, const UpdateWalInsert& record) {
        if (record.verify_rotated.size() != (size_t)header_.verify_dim) {
            throw std::invalid_argument("insert WAL verify vector dimension mismatch");
        }
        if (record.rabitq_record.size() != (size_t)header_.rabitq_record_bytes) {
            throw std::invalid_argument("insert WAL RaBitQ record size mismatch");
        }
        if (record.subspace_cells.size() != (size_t)header_.num_subspaces) {
            throw std::invalid_argument("insert WAL subspace cell count mismatch");
        }
        uint32_t payload_bytes = 0;
        if (!impl::expected_frame_payload_bytes(
                header_, (uint8_t)UpdateWalRecord::Type::Insert, payload_bytes)) {
            throw std::runtime_error("cannot compute insert WAL payload size");
        }
        const std::array<uint32_t, 2> ids = {{record.orig_id, record.global_pid}};
        const std::array<impl::ByteSpan, 4> spans = {{
            {ids.data(), ids.size() * sizeof(uint32_t)},
            {record.verify_rotated.data(), record.verify_rotated.size() * sizeof(float)},
            {record.rabitq_record.data(), record.rabitq_record.size()},
            {record.subspace_cells.data(), record.subspace_cells.size() * sizeof(uint16_t)}
        }};
        write_frame_segments(
            lsn,
            (uint8_t)UpdateWalRecord::Type::Insert,
            payload_bytes,
            spans.data(),
            spans.size());
    }

    void append_insert_parts(uint64_t lsn,
                             uint32_t orig_id,
                             uint32_t global_pid,
                             const float* verify_rotated,
                             const uint8_t* rabitq_record,
                             const uint16_t* subspace_cells) {
        if (verify_rotated == nullptr || rabitq_record == nullptr ||
            subspace_cells == nullptr) {
            throw std::invalid_argument("insert WAL raw payload pointer is null");
        }
        uint32_t payload_bytes = 0;
        if (!impl::expected_frame_payload_bytes(
                header_, (uint8_t)UpdateWalRecord::Type::Insert, payload_bytes)) {
            throw std::runtime_error("cannot compute insert WAL payload size");
        }
        const std::array<uint32_t, 2> ids = {{orig_id, global_pid}};
        const std::array<impl::ByteSpan, 4> spans = {{
            {ids.data(), ids.size() * sizeof(uint32_t)},
            {verify_rotated, (size_t)header_.verify_dim * sizeof(float)},
            {rabitq_record, (size_t)header_.rabitq_record_bytes},
            {subspace_cells, (size_t)header_.num_subspaces * sizeof(uint16_t)}
        }};
        write_frame_segments(
            lsn,
            (uint8_t)UpdateWalRecord::Type::Insert,
            payload_bytes,
            spans.data(),
            spans.size());
    }

    // Append a batch of insert frames built from contiguous SoA encoded rows into
    // one staged buffer, then a single ofstream::write. Produces the exact same
    // per-frame byte layout as append_insert_parts ([lsn][type][payload_bytes]
    // [payload][crc]) — only the buffering granularity and the CRC call count
    // change (one frame_crc_segments over the contiguous payload per frame,
    // instead of 4 spans; one write per batch instead of ~6 per frame). PIDs are
    // pid_begin + row (contiguous), matching the encoded-batch contract.
    void append_insert_rows_batch(uint64_t start_lsn,
                                  size_t row_count,
                                  const uint32_t* orig_ids,
                                  uint32_t pid_begin,
                                  const float* verify_rows,
                                  const uint8_t* rabitq_rows,
                                  const uint16_t* cell_rows) {
        if (row_count == 0) return;
        if (orig_ids == nullptr || verify_rows == nullptr ||
            rabitq_rows == nullptr || cell_rows == nullptr) {
            throw std::invalid_argument("update WAL batch insert row pointer is null");
        }
        uint32_t payload_bytes = 0;
        if (!impl::expected_frame_payload_bytes(
                header_, (uint8_t)UpdateWalRecord::Type::Insert, payload_bytes)) {
            throw std::runtime_error("cannot compute insert WAL payload size");
        }
        constexpr size_t kHeaderBytes =
            sizeof(uint64_t) + sizeof(uint8_t) + sizeof(uint32_t);
        constexpr size_t kCrcBytes = sizeof(uint32_t);
        const size_t verify_bytes = (size_t)header_.verify_dim * sizeof(float);
        const size_t rabitq_bytes = (size_t)header_.rabitq_record_bytes;
        const size_t cells_bytes = (size_t)header_.num_subspaces * sizeof(uint16_t);
        const size_t frame_bytes = kHeaderBytes + (size_t)payload_bytes + kCrcBytes;
        const size_t needed = row_count * frame_bytes;
        // Reuse the staging buffer; resize only grows (zeros once), so steady-state
        // batches overwrite in place without per-call reallocation or zero-fill.
        if (batch_staging_.size() < needed) batch_staging_.resize(needed);
        const uint8_t type = (uint8_t)UpdateWalRecord::Type::Insert;
        uint8_t* base = batch_staging_.data();
        for (size_t row = 0; row < row_count; ++row) {
            uint64_t lsn = start_lsn + row;
            uint32_t pid = pid_begin + (uint32_t)row;
            uint8_t* p = base + row * frame_bytes;
            std::memcpy(p, &lsn, sizeof(lsn)); p += sizeof(lsn);
            std::memcpy(p, &type, sizeof(type)); p += sizeof(type);
            std::memcpy(p, &payload_bytes, sizeof(payload_bytes)); p += sizeof(payload_bytes);
            uint8_t* payload_pos = p;
            std::memcpy(p, &orig_ids[row], sizeof(uint32_t)); p += sizeof(uint32_t);
            std::memcpy(p, &pid, sizeof(uint32_t)); p += sizeof(uint32_t);
            std::memcpy(p, verify_rows + row * (size_t)header_.verify_dim, verify_bytes); p += verify_bytes;
            std::memcpy(p, rabitq_rows + row * (size_t)header_.rabitq_record_bytes, rabitq_bytes); p += rabitq_bytes;
            std::memcpy(p, cell_rows + row * (size_t)header_.num_subspaces, cells_bytes); p += cells_bytes;
            impl::ByteSpan payload_span{payload_pos, payload_bytes};
            uint32_t crc = impl::frame_crc_segments(lsn, type, payload_bytes, &payload_span, 1);
            std::memcpy(p, &crc, sizeof(crc));
        }
        out_.write(reinterpret_cast<const char*>(base), (std::streamsize)needed);
        if (!out_) throw std::runtime_error("failed writing update WAL batch: " + path_);
    }

    void append_delete(uint64_t lsn, const UpdateWalDelete& record) {
        impl::encode_delete_payload_into(record, payload_scratch_);
        write_frame(lsn, (uint8_t)UpdateWalRecord::Type::Delete, payload_scratch_);
    }

    void append_commit(const UpdateWalCommit& commit) {
        impl::encode_commit_payload_into(commit, payload_scratch_);
        write_frame(commit.lsn, 3, payload_scratch_);
    }

    void flush() {
        out_.flush();
        if (!out_) throw std::runtime_error("failed flushing update WAL: " + path_);
    }

    uint64_t current_offset() {
        std::streampos pos = out_.tellp();
        if (pos < std::streampos(0)) {
            throw std::runtime_error("cannot determine update WAL write offset: " + path_);
        }
        return (uint64_t)pos;
    }

private:
    void write_frame(uint64_t lsn, uint8_t type, const std::vector<uint8_t>& payload) {
        if (payload.size() > (size_t)std::numeric_limits<uint32_t>::max()) {
            throw std::invalid_argument("update WAL payload too large");
        }
        uint32_t payload_bytes = (uint32_t)payload.size();
        uint32_t crc = impl::frame_crc(lsn, type, payload_bytes, payload);
        impl::write_frame_header(out_, lsn, type, payload_bytes, path_);
        out_.write(reinterpret_cast<const char*>(payload.data()), payload.size());
        if (!out_) throw std::runtime_error("failed writing update WAL payload: " + path_);
        impl::write_pod(out_, crc, path_);
    }

    void write_frame_segments(uint64_t lsn,
                              uint8_t type,
                              uint32_t payload_bytes,
                              const impl::ByteSpan* spans,
                              size_t span_count) {
        size_t summed_payload_bytes = 0;
        for (size_t i = 0; i < span_count; ++i) {
            if (spans[i].size > (size_t)std::numeric_limits<uint32_t>::max() ||
                summed_payload_bytes >
                    (size_t)std::numeric_limits<uint32_t>::max() - spans[i].size) {
                throw std::invalid_argument("update WAL payload too large");
            }
            summed_payload_bytes += spans[i].size;
        }
        if (summed_payload_bytes != (size_t)payload_bytes) {
            throw std::invalid_argument("update WAL segmented payload size mismatch");
        }
        uint32_t crc =
            impl::frame_crc_segments(lsn, type, payload_bytes, spans, span_count);
        impl::write_frame_header(out_, lsn, type, payload_bytes, path_);
        for (size_t i = 0; i < span_count; ++i) {
            impl::write_bytes(out_, spans[i].data, spans[i].size, path_);
        }
        impl::write_pod(out_, crc, path_);
    }

    std::string path_;
    UpdateWalHeader header_;
    std::ofstream out_;
    std::vector<uint8_t> payload_scratch_;
    std::vector<uint8_t> batch_staging_;
};

inline UpdateWalReplayResult replay_update_wal(const std::string& path,
                                               uint32_t initial_max_pid = 0,
                                               bool validate_append_pids = false,
                                               bool metadata_only = false) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("cannot open update WAL: " + path);

    UpdateWalReplayResult result;
    result.header = impl::read_wal_header(in, path);
    result.max_pid = result.header.base_n;
    result.live_count = result.header.base_n;
    uint32_t validation_max_pid =
        initial_max_pid == 0 ? result.header.base_n : initial_max_pid;

    std::vector<UpdateWalRecord> pending;
    uint64_t last_frame_lsn = 0;
    std::vector<uint8_t> payload;
    while (true) {
        uint64_t lsn = 0;
        uint8_t type = 0;
        uint32_t payload_bytes = 0;
        if (!impl::read_pod_or_tail(in, lsn)) break;
        if (lsn <= last_frame_lsn) {
            throw std::runtime_error("update WAL frame LSN is not strictly increasing: " + path);
        }
        if (!impl::read_pod_or_tail(in, type) ||
            !impl::read_pod_or_tail(in, payload_bytes)) {
            result.ignored_tail = true;
            break;
        }
        uint32_t expected_payload_bytes = 0;
        if (!impl::expected_frame_payload_bytes(result.header, type, expected_payload_bytes) ||
            payload_bytes != expected_payload_bytes) {
            result.ignored_tail = true;
            break;
        }
        payload.resize(payload_bytes);
        if (payload_bytes != 0) {
            in.read(reinterpret_cast<char*>(payload.data()), payload.size());
            if (!in) {
                result.ignored_tail = true;
                break;
            }
        }
        uint32_t stored_crc = 0;
        if (!impl::read_pod_or_tail(in, stored_crc)) {
            result.ignored_tail = true;
            break;
        }
        uint32_t expected_crc = impl::frame_crc(lsn, type, payload_bytes, payload);
        if (stored_crc != expected_crc) {
            result.ignored_tail = true;
            break;
        }
        last_frame_lsn = lsn;

        if (type == (uint8_t)UpdateWalRecord::Type::Insert) {
            UpdateWalRecord record;
            record.type = UpdateWalRecord::Type::Insert;
            record.lsn = lsn;
            if (metadata_only) {
                size_t offset = 0;
                record.insert.orig_id =
                    impl::read_payload_pod<uint32_t>(payload, offset, "insert orig_id");
                record.insert.global_pid =
                    impl::read_payload_pod<uint32_t>(payload, offset, "insert global_pid");
            } else {
                record.insert = impl::decode_insert_payload(result.header, payload);
            }
            pending.push_back(std::move(record));
        } else if (type == (uint8_t)UpdateWalRecord::Type::Delete) {
            UpdateWalRecord record;
            record.type = UpdateWalRecord::Type::Delete;
            record.lsn = lsn;
            record.tombstone = impl::decode_delete_payload(payload);
            pending.push_back(std::move(record));
        } else if (type == 3) {
            UpdateWalCommit commit = impl::decode_commit_payload(lsn, payload);
            if (commit.batch_record_count != pending.size()) {
                throw std::runtime_error("update WAL commit record count mismatch: " + path);
            }
            if (commit.live_count > commit.max_pid) {
                throw std::runtime_error("update WAL commit live_count exceeds max_pid: " + path);
            }
            if (validate_append_pids) {
                impl::validate_append_only_commit_pids(
                    pending, validation_max_pid, commit, path);
                validation_max_pid = commit.max_pid;
            }
            UpdateWalBatch batch;
            batch.commit = commit;
            batch.records = std::move(pending);
            pending.clear();
            result.last_committed_lsn = commit.lsn;
            result.max_pid = commit.max_pid;
            result.live_count = commit.live_count;
            result.committed_batches.push_back(std::move(batch));
        } else {
            throw std::runtime_error("unknown update WAL frame type: " + path);
        }
    }

    return result;
}

inline UpdateWalReplayResult replay_update_wal_metadata(const std::string& path,
                                                        uint32_t initial_max_pid = 0,
                                                        bool validate_append_pids = false) {
    return replay_update_wal(path, initial_max_pid, validate_append_pids,
                             /*metadata_only=*/true);
}

} // namespace score
