#pragma once
#include "core/common.h"
#include "core/rotation.h"
#include "query/sc_score.h"
#include "core/posting_lists.h"
#include "update/update_delta.h"
#include "update/update_io.h"
#include "update/update_manifest.h"
#include "update/update_state.h"
#include <array>
#include <vector>
#include <algorithm>
#include <cstring>
#include <iterator>
#include <limits>
#include <cerrno>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <dirent.h>

#include "rabitqlib/index/query.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "core/fast_estdist.h"
#include "query/base_segment.h"
#include "query/query_engine_types.h"
#include "query/unsealed_delta_view.h"
#include "query/delta_posting_views.h"
#include "query/reader_view.h"

namespace score {

namespace impl {
inline std::string compaction_publish_marker_path(const std::string& index_path) {
    return index_path + "/compaction.publish";
}

// On load: if a stale compaction.publish marker is present (a prior compaction
// was interrupted by a crash), automatically ROLL BACK to the pre-compaction
// state using the hardlinked backup. If no backup exists (old-format compaction
// or the backup was lost), throws an actionable error.
inline void recover_compaction_if_marker_present(const std::string& index_path) {
    std::string marker = compaction_publish_marker_path(index_path);
    if (::access(marker.c_str(), F_OK) != 0) {
        if (errno != ENOENT) {
            throw std::runtime_error("cannot probe compaction publish marker: " + marker +
                                     " (" + std::strerror(errno) + ")");
        }
        return;  // no marker — index is clean
    }

    // Marker present: read the backup_dir path from the marker content.
    std::string backup_dir;
    {
        std::ifstream in(marker);
        if (!in.is_open() || !std::getline(in, backup_dir) || backup_dir.empty()) {
            throw std::runtime_error(
                "index " + index_path + " has a stale compaction.publish marker"
                " but it is unreadable or empty. Manual recovery: re-run"
                " score_compact_index or restore from backup, then remove " + marker);
        }
    }

    // Check backup dir exists.
    struct stat st;
    if (::stat(backup_dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
        throw std::runtime_error(
            "index " + index_path + " has a stale compaction.publish marker pointing"
            " to backup dir " + backup_dir + " which does not exist. Manual recovery:"
            " re-run score_compact_index or restore from a known-good backup,"
            " then remove " + marker);
    }

    // Restore: rename each file in backup_dir -> index_path (overwrites the
    // mixed/new base with the pre-compaction old base + old manifest).
    // Collect names first — readdir during directory modification is UB.
    DIR* dir = ::opendir(backup_dir.c_str());
    if (!dir) {
        throw std::runtime_error("cannot open backup dir for recovery: " + backup_dir);
    }
    std::vector<std::string> backup_names;
    struct dirent* entry;
    while ((entry = ::readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name != "." && name != "..") backup_names.push_back(name);
    }
    ::closedir(dir);
    for (const std::string& name : backup_names) {
        std::string src = backup_dir + "/" + name;
        std::string dst = index_path + "/" + name;
        if (::rename(src.c_str(), dst.c_str()) != 0) {
            throw std::runtime_error("cannot restore backup file during recovery: " +
                                     src + " -> " + dst +
                                     " (" + std::strerror(errno) + ")");
        }
    }
    // Best-effort cleanup of the (now hopefully empty) backup dir. If rmdir
    // fails (e.g., a stray file), it's harmless — next compaction or manual
    // cleanup handles it. The critical step (marker removal) follows.
    ::rmdir(backup_dir.c_str());
    // Recovery complete — remove the marker.
    ::unlink(marker.c_str());
}

class OptionalUpdateReadLock {
public:
    explicit OptionalUpdateReadLock(const std::string& index_path)
        : lock_(index_path) {}

    OptionalUpdateReadLock(const OptionalUpdateReadLock&) = delete;
    OptionalUpdateReadLock& operator=(const OptionalUpdateReadLock&) = delete;

private:
    update_io::UpdatePublishReadLock lock_;
};

inline uint32_t next_query_stamp(std::vector<uint32_t>& stamps, uint32_t& current_stamp) {
    ++current_stamp;
    if (current_stamp == 0) {
        std::fill(stamps.begin(), stamps.end(), 0u);
        current_stamp = 1;
    }
    return current_stamp;
}

inline size_t count_live_posting_ids(const uint32_t* ids,
                                     size_t count,
                                     const LiveTable* live) {
    if (live == nullptr) return count;
    size_t live_count = 0;
    for (size_t i = 0; i < count; ++i) {
        if (live->is_live(ids[i])) ++live_count;
    }
    return live_count;
}

struct L0WalBlockReplayResult {
    UpdateWalHeader header;
    std::vector<std::shared_ptr<const L0Block>> committed_blocks;
    std::shared_ptr<const LiveTable> live;
    uint64_t last_committed_lsn = 0;
    uint64_t last_committed_wal_offset = 0;
    uint32_t max_pid = 0;
    uint32_t live_count = 0;
    bool ignored_tail = false;
    bool committed_insert = false;
    bool committed_delete = false;
    bool committed_insert_reused_pid = false;
    bool committed_base_live_change = false;
    uint32_t committed_base_live_deletes = 0;
    std::vector<uint32_t> committed_dead_pids;
};

inline L0WalBlockReplayResult replay_update_wal_l0_blocks(
    const std::string& path,
    const std::vector<size_t>& cells_per_subspace,
    std::shared_ptr<const LiveTable> initial_live,
    size_t first_block_reserve_rows,
    uint64_t start_after_lsn = 0,
    uint64_t start_offset = 0) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("cannot open update WAL: " + path);
    if (!initial_live) throw std::invalid_argument("L0 WAL replay requires initial live table");

    L0WalBlockReplayResult result;
    result.header = read_wal_header(in, path);
    std::streamoff header_offset = in.tellg();
    if (header_offset < 0) {
        throw std::runtime_error("cannot determine update WAL header offset: " + path);
    }
    if (start_offset != 0) {
        if (start_offset < (uint64_t)header_offset) {
            throw std::runtime_error("incremental L0 WAL replay offset is inside the header: " + path);
        }
        in.seekg((std::streamoff)start_offset, std::ios::beg);
        if (!in) throw std::runtime_error("cannot seek update WAL for incremental replay: " + path);
        result.last_committed_lsn = start_after_lsn;
        result.last_committed_wal_offset = start_offset;
    } else {
        result.last_committed_wal_offset = (uint64_t)header_offset;
    }
    result.max_pid = initial_live->max_pid();
    result.live_count = initial_live->live_count();
    result.live = initial_live;
    std::shared_ptr<LiveTable> replay_live;
    if (result.header.num_subspaces != cells_per_subspace.size()) {
        throw std::runtime_error("update WAL subspace count does not match loaded index");
    }

    std::unique_ptr<L0Block::Builder> pending_block;
    std::vector<uint32_t> set_live;
    std::vector<uint32_t> set_dead;
    std::vector<uint8_t> payload;
    uint64_t last_frame_lsn = start_after_lsn;
    bool reserve_used = false;

    while (true) {
        uint64_t lsn = 0;
        uint8_t type = 0;
        uint32_t payload_bytes = 0;
        if (!read_pod_or_tail(in, lsn)) break;
        if (lsn <= last_frame_lsn) {
            throw std::runtime_error("update WAL frame LSN is not strictly increasing: " + path);
        }
        if (!read_pod_or_tail(in, type) ||
            !read_pod_or_tail(in, payload_bytes)) {
            result.ignored_tail = true;
            break;
        }
        uint32_t expected_payload_bytes = 0;
        if (!expected_frame_payload_bytes(result.header, type, expected_payload_bytes) ||
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
        if (!read_pod_or_tail(in, stored_crc)) {
            result.ignored_tail = true;
            break;
        }
        uint32_t expected_crc = frame_crc(lsn, type, payload_bytes, payload);
        if (stored_crc != expected_crc) {
            result.ignored_tail = true;
            break;
        }
        last_frame_lsn = lsn;

        if (type == (uint8_t)UpdateWalRecord::Type::Insert) {
            if (!pending_block) {
                pending_block = std::make_unique<L0Block::Builder>(
                    cells_per_subspace,
                    result.header.verify_dim,
                    result.header.rabitq_record_bytes,
                    false);
                if (!reserve_used && first_block_reserve_rows != 0) {
                    pending_block->reserve_rows(first_block_reserve_rows);
                    reserve_used = true;
                }
            }
            uint32_t pid = pending_block->append_insert_payload(result.header, payload);
            if (pid < result.header.base_n) {
                throw std::runtime_error("update WAL insert PID overlaps base segment");
            }
            set_live.push_back(pid);
        } else if (type == (uint8_t)UpdateWalRecord::Type::Delete) {
            UpdateWalDelete tombstone = decode_delete_payload(payload);
            set_dead.push_back(tombstone.old_global_pid);
        } else if (type == 3) {
            UpdateWalCommit commit = decode_commit_payload(lsn, payload);
            size_t pending_count = set_live.size() + set_dead.size();
            if (commit.batch_record_count != pending_count) {
                throw std::runtime_error("update WAL commit record count mismatch: " + path);
            }
            if (commit.live_count > commit.max_pid) {
                throw std::runtime_error("update WAL commit live_count exceeds max_pid: " + path);
            }
            if (pending_block) {
                std::shared_ptr<const L0Block> block = pending_block->finish(commit.lsn);
                if (block) result.committed_blocks.push_back(std::move(block));
                pending_block.reset();
            }
            for (uint32_t pid : set_live) {
                if (pid < result.max_pid) result.committed_insert_reused_pid = true;
            }
            uint32_t commit_base_live_deletes = 0;
            for (uint32_t pid : set_dead) {
                if (pid < result.header.base_n && result.live && result.live->is_live(pid)) {
                    ++commit_base_live_deletes;
                }
            }
            if (commit_base_live_deletes != 0) result.committed_base_live_change = true;
            result.committed_base_live_deletes += commit_base_live_deletes;
            if (!set_live.empty()) result.committed_insert = true;
            if (!set_dead.empty()) result.committed_delete = true;
            if (!replay_live) {
                replay_live = std::make_shared<LiveTable>(*result.live);
                result.live = replay_live;
            }
            replay_live->apply_updates_in_place(commit.max_pid, set_live, set_dead);
            if (replay_live->live_count() != commit.live_count) {
                throw std::runtime_error("update WAL live_count does not match replayed live table");
            }
            result.last_committed_lsn = commit.lsn;
            result.max_pid = commit.max_pid;
            result.live_count = commit.live_count;
            result.committed_dead_pids.insert(
                result.committed_dead_pids.end(), set_dead.begin(), set_dead.end());
            std::streamoff commit_offset = in.tellg();
            if (commit_offset >= 0) {
                result.last_committed_wal_offset = (uint64_t)commit_offset;
            }
            set_live.clear();
            set_dead.clear();
        } else {
            throw std::runtime_error("unknown update WAL frame type: " + path);
        }
    }

    return result;
}
} // namespace impl

class QueryIndexHandle;

class QueryEngine {
    // QueryIndexHandle owns the immutable base data and the atomically-published
    // ReaderView snapshots, and reuses this engine as a single-thread mutable
    // query executor (one per thread). It needs access to the ReaderView build
    // helpers and the shared base/rotation attachment path.
    friend class QueryIndexHandle;

public:
    QueryEngine() = default;
    ~QueryEngine() = default;

    void init(const IndexConfig& config) {
        if (config.topk < 0) {
            throw std::invalid_argument("QueryEngine topk must be non-negative");
        }
        config_ = config;
        dim_ = 0;
        n_ = 0;

        rotation_ = std::make_unique<RandomRotation>();
        sc_score_ = std::make_unique<SCScore>();
        subspace_rotation_.reset();
        rotate_subspace_ = false;
        shared_data_.reset();
        reader_view_.reset();
        thread_engines_.clear();
    }

    void load_index(const std::string& index_path) {
        impl::OptionalUpdateReadLock read_lock(index_path);
        load_index_unlocked(index_path);
    }

    void load_index_with_update_write_lock_held(const std::string& index_path) {
        load_index_unlocked(index_path);
    }

private:
    void load_index_unlocked(const std::string& index_path) {
        index_path_ = index_path;
        impl::recover_compaction_if_marker_present(index_path_);
        bool has_update_manifest = update_manifest_exists(index_path_);
        UpdateManifest update_manifest;
        if (has_update_manifest) {
            update_manifest = load_update_manifest_json_for_index(index_path_);
        }

        load_rotations(index_path_, true);

        auto shared = std::make_shared<BaseSegment>();
        shared->dim = dim_;

        std::ifstream cent_check(index_path_ + "/rabitq_centroids.bin", std::ios::binary);
        if (cent_check.is_open()) {
            int K = 0, d = 0, pd = 0;
            cent_check.read(reinterpret_cast<char*>(&K), sizeof(int));
            cent_check.read(reinterpret_cast<char*>(&d), sizeof(int));
            cent_check.read(reinterpret_cast<char*>(&pd), sizeof(int));
            if (!cent_check) {
                throw std::runtime_error("Short rabitq_centroids.bin header");
            }
            if (K <= 0 || K > 1'000'000) {
                throw std::runtime_error("Invalid RaBitQ centroid count in rabitq_centroids.bin");
            }
            if (d != dim_) {
                throw std::runtime_error("RaBitQ centroid dimension does not match rotation dimension");
            }
            if (pd != rotation_->padded_dim()) {
                throw std::runtime_error("RaBitQ centroid padded dimension does not match rotation padded dimension");
            }
            if (pd <= 0 || (size_t)K > std::numeric_limits<size_t>::max() / (size_t)pd ||
                (size_t)K * (size_t)pd > std::numeric_limits<size_t>::max() / sizeof(float)) {
                throw std::runtime_error("RaBitQ centroid payload size overflows");
            }
            shared->rabitq_k = K;
            shared->rabitq_padded_dim = pd;
            shared->rabitq_rot_centroids.resize((size_t)K * pd);
            cent_check.read(reinterpret_cast<char*>(shared->rabitq_rot_centroids.data()),
                           (size_t)K * pd * sizeof(float));
            if (!cent_check) {
                throw std::runtime_error("Short rabitq_centroids.bin payload");
            }
            cent_check.close();

            std::ifstream bindata_in(index_path_ + "/rabitq_bin_data.bin", std::ios::binary);
            if (!bindata_in.is_open()) {
                throw std::runtime_error("Cannot open assignment-embedded rabitq_bin_data.bin");
            }
            int n_bd = 0, record_bytes_int = 0;
            bindata_in.read(reinterpret_cast<char*>(&n_bd), sizeof(int));
            bindata_in.read(reinterpret_cast<char*>(&record_bytes_int), sizeof(int));
            if (!bindata_in || n_bd <= 0) {
                throw std::runtime_error("Bad assignment-embedded rabitq_bin_data.bin header");
            }
            const size_t expected_payload_bytes =
                rabitqlib::BinDataMap<float>::data_bytes(shared->rabitq_padded_dim);
            const size_t expected_record_bytes = sizeof(uint32_t) + expected_payload_bytes;
            if (record_bytes_int != (int)expected_record_bytes) {
                throw std::runtime_error(
                    "Bad assignment-embedded RaBitQ record size: expected " +
                    std::to_string(expected_record_bytes) + ", got " +
                    std::to_string(record_bytes_int));
            }
            shared->n = n_bd;
            shared->rabitq_bin_with_assign_record_bytes_ = (size_t)record_bytes_int;
            shared->rabitq_bin_bytes = expected_payload_bytes;
            {
                if ((size_t)n_bd > std::numeric_limits<size_t>::max() / shared->rabitq_bin_with_assign_record_bytes_) {
                    throw std::runtime_error("assignment-embedded rabitq_bin_data.bin size overflows");
                }
                size_t data_sz = (size_t)n_bd * shared->rabitq_bin_with_assign_record_bytes_;
                void* buf = mmap(nullptr, data_sz, PROT_READ | PROT_WRITE,
                                 MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
                if (buf == MAP_FAILED) throw std::runtime_error("mmap assignment-embedded rabitq_bin_data failed");
                bindata_in.read(static_cast<char*>(buf), data_sz);
                if (!bindata_in) {
                    munmap(buf, data_sz);
                    throw std::runtime_error("Short read from assignment-embedded rabitq_bin_data.bin");
                }
                const char* records = static_cast<const char*>(buf);
                for (int i = 0; i < n_bd; ++i) {
                    uint32_t cid = *reinterpret_cast<const uint32_t*>(records + (size_t)i * shared->rabitq_bin_with_assign_record_bytes_);
                    if (cid >= (uint32_t)shared->rabitq_k) {
                        munmap(buf, data_sz);
                        throw std::runtime_error("RaBitQ embedded cluster id out of range in rabitq_bin_data.bin");
                    }
                }
                madvise(buf, data_sz, MADV_HUGEPAGE);
                shared->rabitq_bin_with_assign_ = static_cast<char*>(buf);
                shared->rabitq_bin_with_assign_alloc_bytes_ = data_sz;
            }
            bindata_in.close();

            shared->query_config = rabitqlib::quant::faster_config(
                shared->rabitq_padded_dim,
                rabitqlib::SplitSingleQuery<float>::kNumBits);

            std::cout << "  [RaBitQ-clustered] K=" << shared->rabitq_k
                      << " padded=" << shared->rabitq_padded_dim
                      << " bin=" << shared->rabitq_bin_bytes << " bytes/pt"
                      << " record=" << shared->rabitq_bin_with_assign_record_bytes_ << " bytes/pt"
                      << " (" << (double)shared->rabitq_bin_with_assign_alloc_bytes_ / 1e6 << " MB)" << std::endl;
        } else {
            std::cerr << "[Error] Index was not built with RaBitQ-clustered mode (missing rabitq_centroids.bin)" << std::endl;
            return;
        }

        int M = config_.num_subspaces;
        {
            int detected_M = 0;
            for (int m = 0; ; m++) {
                std::ifstream probe(index_path + "/ivf_subspace_" + std::to_string(m) + ".bin");
                if (!probe.is_open()) break;
                detected_M++;
            }
            if (detected_M > 0 && detected_M != M) {
                std::cout << "  [IVF] Auto-detected M=" << detected_M
                          << " (config had M=" << M << ")" << std::endl;
                M = detected_M;
                config_.num_subspaces = M;
            }
            shared->ivf_indices.resize(M);
            for (int m = 0; m < M; m++) {
                shared->ivf_indices[m].load(index_path_ + "/ivf_subspace_" + std::to_string(m) + ".bin");
                shared->ivf_indices[m].prepare_simd_centroids();
            }
            std::cout << "  [IVF] Loaded " << M << " subspace indices, K="
                      << shared->ivf_indices[0].K() << std::endl;
        }

        shared->posting_lists = std::make_shared<PostingLists>();
        shared->posting_lists->load(index_path_, M);
        validate_subspace_index_compatibility(*shared, M);
        build_cell_size_quantiles(*shared, M);

        {
            auto [tag, n_rot, d_rot] = detect_vector_format(index_path_ + "/verify_vectors.fbin");
            if (tag != 0) {
                throw std::runtime_error("Cannot open verify_vectors.fbin");
            }
            if (n_rot != shared->n || d_rot != shared->rabitq_padded_dim) {
                throw std::runtime_error("verify_vectors.fbin shape does not match RaBitQ payloads");
            }
            if ((size_t)n_rot > std::numeric_limits<size_t>::max() / (size_t)d_rot ||
                (size_t)n_rot * (size_t)d_rot > std::numeric_limits<size_t>::max() / sizeof(float)) {
                throw std::runtime_error("verify_vectors.fbin payload size overflows");
            }
            size_t total_floats = (size_t)n_rot * d_rot;
            size_t total_bytes = total_floats * sizeof(float);
            size_t skip = 12;

            Timer vt; vt.reset();

            int fd = ::open((index_path_ + "/verify_vectors.fbin").c_str(), O_RDONLY);
            if (fd < 0) throw std::runtime_error("Cannot open verify_vectors.fbin");

            void* buf = mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE,
                             MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
            if (buf == MAP_FAILED) {
                ::close(fd);
                throw std::runtime_error("mmap anonymous failed for verify_vectors");
            }
            parallel_read(fd, buf, total_bytes, skip);
            ::close(fd);
            madvise(buf, total_bytes, MADV_HUGEPAGE);

            shared->verify_vectors_ = static_cast<float*>(buf);
            shared->verify_vectors_bytes_ = total_bytes;
            shared->verify_vectors_dim = d_rot;

            std::cout << "  [Verify] " << n_rot << " x " << d_rot
                      << ", " << vt.elapsed_ms() << " ms" << std::endl;
        }

        {
            std::ifstream perm_in(index_path_ + "/perm_to_orig.bin", std::ios::binary);
            if (perm_in.good()) {
                int n_perm = 0;
                perm_in.read(reinterpret_cast<char*>(&n_perm), sizeof(int));
                if (!perm_in || n_perm != shared->n) {
                    throw std::runtime_error("perm_to_orig.bin length does not match RaBitQ payloads");
                }
                shared->perm_to_orig.resize(n_perm);
                perm_in.read(reinterpret_cast<char*>(shared->perm_to_orig.data()), (size_t)n_perm * sizeof(uint32_t));
                if (!perm_in) throw std::runtime_error("Short read from perm_to_orig.bin");
                std::cout << "  [PID reorder] Loaded perm_to_orig for " << n_perm << " points" << std::endl;
            }
        }

        std::shared_ptr<const ReaderView> loaded_view;
        if (has_update_manifest) {
            loaded_view = make_l0_reader_view(shared, update_manifest, index_path_);
        }
        impl::recover_compaction_if_marker_present(index_path_);
        if (update_manifest_exists(index_path_) != has_update_manifest) {
            throw std::runtime_error("update manifest changed during index load; retry load_index");
        }
        attach_shared_data(shared, index_path_, false, false, loaded_view);
    }

public:
    void reload_update_view() {
        if (!shared_data_) {
            throw std::runtime_error("QueryEngine must load an index before reloading update view");
        }
        impl::OptionalUpdateReadLock read_lock(index_path_);
        std::shared_ptr<const ReaderView> loaded_view;
        bool has_update_manifest = update_manifest_exists(index_path_);
        if (has_update_manifest) {
            UpdateManifest manifest = load_update_manifest_json_for_index(index_path_);
            if (manifest.base_n != (uint32_t)shared_data_->n) {
                throw std::runtime_error("update manifest base_n does not match loaded base segment");
            }
            loaded_view =
                make_l0_reader_view(shared_data_, manifest, index_path_, reader_view_.get());
        } else {
            loaded_view = make_base_reader_view(shared_data_);
        }
        if (update_manifest_exists(index_path_) != has_update_manifest) {
            throw std::runtime_error("update manifest changed during update view reload; retry reload_update_view");
        }
        install_reader_view(std::move(loaded_view), false);
    }

    bool publish_committed_update_delta(const UpdateManifest& manifest,
                                        const CommittedUpdateDelta& delta) {
        if (!shared_data_) {
            throw std::runtime_error("QueryEngine must load an index before publishing update delta");
        }
        if (!committed_delta_matches_loaded_reader(
                shared_data_, manifest, delta, reader_view_.get())) {
            reload_update_view();
            return false;
        }

        std::vector<size_t> cells_per_subspace = reader_cells_per_subspace(shared_data_);
        LoadedSealedDeltas sealed = load_sealed_deltas_for_reader(
            shared_data_, manifest, index_path_, reader_view_.get(), cells_per_subspace);
        validate_l0_pid_floor(manifest);
        L0ReplayState l0 = l0_state_from_committed_delta(
            delta, *reader_view_, cells_per_subspace);
        std::shared_ptr<const ReaderView> loaded_view =
            make_l0_reader_view_from_l0_state(
                shared_data_,
                manifest,
                index_path_,
                reader_view_.get(),
                cells_per_subspace,
                std::move(sealed),
                std::move(l0),
                false);
        install_reader_view(std::move(loaded_view), false);
        return true;
    }

    void reload_index_for_maintenance() {
        if (index_path_.empty()) {
            throw std::runtime_error("QueryEngine must load an index before maintenance reload");
        }
        std::string index_path = index_path_;
        IndexConfig config = config_;
        init(config);
        load_index(index_path);
    }

    QueryResult search(const float* query) {
        QueryResult result;
        search_into(query, result);
        return result;
    }

    void search_into(const float* query, QueryResult& result) {
        const bool profile = config_.query_profile;
        Timer total_timer;
        Timer phase_timer;
        if (profile) {
            total_timer.reset();
            phase_timer.reset();
        }
        reset_query_result(result);
        result.topk.reserve((size_t)config_.topk);
        std::fill(q_rot_.begin(), q_rot_.end(), 0.0f);

        rotation_->apply(query, q_rot_.data());
        if (profile) {
            profile_stats_.rotate_us += phase_timer.elapsed_us();
            phase_timer.reset();
        }

        const float* subspace_query = prepare_subspace_query(query);
        phase5_candidates_.clear();
        std::vector<int>& candidates = phase5_candidates_;
        size_t target_per_sub = (size_t)(config_.collision_ratio * active_point_count());

        run_flat_search_path(subspace_query, target_per_sub, candidates);
        if (profile) phase_timer.reset();
        search_rabitq_clustered(query, q_rot_.data(), candidates, result);
        if (profile) {
            profile_stats_.refine_total_us += phase_timer.elapsed_us();
            profile_stats_.total_us += total_timer.elapsed_us();
            ++profile_stats_.queries;
        }
    }

    void search_rabitq_clustered(
        const float* /*query*/, const float* q_rot,
        const std::vector<int>& candidates,
        QueryResult& qr) {

        qr.topk.clear();
        if (candidates.empty()) return;
        Timer phase_timer;
        if (config_.query_profile) phase_timer.reset();
        if (query_has_updates()) {
            uint32_t base_limit = std::min(
                reader_view_->base_n,
                (uint32_t)std::max(0, shared_data_->n));
            bool has_delta_candidate = false;
            for (int pid : candidates) {
                if (pid < 0 || (uint32_t)pid >= base_limit) {
                    has_delta_candidate = true;
                    break;
                }
            }
            if (has_delta_candidate) {
                resolved_ordered_candidates_.assign(candidates.begin(), candidates.end());
                auto first_delta = std::partition(
                    resolved_ordered_candidates_.begin(),
                    resolved_ordered_candidates_.end(),
                    [base_limit](int pid) {
                        return pid >= 0 && (uint32_t)pid < base_limit;
                    });
                search_rabitq_clustered_resolved(
                    q_rot,
                    resolved_ordered_candidates_,
                    (size_t)std::distance(resolved_ordered_candidates_.begin(), first_delta),
                    qr);
                return;
            }
        }

        rabitqlib::SplitSingleQuery<float> q_obj(
            q_rot, shared_data_->rabitq_padded_dim, 0, shared_data_->query_config, rabitqlib::METRIC_L2);

        const int n = (int)candidates.size();

        const uint32_t stamp = impl::next_query_stamp(rabitq_gadd_stamp_, rabitq_gadd_stamp_value_);

        ranked_size_ = 0;
        if (ranked_.size() < (size_t)n) {
            ranked_.resize((size_t)n);
        }

        const size_t num_blk = shared_data_->rabitq_padded_dim / 64;
        const size_t query_bitplane_words = num_blk * 4;
        if (query_bitplane_.size() < query_bitplane_words) {
            query_bitplane_.resize(query_bitplane_words);
        }
        transpose_query_to_bitplane(q_obj.query_bin(), query_bitplane_.data(), num_blk);
        const float q_delta = q_obj.delta();
        const float q_vl = q_obj.vl();
        const float q_k1 = q_obj.k1xsumq();

        const size_t embedded_stride = shared_data_->rabitq_bin_with_assign_record_bytes_;
        const char* bin_base = shared_data_->rabitq_bin_with_assign_;
        constexpr int BATCH = 8;
        constexpr int PF6A = 16;
        const size_t pack_words = num_blk * (size_t)BATCH;
        if (phase6a_pack_words_.size() < pack_words) {
            phase6a_pack_words_.resize(pack_words);
        }
        alignas(64) std::array<int, BATCH> batch_pids{};
        alignas(64) std::array<uint32_t, BATCH> batch_cids{};
        alignas(64) std::array<RabitqFactorTriplet, BATCH> batch_factors{};
        alignas(64) std::array<uint64_t, BATCH> ip_counts{};
        alignas(64) std::array<uint64_t, BATCH> ppc_counts{};
        const size_t bin_words = shared_data_->rabitq_padded_dim / 64;
        const size_t factor_offset = shared_data_->rabitq_padded_dim / 8;

        for (int i = 0; i < n; i += BATCH) {
            const int active = std::min(BATCH, n - i);
            for (int lane = 0; lane < active; ++lane) {
                const int pf_index = i + PF6A + lane;
                if (pf_index < n) {
                    const int pf_pid = candidates[pf_index];
                    const char* pf = bin_base + (size_t)pf_pid * embedded_stride;
                    __builtin_prefetch(pf, 0, 0);
                    __builtin_prefetch(pf + 64, 0, 0);
                    __builtin_prefetch(pf + 128, 0, 0);
                }

                const int pid = candidates[i + lane];
                const char* record = bin_base + (size_t)pid * embedded_stride;
                const uint32_t cid = *reinterpret_cast<const uint32_t*>(record);
                batch_pids[(size_t)lane] = pid;
                batch_cids[(size_t)lane] = cid;
                if (rabitq_gadd_stamp_[cid] != stamp) {
                    rabitq_gadd_stamp_[cid] = stamp;
                    float dist = l2_distance_sq(
                        q_rot,
                        shared_data_->rabitq_rot_centroids.data() + (size_t)cid * shared_data_->rabitq_padded_dim,
                        shared_data_->rabitq_padded_dim);
                    g_add_[cid] = dist;
                    g_err_[cid] = std::sqrt(dist);
                }

                const char* bin_ptr = record + sizeof(uint32_t);
                const uint64_t* words = reinterpret_cast<const uint64_t*>(bin_ptr);
                for (size_t w = 0; w < bin_words; ++w) {
                    phase6a_pack_words_[w * (size_t)BATCH + (size_t)lane] = words[w];
                }
                std::memcpy(&batch_factors[(size_t)lane], bin_ptr + factor_offset, sizeof(RabitqFactorTriplet));
            }

#if defined(__AVX512F__) && defined(__AVX512VPOPCNTDQ__)
            __m512i ip_vec = _mm512_setzero_si512();
            __m512i ppc_vec = _mm512_setzero_si512();
            for (size_t w = 0; w < bin_words; ++w) {
                const __m512i x = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(phase6a_pack_words_.data() + w * (size_t)BATCH));
                ppc_vec = _mm512_add_epi64(ppc_vec, _mm512_popcnt_epi64(x));
                const __m512i q0 = _mm512_set1_epi64((long long)query_bitplane_[w]);
                const __m512i q1 = _mm512_set1_epi64((long long)query_bitplane_[bin_words + w]);
                const __m512i q2 = _mm512_set1_epi64((long long)query_bitplane_[2 * bin_words + w]);
                const __m512i q3 = _mm512_set1_epi64((long long)query_bitplane_[3 * bin_words + w]);
                __m512i weighted = _mm512_popcnt_epi64(_mm512_and_si512(x, q0));
                weighted = _mm512_add_epi64(weighted,
                    _mm512_slli_epi64(_mm512_popcnt_epi64(_mm512_and_si512(x, q1)), 1));
                weighted = _mm512_add_epi64(weighted,
                    _mm512_slli_epi64(_mm512_popcnt_epi64(_mm512_and_si512(x, q2)), 2));
                weighted = _mm512_add_epi64(weighted,
                    _mm512_slli_epi64(_mm512_popcnt_epi64(_mm512_and_si512(x, q3)), 3));
                ip_vec = _mm512_add_epi64(ip_vec, weighted);
            }
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(ip_counts.data()), ip_vec);
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(ppc_counts.data()), ppc_vec);
#else
            std::fill(ip_counts.begin(), ip_counts.end(), 0);
            std::fill(ppc_counts.begin(), ppc_counts.end(), 0);
            for (size_t w = 0; w < bin_words; ++w) {
                for (int lane = 0; lane < BATCH; ++lane) {
                    const uint64_t x = phase6a_pack_words_[w * (size_t)BATCH + (size_t)lane];
                    ppc_counts[(size_t)lane] += (uint64_t)__builtin_popcountll(x);
                    for (int bit = 0; bit < 4; ++bit) {
                        ip_counts[(size_t)lane] +=
                            (uint64_t)__builtin_popcountll(x & query_bitplane_[(size_t)bit * bin_words + w]) << bit;
                    }
                }
            }
#endif

            for (int lane = 0; lane < active; ++lane) {
                const uint32_t cid = batch_cids[(size_t)lane];
                const auto& factors = batch_factors[(size_t)lane];
                const float ip_x0_qr = q_delta * (float)ip_counts[(size_t)lane] +
                                       q_vl * (float)ppc_counts[(size_t)lane];
                const float est_dist = factors.f_add + g_add_[cid] +
                                       factors.f_rescale * (ip_x0_qr + q_k1);
                const float low_dist = est_dist - factors.f_error * g_err_[cid];
                ranked_[(size_t)(i + lane)] = {est_dist, low_dist, batch_pids[(size_t)lane]};
            }
        }
        ranked_size_ = (size_t)n;

        select_rabitq_verification_candidates();
        if (config_.query_profile) {
            profile_stats_.rabitq_rank_us += phase_timer.elapsed_us();
            phase_timer.reset();
        }

        verify_results_.clear();
        verify_results_.reserve(top_pids_.size());
        const int verify_dim = shared_data_->verify_vectors_dim;
        const float* verify_query = q_rot;
        for (int i = 0; i < (int)top_pids_.size(); i++) {
            int pid = top_pids_[(size_t)i];
            const float* src = shared_data_->verify_vectors_ + (size_t)pid * verify_dim;
            if (i + 16 < (int)top_pids_.size()) {
                const float* pf = shared_data_->verify_vectors_ + (size_t)top_pids_[(size_t)i + 16] * verify_dim;
                __builtin_prefetch(pf, 0, 0);
                __builtin_prefetch(pf + 16, 0, 0);
                __builtin_prefetch(pf + 32, 0, 0);
                __builtin_prefetch(pf + 48, 0, 0);
                __builtin_prefetch(pf + 64, 0, 0);
                __builtin_prefetch(pf + 80, 0, 0);
            }
            float exact_dist_sq = l2_distance_sq(verify_query, src, verify_dim);
            int out_pid = shared_data_->perm_to_orig.empty() ? pid : (int)shared_data_->perm_to_orig[(size_t)pid];
            verify_results_.push_back({std::sqrt(exact_dist_sq), out_pid});
        }
        std::sort(verify_results_.begin(), verify_results_.end());
        size_t keep = std::min(verify_results_.size(), (size_t)config_.topk);
        qr.topk.assign(verify_results_.begin(), verify_results_.begin() + (std::ptrdiff_t)keep);
        if (config_.query_profile) {
            profile_stats_.verify_us += phase_timer.elapsed_us();
        }
    }

    void search_rabitq_clustered_resolved(
        const float* q_rot,
        const std::vector<int>& candidates,
        size_t first_delta_candidate,
        QueryResult& qr) {

        Timer phase_timer;
        if (config_.query_profile) phase_timer.reset();
        rabitqlib::SplitSingleQuery<float> q_obj(
            q_rot, shared_data_->rabitq_padded_dim, 0, shared_data_->query_config, rabitqlib::METRIC_L2);

        const uint32_t stamp = impl::next_query_stamp(rabitq_gadd_stamp_, rabitq_gadd_stamp_value_);
        const size_t num_blk = shared_data_->rabitq_padded_dim / 64;
        const size_t query_bitplane_words = num_blk * 4;
        if (query_bitplane_.size() < query_bitplane_words) {
            query_bitplane_.resize(query_bitplane_words);
        }
        transpose_query_to_bitplane(q_obj.query_bin(), query_bitplane_.data(), num_blk);
        const float q_delta = q_obj.delta();
        const float q_vl = q_obj.vl();
        const float q_k1 = q_obj.k1xsumq();
        const size_t bin_words = shared_data_->rabitq_padded_dim / 64;
        const size_t factor_offset = shared_data_->rabitq_padded_dim / 8;

        ranked_size_ = 0;
        if (ranked_.size() < candidates.size()) ranked_.resize(candidates.size());
        first_delta_candidate = std::min(first_delta_candidate, candidates.size());
        if (config_.query_profile) {
            ++profile_stats_.resolved_refine_queries;
            profile_stats_.resolved_base_candidates += first_delta_candidate;
            profile_stats_.resolved_delta_candidates += candidates.size() - first_delta_candidate;
        }

        const size_t embedded_stride = shared_data_->rabitq_bin_with_assign_record_bytes_;
        const char* base_records = shared_data_->rabitq_bin_with_assign_;
        const UnsealedDeltaView* growing_delta =
            reader_view_ ? reader_view_->growing_delta.get() : nullptr;
        const L0Block* fast_l0_block = nullptr;
        if (growing_delta && growing_delta->committed_blocks.size() == 1 &&
            growing_delta->committed_blocks.front()) {
            fast_l0_block = growing_delta->committed_blocks.front().get();
        }
        const SealedDeltaSegment* fast_sealed_delta = nullptr;
        if (reader_view_ && reader_view_->sealed_deltas.size() == 1 &&
            reader_view_->sealed_deltas.front() &&
            reader_view_->sealed_deltas.front()->dense_pid_order()) {
            fast_sealed_delta = reader_view_->sealed_deltas.front().get();
        }
        auto fast_delta_rabitq_record_for_pid =
            [&](uint32_t pid, const char*& record, uint32_t& record_bytes) -> bool {
                if (fast_l0_block && fast_l0_block->contains_pid(pid)) {
                    uint32_t local_row = fast_l0_block->local_row_for_pid(pid);
                    record = reinterpret_cast<const char*>(
                        fast_l0_block->rabitq_record(local_row));
                    record_bytes = fast_l0_block->rabitq_record_bytes();
                    return true;
                }
                if (growing_delta &&
                    growing_delta->rabitq_record_for_pid(pid, record, record_bytes)) {
                    return true;
                }
                if (fast_sealed_delta && pid >= fast_sealed_delta->pid_begin()) {
                    uint64_t local = (uint64_t)pid -
                                     (uint64_t)fast_sealed_delta->pid_begin();
                    if (local < (uint64_t)fast_sealed_delta->row_count()) {
                        uint32_t local_row = (uint32_t)local;
                        record = reinterpret_cast<const char*>(
                            fast_sealed_delta->rabitq_record(local_row));
                        record_bytes = fast_sealed_delta->rabitq_record_bytes();
                        return true;
                    }
                }
                return false;
            };
        auto fast_delta_verify_for_pid =
            [&](uint32_t pid,
                const float*& verify_vector,
                uint32_t& verify_dim,
                uint32_t& output_id) -> bool {
                if (fast_l0_block && fast_l0_block->contains_pid(pid)) {
                    uint32_t local_row = fast_l0_block->local_row_for_pid(pid);
                    verify_vector = fast_l0_block->verify_vector(local_row);
                    verify_dim = fast_l0_block->verify_dim();
                    output_id = fast_l0_block->orig_id(local_row);
                    return true;
                }
                if (growing_delta &&
                    growing_delta->verify_row_for_pid(
                        pid, verify_vector, verify_dim, output_id)) {
                    return true;
                }
                if (fast_sealed_delta && pid >= fast_sealed_delta->pid_begin()) {
                    uint64_t local = (uint64_t)pid -
                                     (uint64_t)fast_sealed_delta->pid_begin();
                    if (local < (uint64_t)fast_sealed_delta->row_count()) {
                        uint32_t local_row = (uint32_t)local;
                        verify_vector = fast_sealed_delta->verify_vector(local_row);
                        verify_dim = fast_sealed_delta->verify_dim();
                        output_id = fast_sealed_delta->orig_id(local_row);
                        return true;
                    }
                }
                return false;
            };
        auto rank_record_refs = [&](const std::vector<RabitqRecordRef>& refs) {
            constexpr int BATCH = 8;
            constexpr int PF6A = 16;
            if (refs.empty()) return;
            const size_t pack_words = num_blk * (size_t)BATCH;
            if (phase6a_pack_words_.size() < pack_words) {
                phase6a_pack_words_.resize(pack_words);
            }
            alignas(64) std::array<int, BATCH> batch_pids{};
            alignas(64) std::array<uint32_t, BATCH> batch_cids{};
            alignas(64) std::array<RabitqFactorTriplet, BATCH> batch_factors{};
            alignas(64) std::array<uint64_t, BATCH> ip_counts{};
            alignas(64) std::array<uint64_t, BATCH> ppc_counts{};

            const int n_refs = (int)refs.size();
            for (int i = 0; i < n_refs; i += BATCH) {
                const int active = std::min(BATCH, n_refs - i);
                for (int lane = 0; lane < active; ++lane) {
                    const int pf_index = i + PF6A + lane;
                    if (pf_index < n_refs) {
                        const char* pf = refs[(size_t)pf_index].record;
                        __builtin_prefetch(pf, 0, 0);
                        __builtin_prefetch(pf + 64, 0, 0);
                        __builtin_prefetch(pf + 128, 0, 0);
                    }

                    const RabitqRecordRef& ref = refs[(size_t)i + (size_t)lane];
                    const char* record = ref.record;
                    const uint32_t cid = *reinterpret_cast<const uint32_t*>(record);
                    if (cid >= (uint32_t)shared_data_->rabitq_k) {
                        throw std::runtime_error("resolved RaBitQ embedded cluster id out of range");
                    }
                    batch_pids[(size_t)lane] = ref.pid;
                    batch_cids[(size_t)lane] = cid;
                    if (rabitq_gadd_stamp_[cid] != stamp) {
                        rabitq_gadd_stamp_[cid] = stamp;
                        float dist = l2_distance_sq(
                            q_rot,
                            shared_data_->rabitq_rot_centroids.data() +
                                (size_t)cid * shared_data_->rabitq_padded_dim,
                            shared_data_->rabitq_padded_dim);
                        g_add_[cid] = dist;
                        g_err_[cid] = std::sqrt(dist);
                    }

                    const char* bin_ptr = record + sizeof(uint32_t);
                    const uint64_t* words = reinterpret_cast<const uint64_t*>(bin_ptr);
                    for (size_t w = 0; w < bin_words; ++w) {
                        phase6a_pack_words_[w * (size_t)BATCH + (size_t)lane] = words[w];
                    }
                    std::memcpy(&batch_factors[(size_t)lane],
                                bin_ptr + factor_offset,
                                sizeof(RabitqFactorTriplet));
                }

#if defined(__AVX512F__) && defined(__AVX512VPOPCNTDQ__)
                __m512i ip_vec = _mm512_setzero_si512();
                __m512i ppc_vec = _mm512_setzero_si512();
                for (size_t w = 0; w < bin_words; ++w) {
                    const __m512i x = _mm512_loadu_si512(
                        reinterpret_cast<const __m512i*>(
                            phase6a_pack_words_.data() + w * (size_t)BATCH));
                    ppc_vec = _mm512_add_epi64(ppc_vec, _mm512_popcnt_epi64(x));
                    const __m512i q0 = _mm512_set1_epi64((long long)query_bitplane_[w]);
                    const __m512i q1 =
                        _mm512_set1_epi64((long long)query_bitplane_[bin_words + w]);
                    const __m512i q2 =
                        _mm512_set1_epi64((long long)query_bitplane_[2 * bin_words + w]);
                    const __m512i q3 =
                        _mm512_set1_epi64((long long)query_bitplane_[3 * bin_words + w]);
                    __m512i weighted = _mm512_popcnt_epi64(_mm512_and_si512(x, q0));
                    weighted = _mm512_add_epi64(
                        weighted,
                        _mm512_slli_epi64(_mm512_popcnt_epi64(_mm512_and_si512(x, q1)), 1));
                    weighted = _mm512_add_epi64(
                        weighted,
                        _mm512_slli_epi64(_mm512_popcnt_epi64(_mm512_and_si512(x, q2)), 2));
                    weighted = _mm512_add_epi64(
                        weighted,
                        _mm512_slli_epi64(_mm512_popcnt_epi64(_mm512_and_si512(x, q3)), 3));
                    ip_vec = _mm512_add_epi64(ip_vec, weighted);
                }
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(ip_counts.data()), ip_vec);
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(ppc_counts.data()), ppc_vec);
#else
                std::fill(ip_counts.begin(), ip_counts.end(), 0);
                std::fill(ppc_counts.begin(), ppc_counts.end(), 0);
                for (size_t w = 0; w < bin_words; ++w) {
                    for (int lane = 0; lane < BATCH; ++lane) {
                        const uint64_t x =
                            phase6a_pack_words_[w * (size_t)BATCH + (size_t)lane];
                        ppc_counts[(size_t)lane] += (uint64_t)__builtin_popcountll(x);
                        for (int bit = 0; bit < 4; ++bit) {
                            ip_counts[(size_t)lane] +=
                                (uint64_t)__builtin_popcountll(
                                    x & query_bitplane_[(size_t)bit * bin_words + w]) << bit;
                        }
                    }
                }
#endif

                for (int lane = 0; lane < active; ++lane) {
                    const uint32_t cid = batch_cids[(size_t)lane];
                    const auto& factors = batch_factors[(size_t)lane];
                    const float ip_x0_qr = q_delta * (float)ip_counts[(size_t)lane] +
                                           q_vl * (float)ppc_counts[(size_t)lane];
                    const float est_dist = factors.f_add + g_add_[cid] +
                                           factors.f_rescale * (ip_x0_qr + q_k1);
                    const float low_dist = est_dist - factors.f_error * g_err_[cid];
                    ranked_[(size_t)ranked_size_++] =
                        {est_dist, low_dist, batch_pids[(size_t)lane]};
                }
            }
        };
        auto rank_base_records = [&](size_t begin, size_t end) {
            constexpr int BATCH = 8;
            constexpr int PF6A = 16;
            const size_t pack_words = num_blk * (size_t)BATCH;
            if (phase6a_pack_words_.size() < pack_words) {
                phase6a_pack_words_.resize(pack_words);
            }
            alignas(64) std::array<int, BATCH> batch_pids{};
            alignas(64) std::array<uint32_t, BATCH> batch_cids{};
            alignas(64) std::array<RabitqFactorTriplet, BATCH> batch_factors{};
            alignas(64) std::array<uint64_t, BATCH> ip_counts{};
            alignas(64) std::array<uint64_t, BATCH> ppc_counts{};

            if (end <= begin) return;
            const int n_base = (int)(end - begin);
            for (int i = 0; i < n_base; i += BATCH) {
                const int active = std::min(BATCH, n_base - i);
                for (int lane = 0; lane < active; ++lane) {
                    const int pf_index = i + PF6A + lane;
                    if (pf_index < n_base) {
                        const int pf_pid = candidates[begin + (size_t)pf_index];
                        const char* pf = base_records + (size_t)pf_pid * embedded_stride;
                        __builtin_prefetch(pf, 0, 0);
                        __builtin_prefetch(pf + 64, 0, 0);
                        __builtin_prefetch(pf + 128, 0, 0);
                    }

                    const int pid = candidates[begin + (size_t)i + (size_t)lane];
                    if (pid < 0 || (uint32_t)pid >= (uint32_t)shared_data_->n) {
                        throw std::runtime_error("base candidate outside base segment");
                    }
                    const char* record = base_records + (size_t)pid * embedded_stride;
                    const uint32_t cid = *reinterpret_cast<const uint32_t*>(record);
                    batch_pids[(size_t)lane] = pid;
                    batch_cids[(size_t)lane] = cid;
                    if (rabitq_gadd_stamp_[cid] != stamp) {
                        rabitq_gadd_stamp_[cid] = stamp;
                        float dist = l2_distance_sq(
                            q_rot,
                            shared_data_->rabitq_rot_centroids.data() +
                                (size_t)cid * shared_data_->rabitq_padded_dim,
                            shared_data_->rabitq_padded_dim);
                        g_add_[cid] = dist;
                        g_err_[cid] = std::sqrt(dist);
                    }

                    const char* bin_ptr = record + sizeof(uint32_t);
                    const uint64_t* words = reinterpret_cast<const uint64_t*>(bin_ptr);
                    for (size_t w = 0; w < bin_words; ++w) {
                        phase6a_pack_words_[w * (size_t)BATCH + (size_t)lane] = words[w];
                    }
                    std::memcpy(&batch_factors[(size_t)lane],
                                bin_ptr + factor_offset,
                                sizeof(RabitqFactorTriplet));
                }

#if defined(__AVX512F__) && defined(__AVX512VPOPCNTDQ__)
                __m512i ip_vec = _mm512_setzero_si512();
                __m512i ppc_vec = _mm512_setzero_si512();
                for (size_t w = 0; w < bin_words; ++w) {
                    const __m512i x = _mm512_loadu_si512(
                        reinterpret_cast<const __m512i*>(
                            phase6a_pack_words_.data() + w * (size_t)BATCH));
                    ppc_vec = _mm512_add_epi64(ppc_vec, _mm512_popcnt_epi64(x));
                    const __m512i q0 = _mm512_set1_epi64((long long)query_bitplane_[w]);
                    const __m512i q1 =
                        _mm512_set1_epi64((long long)query_bitplane_[bin_words + w]);
                    const __m512i q2 =
                        _mm512_set1_epi64((long long)query_bitplane_[2 * bin_words + w]);
                    const __m512i q3 =
                        _mm512_set1_epi64((long long)query_bitplane_[3 * bin_words + w]);
                    __m512i weighted = _mm512_popcnt_epi64(_mm512_and_si512(x, q0));
                    weighted = _mm512_add_epi64(
                        weighted,
                        _mm512_slli_epi64(_mm512_popcnt_epi64(_mm512_and_si512(x, q1)), 1));
                    weighted = _mm512_add_epi64(
                        weighted,
                        _mm512_slli_epi64(_mm512_popcnt_epi64(_mm512_and_si512(x, q2)), 2));
                    weighted = _mm512_add_epi64(
                        weighted,
                        _mm512_slli_epi64(_mm512_popcnt_epi64(_mm512_and_si512(x, q3)), 3));
                    ip_vec = _mm512_add_epi64(ip_vec, weighted);
                }
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(ip_counts.data()), ip_vec);
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(ppc_counts.data()), ppc_vec);
#else
                std::fill(ip_counts.begin(), ip_counts.end(), 0);
                std::fill(ppc_counts.begin(), ppc_counts.end(), 0);
                for (size_t w = 0; w < bin_words; ++w) {
                    for (int lane = 0; lane < BATCH; ++lane) {
                        const uint64_t x =
                            phase6a_pack_words_[w * (size_t)BATCH + (size_t)lane];
                        ppc_counts[(size_t)lane] += (uint64_t)__builtin_popcountll(x);
                        for (int bit = 0; bit < 4; ++bit) {
                            ip_counts[(size_t)lane] +=
                                (uint64_t)__builtin_popcountll(
                                    x & query_bitplane_[(size_t)bit * bin_words + w]) << bit;
                        }
                    }
                }
#endif

                for (int lane = 0; lane < active; ++lane) {
                    const uint32_t cid = batch_cids[(size_t)lane];
                    const auto& factors = batch_factors[(size_t)lane];
                    const float ip_x0_qr = q_delta * (float)ip_counts[(size_t)lane] +
                                           q_vl * (float)ppc_counts[(size_t)lane];
                    const float est_dist = factors.f_add + g_add_[cid] +
                                           factors.f_rescale * (ip_x0_qr + q_k1);
                    const float low_dist = est_dist - factors.f_error * g_err_[cid];
                    ranked_[(size_t)ranked_size_++] =
                        {est_dist, low_dist, batch_pids[(size_t)lane]};
                }
            }
        };

        rank_base_records(0, first_delta_candidate);

        RowResolver resolver = row_resolver();
        delta_candidate_record_refs_.clear();
        delta_candidate_record_refs_.reserve(candidates.size() - first_delta_candidate);
        for (size_t candidate_idx = first_delta_candidate;
             candidate_idx < candidates.size();
             ++candidate_idx) {
            int candidate_pid = candidates[candidate_idx];
            if (candidate_pid < 0) continue;
            const char* record = nullptr;
            uint32_t record_bytes = 0;
            if (fast_delta_rabitq_record_for_pid(
                    (uint32_t)candidate_pid, record, record_bytes)) {
                if (record_bytes != embedded_stride) {
                    throw std::runtime_error("resolved RaBitQ record stride mismatch");
                }
                delta_candidate_record_refs_.push_back({candidate_pid, record});
                continue;
            }
            ResolvedRow row;
            if (!resolver.resolve((uint32_t)candidate_pid, row)) {
                continue;
            }
            if (row.rabitq_record_bytes != embedded_stride) {
                throw std::runtime_error("resolved RaBitQ record stride mismatch");
            }
            delta_candidate_record_refs_.push_back({candidate_pid, row.rabitq_record});
        }
        rank_record_refs(delta_candidate_record_refs_);

        if (ranked_size_ == 0) return;
        select_rabitq_verification_candidates();
        if (config_.query_profile) {
            profile_stats_.rabitq_rank_us += phase_timer.elapsed_us();
            phase_timer.reset();
        }

        verify_results_.clear();
        verify_results_.reserve(top_pids_.size());
        for (int i = 0; i < (int)top_pids_.size(); ++i) {
            int pid = top_pids_[(size_t)i];
            if (i + 16 < (int)top_pids_.size()) {
                int pf_pid = top_pids_[(size_t)i + 16];
                if (pf_pid >= 0 &&
                    (uint32_t)pf_pid < reader_view_->base_n &&
                    (uint32_t)pf_pid < (uint32_t)shared_data_->n) {
                    const float* pf = shared_data_->verify_vectors_ +
                        (size_t)pf_pid * (size_t)shared_data_->verify_vectors_dim;
                    __builtin_prefetch(pf, 0, 0);
                    __builtin_prefetch(pf + 16, 0, 0);
                    __builtin_prefetch(pf + 32, 0, 0);
                    __builtin_prefetch(pf + 48, 0, 0);
                    __builtin_prefetch(pf + 64, 0, 0);
                    __builtin_prefetch(pf + 80, 0, 0);
                }
            }
            if (pid >= 0 && (uint32_t)pid < reader_view_->base_n &&
                (uint32_t)pid < (uint32_t)shared_data_->n) {
                const float* verify = shared_data_->verify_vectors_ +
                    (size_t)pid * (size_t)shared_data_->verify_vectors_dim;
                float exact_dist_sq = l2_distance_sq(q_rot, verify, shared_data_->verify_vectors_dim);
                int out_pid = shared_data_->perm_to_orig.empty()
                    ? pid
                    : (int)shared_data_->perm_to_orig[(size_t)pid];
                verify_results_.push_back({std::sqrt(exact_dist_sq), out_pid});
                continue;
            }
            const float* fast_verify = nullptr;
            uint32_t fast_verify_dim = 0;
            uint32_t fast_output_id = 0;
            if (fast_delta_verify_for_pid(
                    (uint32_t)pid, fast_verify, fast_verify_dim, fast_output_id)) {
                if ((int)fast_verify_dim != shared_data_->verify_vectors_dim) {
                    throw std::runtime_error("resolved verify vector dimension mismatch");
                }
                float exact_dist_sq = l2_distance_sq(
                    q_rot, fast_verify, shared_data_->verify_vectors_dim);
                verify_results_.push_back({std::sqrt(exact_dist_sq), (int)fast_output_id});
                continue;
            }
            ResolvedRow row;
            if (!resolver.resolve((uint32_t)pid, row)) {
                continue;
            }
            if (row.verify_dim != shared_data_->verify_vectors_dim) {
                throw std::runtime_error("resolved verify vector dimension mismatch");
            }
            float exact_dist_sq = l2_distance_sq(q_rot, row.verify_vector, row.verify_dim);
            verify_results_.push_back({std::sqrt(exact_dist_sq), (int)row.output_id});
        }
        std::sort(verify_results_.begin(), verify_results_.end());
        size_t keep = std::min(verify_results_.size(), (size_t)config_.topk);
        qr.topk.assign(verify_results_.begin(), verify_results_.begin() + (std::ptrdiff_t)keep);
        if (config_.query_profile) {
            profile_stats_.verify_us += phase_timer.elapsed_us();
        }
    }

    void search_batch_preallocate() {
        int nt = configured_num_threads();
        thread_engines_.resize(nt);
        #pragma omp parallel for num_threads(nt) schedule(dynamic, 1)
        for (int t = 0; t < nt; t++) {
            thread_engines_[t].engine = new QueryEngine();
            thread_engines_[t].engine->init(config_);
            thread_engines_[t].engine->attach_shared_data(shared_data_, index_path_, false, true, reader_view_);
        }
    }

    std::vector<QueryResult> search_batch(const float* queries, int num_queries) {
        std::vector<QueryResult> results(num_queries);
        int nt = configured_num_threads();
        thread_engines_.resize(nt);
        #pragma omp parallel for num_threads(nt) schedule(dynamic, 1)
        for (int i = 0; i < num_queries; i++) {
            int t = omp_get_thread_num();
            if (!thread_engines_[t].engine) {
                thread_engines_[t].engine = new QueryEngine();
                thread_engines_[t].engine->init(config_);
                thread_engines_[t].engine->attach_shared_data(shared_data_, index_path_, false, true, reader_view_);
            }
            thread_engines_[t].engine->search_into(queries + (size_t)i * dim_, results[i]);
        }
        return results;
    }

    int dim() const { return dim_; }
    int num_vectors() const { return n_; }
    const SCScore& sc_score() const { return *sc_score_; }
    const std::vector<uint32_t>& perm_to_orig() const { return shared_data_->perm_to_orig; }
    const ReaderView& reader_view() const {
        if (!reader_view_) throw std::runtime_error("QueryEngine reader view is not loaded");
        return *reader_view_;
    }
    // Shared-pointer accessors used by QueryIndexHandle to share the immutable
    // base segment, configuration, and the current reader snapshot across
    // per-thread query executors without copying query scratch.
    std::shared_ptr<const ReaderView> reader_view_shared() const { return reader_view_; }
    const std::shared_ptr<BaseSegment>& shared_data() const { return shared_data_; }
    const IndexConfig& config() const { return config_; }
    using SharedDataType = BaseSegment;
    RowResolver row_resolver() const { return RowResolver(reader_view()); }
    const QueryProfileStats& local_profile_stats() const { return profile_stats_; }
    QueryProfileStats collect_profile_stats() const {
        QueryProfileStats stats = profile_stats_;
        for (const ThreadEngine& thread_engine : thread_engines_) {
            if (thread_engine.engine) {
                stats.add(thread_engine.engine->local_profile_stats());
            }
        }
        return stats;
    }
    void reset_profile_stats() {
        profile_stats_ = QueryProfileStats{};
        for (ThreadEngine& thread_engine : thread_engines_) {
            if (thread_engine.engine) thread_engine.engine->reset_profile_stats();
        }
    }
    size_t effective_candidate_budget() const {
        size_t points = query_has_updates() ? active_point_count() : sc_score_->num_points();
        return std::max((size_t)1, (size_t)(config_.candidate_ratio * points));
    }

private:
    struct Phase4CellRef {
        const uint32_t* base_data = nullptr;
        const uint32_t* sealed_delta_data = nullptr;
        const uint32_t* growing_delta_data = nullptr;
        uint32_t base_size = 0;
        uint32_t sealed_delta_size = 0;
        uint32_t growing_delta_size = 0;
        int subspace = 0;
        int cell = 0;
        uint8_t weight = 1;
    };

    struct RabitqRecordRef {
        int pid = 0;
        const char* record = nullptr;
    };

    enum class Phase4PostingBackend : uint8_t {
        GeneralOverlay,
        SealedOverlay,
        GrowingOnlyGeneral,
        GrowingOverlay,
    };

    static void reset_query_result(QueryResult& result) {
        result.topk.clear();
    }

    static uint32_t checked_phase4_span_count(size_t count) {
        if (count > (size_t)std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("Phase4 posting cell span exceeds uint32 range");
        }
        return (uint32_t)count;
    }

    static constexpr uint64_t kDirectDeltaScoreMaxPostings = 8192;
    bool should_score_delta_directly() const {
        return phase4_delta_posting_count_ != 0 &&
               phase4_delta_posting_count_ <= kDirectDeltaScoreMaxPostings;
    }

    size_t active_point_count() const {
        return active_point_count_;
    }

    bool query_has_updates() const {
        return query_has_updates_;
    }

    bool composite_cell_size_needs_live_count() const {
        return live_filter_needed_;
    }

    bool query_growing_only() const {
        return query_growing_only_;
    }

    Phase4PostingBackend phase4_posting_backend(
        const ReaderView* view,
        const SealedDeltaPostingView* sealed_postings,
        const UnsealedDeltaView* growing) const {
        if (query_growing_only()) {
            if (reader_view_ &&
                !reader_view_->base_live_postings &&
                reader_view_->base_dead_prefix_end == 0 &&
                growing != nullptr &&
                growing->all_pids_live()) {
                return Phase4PostingBackend::GrowingOverlay;
            }
            return Phase4PostingBackend::GrowingOnlyGeneral;
        }
        if (view != nullptr &&
            sealed_postings != nullptr &&
            growing == nullptr &&
            view->base_dead_prefix_end == 0) {
            return Phase4PostingBackend::SealedOverlay;
        }
        return Phase4PostingBackend::GeneralOverlay;
    }

    CandidateSelectionStrategy candidate_selection_strategy() const {
        if (!reader_view_) {
            throw std::runtime_error("candidate selection requested before reader view load");
        }
        return reader_view_->candidate_selection_strategy;
    }

    double select_phase5_candidates(size_t effective_candidates,
                                    Timer& phase_timer,
                                    std::vector<int>& candidates) {
        double candidate_dead_clear_us = 0.0;
        switch (candidate_selection_strategy()) {
        case CandidateSelectionStrategy::BaseAllLive:
        case CandidateSelectionStrategy::UpdateAllLive:
        case CandidateSelectionStrategy::DeadScoresFiltered:
            sc_score_->select_top_n_into(effective_candidates, nullptr, candidates);
            break;
        case CandidateSelectionStrategy::LiveSuffixRange:
            sc_score_->select_top_n_range_into(
                effective_candidates,
                reader_view_->live_suffix_begin,
                reader_view_->max_pid,
                nullptr,
                candidates);
            break;
        case CandidateSelectionStrategy::DeadPidClear:
            sc_score_->clear_scores_for_valid_pids(*reader_view_->selection_dead_pids);
            if (config_.query_profile) {
                candidate_dead_clear_us += phase_timer.elapsed_us();
                profile_stats_.candidate_dead_clear_pids +=
                    reader_view_->selection_dead_pids->size();
                phase_timer.reset();
            }
            sc_score_->select_top_n_into(effective_candidates, nullptr, candidates);
            break;
        case CandidateSelectionStrategy::LiveBitset:
            sc_score_->select_top_n_live_into(
                effective_candidates, *reader_view_->live, nullptr, candidates);
            break;
        }
        return candidate_dead_clear_us;
    }

    PostingLists::CellSpan base_posting_cell_span(int m, int cell) const {
        if (reader_view_ && reader_view_->base_live_postings) {
            BasePostingView::CellSpan span =
                reader_view_->base_live_postings->cell_span(m, cell);
            return {span.data, span.count};
        }
        return shared_data_->posting_lists->cell_span(m, cell);
    }

    size_t posting_cell_query_size(const uint32_t* ids, size_t count) const {
        if (!composite_cell_size_needs_live_count()) return count;
        return impl::count_live_posting_ids(ids, count, reader_view_->live.get());
    }

    static size_t base_cell_size_for_budget(const BaseSegment& shared_data,
                                            const ReaderView* view,
                                            int m,
                                            int cell) {
        if (view != nullptr && view->base_live_postings) {
            return view->base_live_postings->cell_size(m, cell);
        }
        return shared_data.posting_lists->cell_size(m, cell);
    }

    static size_t sealed_delta_cell_size_for_budget(const SealedDeltaSegment& segment,
                                                    uint32_t subspace,
                                                    uint32_t cell,
                                                    const LiveTable* live) {
        size_t count = segment.cell_size(subspace, cell);
        if (count == 0 || segment.live_count() == segment.row_count()) return count;
        if (segment.has_live_cell_counts()) return segment.live_cell_size(subspace, cell);
        if (live == nullptr) {
            throw std::runtime_error("sealed delta cell budget requires live table");
        }
        return impl::count_live_posting_ids(
            segment.cell_data(subspace, cell), count, live);
    }

    static size_t growing_delta_cell_size_for_budget(const UnsealedDeltaView& growing,
                                                     uint32_t subspace,
                                                     uint32_t cell,
                                                     const LiveTable* live) {
        size_t count = growing.cell_size(subspace, cell);
        if (count == 0 || growing.all_pids_live()) return count;
        if (growing.has_live_cell_counts()) return growing.live_cell_size(subspace, cell);
        if (live == nullptr) {
            throw std::runtime_error("growing delta cell budget requires live table");
        }
        const uint32_t* data = growing.cell_data(subspace, cell);
        if (data == nullptr) {
            throw std::runtime_error("growing delta layered cell budget requires live counts");
        }
        return impl::count_live_posting_ids(data, count, live);
    }

    size_t composite_cell_size(int m, int cell) const {
        // Fast path: flat uniform-K composite cache (single deref, no bounds chain).
        if (composite_cell_K_ != 0 && m >= 0 && cell >= 0) {
            const size_t idx = (size_t)m * composite_cell_K_ + (size_t)cell;
            if (idx < composite_cell_sizes_flat_.size()) {
                return (size_t)composite_cell_sizes_flat_[idx];
            }
        }
        if (reader_view_ &&
            (size_t)m < reader_view_->composite_cell_sizes.size()) {
            const std::vector<uint32_t>& sizes =
                reader_view_->composite_cell_sizes[(size_t)m];
            if (cell >= 0 && (size_t)cell < sizes.size()) {
                return (size_t)sizes[(size_t)cell];
            }
        }
        size_t base_count =
            base_cell_size_for_budget(*shared_data_, reader_view_.get(), m, cell);
        if (query_growing_only()) {
            const UnsealedDeltaView* growing = reader_view_->growing_delta.get();
            return base_count + growing_delta_cell_size_for_budget(
                *growing, (uint32_t)m, (uint32_t)cell, reader_view_->live.get());
        }
        size_t total = base_count;
        if (reader_view_) {
            if (reader_view_->sealed_delta_postings) {
                total += reader_view_->sealed_delta_postings->cell_size(
                    (uint32_t)m, (uint32_t)cell);
            } else {
                for (const auto& segment : reader_view_->sealed_deltas) {
                    if (!segment) continue;
                    total += sealed_delta_cell_size_for_budget(
                        *segment, (uint32_t)m, (uint32_t)cell, reader_view_->live.get());
                }
            }
        }
        if (reader_view_ && reader_view_->growing_delta) {
            total += growing_delta_cell_size_for_budget(
                *reader_view_->growing_delta,
                (uint32_t)m,
                (uint32_t)cell,
                reader_view_->live.get());
        }
        return total;
    }

    Phase4CellRef make_phase4_cell_ref(int m, int cell, uint8_t weight) {
        Phase4CellRef ref;
        ref.subspace = m;
        ref.cell = cell;
        ref.weight = weight;

        PostingLists::CellSpan base_span = base_posting_cell_span(m, cell);
        ref.base_data = base_span.data;
        ref.base_size = checked_phase4_span_count(base_span.count);

        if (!reader_view_) return ref;

        if (reader_view_->sealed_delta_postings) {
            SealedDeltaPostingView::CellSpan delta_span =
                reader_view_->sealed_delta_postings->cell_span(
                    (uint32_t)m, (uint32_t)cell);
            ref.sealed_delta_data = delta_span.data;
            ref.sealed_delta_size = checked_phase4_span_count(delta_span.count);
            phase4_delta_posting_count_ += ref.sealed_delta_size;
        }

        if (reader_view_->growing_delta && !reader_view_->growing_delta->empty()) {
            UnsealedDeltaView::CellSpan delta_span =
                reader_view_->growing_delta->cell_span((uint32_t)m, (uint32_t)cell);
            ref.growing_delta_data = delta_span.data;
            ref.growing_delta_size = delta_span.count != 0
                ? checked_phase4_span_count(delta_span.count)
                : checked_phase4_span_count(
                      reader_view_->growing_delta->cell_size(
                          (uint32_t)m, (uint32_t)cell));
            phase4_delta_posting_count_ += ref.growing_delta_size;
        }

        return ref;
    }

    bool trim_active_cells_to_entry_budget(int m,
                                           size_t target_entries,
                                           std::vector<std::pair<float, int>>& cells) const {
        if (target_entries == 0) {
            cells.clear();
            return true;
        }
        size_t trimmed_entries = 0;
        size_t trim_idx = 0;
        for (; trim_idx < cells.size(); ++trim_idx) {
            trimmed_entries += composite_cell_size(m, cells[trim_idx].second);
            if (trimmed_entries >= target_entries) break;
        }
        if (trim_idx + 1 < cells.size()) cells.resize(trim_idx + 1);
        return trimmed_entries >= target_entries;
    }

    void full_fine_entry_budget_activation(int m,
                                           const float* query_subspace,
                                           size_t target_entries,
                                           size_t k_total,
                                           std::vector<std::pair<float, int>>& result) {
        size_t q25_entries = ((size_t)m < shared_data_->cell_size_q25_entries.size())
            ? shared_data_->cell_size_q25_entries[(size_t)m]
            : 0;
        int top_l = 1;
        if (q25_entries > 0) {
            top_l = (int)((target_entries + q25_entries - 1) / q25_entries);
        } else {
            top_l = (int)std::ceil(config_.collision_ratio * (float)k_total);
        }
        top_l = std::min((int)k_total, std::max(1, top_l));
        while (true) {
            shared_data_->ivf_indices[m].dynamic_activation(query_subspace, top_l, result);
            bool budget_met = trim_active_cells_to_entry_budget(m, target_entries, result);
            if (budget_met || top_l >= (int)k_total) break;
            top_l = std::min((int)k_total, std::max(top_l + 1, top_l * 2));
        }
    }

    void validate_subspace_index_compatibility(const BaseSegment& shared, int M) const {
        if (!shared.posting_lists) throw std::runtime_error("Posting lists are not loaded");
        for (int m = 0; m < M; ++m) {
            if ((size_t)m >= shared.ivf_indices.size()) {
                throw std::runtime_error("Missing IVF subspace index");
            }
            int posting_k = shared.posting_lists->K(m);
            size_t expected_k = shared.ivf_indices[(size_t)m].k_total();
            if (expected_k > (size_t)std::numeric_limits<int>::max() || posting_k != (int)expected_k) {
                throw std::runtime_error("Subspace index/posting K mismatch at subspace " + std::to_string(m));
            }
        }
    }

    void build_cell_size_quantiles(BaseSegment& shared, int M) const {
        if (!shared.posting_lists) throw std::runtime_error("Posting lists are not loaded");
        shared.cell_size_q25_entries.assign((size_t)M, 0);
        for (int m = 0; m < M; ++m) {
            int K = shared.posting_lists->K(m);
            std::vector<size_t> sizes;
            sizes.reserve((size_t)K);
            for (int cell = 0; cell < K; ++cell) {
                size_t cell_entries = shared.posting_lists->cell_size(m, cell);
                if (cell_entries > 0) sizes.push_back(cell_entries);
            }
            if (sizes.empty()) continue;
            size_t qidx = (sizes.size() - 1) / 4;
            std::nth_element(sizes.begin(), sizes.begin() + (std::ptrdiff_t)qidx, sizes.end());
            shared.cell_size_q25_entries[(size_t)m] = sizes[qidx];
        }
    }

    const float* prepare_subspace_query(const float* query) {
        if (!rotate_subspace_) return query;
        if (subspace_rotation_) {
            const size_t padded_dim = (size_t)subspace_rotation_->padded_dim();
            if (q_sub_rot_.size() < padded_dim) q_sub_rot_.resize(padded_dim);
            subspace_rotation_->apply(query, q_sub_rot_.data());
            return q_sub_rot_.data();
        }
        return q_rot_.data();
    }

    void run_flat_search_path(const float* subspace_query,
                              size_t target_per_sub,
                              std::vector<int>& candidates) {
        int M = config_.num_subspaces;
        int sub_dim = shared_data_->ivf_indices.empty()
            ? dim_ / M
            : shared_data_->ivf_indices[0].subspace_dim();
        Timer phase_timer;
        if (config_.query_profile) phase_timer.reset();
        for (int m = 0; m < M; m++) {
            full_fine_entry_budget_activation(
                m,
                subspace_query + (size_t)m * sub_dim,
                target_per_sub,
                shared_data_->ivf_indices[(size_t)m].k_total(),
                active_cells_[m]);
        }
        if (config_.query_profile) {
            profile_stats_.activation_us += phase_timer.elapsed_us();
        }

        phase4_uniform_targets_.assign((size_t)M, target_per_sub);
        run_phase45_from_active_cells(phase4_uniform_targets_, candidates);
    }

    void run_phase45_from_active_cells(const std::vector<size_t>& target_per_subspace,
                                       std::vector<int>& candidates) {
        Timer phase_timer;
        if (config_.query_profile) phase_timer.reset();
        sc_score_->reset();
        if (config_.query_profile) {
            profile_stats_.score_reset_us += phase_timer.elapsed_us();
            phase_timer.reset();
        }
        int M = config_.num_subspaces;
        int max_w_int = std::min(std::max(1, config_.max_w), 255);
        phase4_cells_.clear();
        phase4_delta_posting_count_ = 0;

        for (int m = 0; m < M; ++m) {
            const size_t target_per_sub = target_per_subspace[(size_t)m];
            std::array<size_t, 256> bounds{};
            for (int k = 0; k <= max_w_int; ++k) bounds[(size_t)k] = k * target_per_sub / (size_t)max_w_int;
            int total_cells = (int)active_cells_[m].size();
            size_t sub_entries = 0;
            for (int c = 0; c < total_cells; ++c) {
                int cell = active_cells_[(size_t)m][(size_t)c].second;
                if (sub_entries >= target_per_sub) break;
                size_t cell_sz = composite_cell_size(m, cell);
                if (cell_sz == 0) continue;
                uint8_t w = 1;
                if (max_w_int == 3) {
                    w = (sub_entries < bounds[1]) ? 3 : (sub_entries < bounds[2] ? 2 : 1);
                } else {
                    for (int k = 0; k < max_w_int; ++k) {
                        if (sub_entries < bounds[(size_t)k + 1]) {
                            w = (uint8_t)std::max(1, max_w_int - k);
                            break;
                        }
                    }
                }
                sub_entries += cell_sz;
                phase4_cells_.push_back(make_phase4_cell_ref(m, cell, w));
            }
        }
        if (config_.query_profile) {
            profile_stats_.cell_schedule_us += phase_timer.elapsed_us();
            profile_stats_.phase4_cells += phase4_cells_.size();
            phase_timer.reset();
        }
        run_phase4_posting_major_backend();
        if (config_.query_profile) {
            profile_stats_.posting_score_us += phase_timer.elapsed_us();
            phase_timer.reset();
        }

        size_t effective_candidates = effective_candidate_budget();
        double candidate_dead_clear_us =
            select_phase5_candidates(effective_candidates, phase_timer, candidates);
        if (config_.query_profile) {
            double candidate_topn_us = phase_timer.elapsed_us();
            profile_stats_.candidate_dead_clear_us += candidate_dead_clear_us;
            profile_stats_.candidate_topn_us += candidate_topn_us;
            profile_stats_.candidate_select_us += candidate_dead_clear_us + candidate_topn_us;
            profile_stats_.selected_candidates += candidates.size();
        }
    }

    void increment_delta_posting_cell(const uint32_t* data,
                                      size_t count,
                                      size_t live_count,
                                      uint8_t weight) {
        if (count == 0 || live_count == 0) return;
        if (reader_view_ && reader_view_->live &&
            live_count < count && live_count <= count / 4) {
            sc_score_->increment_weighted_live(
                data, count, weight, *reader_view_->live, config_.phase4_score_prefetch);
            return;
        }
        sc_score_->increment_weighted(data, count, weight, config_.phase4_score_prefetch);
    }

    void append_delta_score_buffer(const uint32_t* data, size_t count, uint8_t weight) {
        if (count == 0) return;
        std::vector<uint32_t>& buffer = delta_score_buffers_[(size_t)weight];
        if (buffer.empty()) delta_score_touched_weights_.push_back(weight);
        const size_t old_size = buffer.size();
        buffer.resize(old_size + count);
        std::memcpy(buffer.data() + old_size, data, count * sizeof(uint32_t));
    }

    void score_delta_posting_cell(const uint32_t* data,
                                  size_t count,
                                  size_t live_count,
                                  uint8_t weight,
                                  bool use_buffer_when_all_live) {
        if (count == 0 || live_count == 0) return;
        if (config_.query_profile) ++profile_stats_.delta_posting_cells;
        bool inline_small_span =
            live_count == count &&
            count <= kInlineSmallDeltaPostings;
        if (inline_small_span) {
            if (config_.query_profile) {
                ++profile_stats_.inline_delta_posting_cells;
                profile_stats_.inline_delta_posting_ids += count;
            }
            uint8_t* __restrict__ scores = sc_score_->mutable_scores().data();
            size_t i = 0;
            for (; i + 3 < count; i += 4) {
                scores[data[i]] += weight;
                scores[data[i + 1]] += weight;
                scores[data[i + 2]] += weight;
                scores[data[i + 3]] += weight;
            }
            for (; i < count; ++i) scores[data[i]] += weight;
            return;
        }
        if (use_buffer_when_all_live && live_count == count) {
            if (config_.query_profile) {
                ++profile_stats_.buffered_delta_posting_cells;
                profile_stats_.buffered_delta_posting_ids += count;
            }
            append_delta_score_buffer(data, count, weight);
            return;
        }
        increment_delta_posting_cell(data, count, live_count, weight);
    }

    void score_growing_delta_cell(const UnsealedDeltaView* growing,
                                  const Phase4CellRef& ref,
                                  bool growing_all_live,
                                  bool use_delta_score_buffers) {
        if (growing == nullptr || ref.growing_delta_size == 0) return;
        const uint32_t subspace = (uint32_t)ref.subspace;
        const uint32_t cell = (uint32_t)ref.cell;
        if (ref.growing_delta_data != nullptr) {
            size_t delta_live_size = ref.growing_delta_size;
            if (!growing_all_live) {
                delta_live_size = growing->has_live_cell_counts()
                    ? growing->live_cell_size(subspace, cell)
                    : posting_cell_query_size(ref.growing_delta_data, ref.growing_delta_size);
            }
            score_delta_posting_cell(
                ref.growing_delta_data,
                ref.growing_delta_size,
                delta_live_size,
                ref.weight,
                use_delta_score_buffers);
            if (config_.query_profile) {
                profile_stats_.delta_posting_ids += ref.growing_delta_size;
            }
            return;
        }
        size_t aggregate_live = ref.growing_delta_size;
        if (!growing_all_live) {
            aggregate_live = growing->has_live_cell_counts()
                ? growing->live_cell_size(subspace, cell)
                : ref.growing_delta_size;
            if (aggregate_live == 0) return;
        }
        const bool aggregate_all_live = growing_all_live || aggregate_live == ref.growing_delta_size;
        growing->for_each_cell_span(subspace, cell, [&](UnsealedDeltaView::CellSpan span) {
            size_t span_live = span.count;
            if (!aggregate_all_live) {
                span_live = posting_cell_query_size(span.data, span.count);
                if (span_live == 0) return;
            }
            score_delta_posting_cell(
                span.data, span.count, span_live, ref.weight, use_delta_score_buffers);
            if (config_.query_profile) profile_stats_.delta_posting_ids += span.count;
        });
    }

    void prefetch_next_phase4_base_cell(size_t ref_index) const {
        if (ref_index + 1 >= phase4_cells_.size()) return;
        const Phase4CellRef& next = phase4_cells_[ref_index + 1];
        if (next.base_size > 0) {
            __builtin_prefetch(next.base_data, 0, 0);
        }
    }

    // Prefetch the NEXT cell's delta spans one iteration early. Delta posting
    // arrays live outside the hot hugepage-backed base region, so each cell's
    // first cacheline is cold; warming it ahead (mirroring the base prefetch)
    // hides the latency. The previous code prefetched the CURRENT cell's span,
    // which was loaded a few instructions later and therefore hid nothing.
    void prefetch_next_phase4_delta_span(size_t ref_index) const {
        if (ref_index + 1 >= phase4_cells_.size()) return;
        const Phase4CellRef& next = phase4_cells_[ref_index + 1];
        if (next.sealed_delta_size != 0 && next.sealed_delta_data != nullptr) {
            __builtin_prefetch(next.sealed_delta_data, 0, 0);
        }
        if (next.growing_delta_size != 0 && next.growing_delta_data != nullptr) {
            __builtin_prefetch(next.growing_delta_data, 0, 0);
        }
    }

    void score_phase4_base_cell(const Phase4CellRef& ref, bool profile) {
        if (ref.base_size == 0) return;
        sc_score_->increment_weighted(
            ref.base_data, ref.base_size, ref.weight, config_.phase4_score_prefetch);
        if (profile) {
            profile_stats_.base_posting_ids += ref.base_size;
        }
    }

    void score_phase4_sealed_overlay_cell(const Phase4CellRef& ref,
                                          bool profile,
                                          bool use_delta_score_buffers) {
        if (ref.sealed_delta_size == 0) return;
        score_delta_posting_cell(
            ref.sealed_delta_data,
            ref.sealed_delta_size,
            ref.sealed_delta_size,
            ref.weight,
            use_delta_score_buffers);
        if (profile) profile_stats_.delta_posting_ids += ref.sealed_delta_size;
    }

    void score_phase4_sealed_segment_cells(const ReaderView& view,
                                           const Phase4CellRef& ref,
                                           bool profile,
                                           bool use_delta_score_buffers) {
        const uint32_t subspace = (uint32_t)ref.subspace;
        const uint32_t cell = (uint32_t)ref.cell;
        for (const auto& segment : view.sealed_deltas) {
            if (!segment) continue;
            size_t delta_size = segment->cell_size(subspace, cell);
            if (delta_size == 0) continue;
            const uint32_t* data = segment->cell_data(subspace, cell);
            size_t delta_live_size = delta_size;
            if (segment->live_count() != segment->row_count()) {
                delta_live_size = segment->has_live_cell_counts()
                    ? segment->live_cell_size(subspace, cell)
                    : posting_cell_query_size(data, delta_size);
            }
            score_delta_posting_cell(
                data, delta_size, delta_live_size, ref.weight, use_delta_score_buffers);
            if (profile) profile_stats_.delta_posting_ids += delta_size;
        }
    }

    void score_phase4_growing_overlay_cell(const UnsealedDeltaView* growing,
                                           const Phase4CellRef& ref,
                                           bool profile,
                                           bool use_delta_score_buffers) {
        if (ref.growing_delta_size == 0) return;
        if (ref.growing_delta_data != nullptr) {
            score_delta_posting_cell(
                ref.growing_delta_data,
                ref.growing_delta_size,
                ref.growing_delta_size,
                ref.weight,
                use_delta_score_buffers);
            if (profile) profile_stats_.delta_posting_ids += ref.growing_delta_size;
            return;
        }
        score_growing_delta_cell(growing, ref, true, use_delta_score_buffers);
    }

    void flush_delta_score_buffers() {
        for (uint8_t weight : delta_score_touched_weights_) {
            std::vector<uint32_t>& buffer = delta_score_buffers_[(size_t)weight];
            if (!buffer.empty()) {
                sc_score_->increment_weighted(
                    buffer.data(), buffer.size(), weight, config_.phase4_score_prefetch);
                buffer.clear();
            }
        }
        delta_score_touched_weights_.clear();
    }

    void run_phase4_posting_major_backend() {
        const bool profile = config_.query_profile;
        const ReaderView* view = query_has_updates() ? reader_view_.get() : nullptr;
        const SealedDeltaPostingView* sealed_postings =
            (view && view->sealed_delta_postings)
                ? view->sealed_delta_postings.get()
                : nullptr;
        const UnsealedDeltaView* growing =
            (view && view->growing_delta && !view->growing_delta->empty())
                ? view->growing_delta.get()
                : nullptr;
        switch (phase4_posting_backend(view, sealed_postings, growing)) {
        case Phase4PostingBackend::SealedOverlay:
            run_phase4_posting_major_backend_sealed_overlay();
            return;
        case Phase4PostingBackend::GrowingOnlyGeneral:
            run_phase4_posting_major_backend_growing_only(growing);
            return;
        case Phase4PostingBackend::GrowingOverlay:
            run_phase4_posting_major_backend_growing_overlay(growing);
            return;
        case Phase4PostingBackend::GeneralOverlay:
            break;
        }
        const bool use_delta_score_buffers = !should_score_delta_directly();
        const bool growing_all_live = growing == nullptr || growing->all_pids_live();
        delta_score_touched_weights_.clear();
        for (size_t i = 0; i < phase4_cells_.size(); ++i) {
            const Phase4CellRef& ref = phase4_cells_[i];
            prefetch_next_phase4_base_cell(i);
            prefetch_next_phase4_delta_span(i);
            score_phase4_base_cell(ref, profile);
            if (view != nullptr) {
                if (sealed_postings != nullptr) {
                    score_phase4_sealed_overlay_cell(
                        ref, profile, use_delta_score_buffers);
                } else {
                    score_phase4_sealed_segment_cells(
                        *view, ref, profile, use_delta_score_buffers);
                }
            }
            if (growing != nullptr) {
                score_growing_delta_cell(
                    growing, ref, growing_all_live, use_delta_score_buffers);
            }
        }
        flush_delta_score_buffers();
    }

    void run_phase4_posting_major_backend_sealed_overlay() {
        const bool profile = config_.query_profile;
        const bool use_delta_score_buffers = !should_score_delta_directly();
        delta_score_touched_weights_.clear();
        for (size_t i = 0; i < phase4_cells_.size(); ++i) {
            const Phase4CellRef& ref = phase4_cells_[i];
            prefetch_next_phase4_base_cell(i);
            prefetch_next_phase4_delta_span(i);
            score_phase4_base_cell(ref, profile);
            score_phase4_sealed_overlay_cell(ref, profile, use_delta_score_buffers);
        }
        flush_delta_score_buffers();
    }

    void run_phase4_posting_major_backend_growing_only(const UnsealedDeltaView* growing) {
        const bool profile = config_.query_profile;
        const bool growing_all_live = growing == nullptr || growing->all_pids_live();
        const bool use_delta_score_buffers = !should_score_delta_directly();
        delta_score_touched_weights_.clear();
        for (size_t i = 0; i < phase4_cells_.size(); ++i) {
            const Phase4CellRef& ref = phase4_cells_[i];
            prefetch_next_phase4_base_cell(i);
            prefetch_next_phase4_delta_span(i);
            score_phase4_base_cell(ref, profile);
            score_growing_delta_cell(
                growing, ref, growing_all_live, use_delta_score_buffers);
        }
        flush_delta_score_buffers();
    }

    void run_phase4_posting_major_backend_growing_overlay(
        const UnsealedDeltaView* growing) {
        (void)growing;
        const bool profile = config_.query_profile;
        const bool use_delta_score_buffers = !should_score_delta_directly();
        delta_score_touched_weights_.clear();
        for (size_t i = 0; i < phase4_cells_.size(); ++i) {
            const Phase4CellRef& ref = phase4_cells_[i];
            prefetch_next_phase4_base_cell(i);
            prefetch_next_phase4_delta_span(i);
            score_phase4_base_cell(ref, profile);
            score_phase4_growing_overlay_cell(
                growing, ref, profile, use_delta_score_buffers);
        }
        flush_delta_score_buffers();
    }

    void load_rotations(const std::string& index_path, bool verbose) {
        rotation_ = std::make_unique<RandomRotation>();
        rotation_->load(index_path + "/rotation.bin");
        dim_ = rotation_->dim();

        rotate_subspace_ = false;
        subspace_rotation_.reset();
        std::ifstream flag(index_path + "/rotate_subspace.flag");
        if (!flag.good()) return;

        rotate_subspace_ = true;
        std::ifstream sub_rot_check(index_path + "/subspace_rotation.bin", std::ios::binary);
        if (sub_rot_check.good()) {
            sub_rot_check.close();
            subspace_rotation_ = std::make_unique<RandomRotation>();
            subspace_rotation_->load(index_path + "/subspace_rotation.bin");
            if (subspace_rotation_->dim() != dim_) {
                throw std::runtime_error("subspace_rotation.bin dimension does not match rotation.bin");
            }
            if (subspace_rotation_->padded_dim() < dim_) {
                throw std::runtime_error("subspace_rotation.bin padded dimension is smaller than input dimension");
            }
            if (verbose) {
                std::cout << "  [Subspace] External QR rotation loaded for subspace matching" << std::endl;
            }
        } else if (verbose) {
            std::cout << "  [Subspace] Rotation enabled for subspace matching (using RaBitQ rotation)" << std::endl;
        }
    }

    void preallocate_query_buffers(bool clear_score_buffer) {
        if (!shared_data_) return;
        int M = config_.num_subspaces;
        size_t score_points = reader_view_ ? reader_view_->max_pid : (size_t)n_;
        if (clear_score_buffer) {
            sc_score_->init(score_points, M);
        } else {
            sc_score_->init_uninitialized(score_points, M);
        }
        q_rot_.resize(shared_data_->rabitq_padded_dim, 0.0f);
        if (rotate_subspace_ && subspace_rotation_) {
            q_sub_rot_.resize((size_t)subspace_rotation_->padded_dim());
        }
        active_cells_.resize(M);
        size_t phase4_cell_reserve = 0;
        for (int m = 0; m < M; ++m) {
            int K = shared_data_->posting_lists ? shared_data_->posting_lists->K(m) : 0;
            size_t active_cell_reserve = (size_t)std::max(1, (int)(config_.collision_ratio * std::max(0, K)));
            active_cells_[(size_t)m].reserve(active_cell_reserve);
            phase4_cell_reserve += active_cell_reserve;
        }
        phase4_uniform_targets_.resize((size_t)M);
        phase4_cells_.reserve(phase4_cell_reserve);
        g_add_.resize(shared_data_->rabitq_k);
        g_err_.resize(shared_data_->rabitq_k);
        if (clear_score_buffer || rabitq_gadd_stamp_.size() != (size_t)shared_data_->rabitq_k) {
            rabitq_gadd_stamp_.assign(shared_data_->rabitq_k, 0u);
            rabitq_gadd_stamp_value_ = 0;
        }
        size_t est_candidates = std::max((size_t)1, (size_t)(config_.candidate_ratio * active_point_count()));
        phase5_candidates_.reserve(est_candidates);
        ranked_.resize(est_candidates);
        delta_candidate_record_refs_.reserve(est_candidates);
        size_t verify_reserve = std::max(est_candidates, (size_t)config_.topk);
        top_pids_.reserve(verify_reserve);
        verify_results_.reserve(verify_reserve);
        query_bitplane_.resize((size_t)(shared_data_->rabitq_padded_dim / 64) * 4);
        phase6a_pack_words_.resize((size_t)(shared_data_->rabitq_padded_dim / 64) * 8);
        delta_score_buffers_.resize(256);
        delta_score_touched_weights_.reserve((size_t)std::max(1, config_.max_w));
    }

    void attach_shared_data(std::shared_ptr<BaseSegment> shared_data,
                            const std::string& index_path,
                            bool verbose,
                            bool load_rotation_state,
                            std::shared_ptr<const ReaderView> reader_view = nullptr) {
        shared_data_ = std::move(shared_data);
        index_path_ = index_path;
        dim_ = shared_data_->dim;
        if (load_rotation_state) {
            load_rotations(index_path_, verbose);
        }
        install_reader_view(
            reader_view ? std::move(reader_view) : make_base_reader_view(shared_data_),
            true);
    }

    void install_reader_view(std::shared_ptr<const ReaderView> reader_view,
                             bool clear_score_buffer) {
        if (!reader_view) throw std::runtime_error("cannot install an empty QueryEngine reader view");
        reader_view_ = std::move(reader_view);
        if (reader_view_->max_pid > (uint32_t)std::numeric_limits<int>::max()) {
            throw std::runtime_error("QueryEngine v1 cannot address max_pid above INT_MAX");
        }
        n_ = (int)reader_view_->max_pid;
        active_point_count_ = reader_view_->live_count;
        query_has_updates_ =
            reader_view_->has_deltas() ||
            reader_view_->max_pid != reader_view_->base_n ||
            reader_view_->live_count != reader_view_->max_pid;
        live_filter_needed_ =
            reader_view_->live && reader_view_->live_count < reader_view_->max_pid;
        query_growing_only_ =
            reader_view_->growing_delta &&
            !reader_view_->growing_delta->empty() &&
            reader_view_->sealed_deltas.empty() &&
            !reader_view_->sealed_delta_postings;
        bind_composite_cell_sizes_flat();
        thread_engines_.clear();
        preallocate_query_buffers(clear_score_buffer);
    }

    // Flatten the ReaderView's composite_cell_sizes into one contiguous array
    // when all subspaces share the same K, so the per-cell hot lookup in
    // composite_cell_size() is a single deref instead of two vector
    // bounds-checks + derefs. No-op for base-only or non-uniform layouts (they
    // keep the original fallback path).
    void bind_composite_cell_sizes_flat() {
        composite_cell_sizes_flat_.clear();
        composite_cell_K_ = 0;
        if (!reader_view_) return;
        const std::vector<std::vector<uint32_t>>& ccs =
            reader_view_->composite_cell_sizes;
        if (ccs.empty()) return;
        const size_t M = ccs.size();
        const size_t K = ccs[0].size();
        if (K == 0) return;
        for (const std::vector<uint32_t>& v : ccs) {
            if (v.size() != K) return;  // non-uniform: keep the fallback path
        }
        composite_cell_sizes_flat_.resize(M * K);
        for (size_t m = 0; m < M; ++m) {
            std::copy(ccs[m].begin(), ccs[m].end(),
                      composite_cell_sizes_flat_.begin() + (size_t)m * K);
        }
        composite_cell_K_ = K;
    }

    int configured_num_threads() const {
        return config_.num_threads > 0 ? config_.num_threads : omp_get_max_threads();
    }

    static bool reader_view_has_updates(const ReaderView& view) {
        return view.has_deltas() ||
               view.max_pid != view.base_n ||
               view.live_count != view.max_pid;
    }

    static CandidateSelectionStrategy choose_candidate_selection_strategy(
        const ReaderView& view) {
        if (!reader_view_has_updates(view)) {
            return CandidateSelectionStrategy::BaseAllLive;
        }
        if (!view.live) {
            throw std::runtime_error("update-aware reader view missing live table");
        }
        if (view.live_count == view.max_pid) {
            return CandidateSelectionStrategy::UpdateAllLive;
        }
        if (view.live_suffix_begin < view.max_pid &&
            view.live_count == view.max_pid - view.live_suffix_begin) {
            return CandidateSelectionStrategy::LiveSuffixRange;
        }
        if (view.dead_scores_filtered_for_selection) {
            return CandidateSelectionStrategy::DeadScoresFiltered;
        }
        if (view.selection_dead_pids && !view.selection_dead_pids->empty()) {
            return CandidateSelectionStrategy::DeadPidClear;
        }
        return CandidateSelectionStrategy::LiveBitset;
    }

    static std::shared_ptr<const ReaderView> make_base_reader_view(
        const std::shared_ptr<BaseSegment>& shared_data) {
        if (!shared_data) throw std::runtime_error("Cannot create ReaderView without base segment");
        if (shared_data->n < 0 ||
            (uint64_t)shared_data->n > (uint64_t)std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("Base segment size exceeds v1 uint32 PID range");
        }
        auto view = std::make_shared<ReaderView>();
        view->generation = 0;
        view->base = shared_data;
        view->base_n = (uint32_t)shared_data->n;
        view->max_pid = view->base_n;
        view->live_count = view->base_n;
        view->base_live_count = view->base_n;
        view->base_live_fingerprint =
            mix_base_live_fingerprint(0xcbf29ce484222325ull, view->base_n);
        view->live_suffix_begin = 0;
        view->dead_scores_filtered_for_selection = true;
        view->live = LiveTable::all_live(view->max_pid);
        view->candidate_selection_strategy =
            choose_candidate_selection_strategy(*view);
        return view;
    }

    static std::string join_index_path(const std::string& index_path, const std::string& relative_or_absolute) {
        if (relative_or_absolute.empty()) return relative_or_absolute;
        if (relative_or_absolute[0] == '/') return relative_or_absolute;
        return index_path + "/" + relative_or_absolute;
    }

    static size_t sparse_dead_pid_limit(uint32_t max_pid) {
        constexpr size_t kMaxSparseDeadPids = 65536;
        constexpr size_t kSparseDeadFractionDivisor = 16;
        size_t fraction_limit = std::max<size_t>(1, (size_t)max_pid / kSparseDeadFractionDivisor);
        return std::min(kMaxSparseDeadPids, fraction_limit);
    }

    static size_t parse_count_env_or_default(const char* name,
                                             size_t default_value,
                                             size_t max_value) {
        const char* env = std::getenv(name);
        if (env == nullptr || env[0] == '\0') return default_value;
        char* end = nullptr;
        unsigned long long value = std::strtoull(env, &end, 10);
        if (end == env || (end != nullptr && *end != '\0')) return default_value;
        return (size_t)std::min<unsigned long long>(
            value, (unsigned long long)max_value);
    }

    static size_t base_live_posting_stale_dead_limit(uint32_t base_n) {
        size_t scaled = (size_t)base_n / 512;
        size_t default_value = std::min<size_t>(
            65536,
            std::max<size_t>(4096, scaled));
        return parse_count_env_or_default(
            "SCORE_BASE_LIVE_POSTING_STALE_DEAD_LIMIT",
            default_value,
            (size_t)base_n);
    }

    static std::shared_ptr<const std::vector<uint32_t>> build_sparse_dead_pid_list(
        const LiveTable& live) {
        uint32_t max_pid = live.max_pid();
        uint32_t live_count = live.live_count();
        if (live_count > max_pid) {
            throw std::runtime_error("LiveTable live_count exceeds max_pid");
        }
        size_t dead_count = (size_t)max_pid - (size_t)live_count;
        if (dead_count == 0) return nullptr;

        if (dead_count > sparse_dead_pid_limit(max_pid)) return nullptr;

        auto dead_pids = std::make_shared<std::vector<uint32_t>>();
        dead_pids->reserve(dead_count);
        const std::vector<uint64_t>& words = live.words();
        for (size_t word_idx = 0; word_idx < words.size(); ++word_idx) {
            uint64_t live_word = words[word_idx];
            uint64_t valid_mask = ~uint64_t{0};
            if (word_idx + 1 == words.size()) {
                uint32_t tail = max_pid & 63u;
                if (tail != 0) valid_mask = (uint64_t{1} << tail) - 1;
            }
            uint64_t dead_mask = (~live_word) & valid_mask;
            while (dead_mask) {
                uint32_t bit = (uint32_t)__builtin_ctzll(dead_mask);
                uint32_t pid = (uint32_t)(word_idx * 64) + bit;
                if (pid < max_pid) dead_pids->push_back(pid);
                dead_mask &= dead_mask - 1;
            }
        }
        if (dead_pids->size() != dead_count) {
            throw std::runtime_error("LiveTable sparse dead count mismatch");
        }
        return dead_pids;
    }

    static bool try_extend_sparse_dead_pid_list(
        const ReaderView* previous_view,
        const LiveTable& live,
        const std::vector<uint32_t>& added_dead_pids,
        std::shared_ptr<const std::vector<uint32_t>>& sparse_dead_pids) {
        if (previous_view == nullptr || added_dead_pids.empty()) return false;
        uint32_t max_pid = live.max_pid();
        uint32_t live_count = live.live_count();
        if (live_count > max_pid) {
            throw std::runtime_error("LiveTable live_count exceeds max_pid");
        }
        size_t dead_count = (size_t)max_pid - (size_t)live_count;
        if (dead_count == 0) {
            sparse_dead_pids.reset();
            return true;
        }
        if (dead_count > sparse_dead_pid_limit(max_pid)) {
            sparse_dead_pids.reset();
            return true;
        }

        size_t previous_dead_count =
            (size_t)previous_view->max_pid - (size_t)previous_view->live_count;
        if (previous_dead_count == 0) {
            if (previous_view->sparse_dead_pids &&
                !previous_view->sparse_dead_pids->empty()) {
                return false;
            }
        } else if (!previous_view->sparse_dead_pids ||
                   previous_view->sparse_dead_pids->size() != previous_dead_count) {
            return false;
        }
        if (previous_dead_count + added_dead_pids.size() != dead_count) return false;

        auto dead_pids = std::make_shared<std::vector<uint32_t>>();
        dead_pids->reserve(dead_count);
        if (previous_view->sparse_dead_pids) {
            dead_pids->insert(dead_pids->end(),
                              previous_view->sparse_dead_pids->begin(),
                              previous_view->sparse_dead_pids->end());
        }
        dead_pids->insert(dead_pids->end(), added_dead_pids.begin(), added_dead_pids.end());
        if (!std::is_sorted(dead_pids->begin(), dead_pids->end())) {
            std::sort(dead_pids->begin(), dead_pids->end());
        }
        sparse_dead_pids = dead_pids;
        return true;
    }

    static uint32_t detect_base_dead_prefix_end(const LiveTable& live, uint32_t base_n) {
        if (base_n == 0) return 0;
        uint32_t pid = 0;
        while (pid < base_n && !live.is_live(pid)) ++pid;
        if (pid == 0) return 0;
        size_t min_prefix = std::max<size_t>(4096, (size_t)base_n / 64);
        if ((size_t)pid < min_prefix) return 0;
        if (!live.range_all_live(pid, base_n)) return 0;
        return pid;
    }

    static uint32_t detect_live_suffix_begin(const LiveTable& live) {
        uint32_t max_pid = live.max_pid();
        if (max_pid == 0 || live.live_count() == max_pid) return 0;
        if (live.live_count() == 0) return max_pid;
        uint32_t pid = 0;
        while (pid < max_pid && !live.is_live(pid)) ++pid;
        if (pid == 0 || pid == max_pid) return 0;
        if (live.live_count() != max_pid - pid) return 0;
        if (!live.range_all_live(pid, max_pid)) return 0;
        return pid;
    }

    struct BaseLivePostingStats {
        uint32_t live_count = 0;
        uint64_t fingerprint = 0;
    };

    static uint64_t mix_base_live_fingerprint(uint64_t hash, uint64_t value) {
        value += 0x9e3779b97f4a7c15ull;
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
        value ^= value >> 31;
        hash ^= value;
        hash *= 0x100000001b3ull;
        return hash;
    }

    static BaseLivePostingStats compute_base_live_posting_stats(
        const LiveTable& live,
        uint32_t base_n) {
        if (base_n > live.max_pid()) {
            throw std::runtime_error("base live stats require live table to cover base_n");
        }
        BaseLivePostingStats stats;
        uint64_t hash = 0xcbf29ce484222325ull;
        const std::vector<uint64_t>& words = live.words();
        size_t full_words = (size_t)base_n / 64;
        uint32_t tail_bits = base_n & 63u;
        for (size_t i = 0; i < full_words; ++i) {
            uint64_t word = words[i];
            stats.live_count += (uint32_t)__builtin_popcountll(word);
            hash = mix_base_live_fingerprint(hash, word ^ (uint64_t)i);
        }
        if (tail_bits != 0) {
            uint64_t mask = (uint64_t{1} << tail_bits) - 1;
            uint64_t word = words[full_words] & mask;
            stats.live_count += (uint32_t)__builtin_popcountll(word);
            hash = mix_base_live_fingerprint(hash, word ^ (uint64_t)full_words);
        }
        stats.fingerprint = mix_base_live_fingerprint(hash, base_n);
        return stats;
    }

    static BaseLivePostingStats incremental_base_live_posting_stats(
        const ReaderView& previous_view,
        uint32_t base_live_deletes,
        uint64_t committed_lsn) {
        if (base_live_deletes > previous_view.base_live_count) {
            throw std::runtime_error("incremental base live stats underflow");
        }
        BaseLivePostingStats stats;
        stats.live_count = previous_view.base_live_count - base_live_deletes;
        uint64_t fingerprint = previous_view.base_live_fingerprint;
        fingerprint = mix_base_live_fingerprint(
            fingerprint,
            ((uint64_t)base_live_deletes << 32) ^ committed_lsn);
        fingerprint = mix_base_live_fingerprint(fingerprint, stats.live_count);
        stats.fingerprint = fingerprint;
        return stats;
    }

    struct BaseLivePostingState {
        std::shared_ptr<const BasePostingView> postings;
        bool exact = false;
        uint32_t overlay_base_live_count = 0;
        std::shared_ptr<const std::vector<uint32_t>> unfiltered_dead_pids;
    };

    static std::vector<uint32_t> collect_new_base_dead_pids(
        const ReaderView* previous_view,
        const std::vector<uint32_t>& committed_dead_pids,
        uint32_t base_n) {
        std::vector<uint32_t> new_base_dead;
        if (previous_view == nullptr || !previous_view->live ||
            committed_dead_pids.empty()) {
            return new_base_dead;
        }
        new_base_dead.reserve(committed_dead_pids.size());
        for (uint32_t pid : committed_dead_pids) {
            if (pid >= base_n) continue;
            if (previous_view->live->is_live(pid)) new_base_dead.push_back(pid);
        }
        return new_base_dead;
    }

    static std::shared_ptr<const std::vector<uint32_t>> append_unfiltered_dead_pids(
        const std::shared_ptr<const std::vector<uint32_t>>& previous_unfiltered,
        const std::vector<uint32_t>& new_base_dead) {
        if ((!previous_unfiltered || previous_unfiltered->empty()) &&
            new_base_dead.empty()) {
            return nullptr;
        }
        auto merged = std::make_shared<std::vector<uint32_t>>();
        size_t previous_size = previous_unfiltered ? previous_unfiltered->size() : 0;
        merged->reserve(previous_size + new_base_dead.size());
        if (previous_unfiltered) {
            merged->insert(merged->end(),
                           previous_unfiltered->begin(),
                           previous_unfiltered->end());
        }
        merged->insert(merged->end(), new_base_dead.begin(), new_base_dead.end());
        return merged;
    }

    static BaseLivePostingState maybe_build_base_live_posting_view(
        const std::shared_ptr<BaseSegment>& shared_data,
        const LiveTable& live,
        uint32_t base_n,
        int M,
        const BaseLivePostingStats& stats,
        const ReaderView* previous_view,
        const std::vector<uint32_t>& committed_dead_pids,
        bool committed_insert_reused_pid) {
        BaseLivePostingState state;
        if (!shared_data || !shared_data->posting_lists || base_n == 0) return state;
        if (stats.live_count >= base_n) return state;

        if (previous_view != nullptr &&
            previous_view->base.get() == shared_data.get() &&
            previous_view->base_n == base_n &&
            previous_view->base_live_count == stats.live_count &&
            previous_view->base_live_fingerprint == stats.fingerprint &&
            previous_view->base_live_postings) {
            state.postings = previous_view->base_live_postings;
            state.exact = previous_view->base_live_postings_exact;
            state.overlay_base_live_count =
                previous_view->base_live_postings_base_live_count;
            state.unfiltered_dead_pids =
                previous_view->base_live_postings_unfiltered_dead_pids;
            return state;
        }

        bool can_reuse_previous_overlay =
            previous_view != nullptr &&
            previous_view->base.get() == shared_data.get() &&
            previous_view->base_n == base_n &&
            previous_view->base_live_postings &&
            previous_view->live &&
            !committed_insert_reused_pid;
        if (can_reuse_previous_overlay) {
            std::vector<uint32_t> new_base_dead =
                collect_new_base_dead_pids(previous_view, committed_dead_pids, base_n);
            std::shared_ptr<const std::vector<uint32_t>> unfiltered =
                append_unfiltered_dead_pids(
                    previous_view->base_live_postings_unfiltered_dead_pids,
                    new_base_dead);
            size_t unfiltered_count = unfiltered ? unfiltered->size() : 0;
            size_t rebuild_limit = base_live_posting_stale_dead_limit(base_n);
            if (unfiltered_count == 0 || unfiltered_count < rebuild_limit) {
                state.postings = previous_view->base_live_postings;
                state.exact =
                    previous_view->base_live_postings_exact && unfiltered_count == 0;
                state.overlay_base_live_count =
                    previous_view->base_live_postings_base_live_count;
                state.unfiltered_dead_pids = unfiltered;
                return state;
            }
        }

        auto view = std::make_shared<BasePostingView>();
        view->build(*shared_data->posting_lists, M, live, base_n);
        if (!view->empty()) {
            state.postings = view;
            state.exact = true;
            state.overlay_base_live_count = stats.live_count;
        }
        return state;
    }

    static bool base_dead_scores_filtered_for_selection(
        const ReaderView& view,
        const BaseLivePostingStats& base_live_stats) {
        if (base_live_stats.live_count >= view.base_n) return true;
        if (view.base_live_postings && view.base_live_postings_exact) return true;
        if (view.base_dead_prefix_end != 0 &&
            base_live_stats.live_count == view.base_n - view.base_dead_prefix_end) {
            return true;
        }
        return false;
    }

    static std::vector<std::vector<uint32_t>> build_composite_cell_sizes(
        const std::shared_ptr<BaseSegment>& shared_data,
        const ReaderView& view,
        int M) {
        std::vector<std::vector<uint32_t>> sizes;
        if (!shared_data || !shared_data->posting_lists || M <= 0) return sizes;
        if (!view.has_deltas() && view.live_count == view.max_pid) return sizes;

        sizes.resize((size_t)M);
        for (int m = 0; m < M; ++m) {
            int K_int = shared_data->posting_lists->K(m);
            if (K_int < 0) {
                throw std::runtime_error("negative posting list cell count");
            }
            std::vector<uint32_t>& subspace_sizes = sizes[(size_t)m];
            subspace_sizes.resize((size_t)K_int, 0);
            for (int cell = 0; cell < K_int; ++cell) {
                uint64_t total = (uint64_t)base_cell_size_for_budget(
                    *shared_data, &view, m, cell);
                if (view.sealed_delta_postings) {
                    total += (uint64_t)view.sealed_delta_postings->cell_size(
                        (uint32_t)m, (uint32_t)cell);
                } else {
                    for (const auto& segment : view.sealed_deltas) {
                        if (!segment) continue;
                        total += (uint64_t)sealed_delta_cell_size_for_budget(
                            *segment, (uint32_t)m, (uint32_t)cell, view.live.get());
                    }
                }
                if (view.growing_delta) {
                    total += (uint64_t)growing_delta_cell_size_for_budget(
                        *view.growing_delta,
                        (uint32_t)m,
                        (uint32_t)cell,
                        view.live.get());
                }
                if (total > (uint64_t)std::numeric_limits<uint32_t>::max()) {
                    throw std::runtime_error("composite cell size exceeds uint32 range");
                }
                subspace_sizes[(size_t)cell] = (uint32_t)total;
            }
        }
        return sizes;
    }

    static std::vector<std::vector<uint32_t>> build_l0_block_cell_counts(
        const std::vector<std::shared_ptr<const L0Block>>& blocks,
        const std::vector<size_t>& cells_per_subspace) {
        std::vector<std::vector<uint32_t>> counts;
        counts.resize(cells_per_subspace.size());
        for (size_t m = 0; m < cells_per_subspace.size(); ++m) {
            size_t K = cells_per_subspace[m];
            counts[m].assign(K, 0);
        }
        for (const auto& block : blocks) {
            if (!block) continue;
            for (size_t m = 0; m < cells_per_subspace.size(); ++m) {
                std::vector<uint32_t>& subspace_counts = counts[m];
                if (block->has_row_cells()) {
                    for (uint32_t row = 0; row < block->row_count(); ++row) {
                        uint16_t cell = block->row_cell(row, (uint32_t)m);
                        if ((size_t)cell >= subspace_counts.size()) {
                            throw std::runtime_error("L0 tail row cell exceeds subspace width");
                        }
                        if (subspace_counts[(size_t)cell] ==
                            std::numeric_limits<uint32_t>::max()) {
                            throw std::runtime_error("L0 tail cell count exceeds uint32 range");
                        }
                        ++subspace_counts[(size_t)cell];
                    }
                } else {
                    for (size_t cell = 0; cell < subspace_counts.size(); ++cell) {
                        size_t cell_count = block->cell_size((uint32_t)m, (uint32_t)cell);
                        if (cell_count >
                            (size_t)std::numeric_limits<uint32_t>::max() -
                                (size_t)subspace_counts[cell]) {
                            throw std::runtime_error("L0 tail cell count exceeds uint32 range");
                        }
                        subspace_counts[cell] += (uint32_t)cell_count;
                    }
                }
            }
        }
        return counts;
    }

    static bool add_l0_tail_counts_to_composite(
        std::vector<std::vector<uint32_t>>& sizes,
        const std::vector<std::vector<uint32_t>>& tail_counts) {
        if (sizes.empty() || sizes.size() != tail_counts.size()) return false;
        for (size_t m = 0; m < sizes.size(); ++m) {
            if (sizes[m].size() != tail_counts[m].size()) return false;
        }
        for (size_t m = 0; m < sizes.size(); ++m) {
            for (size_t cell = 0; cell < sizes[m].size(); ++cell) {
                uint32_t add = tail_counts[m][cell];
                if (add >
                    std::numeric_limits<uint32_t>::max() - sizes[m][cell]) {
                    throw std::runtime_error("incremental composite cell size exceeds uint32 range");
                }
                sizes[m][cell] += add;
            }
        }
        return true;
    }

    static size_t matching_sealed_delta_prefix_count(
        const ReaderView* previous_view,
        const UpdateManifest& manifest) {
        if (previous_view == nullptr) return 0;
        size_t count = std::min(previous_view->sealed_deltas.size(),
                                manifest.sealed_deltas.size());
        size_t matched = 0;
        for (size_t i = 0; i < count; ++i) {
            const std::shared_ptr<const SealedDeltaSegment>& segment =
                previous_view->sealed_deltas[i];
            const UpdateManifestDelta& delta = manifest.sealed_deltas[i];
            if (!segment) break;
            if (segment->segment_id() != delta.segment_id ||
                segment->pid_begin() != delta.pid_begin ||
                segment->pid_count() != delta.pid_count ||
                segment->live_count() != delta.live_count) {
                break;
            }
            ++matched;
        }
        return matched;
    }

    static bool sealed_delta_manifest_matches_previous(
        const ReaderView* previous_view,
        const UpdateManifest& manifest) {
        return previous_view != nullptr &&
               previous_view->sealed_deltas.size() == manifest.sealed_deltas.size() &&
               matching_sealed_delta_prefix_count(previous_view, manifest) ==
                   manifest.sealed_deltas.size();
    }

    static bool dead_pids_touch_sealed_deltas(
        const std::vector<uint32_t>& dead_pids,
        const std::vector<std::shared_ptr<const SealedDeltaSegment>>& sealed_deltas) {
        if (dead_pids.empty() || sealed_deltas.empty()) return false;
        for (uint32_t pid : dead_pids) {
            for (const auto& segment : sealed_deltas) {
                if (segment && segment->contains_pid(pid)) return true;
            }
        }
        return false;
    }

    static bool dead_pids_touch_growing_delta(
        const std::vector<uint32_t>& dead_pids,
        const UnsealedDeltaView* growing) {
        if (dead_pids.empty() || growing == nullptr || growing->empty()) return false;
        UnsealedDeltaView::RowRef row;
        for (uint32_t pid : dead_pids) {
            if (growing->resolve_pid(pid, row)) return true;
        }
        return false;
    }

    struct LoadedSealedDeltas {
        std::vector<std::shared_ptr<const SealedDeltaSegment>> segments;
        size_t reusable_prefix = 0;
        bool reuse_exact_view = false;
    };

    struct InitialLiveState {
        std::shared_ptr<const LiveTable> live;
        bool use_checkpoint = false;
    };

    struct L0ReplayState {
        impl::L0WalBlockReplayResult replay;
        std::shared_ptr<const UnsealedDeltaView> growing;
        std::vector<std::vector<uint32_t>> incremental_tail_cell_counts;
        bool incremental = false;
        bool can_increment_composite_sizes = false;
    };

    static std::vector<size_t> reader_cells_per_subspace(
        const std::shared_ptr<BaseSegment>& shared_data) {
        int M = (int)shared_data->ivf_indices.size();
        if (M <= 0) throw std::runtime_error("update-aware query requires IVF subspace indices");
        std::vector<size_t> cells_per_subspace;
        cells_per_subspace.reserve((size_t)M);
        for (int m = 0; m < M; ++m) {
            cells_per_subspace.push_back((size_t)shared_data->posting_lists->K(m));
        }
        return cells_per_subspace;
    }

    static LoadedSealedDeltas load_sealed_deltas_for_reader(
        const std::shared_ptr<BaseSegment>& shared_data,
        const UpdateManifest& manifest,
        const std::string& index_path,
        const ReaderView* previous_view,
        const std::vector<size_t>& cells_per_subspace) {
        LoadedSealedDeltas loaded;
        loaded.reusable_prefix =
            previous_view != nullptr &&
                previous_view->base.get() == shared_data.get()
            ? matching_sealed_delta_prefix_count(previous_view, manifest)
            : 0;
        loaded.segments.reserve(manifest.sealed_deltas.size());
        for (size_t i = 0; i < loaded.reusable_prefix; ++i) {
            loaded.segments.push_back(previous_view->sealed_deltas[i]);
        }
        for (size_t i = loaded.reusable_prefix;
             i < manifest.sealed_deltas.size();
             ++i) {
            const UpdateManifestDelta& delta = manifest.sealed_deltas[i];
            loaded.segments.push_back(SealedDeltaSegment::load(
                delta.segment_id,
                join_index_path(index_path, delta.path),
                cells_per_subspace,
                (uint32_t)shared_data->verify_vectors_dim,
                (uint32_t)shared_data->rabitq_bin_with_assign_record_bytes_,
                delta.pid_begin,
                delta.pid_count,
                delta.live_count));
        }
        loaded.reuse_exact_view =
            previous_view != nullptr &&
            previous_view->base.get() == shared_data.get() &&
            previous_view->generation == manifest.generation &&
            previous_view->sealed_deltas.size() == manifest.sealed_deltas.size() &&
            loaded.reusable_prefix == manifest.sealed_deltas.size();
        return loaded;
    }

    static InitialLiveState load_initial_live_state(
        const UpdateManifest& manifest,
        const std::string& index_path) {
        InitialLiveState state;
        bool checkpoint_exists = false;
        if (!manifest.checkpoint_path.empty()) {
            std::ifstream checkpoint_probe(join_index_path(index_path, manifest.checkpoint_path),
                                           std::ios::binary);
            checkpoint_exists = checkpoint_probe.good();
        }
        state.use_checkpoint = !manifest.sealed_deltas.empty() || checkpoint_exists;
        if (state.use_checkpoint) {
            if (manifest.checkpoint_path.empty()) {
                throw std::runtime_error("update manifest requires live checkpoint");
            }
            state.live = LiveTable::load_binary(join_index_path(index_path, manifest.checkpoint_path));
            if (state.live->max_pid() > manifest.max_pid) {
                throw std::runtime_error("live checkpoint is ahead of update manifest");
            }
        } else {
            state.live = LiveTable::all_live(manifest.base_n);
        }
        return state;
    }

    static void validate_l0_pid_floor(const UpdateManifest& manifest) {
        uint64_t l0_pid_floor = manifest.base_n;
        for (const UpdateManifestDelta& delta : manifest.sealed_deltas) {
            l0_pid_floor = std::max(l0_pid_floor,
                                    (uint64_t)delta.pid_begin + (uint64_t)delta.pid_count);
        }
        if (l0_pid_floor > (uint64_t)std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("update manifest delta PID range exceeds uint32 range");
        }
    }

    static L0ReplayState replay_l0_for_reader(
        const std::shared_ptr<BaseSegment>& shared_data,
        const UpdateManifest& manifest,
        const std::string& index_path,
        const ReaderView* previous_view,
        const std::shared_ptr<const LiveTable>& initial_live,
        const std::vector<size_t>& cells_per_subspace) {
        L0ReplayState state;
        std::shared_ptr<const LiveTable> replay_initial_live = initial_live;
        uint64_t replay_start_lsn = 0;
        uint64_t replay_start_offset = 0;
        if (previous_view != nullptr &&
            previous_view->base.get() == shared_data.get() &&
            previous_view->generation == manifest.generation &&
            previous_view->live &&
            previous_view->growing_delta &&
            previous_view->l0_wal_replay_offset != 0) {
            state.incremental = true;
            replay_start_lsn = previous_view->l0_last_committed_lsn;
            replay_start_offset = previous_view->l0_wal_replay_offset;
            replay_initial_live = previous_view->live;
        }

        uint32_t replay_initial_max_pid = replay_initial_live->max_pid();
        size_t l0_reserve_rows =
            (uint64_t)manifest.max_pid > (uint64_t)replay_initial_max_pid
                ? (size_t)((uint64_t)manifest.max_pid - (uint64_t)replay_initial_max_pid)
                : 0;
        state.replay = impl::replay_update_wal_l0_blocks(
            join_index_path(index_path, manifest.wal_path),
            cells_per_subspace,
            replay_initial_live,
            l0_reserve_rows,
            replay_start_lsn,
            replay_start_offset);

        int M = (int)cells_per_subspace.size();
        if (state.replay.header.generation != manifest.generation ||
            state.replay.header.base_n != manifest.base_n ||
            state.replay.header.num_subspaces != (uint32_t)M ||
            state.replay.header.verify_dim != (uint32_t)shared_data->verify_vectors_dim ||
            state.replay.header.rabitq_record_bytes !=
                (uint32_t)shared_data->rabitq_bin_with_assign_record_bytes_) {
            throw std::runtime_error("update WAL header does not match loaded index/manifest");
        }
        if (state.replay.last_committed_lsn < manifest.last_committed_lsn) {
            throw std::runtime_error("update WAL committed LSN does not match manifest");
        }

        state.can_increment_composite_sizes =
            state.incremental &&
            previous_view != nullptr &&
            previous_view->base.get() == shared_data.get() &&
            previous_view->generation == manifest.generation &&
            !previous_view->composite_cell_sizes.empty() &&
            !state.replay.committed_blocks.empty() &&
            !state.replay.committed_delete &&
            !state.replay.committed_insert_reused_pid;
        if (state.can_increment_composite_sizes) {
            state.incremental_tail_cell_counts =
                build_l0_block_cell_counts(state.replay.committed_blocks, cells_per_subspace);
        }

        bool reuse_previous_growing =
            state.incremental &&
            state.replay.committed_blocks.empty() &&
            state.replay.last_committed_lsn == replay_start_lsn &&
            previous_view != nullptr &&
            previous_view->live.get() == state.replay.live.get() &&
            previous_view->growing_delta;
        if (reuse_previous_growing) {
            state.growing = previous_view->growing_delta;
        } else if (state.incremental) {
            auto next_growing = std::make_shared<UnsealedDeltaView>();
            next_growing->build_incremental_from_previous(
                *previous_view->growing_delta,
                std::move(state.replay.committed_blocks),
                cells_per_subspace,
                state.replay.live.get(),
                true,
                dead_pids_touch_growing_delta(
                    state.replay.committed_dead_pids, previous_view->growing_delta.get()),
                previous_view->live.get(),
                &state.replay.committed_dead_pids);
            state.growing = next_growing;
        } else {
            auto next_growing = std::make_shared<UnsealedDeltaView>();
            next_growing->committed_blocks = std::move(state.replay.committed_blocks);
            next_growing->build_lookup(cells_per_subspace, state.replay.live.get(), true);
            state.growing = next_growing;
        }
        if (!state.growing) {
            throw std::runtime_error("failed to create L0 growing delta reader");
        }
        return state;
    }

    static bool committed_delta_matches_loaded_reader(
        const std::shared_ptr<BaseSegment>& shared_data,
        const UpdateManifest& manifest,
        const CommittedUpdateDelta& delta,
        const ReaderView* previous_view) {
        if (!shared_data || previous_view == nullptr || !previous_view->live) return false;
        if (previous_view->base.get() != shared_data.get()) return false;
        if (manifest.base_path != "." || manifest.base_n != (uint32_t)shared_data->n) {
            return false;
        }
        if (delta.header.generation != manifest.generation ||
            delta.header.base_n != manifest.base_n ||
            delta.header.num_subspaces != shared_data->ivf_indices.size() ||
            delta.header.verify_dim != (uint32_t)shared_data->verify_vectors_dim ||
            delta.header.rabitq_record_bytes !=
                (uint32_t)shared_data->rabitq_bin_with_assign_record_bytes_) {
            return false;
        }
        if (delta.commit_lsn == 0 || delta.wal_offset_after_commit == 0 ||
            delta.commit_lsn <= previous_view->l0_last_committed_lsn) {
            return false;
        }
        if (delta.max_pid != manifest.max_pid || delta.live_count != manifest.live_count) {
            return false;
        }
        if (previous_view->generation != manifest.generation &&
            previous_view->generation != 0) {
            return false;
        }
        if (previous_view->max_pid > delta.max_pid) return false;
        return true;
    }

    static L0ReplayState l0_state_from_committed_delta(
        const CommittedUpdateDelta& delta,
        const ReaderView& previous_view,
        const std::vector<size_t>& cells_per_subspace) {
        L0ReplayState state;
        state.incremental = true;
        state.replay.header = delta.header;
        state.replay.committed_blocks = delta.committed_blocks;
        state.replay.live = previous_view.live->with_updates(
            delta.max_pid, delta.set_live_pids, delta.committed_dead_pids);
        if (state.replay.live->live_count() != delta.live_count) {
            throw std::runtime_error("committed update delta live_count does not match reader live table");
        }
        state.replay.last_committed_lsn = delta.commit_lsn;
        state.replay.last_committed_wal_offset = delta.wal_offset_after_commit;
        state.replay.max_pid = delta.max_pid;
        state.replay.live_count = delta.live_count;
        state.replay.committed_insert = delta.committed_insert;
        state.replay.committed_delete = delta.committed_delete;
        state.replay.committed_insert_reused_pid = delta.committed_insert_reused_pid;
        state.replay.committed_base_live_change = delta.committed_base_live_change;
        state.replay.committed_base_live_deletes = delta.committed_base_live_deletes;
        state.replay.committed_dead_pids = delta.committed_dead_pids;
        state.can_increment_composite_sizes =
            !state.replay.committed_blocks.empty() &&
            !state.replay.committed_delete &&
            !state.replay.committed_insert_reused_pid &&
            !previous_view.composite_cell_sizes.empty();
        if (state.can_increment_composite_sizes) {
            state.incremental_tail_cell_counts =
                build_l0_block_cell_counts(state.replay.committed_blocks, cells_per_subspace);
        }

        bool reuse_previous_growing =
            state.replay.committed_blocks.empty() &&
            previous_view.growing_delta &&
            previous_view.live.get() == state.replay.live.get();
        if (reuse_previous_growing) {
            state.growing = previous_view.growing_delta;
        } else if (previous_view.growing_delta) {
            auto next_growing = std::make_shared<UnsealedDeltaView>();
            next_growing->build_incremental_from_previous(
                *previous_view.growing_delta,
                state.replay.committed_blocks,
                cells_per_subspace,
                state.replay.live.get(),
                true,
                dead_pids_touch_growing_delta(
                    state.replay.committed_dead_pids, previous_view.growing_delta.get()),
                previous_view.live.get(),
                &state.replay.committed_dead_pids);
            state.growing = next_growing;
        } else {
            auto next_growing = std::make_shared<UnsealedDeltaView>();
            next_growing->committed_blocks = state.replay.committed_blocks;
            next_growing->build_lookup(cells_per_subspace, state.replay.live.get(), true);
            state.growing = next_growing;
        }
        if (!state.growing) {
            throw std::runtime_error("failed to create committed L0 growing delta reader");
        }
        return state;
    }

    static std::shared_ptr<const ReaderView> make_l0_reader_view_from_l0_state(
        const std::shared_ptr<BaseSegment>& shared_data,
        const UpdateManifest& manifest,
        const std::string& index_path,
        const ReaderView* previous_view,
        const std::vector<size_t>& cells_per_subspace,
        LoadedSealedDeltas sealed,
        L0ReplayState l0,
        bool initial_live_use_checkpoint) {
        (void)index_path;
        if (!shared_data) throw std::runtime_error("Cannot create update ReaderView without base segment");
        if (manifest.base_path != ".") {
            throw std::runtime_error("update manifest base path must be '.' for in-place query loading");
        }
        if (shared_data->n < 0 || manifest.base_n != (uint32_t)shared_data->n) {
            throw std::runtime_error("update manifest base_n does not match loaded base segment");
        }
        int M = (int)cells_per_subspace.size();
        const impl::L0WalBlockReplayResult& replay = l0.replay;
        std::shared_ptr<const LiveTable> live = replay.live;

        if (replay.last_committed_lsn == manifest.last_committed_lsn &&
            !initial_live_use_checkpoint &&
            (replay.max_pid != manifest.max_pid || replay.live_count != manifest.live_count)) {
            throw std::runtime_error("update WAL replay state does not match manifest PID/live counters");
        }
        if (replay.last_committed_lsn == manifest.last_committed_lsn &&
            (live->max_pid() != manifest.max_pid || live->live_count() != manifest.live_count)) {
            throw std::runtime_error("update WAL replay state does not match manifest PID/live counters");
        }
        if (live->max_pid() > (uint32_t)std::numeric_limits<int>::max()) {
            throw std::runtime_error("update reader max_pid exceeds QueryEngine v1 INT_MAX limit");
        }

        std::shared_ptr<const SealedDeltaPostingView> sealed_delta_postings;
        std::shared_ptr<const SealedDeltaPidLookup> sealed_delta_pid_lookup;
        bool sealed_delta_live_unchanged =
            sealed.reuse_exact_view &&
            previous_view != nullptr &&
            (previous_view->live.get() == live.get() ||
             (l0.incremental &&
              !replay.committed_insert_reused_pid &&
              !dead_pids_touch_sealed_deltas(replay.committed_dead_pids, sealed.segments)));
        if (sealed_delta_live_unchanged) {
            sealed_delta_postings = previous_view->sealed_delta_postings;
        } else {
            bool can_append_sealed_delta_postings =
                previous_view != nullptr &&
                previous_view->base.get() == shared_data.get() &&
                previous_view->sealed_delta_postings &&
                sealed.reusable_prefix == previous_view->sealed_deltas.size() &&
                sealed.reusable_prefix < sealed.segments.size();
            if (can_append_sealed_delta_postings) {
                std::vector<std::shared_ptr<const SealedDeltaSegment>> appended_deltas(
                    sealed.segments.begin() + (std::ptrdiff_t)sealed.reusable_prefix,
                    sealed.segments.end());
                for (const auto& segment : appended_deltas) {
                    if (segment) segment->build_live_cell_counts(*live);
                }
                auto built_postings = std::make_shared<SealedDeltaPostingView>();
                built_postings->build_appending_to_previous(
                    *previous_view->sealed_delta_postings,
                    appended_deltas,
                    cells_per_subspace,
                    live.get());
                if (!built_postings->empty()) sealed_delta_postings = built_postings;
            } else {
                for (const auto& segment : sealed.segments) {
                    if (segment) segment->build_live_cell_counts(*live);
                }
                if (!sealed.segments.empty()) {
                    auto built_postings = std::make_shared<SealedDeltaPostingView>();
                    built_postings->build(sealed.segments, cells_per_subspace, live.get());
                    if (!built_postings->empty()) sealed_delta_postings = built_postings;
                }
            }
        }
        if (sealed.reuse_exact_view) {
            sealed_delta_pid_lookup = previous_view->sealed_delta_pid_lookup;
        } else if (sealed.segments.size() > 1) {
            auto built_lookup = std::make_shared<SealedDeltaPidLookup>();
            built_lookup->build(sealed.segments);
            if (!built_lookup->empty()) sealed_delta_pid_lookup = built_lookup;
        }
        std::shared_ptr<const std::vector<uint32_t>> sparse_dead_pids;
        BaseLivePostingStats base_live_stats;
        bool previous_base_matches =
            previous_view != nullptr &&
            previous_view->base.get() == shared_data.get() &&
            previous_view->base_n == manifest.base_n;
        bool dead_set_unchanged =
            previous_base_matches &&
            (previous_view->live.get() == live.get() ||
             (l0.incremental &&
              !replay.committed_delete &&
              !replay.committed_insert_reused_pid));
        bool base_live_unchanged =
            previous_base_matches &&
            (previous_view->live.get() == live.get() ||
             (l0.incremental && replay.committed_base_live_deletes == 0));
        if (dead_set_unchanged) {
            sparse_dead_pids = previous_view->sparse_dead_pids;
        } else if (l0.incremental &&
                   previous_base_matches &&
                   !replay.committed_insert_reused_pid &&
                   try_extend_sparse_dead_pid_list(
                       previous_view, *live, replay.committed_dead_pids, sparse_dead_pids)) {
            // Incremental WAL replay only appends tombstones and fresh PIDs here, so the
            // sparse dead list can be extended without rescanning the full live bitset.
        } else {
            sparse_dead_pids = build_sparse_dead_pid_list(*live);
        }
        if (base_live_unchanged) {
            base_live_stats.live_count = previous_view->base_live_count;
            base_live_stats.fingerprint = previous_view->base_live_fingerprint;
        } else if (l0.incremental &&
                   previous_base_matches &&
                   replay.committed_base_live_deletes != 0) {
            base_live_stats = incremental_base_live_posting_stats(
                *previous_view,
                replay.committed_base_live_deletes,
                replay.last_committed_lsn);
        } else {
            base_live_stats = compute_base_live_posting_stats(*live, manifest.base_n);
        }
        bool has_growing_phase4_delta =
            l0.growing && !l0.growing->empty();
        bool has_growing_l0_tail = has_growing_phase4_delta;
        BaseLivePostingState base_live_postings;
        size_t total_dead_count =
            (size_t)live->max_pid() - (size_t)live->live_count();
        bool sparse_dead_selection_is_exact =
            total_dead_count <= sparse_dead_pid_limit(live->max_pid());
        bool base_overlay_build_is_material =
            (size_t)manifest.base_n > base_live_posting_stale_dead_limit(manifest.base_n);
        bool defer_base_live_overlay_for_l0 =
            has_growing_l0_tail &&
            sparse_dead_selection_is_exact &&
            base_overlay_build_is_material;
        if (!defer_base_live_overlay_for_l0) {
            base_live_postings = maybe_build_base_live_posting_view(
                shared_data,
                *live,
                manifest.base_n,
                M,
                base_live_stats,
                previous_view,
                replay.committed_dead_pids,
                replay.committed_insert_reused_pid);
        }

        auto view = std::make_shared<ReaderView>();
        view->generation = manifest.generation;
        view->base = shared_data;
        view->base_live_postings = base_live_postings.postings;
        view->base_live_postings_exact = base_live_postings.exact;
        view->base_live_postings_base_live_count =
            base_live_postings.overlay_base_live_count;
        view->base_live_postings_unfiltered_dead_pids =
            base_live_postings.unfiltered_dead_pids;
        view->sealed_deltas = std::move(sealed.segments);
        view->sealed_delta_postings = sealed_delta_postings;
        view->sealed_delta_pid_lookup = sealed_delta_pid_lookup;
        view->growing_delta = l0.growing;
        view->live = live;
        view->sparse_dead_pids = sparse_dead_pids;
        view->selection_dead_pids = sparse_dead_pids;
        view->base_n = manifest.base_n;
        view->max_pid = live->max_pid();
        view->live_count = live->live_count();
        view->base_live_count = base_live_stats.live_count;
        view->base_live_fingerprint = base_live_stats.fingerprint;
        view->base_dead_prefix_end = base_live_unchanged
            ? previous_view->base_dead_prefix_end
            : detect_base_dead_prefix_end(*live, manifest.base_n);
        view->live_suffix_begin = dead_set_unchanged
            ? previous_view->live_suffix_begin
            : detect_live_suffix_begin(*live);
        bool sealed_dead_scores_filtered =
            view->sealed_deltas.empty() || view->sealed_delta_postings != nullptr;
        bool growing_dead_scores_filtered =
            !view->growing_delta || view->growing_delta->all_pids_live();
        bool base_dead_scores_filtered =
            base_dead_scores_filtered_for_selection(*view, base_live_stats);
        view->dead_scores_filtered_for_selection =
            base_dead_scores_filtered &&
            sealed_dead_scores_filtered &&
            growing_dead_scores_filtered;
        if (!view->dead_scores_filtered_for_selection &&
            view->base_live_postings &&
            sealed_dead_scores_filtered &&
            growing_dead_scores_filtered &&
            view->base_live_postings_unfiltered_dead_pids &&
            !view->base_live_postings_unfiltered_dead_pids->empty()) {
            view->selection_dead_pids =
                view->base_live_postings_unfiltered_dead_pids;
        }
        if (previous_view != nullptr &&
            previous_view->base.get() == shared_data.get() &&
            previous_view->generation == view->generation &&
            previous_view->live.get() == view->live.get()) {
            view->composite_cell_sizes = previous_view->composite_cell_sizes;
        } else if (l0.can_increment_composite_sizes) {
            view->composite_cell_sizes = previous_view->composite_cell_sizes;
            if (!add_l0_tail_counts_to_composite(view->composite_cell_sizes,
                                                 l0.incremental_tail_cell_counts)) {
                view->composite_cell_sizes =
                    build_composite_cell_sizes(shared_data, *view, M);
            }
        } else {
            view->composite_cell_sizes =
                build_composite_cell_sizes(shared_data, *view, M);
        }
        view->l0_last_committed_lsn = replay.last_committed_lsn;
        view->l0_wal_replay_offset = replay.last_committed_wal_offset;
        view->candidate_selection_strategy =
            choose_candidate_selection_strategy(*view);
        return view;
    }

    static std::shared_ptr<const ReaderView> make_l0_reader_view(
        const std::shared_ptr<BaseSegment>& shared_data,
        const UpdateManifest& manifest,
        const std::string& index_path,
        const ReaderView* previous_view = nullptr) {
        std::vector<size_t> cells_per_subspace = reader_cells_per_subspace(shared_data);
        LoadedSealedDeltas sealed = load_sealed_deltas_for_reader(
            shared_data, manifest, index_path, previous_view, cells_per_subspace);
        InitialLiveState initial_live = load_initial_live_state(manifest, index_path);
        validate_l0_pid_floor(manifest);
        L0ReplayState l0 = replay_l0_for_reader(
            shared_data,
            manifest,
            index_path,
            previous_view,
            initial_live.live,
            cells_per_subspace);
        return make_l0_reader_view_from_l0_state(
            shared_data,
            manifest,
            index_path,
            previous_view,
            cells_per_subspace,
            std::move(sealed),
            std::move(l0),
            initial_live.use_checkpoint);
    }

    IndexConfig config_;
    std::string index_path_;
    int dim_ = 0;
    int n_ = 0;
    size_t active_point_count_ = 0;
    bool query_has_updates_ = false;
    bool live_filter_needed_ = false;
    bool query_growing_only_ = false;
    // Flat uniform-K cache of reader_view_->composite_cell_sizes for the hot
    // per-cell composite_cell_size() lookup (single deref). Empty when the
    // layout is non-uniform or the view has no overlay.
    std::vector<uint32_t> composite_cell_sizes_flat_;
    size_t composite_cell_K_ = 0;
    std::shared_ptr<BaseSegment> shared_data_;
    std::shared_ptr<const ReaderView> reader_view_;
    QueryProfileStats profile_stats_;

    std::vector<float> q_rot_;
    std::vector<std::vector<std::pair<float, int>>> active_cells_;
    std::vector<float> g_add_;
    std::vector<float> g_err_;
    std::vector<uint32_t> rabitq_gadd_stamp_;
    uint32_t rabitq_gadd_stamp_value_ = 0;
    struct RankEntry {
        float est_dist, low_dist;
        int pid;

        float upper_dist() const { return 2.0f * est_dist - low_dist; }
        bool operator<(const RankEntry& o) const {
            if (est_dist != o.est_dist) return est_dist < o.est_dist;
            if (low_dist != o.low_dist) return low_dist < o.low_dist;
            return pid < o.pid;
        }
    };
    void select_rabitq_verification_candidates() {
        // The k smallest upper bounds witness at least k candidates at or
        // below the cutoff whenever their distance bounds hold.
        float cutoff = std::numeric_limits<float>::infinity();
        const auto ranked_end = ranked_.begin() + static_cast<ptrdiff_t>(ranked_size_);
        if (config_.topk > 0 && static_cast<size_t>(config_.topk) < ranked_size_) {
            const auto top_end = ranked_.begin() + config_.topk;
            std::partial_sort(ranked_.begin(), top_end, ranked_end,
                [](const RankEntry& lhs, const RankEntry& rhs) {
                    return lhs.upper_dist() < rhs.upper_dist();
                });
            cutoff = (top_end - 1)->upper_dist();
        }

        top_pids_.clear();
        for (auto it = ranked_.begin(); it != ranked_end; ++it) {
            if (it->low_dist <= cutoff) top_pids_.push_back(it->pid);
        }
        std::sort(top_pids_.begin(), top_pids_.end());
    }

    std::vector<RankEntry> ranked_;
    size_t ranked_size_ = 0;
    std::vector<int> phase5_candidates_;
    std::vector<RabitqRecordRef> delta_candidate_record_refs_;
    std::vector<int> top_pids_;
    std::vector<std::pair<float, int>> verify_results_;
    std::vector<int> resolved_ordered_candidates_;
    std::vector<uint64_t> phase6a_pack_words_;
    std::vector<size_t> phase4_uniform_targets_;
    std::vector<Phase4CellRef> phase4_cells_;
    std::vector<std::vector<uint32_t>> delta_score_buffers_;
    std::vector<uint8_t> delta_score_touched_weights_;

    static constexpr size_t kInlineSmallDeltaPostings = 16;

    bool rotate_subspace_ = false;
    std::unique_ptr<RandomRotation> subspace_rotation_;
    std::vector<float> q_sub_rot_;

    std::unique_ptr<RandomRotation> rotation_;
    std::unique_ptr<SCScore> sc_score_;
    std::vector<uint64_t> query_bitplane_;
    uint64_t phase4_delta_posting_count_ = 0;
    struct ThreadEngine {
        QueryEngine* engine = nullptr;
        ~ThreadEngine() { delete engine; }
    };
    std::vector<ThreadEngine> thread_engines_;
};

// QueryIndexHandle is the thread-safe owner of the immutable base data and the
// current ReaderView snapshot. Writers publish by atomically swapping the
// snapshot pointer (under publish_mutex_); readers never take that mutex and
// never share mutable query scratch -- each search()/search_batch() call binds a
// thread-local QueryEngine executor to the snapshot it atomically loaded.
//
// QueryEngine remains a single-thread mutable query executor; this handle is the
// only object that may be shared across application threads for concurrent
// update/query.

} // namespace score

#include "query/query_index_handle.h"
