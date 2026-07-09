#pragma once

#include "build/index_builder.h"
#include "update/update_delta.h"
#include "update/update_index.h"
#include "update/update_manifest.h"
#include "update/update_state.h"
#include "update/update_wal.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <sys/stat.h>
#include <utility>
#include <vector>
#include <unistd.h>

namespace score {

struct UpdateCompactionPolicy {
    double flush_fraction = 0.05;
    double compaction_fraction = 0.20;
};

struct UpdateCompactionDecision {
    bool flushed_l0 = false;
    bool compacted = false;

    bool changed() const {
        return flushed_l0 || compacted;
    }
};

struct UpdateCompactionRunStats {
    UpdateCompactionDecision combined;
    size_t passes = 0;
    size_t no_op_passes = 0;
    size_t flushed_l0 = 0;
    size_t compacted = 0;

    bool changed() const { return combined.changed(); }

    void record(const UpdateCompactionDecision& decision) {
        ++passes;
        if (!decision.changed()) {
            ++no_op_passes;
            return;
        }
        if (decision.flushed_l0) {
            combined.flushed_l0 = true;
            ++flushed_l0;
        }
        if (decision.compacted) {
            combined.compacted = true;
            ++compacted;
        }
    }

    void absorb(const UpdateCompactionRunStats& other) {
        combined.flushed_l0 = combined.flushed_l0 || other.combined.flushed_l0;
        combined.compacted = combined.compacted ||
                                   other.combined.compacted;
        passes += other.passes;
        no_op_passes += other.no_op_passes;
        flushed_l0 += other.flushed_l0;
        compacted += other.compacted;
    }
};

struct UpdateCompactionEvaluationContext {
    bool base_artifact_bytes_cached = false;
    uint64_t base_artifact_bytes = 0;

    void invalidate_base_artifact_bytes() {
        base_artifact_bytes_cached = false;
        base_artifact_bytes = 0;
    }
};

struct CompactionOptions {
    bool reorder = false;
    IndexConfig reorder_config;
    size_t io_rows = 65536;
    size_t bucket_buffer_rows = 64;
    int rewrite_workers = 4;
    size_t rewrite_buffer_bytes = 64ull * 1024ull * 1024ull;
};

namespace impl {

inline bool compaction_policy_has_triggers(const UpdateCompactionPolicy& policy) {
    return policy.flush_fraction > 0.0 ||
           policy.compaction_fraction > 0.0;
}


inline uint64_t file_size_bytes(const std::string& path) {
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) {
        throw std::runtime_error("cannot stat file: " + path + " (" + std::strerror(errno) + ")");
    }
    if (st.st_size < 0) throw std::runtime_error("negative file size for: " + path);
    return (uint64_t)st.st_size;
}

inline uint64_t file_size_bytes_if_exists(const std::string& path) {
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) {
        if (errno == ENOENT) return 0;
        throw std::runtime_error("cannot stat file: " + path + " (" + std::strerror(errno) + ")");
    }
    if (st.st_size < 0) throw std::runtime_error("negative file size for: " + path);
    return (uint64_t)st.st_size;
}

inline uint64_t add_bytes_saturated(uint64_t a, uint64_t b) {
    if (a > std::numeric_limits<uint64_t>::max() - b) {
        return std::numeric_limits<uint64_t>::max();
    }
    return a + b;
}

inline void add_existing_file_bytes(uint64_t& total, const std::string& path) {
    total = add_bytes_saturated(total, file_size_bytes_if_exists(path));
}

inline uint64_t delta_segment_artifact_bytes_for_policy(const std::string& index_path,
                                                        const UpdateManifestDelta& delta) {
    if (delta.artifact_bytes != 0) return delta.artifact_bytes;
    std::string dir = join_index_path(index_path, delta.path);
    uint64_t total = 0;
    add_existing_file_bytes(total, dir + "/segment_manifest.json");
    add_existing_file_bytes(total, dir + "/delta_orig_ids.bin");
    add_existing_file_bytes(total, dir + "/delta_pid_to_row.bin");
    add_existing_file_bytes(total, dir + "/delta_row_cells.bin");
    add_existing_file_bytes(total, dir + "/delta_verify_vectors.fbin");
    add_existing_file_bytes(total, dir + "/delta_rabitq_bin_data.bin");

    uint32_t num_subspaces = 0;
    try {
        num_subspaces = impl::load_delta_segment_manifest(dir).num_subspaces;
    } catch (...) {
        // Policy sizing is a heuristic; when the tiny segment manifest is
        // unavailable, count the contiguous posting files that are present.
    }
    if (num_subspaces > 0) {
        for (uint32_t m = 0; m < num_subspaces; ++m) {
            add_existing_file_bytes(
                total, dir + "/delta_posting_subspace_" + std::to_string(m) + ".bin");
        }
        return total;
    }
    for (uint32_t m = 0; m < 1000000u; ++m) {
        std::string path = dir + "/delta_posting_subspace_" + std::to_string(m) + ".bin";
        uint64_t bytes = file_size_bytes_if_exists(path);
        if (bytes == 0) break;
        total = add_bytes_saturated(total, bytes);
    }
    return total;
}

inline uint64_t sealed_delta_artifact_bytes_for_policy(const std::string& index_path,
                                                       const UpdateManifest& manifest) {
    uint64_t total = 0;
    for (const UpdateManifestDelta& delta : manifest.sealed_deltas) {
        total = add_bytes_saturated(
            total, delta_segment_artifact_bytes_for_policy(index_path, delta));
    }
    return total;
}

inline uint64_t base_artifact_bytes_for_policy_uncached(const std::string& index_path) {
    uint64_t total = 0;
    const char* base_files[] = {
        "rotation.bin",
        "rotate_subspace.flag",
        "subspace_rotation.bin",
        "rabitq_centroids.bin",
        "verify_vectors.fbin",
        "rabitq_bin_data.bin",
        "rabitq_assignments.bin",
        "perm_to_orig.bin"
    };
    for (const char* name : base_files) {
        add_existing_file_bytes(total, index_path + "/" + name);
    }
    for (uint32_t m = 0; m < 1000000u; ++m) {
        uint64_t ivf_bytes = file_size_bytes_if_exists(
            index_path + "/ivf_subspace_" + std::to_string(m) + ".bin");
        uint64_t posting_bytes = file_size_bytes_if_exists(
            index_path + "/posting_subspace_" + std::to_string(m) + ".bin");
        if (ivf_bytes == 0 && posting_bytes == 0) break;
        total = add_bytes_saturated(total, ivf_bytes);
        total = add_bytes_saturated(total, posting_bytes);
    }
    return total;
}

inline uint64_t base_artifact_bytes_for_policy(
    const std::string& index_path,
    UpdateCompactionEvaluationContext* context = nullptr) {
    if (context && context->base_artifact_bytes_cached) {
        return context->base_artifact_bytes;
    }
    uint64_t total = base_artifact_bytes_for_policy_uncached(index_path);
    if (context) {
        context->base_artifact_bytes = total;
        context->base_artifact_bytes_cached = true;
    }
    return total;
}

inline uint64_t update_delta_artifact_bytes_for_policy(const std::string& index_path,
                                                       const UpdateManifest& manifest) {
    uint64_t total = sealed_delta_artifact_bytes_for_policy(index_path, manifest);
    if (manifest.last_committed_lsn != 0 && !manifest.wal_path.empty()) {
        total = add_bytes_saturated(
            total, file_size_bytes_if_exists(join_index_path(index_path, manifest.wal_path)));
    }
    return total;
}

inline uint32_t count_wal_insert_records(const UpdateWalReplayResult& replay) {
    uint64_t rows = 0;
    for (const UpdateWalBatch& batch : replay.committed_batches) {
        for (const UpdateWalRecord& record : batch.records) {
            if (record.type == UpdateWalRecord::Type::Insert) ++rows;
        }
    }
    if (rows > (uint64_t)std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("WAL insert record count exceeds uint32 range");
    }
    return (uint32_t)rows;
}

inline uint64_t l0_pid_floor(const UpdateManifest& manifest) {
    uint64_t floor = manifest.base_n;
    for (const UpdateManifestDelta& delta : manifest.sealed_deltas) {
        floor = std::max(floor, (uint64_t)delta.pid_begin + (uint64_t)delta.pid_count);
    }
    return floor;
}

inline bool should_compact_by_policy(const std::string& index_path,
                                           const UpdateManifest& manifest,
                                           const UpdateCompactionPolicy& policy,
                                           UpdateCompactionEvaluationContext* context = nullptr) {
    if (policy.compaction_fraction <= 0.0) return false;
    // Tombstone overhead: dead PIDs / max PID
    if (manifest.max_pid > 0) {
        uint32_t dead_count = manifest.max_pid - manifest.live_count;
        double tombstone_fraction =
            (double)dead_count / (double)std::max<uint32_t>(manifest.max_pid, 1);
        if (tombstone_fraction >= policy.compaction_fraction) return true;
    }
    // Delta overhead: total delta artifact bytes / base bytes
    uint64_t delta_bytes = update_delta_artifact_bytes_for_policy(index_path, manifest);
    uint64_t base_bytes = base_artifact_bytes_for_policy(index_path, context);
    if (base_bytes > 0 &&
        (double)delta_bytes / (double)base_bytes >= policy.compaction_fraction) {
        return true;
    }
    return false;
}

struct UpdateCompactionPolicyEvaluation {
    bool flush_l0 = false;
    bool compaction = false;

    bool any() const {
        return flush_l0 || compaction;
    }
};

inline UpdateCompactionPolicyEvaluation evaluate_compaction_policy(
    const std::string& index_path,
    const UpdateManifest& manifest,
    const UpdateCompactionPolicy& policy,
    UpdateCompactionEvaluationContext* context = nullptr) {
    UpdateCompactionPolicyEvaluation eval;
    // Flush L0 when WAL/base bytes exceed flush_fraction
    if (policy.flush_fraction > 0.0 && manifest.last_committed_lsn != 0) {
        std::string wal_path = join_index_path(index_path, manifest.wal_path);
        uint64_t wal_bytes = file_size_bytes_if_exists(wal_path);
        uint64_t base_bytes = base_artifact_bytes_for_policy(index_path, context);
        if (base_bytes > 0 &&
            (double)wal_bytes / (double)base_bytes >= policy.flush_fraction) {
            eval.flush_l0 = true;
        }
    }
    // Compaction: fold sealed deltas into base when total overhead exceeds compaction_fraction
    eval.compaction = should_compact_by_policy(index_path, manifest, policy, context);
    return eval;
}

inline size_t read_posting_cell_count(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("cannot open posting list header: " + path);
    int K = 0;
    int total = 0;
    in.read(reinterpret_cast<char*>(&K), sizeof(int));
    in.read(reinterpret_cast<char*>(&total), sizeof(int));
    if (!in || K <= 0 || total < 0) {
        throw std::runtime_error("invalid posting list header: " + path);
    }
    return (size_t)K;
}

inline std::string format_delta_segment_id(uint64_t generation, uint64_t lsn) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << generation
        << "_" << std::setw(16) << lsn;
    return out.str();
}

inline std::string choose_unique_delta_segment_path(const std::string& index_path,
                                                    std::string& segment_id) {
    ensure_dir(index_path + "/deltas");
    for (int attempt = 0; attempt < 1000; ++attempt) {
        std::string suffix = attempt == 0 ? "" : ("_" + std::to_string(attempt));
        std::string candidate_id = segment_id + suffix;
        std::string rel = "deltas/" + candidate_id;
        std::string abs = join_index_path(index_path, rel);
        if (!path_exists(abs)) {
            segment_id = candidate_id;
            return rel;
        }
    }
    throw std::runtime_error("cannot choose unique delta segment path");
}

inline void remove_delta_segment_dir_best_effort(const std::string& index_path,
                                                 const std::string& rel_path,
                                                 uint32_t num_subspaces) {
    std::string dir = join_index_path(index_path, rel_path);
    std::remove((dir + "/segment_manifest.json").c_str());
    std::remove((dir + "/delta_orig_ids.bin").c_str());
    std::remove((dir + "/delta_pid_to_row.bin").c_str());
    std::remove((dir + "/delta_row_cells.bin").c_str());
    std::remove((dir + "/delta_verify_vectors.fbin").c_str());
    std::remove((dir + "/delta_rabitq_bin_data.bin").c_str());
    for (uint32_t m = 0; m < num_subspaces; ++m) {
        std::remove((dir + "/delta_posting_subspace_" + std::to_string(m) + ".bin").c_str());
    }
    rmdir(dir.c_str());
}

inline std::string choose_compaction_staging_dir(const std::string& index_path) {
    std::string base = index_path + ".compaction_tmp_" + std::to_string((uint64_t)::getpid());
    for (int attempt = 0; attempt < 1000; ++attempt) {
        std::string candidate = attempt == 0 ? base : base + "_" + std::to_string(attempt);
        if (!path_exists(candidate)) {
            ensure_dir(candidate);
            return candidate;
        }
    }
    throw std::runtime_error("cannot choose unique compaction staging directory");
}

// Backup dir for crash-safe compaction: holds hardlinks to the pre-compaction
// base files + manifest, so a crash mid-publish can be rolled back on load.
inline std::string choose_compaction_backup_dir(const std::string& index_path) {
    std::string base = index_path + ".compaction_backup_" + std::to_string((uint64_t)::getpid());
    for (int attempt = 0; attempt < 1000; ++attempt) {
        std::string candidate = attempt == 0 ? base : base + "_" + std::to_string(attempt);
        if (!path_exists(candidate)) {
            ensure_dir(candidate);
            return candidate;
        }
    }
    throw std::runtime_error("cannot choose unique compaction backup directory");
}

// Hardlink a file (for backup). Cheap: same inode, no data copy.
inline void hardlink_file(const std::string& src, const std::string& dst) {
    if (::link(src.c_str(), dst.c_str()) != 0) {
        throw std::runtime_error("cannot hardlink '" + src + "' to '" + dst +
                                 "' (" + std::strerror(errno) + ")");
    }
}

inline void copy_file_replace(const std::string& src_path, const std::string& dst_path) {
    std::ifstream in(src_path, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("cannot open source file: " + src_path);
    std::ofstream out(dst_path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) throw std::runtime_error("cannot create destination file: " + dst_path);
    out << in.rdbuf();
    if (!out) throw std::runtime_error("failed copying file to: " + dst_path);
}

inline void rename_replace(const std::string& src_path, const std::string& dst_path) {
    if (::rename(src_path.c_str(), dst_path.c_str()) != 0) {
        throw std::runtime_error("rename failed from " + src_path + " to " + dst_path +
                                 ": " + std::strerror(errno));
    }
}

inline void remove_file_best_effort(const std::string& path) {
    std::remove(path.c_str());
}

class CompactionPublishMarker {
public:
    // Writes the backup_dir path into the marker so that crash recovery
    // (recover_compaction_if_marker_present) can find + restore the pre-compaction state.
    CompactionPublishMarker(const std::string& index_path,
                            const std::string& backup_dir,
                            bool do_fsync)
        : path_(compaction_publish_marker_path(index_path)), do_fsync_(do_fsync) {
        int fd = ::open(path_.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0644);
        if (fd < 0) {
            throw std::runtime_error("cannot create compaction publish marker: " + path_ +
                                     " (" + std::strerror(errno) + ")");
        }
        std::string payload = backup_dir + "\n";
        ssize_t written = ::write(fd, payload.data(), payload.size());
        if (written != (ssize_t)payload.size()) {
            int saved = errno;
            ::close(fd);
            ::unlink(path_.c_str());
            throw std::runtime_error("cannot write compaction publish marker: " + path_ +
                                     " (" + std::strerror(saved) + ")");
        }
        if (::close(fd) != 0) {
            int saved = errno;
            ::unlink(path_.c_str());
            throw std::runtime_error("cannot close compaction publish marker: " + path_ +
                                     " (" + std::strerror(saved) + ")");
        }
        fsync_file_if_requested(path_, do_fsync_);
        fsync_parent_dir_if_requested(path_, do_fsync_);
        active_ = true;
    }

    ~CompactionPublishMarker() {
        if (active_) {
            ::unlink(path_.c_str());
        }
    }

    CompactionPublishMarker(const CompactionPublishMarker&) = delete;
    CompactionPublishMarker& operator=(const CompactionPublishMarker&) = delete;

    void remove() {
        if (!active_) return;
        if (::unlink(path_.c_str()) != 0 && errno != ENOENT) {
            throw std::runtime_error("cannot remove compaction publish marker: " + path_ +
                                     " (" + std::strerror(errno) + ")");
        }
        fsync_parent_dir_if_requested(path_, do_fsync_);
        active_ = false;
    }

private:
    std::string path_;
    bool do_fsync_ = false;
    bool active_ = false;
};

inline void fsync_delta_segment_if_requested(const std::string& segment_dir,
                                             uint32_t num_subspaces,
                                             bool do_fsync) {
    if (!do_fsync) return;
    const std::vector<std::string> names = {
        "segment_manifest.json",
        "delta_orig_ids.bin",
        "delta_pid_to_row.bin",
        "delta_verify_vectors.fbin",
        "delta_rabitq_bin_data.bin"
    };
    for (const std::string& name : names) {
        fsync_file_if_requested(segment_dir + "/" + name, true);
    }
    if (path_exists(segment_dir + "/delta_row_cells.bin")) {
        fsync_file_if_requested(segment_dir + "/delta_row_cells.bin", true);
    }
    for (uint32_t m = 0; m < num_subspaces; ++m) {
        fsync_file_if_requested(
            segment_dir + "/delta_posting_subspace_" + std::to_string(m) + ".bin", true);
    }
    fsync_directory_if_requested(segment_dir, true);
    fsync_parent_dir_if_requested(segment_dir, true);
}

inline void remove_update_artifacts_best_effort(const std::string& index_path,
                                                const UpdateManifest& manifest,
                                                uint32_t num_subspaces) {
    remove_file_best_effort(update_manifest_path(index_path));
    remove_file_best_effort(join_index_path(index_path, manifest.wal_path));
    remove_file_best_effort(join_index_path(index_path, manifest.checkpoint_path));
    remove_file_best_effort(join_index_path(index_path, manifest.direct_map_path));
    rmdir((index_path + "/updates").c_str());
    for (const UpdateManifestDelta& delta : manifest.sealed_deltas) {
        remove_delta_segment_dir_best_effort(index_path, delta.path, num_subspaces);
    }
    rmdir((index_path + "/deltas").c_str());
}

inline int detect_num_subspaces(const std::string& index_path) {
    int M = 0;
    for (;;) {
        std::ifstream probe(index_path + "/ivf_subspace_" + std::to_string(M) + ".bin",
                            std::ios::binary);
        if (!probe.is_open()) break;
        ++M;
    }
    if (M <= 0) throw std::runtime_error("compaction requires IVF subspace artifacts");
    return M;
}

struct CompactionBaseMetadata {
    int n = 0;
    int rabitq_k = 0;
    int verify_dim = 0;
    int rabitq_record_bytes = 0;
    uint32_t num_subspaces = 0;
    std::vector<size_t> cells_per_subspace;
    std::vector<uint32_t> perm_to_orig;
};

inline int read_ivf_cell_count(const std::string& path, int expected_subspace) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("cannot open IVF subspace header: " + path);
    int sub_dim = 0;
    int K = 0;
    int subspace_id = 0;
    in.read(reinterpret_cast<char*>(&sub_dim), sizeof(int));
    in.read(reinterpret_cast<char*>(&K), sizeof(int));
    in.read(reinterpret_cast<char*>(&subspace_id), sizeof(int));
    if (!in || sub_dim <= 0 || K <= 0 || subspace_id != expected_subspace) {
        throw std::runtime_error("invalid IVF subspace header: " + path);
    }
    return K;
}

inline CompactionBaseMetadata load_compaction_base_metadata(const std::string& index_path) {
    CompactionBaseMetadata meta;

    RandomRotation::Metadata rotation =
        RandomRotation::load_metadata(index_path + "/rotation.bin");

    int centroid_k = 0;
    int centroid_dim = 0;
    int padded_dim = 0;
    {
        std::ifstream cent_in(index_path + "/rabitq_centroids.bin", std::ios::binary);
        if (!cent_in.is_open()) throw std::runtime_error("Cannot open rabitq_centroids.bin");
        cent_in.read(reinterpret_cast<char*>(&centroid_k), sizeof(int));
        cent_in.read(reinterpret_cast<char*>(&centroid_dim), sizeof(int));
        cent_in.read(reinterpret_cast<char*>(&padded_dim), sizeof(int));
        if (!cent_in || centroid_k <= 0 || centroid_dim != rotation.dim ||
            padded_dim != rotation.padded_dim) {
            throw std::runtime_error("invalid rabitq_centroids.bin header for compaction");
        }
        meta.rabitq_k = centroid_k;
    }

    {
        std::ifstream bin_in(index_path + "/rabitq_bin_data.bin", std::ios::binary);
        if (!bin_in.is_open()) throw std::runtime_error("Cannot open rabitq_bin_data.bin");
        bin_in.read(reinterpret_cast<char*>(&meta.n), sizeof(int));
        bin_in.read(reinterpret_cast<char*>(&meta.rabitq_record_bytes), sizeof(int));
        const size_t expected_payload_bytes = rabitqlib::BinDataMap<float>::data_bytes(padded_dim);
        const size_t expected_record_bytes = sizeof(uint32_t) + expected_payload_bytes;
        if (!bin_in || meta.n <= 0 || meta.rabitq_record_bytes != (int)expected_record_bytes) {
            throw std::runtime_error("invalid rabitq_bin_data.bin header for compaction");
        }
    }

    auto [verify_tag, verify_n, verify_dim] = detect_vector_format(index_path + "/verify_vectors.fbin");
    if (verify_tag != 0 || verify_n != meta.n || verify_dim != padded_dim) {
        throw std::runtime_error("verify_vectors.fbin shape does not match RaBitQ payloads");
    }
    meta.verify_dim = verify_dim;

    int M = detect_num_subspaces(index_path);
    meta.num_subspaces = (uint32_t)M;
    meta.cells_per_subspace.reserve((size_t)M);
    for (int m = 0; m < M; ++m) {
        int ivf_k = read_ivf_cell_count(index_path + "/ivf_subspace_" + std::to_string(m) + ".bin", m);
        size_t posting_k = read_posting_cell_count(
            index_path + "/posting_subspace_" + std::to_string(m) + ".bin");
        if ((size_t)ivf_k != posting_k) {
            throw std::runtime_error("Subspace index/posting K mismatch at subspace " +
                                     std::to_string(m));
        }
        meta.cells_per_subspace.push_back(posting_k);
    }

    {
        std::ifstream perm_in(index_path + "/perm_to_orig.bin", std::ios::binary);
        if (perm_in.good()) {
            int n_perm = 0;
            perm_in.read(reinterpret_cast<char*>(&n_perm), sizeof(int));
            if (!perm_in || n_perm != meta.n) {
                throw std::runtime_error("perm_to_orig.bin length does not match base payloads");
            }
            meta.perm_to_orig.resize((size_t)n_perm);
            perm_in.read(reinterpret_cast<char*>(meta.perm_to_orig.data()),
                         meta.perm_to_orig.size() * sizeof(uint32_t));
            if (!perm_in) throw std::runtime_error("Short read from perm_to_orig.bin");
        }
    }

    return meta;
}

inline std::shared_ptr<const LiveTable> load_compaction_live_table(
    const std::string& index_path,
    const UpdateManifest& manifest) {
    bool checkpoint_exists = false;
    if (!manifest.checkpoint_path.empty()) {
        std::ifstream checkpoint_probe(join_index_path(index_path, manifest.checkpoint_path),
                                       std::ios::binary);
        checkpoint_exists = checkpoint_probe.good();
    }
    if (!manifest.sealed_deltas.empty() && !checkpoint_exists) {
        throw std::runtime_error("compaction requires live checkpoint with sealed deltas");
    }
    std::shared_ptr<const LiveTable> live = checkpoint_exists
        ? LiveTable::load_binary(join_index_path(index_path, manifest.checkpoint_path))
        : LiveTable::all_live(manifest.base_n);
    if (live->max_pid() != manifest.max_pid || live->live_count() != manifest.live_count) {
        throw std::runtime_error("live table does not match update manifest for compaction");
    }
    return live;
}

inline std::vector<std::shared_ptr<const SealedDeltaSegment>> load_compaction_sealed_delta_range(
    const std::string& index_path,
    const UpdateManifest& manifest,
    const CompactionBaseMetadata& meta,
    size_t begin,
    size_t count,
    bool load_row_cells = false);

inline std::vector<std::shared_ptr<const SealedDeltaSegment>> load_compaction_sealed_deltas(
    const std::string& index_path,
    const UpdateManifest& manifest,
    const CompactionBaseMetadata& meta,
    size_t max_segments = 0,
    bool load_row_cells = false) {
    size_t count = manifest.sealed_deltas.size();
    if (max_segments > 0) count = std::min(count, max_segments);
    return load_compaction_sealed_delta_range(
        index_path, manifest, meta, 0, count, load_row_cells);
}

inline std::vector<std::shared_ptr<const SealedDeltaSegment>> load_compaction_sealed_delta_range(
    const std::string& index_path,
    const UpdateManifest& manifest,
    const CompactionBaseMetadata& meta,
    size_t begin,
    size_t count,
    bool load_row_cells) {
    if (begin > manifest.sealed_deltas.size() ||
        count > manifest.sealed_deltas.size() - begin) {
        throw std::out_of_range("sealed delta load range is outside manifest");
    }
    std::vector<std::shared_ptr<const SealedDeltaSegment>> sealed_deltas;
    sealed_deltas.reserve(count);
    for (size_t i = begin; i < begin + count; ++i) {
        const UpdateManifestDelta& delta = manifest.sealed_deltas[i];
        sealed_deltas.push_back(SealedDeltaSegment::load(
            delta.segment_id,
            join_index_path(index_path, delta.path),
            meta.cells_per_subspace,
            (uint32_t)meta.verify_dim,
            (uint32_t)meta.rabitq_record_bytes,
            delta.pid_begin,
            delta.pid_count,
            delta.live_count,
            load_row_cells));
    }
    return sealed_deltas;
}

inline DirectMap rebuild_compaction_direct_map(
    const std::string& index_path,
    const UpdateManifest& manifest,
    const CompactionBaseMetadata& meta,
    const std::vector<std::shared_ptr<const SealedDeltaSegment>>& sealed_deltas,
    const UpdateWalReplayResult& replay,
    std::shared_ptr<const LiveTable>* live_out = nullptr) {
    if ((uint32_t)meta.n != manifest.base_n) {
        throw std::runtime_error("base metadata does not match manifest base_n");
    }
    if (replay.header.generation != manifest.generation ||
        replay.header.base_n != manifest.base_n ||
        replay.header.num_subspaces != meta.num_subspaces ||
        replay.header.verify_dim != (uint32_t)meta.verify_dim ||
        replay.header.rabitq_record_bytes != (uint32_t)meta.rabitq_record_bytes ||
        replay.last_committed_lsn != manifest.last_committed_lsn) {
        throw std::runtime_error("cannot rebuild compaction DirectMap: WAL does not match manifest");
    }

    bool checkpoint_exists = false;
    if (!manifest.checkpoint_path.empty()) {
        std::ifstream checkpoint_probe(join_index_path(index_path, manifest.checkpoint_path),
                                       std::ios::binary);
        checkpoint_exists = checkpoint_probe.good();
    }
    if (!manifest.sealed_deltas.empty() && !checkpoint_exists) {
        throw std::runtime_error("cannot rebuild compaction DirectMap over sealed deltas without live checkpoint");
    }

    std::shared_ptr<const LiveTable> initial_live = checkpoint_exists
        ? LiveTable::load_binary(join_index_path(index_path, manifest.checkpoint_path))
        : LiveTable::all_live(manifest.base_n);
    if (initial_live->max_pid() > manifest.max_pid) {
        throw std::runtime_error("compaction live checkpoint is ahead of update manifest");
    }
    auto live = std::make_shared<LiveTable>(*initial_live);

    DirectMap map = manifest.direct_map_mode == "sparse"
        ? DirectMap::sparse()
        : DirectMap::dense();
    map.reserve_entries((size_t)manifest.live_count);
    for (uint32_t pid = 0; pid < manifest.base_n; ++pid) {
        if (!live->is_live(pid)) continue;
        uint32_t orig = meta.perm_to_orig.empty() ? pid : meta.perm_to_orig[(size_t)pid];
        map.set(orig, pid);
    }
    for (const auto& segment : sealed_deltas) {
        if (!segment) continue;
        for (uint32_t row = 0; row < segment->row_count(); ++row) {
            uint32_t pid = segment->pid(row);
            if (!live->is_live(pid)) continue;
            map.set(segment->orig_id(row), pid);
        }
    }

    for (const UpdateWalBatch& batch : replay.committed_batches) {
        std::vector<uint32_t> set_live;
        std::vector<uint32_t> set_dead;
        set_live.reserve(batch.records.size());
        set_dead.reserve(batch.records.size());
        for (const UpdateWalRecord& record : batch.records) {
            if (record.type == UpdateWalRecord::Type::Delete) {
                map.erase(record.tombstone.orig_id);
                set_dead.push_back(record.tombstone.old_global_pid);
            } else {
                if (record.insert.global_pid < manifest.base_n) {
                    throw std::runtime_error("compaction DirectMap rebuild saw insert PID overlapping base");
                }
                map.set(record.insert.orig_id, record.insert.global_pid);
                set_live.push_back(record.insert.global_pid);
            }
        }
        live->apply_updates_in_place(batch.commit.max_pid, set_live, set_dead);
        if (live->live_count() != batch.commit.live_count) {
            throw std::runtime_error("compaction DirectMap rebuild live_count mismatch");
        }
    }

    if (live->max_pid() != manifest.max_pid || live->live_count() != manifest.live_count ||
        map.live_size() != manifest.live_count) {
        throw std::runtime_error("rebuilt compaction DirectMap does not match manifest live state");
    }
    if (live_out != nullptr) *live_out = live;
    return map;
}

inline std::shared_ptr<const LiveTable> load_compaction_live_table_from_replay(
    const std::string& index_path,
    const UpdateManifest& manifest,
    const CompactionBaseMetadata& meta,
    const UpdateWalReplayResult& replay) {
    if ((uint32_t)meta.n != manifest.base_n) {
        throw std::runtime_error("base metadata does not match manifest base_n");
    }
    if (replay.header.generation != manifest.generation ||
        replay.header.base_n != manifest.base_n ||
        replay.header.num_subspaces != meta.num_subspaces ||
        replay.header.verify_dim != (uint32_t)meta.verify_dim ||
        replay.header.rabitq_record_bytes != (uint32_t)meta.rabitq_record_bytes ||
        replay.last_committed_lsn != manifest.last_committed_lsn) {
        throw std::runtime_error("cannot load compaction live table: WAL does not match manifest");
    }

    bool checkpoint_exists = false;
    if (!manifest.checkpoint_path.empty()) {
        std::ifstream checkpoint_probe(join_index_path(index_path, manifest.checkpoint_path),
                                       std::ios::binary);
        checkpoint_exists = checkpoint_probe.good();
    }
    if (!manifest.sealed_deltas.empty() && !checkpoint_exists) {
        throw std::runtime_error("cannot load compaction live table over sealed deltas without checkpoint");
    }

    std::shared_ptr<const LiveTable> initial_live = checkpoint_exists
        ? LiveTable::load_binary(join_index_path(index_path, manifest.checkpoint_path))
        : LiveTable::all_live(manifest.base_n);
    if (initial_live->max_pid() > manifest.max_pid) {
        throw std::runtime_error("compaction live checkpoint is ahead of update manifest");
    }
    auto live = std::make_shared<LiveTable>(*initial_live);

    for (const UpdateWalBatch& batch : replay.committed_batches) {
        std::vector<uint32_t> set_live;
        std::vector<uint32_t> set_dead;
        set_live.reserve(batch.records.size());
        set_dead.reserve(batch.records.size());
        for (const UpdateWalRecord& record : batch.records) {
            if (record.type == UpdateWalRecord::Type::Insert) {
                if (record.insert.global_pid < manifest.base_n) {
                    throw std::runtime_error("compaction live table saw insert PID overlapping base");
                }
                set_live.push_back(record.insert.global_pid);
            } else {
                set_dead.push_back(record.tombstone.old_global_pid);
            }
        }
        live->apply_updates_in_place(batch.commit.max_pid, set_live, set_dead);
        if (live->live_count() != batch.commit.live_count) {
            throw std::runtime_error("compaction live table replay live_count mismatch");
        }
    }

    if (live->max_pid() != manifest.max_pid ||
        live->live_count() != manifest.live_count) {
        throw std::runtime_error("compaction live table does not match manifest live state");
    }
    return live;
}

inline std::vector<std::pair<uint32_t, uint32_t>> build_compaction_live_entries_preserving_base_order(
    const UpdateManifest& manifest,
    const CompactionBaseMetadata& meta,
    const std::vector<std::shared_ptr<const SealedDeltaSegment>>& sealed_deltas,
    const LiveTable& live,
    const DirectMap& direct_map) {
    std::vector<std::pair<uint32_t, uint32_t>> live_entries;
    live_entries.reserve((size_t)manifest.live_count);
    auto append_live_entry = [&](uint32_t orig_id, uint32_t old_pid) {
        if (!live.is_live(old_pid)) return;
        uint32_t mapped_pid = 0;
        if (!direct_map.lookup(orig_id, mapped_pid) || mapped_pid != old_pid) {
            throw std::runtime_error("DirectMap does not match compaction live row order");
        }
        live_entries.push_back({orig_id, old_pid});
    };
    for (uint32_t old_pid = 0; old_pid < manifest.base_n; ++old_pid) {
        uint32_t orig_id = meta.perm_to_orig.empty()
            ? old_pid
            : meta.perm_to_orig[(size_t)old_pid];
        append_live_entry(orig_id, old_pid);
    }
    for (const auto& segment : sealed_deltas) {
        if (!segment) continue;
        for (uint32_t row = 0; row < segment->row_count(); ++row) {
            append_live_entry(segment->orig_id(row), segment->pid(row));
        }
    }
    if (live_entries.size() != (size_t)manifest.live_count) {
        throw std::runtime_error("cannot compact: live entry count mismatch");
    }
    return live_entries;
}

class CompactionPostingReader {
public:
    explicit CompactionPostingReader(const std::string& path)
        : path_(path), in_(path, std::ios::binary) {
        if (!in_.is_open()) throw std::runtime_error("cannot open posting list: " + path);
        int total = 0;
        in_.read(reinterpret_cast<char*>(&K_), sizeof(int));
        in_.read(reinterpret_cast<char*>(&total), sizeof(int));
        if (!in_ || K_ <= 0 || total < 0) {
            throw std::runtime_error("invalid posting list header: " + path);
        }
        total_ = total;
        offsets_.resize((size_t)K_ + 1);
        in_.read(reinterpret_cast<char*>(offsets_.data()), offsets_.size() * sizeof(int));
        if (!in_ || offsets_.front() != 0 || offsets_.back() != total_) {
            throw std::runtime_error("invalid posting list offsets: " + path);
        }
        for (int cell = 0; cell < K_; ++cell) {
            if (offsets_[(size_t)cell] > offsets_[(size_t)cell + 1]) {
                throw std::runtime_error("non-monotonic posting list offsets: " + path);
            }
        }
        ids_begin_ = (std::streamoff)(2 * sizeof(int) + offsets_.size() * sizeof(int));
    }

    int K() const { return K_; }

    void read_cell(int cell, std::vector<uint32_t>& ids) {
        if (cell < 0 || cell >= K_) throw std::out_of_range("posting cell out of range");
        int begin = offsets_[(size_t)cell];
        int end = offsets_[(size_t)cell + 1];
        int count = end - begin;
        ids.resize((size_t)count);
        if (count == 0) return;
        in_.clear();
        in_.seekg(ids_begin_ + (std::streamoff)begin * (std::streamoff)sizeof(uint32_t));
        in_.read(reinterpret_cast<char*>(ids.data()), ids.size() * sizeof(uint32_t));
        if (!in_) throw std::runtime_error("short posting cell read: " + path_);
    }

private:
    std::string path_;
    std::ifstream in_;
    int K_ = 0;
    int total_ = 0;
    std::vector<int> offsets_;
    std::streamoff ids_begin_ = 0;
};

} // namespace impl

inline bool flush_l0_to_l1_locked(const std::string& index_path) {
    if (!update_manifest_exists(index_path)) {
        throw std::runtime_error("cannot flush L0: update_manifest.json is missing");
    }
    UpdateManifest manifest = load_update_manifest_json_for_index(index_path);
    if (manifest.last_committed_lsn == 0) return false;
    const bool do_fsync = update_durability_fsync_artifacts(manifest.durability_mode);

    std::string wal_path = impl::join_index_path(index_path, manifest.wal_path);
    UpdateWalReplayResult replay = replay_update_wal(wal_path);
    if (replay.header.generation != manifest.generation ||
        replay.header.base_n != manifest.base_n ||
        replay.last_committed_lsn != manifest.last_committed_lsn ||
        replay.max_pid != manifest.max_pid ||
        replay.live_count != manifest.live_count) {
        throw std::runtime_error("cannot flush L0: WAL replay state does not match manifest");
    }

    impl::CompactionBaseMetadata meta = impl::load_compaction_base_metadata(index_path);
    if ((uint32_t)meta.n != manifest.base_n ||
        meta.num_subspaces != replay.header.num_subspaces ||
        (uint32_t)meta.verify_dim != replay.header.verify_dim ||
        (uint32_t)meta.rabitq_record_bytes != replay.header.rabitq_record_bytes) {
        throw std::runtime_error("cannot flush L0: base metadata does not match WAL header");
    }
    std::shared_ptr<const LiveTable> live =
        impl::load_compaction_live_table_from_replay(
            index_path, manifest, meta, replay);

    std::vector<UpdateWalInsert> live_l0_inserts;
    live_l0_inserts.reserve(impl::count_wal_insert_records(replay));
    uint64_t l0_pid_floor = impl::l0_pid_floor(manifest);
    for (UpdateWalBatch& batch : replay.committed_batches) {
        for (UpdateWalRecord& record : batch.records) {
            if (record.type != UpdateWalRecord::Type::Insert) continue;
            if ((uint64_t)record.insert.global_pid < l0_pid_floor) {
                throw std::runtime_error("cannot flush L0: WAL insert PID overlaps sealed delta range");
            }
            if (!live->is_live(record.insert.global_pid)) continue;
            live_l0_inserts.push_back(std::move(record.insert));
        }
    }

    uint64_t new_generation = manifest.generation + 1;
    if (!live_l0_inserts.empty()) {
        std::string segment_id = impl::format_delta_segment_id(
            new_generation, manifest.last_committed_lsn);
        std::string rel_path = impl::choose_unique_delta_segment_path(index_path, segment_id);
        impl::fsync_parent_dir_if_requested(index_path + "/deltas", do_fsync);
        auto segment = SealedDeltaSegment::from_inserts(
            segment_id,
            std::move(live_l0_inserts),
            meta.cells_per_subspace,
            replay.header.verify_dim,
            replay.header.rabitq_record_bytes);
        std::string segment_path = impl::join_index_path(index_path, rel_path);
        segment->save(segment_path);
        impl::fsync_delta_segment_if_requested(
            segment_path, replay.header.num_subspaces, do_fsync);

        UpdateManifestDelta delta;
        delta.segment_id = segment_id;
        delta.path = rel_path;
        delta.pid_begin = segment->pid_begin();
        delta.pid_count = segment->pid_count();
        delta.live_count = segment->live_count();
        delta.artifact_bytes =
            impl::delta_segment_artifact_bytes_for_policy(index_path, delta);
        manifest.sealed_deltas.push_back(std::move(delta));
    }

    if (live->live_count() != manifest.live_count) {
        throw std::runtime_error("cannot flush L0: rebuilt live table does not match manifest live_count");
    }
    impl::ensure_dir(index_path + "/updates");
    impl::fsync_parent_dir_if_requested(index_path + "/updates", do_fsync);
    std::string checkpoint_path = impl::join_index_path(index_path, manifest.checkpoint_path);
    live->save_binary(checkpoint_path, do_fsync);

    manifest.generation = new_generation;
    manifest.last_committed_lsn = 0;
    manifest.last_fsynced_lsn = 0;

    UpdateWalHeader header;
    header.generation = manifest.generation;
    header.base_n = manifest.base_n;
    header.num_subspaces = replay.header.num_subspaces;
    header.verify_dim = replay.header.verify_dim;
    header.rabitq_record_bytes = replay.header.rabitq_record_bytes;
    {
        UpdateWalWriter writer(wal_path, header, false);
        writer.flush();
    }

    impl::fsync_file_if_requested(wal_path, do_fsync);
    impl::fsync_parent_dir_if_requested(wal_path, do_fsync);
    write_update_manifest_json_for_index(manifest, index_path, do_fsync);
    return true;
}

inline bool flush_l0_to_l1(const std::string& index_path) {
    impl::UpdateWriteLock writer_lock(index_path);
    impl::UpdatePublishWriteLock publish_lock(index_path);
    return flush_l0_to_l1_locked(index_path);
}


inline bool flush_delta_segments_locked(const std::string& index_path) {
    return flush_l0_to_l1_locked(index_path);
}

inline bool flush_delta_segments(const std::string& index_path) {
    impl::UpdateWriteLock writer_lock(index_path);
    impl::UpdatePublishWriteLock publish_lock(index_path);
    return flush_delta_segments_locked(index_path);
}

inline bool compact_index_locked(
    const std::string& index_path,
    const CompactionOptions& options = CompactionOptions()) {
    if (!update_manifest_exists(index_path)) return false;
    (void)flush_delta_segments_locked(index_path);

    UpdateManifest manifest = load_update_manifest_json_for_index(index_path);
    const bool do_fsync = update_durability_fsync_artifacts(manifest.durability_mode);
    if (manifest.last_committed_lsn != 0) {
        throw std::runtime_error("cannot compact while L0 has committed records");
    }
    if (manifest.live_count == 0) {
        throw std::runtime_error("compaction cannot publish an empty base index");
    }

    impl::CompactionBaseMetadata meta = impl::load_compaction_base_metadata(index_path);
    if ((uint32_t)meta.n != manifest.base_n) {
        throw std::runtime_error("compaction base_n does not match manifest");
    }
    if (manifest.live_count > (uint32_t)std::numeric_limits<int>::max()) {
        throw std::runtime_error("compaction live_count exceeds v1 INT_MAX limit");
    }
    std::shared_ptr<const LiveTable> live =
        impl::load_compaction_live_table(index_path, manifest);
    std::vector<std::shared_ptr<const SealedDeltaSegment>> sealed_deltas =
        impl::load_compaction_sealed_deltas(index_path, manifest, meta);
    uint32_t M = meta.num_subspaces;
    if (M == 0) throw std::runtime_error("compaction requires IVF subspace indices");
    CompactionOptions effective_options = options;
    if (effective_options.reorder) {
        if (effective_options.io_rows == 0 || effective_options.bucket_buffer_rows == 0 ||
            effective_options.rewrite_workers <= 0) {
            throw std::runtime_error("compaction reorder compaction options must be positive");
        }
        effective_options.reorder_config.pid_reorder = true;
        effective_options.reorder_config.num_subspaces = (int)M;
        effective_options.reorder_config.rabitq_k = meta.rabitq_k;
    }

    UpdateWalReplayResult replay =
        replay_update_wal(impl::join_index_path(index_path, manifest.wal_path));
    DirectMap direct_map = impl::rebuild_compaction_direct_map(
        index_path, manifest, meta, sealed_deltas, replay, &live);
    std::vector<std::pair<uint32_t, uint32_t>> live_entries =
        impl::build_compaction_live_entries_preserving_base_order(
            manifest, meta, sealed_deltas, *live, direct_map);

    const uint32_t missing_new_pid = std::numeric_limits<uint32_t>::max();
    uint64_t dense_pid_map_limit = std::max<uint64_t>(
        1024, (uint64_t)live_entries.size() * 8);
    bool use_dense_pid_map = (uint64_t)manifest.max_pid <= dense_pid_map_limit;
    std::vector<uint32_t> old_pid_to_new_dense;
    SparseDirectMapTable old_pid_to_new_sparse;
    if (use_dense_pid_map) {
        old_pid_to_new_dense.assign((size_t)manifest.max_pid, missing_new_pid);
    } else {
        old_pid_to_new_sparse.reserve(live_entries.size() * 2 + 1);
    }
    auto add_old_to_new_pid = [&](uint32_t old_pid, uint32_t new_pid) {
        if (use_dense_pid_map) {
            if ((size_t)old_pid >= old_pid_to_new_dense.size()) {
                throw std::runtime_error("DirectMap PID is outside compaction PID range");
            }
            uint32_t& mapped = old_pid_to_new_dense[(size_t)old_pid];
            if (mapped != missing_new_pid) {
                throw std::runtime_error("DirectMap contains duplicate PID during compaction");
            }
            mapped = new_pid;
            return;
        }
        if (!old_pid_to_new_sparse.insert_new(old_pid, new_pid)) {
            throw std::runtime_error("DirectMap contains duplicate PID during compaction");
        }
    };
    auto lookup_new_pid = [&](uint32_t old_pid, uint32_t& new_pid) -> bool {
        if (use_dense_pid_map) {
            if ((size_t)old_pid >= old_pid_to_new_dense.size()) return false;
            uint32_t mapped = old_pid_to_new_dense[(size_t)old_pid];
            if (mapped == missing_new_pid) return false;
            new_pid = mapped;
            return true;
        }
        return old_pid_to_new_sparse.lookup(old_pid, new_pid);
    };
    for (uint32_t new_pid = 0; (size_t)new_pid < live_entries.size(); ++new_pid) {
        uint32_t old_pid = live_entries[(size_t)new_pid].second;
        if (!live->is_live(old_pid)) {
            throw std::runtime_error("DirectMap references a dead PID during compaction");
        }
        add_old_to_new_pid(old_pid, new_pid);
    }

    std::string staging_dir = impl::choose_compaction_staging_dir(index_path);
    std::string backup_dir;  // set during publish; cleaned in catch if exception
    impl::fsync_parent_dir_if_requested(staging_dir, do_fsync);
    try {
        const std::vector<std::string> required_static = {
            "rotation.bin",
            "rabitq_centroids.bin"
        };
        for (const std::string& name : required_static) {
            impl::copy_file_replace(index_path + "/" + name, staging_dir + "/" + name);
        }
        const std::vector<std::string> optional_static = {
            "rotate_subspace.flag",
            "subspace_rotation.bin"
        };
        for (const std::string& name : optional_static) {
            std::string src = index_path + "/" + name;
            if (impl::path_exists(src)) {
                impl::copy_file_replace(src, staging_dir + "/" + name);
            }
        }
        for (uint32_t m = 0; m < M; ++m) {
            std::string name = "ivf_subspace_" + std::to_string(m) + ".bin";
            impl::copy_file_replace(index_path + "/" + name, staging_dir + "/" + name);
        }

        int new_n = (int)live_entries.size();
        std::ifstream base_verify_in(index_path + "/verify_vectors.fbin", std::ios::binary);
        if (!base_verify_in.is_open()) {
            throw std::runtime_error("cannot open base verify artifact for compaction");
        }
        std::ifstream base_rabitq_in(index_path + "/rabitq_bin_data.bin", std::ios::binary);
        if (!base_rabitq_in.is_open()) {
            throw std::runtime_error("cannot open base RaBitQ artifact for compaction");
        }
        std::vector<float> base_verify_buffer((size_t)meta.verify_dim);
        std::vector<char> base_rabitq_buffer((size_t)meta.rabitq_record_bytes);

        auto find_delta_row = [&](uint32_t pid,
                                  const SealedDeltaSegment*& segment_out,
                                  uint32_t& local_row_out) {
            for (const auto& segment : sealed_deltas) {
                if (!segment || !segment->contains_pid(pid)) continue;
                segment_out = segment.get();
                local_row_out = segment->local_row_for_pid(pid);
                return true;
            }
            return false;
        };

        auto write_compaction_reorder_payload_inputs =
            [&](const std::string& verify_path,
                const std::string& assignments_path,
                const std::string& rabitq_path) {
            std::ofstream verify_out(verify_path, std::ios::binary | std::ios::trunc);
            if (!verify_out.is_open()) {
                throw std::runtime_error("cannot create compaction reorder verify input: " + verify_path);
            }
            std::ofstream assignments_out(assignments_path, std::ios::binary | std::ios::trunc);
            if (!assignments_out.is_open()) {
                throw std::runtime_error("cannot create compaction reorder assignments input: " +
                                         assignments_path);
            }
            std::ofstream rabitq_out(rabitq_path, std::ios::binary | std::ios::trunc);
            if (!rabitq_out.is_open()) {
                throw std::runtime_error("cannot create compaction reorder RaBitQ input: " + rabitq_path);
            }
            int dim = meta.verify_dim;
            int reserved = 0;
            int record_bytes = meta.rabitq_record_bytes;
            verify_out.write(reinterpret_cast<const char*>(&new_n), sizeof(int));
            verify_out.write(reinterpret_cast<const char*>(&dim), sizeof(int));
            verify_out.write(reinterpret_cast<const char*>(&reserved), sizeof(int));
            assignments_out.write(reinterpret_cast<const char*>(&new_n), sizeof(int));
            rabitq_out.write(reinterpret_cast<const char*>(&new_n), sizeof(int));
            rabitq_out.write(reinterpret_cast<const char*>(&record_bytes), sizeof(int));

            auto write_rabitq_record = [&](const char* record) {
                uint32_t cid = 0;
                std::memcpy(&cid, record, sizeof(uint32_t));
                if (cid >= (uint32_t)meta.rabitq_k) {
                    throw std::runtime_error("compaction reorder RaBitQ assignment out of range");
                }
                assignments_out.write(reinterpret_cast<const char*>(&cid), sizeof(uint32_t));
                rabitq_out.write(record, (size_t)record_bytes);
            };

            for (const auto& entry : live_entries) {
                uint32_t old_pid = entry.second;
                if (old_pid < manifest.base_n) {
                    base_verify_in.clear();
                    base_verify_in.seekg(
                        (std::streamoff)12 +
                        (std::streamoff)old_pid * (std::streamoff)dim *
                            (std::streamoff)sizeof(float));
                    base_verify_in.read(reinterpret_cast<char*>(base_verify_buffer.data()),
                                        base_verify_buffer.size() * sizeof(float));
                    if (!base_verify_in) {
                        throw std::runtime_error("short base verify read during compaction reorder");
                    }
                    verify_out.write(reinterpret_cast<const char*>(base_verify_buffer.data()),
                                     base_verify_buffer.size() * sizeof(float));
                    base_rabitq_in.clear();
                    base_rabitq_in.seekg(
                        (std::streamoff)(2 * sizeof(int)) +
                        (std::streamoff)old_pid * (std::streamoff)record_bytes);
                    base_rabitq_in.read(base_rabitq_buffer.data(), base_rabitq_buffer.size());
                    if (!base_rabitq_in) {
                        throw std::runtime_error("short base RaBitQ read during compaction reorder");
                    }
                    write_rabitq_record(base_rabitq_buffer.data());
                } else {
                    const SealedDeltaSegment* segment = nullptr;
                    uint32_t local_row = 0;
                    if (!find_delta_row(old_pid, segment, local_row)) {
                        throw std::runtime_error("cannot resolve delta payload during compaction reorder");
                    }
                    verify_out.write(reinterpret_cast<const char*>(segment->verify_vector(local_row)),
                                     (size_t)dim * sizeof(float));
                    write_rabitq_record(
                        reinterpret_cast<const char*>(segment->rabitq_record(local_row)));
                }
            }
            if (!verify_out || !assignments_out || !rabitq_out) {
                throw std::runtime_error("failed writing compaction reorder payload inputs");
            }
        };

        auto write_compaction_reorder_perm_to_orig =
            [&](const std::string& raw_perm_path, const std::string& output_path) {
            if (::access(output_path.c_str(), F_OK) == 0) {
                throw std::runtime_error("compaction reorder perm_to_orig output already exists");
            }
            score::impl::disk_reorder_expect_file_size(
                raw_perm_path, (off_t)new_n * (off_t)sizeof(uint32_t));
            int in_fd = score::impl::disk_reorder_open_read(raw_perm_path);
            int out_fd = score::impl::disk_reorder_open_write_exclusive(output_path);
            std::vector<uint32_t> rows(std::max<size_t>(1, effective_options.io_rows));
            try {
                score::impl::disk_reorder_write_all(out_fd, &new_n, sizeof(int), output_path);
                for (int start = 0; start < new_n; start += (int)rows.size()) {
                    int count = std::min<int>((int)rows.size(), new_n - start);
                    score::impl::disk_reorder_read_all(
                        in_fd, rows.data(), (size_t)count * sizeof(uint32_t), raw_perm_path);
                    for (int i = 0; i < count; ++i) {
                        uint32_t row = rows[(size_t)i];
                        if (row >= (uint32_t)live_entries.size()) {
                            throw std::runtime_error("compaction reorder raw perm row out of range");
                        }
                        uint32_t orig = live_entries[(size_t)row].first;
                        score::impl::disk_reorder_write_all(
                            out_fd, &orig, sizeof(uint32_t), output_path);
                    }
                }
            } catch (...) {
                ::close(in_fd);
                ::close(out_fd);
                ::unlink(output_path.c_str());
                throw;
            }
            score::impl::disk_reorder_close_checked(in_fd, raw_perm_path);
            score::impl::disk_reorder_close_checked(out_fd, output_path);
        };

        auto write_compaction_reorder_subspace_assignments =
            [&](uint32_t m, const std::string& path) {
            int K = (int)meta.cells_per_subspace[(size_t)m];
            if (K <= 0 || K > (int)std::numeric_limits<uint16_t>::max()) {
                throw std::runtime_error("compaction reorder posting K exceeds uint16 assignment limit");
            }
            impl::CompactionPostingReader base_postings(
                index_path + "/posting_subspace_" + std::to_string(m) + ".bin");
            if (base_postings.K() != K) {
                throw std::runtime_error("compaction posting K mismatch");
            }
            const uint16_t missing = std::numeric_limits<uint16_t>::max();
            std::vector<uint16_t> assignments((size_t)new_n, missing);
            uint32_t assigned = 0;
            auto assign_old_pid = [&](uint32_t old_pid, uint16_t cell) {
                uint32_t row = 0;
                if (!lookup_new_pid(old_pid, row)) return;
                if (row >= (uint32_t)new_n) {
                    throw std::runtime_error("compaction reorder assignment row out of range");
                }
                uint16_t& slot = assignments[(size_t)row];
                if (slot != missing) {
                    throw std::runtime_error("compaction reorder duplicate row assignment");
                }
                slot = cell;
                ++assigned;
            };

            std::vector<uint32_t> cell_ids;
            for (int cell = 0; cell < K; ++cell) {
                base_postings.read_cell(cell, cell_ids);
                for (uint32_t old_pid : cell_ids) {
                    assign_old_pid(old_pid, (uint16_t)cell);
                }
            }
            for (const auto& segment : sealed_deltas) {
                if (!segment) continue;
                for (int cell = 0; cell < K; ++cell) {
                    const uint32_t* ids = segment->cell_data(m, (uint32_t)cell);
                    size_t count = segment->cell_size(m, (uint32_t)cell);
                    for (size_t i = 0; i < count; ++i) {
                        assign_old_pid(ids[i], (uint16_t)cell);
                    }
                }
            }
            if (assigned != (uint32_t)new_n) {
                throw std::runtime_error("compaction reorder assignments did not cover every live row");
            }
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            if (!out.is_open()) {
                throw std::runtime_error("cannot create compaction reorder assignment: " + path);
            }
            out.write(reinterpret_cast<const char*>(assignments.data()),
                      assignments.size() * sizeof(uint16_t));
            if (!out) throw std::runtime_error("failed writing compaction reorder assignment: " + path);
        };

        if (effective_options.reorder) {
            const std::string verify_input_path = staging_dir + "/verify_vectors.compaction_input.tmp";
            const std::string assignments_input_path =
                staging_dir + "/rabitq_assignments.compaction_input.tmp";
            const std::string bin_input_path = staging_dir + "/rabitq_bin_data.compaction_input.tmp";
            write_compaction_reorder_payload_inputs(
                verify_input_path, assignments_input_path, bin_input_path);

            std::vector<std::string> subspace_assignment_paths;
            subspace_assignment_paths.reserve((size_t)M);
            for (uint32_t m = 0; m < M; ++m) {
                std::string path = staging_dir + "/posting_assign_subspace_" +
                    std::to_string(m) + ".compaction_input.tmp";
                write_compaction_reorder_subspace_assignments(m, path);
                subspace_assignment_paths.push_back(std::move(path));
            }

            const std::string raw_perm_path = staging_dir + "/perm_to_orig.raw.tmp";
            const std::string scratch_perm_path = staging_dir + "/perm_to_orig.scratch.tmp";
            const std::string row_perm_path = staging_dir + "/perm_to_row.compaction_tmp";
            score::impl::DiskPidReorderOptions reorder_options;
            reorder_options.n = new_n;
            reorder_options.M = (int)M;
            reorder_options.assignment_key_range = (size_t)std::max(1, meta.rabitq_k);
            reorder_options.cells_per_subspace = meta.cells_per_subspace;
            reorder_options.subspace_assignment_paths = subspace_assignment_paths;
            reorder_options.rabitq_assignments_path = assignments_input_path;
            reorder_options.perm_to_orig_path = raw_perm_path;
            reorder_options.scratch_perm_path = scratch_perm_path;
            reorder_options.config = effective_options.reorder_config;
            reorder_options.io_rows = effective_options.io_rows;
            reorder_options.bucket_buffer_rows = effective_options.bucket_buffer_rows;
            score::impl::build_pid_reorder_perm_disk_staging(reorder_options);

            score::impl::ReorderedArtifactFinalizeOptions finalize_options;
            finalize_options.n = new_n;
            finalize_options.M = (int)M;
            finalize_options.cells_per_subspace = meta.cells_per_subspace;
            finalize_options.subspace_assignment_paths = subspace_assignment_paths;
            finalize_options.raw_perm_to_orig_path = raw_perm_path;
            finalize_options.verify_input_path = verify_input_path;
            finalize_options.assignments_input_path = assignments_input_path;
            finalize_options.bin_data_input_path = bin_input_path;
            finalize_options.verify_output_path = staging_dir + "/verify_vectors.fbin";
            finalize_options.assignments_output_path = staging_dir + "/rabitq_assignments.bin";
            finalize_options.bin_data_output_path = staging_dir + "/rabitq_bin_data.bin";
            for (uint32_t m = 0; m < M; ++m) {
                finalize_options.posting_output_paths.push_back(
                    staging_dir + "/posting_subspace_" + std::to_string(m) + ".bin");
            }
            finalize_options.final_perm_to_orig_path = row_perm_path;
            finalize_options.io_rows = effective_options.io_rows;
            finalize_options.rewrite_workers = effective_options.rewrite_workers;
            finalize_options.max_cached_record_bytes = 0;
            finalize_options.parallel_buffer_budget_bytes = effective_options.rewrite_buffer_bytes;
            score::impl::finalize_reordered_artifacts_from_staging(finalize_options);
            write_compaction_reorder_perm_to_orig(raw_perm_path, staging_dir + "/perm_to_orig.bin");

            impl::remove_file_best_effort(row_perm_path);
            impl::remove_file_best_effort(raw_perm_path);
            impl::remove_file_best_effort(scratch_perm_path);
            impl::remove_file_best_effort(verify_input_path);
            impl::remove_file_best_effort(assignments_input_path);
            impl::remove_file_best_effort(bin_input_path);
            for (const std::string& path : subspace_assignment_paths) {
                impl::remove_file_best_effort(path);
            }
        } else {

        {
            std::string path = staging_dir + "/verify_vectors.fbin";
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            if (!out.is_open()) throw std::runtime_error("cannot create compaction verify artifact: " + path);
            int dim = meta.verify_dim;
            int reserved = 0;
            out.write(reinterpret_cast<const char*>(&new_n), sizeof(int));
            out.write(reinterpret_cast<const char*>(&dim), sizeof(int));
            out.write(reinterpret_cast<const char*>(&reserved), sizeof(int));
            for (const auto& entry : live_entries) {
                uint32_t old_pid = entry.second;
                if (old_pid < manifest.base_n) {
                    base_verify_in.clear();
                    base_verify_in.seekg(
                        (std::streamoff)12 +
                        (std::streamoff)old_pid * (std::streamoff)dim * (std::streamoff)sizeof(float));
                    base_verify_in.read(reinterpret_cast<char*>(base_verify_buffer.data()),
                                        base_verify_buffer.size() * sizeof(float));
                    if (!base_verify_in) {
                        throw std::runtime_error("short base verify read during compaction");
                    }
                    out.write(reinterpret_cast<const char*>(base_verify_buffer.data()),
                              base_verify_buffer.size() * sizeof(float));
                } else {
                    const SealedDeltaSegment* segment = nullptr;
                    uint32_t local_row = 0;
                    if (!find_delta_row(old_pid, segment, local_row)) {
                        throw std::runtime_error("cannot resolve delta verify payload during compaction");
                    }
                    out.write(reinterpret_cast<const char*>(segment->verify_vector(local_row)),
                              (size_t)dim * sizeof(float));
                }
            }
            if (!out) throw std::runtime_error("failed writing compaction verify artifact: " + path);
        }
        {
            std::string path = staging_dir + "/rabitq_bin_data.bin";
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            if (!out.is_open()) throw std::runtime_error("cannot create compaction RaBitQ artifact: " + path);
            int record_bytes = meta.rabitq_record_bytes;
            out.write(reinterpret_cast<const char*>(&new_n), sizeof(int));
            out.write(reinterpret_cast<const char*>(&record_bytes), sizeof(int));
            for (const auto& entry : live_entries) {
                uint32_t old_pid = entry.second;
                if (old_pid < manifest.base_n) {
                    base_rabitq_in.clear();
                    base_rabitq_in.seekg(
                        (std::streamoff)(2 * sizeof(int)) +
                        (std::streamoff)old_pid * (std::streamoff)record_bytes);
                    base_rabitq_in.read(base_rabitq_buffer.data(), base_rabitq_buffer.size());
                    if (!base_rabitq_in) {
                        throw std::runtime_error("short base RaBitQ read during compaction");
                    }
                    out.write(base_rabitq_buffer.data(), base_rabitq_buffer.size());
                } else {
                    const SealedDeltaSegment* segment = nullptr;
                    uint32_t local_row = 0;
                    if (!find_delta_row(old_pid, segment, local_row)) {
                        throw std::runtime_error("cannot resolve delta RaBitQ payload during compaction");
                    }
                    out.write(reinterpret_cast<const char*>(segment->rabitq_record(local_row)),
                              (size_t)record_bytes);
                }
            }
            if (!out) throw std::runtime_error("failed writing compaction RaBitQ artifact: " + path);
        }
        {
            std::string path = staging_dir + "/perm_to_orig.bin";
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            if (!out.is_open()) throw std::runtime_error("cannot create compaction perm_to_orig: " + path);
            out.write(reinterpret_cast<const char*>(&new_n), sizeof(int));
            for (const auto& entry : live_entries) {
                uint32_t orig = entry.first;
                out.write(reinterpret_cast<const char*>(&orig), sizeof(uint32_t));
            }
            if (!out) throw std::runtime_error("failed writing compaction perm_to_orig: " + path);
        }

        for (uint32_t m = 0; m < M; ++m) {
            int K = (int)meta.cells_per_subspace[(size_t)m];
            impl::CompactionPostingReader base_postings(
                index_path + "/posting_subspace_" + std::to_string(m) + ".bin");
            if (base_postings.K() != K) {
                throw std::runtime_error("compaction posting K mismatch");
            }

            std::vector<int> counts((size_t)K, 0);
            std::vector<uint32_t> cell_ids;
            for (int cell = 0; cell < K; ++cell) {
                base_postings.read_cell(cell, cell_ids);
                for (uint32_t old_pid : cell_ids) {
                    uint32_t ignored_new_pid = 0;
                    if (lookup_new_pid(old_pid, ignored_new_pid)) {
                        ++counts[(size_t)cell];
                    }
                }
            }
            for (const auto& segment : sealed_deltas) {
                if (!segment) continue;
                for (int cell = 0; cell < K; ++cell) {
                    const uint32_t* ids = segment->cell_data(m, (uint32_t)cell);
                    size_t count = segment->cell_size(m, (uint32_t)cell);
                    for (size_t i = 0; i < count; ++i) {
                        uint32_t ignored_new_pid = 0;
                        if (lookup_new_pid(ids[i], ignored_new_pid)) {
                            ++counts[(size_t)cell];
                        }
                    }
                }
            }

            std::vector<int> offsets((size_t)K + 1, 0);
            for (int cell = 0; cell < K; ++cell) {
                offsets[(size_t)cell + 1] = offsets[(size_t)cell] + counts[(size_t)cell];
            }
            if (offsets.back() != new_n) {
                throw std::runtime_error("compaction posting rewrite did not cover every live row");
            }
            std::string path = staging_dir + "/posting_subspace_" + std::to_string(m) + ".bin";
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            if (!out.is_open()) throw std::runtime_error("cannot create compaction posting artifact: " + path);
            out.write(reinterpret_cast<const char*>(&K), sizeof(int));
            out.write(reinterpret_cast<const char*>(&new_n), sizeof(int));
            out.write(reinterpret_cast<const char*>(offsets.data()), offsets.size() * sizeof(int));

            size_t written = 0;
            for (int cell = 0; cell < K; ++cell) {
                int cell_written = 0;
                base_postings.read_cell(cell, cell_ids);
                for (uint32_t old_pid : cell_ids) {
                    uint32_t new_pid = 0;
                    if (!lookup_new_pid(old_pid, new_pid)) continue;
                    out.write(reinterpret_cast<const char*>(&new_pid), sizeof(uint32_t));
                    ++cell_written;
                    ++written;
                }
                for (const auto& segment : sealed_deltas) {
                    if (!segment) continue;
                    const uint32_t* ids = segment->cell_data(m, (uint32_t)cell);
                    size_t count = segment->cell_size(m, (uint32_t)cell);
                    for (size_t i = 0; i < count; ++i) {
                        uint32_t new_pid = 0;
                        if (!lookup_new_pid(ids[i], new_pid)) continue;
                        out.write(reinterpret_cast<const char*>(&new_pid), sizeof(uint32_t));
                        ++cell_written;
                        ++written;
                    }
                }
                if (cell_written != counts[(size_t)cell]) {
                    throw std::runtime_error("compaction posting cell count changed while writing");
                }
            }
            if (written != (size_t)new_n) {
                throw std::runtime_error("compaction posting rewrite wrote wrong row count");
            }
            if (!out) throw std::runtime_error("failed writing compaction posting artifact: " + path);
        }
        }

        // --- Crash-safe publish sequence ---
        // Build the list of base artifacts to publish.
        std::vector<std::string> publish_files = {
            "rotation.bin",
            "rabitq_centroids.bin",
            "verify_vectors.fbin",
            "rabitq_bin_data.bin",
            "perm_to_orig.bin"
        };
        if (effective_options.reorder &&
            impl::path_exists(staging_dir + "/rabitq_assignments.bin")) {
            publish_files.push_back("rabitq_assignments.bin");
        }
        for (const std::string& name : optional_static) {
            if (impl::path_exists(staging_dir + "/" + name)) publish_files.push_back(name);
            else impl::remove_file_best_effort(index_path + "/" + name);
        }
        for (uint32_t m = 0; m < M; ++m) {
            publish_files.push_back("ivf_subspace_" + std::to_string(m) + ".bin");
            publish_files.push_back("posting_subspace_" + std::to_string(m) + ".bin");
        }

        // 1. Fsync all staging files + dir.
        for (const std::string& name : publish_files) {
            impl::fsync_file_if_requested(staging_dir + "/" + name, do_fsync);
        }
        impl::fsync_directory_if_requested(staging_dir, do_fsync);

        // 2. BACKUP: hardlink old base files + manifest (cheap, same inode).
        //    On crash mid-publish, these preserve the pre-compaction state for
        //    automatic rollback on the next load (see recover_if_marker_present).
        backup_dir = impl::choose_compaction_backup_dir(index_path);
        for (const std::string& name : publish_files) {
            std::string live = index_path + "/" + name;
            if (impl::path_exists(live)) {
                impl::hardlink_file(live, backup_dir + "/" + name);
            }
        }
        {
            std::string manifest_live = index_path + "/update_manifest.json";
            if (impl::path_exists(manifest_live)) {
                impl::hardlink_file(manifest_live, backup_dir + "/update_manifest.json");
            }
        }
        impl::fsync_directory_if_requested(backup_dir, do_fsync);

        // 3. Create the publish marker (blocks concurrent query loads).
        impl::CompactionPublishMarker publish_marker(index_path, backup_dir, do_fsync);

        // 4. Rename each new artifact into place (per-file atomic; backup has old).
        for (const std::string& name : publish_files) {
            impl::rename_replace(staging_dir + "/" + name, index_path + "/" + name);
        }
        if (!effective_options.reorder) {
            impl::remove_file_best_effort(index_path + "/rabitq_assignments.bin");
        }
        impl::fsync_directory_if_requested(index_path, do_fsync);

        // 5. Atomic manifest delete (logical query-path commit: load now sees
        //    "no update_manifest.json" -> base-only, ignoring any orphaned delta
        //    files not yet cleaned).
        impl::remove_file_best_effort(index_path + "/update_manifest.json");
        impl::fsync_directory_if_requested(index_path, do_fsync);

        // 6. Remove marker (RECOVERY COMMIT POINT: after this the compaction is
        //    fully committed; a crash here leaves a consistent post-compaction
        //    state — new base + no manifest + orphaned delta files that load
        //    ignores).
        publish_marker.remove();

        // 7. Clean orphaned update artifacts (delta dirs, WAL, checkpoint, etc.).
        //    Moved AFTER marker removal: the manifest is already gone (step 5),
        //    so load sees base-only and ignores these; they're just disk cleanup.
        impl::remove_update_artifacts_best_effort(index_path, manifest, M);

        // 8. Clean backup + staging.
        for (const std::string& name : publish_files) {
            impl::remove_file_best_effort(backup_dir + "/" + name);
        }
        impl::remove_file_best_effort(backup_dir + "/update_manifest.json");
        rmdir(backup_dir.c_str());
        backup_dir.clear();
        rmdir(staging_dir.c_str());
        impl::fsync_directory_if_requested(index_path, do_fsync);
        impl::fsync_parent_dir_if_requested(staging_dir, do_fsync);
        return true;
    } catch (...) {
        for (uint32_t m = 0; m < M; ++m) {
            impl::remove_file_best_effort(staging_dir + "/ivf_subspace_" + std::to_string(m) + ".bin");
            impl::remove_file_best_effort(staging_dir + "/posting_subspace_" + std::to_string(m) + ".bin");
            impl::remove_file_best_effort(staging_dir + "/posting_assign_subspace_" + std::to_string(m) + ".compaction_input.tmp");
        }
        const std::vector<std::string> names = {
            "rotation.bin", "rotate_subspace.flag", "subspace_rotation.bin",
            "rabitq_centroids.bin", "verify_vectors.fbin", "rabitq_bin_data.bin",
            "rabitq_assignments.bin", "perm_to_orig.bin",
            "verify_vectors.compaction_input.tmp", "rabitq_assignments.compaction_input.tmp",
            "rabitq_bin_data.compaction_input.tmp", "perm_to_orig.raw.tmp",
            "perm_to_orig.scratch.tmp", "perm_to_row.compaction_tmp"
        };
        for (const std::string& name : names) impl::remove_file_best_effort(staging_dir + "/" + name);
        rmdir(staging_dir.c_str());
        // Clean backup if it was created before the exception.
        if (!backup_dir.empty()) {
            for (const std::string& name : names) impl::remove_file_best_effort(backup_dir + "/" + name);
            for (uint32_t m = 0; m < M; ++m) {
                impl::remove_file_best_effort(backup_dir + "/ivf_subspace_" + std::to_string(m) + ".bin");
                impl::remove_file_best_effort(backup_dir + "/posting_subspace_" + std::to_string(m) + ".bin");
            }
            impl::remove_file_best_effort(backup_dir + "/update_manifest.json");
            rmdir(backup_dir.c_str());
        }
        throw;
    }
}

inline UpdateCompactionDecision compact_index_by_policy_locked(
    const std::string& index_path,
    const UpdateCompactionPolicy& policy,
    UpdateCompactionEvaluationContext* context = nullptr) {
    UpdateCompactionDecision decision;
    if (!update_manifest_exists(index_path)) return decision;
    UpdateCompactionEvaluationContext local_context;
    if (!context) context = &local_context;

    UpdateManifest manifest = load_update_manifest_json_for_index(index_path);
    impl::UpdateCompactionPolicyEvaluation eval =
        impl::evaluate_compaction_policy(index_path, manifest, policy, context);
    if (eval.flush_l0) {
        decision.flushed_l0 = flush_l0_to_l1_locked(index_path);
    }

    if (!update_manifest_exists(index_path)) return decision;
    manifest = load_update_manifest_json_for_index(index_path);
    eval = impl::evaluate_compaction_policy(index_path, manifest, policy, context);
    if (eval.compaction) {
        decision.compacted = compact_index_locked(index_path);
        if (decision.compacted && context) {
            context->invalidate_base_artifact_bytes();
        }
    }

    return decision;
}

inline UpdateCompactionDecision compact_index_by_policy(
    const std::string& index_path,
    const UpdateCompactionPolicy& policy,
    UpdateCompactionEvaluationContext* context = nullptr) {
    impl::UpdateWriteLock writer_lock(index_path);
    impl::UpdatePublishWriteLock publish_lock(index_path);
    return compact_index_by_policy_locked(index_path, policy, context);
}

inline bool compact_index(const std::string& index_path) {
    if (!update_manifest_exists(index_path)) return false;
    impl::UpdateWriteLock writer_lock(index_path);
    impl::UpdatePublishWriteLock publish_lock(index_path);
    return compact_index_locked(index_path);
}

inline bool compact_index(const std::string& index_path,
                                const CompactionOptions& options) {
    if (!update_manifest_exists(index_path)) return false;
    impl::UpdateWriteLock writer_lock(index_path);
    impl::UpdatePublishWriteLock publish_lock(index_path);
    return compact_index_locked(index_path, options);
}

} // namespace score

#include "update/update_service.h"
