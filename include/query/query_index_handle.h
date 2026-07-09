#pragma once

#include "query/query_engine.h"
#include "query/reader_view.h"
#include "query/base_segment.h"
#include "update/update_manifest.h"
#include "update/update_delta.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <omp.h>

namespace score {
class QueryIndexHandle {
public:
    QueryIndexHandle() = default;
    ~QueryIndexHandle() = default;
    QueryIndexHandle(const QueryIndexHandle&) = delete;
    QueryIndexHandle& operator=(const QueryIndexHandle&) = delete;

    void init(const IndexConfig& config) {
        config_ = config;
        shared_data_.reset();
        current_view_.reset();
        dim_ = 0;
        index_path_.clear();
    }

    // Load the immutable base segment, rotations, and the initial reader view.
    // The heavy on-disk parse is delegated to a temporary QueryEngine; the
    // resulting shared base data and initial snapshot are then shared by every
    // per-thread query executor attached to this handle.
    void load_index(const std::string& index_path) {
        auto loader = std::make_unique<QueryEngine>();
        loader->init(config_);
        loader->load_index(index_path);
        shared_data_ = loader->shared_data();
        config_ = loader->config();
        index_path_ = index_path;
        dim_ = shared_data_->dim;
        std::shared_ptr<const ReaderView> initial = loader->reader_view_shared();
        std::lock_guard<std::mutex> lk(publish_mutex_);
        std::atomic_store(&current_view_, initial);
    }

    // Atomically load the current immutable snapshot. Readers call this without
    // any lock; the returned shared_ptr keeps the snapshot alive even after a
    // newer one is published.
    std::shared_ptr<const ReaderView> snapshot() const {
        return std::atomic_load(&current_view_);
    }

    const IndexConfig& config() const { return config_; }
    const std::string& index_path() const { return index_path_; }
    int dim() const { return dim_; }
    int num_vectors() const { return shared_data_ ? shared_data_->n : 0; }

    // Direct publish of a committed update delta. Reuses the QueryEngine
    // ReaderView-construction helpers: if the delta matches the currently loaded
    // reader, a new snapshot is built incrementally from the in-memory committed
    // L0 blocks; otherwise the view is rebuilt from disk. Returns true when the
    // incremental path was taken, false on fallback reload.
    bool publish_committed_update_delta(const UpdateManifest& manifest,
                                        const CommittedUpdateDelta& delta) {
        if (!shared_data_) {
            throw std::runtime_error(
                "QueryIndexHandle must load an index before publishing update delta");
        }
        std::shared_ptr<const ReaderView> next;
        bool incremental = false;
        {
            std::lock_guard<std::mutex> lk(publish_mutex_);
            std::shared_ptr<const ReaderView> prev = std::atomic_load(&current_view_);
            if (QueryEngine::committed_delta_matches_loaded_reader(
                    shared_data_, manifest, delta, prev.get())) {
                std::vector<size_t> cells_per_subspace =
                    QueryEngine::reader_cells_per_subspace(shared_data_);
                QueryEngine::LoadedSealedDeltas sealed =
                    QueryEngine::load_sealed_deltas_for_reader(
                        shared_data_, manifest, index_path_, prev.get(), cells_per_subspace);
                QueryEngine::validate_l0_pid_floor(manifest);
                QueryEngine::L0ReplayState l0 = QueryEngine::l0_state_from_committed_delta(
                    delta, *prev, cells_per_subspace);
                next = QueryEngine::make_l0_reader_view_from_l0_state(
                    shared_data_, manifest, index_path_, prev.get(), cells_per_subspace,
                    std::move(sealed), std::move(l0), false);
                incremental = true;
            } else {
                next = build_reload_view_locked(prev);
            }
            std::atomic_store(&current_view_, next);
        }
        return incremental;
    }

    // Rebuild the snapshot from disk (used after L0 flush, compaction, or a
    // stale snapshot). Public so maintenance/compaction paths can force a
    // fresh reader.
    void reload_update_view() {
        if (!shared_data_) {
            throw std::runtime_error(
                "QueryIndexHandle must load an index before reloading update view");
        }
        std::lock_guard<std::mutex> lk(publish_mutex_);
        std::shared_ptr<const ReaderView> prev = std::atomic_load(&current_view_);
        std::shared_ptr<const ReaderView> next = build_reload_view_locked(prev);
        std::atomic_store(&current_view_, next);
    }

    QueryResult search(const float* query) {
        std::shared_ptr<const ReaderView> view = std::atomic_load(&current_view_);
        if (!view) {
            throw std::runtime_error("QueryIndexHandle has no loaded snapshot");
        }
        QueryEngine* exec = acquire_executor(view);
        return exec->search(query);
    }

    void search_into(const float* query, QueryResult& result) {
        std::shared_ptr<const ReaderView> view = std::atomic_load(&current_view_);
        if (!view) {
            throw std::runtime_error("QueryIndexHandle has no loaded snapshot");
        }
        QueryEngine* exec = acquire_executor(view);
        exec->search_into(query, result);
    }

    // Concurrent-safe batch search. Each OMP worker binds its own thread-local
    // QueryEngine executor to the atomically-loaded snapshot; no mutable child
    // engine array is shared across concurrent search_batch() calls.
    std::vector<QueryResult> search_batch(const float* queries, int num_queries) {
        std::vector<QueryResult> results(num_queries);
        if (num_queries <= 0) return results;
        std::shared_ptr<const ReaderView> view = std::atomic_load(&current_view_);
        if (!view) {
            throw std::runtime_error("QueryIndexHandle has no loaded snapshot");
        }
        const int dim = dim_;
        const int nt = config_.num_threads > 0 ? config_.num_threads : omp_get_max_threads();
        #pragma omp parallel for num_threads(nt) schedule(dynamic, 1)
        for (int i = 0; i < num_queries; ++i) {
            QueryEngine* exec = acquire_executor(view);
            exec->search_into(queries + (size_t)i * (size_t)dim, results[i]);
        }
        return results;
    }

    // Profile accessors operate on the calling thread's executor. They are
    // accurate for single-thread search (config.num_threads <= 1, where one
    // executor serves the whole batch); a multi-thread search_batch spreads
    // stats across worker executors that the handle does not aggregate.
    void reset_profile_stats() const {
        std::shared_ptr<const ReaderView> v = std::atomic_load(&current_view_);
        if (v) acquire_executor(v)->reset_profile_stats();
    }
    QueryProfileStats collect_profile_stats() const {
        std::shared_ptr<const ReaderView> v = std::atomic_load(&current_view_);
        if (!v) return QueryProfileStats{};
        return acquire_executor(v)->collect_profile_stats();
    }

private:
    // Build a fresh snapshot from disk, mirroring QueryEngine::reload_update_view.
    // Callers hold publish_mutex_ so reloads and direct delta publishes are
    // linearized before publishing current_view_.
    std::shared_ptr<const ReaderView> build_reload_view_locked(
        const std::shared_ptr<const ReaderView>& previous_view) {
        impl::OptionalUpdateReadLock read_lock(index_path_);
        bool has_update_manifest = update_manifest_exists(index_path_);
        if (has_update_manifest) {
            UpdateManifest manifest = load_update_manifest_json_for_index(index_path_);
            if (manifest.base_n != (uint32_t)shared_data_->n) {
                throw std::runtime_error(
                    "update manifest base_n does not match loaded base segment");
            }
            return QueryEngine::make_l0_reader_view(
                shared_data_,
                manifest,
                index_path_,
                previous_view.get());
        }
        return QueryEngine::make_base_reader_view(shared_data_);
    }

    // Acquire (creating or rebinding as needed) a thread-local QueryEngine
    // executor attached to the given snapshot. Each physical thread owns its own
    // executor and its own loaded rotation, so concurrent search()/search_batch()
    // calls never share mutable scratch. The cache is keyed by the base-segment
    // identity, which is unique per loaded handle.
    QueryEngine* acquire_executor(const std::shared_ptr<const ReaderView>& view) const {
        struct CachedExec {
            std::unique_ptr<QueryEngine> engine;
            const BaseSegment* shared_key = nullptr;
            const ReaderView* view = nullptr;
        };
        thread_local CachedExec tls;
        if (tls.shared_key != shared_data_.get()) {
            tls.engine = std::make_unique<QueryEngine>();
            tls.engine->init(config_);
            tls.engine->attach_shared_data(shared_data_, index_path_, false, true, view);
            tls.shared_key = shared_data_.get();
            tls.view = view.get();
        } else if (tls.view != view.get()) {
            tls.engine->install_reader_view(view, false);
            tls.view = view.get();
        }
        return tls.engine.get();
    }

    IndexConfig config_;
    std::string index_path_;
    std::shared_ptr<BaseSegment> shared_data_;
    std::shared_ptr<const ReaderView> current_view_;
    int dim_ = 0;
    mutable std::mutex publish_mutex_;
};
} // namespace score
