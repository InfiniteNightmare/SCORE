#pragma once

#include "update/update_compaction.h"

#include <functional>
#include <mutex>
#include <utility>

namespace score {

// UpdateService is the production update API: it wraps an UpdateIndexWriter
// with a compaction policy, a mutex for multi-thread commit serialization,
// and optional auto-compaction after each commit. (The former
// UpdateIndexOnlineWriter middle layer was merged in here — it had no external
// callers and was a pure pass-through.)
class UpdateService {
public:
    UpdateService(const std::string& index_path,
                  UpdateCompactionPolicy policy = UpdateCompactionPolicy(),
                  std::string durability_mode = "buffered",
                  int threads = 0,
                  const std::vector<uint32_t>& incoming_ids_hint = {},
                  QueryIndexHandle* publish_handle = nullptr,
                  bool auto_compact_after_commit = true)
        : index_path_(index_path),
          policy_(std::move(policy)),
          writer_(index_path,
                  std::move(durability_mode),
                  threads,
                  incoming_ids_hint,
                  publish_handle),
          auto_compact_after_commit_(auto_compact_after_commit) {}

    UpdateService(const UpdateService&) = delete;
    UpdateService& operator=(const UpdateService&) = delete;

    void commit_batch(const UpdateIndexMemoryBatch& batch) {
        std::lock_guard<std::mutex> lock(mutex_);
        writer_.commit_batch(batch);
        if (auto_compact_after_commit_) compact_if_needed_locked();
    }

    UpdateCompactionDecision compact_if_needed() {
        std::lock_guard<std::mutex> lock(mutex_);
        return compact_if_needed_locked();
    }

    UpdateCompactionRunStats compact_until_quiescent(size_t max_passes = 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        UpdateCompactionRunStats stats;
        if (!impl::compaction_policy_has_triggers(policy_)) {
            return stats;
        }
        while (max_passes == 0 || stats.passes < max_passes) {
            UpdateCompactionDecision decision = compact_if_needed_locked();
            stats.record(decision);
            if (!decision.changed() || decision.compacted ||
                writer_.requires_reopen_after_compaction()) {
                break;
            }
        }
        return stats;
    }

    UpdateCompactionDecision flush_delta_segments() {
        std::lock_guard<std::mutex> lock(mutex_);
        UpdateCompactionDecision decision;
        decision = writer_.run_locked_maintenance([&]() {
            UpdateCompactionDecision locked_decision;
            if (!update_manifest_exists(index_path_)) return locked_decision;
            locked_decision.flushed_l0 = flush_l0_to_l1_locked(index_path_);
            return locked_decision;
        });
        return decision;
    }

    UpdateCommitTiming last_commit_timing() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return writer_.last_commit_timing();
    }

    void set_collect_commit_timing(bool enabled) {
        std::lock_guard<std::mutex> lock(mutex_);
        writer_.set_collect_commit_timing(enabled);
    }

    bool collect_commit_timing() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return writer_.collect_commit_timing();
    }

    void flush_buffered_commits() {
        std::lock_guard<std::mutex> lock(mutex_);
        writer_.flush_buffered_commits();
    }

    void set_publish_handle(QueryIndexHandle* publish_handle) {
        std::lock_guard<std::mutex> lock(mutex_);
        writer_.set_publish_handle(publish_handle);
    }

    void set_publish_callback(std::function<void()> publish_callback) {
        std::lock_guard<std::mutex> lock(mutex_);
        writer_.set_publish_callback(std::move(publish_callback));
    }

    const UpdateCompactionPolicy& policy() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return policy_;
    }

    UpdateCompactionPolicy& mutable_policy() {
        std::lock_guard<std::mutex> lock(mutex_);
        return policy_;
    }

    void set_auto_compact_after_commit(bool enabled) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto_compact_after_commit_ = enabled;
    }

    bool auto_compact_after_commit() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return auto_compact_after_commit_;
    }

    bool requires_reopen_after_compaction() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return writer_.requires_reopen_after_compaction();
    }

private:
    UpdateCompactionDecision compact_if_needed_locked() {
        if (!impl::compaction_policy_has_triggers(policy_)) {
            return {};
        }
        impl::UpdateCompactionPolicyEvaluation eval =
            impl::evaluate_compaction_policy(
                index_path_, writer_.manifest(), policy_, &compaction_context_);
        if (!eval.any()) {
            return {};
        }
        UpdateCompactionDecision decision = writer_.run_locked_maintenance([&]() {
            return compact_index_by_policy_locked(index_path_, policy_, &compaction_context_);
        });
        if (decision.compacted) {
            compaction_context_.invalidate_base_artifact_bytes();
        }
        return decision;
    }

    mutable std::mutex mutex_;
    std::string index_path_;
    UpdateCompactionPolicy policy_;
    UpdateCompactionEvaluationContext compaction_context_;
    UpdateIndexWriter writer_;
    bool auto_compact_after_commit_ = true;
};

} // namespace score
