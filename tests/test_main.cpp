#include "test_macros.h"
#include "test_registry.h"

int tests_passed = 0;
int tests_failed = 0;

int main() {
    std::cout << "===== SCORE Unit Tests =====" << std::endl;

    test_rotation();
    test_vector_block_reader_streams_fvecs();
    test_vector_block_reader_streams_fbin();
    test_prefetched_vector_block_reader_streams_fbin();
    test_prefetched_vector_block_reader_explicit_start_and_early_destroy();
    test_prefetched_vector_block_reader_propagates_worker_exception();
    test_prefetched_vector_block_reader_rejects_double_start();
    test_prefetched_vector_block_reader_partial_final_block_eof();
    test_ambiguous_fbin_detects_as_fbin();
    test_streaming_block_mib_row_conversion();
    test_streaming_verify_matches_batch_rotation();
    test_streaming_rabitq_writes_artifact_headers();
    test_streaming_ivf_postings_are_loadable();
    test_fused_no_reorder_artifacts_are_loadable();
    test_reordered_finalized_artifacts_are_query_loadable();
    test_posthoc_reorder_uses_preserved_assignments();
    test_posthoc_reorder_rejects_bad_preserved_assignments();
    test_streaming_fused_reorder_staging_finalizes_query_index();
    test_posthoc_reorder_rejects_subspace_mismatch();
    test_fused_no_reorder_rejects_fvecs();
    test_rabitq();
    test_sc_score();
    test_subspace_repack_stride_equivalence();
    test_ivf_assign_block_cap();
    test_ivf_assign_batch_matches_scalar();
    test_late_rotated_source_from_slabs();
    test_finalize_chunk_rows();
    test_copy_finalize_subspace_adaptive();
    test_large_subspace_gather_equivalence();
    test_adaptive_subspace_gather_equivalence();
    test_pid_reorder_counting_matches_legacy();
    test_pid_reorder_artifact_contract_matches_legacy();
    test_pid_reorder_disk_staging_matches_counting();
    test_pid_reorder_disk_staging_rejects_bad_inputs();
    test_reordered_artifact_finalize_from_staging();
    test_reordered_artifact_finalize_failure_hides_final_perm();
    test_reordered_artifact_finalize_rejects_bad_perm();
    test_uint16_posting_assignment_boundaries();
    test_preserved_assignment_manifest_rejects_v1();
    test_posting_uint32_ids_load_legacy_and_reject_corrupt();
    test_update_manifest_json_parser_validation();
    test_update_state_direct_map_and_live_table();
    test_compaction_live_entry_order_preserves_base_pid_order();
    test_compaction_reorder_compaction_matches_pid_reorder_key();
    test_update_wal_replay_committed_and_torn_tail();
    test_l0_block_materializes_postings_and_payloads();
    test_base_live_posting_overlay_filters_dead_pids();
    test_sealed_delta_load_validates_postings();

    std::cout << "\n===== Results =====" << std::endl;
    std::cout << "Passed: " << tests_passed << std::endl;
    std::cout << "Failed: " << tests_failed << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
