#pragma once

// Declarations for all test_*() entry points (defined in the per-feature
// test_*.cpp files; invoked by tests/test_main.cpp). Global functions
// (they use `using namespace score`, not namespace score).

void test_adaptive_subspace_gather_equivalence();
void test_ambiguous_fbin_detects_as_fbin();
void test_base_live_posting_overlay_filters_dead_pids();
void test_copy_finalize_subspace_adaptive();
void test_finalize_chunk_rows();
void test_fused_no_reorder_artifacts_are_loadable();
void test_fused_no_reorder_rejects_fvecs();
void test_ivf_assign_batch_matches_scalar();
void test_ivf_assign_block_cap();
void test_l0_block_materializes_postings_and_payloads();
void test_large_subspace_gather_equivalence();
void test_late_rotated_source_from_slabs();
void test_compaction_live_entry_order_preserves_base_pid_order();
void test_compaction_reorder_compaction_matches_pid_reorder_key();
void test_pid_reorder_artifact_contract_matches_legacy();
void test_pid_reorder_counting_matches_legacy();
void test_pid_reorder_disk_staging_matches_counting();
void test_pid_reorder_disk_staging_rejects_bad_inputs();
void test_posthoc_reorder_rejects_bad_preserved_assignments();
void test_posthoc_reorder_rejects_subspace_mismatch();
void test_posthoc_reorder_uses_preserved_assignments();
void test_posting_uint32_ids_load_legacy_and_reject_corrupt();
void test_prefetched_vector_block_reader_explicit_start_and_early_destroy();
void test_prefetched_vector_block_reader_partial_final_block_eof();
void test_prefetched_vector_block_reader_propagates_worker_exception();
void test_prefetched_vector_block_reader_rejects_double_start();
void test_prefetched_vector_block_reader_streams_fbin();
void test_preserved_assignment_manifest_rejects_v1();
void test_rabitq();
void test_reordered_artifact_finalize_failure_hides_final_perm();
void test_reordered_artifact_finalize_from_staging();
void test_reordered_artifact_finalize_rejects_bad_perm();
void test_reordered_finalized_artifacts_are_query_loadable();
void test_rotation();
void test_sc_score();
void test_sealed_delta_load_validates_postings();
void test_streaming_block_mib_row_conversion();
void test_streaming_fused_reorder_staging_finalizes_query_index();
void test_streaming_ivf_postings_are_loadable();
void test_streaming_rabitq_writes_artifact_headers();
void test_streaming_verify_matches_batch_rotation();
void test_subspace_repack_stride_equivalence();
void test_uint16_posting_assignment_boundaries();
void test_update_manifest_json_parser_validation();
void test_update_state_direct_map_and_live_table();
void test_update_wal_replay_committed_and_torn_tail();
void test_vector_block_reader_streams_fbin();
void test_vector_block_reader_streams_fvecs();
