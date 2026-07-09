#include "test_macros.h"

void test_update_manifest_json_parser_validation() {
    std::cout << "[Update manifest JSON]" << std::endl;

    const std::string dir = "/tmp/score_update_manifest_" + std::to_string(getpid());
    mkdir(dir.c_str(), 0755);
    write_valid_update_manifest(dir);

    CHECK(update_manifest_exists(dir), "update manifest must be detected");
    UpdateManifest manifest = load_update_manifest_json_for_index(dir);
    CHECK(manifest.format == "score_update_manifest_v1", "update manifest format mismatch");
    CHECK(manifest.generation == 17, "update manifest generation mismatch");
    CHECK(manifest.base_path == ".", "update manifest base path mismatch");
    CHECK(manifest.base_n == 100, "update manifest base_n mismatch");
    CHECK(manifest.max_pid == 112, "update manifest max_pid mismatch");
    CHECK(manifest.live_count == 108, "update manifest live_count mismatch");
    CHECK(manifest.sealed_deltas.size() == 1, "update manifest sealed delta count mismatch");
    CHECK(manifest.sealed_deltas[0].segment_id == "00000015",
          "update manifest sealed delta id mismatch");
    CHECK(manifest.sealed_deltas[0].pid_begin == 100,
          "update manifest sealed delta pid_begin mismatch");
    CHECK(manifest.sealed_deltas[0].artifact_bytes == 0,
          "legacy update manifest delta artifact_bytes should default to zero");
    CHECK(manifest.wal_path == "updates/l0.wal", "update manifest WAL path mismatch");
    CHECK(manifest.direct_map_mode == "dense", "update manifest DirectMap mode mismatch");
    CHECK(manifest.direct_map_optional_for_search,
          "update manifest DirectMap optional flag mismatch");
    CHECK(manifest.durability_mode == "buffered", "update manifest durability mode mismatch");

    {
        std::ofstream bad(dir + "/update_manifest.json", std::ios::trunc);
        bad << "{\n";
        bad << "  \"format\": \"score_update_manifest_v1\",\n";
        bad << "  \"generation\": 18,\n";
        bad << "  \"base\": {\"path\": \".\", \"base_n\": 100},\n";
        bad << "  \"pid\": {\"base_n\": 99, \"max_pid\": 112, \"live_count\": 108},\n";
        bad << "  \"sealed_deltas\": [],\n";
        bad << "  \"growing\": {\"wal_path\": \"updates/l0.wal\",";
        bad << " \"checkpoint_path\": \"updates/l0.checkpoint\"},\n";
        bad << "  \"direct_map\": {\"mode\": \"dense\", \"path\": \"updates/direct_map.bin\"},\n";
        bad << "  \"durability\": {\"mode\": \"buffered\", \"last_committed_lsn\": 2,";
        bad << " \"last_fsynced_lsn\": 2}\n";
        bad << "}\n";
    }
    bool rejected_bad_manifest = false;
    try {
        (void)load_update_manifest_json_for_index(dir);
    } catch (const std::runtime_error&) {
        rejected_bad_manifest = true;
    }
    CHECK(rejected_bad_manifest, "update manifest parser must reject base_n mismatch");

    auto write_manifest_with_delta = [&](uint32_t live_count,
                                         uint32_t delta_pid_begin,
                                         uint32_t delta_pid_count,
                                         uint32_t delta_live_count) {
        std::ofstream bad(dir + "/update_manifest.json", std::ios::trunc);
        bad << "{\n";
        bad << "  \"format\": \"score_update_manifest_v1\",\n";
        bad << "  \"generation\": 18,\n";
        bad << "  \"base\": {\"path\": \".\", \"base_n\": 100},\n";
        bad << "  \"pid\": {\"base_n\": 100, \"max_pid\": 112, \"live_count\": "
            << live_count << "},\n";
        bad << "  \"sealed_deltas\": [\n";
        bad << "    {\"segment_id\": \"bad_delta\", \"path\": \"deltas/bad_delta\",";
        bad << " \"pid_begin\": " << delta_pid_begin << ", \"pid_count\": "
            << delta_pid_count << ", \"live_count\": " << delta_live_count << "}\n";
        bad << "  ],\n";
        bad << "  \"growing\": {\"wal_path\": \"updates/l0.wal\",";
        bad << " \"checkpoint_path\": \"updates/l0.checkpoint\"},\n";
        bad << "  \"direct_map\": {\"mode\": \"dense\", \"path\": \"updates/direct_map.bin\"},\n";
        bad << "  \"durability\": {\"mode\": \"buffered\", \"last_committed_lsn\": 2,";
        bad << " \"last_fsynced_lsn\": 2}\n";
        bad << "}\n";
    };

    write_manifest_with_delta(108, 99, 10, 8);
    bool rejected_base_overlap_delta = false;
    try {
        (void)load_update_manifest_json_for_index(dir);
    } catch (const std::runtime_error&) {
        rejected_base_overlap_delta = true;
    }
    CHECK(rejected_base_overlap_delta,
          "update manifest parser must reject delta PID ranges that overlap base");

    write_manifest_with_delta(7, 100, 10, 8);
    bool rejected_delta_live_overflow = false;
    try {
        (void)load_update_manifest_json_for_index(dir);
    } catch (const std::runtime_error&) {
        rejected_delta_live_overflow = true;
    }
    CHECK(rejected_delta_live_overflow,
          "update manifest parser must reject sealed live_count above global live_count");

    manifest.generation = 19;
    manifest.max_pid = 113;
    manifest.live_count = 109;
    manifest.last_committed_lsn = 9184;
    manifest.last_fsynced_lsn = 9184;
    manifest.sealed_deltas[0].artifact_bytes = 12345;
    write_update_manifest_json_for_index(manifest, dir);
    UpdateManifest rewritten = load_update_manifest_json_for_index(dir);
    CHECK(rewritten.generation == 19, "atomic update manifest rewrite generation mismatch");
    CHECK(rewritten.max_pid == 113 && rewritten.live_count == 109,
          "atomic update manifest rewrite PID state mismatch");
    CHECK(rewritten.sealed_deltas[0].artifact_bytes == 12345,
          "atomic update manifest rewrite must preserve delta artifact bytes");
    CHECK(access((update_manifest_path(dir) + ".tmp." + std::to_string((uint64_t)getpid())).c_str(), F_OK) != 0,
          "atomic update manifest rewrite must not leave a temp file");

    manifest.durability_mode = "buffered";
    manifest.last_committed_lsn = 9190;
    manifest.last_fsynced_lsn = 9184;
    write_update_manifest_json_for_index(manifest, dir);
    UpdateManifest pending_buffered = load_update_manifest_json_for_index(dir);
    CHECK(pending_buffered.last_committed_lsn == 9190 &&
              pending_buffered.last_fsynced_lsn == 9184,
          "buffered manifest parser must allow pending unfsynced commits");

    manifest.durability_mode = "strict";
    write_update_manifest_json_for_index(manifest, dir);
    bool rejected_unsynced_strict = false;
    try {
        (void)load_update_manifest_json_for_index(dir);
    } catch (const std::runtime_error&) {
        rejected_unsynced_strict = true;
    }
    CHECK(rejected_unsynced_strict,
          "strict manifest parser must reject unfsynced committed LSN");

    std::remove((dir + "/update_manifest.json").c_str());
    rmdir(dir.c_str());
    TEST("Update manifest JSON parser validation"); PASS();
}

void test_update_state_direct_map_and_live_table() {
    std::cout << "[Update state]" << std::endl;

    std::vector<uint8_t> keep_mask;
    std::vector<uint32_t> kept_ids;
    CHECK(!impl::build_last_wins_keep_mask(
              std::vector<uint32_t>({1, 2, 3, 4}), keep_mask, kept_ids),
          "ordered unique upsert IDs should skip last-wins coalescing");
    CHECK(keep_mask.empty() && kept_ids.empty(),
          "ordered unique upsert coalescing should not allocate output masks");
    CHECK(impl::build_last_wins_keep_mask(
              std::vector<uint32_t>({5, 7, 5, 9, 7}), keep_mask, kept_ids),
          "duplicate upsert IDs should trigger last-wins coalescing");
    CHECK(kept_ids == std::vector<uint32_t>({5, 9, 7}),
          "last-wins coalescing should preserve final versions in input order");
    CHECK(keep_mask == std::vector<uint8_t>({0, 0, 1, 1, 1}),
          "last-wins coalescing keep mask mismatch");

    std::shared_ptr<const LiveTable> base_live =
        LiveTable::from_live_pids(8, {0, 1, 2, 3, 4, 5, 6, 7});
    std::shared_ptr<const LiveTable> immutable_live =
        base_live->with_updates(11, {8, 10}, {2, 6});
    LiveTable mutable_live = *base_live;
    mutable_live.apply_updates_in_place(11, {8, 10}, {2, 6});
    CHECK(mutable_live.max_pid() == immutable_live->max_pid() &&
              mutable_live.live_count() == immutable_live->live_count(),
          "in-place LiveTable update must match immutable counters");
    for (uint32_t pid = 0; pid < immutable_live->max_pid(); ++pid) {
        CHECK(mutable_live.is_live(pid) == immutable_live->is_live(pid),
              "in-place LiveTable update must match immutable bit state");
    }

    DirectMap dense_map = DirectMap::dense(4);
    dense_map.set(2, 42);
    uint32_t erased_pid = 0;
    CHECK(dense_map.erase(2, &erased_pid) && erased_pid == 42,
          "dense DirectMap erase should return removed PID");
    CHECK(!dense_map.contains(2) && dense_map.live_size() == 0,
          "dense DirectMap erase should remove mapping");
    DirectMap sparse_map = DirectMap::sparse();
    sparse_map.set(9001, 77);
    erased_pid = 0;
    CHECK(sparse_map.erase(9001, &erased_pid) && erased_pid == 77,
          "sparse DirectMap erase should return removed PID");
    CHECK(!sparse_map.contains(9001) && sparse_map.live_size() == 0,
          "sparse DirectMap erase should remove mapping");

    std::vector<uint32_t> large_unsorted_ids;
    large_unsorted_ids.reserve(impl::kUpdateLastWinsSmallScanRows + 4);
    large_unsorted_ids.push_back(100);
    for (uint32_t id = 200;
         id < 200 + (uint32_t)impl::kUpdateLastWinsSmallScanRows + 1;
         ++id) {
        large_unsorted_ids.push_back(id);
    }
    large_unsorted_ids.push_back(100);
    CHECK(impl::build_last_wins_keep_mask(
              large_unsorted_ids, keep_mask, kept_ids),
          "large duplicate upsert IDs should trigger hash last-wins coalescing");
    CHECK(!kept_ids.empty() && kept_ids.back() == 100,
          "large last-wins coalescing should keep final duplicate position");
    CHECK(keep_mask.size() == large_unsorted_ids.size() &&
              keep_mask.front() == 0 && keep_mask.back() == 1,
          "large last-wins coalescing keep mask mismatch");

    bool rejected_large_duplicate_insert = false;
    try {
        impl::validate_no_insert_update_id_conflicts(large_unsorted_ids, {});
    } catch (const std::runtime_error&) {
        rejected_large_duplicate_insert = true;
    }
    CHECK(rejected_large_duplicate_insert,
          "large insert validation should reject duplicate orig_id without unordered_set");

    std::vector<uint32_t> large_unique_insert_ids;
    large_unique_insert_ids.reserve(impl::kUpdateLastWinsSmallScanRows + 2);
    for (uint32_t id = 500;
         id < 500 + (uint32_t)impl::kUpdateLastWinsSmallScanRows + 2;
         ++id) {
        large_unique_insert_ids.push_back(id);
    }
    bool rejected_large_insert_upsert_overlap = false;
    try {
        impl::validate_no_insert_update_id_conflicts(
            large_unique_insert_ids, {large_unique_insert_ids.back()});
    } catch (const std::runtime_error&) {
        rejected_large_insert_upsert_overlap = true;
    }
    CHECK(rejected_large_insert_upsert_overlap,
          "large insert validation should reject upsert overlap without unordered_set");
    CHECK(impl::adaptive_update_threads_for_work(500000, 48) == 1,
          "tiny update work should run single-threaded");
    CHECK(impl::adaptive_update_threads_for_work(3ull * 1000ull * 1000ull, 48) == 4,
          "small update work should cap at four threads");
    CHECK(impl::adaptive_update_threads_for_work(26ull * 1000ull * 1000ull, 48) == 8,
          "small-batch update work should cap at eight threads");
    CHECK(impl::adaptive_update_threads_for_work(100ull * 1000ull * 1000ull, 48) == 16,
          "medium update work should cap at sixteen threads");
    CHECK(impl::adaptive_update_threads_for_work(300ull * 1000ull * 1000ull, 48) == 16,
          "medium-large update work should still cap at sixteen threads");
    CHECK(impl::adaptive_update_threads_for_work(700ull * 1000ull * 1000ull, 48) == 32,
          "large update work should cap at thirty-two threads");
    CHECK(impl::adaptive_update_threads_for_work(2ull * 1000ull * 1000ull * 1000ull, 48) == 32,
          "multi-billion update work should still cap at thirty-two threads");
    CHECK(impl::adaptive_update_threads_for_work(8ull * 1000ull * 1000ull * 1000ull, 48) == 48,
          "very large update work should keep configured thread budget");
    CHECK(impl::adaptive_update_threads_for_work(26ull * 1000ull * 1000ull, 8) == 8,
          "adaptive update threads must honor configured thread cap");

    const std::string policy_dir =
        "/tmp/score_delta_policy_" + std::to_string(getpid());
    mkdir(policy_dir.c_str(), 0755);

    mkdir((policy_dir + "/deltas").c_str(), 0755);
    auto write_policy_delta_bytes = [&](const std::string& rel_path, size_t bytes) {
        std::string dir = policy_dir + "/" + rel_path;
        mkdir(dir.c_str(), 0755);
        std::ofstream out(dir + "/delta_orig_ids.bin", std::ios::binary | std::ios::trunc);
        CHECK(out.is_open(), "policy delta byte test must create artifact file");
        std::vector<char> payload(bytes, 'x');
        if (!payload.empty()) out.write(payload.data(), (std::streamsize)payload.size());
        CHECK((bool)out, "policy delta byte test must write artifact file");
    };
    write_policy_delta_bytes("deltas/large_a", 4096);
    write_policy_delta_bytes("deltas/small_a", 128);
    write_policy_delta_bytes("deltas/small_b", 160);
    write_policy_delta_bytes("deltas/large_b", 4096);
    {
        std::ofstream base_out(policy_dir + "/rabitq_bin_data.bin",
                               std::ios::binary | std::ios::trunc);
        CHECK(base_out.is_open(), "policy byte-fraction test must create base artifact");
        std::vector<char> base_payload(1024, 'b');
        base_out.write(base_payload.data(), (std::streamsize)base_payload.size());
        CHECK((bool)base_out, "policy byte-fraction test must write base artifact");
    }
    UpdateManifest l1_policy_manifest;
    l1_policy_manifest.base_n = 100;
    l1_policy_manifest.sealed_deltas = {
        {"large_a", "deltas/large_a", 5, 1, 1},
        {"small_a", "deltas/small_a", 6, 1, 1},
        {"small_b", "deltas/small_b", 7, 1, 1},
        {"large_b", "deltas/large_b", 8, 1, 1}
    };
    CHECK(impl::sealed_delta_artifact_bytes_for_policy(
              policy_dir, l1_policy_manifest) > 0,
          "L1 byte-fraction policy test should see sealed delta bytes");
    CHECK(impl::base_artifact_bytes_for_policy(policy_dir) >= 1024,
          "L1 byte-fraction policy test should see base artifact bytes");
    impl::remove_delta_segment_dir_best_effort(policy_dir, "deltas/large_a", 0);
    impl::remove_delta_segment_dir_best_effort(policy_dir, "deltas/small_a", 0);
    impl::remove_delta_segment_dir_best_effort(policy_dir, "deltas/small_b", 0);
    impl::remove_delta_segment_dir_best_effort(policy_dir, "deltas/large_b", 0);
    std::remove((policy_dir + "/rabitq_bin_data.bin").c_str());
    rmdir((policy_dir + "/deltas").c_str());
    std::remove((policy_dir + "/direct_map.bin").c_str());
    rmdir(policy_dir.c_str());

    DirectMap dense = DirectMap::dense(2);
    uint32_t pid = 0;
    CHECK(!dense.lookup(1, pid), "dense DirectMap missing lookup should fail");
    dense.set(0, 10);
    dense.set(2, 12);
    dense.set(0, 11);
    dense.set(5, 15);
    CHECK(dense.live_size() == 3, "dense DirectMap live size mismatch");
    CHECK(dense.lookup(0, pid) && pid == 11, "dense DirectMap overwrite mismatch");
    CHECK(dense.lookup(5, pid) && pid == 15, "dense DirectMap grow lookup mismatch");
    uint32_t old_pid = 0;
    CHECK(dense.set_with_old_pid(0, 21, &old_pid),
          "dense DirectMap replace should report an old PID");
    CHECK(old_pid == 11 && dense.lookup(0, pid) && pid == 21,
          "dense DirectMap replace should return and publish the old PID");
    CHECK(!dense.set_with_old_pid(6, 16, &old_pid),
          "dense DirectMap insert should not report an old PID");
    CHECK(dense.live_size() == 4, "dense DirectMap replace/insert live size mismatch");
    CHECK(dense.erase(2), "dense DirectMap erase should remove existing entry");
    CHECK(!dense.lookup(2, pid), "dense DirectMap erased entry should be missing");
    CHECK(!dense.erase(2), "dense DirectMap erase should report missing entry");
    CHECK(dense.live_size() == 3, "dense DirectMap live size after erase mismatch");
    bool rejected_sentinel = false;
    try {
        dense.set(7, DirectMap::kMissingPid);
    } catch (const std::invalid_argument&) {
        rejected_sentinel = true;
    }
    CHECK(rejected_sentinel, "DirectMap must reject UINT32_MAX as a valid PID");
    std::vector<uint32_t> dense_live = dense.live_pids_sorted_by_orig();
    CHECK(dense_live.size() == 3 && dense_live[0] == 21 && dense_live[1] == 15 &&
              dense_live[2] == 16,
          "dense DirectMap live PIDs should be sorted by orig_id");
    const std::string dense_map_path = "/tmp/score_direct_map_dense_" + std::to_string(getpid()) + ".bin";
    const std::string dense_map_tmp_path = dense_map_path + ".tmp." + std::to_string((uint64_t)getpid());
    dense.save_binary(dense_map_path);
    CHECK(access(dense_map_tmp_path.c_str(), F_OK) != 0,
          "dense DirectMap atomic save must not leave a temp file");
    DirectMap dense_loaded = DirectMap::load_binary(dense_map_path);
    CHECK(dense_loaded.mode() == DirectMap::Mode::Dense, "dense DirectMap load mode mismatch");
    CHECK(dense_loaded.live_size() == dense.live_size(), "dense DirectMap load live size mismatch");
    CHECK(dense_loaded.lookup(0, pid) && pid == 21, "dense DirectMap load pid 0 mismatch");
    CHECK(dense_loaded.lookup(5, pid) && pid == 15, "dense DirectMap load pid 5 mismatch");
    dense.set(1, 31);
    dense.save_binary(dense_map_path);
    CHECK(access(dense_map_tmp_path.c_str(), F_OK) != 0,
          "dense DirectMap atomic rewrite must not leave a temp file");
    DirectMap dense_rewritten = DirectMap::load_binary(dense_map_path);
    CHECK(dense_rewritten.live_size() == dense.live_size(),
          "dense DirectMap rewritten live size mismatch");
    CHECK(dense_rewritten.lookup(1, pid) && pid == 31,
          "dense DirectMap rewritten pid mismatch");
    std::remove(dense_map_path.c_str());

    DirectMap sparse = DirectMap::sparse();
    sparse.set(1000, 42);
    sparse.set(2, 7);
    sparse.set(1000, 43);
    CHECK(sparse.live_size() == 2, "sparse DirectMap live size mismatch");
    CHECK(sparse.lookup(1000, pid) && pid == 43, "sparse DirectMap overwrite mismatch");
    CHECK(sparse.set_with_old_pid(1000, 44, &old_pid),
          "sparse DirectMap replace should report an old PID");
    CHECK(old_pid == 43 && sparse.lookup(1000, pid) && pid == 44,
          "sparse DirectMap replace should return and publish the old PID");
    CHECK(!sparse.set_with_old_pid(2000, 45, &old_pid),
          "sparse DirectMap insert should not report an old PID");
    CHECK(sparse.live_size() == 3, "sparse DirectMap replace/insert live size mismatch");
    CHECK(sparse.erase(2), "sparse DirectMap erase should remove existing entry");
    CHECK(!sparse.lookup(2, pid), "sparse DirectMap erased entry should be missing");
    const std::string sparse_map_path = "/tmp/score_direct_map_sparse_" + std::to_string(getpid()) + ".bin";
    const std::string sparse_map_tmp_path = sparse_map_path + ".tmp." + std::to_string((uint64_t)getpid());
    sparse.save_binary(sparse_map_path);
    CHECK(access(sparse_map_tmp_path.c_str(), F_OK) != 0,
          "sparse DirectMap atomic save must not leave a temp file");
    DirectMap sparse_loaded = DirectMap::load_binary(sparse_map_path);
    CHECK(sparse_loaded.mode() == DirectMap::Mode::Sparse, "sparse DirectMap load mode mismatch");
    CHECK(sparse_loaded.live_size() == 2, "sparse DirectMap load live size mismatch");
    CHECK(sparse_loaded.lookup(1000, pid) && pid == 44, "sparse DirectMap load pid mismatch");
    sparse.set(5, 55);
    sparse.save_binary(sparse_map_path);
    CHECK(access(sparse_map_tmp_path.c_str(), F_OK) != 0,
          "sparse DirectMap atomic rewrite must not leave a temp file");
    DirectMap sparse_rewritten = DirectMap::load_binary(sparse_map_path);
    CHECK(sparse_rewritten.live_size() == 3, "sparse DirectMap rewritten live size mismatch");
    CHECK(sparse_rewritten.lookup(5, pid) && pid == 55,
          "sparse DirectMap rewritten pid mismatch");
    std::remove(sparse_map_path.c_str());

    const std::string duplicate_sparse_path =
        "/tmp/score_direct_map_sparse_duplicate_" + std::to_string(getpid()) + ".bin";
    {
        std::ofstream out(duplicate_sparse_path, std::ios::binary | std::ios::trunc);
        uint32_t magic = 0x314d4453u;
        uint32_t version = 1;
        uint32_t mode = 2;
        uint64_t live = 1;
        uint64_t count = 2;
        uint32_t orig = 42;
        uint32_t pid_a = 100;
        uint32_t pid_b = 101;
        out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));
        out.write(reinterpret_cast<const char*>(&mode), sizeof(mode));
        out.write(reinterpret_cast<const char*>(&live), sizeof(live));
        out.write(reinterpret_cast<const char*>(&count), sizeof(count));
        out.write(reinterpret_cast<const char*>(&orig), sizeof(orig));
        out.write(reinterpret_cast<const char*>(&pid_a), sizeof(pid_a));
        out.write(reinterpret_cast<const char*>(&orig), sizeof(orig));
        out.write(reinterpret_cast<const char*>(&pid_b), sizeof(pid_b));
    }
    bool rejected_duplicate_sparse_orig = false;
    try {
        (void)DirectMap::load_binary(duplicate_sparse_path);
    } catch (const std::runtime_error&) {
        rejected_duplicate_sparse_orig = true;
    }
    CHECK(rejected_duplicate_sparse_orig,
          "sparse DirectMap load must reject duplicate orig_id entries");
    std::remove(duplicate_sparse_path.c_str());

    auto live0 = LiveTable::empty(5);
    CHECK(live0->max_pid() == 5 && live0->live_count() == 0,
          "empty LiveTable metadata mismatch");
    CHECK(!live0->is_live(0) && !live0->is_live(4),
          "empty LiveTable should mark all PIDs dead");

    auto live1 = live0->with_updates(8, {1, 6}, {});
    CHECK(live0->live_count() == 0 && !live0->is_live(1),
          "LiveTable update must not mutate previous snapshot");
    CHECK(live1->max_pid() == 8 && live1->live_count() == 2,
          "LiveTable grow/set metadata mismatch");
    CHECK(live1->is_live(1) && live1->is_live(6) && !live1->is_live(7),
          "LiveTable grow/set bit mismatch");
    CHECK(live1->range_all_live(1, 2), "LiveTable single live range mismatch");
    CHECK(!live1->range_all_live(1, 7), "LiveTable sparse range should not be all live");
    CHECK(!live1->range_all_live(7, 9), "LiveTable out-of-range range should be rejected");

    auto live2 = live1->with_updates(8, {2, 2}, {1, 1});
    CHECK(live2->live_count() == 2, "LiveTable duplicate update live_count mismatch");
    CHECK(!live2->is_live(1) && live2->is_live(2) && live2->is_live(6),
          "LiveTable set/clear bit mismatch");
    const std::string live_table_path = "/tmp/score_live_table_" + std::to_string(getpid()) + ".bin";
    const std::string live_table_tmp_path = live_table_path + ".tmp." + std::to_string((uint64_t)getpid());
    live2->save_binary(live_table_path);
    CHECK(access(live_table_tmp_path.c_str(), F_OK) != 0,
          "LiveTable atomic save must not leave a temp file");
    auto live_loaded = LiveTable::load_binary(live_table_path);
    CHECK(live_loaded->max_pid() == live2->max_pid() &&
              live_loaded->live_count() == live2->live_count(),
          "LiveTable loaded metadata mismatch");
    CHECK(!live_loaded->is_live(1) && live_loaded->is_live(2) && live_loaded->is_live(6),
          "LiveTable loaded bit mismatch");
    auto live3 = live2->with_updates(10, {8}, {6});
    live3->save_binary(live_table_path);
    CHECK(access(live_table_tmp_path.c_str(), F_OK) != 0,
          "LiveTable atomic rewrite must not leave a temp file");
    auto live_rewritten = LiveTable::load_binary(live_table_path);
    CHECK(live_rewritten->max_pid() == 10 && live_rewritten->live_count() == 2,
          "LiveTable rewritten metadata mismatch");
    CHECK(live_rewritten->is_live(2) && !live_rewritten->is_live(6) &&
              live_rewritten->is_live(8),
          "LiveTable rewritten bit mismatch");
    auto all_live_range = LiveTable::all_live(130);
    CHECK(all_live_range->range_all_live(0, 130),
          "LiveTable all-live range should cross word boundaries");
    auto tail_dead_range = all_live_range->with_updates(130, {}, {129});
    CHECK(!tail_dead_range->range_all_live(64, 130),
          "LiveTable range_all_live should honor tail bits");
    std::remove(live_table_path.c_str());

    bool rejected_shrink = false;
    try {
        (void)live2->with_updates(7, {}, {});
    } catch (const std::invalid_argument&) {
        rejected_shrink = true;
    }
    CHECK(rejected_shrink, "LiveTable must reject shrinking max_pid");

    std::vector<uint32_t> posting_ids = {0, 1, 2, 6, 7, 9};
    CHECK(impl::count_live_posting_ids(posting_ids.data(), posting_ids.size(), nullptr) ==
              posting_ids.size(),
          "posting live count without LiveTable should return raw count");
    CHECK(impl::count_live_posting_ids(posting_ids.data(), posting_ids.size(), live2.get()) == 2,
          "posting live count must skip dead and out-of-range PIDs");

    TEST("DirectMap and LiveTable snapshots"); PASS();
}

void test_compaction_live_entry_order_preserves_base_pid_order() {
    std::cout << "[Update compaction]" << std::endl;

    UpdateManifest manifest;
    manifest.base_n = 5;
    manifest.max_pid = 7;
    manifest.live_count = 5;

    impl::CompactionBaseMetadata meta;
    meta.n = 5;
    meta.perm_to_orig = {40, 10, 30, 20, 0};

    UpdateWalInsert dead_delta;
    dead_delta.orig_id = 50;
    dead_delta.global_pid = 5;
    dead_delta.verify_rotated = {1.0f, 2.0f};
    dead_delta.rabitq_record = {1, 2, 3, 4};
    dead_delta.subspace_cells = {0};

    UpdateWalInsert live_delta = dead_delta;
    live_delta.orig_id = 60;
    live_delta.global_pid = 6;
    live_delta.subspace_cells = {1};

    std::vector<std::shared_ptr<const SealedDeltaSegment>> sealed_deltas;
    sealed_deltas.push_back(SealedDeltaSegment::from_inserts(
        "compaction_order",
        {dead_delta, live_delta},
        {2},
        2,
        4));

    std::shared_ptr<const LiveTable> live =
        LiveTable::from_live_pids(7, {0, 2, 3, 4, 6});

    DirectMap direct_map = DirectMap::dense(60);
    direct_map.set(40, 0);
    direct_map.set(30, 2);
    direct_map.set(20, 3);
    direct_map.set(0, 4);
    direct_map.set(60, 6);

    std::vector<std::pair<uint32_t, uint32_t>> entries =
        impl::build_compaction_live_entries_preserving_base_order(
            manifest, meta, sealed_deltas, *live, direct_map);
    std::vector<std::pair<uint32_t, uint32_t>> expected = {
        {40, 0},
        {30, 2},
        {20, 3},
        {0, 4},
        {60, 6}
    };
    CHECK(entries == expected,
          "compaction must preserve reordered base PID order before appending delta rows");

    TEST("Compaction live entry order preserves base PID order"); PASS();
}

void test_compaction_reorder_compaction_matches_pid_reorder_key() {
    std::cout << "[Update compaction reorder]" << std::endl;

    const std::string suffix = std::to_string(getpid());
    const std::string input_path = "/tmp/score_compaction_reorder_input_" + suffix + ".fbin";
    const std::string insert_path = "/tmp/score_compaction_reorder_insert_" + suffix + ".fbin";
    const std::string insert_ids_path = "/tmp/score_compaction_reorder_insert_ids_" + suffix + ".txt";
    const std::string index_dir = "/tmp/score_compaction_reorder_index_" + suffix;
    const int n = 28;
    const int insert_n = 2;
    const int live_n = n + insert_n;
    const int dim = 64;
    const int M = 4;
    const int ivf_k = 4;
    const int rabitq_k = 4;

    std::vector<float> rows((size_t)n * (size_t)dim);
    for (int i = 0; i < n; ++i) {
        for (int d = 0; d < dim; ++d) {
            rows[(size_t)i * (size_t)dim + (size_t)d] =
                (float)(((i * 37 + d * 11 + 17) % 251) - 120) * 0.01f;
        }
    }
    write_test_fbin(input_path, rows, n, dim);
    mkdir(index_dir.c_str(), 0755);

    RandomRotation rotation;
    rotation.init(dim, 42, RotationType::FhtKac);
    StreamingRabitqOptions rabitq_options;
    rabitq_options.rabitq_k = rabitq_k;
    rabitq_options.rabitq_seed = 42;
    rabitq_options.block_rows = 8;
    rabitq_options.sample_rows = 16;
    rabitq_options.train_iters = 2;
    StreamingIvfPostingOptions ivf_options;
    ivf_options.M = M;
    ivf_options.ivf_k = ivf_k;
    ivf_options.block_rows = rabitq_options.block_rows;
    ivf_options.sample_rows = rabitq_options.sample_rows;
    ivf_options.train_iters = rabitq_options.train_iters;
    ivf_options.subspace_workers = 2;
    write_streaming_no_reorder_fused_artifacts(input_path, index_dir, rotation,
                                               rabitq_options, ivf_options);

    std::vector<uint32_t> top_assignments((size_t)live_n);
    {
        std::ifstream in(index_dir + "/rabitq_assignments.bin", std::ios::binary);
        int count = 0;
        in.read(reinterpret_cast<char*>(&count), sizeof(int));
        CHECK(count == n, "compaction reorder test base assignment count mismatch");
        in.read(reinterpret_cast<char*>(top_assignments.data()), (size_t)n * sizeof(uint32_t));
        CHECK((bool)in, "compaction reorder test short base assignment read");
    }

    PostingLists base_postings;
    base_postings.load(index_dir, M);
    std::vector<size_t> cells_per_subspace((size_t)M);
    std::vector<int> submajor_cell_ids((size_t)M * (size_t)live_n, -1);
    for (int m = 0; m < M; ++m) {
        cells_per_subspace[(size_t)m] = (size_t)base_postings.K(m);
        CHECK(base_postings.K(m) == ivf_k, "compaction reorder test posting K mismatch");
        for (int cell = 0; cell < base_postings.K(m); ++cell) {
            const uint32_t* ids = base_postings.cell_data(m, cell);
            size_t count = base_postings.cell_size(m, cell);
            for (size_t i = 0; i < count; ++i) {
                CHECK(ids[i] < (uint32_t)n, "compaction reorder test base posting PID out of range");
                submajor_cell_ids[(size_t)m * (size_t)live_n + (size_t)ids[i]] = cell;
            }
        }
    }
    std::vector<uint32_t> noorder_orig((size_t)live_n);
    for (int i = 0; i < n; ++i) noorder_orig[(size_t)i] = (uint32_t)i;

    std::vector<float> inserts((size_t)insert_n * (size_t)dim);
    for (int i = 0; i < insert_n; ++i) {
        for (int d = 0; d < dim; ++d) {
            inserts[(size_t)i * (size_t)dim + (size_t)d] =
                (float)((((i + 29) * 41 + d * 7) % 223) - 90) * 0.0125f;
        }
    }
    write_test_fbin(insert_path, inserts, insert_n, dim);
    {
        std::ofstream ids(insert_ids_path);
        ids << "1000\n1001\n";
    }
    UpdateIndexBatch insert_batch;
    insert_batch.insert_fbin = insert_path;
    insert_batch.insert_ids = insert_ids_path;
    insert_batch.durability_mode = "none";
    insert_batch.threads = 2;
    commit_update_batch(index_dir, insert_batch);

    UpdateManifest manifest = load_update_manifest_json_for_index(index_dir);
    UpdateWalReplayResult replay =
        replay_update_wal(index_dir + "/" + manifest.wal_path);
    int next_row = n;
    for (const UpdateWalBatch& batch : replay.committed_batches) {
        for (const UpdateWalRecord& record : batch.records) {
            if (record.type != UpdateWalRecord::Type::Insert) continue;
            CHECK(next_row < live_n, "compaction reorder test too many WAL inserts");
            uint32_t cid = std::numeric_limits<uint32_t>::max();
            std::memcpy(&cid, record.insert.rabitq_record.data(), sizeof(uint32_t));
            top_assignments[(size_t)next_row] = cid;
            noorder_orig[(size_t)next_row] = record.insert.orig_id;
            CHECK(record.insert.subspace_cells.size() == (size_t)M,
                  "compaction reorder test insert subspace count mismatch");
            for (int m = 0; m < M; ++m) {
                submajor_cell_ids[(size_t)m * (size_t)live_n + (size_t)next_row] =
                    (int)record.insert.subspace_cells[(size_t)m];
            }
            ++next_row;
        }
    }
    CHECK(next_row == live_n, "compaction reorder test WAL insert count mismatch");

    IndexConfig reorder_config;
    reorder_config.num_subspaces = M;
    reorder_config.rabitq_k = rabitq_k;
    std::vector<uint32_t> expected_rows = impl::build_pid_reorder_perm_counting(
        top_assignments,
        submajor_cell_ids.data(),
        live_n,
        M,
        cells_per_subspace,
        reorder_config);
    std::vector<uint32_t> expected_orig((size_t)live_n);
    for (int i = 0; i < live_n; ++i) {
        expected_orig[(size_t)i] = noorder_orig[(size_t)expected_rows[(size_t)i]];
    }

    CompactionOptions options;
    options.reorder = true;
    options.io_rows = 8;
    options.bucket_buffer_rows = 2;
    options.rewrite_workers = 2;
    bool changed = compact_index(index_dir, options);
    CHECK(changed, "compaction reorder compaction should publish a compact base");
    CHECK(access((index_dir + "/update_manifest.json").c_str(), F_OK) != 0,
          "compaction reorder compaction must remove update manifest");
    CHECK(access((index_dir + "/rabitq_assignments.bin").c_str(), F_OK) == 0,
          "compaction reorder compaction should publish reordered assignments");

    std::vector<uint32_t> final_perm((size_t)live_n);
    {
        std::ifstream in(index_dir + "/perm_to_orig.bin", std::ios::binary);
        int count = 0;
        in.read(reinterpret_cast<char*>(&count), sizeof(int));
        CHECK(count == live_n, "compaction reorder perm_to_orig count mismatch");
        in.read(reinterpret_cast<char*>(final_perm.data()), final_perm.size() * sizeof(uint32_t));
        CHECK((bool)in, "compaction reorder short perm_to_orig read");
    }
    CHECK(final_perm == expected_orig,
          "compaction reorder compaction perm_to_orig must match clean PID reorder key");

    IndexConfig query_config;
    query_config.num_subspaces = M;
    query_config.rabitq_k = rabitq_k;
    query_config.num_threads = 1;
    QueryEngine engine;
    engine.init(query_config);
    engine.load_index(index_dir);
    CHECK(!engine.reader_view().has_deltas(),
          "compaction reorder compacted reader should be base-only");
    CHECK(engine.reader_view().base_n == (uint32_t)live_n,
          "compaction reorder compacted base_n mismatch");
    auto inserted_it = std::find(engine.perm_to_orig().begin(),
                                 engine.perm_to_orig().end(),
                                 1000u);
    CHECK(inserted_it != engine.perm_to_orig().end(),
          "compaction reorder compacted perm_to_orig should keep inserted ID");
    uint32_t inserted_pid =
        (uint32_t)std::distance(engine.perm_to_orig().begin(), inserted_it);
    ResolvedRow resolved;
    CHECK(engine.row_resolver().resolve(inserted_pid, resolved),
          "compaction reorder RowResolver should resolve inserted compacted PID");
    CHECK(resolved.segment == ResolvedRow::SegmentKind::Base &&
              resolved.output_id == 1000u,
          "compaction reorder RowResolver must map compacted PID through perm_to_orig");

    const char* base_names[] = {
        "rotation.bin", "rotate_subspace.flag", "subspace_rotation.bin",
        "rabitq_centroids.bin", "verify_vectors.fbin", "rabitq_assignments.bin",
        "rabitq_bin_data.bin", "perm_to_orig.bin", "update_writer.lock",
        "update_publish.lock", "compaction.publish"
    };
    for (const char* name : base_names) std::remove((index_dir + "/" + name).c_str());
    for (int m = 0; m < M; ++m) {
        std::remove((index_dir + "/ivf_subspace_" + std::to_string(m) + ".bin").c_str());
        std::remove((index_dir + "/posting_subspace_" + std::to_string(m) + ".bin").c_str());
        std::remove((index_dir + "/posting_assign_subspace_" + std::to_string(m) + ".compaction_input.tmp").c_str());
    }
    rmdir(index_dir.c_str());
    std::remove(input_path.c_str());
    std::remove(insert_path.c_str());
    std::remove(insert_ids_path.c_str());

    TEST("Compaction reorder compaction matches clean PID reorder key"); PASS();
}

void test_update_wal_replay_committed_and_torn_tail() {
    std::cout << "[Update WAL]" << std::endl;

    const std::string path = "/tmp/score_update_wal_" + std::to_string(getpid()) + ".wal";
    UpdateWalHeader header;
    header.generation = 3;
    header.base_n = 10;
    header.num_subspaces = 2;
    header.verify_dim = 3;
    header.rabitq_record_bytes = 5;

    {
        std::vector<uint8_t> payload(4099);
        for (size_t i = 0; i < payload.size(); ++i) {
            payload[i] = (uint8_t)((i * 131u + 17u) & 0xffu);
        }
        uint64_t lsn = 0x1122334455667788ull;
        uint8_t type = (uint8_t)UpdateWalRecord::Type::Insert;
        uint32_t payload_bytes = (uint32_t)payload.size();
        uint32_t sw_crc = impl::crc32c_update_sw(0xffffffffu, &lsn, sizeof(lsn));
        sw_crc = impl::crc32c_update_sw(sw_crc, &type, sizeof(type));
        sw_crc = impl::crc32c_update_sw(sw_crc, &payload_bytes, sizeof(payload_bytes));
        sw_crc = impl::crc32c_update_sw(sw_crc, payload.data(), payload.size());
        sw_crc = impl::crc32c_finish(sw_crc);

        std::array<impl::ByteSpan, 3> spans = {{
            {payload.data(), 17},
            {payload.data() + 17, 2113},
            {payload.data() + 2130, payload.size() - 2130}
        }};
        uint32_t frame_crc = impl::frame_crc(lsn, type, payload_bytes, payload);
        uint32_t segmented_crc =
            impl::frame_crc_segments(lsn, type, payload_bytes, spans.data(), spans.size());
        CHECK(frame_crc == sw_crc, "WAL CRC32C implementation must match software reference");
        CHECK(segmented_crc == frame_crc, "WAL segmented CRC32C must match contiguous payload");
        TEST("Update WAL CRC32C implementation equivalence"); PASS();
    }

    {
        UpdateWalWriter writer(path, header);
        UpdateWalInsert insert;
        insert.orig_id = 100;
        insert.global_pid = 10;
        insert.verify_rotated = {1.0f, 2.0f, 3.0f};
        insert.rabitq_record = {1, 2, 3, 4, 5};
        insert.subspace_cells = {7, 8};
        writer.append_insert(1, insert);

        UpdateWalDelete tombstone;
        tombstone.orig_id = 4;
        tombstone.old_global_pid = 4;
        writer.append_delete(2, tombstone);

        UpdateWalCommit commit;
        commit.lsn = 3;
        commit.batch_record_count = 2;
        commit.max_pid = 11;
        commit.live_count = 10;
        writer.append_commit(commit);

        UpdateWalInsert uncommitted = insert;
        uncommitted.orig_id = 101;
        uncommitted.global_pid = 11;
        writer.append_insert(4, uncommitted);
        writer.flush();
    }

    UpdateWalReplayResult replay = replay_update_wal(path);
    CHECK(replay.header.generation == 3, "WAL replay generation mismatch");
    CHECK(replay.header.base_n == 10, "WAL replay base_n mismatch");
    CHECK(replay.committed_batches.size() == 1,
          "WAL replay must ignore uncommitted trailing records");
    CHECK(replay.last_committed_lsn == 3, "WAL replay committed LSN mismatch");
    CHECK(replay.max_pid == 11 && replay.live_count == 10,
          "WAL replay committed counts mismatch");
    const UpdateWalBatch& batch = replay.committed_batches[0];
    CHECK(batch.records.size() == 2, "WAL replay batch record count mismatch");
    CHECK(batch.records[0].type == UpdateWalRecord::Type::Insert,
          "WAL replay first record type mismatch");
    CHECK(batch.records[0].insert.orig_id == 100 &&
          batch.records[0].insert.global_pid == 10,
          "WAL replay insert IDs mismatch");
    CHECK(batch.records[0].insert.verify_rotated == std::vector<float>({1.0f, 2.0f, 3.0f}),
          "WAL replay insert verify payload mismatch");
    CHECK(batch.records[0].insert.rabitq_record == std::vector<uint8_t>({1, 2, 3, 4, 5}),
          "WAL replay insert RaBitQ payload mismatch");
    CHECK(batch.records[0].insert.subspace_cells == std::vector<uint16_t>({7, 8}),
          "WAL replay insert cell payload mismatch");
    CHECK(batch.records[1].type == UpdateWalRecord::Type::Delete,
          "WAL replay second record type mismatch");
    CHECK(batch.records[1].tombstone.orig_id == 4 &&
          batch.records[1].tombstone.old_global_pid == 4,
          "WAL replay delete payload mismatch");
    CHECK(!replay.ignored_tail, "complete uncommitted frame should not count as torn tail");

    UpdateWalReplayResult metadata_replay = replay_update_wal_metadata(path);
    CHECK(metadata_replay.committed_batches.size() == 1,
          "metadata WAL replay committed batch count mismatch");
    const UpdateWalBatch& metadata_batch = metadata_replay.committed_batches[0];
    CHECK(metadata_batch.records.size() == 2,
          "metadata WAL replay batch record count mismatch");
    CHECK(metadata_batch.records[0].type == UpdateWalRecord::Type::Insert,
          "metadata WAL replay insert record type mismatch");
    CHECK(metadata_batch.records[0].insert.orig_id == 100 &&
          metadata_batch.records[0].insert.global_pid == 10,
          "metadata WAL replay insert IDs mismatch");
    CHECK(metadata_batch.records[0].insert.verify_rotated.empty() &&
          metadata_batch.records[0].insert.rabitq_record.empty() &&
          metadata_batch.records[0].insert.subspace_cells.empty(),
          "metadata WAL replay must omit insert payload vectors");
    CHECK(metadata_batch.records[1].type == UpdateWalRecord::Type::Delete &&
          metadata_batch.records[1].tombstone.orig_id == 4 &&
          metadata_batch.records[1].tombstone.old_global_pid == 4,
          "metadata WAL replay delete payload mismatch");

    const std::string reused_pid_path =
        "/tmp/score_update_wal_reuse_" + std::to_string(getpid()) + ".wal";
    {
        UpdateWalWriter writer(reused_pid_path, header);
        UpdateWalInsert insert;
        insert.orig_id = 120;
        insert.global_pid = 10;
        insert.verify_rotated = {1.0f, 2.0f, 3.0f};
        insert.rabitq_record = {1, 2, 3, 4, 5};
        insert.subspace_cells = {0, 1};
        writer.append_insert(1, insert);
        UpdateWalCommit first_commit;
        first_commit.lsn = 2;
        first_commit.batch_record_count = 1;
        first_commit.max_pid = 11;
        first_commit.live_count = 11;
        writer.append_commit(first_commit);

        UpdateWalInsert reused = insert;
        reused.orig_id = 121;
        reused.global_pid = 10;
        writer.append_insert(3, reused);
        UpdateWalCommit reused_commit;
        reused_commit.lsn = 4;
        reused_commit.batch_record_count = 1;
        reused_commit.max_pid = 12;
        reused_commit.live_count = 12;
        writer.append_commit(reused_commit);
        writer.flush();
    }
    bool rejected_reused_pid = false;
    try {
        (void)replay_update_wal(reused_pid_path, header.base_n, true);
    } catch (const std::runtime_error&) {
        rejected_reused_pid = true;
    }
    CHECK(rejected_reused_pid, "WAL replay must reject reused insert PIDs");

    bool rejected_reused_pid_metadata = false;
    try {
        (void)replay_update_wal_metadata(reused_pid_path, header.base_n, true);
    } catch (const std::runtime_error&) {
        rejected_reused_pid_metadata = true;
    }
    CHECK(rejected_reused_pid_metadata,
          "metadata WAL replay must reject reused insert PIDs");
    std::remove(reused_pid_path.c_str());

    UpdateWalHeader mismatched_header = header;
    mismatched_header.generation = header.generation + 1;
    bool rejected_append_header_mismatch = false;
    try {
        UpdateWalWriter writer(path, mismatched_header, true);
        (void)writer;
    } catch (const std::runtime_error&) {
        rejected_append_header_mismatch = true;
    }
    CHECK(rejected_append_header_mismatch,
          "WAL append must reject an existing file with a mismatched header");

    {
        std::ofstream out(path, std::ios::binary | std::ios::app);
        uint64_t partial_lsn = 99;
        out.write(reinterpret_cast<const char*>(&partial_lsn), sizeof(partial_lsn));
    }
    UpdateWalReplayResult torn = replay_update_wal(path);
    CHECK(torn.committed_batches.size() == 1,
          "WAL replay must preserve committed batch before torn tail");
    CHECK(torn.ignored_tail, "WAL replay must report ignored torn tail");

    {
        UpdateWalWriter writer(path, header);
        UpdateWalInsert insert;
        insert.orig_id = 200;
        insert.global_pid = 10;
        insert.verify_rotated = {4.0f, 5.0f, 6.0f};
        insert.rabitq_record = {6, 7, 8, 9, 10};
        insert.subspace_cells = {1, 0};
        writer.append_insert(1, insert);
        UpdateWalCommit duplicate_lsn_commit;
        duplicate_lsn_commit.lsn = 1;
        duplicate_lsn_commit.batch_record_count = 1;
        duplicate_lsn_commit.max_pid = 11;
        duplicate_lsn_commit.live_count = 11;
        writer.append_commit(duplicate_lsn_commit);
        writer.flush();
    }
    bool rejected_duplicate_lsn = false;
    try {
        (void)replay_update_wal(path);
    } catch (const std::runtime_error&) {
        rejected_duplicate_lsn = true;
    }
    CHECK(rejected_duplicate_lsn, "WAL replay must reject non-increasing frame LSNs");

    {
        UpdateWalWriter writer(path, header);
        UpdateWalInsert insert;
        insert.orig_id = 300;
        insert.global_pid = 10;
        insert.verify_rotated = {7.0f, 8.0f, 9.0f};
        insert.rabitq_record = {11, 12, 13, 14, 15};
        insert.subspace_cells = {0, 1};
        writer.append_insert(1, insert);
        UpdateWalCommit commit;
        commit.lsn = 2;
        commit.batch_record_count = 1;
        commit.max_pid = 11;
        commit.live_count = 11;
        writer.append_commit(commit);
        writer.flush();
    }
    {
        std::ofstream out(path, std::ios::binary | std::ios::app);
        uint64_t bad_lsn = 3;
        uint8_t bad_type = (uint8_t)UpdateWalRecord::Type::Insert;
        uint32_t bad_payload_bytes = std::numeric_limits<uint32_t>::max();
        out.write(reinterpret_cast<const char*>(&bad_lsn), sizeof(bad_lsn));
        out.write(reinterpret_cast<const char*>(&bad_type), sizeof(bad_type));
        out.write(reinterpret_cast<const char*>(&bad_payload_bytes), sizeof(bad_payload_bytes));
    }
    UpdateWalReplayResult oversized_tail = replay_update_wal(path);
    CHECK(oversized_tail.committed_batches.size() == 1,
          "WAL replay must preserve committed batch before oversized tail frame");
    CHECK(oversized_tail.last_committed_lsn == 2,
          "WAL replay oversized tail committed LSN mismatch");
    CHECK(oversized_tail.ignored_tail,
          "WAL replay must ignore oversized malformed tail without allocating payload");

    std::remove(path.c_str());
    TEST("Update WAL replay committed batches and torn tail"); PASS();
}

void test_l0_block_materializes_postings_and_payloads() {
    std::cout << "[L0 block]" << std::endl;

    std::vector<UpdateWalInsert> inserts(3);
    for (int i = 0; i < 3; ++i) {
        inserts[(size_t)i].orig_id = (uint32_t)(100 + i);
        inserts[(size_t)i].global_pid = (uint32_t)(50 + i);
        inserts[(size_t)i].verify_rotated = {
            (float)i, (float)i + 0.25f
        };
        inserts[(size_t)i].rabitq_record = {
            (uint8_t)(10 + i), (uint8_t)(20 + i), (uint8_t)(30 + i)
        };
    }
    inserts[0].subspace_cells = {1, 0};
    inserts[1].subspace_cells = {2, 1};
    inserts[2].subspace_cells = {1, 1};

    std::shared_ptr<const L0Block> block =
        L0Block::from_inserts(77, inserts, {3, 2}, 2, 3);
    CHECK(block->commit_lsn() == 77, "L0Block commit LSN mismatch");
    CHECK(block->pid_begin() == 50 && block->pid_end() == 53,
          "L0Block PID range mismatch");
    CHECK(block->pid_count() == 3, "L0Block PID count mismatch");
    CHECK(block->num_subspaces() == 2, "L0Block subspace count mismatch");
    CHECK(block->orig_id(1) == 101 && block->pid(1) == 51,
          "L0Block row id payload mismatch");
    CHECK(block->verify_vector(2)[0] == 2.0f && block->verify_vector(2)[1] == 2.25f,
          "L0Block verify payload mismatch");
    CHECK(block->rabitq_record(1)[0] == 11 && block->rabitq_record(1)[2] == 31,
          "L0Block RaBitQ payload mismatch");

    CHECK(block->cell_size(0, 0) == 0, "L0Block empty cell size mismatch");
    CHECK(block->cell_size(0, 1) == 2, "L0Block subspace 0 cell 1 size mismatch");
    const uint32_t* s0c1 = block->cell_data(0, 1);
    CHECK(s0c1[0] == 50 && s0c1[1] == 52,
          "L0Block subspace 0 cell 1 postings mismatch");
    CHECK(block->cell_size(0, 2) == 1, "L0Block subspace 0 cell 2 size mismatch");
    CHECK(block->cell_data(0, 2)[0] == 51,
          "L0Block subspace 0 cell 2 postings mismatch");
    CHECK(block->cell_size(1, 0) == 1 && block->cell_data(1, 0)[0] == 50,
          "L0Block subspace 1 cell 0 postings mismatch");
    CHECK(block->cell_size(1, 1) == 2,
          "L0Block subspace 1 cell 1 size mismatch");
    const uint32_t* s1c1 = block->cell_data(1, 1);
    CHECK(s1c1[0] == 51 && s1c1[1] == 52,
          "L0Block subspace 1 cell 1 postings mismatch");

    std::vector<UpdateWalInsert> wide_cell_inserts(1);
    wide_cell_inserts[0].orig_id = 123;
    wide_cell_inserts[0].global_pid = 900;
    wide_cell_inserts[0].verify_rotated = {0.0f, 1.0f};
    wide_cell_inserts[0].rabitq_record = {1, 2, 3};
    wide_cell_inserts[0].subspace_cells = {65534u};
    bool accepted_uint16_l0_cell = false;
    try {
        std::shared_ptr<const L0Block> wide_cell_block =
            L0Block::from_inserts(79, wide_cell_inserts, {65535}, 2, 3);
        accepted_uint16_l0_cell =
            wide_cell_block && wide_cell_block->row_cell(0, 0) == (uint16_t)65534u;
    } catch (...) {
        accepted_uint16_l0_cell = false;
    }
    CHECK(accepted_uint16_l0_cell,
          "L0Block row cells should accept uint16 assignment boundary");
    bool rejected_oversized_l0_cell_range = false;
    try {
        (void)L0Block::from_inserts(80, wide_cell_inserts, {65536}, 2, 3);
    } catch (const std::invalid_argument&) {
        rejected_oversized_l0_cell_range = true;
    }
    CHECK(rejected_oversized_l0_cell_range,
          "L0Block row cells should reject K beyond uint16 assignment boundary");

    std::shared_ptr<const LiveTable> l0_live =
        LiveTable::from_live_pids(53, {50, 52});
    UnsealedDeltaView growing;
    growing.committed_blocks.push_back(block);
    growing.build_lookup({3, 2}, l0_live.get());
    CHECK(growing.has_live_cell_counts(),
          "UnsealedDeltaView should cache live cell counts when L0 has tombstones");
    CHECK(growing.live_cell_size(0, 1) == 2 && growing.live_cell_size(0, 2) == 0,
          "UnsealedDeltaView live cell counts must skip dead L0 PIDs");
    CHECK(growing.live_cell_size(1, 0) == 1 && growing.live_cell_size(1, 1) == 1,
          "UnsealedDeltaView live cell counts must be per subspace");

    std::vector<UpdateWalInsert> tail_inserts = inserts;
    tail_inserts.resize(2);
    for (size_t i = 0; i < tail_inserts.size(); ++i) {
        tail_inserts[i].global_pid = 53 + (uint32_t)i;
        tail_inserts[i].orig_id = 200 + (uint32_t)i;
    }
    tail_inserts[0].subspace_cells = {0, 1};
    tail_inserts[1].subspace_cells = {1, 0};
    std::shared_ptr<const L0Block> tail_block =
        L0Block::from_inserts(78, tail_inserts, {3, 2}, 2, 3);
    std::shared_ptr<const LiveTable> incremental_live =
        LiveTable::from_live_pids(55, {50, 52, 53, 54});

    UnsealedDeltaView full_growing;
    full_growing.committed_blocks.push_back(block);
    full_growing.committed_blocks.push_back(tail_block);
    full_growing.build_lookup({3, 2}, incremental_live.get());

    UnsealedDeltaView incremental_growing;
    std::vector<std::shared_ptr<const L0Block>> tail_blocks = {tail_block};
    incremental_growing.build_incremental_from_previous(
        growing, std::move(tail_blocks), {3, 2}, incremental_live.get());
    CHECK(!incremental_growing.posting_layers.empty(),
          "incremental L0 lookup should retain reusable posting layers");
    CHECK(incremental_growing.posting_layers.size() <= 2,
          "one incremental tail should add at most one posting layer");
    auto collect_growing_cell = [](const UnsealedDeltaView& view,
                                   uint32_t m,
                                   uint32_t cell) {
        std::vector<uint32_t> ids;
        view.for_each_cell_span(m, cell, [&](UnsealedDeltaView::CellSpan span) {
            ids.insert(ids.end(), span.data, span.data + span.count);
        });
        return ids;
    };
    for (uint32_t m = 0; m < 2; ++m) {
        uint32_t K = m == 0 ? 3 : 2;
        for (uint32_t cell = 0; cell < K; ++cell) {
            std::vector<uint32_t> full_ids = collect_growing_cell(full_growing, m, cell);
            std::vector<uint32_t> incremental_ids =
                collect_growing_cell(incremental_growing, m, cell);
            CHECK(full_ids.size() == incremental_ids.size(),
                  "incremental L0 lookup cell size must match full rebuild");
            CHECK(full_growing.cell_size(m, cell) ==
                      incremental_growing.cell_size(m, cell),
                  "incremental L0 cached cell count must match full rebuild");
            for (size_t i = 0; i < full_ids.size(); ++i) {
                CHECK(full_ids[i] == incremental_ids[i],
                      "incremental L0 lookup postings must match full rebuild");
            }
            CHECK(full_growing.live_cell_size(m, cell) ==
                      incremental_growing.live_cell_size(m, cell),
                  "incremental L0 lookup live cell count must match full rebuild");
        }
    }

    std::shared_ptr<const LiveTable> l0_deleted_live =
        LiveTable::from_live_pids(55, {50, 52, 54});
    std::vector<uint32_t> l0_dead_pids = {53};
    UnsealedDeltaView l0_incremental_counts;
    l0_incremental_counts.committed_blocks = incremental_growing.committed_blocks;
    l0_incremental_counts.build_pid_lookup(l0_deleted_live.get());
    bool incrementally_rebuilt_live_counts =
        l0_incremental_counts.try_rebuild_live_cell_counts_incremental(
            incremental_growing,
            incremental_growing.committed_blocks.size(),
            {3, 2},
            incremental_live.get(),
            l0_deleted_live.get(),
            &l0_dead_pids);
    CHECK(incrementally_rebuilt_live_counts,
          "L0 live cell counts should update incrementally from dead PID row cells");
    CHECK(l0_incremental_counts.live_cell_size(0, 0) == 0 &&
              l0_incremental_counts.live_cell_size(0, 1) == 3,
          "incremental L0 live counts must decrement dead PID subspace 0 cell");
    CHECK(l0_incremental_counts.live_cell_size(1, 0) == 2 &&
              l0_incremental_counts.live_cell_size(1, 1) == 1,
          "incremental L0 live counts must decrement dead PID subspace 1 cell");

    UnsealedDeltaView l0_delete_growing;
    l0_delete_growing.build_incremental_from_previous(
        incremental_growing,
        {},
        {3, 2},
        l0_deleted_live.get(),
        false,
        true,
        incremental_live.get(),
        &l0_dead_pids);
    CHECK(!l0_delete_growing.all_pids_live(),
          "incremental L0 delete should refresh growing liveness");
    CHECK(l0_delete_growing.has_live_cell_counts(),
          "incremental L0 delete should rebuild live cell counts");
    CHECK(l0_delete_growing.live_cell_size(0, 0) == 0 &&
              l0_delete_growing.live_cell_size(0, 1) == 3,
          "incremental L0 delete must remove dead PID from subspace 0 counts");
    CHECK(l0_delete_growing.live_cell_size(1, 0) == 2 &&
              l0_delete_growing.live_cell_size(1, 1) == 1,
          "incremental L0 delete must remove dead PID from subspace 1 counts");

    UnsealedDeltaView layered_growing;
    UnsealedDeltaView::PostingVectors layer0_offsets = {{0, 1, 2}, {0, 1, 1}};
    UnsealedDeltaView::PostingVectors layer0_ids = {{50, 51}, {52}};
    UnsealedDeltaView::PostingVectors layer1_offsets = {{0, 2, 2}, {0, 0, 2}};
    UnsealedDeltaView::PostingVectors layer1_ids = {{53, 54}, {55, 56}};
    layered_growing.posting_layers.push_back(
        layered_growing.make_layer(std::move(layer0_offsets), std::move(layer0_ids), 0));
    layered_growing.posting_layers.push_back(
        layered_growing.make_layer(std::move(layer1_offsets), std::move(layer1_ids), 1));
    layered_growing.rebuild_posting_cell_counts({2, 2});
    CHECK(layered_growing.cell_size(0, 0) == 3 &&
              layered_growing.cell_size(0, 1) == 1 &&
              layered_growing.cell_size(1, 0) == 1 &&
              layered_growing.cell_size(1, 1) == 2,
          "multi-layer L0 posting counts must aggregate layers");
    std::vector<uint32_t> layered_s0c0 =
        collect_growing_cell(layered_growing, 0, 0);
    std::vector<uint32_t> layered_s1c1 =
        collect_growing_cell(layered_growing, 1, 1);
    CHECK(layered_s0c0 == std::vector<uint32_t>({50, 53, 54}) &&
              layered_s1c1 == std::vector<uint32_t>({55, 56}),
          "multi-layer L0 span traversal must preserve layer order");

    UnsealedDeltaView compactable_growing;
    for (uint32_t layer = 0; layer < 4; ++layer) {
        UnsealedDeltaView::PostingVectors offsets = {{0, 1, 2}, {0, 1, 1}};
        UnsealedDeltaView::PostingVectors ids = {
            {100u + layer * 10u, 101u + layer * 10u},
            {102u + layer * 10u}
        };
        compactable_growing.posting_layers.push_back(
            compactable_growing.make_layer(std::move(offsets), std::move(ids), layer));
    }
    bool compacted_layers =
        compactable_growing.maybe_consolidate_layers_for_query({2, 2});
    CHECK(compacted_layers && compactable_growing.posting_layers.size() == 1,
          "small multi-layer L0 postings should consolidate for query");
    CHECK(compactable_growing.cell_span(0, 0).count == 4 &&
              compactable_growing.cell_span(0, 1).count == 4 &&
              compactable_growing.cell_span(1, 0).count == 4,
          "consolidated L0 cell counts must be query-addressable");
    std::vector<uint32_t> compacted_s0c0 =
        collect_growing_cell(compactable_growing, 0, 0);
    CHECK(compacted_s0c0 == std::vector<uint32_t>({100, 110, 120, 130}),
          "consolidated L0 postings must preserve layer order");

    std::vector<UpdateWalInsert> sparse = inserts;
    sparse[2].global_pid = 60;
    std::shared_ptr<const L0Block> sparse_block =
        L0Block::from_inserts(78, sparse, {3, 2}, 2, 3);
    CHECK(sparse_block->pid_begin() == 50 &&
              sparse_block->pid_end() == 61 &&
              sparse_block->pid_count() == 11 &&
              sparse_block->row_count() == 3,
          "sparse L0Block must distinguish PID range from row count");
    CHECK(!sparse_block->dense_pid_order(),
          "sparse L0Block must disable dense PID row fast path");
    CHECK(sparse_block->contains_pid(50) &&
              sparse_block->contains_pid(51) &&
              !sparse_block->contains_pid(52) &&
              sparse_block->contains_pid(60),
          "sparse L0Block PID membership mismatch");
    CHECK(sparse_block->local_row_for_pid(60) == 2,
          "sparse L0Block PID lookup must resolve local row");
    UnsealedDeltaView sparse_growing;
    sparse_growing.committed_blocks.push_back(block);
    sparse_growing.committed_blocks.push_back(sparse_block);
    sparse_growing.build_lookup({3, 2}, nullptr);
    UnsealedDeltaView::RowRef sparse_row;
    CHECK(sparse_growing.resolve_pid(60, sparse_row) &&
              sparse_row.block == sparse_block.get() &&
              sparse_row.local_row == 2,
          "UnsealedDeltaView must resolve sparse L0 block PIDs");
    CHECK(sparse_growing.resolve_pid(50, sparse_row) &&
              sparse_row.block == sparse_block.get() &&
              sparse_row.local_row == 0,
          "later sparse L0 block should override older duplicate PID");
    CHECK(sparse_growing.resolve_pid(52, sparse_row) &&
              sparse_row.block == block.get() &&
              sparse_row.local_row == 2,
          "sparse L0 block holes must fall through to older blocks");

    TEST("L0 block materializes postings and payloads"); PASS();
}

void test_base_live_posting_overlay_filters_dead_pids() {
    std::cout << "[Base live posting overlay]" << std::endl;

    const std::string dir = "/tmp/score_base_live_postings_" + std::to_string(getpid());
    mkdir(dir.c_str(), 0755);

    PostingLists postings;
    std::vector<size_t> cells_per_subspace = {3, 2};
    std::vector<int> cell_ids = {
        0, 0,
        1, 1,
        1, 0,
        2, 1,
        1, 0,
        0, 1
    };
    postings.build(cells_per_subspace, cell_ids, 6, 2, dir);
    postings.load(dir, 2);

    std::shared_ptr<const LiveTable> live =
        LiveTable::from_live_pids(6, {0, 2, 3, 5});
    BasePostingView overlay;
    overlay.build(postings, 2, *live, 6);

    CHECK(overlay.cell_size(0, 0) == 2,
          "overlay should preserve live PIDs in subspace 0 cell 0");
    const uint32_t* s0c0 = overlay.cell_data(0, 0);
    CHECK(s0c0[0] == 0 && s0c0[1] == 5,
          "overlay subspace 0 cell 0 data mismatch");
    CHECK(overlay.cell_size(0, 1) == 1 && overlay.cell_data(0, 1)[0] == 2,
          "overlay must filter dead PID from subspace 0 cell 1");
    CHECK(overlay.cell_size(0, 2) == 1 && overlay.cell_data(0, 2)[0] == 3,
          "overlay subspace 0 cell 2 data mismatch");
    CHECK(overlay.cell_size(1, 0) == 2,
          "overlay subspace 1 cell 0 size mismatch");
    const uint32_t* s1c0 = overlay.cell_data(1, 0);
    CHECK(s1c0[0] == 0 && s1c0[1] == 2,
          "overlay subspace 1 cell 0 data mismatch");
    CHECK(overlay.cell_size(1, 1) == 2,
          "overlay subspace 1 cell 1 size mismatch");
    const uint32_t* s1c1 = overlay.cell_data(1, 1);
    CHECK(s1c1[0] == 3 && s1c1[1] == 5,
          "overlay must preserve live PID order within a cell");
    CHECK(overlay.total_postings == 8,
          "overlay total postings should equal live base rows times subspaces");

    std::vector<UpdateWalInsert> delta_inserts(2);
    for (int i = 0; i < 2; ++i) {
        delta_inserts[(size_t)i].orig_id = (uint32_t)(700 + i);
        delta_inserts[(size_t)i].global_pid = (uint32_t)(6 + i);
        delta_inserts[(size_t)i].verify_rotated = {(float)i, (float)i + 0.25f};
        delta_inserts[(size_t)i].rabitq_record = {
            (uint8_t)(10 + i), (uint8_t)(20 + i), (uint8_t)(30 + i)
        };
    }
    delta_inserts[0].subspace_cells = {0, 1};
    delta_inserts[1].subspace_cells = {1, 0};
    std::shared_ptr<const SealedDeltaSegment> delta =
        SealedDeltaSegment::from_inserts(
            "merged_phase4_delta", delta_inserts, cells_per_subspace, 2, 3);
    std::shared_ptr<const LiveTable> delta_live =
        LiveTable::from_live_pids(8, {6, 7});
    SealedDeltaPostingView delta_postings;
    delta_postings.build({delta}, cells_per_subspace, delta_live.get());

    std::remove((dir + "/posting_subspace_0.bin").c_str());
    std::remove((dir + "/posting_subspace_1.bin").c_str());
    rmdir(dir.c_str());
    TEST("Base live posting overlay filters dead PIDs"); PASS();
}

void test_sealed_delta_load_validates_postings() {
    std::cout << "[Sealed delta]" << std::endl;

    const std::string dir = "/tmp/score_delta_validate_" + std::to_string(getpid());
    mkdir(dir.c_str(), 0755);

    std::vector<UpdateWalInsert> inserts(2);
    for (int i = 0; i < 2; ++i) {
        inserts[(size_t)i].orig_id = (uint32_t)(500 + i);
        inserts[(size_t)i].global_pid = i == 0 ? 100u : 102u;
        inserts[(size_t)i].verify_rotated = {(float)i, (float)i + 0.5f};
        inserts[(size_t)i].rabitq_record = {
            (uint8_t)(1 + i), (uint8_t)(2 + i), (uint8_t)(3 + i)
        };
    }
    inserts[0].subspace_cells = {0, 1};
    inserts[1].subspace_cells = {1, 0};

    std::shared_ptr<const SealedDeltaSegment> segment =
        SealedDeltaSegment::from_inserts("delta_validate", inserts, {2, 2}, 2, 3);
    CHECK(segment->pid_begin() == 100 && segment->pid_count() == 3 &&
              segment->row_count() == 2,
          "sealed delta must distinguish PID range count from row count");
    CHECK(!segment->dense_pid_order(),
          "sealed delta dense PID fast path must reject sparse PID ranges");
    std::vector<UpdateWalInsert> dense_inserts = inserts;
    dense_inserts[1].global_pid = 101;
    std::shared_ptr<const SealedDeltaSegment> dense_segment =
        SealedDeltaSegment::from_inserts(
            "delta_dense", dense_inserts, {2, 2}, 2, 3);
    CHECK(dense_segment->dense_pid_order(),
          "sealed delta dense PID fast path should accept contiguous PID ranges");
    segment->save(dir);
    {
        std::ifstream row_cells(dir + "/delta_row_cells.bin", std::ios::binary);
        CHECK(!row_cells.good(), "small sealed delta should not write optional row cell artifact");
    }
    auto write_manifest_with_row_cells = [&]() {
        std::ofstream manifest(dir + "/segment_manifest.json", std::ios::trunc);
        manifest << "{"
                 << "\"format\":\"score_delta_segment_v1\","
                 << "\"segment_id\":\"delta_validate\","
                 << "\"pid\":{\"pid_begin\":100,\"pid_count\":3,"
                 << "\"row_count\":2},"
                 << "\"layout\":{\"num_subspaces\":2,\"verify_dim\":2,"
                 << "\"rabitq_record_bytes\":3},"
                 << "\"artifacts\":{\"orig_ids\":\"delta_orig_ids.bin\","
                 << "\"pid_to_row\":\"delta_pid_to_row.bin\","
                 << "\"row_cells\":\"delta_row_cells.bin\","
                 << "\"verify_vectors\":\"delta_verify_vectors.fbin\","
                 << "\"rabitq_records\":\"delta_rabitq_bin_data.bin\","
                 << "\"posting_prefix\":\"delta_posting_subspace_\"}"
                 << "}";
    };
    {
        std::ofstream row_cells(dir + "/delta_row_cells.bin",
                                std::ios::binary | std::ios::trunc);
        uint32_t n = 2;
        uint32_t m = 2;
        uint16_t cells[4] = {0, 1, 1, 0};
        row_cells.write(reinterpret_cast<const char*>(&n), sizeof(n));
        row_cells.write(reinterpret_cast<const char*>(&m), sizeof(m));
        row_cells.write(reinterpret_cast<const char*>(cells), sizeof(cells));
    }
    write_manifest_with_row_cells();
    std::shared_ptr<const SealedDeltaSegment> loaded =
        SealedDeltaSegment::load("delta_validate", dir, {2, 2}, 2, 3, 100, 3, 2);
    CHECK(loaded->pid_count() == 3 && loaded->row_count() == 2 &&
              loaded->contains_pid(100) && !loaded->contains_pid(101) &&
              loaded->contains_pid(102),
          "sealed delta load should accept valid postings");
    CHECK(!loaded->dense_pid_order(),
          "sealed delta load must preserve sparse PID fast-path guard");

    {
        std::fstream row_cells(dir + "/delta_row_cells.bin",
                               std::ios::binary | std::ios::in | std::ios::out);
        uint16_t wrong_cell = 1;
        row_cells.seekp((std::streamoff)(2 * sizeof(uint32_t)));
        row_cells.write(reinterpret_cast<const char*>(&wrong_cell), sizeof(wrong_cell));
    }
    bool rejected_bad_row_cells = false;
    try {
        (void)SealedDeltaSegment::load("delta_validate", dir, {2, 2}, 2, 3, 100, 3, 2, true);
    } catch (const std::runtime_error&) {
        rejected_bad_row_cells = true;
    }
    CHECK(rejected_bad_row_cells,
          "sealed delta load must reject row cell artifacts that disagree with postings");
    segment->save(dir);

    std::shared_ptr<const SealedDeltaSegment> loaded_with_tombstone =
        SealedDeltaSegment::load("delta_validate", dir, {2, 2}, 2, 3, 100, 3, 1);
    std::shared_ptr<const LiveTable> sealed_live =
        LiveTable::from_live_pids(103, {102});
    loaded_with_tombstone->build_live_cell_counts(*sealed_live);
    CHECK(loaded_with_tombstone->has_live_cell_counts(),
          "sealed delta should cache live cell counts when it has tombstones");
    CHECK(loaded_with_tombstone->live_cell_size(0, 0) == 0 &&
              loaded_with_tombstone->live_cell_size(0, 1) == 1,
          "sealed delta live cell counts must skip dead PIDs");
    CHECK(loaded_with_tombstone->live_cell_size(1, 0) == 1 &&
              loaded_with_tombstone->live_cell_size(1, 1) == 0,
          "sealed delta live cell counts must be per subspace");
    std::vector<UpdateWalInsert> appended_inserts(1);
    appended_inserts[0].orig_id = 502;
    appended_inserts[0].global_pid = 103;
    appended_inserts[0].verify_rotated = {3.0f, 3.5f};
    appended_inserts[0].rabitq_record = {4, 5, 6};
    appended_inserts[0].subspace_cells = {0, 0};
    std::shared_ptr<const SealedDeltaSegment> appended_segment =
        SealedDeltaSegment::from_inserts(
            "delta_validate_appended", appended_inserts, {2, 2}, 2, 3);
    CHECK(appended_segment->dense_pid_order(),
          "single-row appended sealed delta should be dense-PID addressable");
    std::shared_ptr<const LiveTable> appended_live =
        LiveTable::from_live_pids(104, {102, 103});
    SealedDeltaPostingView prefix_postings;
    prefix_postings.build({loaded_with_tombstone}, {2, 2}, appended_live.get());
    SealedDeltaPostingView full_postings;
    full_postings.build(
        {loaded_with_tombstone, appended_segment}, {2, 2}, appended_live.get());
    SealedDeltaPostingView appended_postings;
    appended_postings.build_appending_to_previous(
        prefix_postings, {appended_segment}, {2, 2}, appended_live.get());
    for (uint32_t m = 0; m < 2; ++m) {
        for (uint32_t cell = 0; cell < 2; ++cell) {
            SealedDeltaPostingView::CellSpan full_span =
                full_postings.cell_span(m, cell);
            SealedDeltaPostingView::CellSpan appended_span =
                appended_postings.cell_span(m, cell);
            CHECK(full_span.count == appended_span.count,
                  "incremental sealed posting append cell count mismatch");
            for (size_t i = 0; i < full_span.count; ++i) {
                CHECK(full_span.data[i] == appended_span.data[i],
                      "incremental sealed posting append cell payload mismatch");
            }
        }
    }

    {
        std::ofstream manifest(dir + "/segment_manifest.json", std::ios::trunc);
        manifest << "{"
                 << "\"format\":\"score_delta_segment_v1\","
                 << "\"segment_id\":\"delta_validate\","
                 << "\"pid\":{\"pid_begin\":100,\"pid_count\":3,"
                 << "\"row_count\":1},"
                 << "\"layout\":{\"num_subspaces\":2,\"verify_dim\":2,"
                 << "\"rabitq_record_bytes\":3},"
                 << "\"artifacts\":{\"orig_ids\":\"delta_orig_ids.bin\","
                 << "\"pid_to_row\":\"delta_pid_to_row.bin\","
                 << "\"verify_vectors\":\"delta_verify_vectors.fbin\","
                 << "\"rabitq_records\":\"delta_rabitq_bin_data.bin\","
                 << "\"posting_prefix\":\"delta_posting_subspace_\"}"
                 << "}";
    }
    bool rejected_manifest_row_mismatch = false;
    try {
        (void)SealedDeltaSegment::load("delta_validate", dir, {2, 2}, 2, 3, 100, 3, 2);
    } catch (const std::runtime_error&) {
        rejected_manifest_row_mismatch = true;
    }
    CHECK(rejected_manifest_row_mismatch,
          "sealed delta load must reject segment manifest row count mismatch");
    segment->save(dir);

    {
        std::ofstream out(dir + "/delta_posting_subspace_0.bin",
                          std::ios::binary | std::ios::trunc);
        int K = 2;
        int total = 2;
        uint32_t offsets[3] = {0, 1, 2};
        uint32_t ids[2] = {100, 999};
        out.write(reinterpret_cast<const char*>(&K), sizeof(K));
        out.write(reinterpret_cast<const char*>(&total), sizeof(total));
        out.write(reinterpret_cast<const char*>(offsets), sizeof(offsets));
        out.write(reinterpret_cast<const char*>(ids), sizeof(ids));
    }
    bool rejected_outside_pid = false;
    try {
        (void)SealedDeltaSegment::load("delta_validate", dir, {2, 2}, 2, 3, 100, 3, 2);
    } catch (const std::runtime_error&) {
        rejected_outside_pid = true;
    }
    CHECK(rejected_outside_pid,
          "sealed delta load must reject postings that reference PIDs outside the segment");

    std::remove((dir + "/segment_manifest.json").c_str());
    std::remove((dir + "/delta_orig_ids.bin").c_str());
    std::remove((dir + "/delta_pid_to_row.bin").c_str());
    std::remove((dir + "/delta_row_cells.bin").c_str());
    std::remove((dir + "/delta_verify_vectors.fbin").c_str());
    std::remove((dir + "/delta_rabitq_bin_data.bin").c_str());
    std::remove((dir + "/delta_posting_subspace_0.bin").c_str());
    std::remove((dir + "/delta_posting_subspace_1.bin").c_str());
    rmdir(dir.c_str());
    TEST("Sealed delta posting validation"); PASS();
}



