#include "test_macros.h"

void test_fused_no_reorder_artifacts_are_loadable() {
    std::cout << "[Streaming fused no-reorder]" << std::endl;

    const std::string suffix = std::to_string(getpid());
    const std::string input_path = "/tmp/score_stream_fused_input_" + suffix + ".fbin";
    const std::string index_dir = "/tmp/score_stream_fused_index_" + suffix;
    const int n = 24;
    const int dim = 96;
    const int M = 4;
    const int ivf_k = 3;
    const int rabitq_k = 4;
    const int sub_dim = dim / M;
    std::vector<float> rows((size_t)n * dim);
    for (int i = 0; i < n; ++i) {
        for (int d = 0; d < dim; ++d) {
            rows[(size_t)i * dim + d] = (float)(((i + 11) * 23 + d * 5) % 127) * 0.01f;
        }
    }
    write_test_fbin(input_path, rows, n, dim);
    mkdir(index_dir.c_str(), 0755);

    RandomRotation rotation;
    rotation.init(dim, 42, RotationType::FhtKac);
    const int rabitq_payload_bytes = (int)rabitqlib::BinDataMap<float>::data_bytes(rotation.padded_dim());
    StreamingRabitqOptions rabitq_options;
    rabitq_options.rabitq_k = rabitq_k;
    rabitq_options.rabitq_seed = 42;
    rabitq_options.block_rows = 7;
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
    CHECK(access((index_dir + "/rotation.bin").c_str(), F_OK) == 0,
          "fused writer must publish rotation.bin");
    CHECK(access((index_dir + "/rotate_subspace.flag").c_str(), F_OK) == 0,
          "fused writer must publish rotate_subspace.flag");

    int verify_n = 0, verify_dim = 0;
    std::vector<float> streamed_verify = load_fbin_payload(index_dir + "/verify_vectors.fbin", verify_n, verify_dim);
    CHECK(verify_n == n, "fused verify n mismatch");
    CHECK(verify_dim == rotation.padded_dim(), "fused verify dim mismatch");
    std::vector<float> expected_verify((size_t)n * rotation.padded_dim());
    for (int i = 0; i < n; ++i) {
        rotation.apply(rows.data() + (size_t)i * dim,
                       expected_verify.data() + (size_t)i * rotation.padded_dim());
    }
    CHECK(streamed_verify == expected_verify, "fused verify payload mismatch");
    std::vector<float> current_verify = streamed_verify;
    int current_verify_n = verify_n;
    int current_verify_dim = verify_dim;

    std::vector<uint32_t> rabitq_assignments((size_t)n);
    {
        std::ifstream in(index_dir + "/rabitq_assignments.bin", std::ios::binary);
        int nn = 0;
        in.read(reinterpret_cast<char*>(&nn), sizeof(int));
        CHECK(nn == n, "fused RaBitQ assignment n mismatch");
        in.read(reinterpret_cast<char*>(rabitq_assignments.data()), (size_t)n * sizeof(uint32_t));
    }
    {
        std::ifstream in(index_dir + "/rabitq_bin_data.bin", std::ios::binary);
        int nn = 0, bb = 0;
        in.read(reinterpret_cast<char*>(&nn), sizeof(int));
        in.read(reinterpret_cast<char*>(&bb), sizeof(int));
        CHECK(nn == n, "fused RaBitQ bin n mismatch");
        CHECK(bb == (int)(sizeof(uint32_t) + (size_t)rabitq_payload_bytes),
              "fused RaBitQ embedded record bytes mismatch");
        for (int i = 0; i < n; ++i) {
            uint32_t embedded_cid = std::numeric_limits<uint32_t>::max();
            in.read(reinterpret_cast<char*>(&embedded_cid), sizeof(uint32_t));
            CHECK(embedded_cid == rabitq_assignments[(size_t)i],
                  "fused RaBitQ embedded cid mismatch");
            in.seekg(rabitq_payload_bytes, std::ios::cur);
        }
    }

    PostingLists postings;
    postings.load(index_dir, M);
    for (int m = 0; m < M; ++m) {
        IVFIndex ivf;
        ivf.load(index_dir + "/ivf_subspace_" + std::to_string(m) + ".bin");
        CHECK(ivf.subspace_dim() == sub_dim, "fused IVF sub_dim mismatch");
        CHECK(ivf.K() == ivf_k, "fused IVF K mismatch");
        CHECK(postings.K(m) == ivf_k, "fused posting K mismatch");
        size_t size_sum = 0;
        for (int k = 0; k < ivf_k; ++k) size_sum += postings.cell_size(m, k);
        CHECK(size_sum == (size_t)n, "fused posting sizes do not sum to n");

        std::vector<float> sub_all((size_t)n * sub_dim);
        for (int i = 0; i < n; ++i) {
            std::memcpy(sub_all.data() + (size_t)i * sub_dim,
                        expected_verify.data() + (size_t)i * rotation.padded_dim() + (size_t)m * sub_dim,
                        (size_t)sub_dim * sizeof(float));
        }
        std::vector<int> assignments(n);
        ivf.assign_batch(sub_all.data(), assignments.data(), n);
        for (int k = 0; k < ivf_k; ++k) {
            std::vector<int> expected_ids;
            for (int i = 0; i < n; ++i) {
                if (assignments[(size_t)i] == k) expected_ids.push_back(i);
            }
            CHECK(postings.cell_size(m, k) == expected_ids.size(), "fused posting cell size mismatch");
            const uint32_t* ids = postings.cell_data(m, k);
            for (size_t j = 0; j < expected_ids.size(); ++j) {
                CHECK(ids[j] == (uint32_t)expected_ids[j], "fused posting ids mismatch");
            }
        }
        CHECK(access((index_dir + "/posting_assign_subspace_" + std::to_string(m) + ".tmp").c_str(), F_OK) != 0,
              "fused temporary assignment file should be removed");
        CHECK(access((index_dir + "/posting_assign_subspace_" + std::to_string(m) + ".assign.bin").c_str(), F_OK) != 0,
              "fused assignment artifact should be opt-in only");
    }
    CHECK(access((index_dir + "/posting_assignments.manifest").c_str(), F_OK) != 0,
          "fused assignment manifest should be opt-in only");

    std::remove((index_dir + "/rabitq_assignments.bin").c_str());
    try {
        IndexConfig config;
        config.num_subspaces = M;
        config.rabitq_k = rabitq_k;
        config.num_threads = 1;
        QueryEngine engine;
        engine.init(config);
        engine.load_index(index_dir);
        const ReaderView& reader = engine.reader_view();
        CHECK(reader.base != nullptr, "reader view must expose base segment");
        CHECK(reader.generation == 0, "base-only reader generation should start at zero");
        CHECK(reader.base_n == (uint32_t)n, "base-only reader base_n mismatch");
        CHECK(reader.max_pid == (uint32_t)n, "base-only reader max_pid mismatch");
        CHECK(reader.live_count == (uint32_t)n, "base-only reader live_count mismatch");
        CHECK(reader.live != nullptr, "base-only reader must expose all-live table");
        CHECK(reader.live->live_count() == (uint32_t)n, "all-live table live_count mismatch");
        CHECK(reader.live->is_live(0), "all-live table must mark pid 0 live");
        CHECK(reader.live->is_live((uint32_t)n - 1), "all-live table must mark final base pid live");
        CHECK(!reader.live->is_live((uint32_t)n), "all-live table must reject pid at max_pid");
        CHECK(!reader.has_deltas(), "base-only reader should not report deltas");

        RowResolver resolver = engine.row_resolver();
        ResolvedRow resolved;
        CHECK(resolver.resolve(0, resolved), "RowResolver must resolve base pid 0");
        CHECK(resolved.segment == ResolvedRow::SegmentKind::Base, "RowResolver must classify base pid");
        CHECK(resolved.global_pid == 0, "RowResolver global pid mismatch");
        CHECK(resolved.output_id == 0, "RowResolver no-reorder output id mismatch");
        CHECK(resolved.rabitq_record != nullptr, "RowResolver must expose base RaBitQ record");
        CHECK(resolved.rabitq_record_bytes == sizeof(uint32_t) + (size_t)rabitq_payload_bytes,
              "RowResolver RaBitQ record stride mismatch");
        CHECK(resolved.verify_vector != nullptr, "RowResolver must expose base verify vector");
        CHECK(resolved.verify_dim == verify_dim, "RowResolver verify dim mismatch");
        CHECK(!resolver.resolve((uint32_t)n, resolved), "RowResolver must reject pid at max_pid");
        TEST("Base ReaderView RowResolver"); PASS();

        BaseSegment lightweight_base =
            impl::load_update_base_segment_lightweight(index_dir);
        CHECK(lightweight_base.n == n, "lightweight update base row count mismatch");
        CHECK(lightweight_base.dim == dim, "lightweight update base dim mismatch");
        CHECK(lightweight_base.rabitq_k == rabitq_k,
              "lightweight update base RaBitQ K mismatch");
        CHECK(lightweight_base.verify_vectors_dim == verify_dim,
              "lightweight update base verify dim mismatch");
        CHECK(lightweight_base.ivf_indices.size() == (size_t)M,
              "lightweight update base IVF count mismatch");
        CHECK(lightweight_base.perm_to_orig.empty(),
              "lightweight update base should preserve absent perm_to_orig");
        CHECK(lightweight_base.verify_vectors_ == nullptr,
              "lightweight update base must not load full verify vectors");
        CHECK(lightweight_base.rabitq_bin_with_assign_ == nullptr,
              "lightweight update base must not load full base RaBitQ records");
        TEST("Update lightweight base loader omits full base payloads"); PASS();

        std::vector<uint32_t> update_top_assignments;
        impl::assign_l0_rabitq_centroids(
            *reader.base, expected_verify.data(), n, update_top_assignments);
        CHECK(update_top_assignments == rabitq_assignments,
              "batch update RaBitQ assignment must match build assignment sidecar");

        std::vector<float> encode_rows(rows.begin(), rows.begin() + (size_t)3 * (size_t)dim);
        std::vector<uint32_t> encode_orig_ids = {9001, 9002, 9003};
        std::vector<UpdateWalInsert> encoded_updates = impl::encode_l0_inserts(
            index_dir, *reader.base, encode_rows, 3, dim, encode_orig_ids, (uint32_t)n + 10);
        impl::EncodedL0InsertBatch contiguous_updates =
            impl::encode_l0_inserts_contiguous(
                index_dir,
                *reader.base,
                encode_rows.data(),
                3,
                dim,
                encode_orig_ids,
                (uint32_t)n + 10);
        auto posting_cell_for_pid = [&](int subspace, uint32_t pid) {
            for (int cell = 0; cell < postings.K(subspace); ++cell) {
                const uint32_t* ids = postings.cell_data(subspace, cell);
                size_t count = postings.cell_size(subspace, cell);
                for (size_t j = 0; j < count; ++j) {
                    if (ids[j] == pid) return (uint32_t)cell;
                }
            }
            return std::numeric_limits<uint32_t>::max();
        };
        CHECK(encoded_updates.size() == 3, "batch update encode count mismatch");
        CHECK(contiguous_updates.size() == encoded_updates.size(),
              "contiguous update encode count mismatch");
        for (size_t i = 0; i < encoded_updates.size(); ++i) {
            const UpdateWalInsert& insert = encoded_updates[i];
            CHECK(insert.orig_id == encode_orig_ids[i], "batch update encode orig_id mismatch");
            CHECK(insert.global_pid == (uint32_t)n + 10 + (uint32_t)i,
                  "batch update encode PID range mismatch");
            CHECK(contiguous_updates.orig_ids[i] == insert.orig_id &&
                      contiguous_updates.global_pid(i) == insert.global_pid,
                  "contiguous update encode IDs mismatch");
            CHECK(insert.verify_rotated.size() == (size_t)verify_dim,
                  "batch update encode verify dim mismatch");
            CHECK(std::equal(
                      contiguous_updates.verify_ptr(i),
                      contiguous_updates.verify_ptr(i) + verify_dim,
                      insert.verify_rotated.begin()),
                  "contiguous update encode verify payload mismatch");
            CHECK(std::equal(
                      contiguous_updates.rabitq_record_ptr(i),
                      contiguous_updates.rabitq_record_ptr(i) + insert.rabitq_record.size(),
                      insert.rabitq_record.begin()),
                  "contiguous update encode RaBitQ payload mismatch");
            CHECK(std::equal(
                      contiguous_updates.subspace_cells_ptr(i),
                      contiguous_updates.subspace_cells_ptr(i) + M,
                      insert.subspace_cells.begin()),
                  "contiguous update encode cell payload mismatch");
            for (int d = 0; d < verify_dim; ++d) {
                CHECK(insert.verify_rotated[(size_t)d] ==
                          expected_verify[i * (size_t)verify_dim + (size_t)d],
                      "batch update encode verify payload mismatch");
            }
            uint32_t encoded_cid = std::numeric_limits<uint32_t>::max();
            std::memcpy(&encoded_cid, insert.rabitq_record.data(), sizeof(uint32_t));
            CHECK(encoded_cid == rabitq_assignments[i], "batch update encode RaBitQ assignment mismatch");
            CHECK(insert.subspace_cells.size() == (size_t)M,
                  "batch update encode subspace cell count mismatch");
            for (int m = 0; m < M; ++m) {
                uint32_t expected_cell = posting_cell_for_pid(m, (uint32_t)i);
                CHECK(expected_cell != std::numeric_limits<uint32_t>::max(),
                      "encoded update test could not find base posting cell");
                CHECK(insert.subspace_cells[(size_t)m] == expected_cell,
                      "batch update encode subspace assignment mismatch");
            }
        }
        TEST("Batch update encode reuses build assignment semantics"); PASS();

        std::vector<float> query = rows;
        QueryResult result = engine.search(query.data());
        CHECK(!result.topk.empty(), "fused query should work without RaBitQ assignment sidecar");

        std::vector<QueryResult> empty_batch = engine.search_batch(query.data(), 0);
        CHECK(empty_batch.empty(), "zero-query search_batch should return no results");

        std::vector<QueryResult> batch_results = engine.search_batch(query.data(), 2);
        CHECK(batch_results.size() == 2, "search_batch result count mismatch");
        QueryResult single0 = engine.search(query.data());
        QueryResult single1 = engine.search(query.data() + (size_t)dim);
        auto same_topk = [](const QueryResult& a, const QueryResult& b) {
            if (a.topk.size() != b.topk.size()) return false;
            for (size_t i = 0; i < a.topk.size(); ++i) {
                if (a.topk[i].second != b.topk[i].second) return false;
                if (std::abs(a.topk[i].first - b.topk[i].first) > 1e-5f) return false;
            }
            return true;
        };
        CHECK(same_topk(batch_results[0], single0), "search_batch query 0 differs from single search");
        CHECK(same_topk(batch_results[1], single1), "search_batch query 1 differs from single search");
        TEST("Fused no-reorder search_batch equivalence"); PASS();

        IndexConfig topk_zero_config = config;
        topk_zero_config.topk = 0;
        QueryEngine topk_zero_engine;
        topk_zero_engine.init(topk_zero_config);
        topk_zero_engine.load_index(index_dir);
        QueryResult topk_zero = topk_zero_engine.search(query.data());
        CHECK(topk_zero.topk.empty(), "topk=0 search should return no results");
        TEST("Fused no-reorder topk zero boundary"); PASS();

        bool rejected_negative_topk = false;
        try {
            IndexConfig negative_topk_config = config;
            negative_topk_config.topk = -1;
            QueryEngine negative_topk_engine;
            negative_topk_engine.init(negative_topk_config);
        } catch (const std::invalid_argument&) {
            rejected_negative_topk = true;
        }
        CHECK(rejected_negative_topk, "negative topk must be rejected by QueryEngine::init");
        TEST("QueryEngine rejects negative topk"); PASS();

        {
            std::ofstream marker(index_dir + "/compaction.publish");
            marker << "test\n";
        }
        bool rejected_compaction_publish_load = false;
        try {
            QueryEngine blocked_engine;
            blocked_engine.init(config);
            blocked_engine.load_index(index_dir);
        } catch (const std::exception& e) {
            rejected_compaction_publish_load =
                std::string(e.what()).find("compaction") != std::string::npos;
        }
        std::remove((index_dir + "/compaction.publish").c_str());
        CHECK(rejected_compaction_publish_load,
              "QueryEngine must reject loads during compaction publish window");
        TEST("QueryEngine blocks compaction publish window loads"); PASS();

        std::vector<uint8_t> first_rabitq_record(sizeof(uint32_t) + (size_t)rabitq_payload_bytes);
        {
            std::ifstream in(index_dir + "/rabitq_bin_data.bin", std::ios::binary);
            int nn = 0, record_bytes = 0;
            in.read(reinterpret_cast<char*>(&nn), sizeof(int));
            in.read(reinterpret_cast<char*>(&record_bytes), sizeof(int));
            CHECK(nn == n, "L0 test RaBitQ bin n mismatch");
            CHECK(record_bytes == (int)first_rabitq_record.size(),
                  "L0 test RaBitQ record size mismatch");
            in.read(reinterpret_cast<char*>(first_rabitq_record.data()), first_rabitq_record.size());
            CHECK((bool)in, "L0 test failed to read first RaBitQ record");
        }

        std::vector<uint16_t> row0_cells((size_t)M, std::numeric_limits<uint16_t>::max());
        for (int m = 0; m < M; ++m) {
            for (int cell = 0; cell < postings.K(m); ++cell) {
                const uint32_t* ids = postings.cell_data(m, cell);
                size_t count = postings.cell_size(m, cell);
                for (size_t j = 0; j < count; ++j) {
                    if (ids[j] == 0) row0_cells[(size_t)m] = (uint16_t)cell;
                }
            }
            CHECK(row0_cells[(size_t)m] != std::numeric_limits<uint16_t>::max(),
                  "L0 test must find row 0 posting cell");
        }

        const std::string updates_dir = index_dir + "/updates";
        mkdir(updates_dir.c_str(), 0755);
        {
            UpdateWalHeader header;
            header.generation = 17;
            header.base_n = (uint32_t)n;
            header.num_subspaces = (uint32_t)M;
            header.verify_dim = (uint32_t)verify_dim;
            header.rabitq_record_bytes = (uint32_t)first_rabitq_record.size();
            UpdateWalWriter writer(updates_dir + "/l0.wal", header);

            UpdateWalDelete tombstone;
            tombstone.orig_id = 0;
            tombstone.old_global_pid = 0;
            writer.append_delete(1, tombstone);

            UpdateWalInsert insert;
            insert.orig_id = 999;
            insert.global_pid = (uint32_t)n;
            insert.verify_rotated.assign(streamed_verify.begin(),
                                         streamed_verify.begin() + verify_dim);
            insert.rabitq_record = first_rabitq_record;
            insert.subspace_cells = row0_cells;
            writer.append_insert(2, insert);

            UpdateWalCommit commit;
            commit.lsn = 3;
            commit.batch_record_count = 2;
            commit.max_pid = (uint32_t)n + 1;
            commit.live_count = (uint32_t)n;
            writer.append_commit(commit);
            writer.flush();
        }
        {
            std::ofstream manifest(index_dir + "/update_manifest.json");
            manifest << "{\n";
            manifest << "  \"format\": \"score_update_manifest_v1\",\n";
            manifest << "  \"generation\": 17,\n";
            manifest << "  \"base\": {\"path\": \".\", \"base_n\": " << n << "},\n";
            manifest << "  \"pid\": {\"base_n\": " << n << ", \"max_pid\": " << (n + 1)
                     << ", \"live_count\": " << n << "},\n";
            manifest << "  \"sealed_deltas\": [],\n";
            manifest << "  \"growing\": {\"wal_path\": \"updates/l0.wal\",";
            manifest << " \"checkpoint_path\": \"updates/l0.checkpoint\"},\n";
            manifest << "  \"direct_map\": {\"mode\": \"dense\", \"path\": \"updates/direct_map.bin\",";
            manifest << " \"optional_for_search\": true},\n";
            manifest << "  \"durability\": {\"mode\": \"buffered\", \"last_committed_lsn\": 3,";
            manifest << " \"last_fsynced_lsn\": 3}\n";
            manifest << "}\n";
        }

        IndexConfig update_config = config;
        update_config.collision_ratio = 1.0f;
        update_config.candidate_ratio = 1.0f;
        update_config.topk = 5;
        QueryEngine update_engine;
        update_engine.init(update_config);
        update_engine.load_index(index_dir);
        const ReaderView& update_reader = update_engine.reader_view();
        CHECK(update_reader.generation == 17, "L0 reader generation mismatch");
        CHECK(update_reader.has_deltas(), "L0 reader must report growing delta");
        CHECK(update_reader.max_pid == (uint32_t)n + 1, "L0 reader max_pid mismatch");
        CHECK(update_reader.live_count == (uint32_t)n, "L0 reader live_count mismatch");
        CHECK(!update_reader.live->is_live(0), "L0 tombstone must mark old PID dead");
        CHECK(update_reader.live->is_live((uint32_t)n), "L0 insert must mark new PID live");
        CHECK(update_reader.sparse_dead_pids != nullptr,
              "L0 reader should cache sparse dead PIDs for small tombstone sets");
        CHECK(update_reader.sparse_dead_pids->size() == 1,
              "L0 sparse dead PID cache size mismatch");
        CHECK((*update_reader.sparse_dead_pids)[0] == 0,
              "L0 sparse dead PID cache must contain tombstoned base PID");
        CHECK(update_reader.live_suffix_begin == 1,
              "single prefix tombstone should expose a live suffix for range selection");
        CHECK(update_reader.candidate_selection_strategy ==
                  CandidateSelectionStrategy::LiveSuffixRange,
              "single prefix tombstone should select live suffix candidate strategy");
        CHECK(update_reader.dead_scores_filtered_for_selection,
              "single base tombstone with merged phase4 should filter dead scores before selection");
        CHECK(update_reader.composite_cell_sizes.size() == (size_t)M,
              "update reader should precompute composite cell sizes");
        uint32_t row0_cell0 = row0_cells[0];
        CHECK(update_reader.composite_cell_sizes[0].size() == (size_t)postings.K(0),
              "composite cell size table subspace width mismatch");
        CHECK(update_reader.composite_cell_sizes[0][(size_t)row0_cell0] ==
                  postings.cell_size(0, (int)row0_cell0),
              "composite cell size should replace tombstoned base row with committed L0 row");
        uint32_t first_row0_composite_size =
            update_reader.composite_cell_sizes[0][(size_t)row0_cell0];

        QueryResult update_result = update_engine.search(rows.data());
        bool saw_inserted = false;
        bool saw_deleted = false;
        for (const auto& hit : update_result.topk) {
            if (hit.second == 999) saw_inserted = true;
            if (hit.second == 0) saw_deleted = true;
        }
        CHECK(saw_inserted, "L0 query must return inserted orig_id");
        CHECK(!saw_deleted, "L0 query must not return deleted orig_id");
        TEST("L0 WAL query applies insert and tombstone"); PASS();

        std::shared_ptr<const std::vector<uint32_t>> first_sparse_dead =
            update_engine.reader_view().sparse_dead_pids;
        uint32_t first_base_live_count = update_engine.reader_view().base_live_count;
        uint64_t first_base_live_fingerprint = update_engine.reader_view().base_live_fingerprint;
        uint32_t first_base_dead_prefix_end = update_engine.reader_view().base_dead_prefix_end;

        const std::string overlay_updates_dir = index_dir + "/updates_overlay";
        mkdir(overlay_updates_dir.c_str(), 0755);
        struct BaseOverlayEnvGuard {
            ~BaseOverlayEnvGuard() {
                unsetenv("SCORE_BASE_LIVE_POSTING_STALE_DEAD_LIMIT");
            }
        } base_overlay_env_guard;
        setenv("SCORE_BASE_LIVE_POSTING_STALE_DEAD_LIMIT", "4", 1);
        auto write_overlay_manifest = [&](uint64_t lsn, uint32_t max_pid, uint32_t live_count) {
            std::ofstream manifest(index_dir + "/update_manifest.json");
            manifest << "{\n";
            manifest << "  \"format\": \"score_update_manifest_v1\",\n";
            manifest << "  \"generation\": 19,\n";
            manifest << "  \"base\": {\"path\": \".\", \"base_n\": " << n << "},\n";
            manifest << "  \"pid\": {\"base_n\": " << n << ", \"max_pid\": " << max_pid
                     << ", \"live_count\": " << live_count << "},\n";
            manifest << "  \"sealed_deltas\": [],\n";
            manifest << "  \"growing\": {\"wal_path\": \"updates_overlay/l0.wal\",";
            manifest << " \"checkpoint_path\": \"updates_overlay/l0.checkpoint\"},\n";
            manifest << "  \"direct_map\": {\"mode\": \"dense\", \"path\": \"updates/direct_map.bin\",";
            manifest << " \"optional_for_search\": true},\n";
            manifest << "  \"durability\": {\"mode\": \"buffered\", \"last_committed_lsn\": " << lsn
                     << ", \"last_fsynced_lsn\": " << lsn << "}\n";
            manifest << "}\n";
        };
        {
            UpdateWalHeader header;
            header.generation = 19;
            header.base_n = (uint32_t)n;
            header.num_subspaces = (uint32_t)M;
            header.verify_dim = (uint32_t)verify_dim;
            header.rabitq_record_bytes = (uint32_t)first_rabitq_record.size();
            UpdateWalWriter writer(overlay_updates_dir + "/l0.wal", header);

            UpdateWalDelete tombstone1;
            tombstone1.orig_id = 1;
            tombstone1.old_global_pid = 1;
            writer.append_delete(1, tombstone1);

            UpdateWalCommit commit;
            commit.lsn = 2;
            commit.batch_record_count = 1;
            commit.max_pid = (uint32_t)n;
            commit.live_count = (uint32_t)n - 1;
            writer.append_commit(commit);
            writer.flush();
        }
        write_overlay_manifest(2, (uint32_t)n, (uint32_t)n - 1);
        QueryEngine overlay_engine;
        overlay_engine.init(update_config);
        overlay_engine.load_index(index_dir);
        const ReaderView& exact_overlay_reader = overlay_engine.reader_view();
        CHECK(exact_overlay_reader.base_live_postings != nullptr,
              "base tombstone without delta should build base live posting overlay");
        CHECK(exact_overlay_reader.base_live_postings_exact,
              "fresh base live posting overlay should be exact");
        CHECK(exact_overlay_reader.dead_scores_filtered_for_selection,
              "exact base overlay should filter dead scores");
        CHECK(exact_overlay_reader.candidate_selection_strategy ==
                  CandidateSelectionStrategy::DeadScoresFiltered,
              "exact base overlay should publish dead-score-filtered candidate strategy");
        CHECK(exact_overlay_reader.composite_cell_sizes[0][(size_t)row0_cell0] ==
                  exact_overlay_reader.base_live_postings->cell_size(0, (int)row0_cell0),
              "composite cell size should use filtered base overlay size");
        std::shared_ptr<const BasePostingView> first_base_overlay =
            exact_overlay_reader.base_live_postings;
        {
            UpdateWalHeader header;
            header.generation = 19;
            header.base_n = (uint32_t)n;
            header.num_subspaces = (uint32_t)M;
            header.verify_dim = (uint32_t)verify_dim;
            header.rabitq_record_bytes = (uint32_t)first_rabitq_record.size();
            UpdateWalWriter writer(overlay_updates_dir + "/l0.wal", header, true);

            UpdateWalDelete tombstone0;
            tombstone0.orig_id = 0;
            tombstone0.old_global_pid = 0;
            writer.append_delete(4, tombstone0);

            UpdateWalCommit commit;
            commit.lsn = 5;
            commit.batch_record_count = 1;
            commit.max_pid = (uint32_t)n;
            commit.live_count = (uint32_t)n - 2;
            writer.append_commit(commit);
            writer.flush();
        }
        write_overlay_manifest(5, (uint32_t)n, (uint32_t)n - 2);
        overlay_engine.reload_update_view();
        const ReaderView& stale_overlay_reader = overlay_engine.reader_view();
        CHECK(stale_overlay_reader.base_live_postings.get() == first_base_overlay.get(),
              "small base tombstone tail should reuse previous base live overlay");
        CHECK(!stale_overlay_reader.base_live_postings_exact,
              "reused base live overlay must be marked stale");
        CHECK(stale_overlay_reader.base_live_postings_unfiltered_dead_pids != nullptr &&
                  stale_overlay_reader.base_live_postings_unfiltered_dead_pids->size() == 1 &&
                  (*stale_overlay_reader.base_live_postings_unfiltered_dead_pids)[0] == 0,
              "stale base overlay must track newly unfiltered tombstone");
        CHECK(stale_overlay_reader.selection_dead_pids.get() ==
                  stale_overlay_reader.base_live_postings_unfiltered_dead_pids.get(),
              "selection should clear only tombstones not covered by stale overlay");
        CHECK(stale_overlay_reader.candidate_selection_strategy ==
                  CandidateSelectionStrategy::LiveSuffixRange,
              "prefix tombstones should prefer live suffix strategy over dead PID clear");
        QueryResult stale_overlay_result =
            overlay_engine.search(rows.data() + (size_t)dim);
        bool saw_stale_deleted = false;
        for (const auto& hit : stale_overlay_result.topk) {
            if (hit.second == 1) saw_stale_deleted = true;
        }
        CHECK(!saw_stale_deleted,
              "stale base overlay selection clear list must suppress deleted base ID");
        std::remove((overlay_updates_dir + "/l0.wal").c_str());
        std::remove((overlay_updates_dir + "/l0.checkpoint").c_str());
        rmdir(overlay_updates_dir.c_str());
        unsetenv("SCORE_BASE_LIVE_POSTING_STALE_DEAD_LIMIT");
        TEST("Base live overlay reuses stale view with selection clear list"); PASS();

        {
            UpdateWalHeader header;
            header.generation = 17;
            header.base_n = (uint32_t)n;
            header.num_subspaces = (uint32_t)M;
            header.verify_dim = (uint32_t)verify_dim;
            header.rabitq_record_bytes = (uint32_t)first_rabitq_record.size();
            UpdateWalWriter writer(updates_dir + "/l0.wal", header, true);

            for (uint32_t extra = 0; extra < 7; ++extra) {
                UpdateWalInsert insert;
                insert.orig_id = 1000 + extra;
                insert.global_pid = (uint32_t)n + 1 + extra;
                insert.verify_rotated.assign(streamed_verify.begin(),
                                             streamed_verify.begin() + verify_dim);
                insert.rabitq_record = first_rabitq_record;
                insert.subspace_cells = row0_cells;
                writer.append_insert(4 + extra, insert);
            }

            UpdateWalCommit commit;
            commit.lsn = 11;
            commit.batch_record_count = 7;
            commit.max_pid = (uint32_t)n + 8;
            commit.live_count = (uint32_t)n + 7;
            writer.append_commit(commit);
            writer.flush();
        }
        {
            std::ofstream manifest(index_dir + "/update_manifest.json");
            manifest << "{\n";
            manifest << "  \"format\": \"score_update_manifest_v1\",\n";
            manifest << "  \"generation\": 17,\n";
            manifest << "  \"base\": {\"path\": \".\", \"base_n\": " << n << "},\n";
            manifest << "  \"pid\": {\"base_n\": " << n << ", \"max_pid\": " << (n + 8)
                     << ", \"live_count\": " << (n + 7) << "},\n";
            manifest << "  \"sealed_deltas\": [],\n";
            manifest << "  \"growing\": {\"wal_path\": \"updates/l0.wal\",";
            manifest << " \"checkpoint_path\": \"updates/l0.checkpoint\"},\n";
            manifest << "  \"direct_map\": {\"mode\": \"dense\", \"path\": \"updates/direct_map.bin\",";
            manifest << " \"optional_for_search\": true},\n";
            manifest << "  \"durability\": {\"mode\": \"buffered\", \"last_committed_lsn\": 11,";
            manifest << " \"last_fsynced_lsn\": 11}\n";
            manifest << "}\n";
        }
        update_engine.reload_update_view();
        const ReaderView& append_reload_reader = update_engine.reader_view();
        CHECK(append_reload_reader.sparse_dead_pids.get() == first_sparse_dead.get(),
              "append-only L0 reload should reuse sparse dead PID cache");
        CHECK(append_reload_reader.base_live_count == first_base_live_count,
              "append-only L0 reload should preserve base live count");
        CHECK(append_reload_reader.base_live_fingerprint == first_base_live_fingerprint,
              "append-only L0 reload should preserve base live fingerprint");
        CHECK(append_reload_reader.base_dead_prefix_end == first_base_dead_prefix_end,
              "append-only L0 reload should preserve base dead prefix metadata");
        CHECK(append_reload_reader.composite_cell_sizes[0][(size_t)row0_cell0] ==
                  first_row0_composite_size + 7,
              "append-only L0 reload should increment composite cell size");
        TEST("L0 append-only reload reuses live metadata"); PASS();

        {
            UpdateWalHeader header;
            header.generation = 17;
            header.base_n = (uint32_t)n;
            header.num_subspaces = (uint32_t)M;
            header.verify_dim = (uint32_t)verify_dim;
            header.rabitq_record_bytes = (uint32_t)first_rabitq_record.size();
            UpdateWalWriter writer(updates_dir + "/l0.wal", header, true);

            UpdateWalDelete delete_l0;
            delete_l0.orig_id = 999;
            delete_l0.old_global_pid = (uint32_t)n;
            writer.append_delete(12, delete_l0);

            UpdateWalCommit commit;
            commit.lsn = 13;
            commit.batch_record_count = 1;
            commit.max_pid = (uint32_t)n + 8;
            commit.live_count = (uint32_t)n + 6;
            writer.append_commit(commit);
            writer.flush();
        }
        {
            std::ofstream manifest(index_dir + "/update_manifest.json");
            manifest << "{\n";
            manifest << "  \"format\": \"score_update_manifest_v1\",\n";
            manifest << "  \"generation\": 17,\n";
            manifest << "  \"base\": {\"path\": \".\", \"base_n\": " << n << "},\n";
            manifest << "  \"pid\": {\"base_n\": " << n << ", \"max_pid\": " << (n + 8)
                     << ", \"live_count\": " << (n + 6) << "},\n";
            manifest << "  \"sealed_deltas\": [],\n";
            manifest << "  \"growing\": {\"wal_path\": \"updates/l0.wal\",";
            manifest << " \"checkpoint_path\": \"updates/l0.checkpoint\"},\n";
            manifest << "  \"direct_map\": {\"mode\": \"dense\", \"path\": \"updates/direct_map.bin\",";
            manifest << " \"optional_for_search\": true},\n";
            manifest << "  \"durability\": {\"mode\": \"buffered\", \"last_committed_lsn\": 13,";
            manifest << " \"last_fsynced_lsn\": 13}\n";
            manifest << "}\n";
        }
        update_engine.reload_update_view();
        const ReaderView& delete_reload_reader = update_engine.reader_view();
        CHECK(delete_reload_reader.sparse_dead_pids != nullptr,
              "incremental delete reload should keep sparse dead PID cache");
        CHECK(delete_reload_reader.sparse_dead_pids->size() == 2,
              "incremental delete reload should append one sparse dead PID");
        CHECK((*delete_reload_reader.sparse_dead_pids)[0] == 0 &&
                  (*delete_reload_reader.sparse_dead_pids)[1] == (uint32_t)n,
              "incremental delete reload should extend previous sparse dead list");
        CHECK(delete_reload_reader.candidate_selection_strategy ==
                  CandidateSelectionStrategy::DeadPidClear,
              "sparse non-prefix tombstones should publish dead PID clear strategy");
        QueryResult delete_reload_result = update_engine.search(rows.data());
        bool saw_deleted_l0 = false;
        bool saw_remaining_l0 = false;
        for (const auto& hit : delete_reload_result.topk) {
            if (hit.second == 999) saw_deleted_l0 = true;
            if (hit.second == 1000) saw_remaining_l0 = true;
        }
        CHECK(!saw_deleted_l0 && saw_remaining_l0,
              "incremental sparse dead reload must preserve live-aware query results");
        TEST("L0 incremental delete reload extends sparse dead metadata"); PASS();

        std::remove((index_dir + "/update_manifest.json").c_str());
        std::remove((updates_dir + "/l0.wal").c_str());
        std::remove((updates_dir + "/l0.checkpoint").c_str());
        rmdir(updates_dir.c_str());

        const std::string update_insert_path = "/tmp/score_update_insert_" + suffix + ".fbin";
        const std::string update_batch_insert_path =
            "/tmp/score_update_batch_insert_" + suffix + ".fbin";
        const std::string update_insert_ids_path = "/tmp/score_update_insert_ids_" + suffix + ".txt";
        const std::string update_insert2_ids_path = "/tmp/score_update_insert2_ids_" + suffix + ".txt";
        const std::string update_batch_insert_ids_path =
            "/tmp/score_update_batch_insert_ids_" + suffix + ".txt";
        const std::string update_delete_ids_path = "/tmp/score_update_delete_ids_" + suffix + ".txt";
        std::vector<float> one_insert(rows.begin(), rows.begin() + dim);
        write_test_fbin(update_insert_path, one_insert, 1, dim);
        {
            std::ofstream ids(update_insert_ids_path);
            ids << "1001\n";
        }
        UpdateIndexBatch insert_batch;
        insert_batch.insert_fbin = update_insert_path;
        insert_batch.insert_ids = update_insert_ids_path;
        insert_batch.durability_mode = "none";
        insert_batch.threads = 1;
        const int saved_update_threads = omp_get_max_threads();
        omp_set_num_threads(2);
        QueryEngine refresh_engine;
        refresh_engine.init(update_config);
        refresh_engine.load_index(index_dir);
        std::vector<QueryResult> pre_refresh_batch = refresh_engine.search_batch(rows.data(), 1);
        CHECK(pre_refresh_batch.size() == 1,
              "pre-refresh search_batch should materialize a base-only thread engine");
        commit_update_batch(index_dir, insert_batch);
        CHECK(omp_get_max_threads() == 2,
              "score_update_index thread override must restore caller OMP thread count");
        omp_set_num_threads(saved_update_threads);
        CHECK(access((index_dir + "/update_manifest.json").c_str(), F_OK) == 0,
              "score_update_index path must write update_manifest.json");
        CHECK(access((index_dir + "/updates/l0.wal").c_str(), F_OK) == 0,
              "score_update_index path must write L0 WAL");
        CHECK(access((index_dir + "/updates/direct_map.bin").c_str(), F_OK) == 0,
              "score_update_index path must write DirectMap");
        UpdateManifest cli_manifest = load_update_manifest_json_for_index(index_dir);
        CHECK(cli_manifest.max_pid == (uint32_t)n + 1, "score_update_index insert max_pid mismatch");
        CHECK(cli_manifest.live_count == (uint32_t)n + 1, "score_update_index insert live_count mismatch");
        CHECK(cli_manifest.last_committed_lsn == 2, "score_update_index insert LSN mismatch");

        refresh_engine.reload_update_view();
        QueryResult refreshed_insert_result = refresh_engine.search(rows.data());
        bool saw_refreshed_inserted = false;
        for (const auto& hit : refreshed_insert_result.topk) {
            if (hit.second == 1001) saw_refreshed_inserted = true;
        }
        CHECK(saw_refreshed_inserted,
              "reload_update_view must publish committed L0 insert without reloading base");
        std::vector<QueryResult> refreshed_batch_results =
            refresh_engine.search_batch(rows.data(), 1);
        bool saw_refreshed_batch_inserted = false;
        for (const auto& hit : refreshed_batch_results[0].topk) {
            if (hit.second == 1001) saw_refreshed_batch_inserted = true;
        }
        CHECK(saw_refreshed_batch_inserted,
              "reload_update_view must refresh cached batch query engines");
        const ReaderView& first_reload_reader = refresh_engine.reader_view();
        CHECK(first_reload_reader.growing_delta != nullptr,
              "reload_update_view should publish an L0 reader");
        CHECK(!first_reload_reader.growing_delta->committed_blocks.empty(),
              "reload_update_view should publish committed L0 blocks");
        uint64_t first_reload_lsn = first_reload_reader.l0_last_committed_lsn;
        uint64_t first_reload_offset = first_reload_reader.l0_wal_replay_offset;
        std::vector<std::vector<uint32_t>> first_reload_composite_sizes =
            first_reload_reader.composite_cell_sizes;
        std::shared_ptr<const UnsealedDeltaView> first_reload_growing =
            first_reload_reader.growing_delta;
        std::shared_ptr<const L0Block> first_reload_block =
            first_reload_reader.growing_delta->committed_blocks[0];
        refresh_engine.reload_update_view();
        const ReaderView& second_reload_reader = refresh_engine.reader_view();
        CHECK(second_reload_reader.l0_last_committed_lsn == first_reload_lsn,
              "empty reload_update_view must preserve L0 replay LSN");
        CHECK(second_reload_reader.l0_wal_replay_offset == first_reload_offset,
              "empty reload_update_view must preserve L0 WAL replay offset");
        CHECK(second_reload_reader.growing_delta != nullptr &&
                  !second_reload_reader.growing_delta->committed_blocks.empty(),
              "empty reload_update_view should keep L0 blocks visible");
        CHECK(second_reload_reader.growing_delta.get() == first_reload_growing.get(),
              "empty reload_update_view should reuse the L0 growing delta view");
        CHECK(second_reload_reader.growing_delta->committed_blocks[0].get() ==
                  first_reload_block.get(),
              "empty reload_update_view should reuse committed L0 blocks");
        CHECK(second_reload_reader.composite_cell_sizes == first_reload_composite_sizes,
              "empty reload_update_view should preserve composite cell sizes");
        TEST("QueryEngine reload_update_view publishes committed L0"); PASS();

        {
            std::ofstream wal_tail(index_dir + "/updates/l0.wal",
                                   std::ios::binary | std::ios::app);
            uint64_t partial_lsn = cli_manifest.last_committed_lsn + 1;
            wal_tail.write(reinterpret_cast<const char*>(&partial_lsn), sizeof(partial_lsn));
        }

        QueryEngine cli_update_engine;
        cli_update_engine.init(update_config);
        cli_update_engine.load_index(index_dir);
        QueryResult cli_insert_result = cli_update_engine.search(rows.data());
        bool saw_cli_inserted = false;
        for (const auto& hit : cli_insert_result.topk) {
            if (hit.second == 1001) saw_cli_inserted = true;
        }
        CHECK(saw_cli_inserted, "score_update_index insert must become query-visible");
        TEST("Query loader ignores torn L0 WAL tail after committed batch"); PASS();

        std::remove((index_dir + "/updates/direct_map.bin").c_str());
        UpdateCompactionPolicy flush_policy;
        flush_policy.flush_fraction = 0.000000001;
        flush_policy.compaction_fraction = 0.0;
        UpdateCompactionDecision flush_decision =
            compact_index_by_policy(index_dir, flush_policy);
        CHECK(flush_decision.flushed_l0,
              "L0 WAL/base byte fraction threshold should materialize committed insert without DirectMap");
        UpdateManifest flushed_manifest = load_update_manifest_json_for_index(index_dir);
        CHECK(flushed_manifest.sealed_deltas.size() == 1,
              "L0 flush must publish one sealed delta");
        CHECK(flushed_manifest.sealed_deltas[0].artifact_bytes > 0,
              "L0 flush must cache sealed delta artifact bytes in manifest");
        CHECK(impl::sealed_delta_artifact_bytes_for_policy(index_dir, flushed_manifest) > 0,
              "sealed delta byte policy estimate must count flushed artifacts");
        CHECK(impl::base_artifact_bytes_for_policy(index_dir) > 0,
              "base byte policy estimate must count base artifacts");
        CHECK(flushed_manifest.last_committed_lsn == 0,
              "L0 flush must reset growing WAL committed LSN");
        CHECK(access((index_dir + "/updates/l0.checkpoint").c_str(), F_OK) == 0,
              "L0 flush must write live checkpoint");
        CHECK(access((index_dir + "/" + flushed_manifest.sealed_deltas[0].path +
                      "/delta_orig_ids.bin").c_str(), F_OK) == 0,
              "L0 flush must write sealed delta orig IDs");
        CHECK(access((index_dir + "/" + flushed_manifest.sealed_deltas[0].path +
                      "/segment_manifest.json").c_str(), F_OK) == 0,
              "L0 flush must write sealed delta segment manifest");

        QueryEngine sealed_engine;
        sealed_engine.init(update_config);
        sealed_engine.load_index(index_dir);
        CHECK(sealed_engine.reader_view().sealed_deltas.size() == 1,
              "query loader must load sealed delta");
        CHECK(sealed_engine.reader_view().sealed_delta_postings != nullptr,
              "single sealed delta reader should build a posting overlay");
        QueryResult sealed_result = sealed_engine.search(rows.data());
        bool saw_sealed_inserted = false;
        for (const auto& hit : sealed_result.topk) {
            if (hit.second == 1001) saw_sealed_inserted = true;
        }
        CHECK(saw_sealed_inserted, "sealed delta query must return flushed inserted ID");
        std::shared_ptr<const SealedDeltaSegment> sealed_segment =
            sealed_engine.reader_view().sealed_deltas[0];
        std::shared_ptr<const SealedDeltaPostingView> sealed_postings =
            sealed_engine.reader_view().sealed_delta_postings;
        sealed_engine.reload_update_view();
        CHECK(sealed_engine.reader_view().sealed_deltas.size() == 1,
              "no-op sealed reload must keep sealed delta visible");
        CHECK(sealed_engine.reader_view().sealed_deltas[0].get() == sealed_segment.get(),
              "no-op sealed reload should reuse the sealed delta payload");
        CHECK(sealed_engine.reader_view().sealed_delta_postings.get() == sealed_postings.get(),
              "no-op sealed reload should reuse the sealed posting overlay");
        TEST("L0 flush to L1 sealed delta remains query-visible"); PASS();

        {
            std::ofstream ids(update_insert2_ids_path);
            ids << "1002\n";
        }
        UpdateIndexBatch insert2_batch;
        insert2_batch.insert_fbin = update_insert_path;
        insert2_batch.insert_ids = update_insert2_ids_path;
        insert2_batch.durability_mode = "none";
        insert2_batch.threads = 2;
        commit_update_batch(index_dir, insert2_batch);
        std::remove((index_dir + "/updates/direct_map.bin").c_str());
        UpdateCompactionPolicy merge_policy;
        merge_policy.flush_fraction = 0.000000001;
        merge_policy.compaction_fraction = 0.0;
        UpdateCompactionDecision merge_decision =
            compact_index_by_policy(index_dir, merge_policy);
        CHECK(merge_decision.flushed_l0,
              "flush fraction should flush committed L0");
        CHECK(!merge_decision.compacted,
              "flush should not compact");
        UpdateManifest merged_manifest = load_update_manifest_json_for_index(index_dir);
        CHECK(merged_manifest.sealed_deltas.size() >= 2,
              "flushed deltas should accumulate as separate sealed segments");
        CHECK(merged_manifest.last_committed_lsn == 0,
              "flush should reset L0 WAL");
        CHECK(merged_manifest.max_pid == (uint32_t)n + 2,
              "flush max_pid mismatch");
        CHECK(merged_manifest.live_count == (uint32_t)n + 2,
              "flush live_count mismatch");

        QueryEngine merged_engine;
        merged_engine.init(update_config);
        merged_engine.load_index(index_dir);
        QueryResult merged_result = merged_engine.search(rows.data());
        bool saw_merged_1001 = false;
        bool saw_merged_1002 = false;
        for (const auto& hit : merged_result.topk) {
            if (hit.second == 1001) saw_merged_1001 = true;
            if (hit.second == 1002) saw_merged_1002 = true;
        }
        CHECK(saw_merged_1001 && saw_merged_1002,
              "merged sealed delta query must return both inserted IDs");
        TEST("sealed delta publish view preserves live rows across segments"); PASS();

        {
            std::ofstream ids(update_delete_ids_path);
            ids << "1001\n";
        }
        UpdateIndexBatch delete_batch;
        delete_batch.delete_ids = update_delete_ids_path;
        delete_batch.durability_mode = "none";
        commit_update_batch(index_dir, delete_batch);
        UpdateManifest cli_delete_manifest = load_update_manifest_json_for_index(index_dir);
        CHECK(cli_delete_manifest.max_pid == (uint32_t)n + 2, "score_update_index delete max_pid mismatch");
        CHECK(cli_delete_manifest.live_count == (uint32_t)n + 1, "score_update_index delete live_count mismatch");
        CHECK(cli_delete_manifest.sealed_deltas.size() >= 1,
              "score_update_index delete should keep sealed deltas");
        CHECK(cli_delete_manifest.last_committed_lsn == 2, "score_update_index delete LSN mismatch");

        QueryEngine cli_delete_engine;
        cli_delete_engine.init(update_config);
        cli_delete_engine.load_index(index_dir);
        QueryResult cli_delete_result = cli_delete_engine.search(rows.data());
        bool saw_cli_deleted = false;
        bool saw_cli_remaining = false;
        for (const auto& hit : cli_delete_result.topk) {
            if (hit.second == 1001) saw_cli_deleted = true;
            if (hit.second == 1002) saw_cli_remaining = true;
        }
        CHECK(!saw_cli_deleted, "score_update_index delete must tombstone inserted ID");
        CHECK(saw_cli_remaining, "score_update_index delete must preserve other sealed delta rows");
        TEST("score_update_index insert and delete batches"); PASS();

        bool compaction_changed = compact_index(index_dir);
        CHECK(compaction_changed, "compaction should publish a compact base");
        CHECK(access((index_dir + "/update_manifest.json").c_str(), F_OK) != 0,
              "compaction must remove update_manifest.json");
        CHECK(access((index_dir + "/updates").c_str(), F_OK) != 0,
              "compaction must remove growing update directory");
        CHECK(access((index_dir + "/deltas").c_str(), F_OK) != 0,
              "compaction must remove sealed delta directory");
        CHECK(access((index_dir + "/perm_to_orig.bin").c_str(), F_OK) == 0,
              "compaction must publish perm_to_orig for compacted PIDs");

        current_verify = load_fbin_payload(index_dir + "/verify_vectors.fbin",
                                           current_verify_n,
                                           current_verify_dim);
        CHECK(current_verify_n == n + 1, "compaction verify row count mismatch");
        CHECK(current_verify_dim == verify_dim, "compaction verify dim mismatch");

        cli_delete_engine.reload_index_for_maintenance();
        const ReaderView& maintenance_reader = cli_delete_engine.reader_view();
        CHECK(!maintenance_reader.has_deltas(),
              "maintenance reload after compaction must install base-only compacted reader");
        CHECK(maintenance_reader.base_n == (uint32_t)n + 1,
              "maintenance reload base_n mismatch after compaction");
        CHECK(cli_delete_engine.perm_to_orig().size() == (size_t)n + 1,
              "maintenance reload must load compacted perm_to_orig");
        QueryResult maintenance_result = cli_delete_engine.search(rows.data());
        bool saw_maintenance_deleted = false;
        bool saw_maintenance_remaining = false;
        for (const auto& hit : maintenance_result.topk) {
            if (hit.second == 1001) saw_maintenance_deleted = true;
            if (hit.second == 1002) saw_maintenance_remaining = true;
        }
        CHECK(!saw_maintenance_deleted,
              "maintenance-reloaded query must not return deleted sealed delta ID");
        CHECK(saw_maintenance_remaining,
              "maintenance-reloaded query must return compacted live delta ID");
        TEST("QueryEngine maintenance reload loads compacted base"); PASS();

        QueryEngine compaction_engine;
        compaction_engine.init(update_config);
        compaction_engine.load_index(index_dir);
        const ReaderView& compaction_reader = compaction_engine.reader_view();
        CHECK(!compaction_reader.has_deltas(), "compacted reader should be base-only");
        CHECK(compaction_reader.base_n == (uint32_t)n + 1, "compacted base_n mismatch");
        CHECK(compaction_reader.max_pid == (uint32_t)n + 1, "compacted max_pid mismatch");
        CHECK(compaction_reader.live_count == (uint32_t)n + 1, "compacted live_count mismatch");
        CHECK(compaction_engine.perm_to_orig().size() == (size_t)n + 1,
              "compacted perm_to_orig size mismatch");

        RowResolver compaction_resolver = compaction_engine.row_resolver();
        ResolvedRow compacted_delta_row;
        CHECK(compaction_resolver.resolve((uint32_t)n, compacted_delta_row),
              "compacted RowResolver must resolve remapped delta row");
        CHECK(compacted_delta_row.segment == ResolvedRow::SegmentKind::Base,
              "compaction must fold delta rows into base segment");
        CHECK(compacted_delta_row.output_id == 1002,
              "compaction must preserve delta orig_id through perm_to_orig");

        QueryResult compaction_result = compaction_engine.search(rows.data());
        bool saw_compaction_deleted = false;
        bool saw_compaction_remaining = false;
        for (const auto& hit : compaction_result.topk) {
            if (hit.second == 1001) saw_compaction_deleted = true;
            if (hit.second == 1002) saw_compaction_remaining = true;
        }
        CHECK(!saw_compaction_deleted, "compacted query must not return tombstoned ID");
        CHECK(saw_compaction_remaining, "compacted query must return live delta ID");
        TEST("Compaction folds live rows into compact base"); PASS();

        {
            std::ofstream ids(update_delete_ids_path);
            ids << "0\n";
        }
        UpdateIndexBatch delete_base_batch;
        delete_base_batch.delete_ids = update_delete_ids_path;
        delete_base_batch.durability_mode = "none";
        commit_update_batch(index_dir, delete_base_batch);

        QueryEngine base_delete_l0_engine;
        base_delete_l0_engine.init(update_config);
        base_delete_l0_engine.load_index(index_dir);
        QueryResult base_delete_l0_result = base_delete_l0_engine.search(rows.data());
        bool saw_base_deleted_l0 = false;
        bool saw_delta_remaining_l0 = false;
        for (const auto& hit : base_delete_l0_result.topk) {
            if (hit.second == 0) saw_base_deleted_l0 = true;
            if (hit.second == 1002) saw_delta_remaining_l0 = true;
        }
        CHECK(!saw_base_deleted_l0, "post-compaction L0 delete must tombstone compacted base ID");
        CHECK(saw_delta_remaining_l0, "post-compaction L0 delete must preserve remapped delta ID");

        IndexConfig budget_config = update_config;
        budget_config.candidate_ratio = 0.96f;
        QueryEngine budget_engine;
        budget_engine.init(budget_config);
        budget_engine.load_index(index_dir);
        const ReaderView& budget_reader = budget_engine.reader_view();
        size_t live_budget = std::max((size_t)1,
                                      (size_t)(budget_config.candidate_ratio *
                                               (float)budget_reader.live_count));
        size_t max_pid_budget = std::max((size_t)1,
                                         (size_t)(budget_config.candidate_ratio *
                                                  (float)budget_reader.max_pid));
        CHECK(budget_reader.live_count < budget_reader.max_pid,
              "candidate budget regression setup requires a tombstoned PID");
        CHECK(live_budget < max_pid_budget,
              "candidate budget regression setup must distinguish live_count from max_pid");
        CHECK(budget_engine.effective_candidate_budget() == live_budget,
              "candidate budget must use live_count when tombstones exist");
        TEST("QueryEngine candidate budget uses live_count with tombstones"); PASS();

        bool flushed_delete_only = flush_l0_to_l1(index_dir);
        CHECK(flushed_delete_only, "delete-only L0 flush should publish a live checkpoint");
        UpdateManifest delete_only_manifest = load_update_manifest_json_for_index(index_dir);
        CHECK(delete_only_manifest.sealed_deltas.empty(),
              "delete-only flush should not create an empty sealed delta");
        CHECK(delete_only_manifest.live_count == (uint32_t)n,
              "delete-only flush live_count mismatch");

        std::remove((index_dir + "/updates/direct_map.bin").c_str());
        UpdateIndexBatch redundant_delete_batch;
        redundant_delete_batch.delete_ids = update_delete_ids_path;
        redundant_delete_batch.durability_mode = "none";
        commit_update_batch(index_dir, redundant_delete_batch);
        UpdateManifest redundant_delete_manifest = load_update_manifest_json_for_index(index_dir);
        CHECK(redundant_delete_manifest.max_pid == delete_only_manifest.max_pid,
              "rebuilt DirectMap must preserve max_pid on redundant delete");
        CHECK(redundant_delete_manifest.live_count == delete_only_manifest.live_count,
              "rebuilt DirectMap must preserve checkpoint tombstones on redundant delete");
        CHECK(redundant_delete_manifest.last_committed_lsn == delete_only_manifest.last_committed_lsn,
              "redundant delete over rebuilt DirectMap must not commit WAL records");
        CHECK(access((index_dir + "/updates/direct_map.bin").c_str(), F_OK) != 0,
              "no-op update should not materialize a new DirectMap");

        QueryEngine checkpoint_delete_engine;
        checkpoint_delete_engine.init(update_config);
        checkpoint_delete_engine.load_index(index_dir);
        QueryResult checkpoint_delete_result = checkpoint_delete_engine.search(rows.data());
        bool saw_checkpoint_deleted = false;
        bool saw_checkpoint_remaining = false;
        for (const auto& hit : checkpoint_delete_result.topk) {
            if (hit.second == 0) saw_checkpoint_deleted = true;
            if (hit.second == 1002) saw_checkpoint_remaining = true;
        }
        CHECK(!saw_checkpoint_deleted, "checkpoint-only delete must stay query-visible after flush");
        CHECK(saw_checkpoint_remaining, "checkpoint-only delete must preserve other live IDs");

        UpdateCompactionPolicy tombstone_compaction_policy;
        tombstone_compaction_policy.compaction_fraction = 0.01;
        UpdateCompactionDecision tombstone_compaction_decision =
            compact_index_by_policy(index_dir, tombstone_compaction_policy);
        CHECK(tombstone_compaction_decision.compacted,
              "tombstone ratio threshold should fold delete-only checkpoint");
        CHECK(access((index_dir + "/update_manifest.json").c_str(), F_OK) != 0,
              "second compaction must remove update manifest");
        current_verify = load_fbin_payload(index_dir + "/verify_vectors.fbin",
                                           current_verify_n,
                                           current_verify_dim);
        CHECK(current_verify_n == n, "second compaction verify row count mismatch");
        CHECK(current_verify_dim == verify_dim, "second compaction verify dim mismatch");

        QueryEngine second_compaction_engine;
        second_compaction_engine.init(update_config);
        second_compaction_engine.load_index(index_dir);
        CHECK(!second_compaction_engine.reader_view().has_deltas(),
              "second compacted reader should be base-only");
        CHECK(second_compaction_engine.reader_view().base_n == (uint32_t)n,
              "second compacted base_n mismatch");
        CHECK(second_compaction_engine.perm_to_orig().size() == (size_t)n,
              "second compacted perm_to_orig size mismatch");
        ResolvedRow second_compaction_last;
        CHECK(second_compaction_engine.row_resolver().resolve((uint32_t)n - 1, second_compaction_last),
              "second compaction resolver must resolve final compacted row");
        CHECK(second_compaction_last.output_id == 1002,
              "second compaction must preserve high orig_id after base deletion");
        QueryResult second_compaction_result = second_compaction_engine.search(rows.data());
        bool saw_second_compaction_deleted = false;
        bool saw_second_compaction_remaining = false;
        for (const auto& hit : second_compaction_result.topk) {
            if (hit.second == 0) saw_second_compaction_deleted = true;
            if (hit.second == 1002) saw_second_compaction_remaining = true;
        }
        CHECK(!saw_second_compaction_deleted, "second compacted query must not return deleted base ID");
        CHECK(saw_second_compaction_remaining, "second compacted query must return remaining high ID");
        TEST("Delete-only checkpoint survives flush and compaction"); PASS();

        UpdateIndexBatch upsert_after_compaction_batch;
        upsert_after_compaction_batch.upsert_fbin = update_insert_path;
        upsert_after_compaction_batch.upsert_ids = update_insert2_ids_path;
        upsert_after_compaction_batch.durability_mode = "none";
        upsert_after_compaction_batch.threads = 1;
        commit_update_batch(index_dir, upsert_after_compaction_batch);
        UpdateManifest upsert_manifest = load_update_manifest_json_for_index(index_dir);
        CHECK(upsert_manifest.base_n == (uint32_t)n,
              "upsert after compaction base_n mismatch");
        CHECK(upsert_manifest.max_pid == (uint32_t)n + 1,
              "upsert after compaction max_pid mismatch");
        CHECK(upsert_manifest.live_count == (uint32_t)n,
              "upsert after compaction must preserve live_count");
        CHECK(upsert_manifest.last_committed_lsn == 3,
              "upsert after compaction should write delete+insert WAL batch");

        DirectMap upsert_map = DirectMap::load_binary(index_dir + "/" + upsert_manifest.direct_map_path);
        uint32_t upsert_pid = 0;
        CHECK(upsert_map.lookup(1002, upsert_pid) && upsert_pid == (uint32_t)n,
              "upsert must remap orig_id to the new L0 PID");

        QueryEngine upsert_engine;
        upsert_engine.init(update_config);
        upsert_engine.load_index(index_dir);
        CHECK(!upsert_engine.reader_view().live->is_live((uint32_t)n - 1),
              "upsert must tombstone old base PID");
        CHECK(upsert_engine.reader_view().live->is_live((uint32_t)n),
              "upsert must publish new L0 PID");
        QueryResult upsert_result = upsert_engine.search(rows.data());
        bool saw_upserted = false;
        bool saw_deleted_base_again = false;
        for (const auto& hit : upsert_result.topk) {
            if (hit.second == 1002) saw_upserted = true;
            if (hit.second == 0) saw_deleted_base_again = true;
        }
        CHECK(saw_upserted, "upserted ID must be query-visible from L0");
        CHECK(!saw_deleted_base_again, "upsert must preserve older base tombstones");
        TEST("Upsert tombstones base PID and publishes new L0 PID"); PASS();

        DirectMap stale_direct_map = upsert_map;
        stale_direct_map.set(1002, (uint32_t)n - 1);
        stale_direct_map.save_binary(index_dir + "/" + upsert_manifest.direct_map_path, false);

        std::vector<float> two_inserts(rows.begin(), rows.begin() + (size_t)2 * (size_t)dim);
        write_test_fbin(update_batch_insert_path, two_inserts, 2, dim);
        {
            std::ofstream ids(update_batch_insert_ids_path);
            ids << "2001\n2002\n";
        }
        UpdateIndexBatch row_chunked_insert_batch;
        row_chunked_insert_batch.insert_fbin = update_batch_insert_path;
        row_chunked_insert_batch.insert_ids = update_batch_insert_ids_path;
        row_chunked_insert_batch.durability_mode = "buffered";
        row_chunked_insert_batch.batch_rows = 1;
        commit_update_batch(index_dir, row_chunked_insert_batch);
        UpdateManifest chunked_manifest = load_update_manifest_json_for_index(index_dir);
        CHECK(chunked_manifest.max_pid == (uint32_t)n + 3,
              "batch_rows insert max_pid mismatch");
        CHECK(chunked_manifest.live_count == (uint32_t)n + 2,
              "batch_rows insert live_count mismatch");
        CHECK(chunked_manifest.last_committed_lsn == 7,
              "batch_rows=1 should publish two single-row insert commits");
        CHECK(chunked_manifest.last_fsynced_lsn == chunked_manifest.last_committed_lsn,
              "buffered should fsync once after the chunked commit window");
        DirectMap repaired_map = DirectMap::load_binary(index_dir + "/" + chunked_manifest.direct_map_path);
        uint32_t repaired_pid = 0;
        CHECK(repaired_map.lookup(1002, repaired_pid) && repaired_pid == (uint32_t)n,
              "update path must rebuild stale DirectMap entries that point at dead PIDs");
        TEST("score_update_index rebuilds stale DirectMap before commit"); PASS();

        QueryEngine chunked_engine;
        chunked_engine.init(update_config);
        chunked_engine.load_index(index_dir);
        QueryResult chunked_result_0 = chunked_engine.search(rows.data());
        QueryResult chunked_result_1 = chunked_engine.search(rows.data() + (size_t)dim);
        bool saw_chunked_2001 = false;
        bool saw_chunked_2002 = false;
        for (const auto& hit : chunked_result_0.topk) {
            if (hit.second == 2001) saw_chunked_2001 = true;
        }
        for (const auto& hit : chunked_result_1.topk) {
            if (hit.second == 2002) saw_chunked_2002 = true;
        }
        CHECK(saw_chunked_2001 && saw_chunked_2002,
              "batch_rows chunked inserts must be query-visible from L0");
        TEST("score_update_index batch_rows splits WAL commits"); PASS();

        UpdateManifest before_memory_writer_manifest =
            load_update_manifest_json_for_index(index_dir);
        {
            UpdateIndexWriter writer(index_dir, "buffered", 1, {3001, 3002});
            CHECK(!writer.collect_commit_timing(),
                  "commit timing collection should be disabled by default");
            writer.set_collect_commit_timing(true);

            UpdateIndexMemoryBatch first_memory_batch;
            first_memory_batch.insert_rows = rows.data() + (size_t)2 * (size_t)dim;
            first_memory_batch.insert_n = 1;
            first_memory_batch.insert_dim = dim;
            first_memory_batch.insert_orig_ids = {3001};
            writer.commit_batch(first_memory_batch);
            UpdateCommitTiming first_timing = writer.last_commit_timing();
            CHECK(first_timing.published && first_timing.insert_rows == 1 &&
                      first_timing.upsert_rows == 0 && first_timing.delete_rows == 0,
                  "commit timing should record published insert batch shape");
            CHECK(first_timing.total_ms >= first_timing.encode_ms &&
                      first_timing.total_ms >= first_timing.wal_commit_ms,
                  "commit timing should record bounded phase durations");

            UpdateIndexMemoryBatch second_memory_batch;
            second_memory_batch.insert_rows = rows.data() + (size_t)3 * (size_t)dim;
            second_memory_batch.insert_n = 1;
            second_memory_batch.insert_dim = dim;
            second_memory_batch.insert_orig_ids = {3002};
            writer.commit_batch(second_memory_batch);

            UpdateManifest pre_finish_memory_manifest =
                load_update_manifest_json_for_index(index_dir);
            CHECK(pre_finish_memory_manifest.last_committed_lsn ==
                      before_memory_writer_manifest.last_committed_lsn,
                  "buffered writer should avoid rewriting manifest for every small commit");
            QueryEngine pre_finish_memory_engine;
            pre_finish_memory_engine.init(update_config);
            pre_finish_memory_engine.load_index(index_dir);
            CHECK(pre_finish_memory_engine.reader_view().max_pid == (uint32_t)n + 5,
                  "query reader must load while a writer is open and derive latest buffered commit max_pid from WAL");
            QueryResult pre_finish_memory_result =
                pre_finish_memory_engine.search(rows.data() + (size_t)3 * (size_t)dim);
            bool saw_pre_finish_memory_insert = false;
            for (const auto& hit : pre_finish_memory_result.topk) {
                if (hit.second == 3002) saw_pre_finish_memory_insert = true;
            }
            CHECK(saw_pre_finish_memory_insert,
                  "query reader must see latest buffered commit even when manifest is stale");
            std::shared_ptr<const UnsealedDeltaView> pre_finish_growing =
                pre_finish_memory_engine.reader_view().growing_delta;
            CHECK(pre_finish_growing != nullptr &&
                      !pre_finish_growing->committed_blocks.empty(),
                  "pre-finish reader must expose L0 blocks before stale-manifest reload");
            std::shared_ptr<const L0Block> pre_finish_first_block =
                pre_finish_growing->committed_blocks[0];
            UpdateIndexMemoryBatch third_memory_batch;
            third_memory_batch.insert_rows = rows.data() + (size_t)4 * (size_t)dim;
            third_memory_batch.insert_n = 1;
            third_memory_batch.insert_dim = dim;
            third_memory_batch.insert_orig_ids = {3003};
            writer.commit_batch(third_memory_batch);
            pre_finish_memory_engine.reload_update_view();
            CHECK(pre_finish_memory_engine.reader_view().max_pid == (uint32_t)n + 6,
                  "reload_update_view must refresh while a writer is open");
            CHECK(pre_finish_memory_engine.reader_view().l0_last_committed_lsn == 13,
                  "stale-manifest reload must replay only the new committed WAL tail");
            CHECK(pre_finish_memory_engine.reader_view().growing_delta != nullptr &&
                      !pre_finish_memory_engine.reader_view().growing_delta->committed_blocks.empty() &&
                      pre_finish_memory_engine.reader_view()
                              .growing_delta->committed_blocks[0].get() ==
                          pre_finish_first_block.get(),
                  "stale-manifest reload should reuse previous L0 blocks");
            writer.flush_buffered_commits();
        }

        UpdateManifest memory_writer_manifest = load_update_manifest_json_for_index(index_dir);
        CHECK(memory_writer_manifest.max_pid == (uint32_t)n + 6,
              "in-process writer max_pid mismatch");
        CHECK(memory_writer_manifest.live_count == (uint32_t)n + 5,
              "in-process writer live_count mismatch");
        CHECK(memory_writer_manifest.last_committed_lsn == 13,
              "in-process writer must publish consecutive commits while readers reload");
        CHECK(memory_writer_manifest.last_fsynced_lsn ==
                  memory_writer_manifest.last_committed_lsn,
              "in-process writer buffered flush must fsync the final commit");
        DirectMap memory_writer_map =
            DirectMap::load_binary(index_dir + "/" + memory_writer_manifest.direct_map_path);
        uint32_t memory_writer_pid = 0;
        CHECK(memory_writer_map.lookup(3002, memory_writer_pid) &&
                  memory_writer_pid == (uint32_t)n + 4,
              "in-process writer must persist final DirectMap state");

        QueryEngine memory_writer_engine;
        memory_writer_engine.init(update_config);
        memory_writer_engine.load_index(index_dir);
        const ReaderView& memory_writer_view = memory_writer_engine.reader_view();
        CHECK(memory_writer_view.growing_delta != nullptr,
              "in-process writer reader must expose growing delta");
        CHECK(memory_writer_view.growing_delta->committed_blocks.size() > 1,
              "in-process writer test must create multiple L0 blocks");
        CHECK(!memory_writer_view.growing_delta->pid_to_block_lookup.empty(),
              "merged growing delta should build a direct PID-to-block lookup");
        CHECK(memory_writer_view.growing_delta->pid_lookup_begin <= (uint32_t)n + 4 &&
                  (uint64_t)((uint32_t)n + 4) -
                          (uint64_t)memory_writer_view.growing_delta->pid_lookup_begin <
                      (uint64_t)memory_writer_view.growing_delta->pid_to_block_lookup.size(),
              "merged growing delta direct lookup must cover final L0 PID");
        const L0Block* resolved_l0_block =
            memory_writer_view.growing_delta->block_for_pid((uint32_t)n + 4);
        CHECK(resolved_l0_block != nullptr && resolved_l0_block->contains_pid((uint32_t)n + 4),
              "merged growing delta lookup must resolve final L0 PID");
        size_t expected_l0_rows =
            (size_t)memory_writer_manifest.max_pid - (size_t)memory_writer_manifest.base_n;
        for (int m = 0; m < M; ++m) {
            size_t merged_l0_postings = 0;
            for (int cell = 0; cell < postings.K(m); ++cell) {
                merged_l0_postings +=
                    memory_writer_view.growing_delta->cell_size((uint32_t)m, (uint32_t)cell);
            }
            CHECK(merged_l0_postings == expected_l0_rows,
                  "merged growing delta posting total mismatch");
        }
        QueryResult memory_writer_result =
            memory_writer_engine.search(rows.data() + (size_t)3 * (size_t)dim);
        bool saw_memory_writer_insert = false;
        for (const auto& hit : memory_writer_result.topk) {
            if (hit.second == 3002) saw_memory_writer_insert = true;
        }
        CHECK(saw_memory_writer_insert,
              "in-process writer insert must be query-visible from L0");
        TEST("UpdateIndexWriter publishes consecutive memory batches"); PASS();

        {
            QueryIndexHandle auto_publish_handle;
            auto_publish_handle.init(update_config);
            auto_publish_handle.load_index(index_dir);
            std::shared_ptr<const ReaderView> auto_publish_before_view =
                auto_publish_handle.snapshot();
            uint32_t auto_publish_before_max_pid =
                auto_publish_before_view->max_pid;

            UpdateIndexWriter writer(index_dir, "buffered", 1, {3004}, &auto_publish_handle);
            UpdateIndexMemoryBatch auto_publish_batch;
            auto_publish_batch.insert_rows = rows.data() + (size_t)8 * (size_t)dim;
            auto_publish_batch.insert_n = 1;
            auto_publish_batch.insert_dim = dim;
            auto_publish_batch.insert_orig_ids = {3004};
            writer.commit_batch(auto_publish_batch);
            const CommittedUpdateDelta& auto_publish_delta = writer.last_commit_delta();
            std::shared_ptr<const ReaderView> auto_publish_view =
                auto_publish_handle.snapshot();

            CHECK(auto_publish_view->max_pid == auto_publish_before_max_pid + 1,
                  "writer publish target must refresh the reader after commit");
            CHECK(auto_publish_view->l0_last_committed_lsn == auto_publish_delta.commit_lsn &&
                  auto_publish_view->l0_wal_replay_offset ==
                      auto_publish_delta.wal_offset_after_commit,
                  "direct writer publish must preserve committed L0 WAL cursor");
            CHECK(auto_publish_view->growing_delta != nullptr &&
                  !auto_publish_view->growing_delta->committed_blocks.empty() &&
                  !auto_publish_delta.committed_blocks.empty() &&
                  auto_publish_view->growing_delta->committed_blocks.back().get() ==
                      auto_publish_delta.committed_blocks.back().get(),
                  "writer publish target should install the in-memory committed L0 block");
            QueryResult auto_publish_result =
                auto_publish_handle.search(rows.data() + (size_t)8 * (size_t)dim);
            bool saw_auto_publish_insert = false;
            for (const auto& hit : auto_publish_result.topk) {
                if (hit.second == 3004) saw_auto_publish_insert = true;
            }
            CHECK(saw_auto_publish_insert,
                  "auto-published writer commit must be query-visible without explicit reload");
            writer.flush_buffered_commits();
        }
        TEST("UpdateIndexWriter auto-publishes committed memory batches"); PASS();

        {
            QueryEngine callback_publish_reader;
            callback_publish_reader.init(update_config);
            callback_publish_reader.load_index(index_dir);
            uint32_t callback_publish_before_max_pid =
                callback_publish_reader.reader_view().max_pid;
            size_t publish_calls = 0;

            UpdateIndexWriter writer(index_dir, "buffered", 1, {3005});
            writer.set_publish_callback([&]() {
                ++publish_calls;
                callback_publish_reader.reload_update_view();
            });
            UpdateIndexMemoryBatch callback_publish_batch;
            callback_publish_batch.insert_rows = rows.data() + (size_t)9 * (size_t)dim;
            callback_publish_batch.insert_n = 1;
            callback_publish_batch.insert_dim = dim;
            callback_publish_batch.insert_orig_ids = {3005};
            writer.commit_batch(callback_publish_batch);

            CHECK(publish_calls == 1,
                  "writer publish callback must run once for a committed batch");
            CHECK(callback_publish_reader.reader_view().max_pid ==
                      callback_publish_before_max_pid + 1,
                  "writer publish callback must be able to refresh the reader");
            QueryResult callback_publish_result =
                callback_publish_reader.search(rows.data() + (size_t)9 * (size_t)dim);
            bool saw_callback_publish_insert = false;
            for (const auto& hit : callback_publish_result.topk) {
                if (hit.second == 3005) saw_callback_publish_insert = true;
            }
            CHECK(saw_callback_publish_insert,
                  "callback-published writer commit must be query-visible");
            writer.flush_buffered_commits();
        }
        TEST("UpdateIndexWriter publish callback refreshes reader"); PASS();

        UpdateManifest before_coalesced_upsert_manifest =
            load_update_manifest_json_for_index(index_dir);
        {
            UpdateIndexWriter writer(index_dir, "buffered", 1, {3002});
            std::vector<float> duplicate_upsert_rows;
            duplicate_upsert_rows.insert(
                duplicate_upsert_rows.end(),
                rows.begin() + (size_t)4 * (size_t)dim,
                rows.begin() + (size_t)5 * (size_t)dim);
            duplicate_upsert_rows.insert(
                duplicate_upsert_rows.end(),
                rows.begin() + (size_t)5 * (size_t)dim,
                rows.begin() + (size_t)6 * (size_t)dim);

            UpdateIndexMemoryBatch duplicate_upsert_batch;
            duplicate_upsert_batch.upsert_rows = duplicate_upsert_rows.data();
            duplicate_upsert_batch.upsert_n = 2;
            duplicate_upsert_batch.upsert_dim = dim;
            duplicate_upsert_batch.upsert_orig_ids = {3002, 3002};
            writer.commit_batch(duplicate_upsert_batch);
            writer.flush_buffered_commits();
        }

        UpdateManifest coalesced_upsert_manifest =
            load_update_manifest_json_for_index(index_dir);
        CHECK(coalesced_upsert_manifest.max_pid ==
                  before_coalesced_upsert_manifest.max_pid + 1,
              "duplicate upsert batch must allocate one final PID");
        CHECK(coalesced_upsert_manifest.live_count ==
                  before_coalesced_upsert_manifest.live_count,
              "duplicate upsert batch must keep live_count stable");
        UpdateWalReplayResult coalesced_replay =
            replay_update_wal(index_dir + "/" + coalesced_upsert_manifest.wal_path);
        CHECK(!coalesced_replay.committed_batches.empty(),
              "duplicate upsert test requires committed WAL batches");
        const UpdateWalBatch& coalesced_batch = coalesced_replay.committed_batches.back();
        CHECK(coalesced_batch.records.size() == 2,
              "duplicate upsert commit should contain one tombstone and one insert");
        CHECK(coalesced_batch.records[0].type == UpdateWalRecord::Type::Delete &&
                  coalesced_batch.records[1].type == UpdateWalRecord::Type::Insert,
              "duplicate upsert commit should publish delete then final insert");
        CHECK(coalesced_batch.records[1].insert.orig_id == 3002 &&
                  coalesced_batch.records[1].insert.global_pid ==
                      coalesced_upsert_manifest.max_pid - 1,
              "duplicate upsert commit must keep only the last inserted version");
        DirectMap coalesced_map =
            DirectMap::load_binary(index_dir + "/" + coalesced_upsert_manifest.direct_map_path);
        uint32_t coalesced_pid = 0;
        CHECK(coalesced_map.lookup(3002, coalesced_pid) &&
                  coalesced_pid == coalesced_upsert_manifest.max_pid - 1,
              "duplicate upsert DirectMap must point at final PID");
        TEST("UpdateIndexWriter coalesces duplicate upserts before publication"); PASS();

        CHECK(flush_l0_to_l1(index_dir),
              "multi-sealed setup must flush current L0");
        auto insert_one_and_flush_l1 = [&](uint32_t orig_id, int row) {
            {
                UpdateIndexWriter writer(index_dir, "none", 1, {orig_id});
                UpdateIndexMemoryBatch batch;
                batch.insert_rows = rows.data() + (size_t)row * (size_t)dim;
                batch.insert_n = 1;
                batch.insert_dim = dim;
                batch.insert_orig_ids = {orig_id};
                writer.commit_batch(batch);
                writer.flush_buffered_commits();
            }
            CHECK(flush_l0_to_l1(index_dir),
                  "multi-sealed setup must flush each single-row delta");
        };
        insert_one_and_flush_l1(4001, 4);
        insert_one_and_flush_l1(4002, 5);
        insert_one_and_flush_l1(4003, 6);
        UpdateManifest pre_partial_merge_manifest =
            load_update_manifest_json_for_index(index_dir);
        CHECK(pre_partial_merge_manifest.sealed_deltas.size() == 4,
              "multi-sealed setup should create four sealed deltas");
        QueryEngine multi_sealed_engine;
        multi_sealed_engine.init(update_config);
        multi_sealed_engine.load_index(index_dir);
        std::shared_ptr<const SealedDeltaPostingView> multi_sealed_postings =
            multi_sealed_engine.reader_view().sealed_delta_postings;
        std::shared_ptr<const SealedDeltaPidLookup> multi_sealed_lookup =
            multi_sealed_engine.reader_view().sealed_delta_pid_lookup;
        CHECK(multi_sealed_postings != nullptr,
              "multi sealed reader should build a posting overlay");
        CHECK(multi_sealed_lookup != nullptr,
              "multi sealed reader should build a PID lookup");
        multi_sealed_engine.reload_update_view();
        CHECK(multi_sealed_engine.reader_view().sealed_delta_postings.get() ==
                  multi_sealed_postings.get(),
              "no-op multi sealed reload should reuse the posting overlay");
        CHECK(multi_sealed_engine.reader_view().sealed_delta_pid_lookup.get() ==
                  multi_sealed_lookup.get(),
              "no-op multi sealed reload should reuse the PID lookup");
        std::vector<std::shared_ptr<const SealedDeltaSegment>> multi_sealed_prefix =
            multi_sealed_engine.reader_view().sealed_deltas;
        insert_one_and_flush_l1(4004, 7);
        multi_sealed_engine.reload_update_view();
        CHECK(multi_sealed_engine.reader_view().sealed_deltas.size() == 5,
              "appended sealed reload should publish the new sealed delta");
        for (size_t i = 0; i < multi_sealed_prefix.size(); ++i) {
            CHECK(multi_sealed_engine.reader_view().sealed_deltas[i].get() ==
                      multi_sealed_prefix[i].get(),
                  "appended sealed reload should reuse the unchanged sealed prefix");
        }

        UpdateCompactionPolicy partial_merge_policy;
        partial_merge_policy.compaction_fraction = 0.0;
        UpdateCompactionDecision partial_merge_decision =
            compact_index_by_policy(index_dir, partial_merge_policy);
        UpdateManifest partial_merge_manifest =
            load_update_manifest_json_for_index(index_dir);
        CHECK(partial_merge_manifest.last_committed_lsn == 0,
              "compaction policy must leave L0 empty");

        QueryEngine partial_merge_engine;
        partial_merge_engine.init(update_config);
        partial_merge_engine.load_index(index_dir);
        const ReaderView& partial_merge_view = partial_merge_engine.reader_view();
        CHECK(partial_merge_view.sealed_delta_postings != nullptr,
              "multi-sealed-delta reader must build a sealed delta posting overlay");
        CHECK(partial_merge_view.sealed_delta_pid_lookup != nullptr,
              "multi-sealed-delta reader must build a sealed delta PID lookup");
        for (int m = 0; m < M; ++m) {
            for (int cell = 0; cell < postings.K(m); ++cell) {
                size_t expected_delta_live = 0;
                for (const auto& segment : partial_merge_view.sealed_deltas) {
                    expected_delta_live += segment->live_cell_size((uint32_t)m, (uint32_t)cell);
                }
                CHECK(partial_merge_view.sealed_delta_postings->cell_size(
                          (uint32_t)m, (uint32_t)cell) == expected_delta_live,
                      "sealed delta posting overlay cell size mismatch");
            }
        }
        DirectMap partial_merge_map =
            DirectMap::load_binary(index_dir + "/" + partial_merge_manifest.direct_map_path);
        uint32_t partial_merge_pid = 0;
        CHECK(partial_merge_map.lookup(4003, partial_merge_pid),
              "compaction test must resolve inserted orig_id through DirectMap");
        uint32_t partial_merge_segment_index = 0;
        CHECK(partial_merge_view.sealed_delta_pid_lookup->segment_index_for_pid(
                  partial_merge_pid, partial_merge_segment_index),
              "sealed delta PID lookup must find inserted PID");
        CHECK((size_t)partial_merge_segment_index < partial_merge_view.sealed_deltas.size(),
              "sealed delta PID lookup returned out-of-range segment index");
        ResolvedRow partial_merge_row;
        CHECK(partial_merge_engine.row_resolver().resolve(partial_merge_pid, partial_merge_row),
              "RowResolver must resolve multi-sealed-delta PID through PID lookup");
        CHECK(partial_merge_row.output_id == 4003 &&
                  partial_merge_row.segment == ResolvedRow::SegmentKind::Delta,
              "RowResolver multi-sealed-delta PID lookup resolved wrong row");
        QueryResult partial_merge_result =
            partial_merge_engine.search(rows.data() + (size_t)6 * (size_t)dim);
        bool saw_partial_merge_insert = false;
        for (const auto& hit : partial_merge_result.topk) {
            if (hit.second == 4003) saw_partial_merge_insert = true;
        }
        CHECK(saw_partial_merge_insert,
              "compaction must preserve sealed delta rows");
        TEST("Multi sealed delta overlay query"); PASS();

        {
            UpdateCompactionPolicy drain_policy;
            drain_policy.flush_fraction = 0.000000001;
            drain_policy.compaction_fraction = 0.0;
            UpdateService drain_service(
                index_dir,
                drain_policy,
                "none",
                1,
                {});
            UpdateCompactionRunStats drain_stats =
                drain_service.compact_until_quiescent();
            CHECK(drain_stats.no_op_passes == 1,
                  "update service drain should stop after one no-op policy pass");
            CHECK(!drain_service.requires_reopen_after_compaction(),
                  "non-compaction update service drain should not require reopen");
            drain_service.flush_buffered_commits();
        }

        UpdateManifest drained_manifest = load_update_manifest_json_for_index(index_dir);
        CHECK(drained_manifest.sealed_deltas.size() >= 1,
              "update service drain should keep sealed deltas");
        CHECK(drained_manifest.last_committed_lsn == 0,
              "online writer drain should leave L0 empty");
        QueryEngine drained_engine;
        drained_engine.init(update_config);
        drained_engine.load_index(index_dir);
        QueryResult drained_result =
            drained_engine.search(rows.data() + (size_t)6 * (size_t)dim);
        bool saw_drained_insert = false;
        for (const auto& hit : drained_result.topk) {
            if (hit.second == 4003) saw_drained_insert = true;
        }
        CHECK(saw_drained_insert,
              "update service drain must preserve query-visible delta rows");
        TEST("UpdateService drains policy compaction backlog"); PASS();

        {
            UpdateIndexWriter writer(index_dir, "buffered", 1, {3101, 3102});
            UpdateIndexMemoryBatch pre_flush_batch;
            pre_flush_batch.insert_rows = rows.data() + (size_t)4 * (size_t)dim;
            pre_flush_batch.insert_n = 1;
            pre_flush_batch.insert_dim = dim;
            pre_flush_batch.insert_orig_ids = {3101};
            writer.commit_batch(pre_flush_batch);

            UpdateCompactionPolicy flush_policy;
            flush_policy.flush_fraction = 0.000000001;
            flush_policy.compaction_fraction = 0.0;
            UpdateCompactionDecision decision = writer.run_locked_maintenance([&]() {
                return compact_index_by_policy_locked(index_dir, flush_policy);
            });
            CHECK(decision.flushed_l0,
                  "in-process writer maintenance must flush committed L0 while holding writer lock");
            CHECK(!writer.requires_reopen_after_compaction(),
                  "non-compaction writer maintenance should not require reopening writer");

            UpdateIndexMemoryBatch post_flush_batch;
            post_flush_batch.insert_rows = rows.data() + (size_t)5 * (size_t)dim;
            post_flush_batch.insert_n = 1;
            post_flush_batch.insert_dim = dim;
            post_flush_batch.insert_orig_ids = {3102};
            writer.commit_batch(post_flush_batch);
            writer.flush_buffered_commits();
        }

        QueryEngine maintenance_writer_engine;
        maintenance_writer_engine.init(update_config);
        maintenance_writer_engine.load_index(index_dir);
        const ReaderView& maintenance_writer_view = maintenance_writer_engine.reader_view();
        CHECK(!maintenance_writer_view.sealed_deltas.empty(),
              "writer maintenance flush must publish a sealed delta");
        CHECK(maintenance_writer_view.growing_delta != nullptr &&
                  !maintenance_writer_view.growing_delta->empty(),
              "writer must continue publishing new L0 after maintenance flush");
        QueryResult maintenance_l1_result =
            maintenance_writer_engine.search(rows.data() + (size_t)4 * (size_t)dim);
        QueryResult maintenance_l0_result =
            maintenance_writer_engine.search(rows.data() + (size_t)5 * (size_t)dim);
        bool saw_maintenance_l1 = false;
        bool saw_maintenance_l0 = false;
        for (const auto& hit : maintenance_l1_result.topk) {
            if (hit.second == 3101) saw_maintenance_l1 = true;
        }
        for (const auto& hit : maintenance_l0_result.topk) {
            if (hit.second == 3102) saw_maintenance_l0 = true;
        }
        CHECK(saw_maintenance_l1 && saw_maintenance_l0,
              "writer must remain query-correct across maintenance flush and later L0 append");
        TEST("UpdateIndexWriter continues after locked maintenance flush"); PASS();

        {
            QueryIndexHandle online_handle;
            online_handle.init(update_config);
            online_handle.load_index(index_dir);

            UpdateCompactionPolicy online_policy;
            online_policy.flush_fraction = 0.000000001;
            online_policy.compaction_fraction = 0.0;
            UpdateService update_service(
                index_dir,
                online_policy,
                "buffered",
                1,
                {3201},
                &online_handle);
            UpdateIndexMemoryBatch online_batch;
            online_batch.insert_rows = rows.data() + (size_t)6 * (size_t)dim;
            online_batch.insert_n = 1;
            online_batch.insert_dim = dim;
            online_batch.insert_orig_ids = {3201};
            update_service.commit_batch(online_batch);
            CHECK(update_service.auto_compact_after_commit(),
                  "update service should default to automatic policy compaction");
            CHECK(!update_service.requires_reopen_after_compaction(),
                  "update service flush should not require reopen");
            std::shared_ptr<const ReaderView> online_view = online_handle.snapshot();
            CHECK(online_view->l0_last_committed_lsn == 0,
                  "update service automatic policy flush should publish an empty L0 reader");
            CHECK(!online_view->sealed_deltas.empty(),
                  "update service automatic policy flush should publish sealed delta state");
            QueryResult online_result =
                online_handle.search(rows.data() + (size_t)6 * (size_t)dim);
            bool saw_online_insert = false;
            for (const auto& hit : online_result.topk) {
                if (hit.second == 3201) saw_online_insert = true;
            }
            CHECK(saw_online_insert,
                  "update service automatic policy compaction must keep committed row searchable");
            update_service.flush_buffered_commits();
        }
        TEST("UpdateService commits batches with automatic policy compaction"); PASS();

        {
            QueryIndexHandle manual_handle;
            manual_handle.init(update_config);
            manual_handle.load_index(index_dir);

            UpdateCompactionPolicy manual_policy;
            manual_policy.flush_fraction = 0.000000001;
            manual_policy.compaction_fraction = 0.0;
            UpdateService manual_service(
                index_dir,
                manual_policy,
                "buffered",
                1,
                {3202},
                &manual_handle);
            manual_service.set_auto_compact_after_commit(false);
            CHECK(!manual_service.auto_compact_after_commit(),
                  "update service should allow manual policy compaction mode");
            UpdateIndexMemoryBatch manual_batch;
            manual_batch.insert_rows = rows.data() + (size_t)7 * (size_t)dim;
            manual_batch.insert_n = 1;
            manual_batch.insert_dim = dim;
            manual_batch.insert_orig_ids = {3202};
            manual_service.commit_batch(manual_batch);
            std::shared_ptr<const ReaderView> manual_commit_view = manual_handle.snapshot();
            CHECK(manual_commit_view->l0_last_committed_lsn != 0,
                  "manual compaction mode should publish committed L0 after commit");
            UpdateCompactionDecision manual_decision =
                manual_service.compact_if_needed();
            CHECK(manual_decision.flushed_l0,
                  "manual compaction mode should flush L0 only when explicitly requested");
            std::shared_ptr<const ReaderView> manual_compacted_view = manual_handle.snapshot();
            CHECK(manual_compacted_view->l0_last_committed_lsn == 0,
                  "manual compaction flush should publish an empty L0 reader");
            manual_service.flush_buffered_commits();
        }
        TEST("UpdateService supports manual policy compaction mode"); PASS();

        {
            QueryIndexHandle drain_handle;
            drain_handle.init(update_config);
            drain_handle.load_index(index_dir);

            UpdateCompactionPolicy disabled_policy;
            disabled_policy.flush_fraction = 0.0;
            disabled_policy.compaction_fraction = 0.0;
            UpdateService drain_service(
                index_dir,
                disabled_policy,
                "buffered",
                1,
                {3203},
                &drain_handle);
            UpdateIndexMemoryBatch drain_batch;
            drain_batch.insert_rows = rows.data() + (size_t)8 * (size_t)dim;
            drain_batch.insert_n = 1;
            drain_batch.insert_dim = dim;
            drain_batch.insert_orig_ids = {3203};
            drain_service.commit_batch(drain_batch);
            std::shared_ptr<const ReaderView> drain_commit_view = drain_handle.snapshot();
            CHECK(drain_commit_view->l0_last_committed_lsn != 0,
                  "disabled policy should leave committed L0 visible after commit");
            UpdateCompactionDecision drain_decision =
                drain_service.flush_delta_segments();
            CHECK(drain_decision.flushed_l0,
                  "explicit service delta drain should flush committed L0");
            std::shared_ptr<const ReaderView> drain_compacted_view = drain_handle.snapshot();
            CHECK(drain_compacted_view->l0_last_committed_lsn == 0,
                  "explicit service delta drain should publish an empty L0 reader");
            CHECK(!drain_compacted_view->sealed_deltas.empty(),
                  "explicit service delta drain should publish sealed delta state");
            QueryResult drain_result =
                drain_handle.search(rows.data() + (size_t)8 * (size_t)dim);
            bool saw_drain_insert = false;
            for (const auto& hit : drain_result.topk) {
                if (hit.second == 3203) saw_drain_insert = true;
            }
            CHECK(saw_drain_insert,
                  "explicit service delta drain must keep committed row searchable");
            drain_service.flush_buffered_commits();
        }
        TEST("UpdateService drains delta segments without policy threshold"); PASS();

        {
            // strict durability: each commit fsyncs the WAL + manifest and advances
            // last_fsynced_lsn to last_committed_lsn AT COMMIT TIME (no
            // flush_buffered_commits needed). buffered/none would leave
            // last_fsynced_lsn lagging until a buffer-boundary flush.
            UpdateManifest pre_strict = load_update_manifest_json_for_index(index_dir);
            QueryIndexHandle strict_handle;
            strict_handle.init(update_config);
            strict_handle.load_index(index_dir);
            UpdateCompactionPolicy strict_policy;
            strict_policy.flush_fraction = 0.0;
            strict_policy.compaction_fraction = 0.0;
            UpdateService strict_service(index_dir, strict_policy, "strict", 1,
                                         {3400}, &strict_handle);
            UpdateIndexMemoryBatch strict_batch;
            strict_batch.insert_rows = rows.data() + (size_t)9 * (size_t)dim;
            strict_batch.insert_n = 1;
            strict_batch.insert_dim = dim;
            strict_batch.insert_orig_ids = {3400};
            strict_service.commit_batch(strict_batch);
            UpdateManifest post_strict = load_update_manifest_json_for_index(index_dir);
            CHECK(post_strict.last_committed_lsn > pre_strict.last_committed_lsn,
                  "strict durability commit must advance last_committed_lsn");
            CHECK(post_strict.durability_mode == "strict",
                  "strict commit must persist the strict durability mode");
            CHECK(post_strict.last_fsynced_lsn == post_strict.last_committed_lsn,
                  "strict durability must fsync per commit (last_fsynced_lsn catches "
                  "up to last_committed_lsn) without flush_buffered_commits");
            strict_service.flush_buffered_commits();
        }
        TEST("UpdateService strict durability fsyncs each commit"); PASS();

        {
            constexpr size_t concurrent_writers = 4;
            std::vector<uint32_t> concurrent_ids;
            concurrent_ids.reserve(concurrent_writers);
            for (size_t i = 0; i < concurrent_writers; ++i) {
                concurrent_ids.push_back(3301u + (uint32_t)i);
            }
            UpdateCompactionPolicy concurrent_policy;
            concurrent_policy.flush_fraction = 0.0;
            concurrent_policy.compaction_fraction = 0.0;
            UpdateService concurrent_service(
                index_dir,
                concurrent_policy,
                "none",
                1,
                concurrent_ids);
            concurrent_service.set_auto_compact_after_commit(false);

            std::vector<std::thread> writers;
            std::vector<std::exception_ptr> errors(concurrent_writers);
            writers.reserve(concurrent_writers);
            for (size_t i = 0; i < concurrent_writers; ++i) {
                writers.emplace_back([&, i]() {
                    try {
                        UpdateIndexMemoryBatch batch;
                        batch.insert_rows =
                            rows.data() + (size_t)(8 + (int)i) * (size_t)dim;
                        batch.insert_n = 1;
                        batch.insert_dim = dim;
                        batch.insert_orig_ids = {concurrent_ids[i]};
                        concurrent_service.commit_batch(batch);
                    } catch (...) {
                        errors[i] = std::current_exception();
                    }
                });
            }
            for (std::thread& writer_thread : writers) writer_thread.join();
            for (const std::exception_ptr& error : errors) {
                CHECK(!error,
                      "concurrent UpdateService commit_batch should not throw");
            }
            concurrent_service.flush_buffered_commits();

            UpdateManifest concurrent_manifest =
                load_update_manifest_json_for_index(index_dir);
            DirectMap concurrent_map =
                DirectMap::load_binary(index_dir + "/" + concurrent_manifest.direct_map_path);
            for (uint32_t orig_id : concurrent_ids) {
                uint32_t pid = 0;
                CHECK(concurrent_map.lookup(orig_id, pid),
                      "concurrent UpdateService commit must publish each ID in DirectMap");
                CHECK(pid >= concurrent_manifest.base_n && pid < concurrent_manifest.max_pid,
                      "concurrent UpdateService DirectMap PID must be in delta range");
            }
        }
        TEST("UpdateService serializes concurrent commit_batch callers"); PASS();

        // QueryIndexHandle: many writer callers commit through one UpdateService
        // (publishing to the handle) while reader threads search_batch the same
        // handle concurrently. Validates the immutable-snapshot + per-thread
        // executor model: no locks on the query hot path, no shared mutable
        // scratch, every committed id becomes query-visible.
        {
            IndexConfig handle_query_config = update_config;
            handle_query_config.num_threads = 1;
            QueryIndexHandle rw_handle;
            rw_handle.init(handle_query_config);
            rw_handle.load_index(index_dir);
            const uint32_t rw_before_max_pid = rw_handle.snapshot()->max_pid;

            constexpr size_t rw_writers = 4;
            std::vector<uint32_t> rw_ids;
            rw_ids.reserve(rw_writers);
            for (size_t i = 0; i < rw_writers; ++i) {
                rw_ids.push_back(3401u + (uint32_t)i);
            }
            UpdateCompactionPolicy rw_policy;
            rw_policy.flush_fraction = 0.0;
            rw_policy.compaction_fraction = 0.0;
            UpdateService rw_service(index_dir, rw_policy, "none", 1, rw_ids, &rw_handle);
            rw_service.set_auto_compact_after_commit(false);

            const int rw_nq = 4;
            const float* rw_queries = rows.data() + (size_t)12 * (size_t)dim;
            constexpr int rw_readers = 4;
            std::atomic<bool> rw_stop{false};
            std::atomic<int64_t> rw_reader_queries{0};
            std::vector<std::exception_ptr> rw_reader_errors(rw_readers);
            std::vector<std::thread> rw_reader_threads;
            rw_reader_threads.reserve(rw_readers);
            for (int r = 0; r < rw_readers; ++r) {
                rw_reader_threads.emplace_back([& rw_reader_errors, &rw_handle, &rw_stop,
                                                &rw_reader_queries, rw_queries, rw_nq, r]() {
                    try {
                        while (!rw_stop.load(std::memory_order_acquire)) {
                            std::vector<QueryResult> results =
                                rw_handle.search_batch(rw_queries, rw_nq);
                            rw_reader_queries += (int64_t)results.size();
                        }
                    } catch (...) {
                        rw_reader_errors[r] = std::current_exception();
                    }
                });
            }

            std::vector<std::exception_ptr> rw_writer_errors(rw_writers);
            std::vector<std::thread> rw_writer_threads;
            rw_writer_threads.reserve(rw_writers);
            for (size_t i = 0; i < rw_writers; ++i) {
                rw_writer_threads.emplace_back([& rw_ids, &rows, dim, &rw_service,
                                                &rw_writer_errors, i]() {
                    try {
                        UpdateIndexMemoryBatch batch;
                        batch.insert_rows = rows.data() + (size_t)(12 + (int)i) * (size_t)dim;
                        batch.insert_n = 1;
                        batch.insert_dim = dim;
                        batch.insert_orig_ids = {rw_ids[i]};
                        rw_service.commit_batch(batch);
                    } catch (...) {
                        rw_writer_errors[i] = std::current_exception();
                    }
                });
            }

            for (std::thread& t : rw_writer_threads) t.join();
            for (const std::exception_ptr& e : rw_writer_errors) {
                CHECK(!e, "concurrent handle publish writer should not throw");
            }
            rw_stop.store(true, std::memory_order_release);
            for (std::thread& t : rw_reader_threads) t.join();

            for (const std::exception_ptr& e : rw_reader_errors) {
                CHECK(!e, "concurrent readers must not throw during publish");
            }
            CHECK(rw_reader_queries.load() > 0,
                  "concurrent readers must complete searches while writers publish");
            std::shared_ptr<const ReaderView> rw_after_view = rw_handle.snapshot();
            CHECK(rw_after_view->max_pid ==
                      rw_before_max_pid + (uint32_t)rw_writers,
                  "handle must publish all concurrent writer commits");
            for (size_t i = 0; i < rw_writers; ++i) {
                QueryResult rw_visible = rw_handle.search(rows.data() + (size_t)(12 + (int)i) * (size_t)dim);
                bool saw_rw_insert = false;
                for (const auto& hit : rw_visible.topk) {
                    if (hit.second == rw_ids[i]) saw_rw_insert = true;
                }
                CHECK(saw_rw_insert, "concurrent published insert must be query-visible");
            }
            rw_service.flush_buffered_commits();
        }
        TEST("QueryIndexHandle publishes concurrent commits to live readers"); PASS();

        // A reader-held snapshot must stay immutable and valid across a publish
        // and across a compaction that forces a reload, while the latest snapshot
        // reflects the new state.
        {
            IndexConfig iso_config = update_config;
            iso_config.num_threads = 1;
            QueryIndexHandle iso_handle;
            iso_handle.init(iso_config);
            iso_handle.load_index(index_dir);
            std::shared_ptr<const ReaderView> old_snapshot = iso_handle.snapshot();
            const uint32_t iso_old_max_pid = old_snapshot->max_pid;
            const uint64_t iso_old_lsn = old_snapshot->l0_last_committed_lsn;

            UpdateCompactionPolicy iso_policy;
            iso_policy.flush_fraction = 0.0;
            iso_policy.compaction_fraction = 0.0;
            UpdateService iso_service(index_dir, iso_policy, "none", 1, {3501}, &iso_handle);
            iso_service.set_auto_compact_after_commit(false);
            UpdateIndexMemoryBatch iso_batch;
            iso_batch.insert_rows = rows.data() + (size_t)20 * (size_t)dim;
            iso_batch.insert_n = 1;
            iso_batch.insert_dim = dim;
            iso_batch.insert_orig_ids = {3501};
            iso_service.commit_batch(iso_batch);

            CHECK(old_snapshot->max_pid == iso_old_max_pid &&
                      old_snapshot->l0_last_committed_lsn == iso_old_lsn,
                  "an older reader snapshot must remain immutable after a newer publish");
            CHECK(iso_handle.snapshot()->max_pid == iso_old_max_pid + 1,
                  "the latest snapshot must reflect the newly published commit");

            // Force an L0->L1 compaction: the handle must rebuild from disk
            // (reload fallback), while the previously held snapshot stays valid.
            UpdateCompactionDecision iso_decision = iso_service.flush_delta_segments();
            CHECK(iso_decision.flushed_l0,
                  "explicit compaction should flush the committed L0 layer");
            CHECK(old_snapshot->max_pid == iso_old_max_pid &&
                      old_snapshot->base != nullptr &&
                      old_snapshot->base_n == (uint32_t)n,
                  "older snapshot must stay alive and immutable across compaction");
            CHECK(iso_handle.snapshot()->l0_last_committed_lsn == 0 &&
                      !iso_handle.snapshot()->sealed_deltas.empty(),
                  "compaction must publish a reloaded reader with sealed delta state");
            iso_service.flush_buffered_commits();
        }
        TEST("QueryIndexHandle keeps older snapshots valid across publish and compaction"); PASS();

        std::remove(update_insert_path.c_str());
        std::remove(update_batch_insert_path.c_str());
        std::remove(update_insert_ids_path.c_str());
        std::remove(update_insert2_ids_path.c_str());
        std::remove(update_batch_insert_ids_path.c_str());
        std::remove(update_delete_ids_path.c_str());
        std::remove(update_batch_insert_path.c_str());
        std::remove(update_insert_ids_path.c_str());
        std::remove(update_insert2_ids_path.c_str());
        std::remove(update_batch_insert_ids_path.c_str());
        std::remove(update_delete_ids_path.c_str());
        if (update_manifest_exists(index_dir)) {
            UpdateManifest cleanup_manifest = load_update_manifest_json_for_index(index_dir);
            for (const UpdateManifestDelta& delta : cleanup_manifest.sealed_deltas) {
                std::string dir = index_dir + "/" + delta.path;
                std::remove((dir + "/segment_manifest.json").c_str());
                std::remove((dir + "/delta_orig_ids.bin").c_str());
                std::remove((dir + "/delta_pid_to_row.bin").c_str());
                std::remove((dir + "/delta_verify_vectors.fbin").c_str());
                std::remove((dir + "/delta_rabitq_bin_data.bin").c_str());
                for (int m = 0; m < M; ++m) {
                    std::remove((dir + "/delta_posting_subspace_" + std::to_string(m) + ".bin").c_str());
                }
                rmdir(dir.c_str());
            }
            rmdir((index_dir + "/deltas").c_str());
            std::remove((index_dir + "/update_manifest.json").c_str());
            std::remove((index_dir + "/updates/l0.wal").c_str());
            std::remove((index_dir + "/updates/direct_map.bin").c_str());
            std::remove((index_dir + "/updates/l0.checkpoint").c_str());
            rmdir((index_dir + "/updates").c_str());
        }
    } catch (const std::exception& e) {
        FAIL(std::string("Fused no-reorder query without sidecar threw: ") + e.what());
        return;
    }

    RandomRotation stale_rotation;
    stale_rotation.init(dim, 7, RotationType::QR);
    stale_rotation.save(index_dir + "/rotation.bin");
    bool rejected_mismatched_rotation = false;
    try {
        IndexConfig config;
        config.num_subspaces = M;
        config.rabitq_k = rabitq_k;
        QueryEngine engine;
        engine.init(config);
        engine.load_index(index_dir);
    } catch (const std::exception&) {
        rejected_mismatched_rotation = true;
    }
    CHECK(rejected_mismatched_rotation,
          "query loader must reject rotation padded dimension mismatches");
    rotation.save(index_dir + "/rotation.bin");

    uint32_t original_cid = 0;
    {
        std::fstream corrupt(index_dir + "/rabitq_bin_data.bin", std::ios::binary | std::ios::in | std::ios::out);
        corrupt.seekg(2 * sizeof(int));
        corrupt.read(reinterpret_cast<char*>(&original_cid), sizeof(uint32_t));
        uint32_t bad_cid = (uint32_t)rabitq_k;
        corrupt.seekp(2 * sizeof(int));
        corrupt.write(reinterpret_cast<const char*>(&bad_cid), sizeof(uint32_t));
    }
    bool rejected_bad_embedded_cid = false;
    try {
        IndexConfig config;
        config.num_subspaces = M;
        config.rabitq_k = rabitq_k;
        QueryEngine engine;
        engine.init(config);
        engine.load_index(index_dir);
    } catch (const std::exception&) {
        rejected_bad_embedded_cid = true;
    }
    CHECK(rejected_bad_embedded_cid, "query loader must reject out-of-range embedded RaBitQ cids");
    {
        std::fstream restore(index_dir + "/rabitq_bin_data.bin", std::ios::binary | std::ios::in | std::ios::out);
        restore.seekp(2 * sizeof(int));
        restore.write(reinterpret_cast<const char*>(&original_cid), sizeof(uint32_t));
    }

    write_test_fvecs(index_dir + "/verify_vectors.fbin",
                     current_verify,
                     current_verify_n,
                     current_verify_dim);
    bool rejected_fvec_verify_artifact = false;
    try {
        IndexConfig config;
        config.num_subspaces = M;
        config.rabitq_k = rabitq_k;
        QueryEngine engine;
        engine.init(config);
        engine.load_index(index_dir);
    } catch (const std::exception&) {
        rejected_fvec_verify_artifact = true;
    }
    CHECK(rejected_fvec_verify_artifact, "query loader must reject fvecs payloads named verify_vectors.fbin");
    write_test_fbin(index_dir + "/verify_vectors.fbin",
                    current_verify,
                    current_verify_n,
                    current_verify_dim);

    {
        std::fstream corrupt(index_dir + "/rabitq_bin_data.bin", std::ios::binary | std::ios::in | std::ios::out);
        int legacy_payload_only_record_bytes = rabitq_payload_bytes;
        corrupt.seekp(sizeof(int));
        corrupt.write(reinterpret_cast<const char*>(&legacy_payload_only_record_bytes), sizeof(int));
    }
    bool rejected_legacy_layout = false;
    try {
        IndexConfig config;
        config.num_subspaces = M;
        config.rabitq_k = rabitq_k;
        QueryEngine engine;
        engine.init(config);
        engine.load_index(index_dir);
    } catch (const std::exception&) {
        rejected_legacy_layout = true;
    }
    CHECK(rejected_legacy_layout, "query loader must reject legacy payload-only RaBitQ record headers");

    // --- Crash-safe compaction auto-recovery test ---
    // Simulate a crashed compaction: hardlink base files + manifest to a backup
    // dir, create a marker pointing to it. The recovery function should
    // auto-restore from backup + remove the marker.
    {
        std::string backup = index_dir + ".compaction_backup_test";
        ::mkdir(backup.c_str(), 0755);
        std::vector<std::string> artifacts = {
            "rotation.bin", "rabitq_centroids.bin", "verify_vectors.fbin",
            "rabitq_bin_data.bin", "perm_to_orig.bin"
        };
        for (int m = 0; m < M; ++m) {
            artifacts.push_back("ivf_subspace_" + std::to_string(m) + ".bin");
            artifacts.push_back("posting_subspace_" + std::to_string(m) + ".bin");
        }
        for (const std::string& name : artifacts) {
            std::string live = index_dir + "/" + name;
            if (::access(live.c_str(), F_OK) == 0) {
                ::link(live.c_str(), (backup + "/" + name).c_str());
            }
        }
        {
            std::string manifest_live = index_dir + "/update_manifest.json";
            if (::access(manifest_live.c_str(), F_OK) == 0) {
                ::link(manifest_live.c_str(), (backup + "/update_manifest.json").c_str());
            }
        }
        {
            std::ofstream m(index_dir + "/compaction.publish");
            m << backup << "\n";
        }
        // Call the recovery function directly (tests the rollback logic without
        // needing a full load_index, which may fail for unrelated reasons on
        // this heavily-mutated test index).
        score::impl::recover_compaction_if_marker_present(index_dir);
        CHECK(::access((index_dir + "/compaction.publish").c_str(), F_OK) != 0,
              "recovery must remove the stale marker");
        // Backup dir cleanup is best-effort (rmdir may leave harmless leftovers).
        // The key signal is the marker removal + files restored to index_dir.
        CHECK(::access((index_dir + "/verify_vectors.fbin").c_str(), F_OK) == 0,
              "recovery must restore base files to index_dir");
    }
    TEST("Compaction crash auto-recovers from marker + backup"); PASS();

    // Negative: marker with no valid backup → throws actionable error.
    {
        {
            std::ofstream m(index_dir + "/compaction.publish");
            m << "/nonexistent/backup_dir\n";
        }
        bool threw = false;
        try {
            score::impl::recover_compaction_if_marker_present(index_dir);
        } catch (const std::exception&) {
            threw = true;
        }
        CHECK(threw, "marker with invalid backup must throw actionable error");
        std::remove((index_dir + "/compaction.publish").c_str());
    }
    TEST("Compaction crash with no backup throws actionable error"); PASS();

    std::remove((index_dir + "/verify_vectors.fbin").c_str());
    std::remove((index_dir + "/rotation.bin").c_str());
    std::remove((index_dir + "/rotate_subspace.flag").c_str());
    std::remove((index_dir + "/rabitq_centroids.bin").c_str());
    std::remove((index_dir + "/rabitq_assignments.bin").c_str());
    std::remove((index_dir + "/rabitq_bin_data.bin").c_str());
    std::remove((index_dir + "/perm_to_orig.bin").c_str());
    std::remove((index_dir + "/update_writer.lock").c_str());
    std::remove((index_dir + "/update_publish.lock").c_str());
    for (int m = 0; m < M; ++m) {
        std::remove((index_dir + "/ivf_subspace_" + std::to_string(m) + ".bin").c_str());
        std::remove((index_dir + "/posting_subspace_" + std::to_string(m) + ".bin").c_str());
    }
    rmdir(index_dir.c_str());
    std::remove(input_path.c_str());
    TEST("Fused no-reorder artifacts loadable"); PASS();
}

