#include "test_macros.h"

void test_reordered_finalized_artifacts_are_query_loadable() {
    std::cout << "[PID reorder finalized query-load]" << std::endl;

    const std::string suffix = std::to_string(getpid());
    const std::string input_path = "/tmp/score_reorder_query_input_" + suffix + ".fbin";
    const std::string base_dir = "/tmp/score_reorder_query_base_" + suffix;
    const std::string final_dir = "/tmp/score_reorder_query_final_" + suffix;
    const int n = 24;
    const int dim = 96;
    const int M = 4;
    const int ivf_k = 3;
    const int rabitq_k = 4;

    std::vector<float> rows((size_t)n * dim);
    for (int i = 0; i < n; ++i) {
        for (int d = 0; d < dim; ++d) {
            rows[(size_t)i * dim + d] = (float)(((i + 19) * 17 + d * 7) % 131) * 0.01f;
        }
    }
    write_test_fbin(input_path, rows, n, dim);
    mkdir(base_dir.c_str(), 0755);
    mkdir(final_dir.c_str(), 0755);

    auto cleanup = [&]() {
        const std::vector<std::string> base_names = streaming_no_reorder_artifact_names(M);
        for (const std::string& name : base_names) std::remove((base_dir + "/" + name).c_str());
        std::remove((base_dir + "/subspace_rotation.bin").c_str());
        std::remove((base_dir + "/perm_to_orig.bin").c_str());
        std::remove((base_dir + "/posting_assignments.manifest").c_str());
        rmdir(base_dir.c_str());
        std::remove(input_path.c_str());
        std::remove((final_dir + "/rotation.bin").c_str());
        std::remove((final_dir + "/rotate_subspace.flag").c_str());
        std::remove((final_dir + "/subspace_rotation.bin").c_str());
        std::remove((final_dir + "/rabitq_centroids.bin").c_str());
        std::remove((final_dir + "/verify_vectors.fbin").c_str());
        std::remove((final_dir + "/rabitq_assignments.bin").c_str());
        std::remove((final_dir + "/rabitq_bin_data.bin").c_str());
        std::remove((final_dir + "/perm_to_orig.bin").c_str());
        std::remove((final_dir + "/perm_to_orig.raw.tmp").c_str());
        std::remove((final_dir + "/perm_to_orig.scratch.tmp").c_str());
        std::remove((final_dir + "/orig_to_new.reorder.tmp").c_str());
        for (int m = 0; m < M; ++m) {
            std::remove((base_dir + "/posting_assign_subspace_" + std::to_string(m) + ".reorder.tmp").c_str());
            std::remove((base_dir + "/posting_assign_subspace_" + std::to_string(m) + ".assign.bin").c_str());
            std::remove((final_dir + "/posting_assign_subspace_" + std::to_string(m) + ".reorder.tmp").c_str());
            std::remove((final_dir + "/ivf_subspace_" + std::to_string(m) + ".bin").c_str());
            std::remove((final_dir + "/posting_subspace_" + std::to_string(m) + ".bin").c_str());
        }
        rmdir(final_dir.c_str());
    };

    try {
        RandomRotation rotation;
        rotation.init(dim, 42, RotationType::FhtKac);
        save_rotation_exclusive(rotation, base_dir + "/rotation.bin");
        write_rotate_subspace_flag_exclusive(base_dir);
        copy_test_file(base_dir + "/rotation.bin", base_dir + "/subspace_rotation.bin");
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
        write_streaming_no_reorder_fused_artifacts(input_path, base_dir, rotation,
                                                   rabitq_options, ivf_options);

        IndexConfig reorder_config;
        reorder_config.num_subspaces = M;
        reorder_config.rabitq_k = rabitq_k;
        reorder_config.pid_reorder_key_mode = 1;
        reorder_config.pid_reorder_cell_key_depth = 2;
        impl::PosthocReorderIndexOptions posthoc_options;
        posthoc_options.source_index_dir = base_dir;
        posthoc_options.output_index_dir = final_dir;
        posthoc_options.n = n;
        posthoc_options.M = M;
        posthoc_options.config = reorder_config;
        posthoc_options.io_rows = 5;
        posthoc_options.bucket_buffer_rows = 2;
        impl::posthoc_reorder_no_reorder_index(posthoc_options);

        QueryEngine engine;
        engine.init(reorder_config);
        engine.load_index(final_dir);
        CHECK(engine.reader_view().base_n == (uint32_t)n,
              "reordered reader view base_n mismatch");
        CHECK(!engine.perm_to_orig().empty(),
              "reordered reader must load perm_to_orig");
        RowResolver resolver = engine.row_resolver();
        ResolvedRow resolved;
        CHECK(resolver.resolve(0, resolved),
              "reordered RowResolver must resolve base pid 0");
        CHECK(resolved.output_id == engine.perm_to_orig()[0],
              "reordered RowResolver output id must use perm_to_orig");
        std::vector<float> query = rows;
        QueryResult result = engine.search(query.data());
        CHECK(!result.topk.empty(), "final query-load index with preserved subspace rotation must search");
    } catch (const std::exception& e) {
        cleanup();
        FAIL(std::string("Finalized reordered query-load threw: ") + e.what());
        return;
    }

    CHECK(access((final_dir + "/perm_to_orig.bin").c_str(), F_OK) == 0,
          "final query-load index must expose perm_to_orig.bin");
    CHECK(access((final_dir + "/orig_to_new.reorder.tmp").c_str(), F_OK) != 0,
          "query-load index must not leave inverse scratch");
    CHECK(access((final_dir + "/perm_to_orig.raw.tmp").c_str(), F_OK) != 0,
          "query-load index must not leave raw perm scratch");
    CHECK(access((final_dir + "/subspace_rotation.bin").c_str(), F_OK) == 0,
          "posthoc reorder must preserve external subspace rotation artifact when present");

    {
        std::vector<uint32_t> final_assignments((size_t)n);
        std::ifstream assign_in(final_dir + "/rabitq_assignments.bin", std::ios::binary);
        int assign_n = 0;
        assign_in.read(reinterpret_cast<char*>(&assign_n), sizeof(int));
        CHECK(assign_n == n, "final reordered assignment n mismatch");
        assign_in.read(reinterpret_cast<char*>(final_assignments.data()), (size_t)n * sizeof(uint32_t));

        std::ifstream bin_in(final_dir + "/rabitq_bin_data.bin", std::ios::binary);
        int bin_n = 0, record_bytes = 0;
        bin_in.read(reinterpret_cast<char*>(&bin_n), sizeof(int));
        bin_in.read(reinterpret_cast<char*>(&record_bytes), sizeof(int));
        CHECK(bin_n == n, "final reordered bin n mismatch");
        const int payload_bytes = (int)rabitqlib::BinDataMap<float>::data_bytes(
            RandomRotation::padding_requirement(dim, RotationType::FhtKac));
        CHECK(record_bytes == (int)(sizeof(uint32_t) + (size_t)payload_bytes),
              "final reordered embedded record bytes mismatch");
        for (int i = 0; i < n; ++i) {
            uint32_t embedded_cid = std::numeric_limits<uint32_t>::max();
            bin_in.read(reinterpret_cast<char*>(&embedded_cid), sizeof(uint32_t));
            CHECK(embedded_cid == final_assignments[(size_t)i],
                  "final reordered embedded cid mismatch");
            bin_in.seekg(payload_bytes, std::ios::cur);
        }
    }

    cleanup();
    TEST("Finalized reordered artifacts query-load"); PASS();
}

void test_posthoc_reorder_uses_preserved_assignments() {
    std::cout << "[PID reorder posthoc preserved assignments]" << std::endl;

    const std::string suffix = std::to_string(getpid());
    const std::string input_path = "/tmp/score_preserved_assign_input_" + suffix + ".fbin";
    const std::string base_dir = "/tmp/score_preserved_assign_base_" + suffix;
    const std::string final_dir = "/tmp/score_preserved_assign_final_" + suffix;
    const int n = 20;
    const int dim = 64;
    const int M = 4;
    const int ivf_k = 3;
    const int rabitq_k = 4;

    std::vector<float> rows((size_t)n * dim);
    for (int i = 0; i < n; ++i) {
        for (int d = 0; d < dim; ++d) {
            rows[(size_t)i * dim + d] = (float)(((i + 5) * 13 + d * 11) % 97) * 0.02f;
        }
    }
    write_test_fbin(input_path, rows, n, dim);
    mkdir(base_dir.c_str(), 0755);
    mkdir(final_dir.c_str(), 0755);

    auto cleanup = [&]() {
        const std::vector<std::string> base_names = streaming_no_reorder_artifact_names(M);
        for (const std::string& name : base_names) std::remove((base_dir + "/" + name).c_str());
        rmdir(base_dir.c_str());
        std::remove(input_path.c_str());
        std::remove((final_dir + "/rotation.bin").c_str());
        std::remove((final_dir + "/rotate_subspace.flag").c_str());
        std::remove((final_dir + "/rabitq_centroids.bin").c_str());
        std::remove((final_dir + "/verify_vectors.fbin").c_str());
        std::remove((final_dir + "/rabitq_assignments.bin").c_str());
        std::remove((final_dir + "/rabitq_bin_data.bin").c_str());
        std::remove((final_dir + "/perm_to_orig.bin").c_str());
        std::remove((final_dir + "/perm_to_orig.raw.tmp").c_str());
        std::remove((final_dir + "/perm_to_orig.scratch.tmp").c_str());
        std::remove((final_dir + "/orig_to_new.reorder.tmp").c_str());
        for (int m = 0; m < M; ++m) {
            std::remove((final_dir + "/posting_assign_subspace_" + std::to_string(m) + ".reorder.tmp").c_str());
            std::remove((final_dir + "/ivf_subspace_" + std::to_string(m) + ".bin").c_str());
            std::remove((final_dir + "/posting_subspace_" + std::to_string(m) + ".bin").c_str());
        }
        rmdir(final_dir.c_str());
    };

    try {
        RandomRotation rotation;
        rotation.init(dim, 42, RotationType::FhtKac);
        save_rotation_exclusive(rotation, base_dir + "/rotation.bin");
        write_rotate_subspace_flag_exclusive(base_dir);
        StreamingRabitqOptions rabitq_options;
        rabitq_options.rabitq_k = rabitq_k;
        rabitq_options.rabitq_seed = 42;
        rabitq_options.block_rows = 6;
        rabitq_options.sample_rows = 16;
        rabitq_options.train_iters = 2;
        StreamingIvfPostingOptions ivf_options;
        ivf_options.M = M;
        ivf_options.ivf_k = ivf_k;
        ivf_options.block_rows = rabitq_options.block_rows;
        ivf_options.sample_rows = rabitq_options.sample_rows;
        ivf_options.train_iters = rabitq_options.train_iters;
        ivf_options.preserve_assignment_artifacts = true;
        write_streaming_no_reorder_fused_artifacts(input_path, base_dir, rotation,
                                                   rabitq_options, ivf_options);

        CHECK(access((base_dir + "/posting_assignments.manifest").c_str(), F_OK) == 0,
              "preserved assignment manifest must exist");
        for (int m = 0; m < M; ++m) {
            CHECK(access((base_dir + "/posting_assign_subspace_" + std::to_string(m) + ".assign.bin").c_str(), F_OK) == 0,
                  "preserved assignment file must exist");
        }

        IndexConfig reorder_config;
        reorder_config.num_subspaces = M;
        reorder_config.rabitq_k = rabitq_k;
        reorder_config.pid_reorder_key_mode = 1;
        reorder_config.pid_reorder_cell_key_depth = 2;
        impl::PosthocReorderIndexOptions posthoc_options;
        posthoc_options.source_index_dir = base_dir;
        posthoc_options.output_index_dir = final_dir;
        posthoc_options.n = n;
        posthoc_options.M = M;
        posthoc_options.config = reorder_config;
        posthoc_options.io_rows = 4;
        posthoc_options.bucket_buffer_rows = 2;
        impl::posthoc_reorder_no_reorder_index(posthoc_options);

        QueryEngine engine;
        engine.init(reorder_config);
        engine.load_index(final_dir);
    } catch (const std::exception& e) {
        cleanup();
        FAIL(std::string("Posthoc preserved assignment path threw: ") + e.what());
        return;
    }

    CHECK(access((final_dir + "/perm_to_orig.bin").c_str(), F_OK) == 0,
          "preserved assignment posthoc must expose final perm marker");
    for (int m = 0; m < M; ++m) {
        CHECK(access((base_dir + "/posting_assign_subspace_" + std::to_string(m) + ".assign.bin").c_str(), F_OK) == 0,
              "posthoc must not delete source preserved assignment artifacts");
        CHECK(access((final_dir + "/posting_assign_subspace_" + std::to_string(m) + ".reorder.tmp").c_str(), F_OK) != 0,
              "trusted assignment path should not leave output temp assignments");
    }

    cleanup();
    TEST("Posthoc reuses preserved assignments"); PASS();
}

void test_posthoc_reorder_rejects_bad_preserved_assignments() {
    std::cout << "[PID reorder posthoc bad preserved assignments]" << std::endl;

    const std::string suffix = std::to_string(getpid());
    const std::string input_path = "/tmp/score_bad_preserved_assign_input_" + suffix + ".fbin";
    const std::string base_dir = "/tmp/score_bad_preserved_assign_base_" + suffix;
    const std::string final_dir = "/tmp/score_bad_preserved_assign_final_" + suffix;
    const int n = 18;
    const int dim = 64;
    const int M = 4;
    const int ivf_k = 3;
    const int rabitq_k = 4;

    std::vector<float> rows((size_t)n * dim);
    for (int i = 0; i < n; ++i) {
        for (int d = 0; d < dim; ++d) rows[(size_t)i * dim + d] = (float)(((i + 3) * 7 + d) % 53);
    }
    write_test_fbin(input_path, rows, n, dim);
    mkdir(base_dir.c_str(), 0755);
    mkdir(final_dir.c_str(), 0755);

    RandomRotation rotation;
    rotation.init(dim, 42, RotationType::FhtKac);
    save_rotation_exclusive(rotation, base_dir + "/rotation.bin");
    write_rotate_subspace_flag_exclusive(base_dir);
    StreamingRabitqOptions rabitq_options;
    rabitq_options.rabitq_k = rabitq_k;
    rabitq_options.block_rows = 6;
    rabitq_options.sample_rows = 16;
    rabitq_options.train_iters = 2;
    StreamingIvfPostingOptions ivf_options;
    ivf_options.M = M;
    ivf_options.ivf_k = ivf_k;
    ivf_options.block_rows = rabitq_options.block_rows;
    ivf_options.sample_rows = rabitq_options.sample_rows;
    ivf_options.train_iters = rabitq_options.train_iters;
    ivf_options.preserve_assignment_artifacts = true;
    write_streaming_no_reorder_fused_artifacts(input_path, base_dir, rotation,
                                               rabitq_options, ivf_options);

    {
        std::ofstream manifest(base_dir + "/posting_assignments.manifest", std::ios::app);
        manifest << "trailing_garbage\n";
    }
    bool rejected_trailing_manifest = false;
    try {
        impl::load_preserved_posting_assignments(base_dir, n, M);
    } catch (const std::exception&) {
        rejected_trailing_manifest = true;
    }
    CHECK(rejected_trailing_manifest, "preserved assignment manifest must reject trailing data");

    const std::string removed_assignment = base_dir + "/posting_assign_subspace_0.assign.bin";
    std::remove(removed_assignment.c_str());

    IndexConfig reorder_config;
    reorder_config.num_subspaces = M;
    reorder_config.rabitq_k = rabitq_k;
    impl::PosthocReorderIndexOptions posthoc_options;
    posthoc_options.source_index_dir = base_dir;
    posthoc_options.output_index_dir = final_dir;
    posthoc_options.n = n;
    posthoc_options.M = M;
    posthoc_options.config = reorder_config;

    bool threw = false;
    try {
        impl::posthoc_reorder_no_reorder_index(posthoc_options);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw, "posthoc must reject malformed preserved assignment manifest");
    CHECK(access((final_dir + "/perm_to_orig.bin").c_str(), F_OK) != 0,
          "bad preserved assignment path must not expose final perm marker");
    CHECK(access((base_dir + "/posting_assignments.manifest").c_str(), F_OK) == 0,
          "bad preserved assignment failure must not delete source manifest");

    const std::vector<std::string> base_names = streaming_no_reorder_artifact_names(M);
    for (const std::string& name : base_names) std::remove((base_dir + "/" + name).c_str());
    std::remove(input_path.c_str());
    rmdir(base_dir.c_str());
    std::remove((final_dir + "/rotation.bin").c_str());
    std::remove((final_dir + "/rotate_subspace.flag").c_str());
    std::remove((final_dir + "/rabitq_centroids.bin").c_str());
    for (int m = 0; m < M; ++m) {
        std::remove((final_dir + "/ivf_subspace_" + std::to_string(m) + ".bin").c_str());
        std::remove((final_dir + "/posting_assign_subspace_" + std::to_string(m) + ".reorder.tmp").c_str());
    }
    rmdir(final_dir.c_str());
    TEST("Posthoc rejects bad preserved assignments"); PASS();
}

void test_streaming_fused_reorder_staging_finalizes_query_index() {
    std::cout << "[Streaming fused reorder staging]" << std::endl;

    const std::string suffix = std::to_string(getpid());
    const std::string input_path = "/tmp/score_fused_reorder_input_" + suffix + ".fbin";
    const std::string staging_dir = "/tmp/score_fused_reorder_staging_" + suffix;
    const std::string final_dir = "/tmp/score_fused_reorder_final_" + suffix;
    const int n = 20;
    const int dim = 64;
    const int M = 4;
    const int ivf_k = 3;
    const int rabitq_k = 4;

    std::vector<float> rows((size_t)n * dim);
    for (int i = 0; i < n; ++i) {
        for (int d = 0; d < dim; ++d) {
            rows[(size_t)i * dim + d] = (float)(((i + 11) * 17 + d * 5) % 101) * 0.015f;
        }
    }
    write_test_fbin(input_path, rows, n, dim);
    mkdir(staging_dir.c_str(), 0755);
    mkdir(final_dir.c_str(), 0755);

    auto cleanup = [&]() {
        const std::vector<std::string> staging_names = streaming_no_reorder_artifact_names(M);
        for (const std::string& name : staging_names) std::remove((staging_dir + "/" + name).c_str());
        for (int m = 0; m < M; ++m) {
            std::remove((staging_dir + "/posting_assign_subspace_" + std::to_string(m) + ".tmp").c_str());
        }
        rmdir(staging_dir.c_str());
        std::remove(input_path.c_str());
        std::remove((final_dir + "/rotation.bin").c_str());
        std::remove((final_dir + "/rotate_subspace.flag").c_str());
        std::remove((final_dir + "/rabitq_centroids.bin").c_str());
        std::remove((final_dir + "/verify_vectors.fbin").c_str());
        std::remove((final_dir + "/rabitq_assignments.bin").c_str());
        std::remove((final_dir + "/rabitq_bin_data.bin").c_str());
        std::remove((final_dir + "/perm_to_orig.bin").c_str());
        std::remove((final_dir + "/perm_to_orig.raw.tmp").c_str());
        std::remove((final_dir + "/perm_to_orig.scratch.tmp").c_str());
        std::remove((final_dir + "/orig_to_new.reorder.tmp").c_str());
        for (int m = 0; m < M; ++m) {
            std::remove((final_dir + "/ivf_subspace_" + std::to_string(m) + ".bin").c_str());
            std::remove((final_dir + "/posting_subspace_" + std::to_string(m) + ".bin").c_str());
        }
        rmdir(final_dir.c_str());
    };

    try {
        RandomRotation rotation;
        rotation.init(dim, 42, RotationType::FhtKac);
        save_rotation_exclusive(rotation, staging_dir + "/rotation.bin");
        write_rotate_subspace_flag_exclusive(staging_dir);
        StreamingRabitqOptions rabitq_options;
        rabitq_options.rabitq_k = rabitq_k;
        rabitq_options.rabitq_seed = 42;
        rabitq_options.block_rows = 6;
        rabitq_options.sample_rows = 16;
        rabitq_options.train_iters = 2;
        StreamingIvfPostingOptions ivf_options;
        ivf_options.M = M;
        ivf_options.ivf_k = ivf_k;
        ivf_options.block_rows = rabitq_options.block_rows;
        ivf_options.sample_rows = rabitq_options.sample_rows;
        ivf_options.train_iters = rabitq_options.train_iters;
        ivf_options.materialize_no_reorder_postings = false;
        StreamingReorderStagingArtifacts staging = write_streaming_no_reorder_fused_artifacts(
            input_path, staging_dir, rotation, rabitq_options, ivf_options);

        for (int m = 0; m < M; ++m) {
            CHECK(access((staging_dir + "/posting_subspace_" + std::to_string(m) + ".bin").c_str(), F_OK) != 0,
                  "fused reorder staging must skip no-reorder posting materialization");
            CHECK(access(staging.subspace_assignment_paths[(size_t)m].c_str(), F_OK) == 0,
                  "fused reorder staging must keep temp assignments until finalization");
        }
        CHECK(access((staging_dir + "/posting_assignments.manifest").c_str(), F_OK) != 0,
              "fused reorder staging must not publish preserved-assignment manifest");

        impl::copy_file_exclusive(staging_dir + "/rotation.bin", final_dir + "/rotation.bin");
        impl::copy_file_exclusive(staging_dir + "/rotate_subspace.flag", final_dir + "/rotate_subspace.flag");
        impl::copy_file_exclusive(staging.centroids_path, final_dir + "/rabitq_centroids.bin");
        for (int m = 0; m < M; ++m) {
            impl::copy_file_exclusive(staging.ivf_paths[(size_t)m],
                                        final_dir + "/ivf_subspace_" + std::to_string(m) + ".bin");
        }

        IndexConfig reorder_config;
        reorder_config.num_subspaces = M;
        reorder_config.rabitq_k = rabitq_k;
        reorder_config.pid_reorder_key_mode = 1;
        reorder_config.pid_reorder_cell_key_depth = 2;
        const std::string raw_perm = final_dir + "/perm_to_orig.raw.tmp";
        const std::string scratch_perm = final_dir + "/perm_to_orig.scratch.tmp";
        impl::DiskPidReorderOptions reorder_options;
        reorder_options.n = staging.n;
        reorder_options.M = staging.M;
        reorder_options.assignment_key_range = (size_t)rabitq_k;
        reorder_options.cells_per_subspace = staging.cells_per_subspace;
        reorder_options.subspace_assignment_paths = staging.subspace_assignment_paths;
        reorder_options.rabitq_assignments_path = staging.assignments_path;
        reorder_options.perm_to_orig_path = raw_perm;
        reorder_options.scratch_perm_path = scratch_perm;
        reorder_options.config = reorder_config;
        reorder_options.io_rows = 4;
        reorder_options.bucket_buffer_rows = 2;
        impl::build_pid_reorder_perm_disk_staging(reorder_options);

        impl::ReorderedArtifactFinalizeOptions finalize_options;
        finalize_options.n = staging.n;
        finalize_options.M = staging.M;
        finalize_options.cells_per_subspace = staging.cells_per_subspace;
        finalize_options.subspace_assignment_paths = staging.subspace_assignment_paths;
        finalize_options.raw_perm_to_orig_path = raw_perm;
        finalize_options.verify_input_path = staging.verify_path;
        finalize_options.assignments_input_path = staging.assignments_path;
        finalize_options.bin_data_input_path = staging.bin_data_path;
        finalize_options.verify_output_path = final_dir + "/verify_vectors.fbin";
        finalize_options.assignments_output_path = final_dir + "/rabitq_assignments.bin";
        finalize_options.bin_data_output_path = final_dir + "/rabitq_bin_data.bin";
        for (int m = 0; m < M; ++m) {
            finalize_options.posting_output_paths.push_back(final_dir + "/posting_subspace_" + std::to_string(m) + ".bin");
        }
        finalize_options.final_perm_to_orig_path = final_dir + "/perm_to_orig.bin";
        finalize_options.io_rows = 4;
        impl::finalize_reordered_artifacts_from_staging(finalize_options);

        for (const std::string& path : staging.subspace_assignment_paths) std::remove(path.c_str());
        std::remove(raw_perm.c_str());
        std::remove(scratch_perm.c_str());

        QueryEngine engine;
        engine.init(reorder_config);
        engine.load_index(final_dir);
    } catch (const std::exception& e) {
        cleanup();
        FAIL(std::string("Fused reorder staging threw: ") + e.what());
        return;
    }

    CHECK(access((final_dir + "/perm_to_orig.bin").c_str(), F_OK) == 0,
          "fused reorder final index must expose final perm marker");
    for (int m = 0; m < M; ++m) {
        CHECK(access((staging_dir + "/posting_assign_subspace_" + std::to_string(m) + ".tmp").c_str(), F_OK) != 0,
              "fused reorder success cleanup should remove staging temp assignments");
    }

    cleanup();
    TEST("Fused reorder staging finalizes query index"); PASS();
}

void test_posthoc_reorder_rejects_subspace_mismatch() {
    std::cout << "[PID reorder posthoc subspace mismatch]" << std::endl;

    const std::string suffix = std::to_string(getpid());
    const std::string base_dir = "/tmp/score_posthoc_mismatch_base_" + suffix;
    const std::string final_dir = "/tmp/score_posthoc_mismatch_final_" + suffix;
    mkdir(base_dir.c_str(), 0755);
    mkdir(final_dir.c_str(), 0755);

    const std::string extra_ivf = base_dir + "/ivf_subspace_3.bin";
    const std::string extra_posting = base_dir + "/posting_subspace_3.bin";
    {
        std::ofstream out(extra_ivf, std::ios::binary);
        out.put('\0');
    }
    {
        std::ofstream out(extra_posting, std::ios::binary);
        out.put('\0');
    }

    IndexConfig config;
    config.num_subspaces = 3;
    config.rabitq_k = 1;
    impl::PosthocReorderIndexOptions options;
    options.source_index_dir = base_dir;
    options.output_index_dir = final_dir;
    options.n = 1;
    options.M = 3;
    options.config = config;

    bool threw = false;
    try {
        impl::posthoc_reorder_no_reorder_index(options);
    } catch (const std::exception&) {
        threw = true;
    }

    CHECK(threw, "posthoc reorder must reject M smaller than source subspace count");
    CHECK(access(extra_ivf.c_str(), F_OK) == 0, "posthoc mismatch rejection must not delete source IVF artifact");
    CHECK(access((final_dir + "/perm_to_orig.bin").c_str(), F_OK) != 0,
          "posthoc mismatch rejection must not expose final perm marker");

    std::remove(extra_ivf.c_str());
    std::remove(extra_posting.c_str());
    rmdir(base_dir.c_str());
    rmdir(final_dir.c_str());
    TEST("Posthoc rejects subspace mismatch"); PASS();
}

void test_fused_no_reorder_rejects_fvecs() {
    std::cout << "[Streaming fused fbin-only]" << std::endl;

    const std::string suffix = std::to_string(getpid());
    const std::string input_path = "/tmp/score_stream_fused_reject_" + suffix + ".fvecs";
    const std::string index_dir = "/tmp/score_stream_fused_reject_index_" + suffix;
    const int n = 8;
    const int dim = 96;
    std::vector<float> rows((size_t)n * dim);
    for (int i = 0; i < n * dim; ++i) rows[(size_t)i] = (float)(i % 17) * 0.1f;
    write_test_fvecs(input_path, rows, n, dim);
    mkdir(index_dir.c_str(), 0755);

    RandomRotation rotation;
    rotation.init(dim, 42, RotationType::FhtKac);
    StreamingRabitqOptions rabitq_options;
    rabitq_options.rabitq_k = 2;
    rabitq_options.sample_rows = 4;
    rabitq_options.block_rows = 4;
    rabitq_options.train_iters = 1;
    StreamingIvfPostingOptions ivf_options;
    ivf_options.M = 4;
    ivf_options.ivf_k = 2;
    ivf_options.sample_rows = rabitq_options.sample_rows;
    ivf_options.block_rows = rabitq_options.block_rows;
    ivf_options.train_iters = rabitq_options.train_iters;
    ivf_options.subspace_workers = 2;

    bool rejected = false;
    try {
        write_streaming_no_reorder_fused_artifacts(input_path, index_dir, rotation,
                                                   rabitq_options, ivf_options);
    } catch (const std::exception& e) {
        rejected = std::string(e.what()).find("fbin") != std::string::npos;
    }
    CHECK(rejected, "fused builder should reject fvecs input");

    rmdir(index_dir.c_str());
    std::remove(input_path.c_str());
    TEST("Fused no-reorder rejects fvecs"); PASS();
}

