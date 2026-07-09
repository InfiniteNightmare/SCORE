#include "test_macros.h"

void test_rabitq() {
    std::cout << "[RaBitQ]" << std::endl;
    const int dim = 128;
    RaBitQuantizer q;
    q.init(dim);

    TEST("Init");
    CHECK(q.dim() == dim, "dim mismatch");
    CHECK(q.words_per_vector() == 2, "words_per_vec should be 2 for dim=128");
    PASS();

    // Test binary quantization
    std::vector<float> x_rot(dim);
    std::mt19937 rng(456);
    std::normal_distribution<float> dist(0, 1);
    for (int i = 0; i < dim; i++) x_rot[i] = dist(rng);

    std::vector<uint64_t> code(q.words_per_vector(), 0);
    q.quantize_binary(x_rot.data(), code.data());

    // Verify: bit set iff x_rot[i] >= 0
    for (int i = 0; i < dim; i++) {
        int word = i / 64;
        int bit = i % 64;
        bool code_bit = (code[word] >> bit) & 1;
        bool expected = x_rot[i] >= 0;
        CHECK(code_bit == expected,
              "Binary code mismatch at dim " + std::to_string(i));
    }
    TEST("Binary quantization"); PASS();

    // Test Hamming distance
    std::vector<float> y_rot(dim);
    for (int i = 0; i < dim; i++) y_rot[i] = dist(rng);
    std::vector<uint64_t> code2(q.words_per_vector(), 0);
    q.quantize_binary(y_rot.data(), code2.data());

    int hamming = q.hamming_distance(code.data(), code2.data());
    // Manually count
    int manual = 0;
    for (int i = 0; i < dim; i++) {
        bool b1 = x_rot[i] >= 0;
        bool b2 = y_rot[i] >= 0;
        if (b1 != b2) manual++;
    }
    CHECK(hamming == manual, "Hamming distance mismatch: got " + std::to_string(hamming) +
          " expected " + std::to_string(manual));
    TEST("Hamming distance"); PASS();

    // Test batch quantization
    int n = 50;
    std::vector<float> data(n * dim);
    for (int i = 0; i < n * dim; i++) data[i] = dist(rng);
    std::vector<uint64_t> codes(n * q.words_per_vector());
    q.quantize_binary_batch(data.data(), codes.data(), n);

    // Verify first matches single
    std::vector<uint64_t> single_code(q.words_per_vector());
    q.quantize_binary(data.data(), single_code.data());
    CHECK(single_code[0] == codes[0], "Batch quantize mismatch");
    TEST("Batch quantization"); PASS();

    // Test corrective factors
    std::vector<float> x(dim);
    for (int i = 0; i < dim; i++) x[i] = dist(rng);
    float norm_x = 0;
    for (float v : x) norm_x += v * v;
    norm_x = std::sqrt(norm_x);

    auto factor = q.compute_factor(x_rot.data(), nullptr, code.data(), norm_x);
    CHECK(factor.sqr_res >= 0, "sqr_res should be non-negative");
    TEST("Corrective factors"); PASS();
}

void test_sc_score() {
    std::cout << "[SC-Score]" << std::endl;
    const int n = 100;
    const int M = 4;

    SCScore sc;
    sc.init(n, M);

    // Test increment
    uint32_t ids1[] = {0, 1, 2, 3, 5, 8};
    sc.increment(ids1, 6);
    for (int i = 0; i < 6; i++) {
        CHECK(sc.score(ids1[i]) == 1, "Score should be 1 after one increment");
    }
    CHECK(sc.score(4) == 0, "Unincremented score should be 0");
    TEST("Increment"); PASS();

    // Add more increments
    uint32_t ids2[] = {0, 1, 4, 5};
    sc.increment(ids2, 4);
    CHECK(sc.score(0) == 2, "Score should be 2");
    CHECK(sc.score(4) == 1, "Score should be 1");
    TEST("Multiple increments"); PASS();

    SCScore live_increment_sc;
    live_increment_sc.init(16, M);
    uint32_t live_increment_ids[] = {1, 2, 3, 4, 5, 6, 7, 8};
    auto increment_live = LiveTable::from_live_pids(16, {2, 4, 8});
    live_increment_sc.increment_weighted_live(
        live_increment_ids, 8, 3, *increment_live, 0);
    CHECK(live_increment_sc.score(2) == 3 &&
              live_increment_sc.score(4) == 3 &&
              live_increment_sc.score(8) == 3,
          "live-aware weighted increment must score live PIDs");
    CHECK(live_increment_sc.score(1) == 0 &&
              live_increment_sc.score(3) == 0 &&
              live_increment_sc.score(5) == 0 &&
              live_increment_sc.score(6) == 0 &&
              live_increment_sc.score(7) == 0,
          "live-aware weighted increment must skip dead PIDs");
    TEST("Live-aware weighted increment skips dead PIDs"); PASS();

    SCScore clear_sc;
    clear_sc.init(512, M);
    std::vector<uint32_t> clear_ids;
    clear_ids.reserve(256);
    for (uint32_t pid = 0; pid < 512; pid += 2) {
        clear_sc.mutable_scores()[pid] = 7;
        clear_sc.mutable_scores()[pid + 1] = 5;
        clear_ids.push_back(pid);
    }
    clear_sc.clear_scores_for_pids(clear_ids);
    for (uint32_t pid = 0; pid < 512; pid += 2) {
        CHECK(clear_sc.score((int)pid) == 0,
              "large sparse clear must clear requested PID");
        CHECK(clear_sc.score((int)pid + 1) == 5,
              "large sparse clear must not touch adjacent PID");
    }
    TEST("Sparse clear handles large PID lists"); PASS();

    SCScore fast_clear_sc;
    fast_clear_sc.init(512, M);
    for (uint32_t pid = 0; pid < 512; pid += 2) {
        fast_clear_sc.mutable_scores()[pid] = 9;
        fast_clear_sc.mutable_scores()[pid + 1] = 4;
    }
    fast_clear_sc.clear_scores_for_valid_pids(clear_ids);
    for (uint32_t pid = 0; pid < 512; pid += 2) {
        CHECK(fast_clear_sc.score((int)pid) == 0,
              "valid sparse clear must clear requested PID");
    CHECK(fast_clear_sc.score((int)pid + 1) == 4,
              "valid sparse clear must not touch adjacent PID");
    }
    TEST("Valid sparse clear handles large PID lists"); PASS();

    SCScore uninitialized_sc;
    uninitialized_sc.init(128, M);
    uninitialized_sc.mutable_scores()[7] = 42;
    uninitialized_sc.init_uninitialized(64, M);
    CHECK(uninitialized_sc.score(7) == 42,
          "uninitialized score resize should not clear existing mapped storage");
    uninitialized_sc.reset();
    CHECK(uninitialized_sc.score(7) == 0,
          "reset must clear scores after uninitialized resize");
    TEST("Uninitialized score resize relies on explicit reset"); PASS();

    // Select top
    auto top = sc.select_top(0.05f); // 5% = 5 candidates
    CHECK(!top.empty(), "Should return candidates");
    CHECK(top.size() > 0, "Should have at least 1 candidate");
    TEST("Select top"); PASS();

    SCScore live_sc;
    live_sc.init(8, M);
    live_sc.mutable_scores()[0] = 20; // dead high-score PID must not be selected.
    live_sc.mutable_scores()[1] = 5;
    live_sc.mutable_scores()[2] = 4;
    live_sc.mutable_scores()[3] = 3;  // dead lower-score PID.
    live_sc.mutable_scores()[4] = 2;
    auto live = LiveTable::from_live_pids(8, {1, 2, 4});
    std::vector<int> live_top;
    SCSelectStats live_stats;
    live_sc.select_top_n_live_into(3, *live, &live_stats, live_top);
    CHECK(live_top.size() == 3, "live-aware select should return requested live candidates");
    CHECK(std::find(live_top.begin(), live_top.end(), 0) == live_top.end(),
          "live-aware select must skip dead high-score PID");
    CHECK(live_top[0] == 1 && live_top[1] == 2 && live_top[2] == 4,
          "live-aware select order should follow live score threshold scan");
    CHECK(live_stats.nonzero_scores == 3, "live-aware stats should count only live nonzero PIDs");
    CHECK(live_stats.selected_total == 3, "live-aware stats selected_total mismatch");
    TEST("Live-aware select skips dead PIDs"); PASS();

    SCScore vector_live_sc;
    vector_live_sc.init(130, M);
    vector_live_sc.mutable_scores()[0] = 250;
    vector_live_sc.mutable_scores()[64] = 240;
    vector_live_sc.mutable_scores()[129] = 230;
    vector_live_sc.mutable_scores()[2] = 6;
    vector_live_sc.mutable_scores()[5] = 10;
    vector_live_sc.mutable_scores()[63] = 9;
    vector_live_sc.mutable_scores()[65] = 8;
    vector_live_sc.mutable_scores()[128] = 7;
    std::vector<uint32_t> vector_live_pids;
    vector_live_pids.reserve(127);
    for (uint32_t pid = 0; pid < 130; ++pid) {
        if (pid == 0 || pid == 64 || pid == 129) continue;
        vector_live_pids.push_back(pid);
    }
    auto vector_live = LiveTable::from_live_pids(130, vector_live_pids);
    std::vector<int> vector_live_top;
    SCSelectStats vector_live_stats;
    vector_live_sc.select_top_n_live_into(5, *vector_live, &vector_live_stats, vector_live_top);
    CHECK(vector_live_top.size() == 5, "vector live select should return all live nonzero candidates");
    for (int dead_pid : {0, 64, 129}) {
        CHECK(std::find(vector_live_top.begin(), vector_live_top.end(), dead_pid) == vector_live_top.end(),
              "vector live select must skip dead high-score PID");
    }
    for (int expected_pid : {2, 5, 63, 65, 128}) {
        CHECK(std::find(vector_live_top.begin(), vector_live_top.end(), expected_pid) != vector_live_top.end(),
              "vector live select dropped a live nonzero PID");
    }
    CHECK(vector_live_stats.nonzero_scores == 5,
          "vector live stats should count only live nonzero PIDs");
    TEST("Live-aware select handles vector masks"); PASS();

    SCScore range_live_sc;
    range_live_sc.init(181, M);
    range_live_sc.mutable_scores()[0] = 250;
    range_live_sc.mutable_scores()[36] = 240;
    range_live_sc.mutable_scores()[41] = 10;
    range_live_sc.mutable_scores()[127] = 9;
    range_live_sc.mutable_scores()[180] = 8;
    std::vector<uint32_t> suffix_live_pids;
    for (uint32_t pid = 37; pid < 181; ++pid) suffix_live_pids.push_back(pid);
    auto suffix_live = LiveTable::from_live_pids(181, suffix_live_pids);
    std::vector<int> suffix_live_top;
    std::vector<int> range_live_top;
    SCSelectStats range_stats;
    range_live_sc.select_top_n_live_into(3, *suffix_live, nullptr, suffix_live_top);
    range_live_sc.select_top_n_range_into(3, 37, 181, &range_stats, range_live_top);
    CHECK(range_live_top == suffix_live_top,
          "range select should match equivalent live suffix select");
    for (int dead_pid : {0, 36}) {
        CHECK(std::find(range_live_top.begin(), range_live_top.end(), dead_pid) == range_live_top.end(),
              "range select must skip dead prefix PID");
    }
    CHECK(range_stats.nonzero_scores == 3 && range_stats.selected_total == 3,
          "range select stats should cover only the live suffix");
    TEST("Live suffix range select matches live-aware select"); PASS();

    SCScore sparse_live_sc;
    sparse_live_sc.init(1024, M);
    sparse_live_sc.mutable_scores()[0] = 250;
    sparse_live_sc.mutable_scores()[64] = 240;
    sparse_live_sc.mutable_scores()[1023] = 230;
    sparse_live_sc.mutable_scores()[1] = 12;
    sparse_live_sc.mutable_scores()[2] = 11;
    sparse_live_sc.mutable_scores()[65] = 10;
    sparse_live_sc.mutable_scores()[128] = 9;
    sparse_live_sc.mutable_scores()[700] = 8;
    sparse_live_sc.mutable_scores()[900] = 7;
    auto sparse_live = LiveTable::all_live(1024)->with_updates(1024, {}, {0, 64, 1023});
    std::vector<int> sparse_exact;
    std::vector<int> sparse_fast;
    SCSelectStats sparse_stats;
    sparse_live_sc.select_top_n_live_into(6, *sparse_live, &sparse_stats, sparse_exact);
    sparse_live_sc.select_top_n_live_into(6, *sparse_live, nullptr, sparse_fast);
    CHECK(sparse_fast == sparse_exact,
          "live select without stats should match exact live-aware select");
    for (int dead_pid : {0, 64, 1023}) {
        CHECK(std::find(sparse_fast.begin(), sparse_fast.end(), dead_pid) == sparse_fast.end(),
              "live select without stats must skip dead high-score PID");
    }
    TEST("Sparse live select without stats matches exact"); PASS();

}

void test_subspace_repack_stride_equivalence() {
    std::cout << "[Build repack]" << std::endl;

    auto repack = [](const float* src, int n, int row_stride, int dim, int M, int sub_dim) {
        std::vector<float> out((size_t)n * dim, -1.0f);
        for (int m = 0; m < M; ++m) {
            for (int i = 0; i < n; ++i) {
                const float* row = src + (size_t)i * row_stride;
                std::memcpy(out.data() + (size_t)m * n * sub_dim + (size_t)i * sub_dim,
                            row + (size_t)m * sub_dim,
                            (size_t)sub_dim * sizeof(float));
            }
        }
        return out;
    };

    {
        const int n = 7;
        const int dim = 12;
        const int padded_dim = 16;
        const int M = 3;
        const int sub_dim = dim / M;

        std::vector<float> dim_stride((size_t)n * dim);
        std::vector<float> padded_stride((size_t)n * padded_dim, -777.0f);
        for (int i = 0; i < n; ++i) {
            for (int d = 0; d < dim; ++d) {
                float v = (float)(100 * i + d + 1);
                dim_stride[(size_t)i * dim + d] = v;
                padded_stride[(size_t)i * padded_dim + d] = v;
            }
        }

        auto from_dim = repack(dim_stride.data(), n, dim, dim, M, sub_dim);
        auto from_padded = repack(padded_stride.data(), n, padded_dim, dim, M, sub_dim);
        CHECK(from_dim == from_padded, "Padded-stride repack mismatch when padded_dim > dim");
        TEST("Padded-stride repack equivalence"); PASS();
    }

    {
        const int n = 5;
        const int dim = 16;
        const int padded_dim = 16;
        const int M = 4;
        const int sub_dim = dim / M;

        std::vector<float> packed((size_t)n * dim);
        for (int i = 0; i < n; ++i) {
            for (int d = 0; d < dim; ++d) {
                packed[(size_t)i * dim + d] = (float)(10 * i + d);
            }
        }

        auto from_dim = repack(packed.data(), n, dim, dim, M, sub_dim);
        auto from_same = repack(packed.data(), n, padded_dim, dim, M, sub_dim);
        CHECK(from_dim == from_same, "Repack mismatch when padded_dim == dim");
        TEST("Equal-stride repack equivalence"); PASS();
    }

    {
        const float* data_ptr = reinterpret_cast<const float*>(0x1000);
        const float* pre_rotated = data_ptr;
        const float* shared_subspace = pre_rotated;
        const float* external_subspace = reinterpret_cast<const float*>(0x2000);
        const float* rot_padded = reinterpret_cast<const float*>(0x3000);

        CHECK(impl::should_repack_ivf_from_rot_padded(
                  true, shared_subspace, pre_rotated, data_ptr, rot_padded, 128, 96),
              "Shared FhtKac path should repack from rot_padded when padded_dim > dim");

        CHECK(!impl::should_repack_ivf_from_rot_padded(
                   true, external_subspace, pre_rotated, data_ptr, rot_padded, 128, 96),
              "External subspace rotation path must not be overridden by rot_padded repack");

        CHECK(!impl::should_repack_ivf_from_rot_padded(
                   true, shared_subspace, pre_rotated, data_ptr, rot_padded, 96, 96),
              "Equal-stride path should not force rot_padded repack");

        bool raw_forced_path = true;
        bool use_rot_padded = !raw_forced_path && impl::should_repack_ivf_from_rot_padded(
            true, shared_subspace, pre_rotated, data_ptr, rot_padded, 128, 96);
        CHECK(!use_rot_padded,
              "subspace_force_raw-style path must suppress rot_padded repack even when shared rotated path would match");

        TEST("Repack source selection guard"); PASS();
    }
}

void test_ivf_assign_block_cap() {
    std::cout << "[IVF assign cap]" << std::endl;

    CHECK(impl::capped_assign_block_size(2500) < 50000,
          "Scratch cap should reduce assign_batch block size for K=2500");
    TEST("Cap reduces large-K assign block"); PASS();

    CHECK(impl::capped_assign_block_size(2500, 32ull * 1024ull * 1024ull) <
          impl::capped_assign_block_size(2500),
          "Smaller scratch cap should reduce assign_batch block size further");
    TEST("Custom cap reduces block further"); PASS();

    CHECK(impl::capped_assign_block_size(1) == 50000,
          "Small-K path should keep default assign_batch block size");
    TEST("Small-K keeps default block"); PASS();

    CHECK(impl::capped_assign_block_size(0) == 1,
          "Non-positive K should clamp to a valid minimum block size");
    TEST("Zero-K clamp"); PASS();
}

void test_ivf_assign_batch_matches_scalar() {
    std::cout << "[IVF assign equivalence]" << std::endl;

    const int n_train = 96;
    const int n_query = 37;
    const int sub_dim = 8;
    const int row_stride = 13;
    const int K = 19;
    std::vector<float> train((size_t)n_train * sub_dim);
    for (int i = 0; i < n_train; ++i) {
        for (int d = 0; d < sub_dim; ++d) {
            train[(size_t)i * sub_dim + d] = (float)(((i + 5) * 31 + d * 11) % 97) * 0.013f;
        }
    }

    IVFIndex ivf;
    ivf.init(sub_dim, K, 0);
    ivf.train(train.data(), n_train, 3, 123);

    std::vector<float> queries((size_t)n_query * row_stride, -123.0f);
    for (int i = 0; i < n_query; ++i) {
        for (int d = 0; d < sub_dim; ++d) {
            queries[(size_t)i * row_stride + d] = (float)(((i + 7) * 17 + d * 19) % 113) * 0.017f;
        }
    }

    std::vector<int> batch((size_t)n_query);
    ivf.assign_batch_strided_with_scratch_cap(queries.data(), row_stride, batch.data(), n_query,
                                              1ull * 1024ull * 1024ull);
    for (int i = 0; i < n_query; ++i) {
        int scalar = ivf.assign(queries.data() + (size_t)i * row_stride);
        CHECK(batch[(size_t)i] == scalar, "Strided batch assignment must match scalar assign");
    }
    TEST("Strided batch matches scalar"); PASS();

    ivf.prepare_simd_centroids();
    std::vector<int> simd_batch((size_t)n_query);
    ivf.assign_batch_strided_simd_soa(queries.data(), row_stride, simd_batch.data(), n_query);
    for (int i = 0; i < n_query; ++i) {
        int scalar = ivf.assign(queries.data() + (size_t)i * row_stride);
        CHECK(simd_batch[(size_t)i] == scalar, "SoA SIMD strided assignment must match scalar assign");
    }
    TEST("SoA SIMD strided assignment matches scalar"); PASS();

    for (int test_sub_dim = 1; test_sub_dim <= 8; ++test_sub_dim) {
        const int test_row_stride = test_sub_dim + 5;
        std::vector<float> dim_train((size_t)n_train * (size_t)test_sub_dim);
        for (int i = 0; i < n_train; ++i) {
            for (int d = 0; d < test_sub_dim; ++d) {
                dim_train[(size_t)i * (size_t)test_sub_dim + (size_t)d] =
                    (float)(((i + 3) * 23 + d * 29 + test_sub_dim * 7) % 101) * 0.011f;
            }
        }
        IVFIndex dim_ivf;
        dim_ivf.init(test_sub_dim, K, 0);
        dim_ivf.train(dim_train.data(), n_train, 2, (uint64_t)(1000 + test_sub_dim));
        dim_ivf.prepare_simd_centroids();

        std::vector<float> dim_queries((size_t)n_query * (size_t)test_row_stride, -77.0f);
        for (int i = 0; i < n_query; ++i) {
            for (int d = 0; d < test_sub_dim; ++d) {
                dim_queries[(size_t)i * (size_t)test_row_stride + (size_t)d] =
                    (float)(((i + 11) * 13 + d * 17 + test_sub_dim * 5) % 89) * 0.019f;
            }
        }
        std::vector<int> dim_batch((size_t)n_query);
        dim_ivf.assign_batch_strided_simd_soa(dim_queries.data(), test_row_stride,
                                              dim_batch.data(), n_query);
        for (int i = 0; i < n_query; ++i) {
            int scalar = dim_ivf.assign(dim_queries.data() + (size_t)i * (size_t)test_row_stride);
            CHECK(dim_batch[(size_t)i] == scalar,
                  "SoA SIMD assignment must match scalar for sub_dim 1..8");
        }
    }
    TEST("SoA SIMD low-subdim assignment matches scalar"); PASS();

    {
        const int high_sub_dim = 60;
        const int high_row_stride = 67;
        std::vector<float> high_train((size_t)n_train * (size_t)high_sub_dim);
        for (int i = 0; i < n_train; ++i) {
            for (int d = 0; d < high_sub_dim; ++d) {
                high_train[(size_t)i * (size_t)high_sub_dim + (size_t)d] =
                    (float)(((i + 13) * 7 + d * 31) % 127) * 0.007f;
            }
        }
        IVFIndex high_ivf;
        high_ivf.init(high_sub_dim, K, 0);
        high_ivf.train(high_train.data(), n_train, 2, 4242);

        std::vector<float> high_queries((size_t)n_query * (size_t)high_row_stride, -11.0f);
        for (int i = 0; i < n_query; ++i) {
            for (int d = 0; d < high_sub_dim; ++d) {
                high_queries[(size_t)i * (size_t)high_row_stride + (size_t)d] =
                    (float)(((i + 19) * 5 + d * 23) % 131) * 0.009f;
            }
        }
        std::vector<int> high_batch((size_t)n_query);
        high_ivf.assign_batch_strided_simd_points16(high_queries.data(), high_row_stride,
                                                    high_batch.data(), n_query);
        for (int i = 0; i < n_query; ++i) {
            int scalar = high_ivf.assign(high_queries.data() + (size_t)i * (size_t)high_row_stride);
            CHECK(high_batch[(size_t)i] == scalar,
                  "Point16 SIMD high-subdim assignment must match scalar assign");
        }
    }
    TEST("Point16 SIMD high-subdim assignment matches scalar"); PASS();

    auto write_ivf_fixture = [](const std::string& path, int sub_dim, int K,
                                const std::vector<float>& centroids) {
        std::ofstream out(path, std::ios::binary);
        int subspace_id = 0;
        std::vector<size_t> cell_sizes((size_t)K, 0);
        out.write(reinterpret_cast<const char*>(&sub_dim), sizeof(int));
        out.write(reinterpret_cast<const char*>(&K), sizeof(int));
        out.write(reinterpret_cast<const char*>(&subspace_id), sizeof(int));
        out.write(reinterpret_cast<const char*>(centroids.data()), centroids.size() * sizeof(float));
        out.write(reinterpret_cast<const char*>(cell_sizes.data()), cell_sizes.size() * sizeof(size_t));
    };

    {
        const std::string path = "/tmp/score_ivf_tie_" + std::to_string(getpid()) + ".bin";
        const int tie_k = 32;
        std::vector<float> centroids((size_t)tie_k, 10.0f);
        centroids[5] = 1.0f;
        centroids[20] = 1.0f;
        write_ivf_fixture(path, 1, tie_k, centroids);

        IVFIndex tie_ivf;
        tie_ivf.load(path);
        tie_ivf.prepare_simd_centroids();
        std::remove(path.c_str());

        float query[1] = {0.0f};
        int batch_id = -1;
        tie_ivf.assign_batch_strided_simd_soa(query, 1, &batch_id, 1);
        CHECK(batch_id == tie_ivf.assign(query), "AVX tie assignment must match scalar assign");
        CHECK(batch_id == 5, "Exact ties must keep scalar first-minimum centroid id");
    }
    TEST("AVX exact tie keeps first minimum"); PASS();

    {
        const std::string path = "/tmp/score_ivf_smallk_" + std::to_string(getpid()) + ".bin";
        const int small_k = 8;
        std::vector<float> centroids((size_t)small_k, 10.0f);
        centroids[3] = 1.0f;
        write_ivf_fixture(path, 1, small_k, centroids);

        IVFIndex small_ivf;
        small_ivf.load(path);
        small_ivf.prepare_simd_centroids();
        std::remove(path.c_str());

        float strided_query[4] = {0.0f, -99.0f, -99.0f, -99.0f};
        int batch_id = -1;
        small_ivf.assign_batch_strided_simd_soa(strided_query, 4, &batch_id, 1);
        CHECK(batch_id == small_ivf.assign(strided_query), "K<16 tail assignment must match scalar assign");
        CHECK(batch_id == 3, "K<16 tail path should find the nearest centroid");
    }
    TEST("K<16 tail path matches scalar"); PASS();
}

void test_late_rotated_source_from_slabs() {
    std::cout << "[Build late rotated source]" << std::endl;

    const float* dim_stride = reinterpret_cast<const float*>(0x2000);
    const float* rot_padded = reinterpret_cast<const float*>(0x3000);
    const float* external_subspace = reinterpret_cast<const float*>(0x4000);

    CHECK(impl::can_finalize_rotated_outputs_from_ivf_slabs(
              true, false, dim_stride, dim_stride, rot_padded, 96, 96, true),
          "Equal-stride shared-rotation slabs should be able to serve late rotated consumers");
    TEST("Equal-stride slab late-source guard"); PASS();

    CHECK(!impl::can_finalize_rotated_outputs_from_ivf_slabs(
               true, false, dim_stride, dim_stride, rot_padded, 128, 96, true),
          "Padded-tail path must not replace rot_padded with IVF slabs");
    TEST("Padded-tail guard"); PASS();

    CHECK(!impl::can_finalize_rotated_outputs_from_ivf_slabs(
               true, false, external_subspace, dim_stride, rot_padded, 96, 96, true),
          "External subspace-only rotation must not replace the verify backing");
    TEST("External subspace guard"); PASS();

    CHECK(!impl::can_finalize_rotated_outputs_from_ivf_slabs(
               true, false, dim_stride, dim_stride, rot_padded, 96, 96, false),
          "Without slab backing reuse there is no late rotated source to keep");
    TEST("No slab reuse guard"); PASS();

    const int M = 3;
    const int sub_dim = 2;
    std::vector<float> slab0 = {1, 2, 7, 8, 13, 14, 19, 20};
    std::vector<float> slab1 = {3, 4, 9, 10, 15, 16, 21, 22};
    std::vector<float> slab2 = {5, 6, 11, 12, 17, 18, 23, 24};
    std::vector<float*> slab_ptrs = {slab0.data(), slab1.data(), slab2.data()};
    std::vector<uint32_t> orig_ids = {2, 0, 3};
    std::vector<float> gathered((size_t)orig_ids.size() * M * sub_dim, -1.0f);

    impl::gather_rotated_rows_from_ivf_slabs(
        gathered.data(), slab_ptrs, orig_ids.data(), (int)orig_ids.size(), M, sub_dim);

    std::vector<float> expected = {
        13, 14, 15, 16, 17, 18,
        1,  2,  3,  4,  5,  6,
        19, 20, 21, 22, 23, 24,
    };
    CHECK(gathered == expected, "Slab gather should reconstruct full rotated rows in requested order");
    TEST("Slab gather reconstruction"); PASS();
}

void test_finalize_chunk_rows() {
    std::cout << "[Build finalize chunking]" << std::endl;

    const int threads = 48;
    size_t verify_record_bytes = (size_t)960 * sizeof(float);
    size_t verify_rows = impl::finalize_rows_per_chunk(verify_record_bytes, threads);
    CHECK(verify_rows >= (size_t)threads * 64,
          "Verify finalize rows should scale up to a useful per-thread floor");
    TEST("Verify rows-per-chunk floor"); PASS();

    size_t bindata_record_bytes = 132;
    size_t bindata_rows = impl::finalize_rows_per_chunk(bindata_record_bytes, threads);
    CHECK(bindata_rows >= (size_t)(256 * PAGE_SIZE) / bindata_record_bytes,
          "Finalize chunk helper must never shrink below the historical base chunk size");
    TEST("Finalize rows preserve base floor"); PASS();
}

void test_copy_finalize_subspace_adaptive() {
    std::cout << "[Build adaptive finalize copy]" << std::endl;

    const std::vector<int> widths = {0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 48, 59, 60, 61, 63, 64, 65, 96, 120};
    for (int width : widths) {
        std::vector<float> src((size_t)width + 8, -1.0f);
        std::vector<float> dst((size_t)width + 8, -777.0f);
        for (int i = 0; i < width; ++i) {
            src[(size_t)i] = (float)(100 + i);
        }
        for (int i = 0; i < 4; ++i) {
            dst[(size_t)width + i] = -999.0f;
        }

        impl::copy_finalize_subspace(dst.data(), src.data(), width);

        bool copied = true;
        for (int i = 0; i < width; ++i) {
            if (dst[(size_t)i] != src[(size_t)i]) {
                copied = false;
                break;
            }
        }
        CHECK(copied, "Adaptive finalize copy mismatch at width " + std::to_string(width));

        bool sentinel_ok = true;
        for (int i = 0; i < 4; ++i) {
            if (dst[(size_t)width + i] != -999.0f) {
                sentinel_ok = false;
                break;
            }
        }
        CHECK(sentinel_ok, "Adaptive finalize copy overwrote tail sentinel at width " + std::to_string(width));
    }
    TEST("Adaptive finalize copy widths"); PASS();
}

void test_large_subspace_gather_equivalence() {
    std::cout << "[Build gather subspace60]" << std::endl;

    const int n = 3;
    const int M = 4;
    const int sub_dim = 60;
    const int dim = M * sub_dim;
    std::vector<float> full_rows((size_t)n * dim);
    for (int i = 0; i < n; ++i) {
        for (int d = 0; d < dim; ++d) {
            full_rows[(size_t)i * dim + d] = (float)(1000 * i + d);
        }
    }

    std::vector<float> slab_storage((size_t)n * dim);
    std::vector<float*> slab_ptrs(M);
    for (int m = 0; m < M; ++m) {
        slab_ptrs[m] = slab_storage.data() + (size_t)m * n * sub_dim;
        for (int i = 0; i < n; ++i) {
            std::memcpy(slab_ptrs[m] + (size_t)i * sub_dim,
                        full_rows.data() + (size_t)i * dim + (size_t)m * sub_dim,
                        (size_t)sub_dim * sizeof(float));
        }
    }

    std::vector<uint32_t> orig_ids = {2, 0};
    std::vector<float> gathered((size_t)orig_ids.size() * dim, -1.0f);
    impl::gather_rotated_rows_from_ivf_slabs(
        gathered.data(), slab_ptrs, orig_ids.data(), (int)orig_ids.size(), M, sub_dim);

    std::vector<float> expected((size_t)orig_ids.size() * dim);
    for (size_t j = 0; j < orig_ids.size(); ++j) {
        std::memcpy(expected.data() + j * dim,
                    full_rows.data() + (size_t)orig_ids[j] * dim,
                    (size_t)dim * sizeof(float));
    }

    CHECK(gathered == expected, "sub_dim=60 gather must reconstruct rows exactly");
    TEST("Subspace60 gather equivalence"); PASS();
}

void test_adaptive_subspace_gather_equivalence() {
    std::cout << "[Build gather adaptive widths]" << std::endl;

    const std::vector<int> widths = {5, 17, 33, 61};
    for (int sub_dim : widths) {
        const int n = 3;
        const int M = 4;
        const int dim = M * sub_dim;
        std::vector<float> full_rows((size_t)n * dim);
        for (int i = 0; i < n; ++i) {
            for (int d = 0; d < dim; ++d) {
                full_rows[(size_t)i * dim + d] = (float)(1000 * i + d + sub_dim);
            }
        }

        std::vector<float> slab_storage((size_t)n * dim);
        std::vector<float*> slab_ptrs(M);
        for (int m = 0; m < M; ++m) {
            slab_ptrs[m] = slab_storage.data() + (size_t)m * n * sub_dim;
            for (int i = 0; i < n; ++i) {
                std::memcpy(slab_ptrs[m] + (size_t)i * sub_dim,
                            full_rows.data() + (size_t)i * dim + (size_t)m * sub_dim,
                            (size_t)sub_dim * sizeof(float));
            }
        }

        std::vector<uint32_t> orig_ids = {2, 0};
        std::vector<float> gathered((size_t)orig_ids.size() * dim, -1.0f);
        impl::gather_rotated_rows_from_ivf_slabs(
            gathered.data(), slab_ptrs, orig_ids.data(), (int)orig_ids.size(), M, sub_dim);

        std::vector<float> expected((size_t)orig_ids.size() * dim);
        for (size_t j = 0; j < orig_ids.size(); ++j) {
            std::memcpy(expected.data() + j * dim,
                        full_rows.data() + (size_t)orig_ids[j] * dim,
                        (size_t)dim * sizeof(float));
        }

        CHECK(gathered == expected,
              "Adaptive gather mismatch at sub_dim " + std::to_string(sub_dim));
    }

    TEST("Adaptive gather width matrix"); PASS();
}

