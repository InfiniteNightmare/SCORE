#include "test_macros.h"

void test_streaming_block_mib_row_conversion() {
    std::cout << "[Streaming block_mib conversion]" << std::endl;

    CHECK(streaming_block_rows_for_mib(2048, 64) == 8192,
          "64MiB block for dim=2048 should produce 8192 rows");
    CHECK(streaming_block_rows_for_mib(2048, 128) == 16384,
          "128MiB block for dim=2048 should produce 16384 rows");
    CHECK(streaming_block_rows_for_mib(768, 64) == 21845,
          "64MiB block for dim=768 should floor to 21845 rows");

    bool rejected_zero_dim = false;
    try {
        (void)streaming_block_rows_for_mib(0, 64);
    } catch (const std::exception&) {
        rejected_zero_dim = true;
    }
    CHECK(rejected_zero_dim, "block_mib conversion should reject zero dimension");

    bool rejected_too_small = false;
    try {
        (void)streaming_block_rows_for_mib(1024 * 1024, 1);
    } catch (const std::exception&) {
        rejected_too_small = true;
    }
    CHECK(rejected_too_small, "block_mib conversion should reject sizes below one row");

    TEST("Streaming block_mib row conversion"); PASS();
}

void test_streaming_verify_matches_batch_rotation() {
    std::cout << "[Streaming rotated verify]" << std::endl;

    const std::string suffix = std::to_string(getpid());
    const std::string input_path = "/tmp/score_stream_verify_input_" + suffix + ".fvecs";
    const std::string output_path = "/tmp/score_stream_verify_output_" + suffix + ".fbin";
    const int n = 5;
    const int dim = 96;
    std::vector<float> rows((size_t)n * dim);
    for (int i = 0; i < n * dim; ++i) rows[(size_t)i] = (float)((i % 11) - 5) * 0.25f;
    write_test_fvecs(input_path, rows, n, dim);

    RandomRotation rotation;
    rotation.init(dim, 42, RotationType::FhtKac);
    write_streaming_verify_vectors(input_path, output_path, rotation, 2);

    int out_n = 0, out_dim = 0;
    std::vector<float> streamed = load_fbin_payload(output_path, out_n, out_dim);
    CHECK(out_n == n, "streamed verify n mismatch");
    CHECK(out_dim == rotation.padded_dim(), "streamed verify dim mismatch");

    std::vector<float> expected((size_t)n * rotation.padded_dim());
    for (int i = 0; i < n; ++i) {
        rotation.apply(rows.data() + (size_t)i * dim,
                       expected.data() + (size_t)i * rotation.padded_dim());
    }
    CHECK(streamed.size() == expected.size(), "streamed verify payload size mismatch");
    bool equal = true;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(streamed[i] - expected[i]) > 1e-6f) {
            equal = false;
            break;
        }
    }
    CHECK(equal, "streamed verify payload differs from batch rotation");

    std::remove(input_path.c_str());
    std::remove(output_path.c_str());
    TEST("Streaming verify matches batch rotation"); PASS();
}

void test_streaming_rabitq_writes_artifact_headers() {
    std::cout << "[Streaming RaBitQ artifacts]" << std::endl;

    const std::string suffix = std::to_string(getpid());
    const std::string input_path = "/tmp/score_stream_rabitq_input_" + suffix + ".fvecs";
    const std::string index_dir = "/tmp/score_stream_rabitq_index_" + suffix;
    const int n = 24;
    const int dim = 96;
    const int k = 4;
    std::vector<float> rows((size_t)n * dim);
    for (int i = 0; i < n; ++i) {
        for (int d = 0; d < dim; ++d) {
            rows[(size_t)i * dim + d] = (float)((i * 17 + d * 3) % 101) * 0.01f;
        }
    }
    write_test_fvecs(input_path, rows, n, dim);
    mkdir(index_dir.c_str(), 0755);

    RandomRotation rotation;
    rotation.init(dim, 42, RotationType::FhtKac);
    StreamingRabitqOptions options;
    options.rabitq_k = k;
    options.rabitq_seed = 42;
    options.block_rows = 7;
    options.sample_rows = 16;
    options.train_iters = 2;
    write_streaming_rabitq_artifacts(input_path, index_dir, rotation, options);

    {
        std::ifstream in(index_dir + "/rabitq_centroids.bin", std::ios::binary);
        int kk = 0, dd = 0, pd = 0;
        in.read(reinterpret_cast<char*>(&kk), sizeof(int));
        in.read(reinterpret_cast<char*>(&dd), sizeof(int));
        in.read(reinterpret_cast<char*>(&pd), sizeof(int));
        CHECK(kk == k, "streaming rabitq centroid K mismatch");
        CHECK(dd == dim, "streaming rabitq centroid dim mismatch");
        CHECK(pd == rotation.padded_dim(), "streaming rabitq centroid padded dim mismatch");
    }
    std::vector<uint32_t> rabitq_assignments((size_t)n);
    {
        std::ifstream in(index_dir + "/rabitq_assignments.bin", std::ios::binary);
        int nn = 0;
        in.read(reinterpret_cast<char*>(&nn), sizeof(int));
        CHECK(nn == n, "streaming rabitq assignment n mismatch");
        in.read(reinterpret_cast<char*>(rabitq_assignments.data()), (size_t)n * sizeof(uint32_t));
    }
    {
        std::ifstream in(index_dir + "/rabitq_bin_data.bin", std::ios::binary);
        int nn = 0, bb = 0;
        in.read(reinterpret_cast<char*>(&nn), sizeof(int));
        in.read(reinterpret_cast<char*>(&bb), sizeof(int));
        CHECK(nn == n, "streaming rabitq bin n mismatch");
        const int payload_bytes = (int)rabitqlib::BinDataMap<float>::data_bytes(rotation.padded_dim());
        CHECK(bb == (int)(sizeof(uint32_t) + (size_t)payload_bytes),
              "streaming rabitq embedded record bytes mismatch");
        for (int i = 0; i < n; ++i) {
            uint32_t embedded_cid = std::numeric_limits<uint32_t>::max();
            in.read(reinterpret_cast<char*>(&embedded_cid), sizeof(uint32_t));
            CHECK(embedded_cid == rabitq_assignments[(size_t)i],
                  "streaming rabitq embedded cid mismatch");
            in.seekg(payload_bytes, std::ios::cur);
        }
    }

    std::remove((index_dir + "/rabitq_centroids.bin").c_str());
    std::remove((index_dir + "/rabitq_assignments.bin").c_str());
    std::remove((index_dir + "/rabitq_bin_data.bin").c_str());
    rmdir(index_dir.c_str());
    std::remove(input_path.c_str());
    TEST("Streaming RaBitQ artifact headers"); PASS();
}

void test_streaming_ivf_postings_are_loadable() {
    std::cout << "[Streaming IVF postings]" << std::endl;

    const std::string suffix = std::to_string(getpid());
    const std::string input_path = "/tmp/score_stream_ivf_input_" + suffix + ".fvecs";
    const std::string index_dir = "/tmp/score_stream_ivf_index_" + suffix;
    const int n = 20;
    const int dim = 96;
    const int M = 4;
    const int K = 3;
    const int sub_dim = dim / M;
    std::vector<float> rows((size_t)n * dim);
    for (int i = 0; i < n; ++i) {
        for (int d = 0; d < dim; ++d) {
            rows[(size_t)i * dim + d] = (float)(((i + 3) * 19 + d * 7) % 113) * 0.02f;
        }
    }
    write_test_fvecs(input_path, rows, n, dim);
    mkdir(index_dir.c_str(), 0755);

    RandomRotation rotation;
    rotation.init(dim, 42, RotationType::FhtKac);
    StreamingIvfPostingOptions options;
    options.M = M;
    options.ivf_k = K;
    options.block_rows = 6;
    options.sample_rows = 12;
    options.train_iters = 2;
    write_streaming_ivf_posting_artifacts(input_path, index_dir, rotation, options);

    std::vector<float> rotated((size_t)n * rotation.padded_dim());
    for (int i = 0; i < n; ++i) {
        rotation.apply(rows.data() + (size_t)i * dim,
                       rotated.data() + (size_t)i * rotation.padded_dim());
    }

    {
        PostingLists postings;
        postings.load(index_dir, M);
        for (int m = 0; m < M; ++m) {
            IVFIndex ivf;
            ivf.load(index_dir + "/ivf_subspace_" + std::to_string(m) + ".bin");
            CHECK(ivf.subspace_dim() == sub_dim, "streaming IVF sub_dim mismatch");
            CHECK(ivf.K() == K, "streaming IVF K mismatch");
            CHECK(ivf.subspace_id() == m, "streaming IVF subspace id mismatch");
            CHECK(postings.K(m) == K, "streaming posting K mismatch");

            size_t size_sum = 0;
            for (int k = 0; k < K; ++k) size_sum += ivf.cell_sizes()[(size_t)k];
            CHECK(size_sum == (size_t)n, "streaming IVF cell sizes do not sum to n");

            std::vector<float> sub_all((size_t)n * sub_dim);
            for (int i = 0; i < n; ++i) {
                std::memcpy(sub_all.data() + (size_t)i * sub_dim,
                            rotated.data() + (size_t)i * rotation.padded_dim() + (size_t)m * sub_dim,
                            (size_t)sub_dim * sizeof(float));
            }
            std::vector<int> assignments(n);
            ivf.assign_batch(sub_all.data(), assignments.data(), n);

            for (int k = 0; k < K; ++k) {
                std::vector<int> expected;
                for (int i = 0; i < n; ++i) {
                    if (assignments[(size_t)i] == k) expected.push_back(i);
                }
                CHECK(postings.cell_size(m, k) == expected.size(), "streaming posting cell size mismatch");
                const uint32_t* ids = postings.cell_data(m, k);
                for (size_t j = 0; j < expected.size(); ++j) {
                    CHECK(ids[j] == (uint32_t)expected[j], "streaming posting ids mismatch");
                }
            }
        }
    }

    for (int m = 0; m < M; ++m) {
        std::remove((index_dir + "/ivf_subspace_" + std::to_string(m) + ".bin").c_str());
        std::remove((index_dir + "/posting_subspace_" + std::to_string(m) + ".bin").c_str());
    }
    rmdir(index_dir.c_str());
    std::remove(input_path.c_str());
    TEST("Streaming IVF postings loadable"); PASS();
}

