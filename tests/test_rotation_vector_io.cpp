#include "test_macros.h"

void test_rotation() {
    std::cout << "[Rotation]" << std::endl;
    const int dim = 128;
    RandomRotation rot;
    rot.init(dim, 42);

    // Test: normalized Hadamard is orthogonal — preserves distance exactly
    // ||H @ x - H @ y||² = ||x - y||²  (H includes 1/√n normalization)
    std::vector<float> x(dim), y(dim), x_rot(dim), y_rot(dim);
    std::mt19937 rng(123);
    std::normal_distribution<float> dist_n(0, 1);
    for (int i = 0; i < dim; i++) {
        x[i] = dist_n(rng); y[i] = dist_n(rng);
    }

    rot.apply(x.data(), x_rot.data());
    rot.apply(y.data(), y_rot.data());

    float dist_xy = 0, dist_rot = 0;
    for (int i = 0; i < dim; i++) {
        float dx = x[i] - y[i];
        dist_xy += dx * dx;
        float dr = x_rot[i] - y_rot[i];
        dist_rot += dr * dr;
    }
    CHECK(std::abs(dist_rot / dist_xy - 1.0f) < 0.01f, "Hadamard: ||H(x-y)||² != ||x-y||² (not orthogonal)");
    TEST("Hadamard distance preservation"); PASS();

    // Test batch apply matches single apply
    int n = 100;
    std::vector<float> data(n * dim), rotated(n * dim);
    for (int i = 0; i < n * dim; i++) data[i] = dist_n(rng);
    rot.apply_batch(data.data(), rotated.data(), n);

    bool batch_ok = true;
    for (int row = 0; row < std::min(5, n); row++) {
        std::vector<float> single_rot(dim);
        rot.apply(data.data() + (size_t)row * dim, single_rot.data());
        for (int d = 0; d < dim; d++) {
            float diff = std::abs(rotated[(size_t)row * dim + d] - single_rot[d]);
            if (diff > 1e-5f) { batch_ok = false; break; }
        }
        if (!batch_ok) break;
    }
    CHECK(batch_ok, "Batch apply mismatch");
    TEST("Batch apply"); PASS();

    // Test involution: H @ H @ x = x  (normalized H satisfies H² = I)
    std::vector<float> tmp(dim), tmp2(dim);
    rot.apply(x.data(), tmp.data());
    rot.apply(tmp.data(), tmp2.data());
    bool idempotent = true;
    for (int i = 0; i < dim; i++) {
        if (std::abs(tmp2[i] - x[i]) > 1e-4f) { idempotent = false; break; }
    }
    CHECK(idempotent, "Hadamard not involution: H @ H @ x != x");
    TEST("Hadamard idempotence"); PASS();

    const std::string qr_path =
        "/tmp/score_rotation_qr_" + std::to_string(getpid()) + ".bin";
    const std::string qr_header_path = qr_path + ".header";
    RandomRotation qr;
    qr.init(8, 7, RotationType::QR);
    qr.save(qr_path);
    {
        std::ifstream in(qr_path, std::ios::binary);
        std::ofstream out(qr_header_path, std::ios::binary | std::ios::trunc);
        std::vector<char> header(sizeof(int) + sizeof(uint64_t) + sizeof(int) + sizeof(int));
        in.read(header.data(), (std::streamsize)header.size());
        CHECK((bool)in, "QR rotation metadata test must read header");
        out.write(header.data(), (std::streamsize)header.size());
        CHECK((bool)out, "QR rotation metadata test must write header");
    }
    RandomRotation::Metadata qr_meta = RandomRotation::load_metadata(qr_header_path);
    CHECK(qr_meta.dim == 8 && qr_meta.padded_dim == 8 && qr_meta.type == RotationType::QR,
          "rotation metadata should read QR header without payload");
    bool rejected_short_qr_payload = false;
    try {
        RandomRotation full_qr;
        full_qr.load(qr_header_path);
    } catch (const std::runtime_error&) {
        rejected_short_qr_payload = true;
    }
    CHECK(rejected_short_qr_payload, "full QR rotation load should still require payload");
    std::remove(qr_path.c_str());
    std::remove(qr_header_path.c_str());
    TEST("Header-only metadata load"); PASS();
}

void test_vector_block_reader_streams_fvecs() {
    std::cout << "[Streaming block reader fvecs]" << std::endl;

    const std::string path = "/tmp/score_stream_reader_test_" + std::to_string(getpid()) + ".fvecs";
    const int n = 5;
    const int dim = 4;
    std::vector<float> rows((size_t)n * dim);
    for (int i = 0; i < n * dim; ++i) rows[(size_t)i] = (float)(i + 1);
    write_test_fvecs(path, rows, n, dim);

    VectorBlockReader reader(path);
    CHECK(reader.n() == n, "fvecs reader n mismatch");
    CHECK(reader.dim() == dim, "fvecs reader dim mismatch");
    CHECK(reader.format_tag() == 1, "fvecs reader format mismatch");

    std::vector<float> block;
    int rows0 = reader.read_block(2, block);
    CHECK(rows0 == 2, "first fvecs block size mismatch");
    CHECK(std::equal(block.begin(), block.end(), rows.begin()), "first fvecs block payload mismatch");

    int rows1 = reader.read_block(2, block);
    CHECK(rows1 == 2, "second fvecs block size mismatch");
    CHECK(std::equal(block.begin(), block.end(), rows.begin() + 8), "second fvecs block payload mismatch");

    int rows2 = reader.read_block(2, block);
    CHECK(rows2 == 1, "third fvecs block size mismatch");
    CHECK(std::equal(block.begin(), block.end(), rows.begin() + 16), "third fvecs block payload mismatch");
    CHECK(reader.read_block(2, block) == 0, "fvecs reader should reach EOF");

    std::remove(path.c_str());
    TEST("Fvecs block reader preserves order"); PASS();
}

void test_vector_block_reader_streams_fbin() {
    std::cout << "[Streaming block reader fbin]" << std::endl;

    const std::string path = "/tmp/score_stream_reader_test_" + std::to_string(getpid()) + ".fbin";
    const int n = 4;
    const int dim = 5;
    std::vector<float> rows((size_t)n * dim);
    for (int i = 0; i < n * dim; ++i) rows[(size_t)i] = (float)(100 + i);
    write_test_fbin(path, rows, n, dim);

    VectorBlockReader reader(path);
    CHECK(reader.n() == n, "fbin reader n mismatch");
    CHECK(reader.dim() == dim, "fbin reader dim mismatch");
    CHECK(reader.format_tag() == 0, "fbin reader format mismatch");

    std::vector<float> block;
    int rows0 = reader.read_block(3, block);
    CHECK(rows0 == 3, "first fbin block size mismatch");
    CHECK(std::equal(block.begin(), block.end(), rows.begin()), "first fbin block payload mismatch");

    int rows1 = reader.read_block(3, block);
    CHECK(rows1 == 1, "second fbin block size mismatch");
    CHECK(std::equal(block.begin(), block.end(), rows.begin() + 15), "second fbin block payload mismatch");
    CHECK(reader.read_block(3, block) == 0, "fbin reader should reach EOF");

    std::remove(path.c_str());
    TEST("Fbin block reader preserves order"); PASS();
}

void test_prefetched_vector_block_reader_streams_fbin() {
    std::cout << "[Streaming prefetched block reader fbin]" << std::endl;

    const std::string path = "/tmp/score_prefetch_reader_test_" + std::to_string(getpid()) + ".fbin";
    const int n = 7;
    const int dim = 6;
    std::vector<float> rows((size_t)n * dim);
    for (int i = 0; i < n * dim; ++i) rows[(size_t)i] = (float)(300 + i);
    write_test_fbin(path, rows, n, dim);

    PrefetchedVectorBlockReader reader(path, 3);
    CHECK(reader.n() == n, "prefetched reader n mismatch");
    CHECK(reader.dim() == dim, "prefetched reader dim mismatch");
    CHECK(reader.format_tag() == 0, "prefetched reader format mismatch");

    auto block0 = reader.next();
    CHECK(block0.rows == 3, "first prefetched block size mismatch");
    CHECK(block0.data != nullptr, "first prefetched block data missing");
    CHECK(std::equal(block0.data->begin(), block0.data->end(), rows.begin()), "first prefetched block payload mismatch");

    auto block1 = reader.next();
    CHECK(block1.rows == 3, "second prefetched block size mismatch");
    CHECK(block1.data != nullptr, "second prefetched block data missing");
    CHECK(std::equal(block1.data->begin(), block1.data->end(), rows.begin() + 18), "second prefetched block payload mismatch");

    auto block2 = reader.next();
    CHECK(block2.rows == 1, "third prefetched block size mismatch");
    CHECK(block2.data != nullptr, "third prefetched block data missing");
    CHECK(std::equal(block2.data->begin(), block2.data->end(), rows.begin() + 36), "third prefetched block payload mismatch");

    auto eof = reader.next();
    CHECK(eof.rows == 0, "prefetched reader should reach EOF");

    std::remove(path.c_str());
    TEST("Prefetched fbin block reader preserves order"); PASS();
}

void test_prefetched_vector_block_reader_explicit_start_and_early_destroy() {
    std::cout << "[Streaming prefetched block reader explicit start]" << std::endl;

    const std::string path = "/tmp/score_prefetch_reader_start_test_" + std::to_string(getpid()) + ".fbin";
    const int n = 5;
    const int dim = 4;
    std::vector<float> rows((size_t)n * dim);
    for (int i = 0; i < n * dim; ++i) rows[(size_t)i] = (float)(500 + i);
    write_test_fbin(path, rows, n, dim);

    {
        PrefetchedVectorBlockReader reader(path, 2);
        reader.start();
        auto block0 = reader.next();
        CHECK(block0.rows == 2, "explicit-start prefetched first block size mismatch");
        CHECK(block0.data != nullptr, "explicit-start prefetched first block data missing");
        CHECK(std::equal(block0.data->begin(), block0.data->end(), rows.begin()),
              "explicit-start prefetched first block payload mismatch");
    }

    std::remove(path.c_str());
    TEST("Prefetched fbin block reader supports explicit start and early destroy"); PASS();
}

void test_prefetched_vector_block_reader_propagates_worker_exception() {
    std::cout << "[Streaming prefetched block reader exception propagation]" << std::endl;

    const std::string path = "/tmp/score_prefetch_reader_bad_fvecs_" + std::to_string(getpid()) + ".fvecs";
    const int dim = 4;
    std::vector<float> row((size_t)dim, 1.0f);
    {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&dim), sizeof(int));
        out.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
        int bad_dim = dim + 1;
        out.write(reinterpret_cast<const char*>(&bad_dim), sizeof(int));
        out.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
    }

    bool threw = false;
    try {
        PrefetchedVectorBlockReader reader(path, 2);
        (void)reader.next();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    std::remove(path.c_str());
    CHECK(threw, "prefetched reader should propagate worker read exceptions");
    TEST("Prefetched reader propagates worker exceptions"); PASS();
}

void test_prefetched_vector_block_reader_rejects_double_start() {
    std::cout << "[Streaming prefetched block reader double start]" << std::endl;

    const std::string path = "/tmp/score_prefetch_reader_double_start_" + std::to_string(getpid()) + ".fbin";
    const int n = 2;
    const int dim = 4;
    std::vector<float> rows((size_t)n * dim);
    for (int i = 0; i < n * dim; ++i) rows[(size_t)i] = (float)(700 + i);
    write_test_fbin(path, rows, n, dim);

    bool threw = false;
    {
        PrefetchedVectorBlockReader reader(path, 1);
        reader.start();
        try {
            reader.start();
        } catch (const std::runtime_error&) {
            threw = true;
        }
    }
    std::remove(path.c_str());
    CHECK(threw, "prefetched reader should reject a second start call");
    TEST("Prefetched reader rejects double start"); PASS();
}

void test_prefetched_vector_block_reader_partial_final_block_eof() {
    std::cout << "[Streaming prefetched block reader partial final EOF]" << std::endl;

    const std::string path = "/tmp/score_prefetch_reader_partial_eof_" + std::to_string(getpid()) + ".fbin";
    const int n = 3;
    const int dim = 4;
    std::vector<float> rows((size_t)n * dim);
    for (int i = 0; i < n * dim; ++i) rows[(size_t)i] = (float)(900 + i);
    write_test_fbin(path, rows, n, dim);

    PrefetchedVectorBlockReader reader(path, 2);
    auto block0 = reader.next();
    CHECK(block0.rows == 2, "partial-EOF first block size mismatch");
    auto block1 = reader.next();
    CHECK(block1.rows == 1, "partial-EOF final block size mismatch");
    CHECK(block1.data != nullptr, "partial-EOF final block data missing");
    CHECK(std::equal(block1.data->begin(), block1.data->end(), rows.begin() + 8),
          "partial-EOF final block payload mismatch");
    auto eof0 = reader.next();
    auto eof1 = reader.next();
    CHECK(eof0.rows == 0 && eof0.data == nullptr, "partial-EOF first EOF block mismatch");
    CHECK(eof1.rows == 0 && eof1.data == nullptr, "partial-EOF repeated EOF block mismatch");

    std::remove(path.c_str());
    TEST("Prefetched reader handles partial final block EOF"); PASS();
}

void test_ambiguous_fbin_detects_as_fbin() {
    std::cout << "[Streaming ambiguous fbin detection]" << std::endl;

    const std::string path = "/tmp/score_stream_ambiguous_" + std::to_string(getpid()) + ".fbin";
    const int n = 4;
    const int dim = 8;
    std::vector<float> rows((size_t)n * dim);
    for (int i = 0; i < n * dim; ++i) rows[(size_t)i] = (float)(200 + i);
    write_test_fbin(path, rows, n, dim);

    auto [tag, detected_n, detected_dim] = detect_vector_format(path);
    CHECK(tag == 0, "ambiguous .fbin should be detected as fbin by extension");
    CHECK(detected_n == n, "ambiguous .fbin n mismatch");
    CHECK(detected_dim == dim, "ambiguous .fbin dim mismatch");

    VectorBlockReader reader(path);
    std::vector<float> block;
    int rows0 = reader.read_block(10, block);
    CHECK(rows0 == n, "ambiguous .fbin reader row count mismatch");
    CHECK(block == rows, "ambiguous .fbin reader payload mismatch");

    std::remove(path.c_str());
    TEST("Ambiguous fbin stays fbin"); PASS();
}

