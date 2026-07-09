#include "test_macros.h"

void test_pid_reorder_counting_matches_legacy() {
    std::cout << "[PID reorder]" << std::endl;

    auto check_perm_valid = [](const std::vector<uint32_t>& perm) -> bool {
        std::vector<uint32_t> sorted = perm;
        std::sort(sorted.begin(), sorted.end());
        for (size_t i = 0; i < sorted.size(); ++i) {
            if (sorted[i] != (uint32_t)i) return false;
        }
        return true;
    };

    struct SyntheticCase {
        std::string name;
        int n;
        int M;
        std::vector<uint32_t> assignments;
        std::vector<size_t> cells_per_subspace;
        std::vector<int> submajor_cell_ids;
    };

    std::vector<SyntheticCase> cases;
    {
        SyntheticCase c;
        c.name = "mixed-ties";
        c.n = 97;
        c.M = 4;
        c.assignments.resize((size_t)c.n);
        c.cells_per_subspace = {7, 5, 9, 3};
        c.submajor_cell_ids.resize((size_t)c.M * c.n);
        for (int i = 0; i < c.n; ++i) {
            c.assignments[(size_t)i] = (uint32_t)((i * 7 + i / 3 + 2) % 5);
        }
        for (int m = 0; m < c.M; ++m) {
            for (int i = 0; i < c.n; ++i) {
                c.submajor_cell_ids[(size_t)m * c.n + i] =
                    (i * (m + 3) + m * 7 + i / 5) % (int)c.cells_per_subspace[(size_t)m];
            }
        }
        cases.push_back(c);
    }
    {
        SyntheticCase c;
        c.name = "all-equal";
        c.n = 31;
        c.M = 4;
        c.assignments.assign((size_t)c.n, 2u);
        c.cells_per_subspace = {4, 4, 4, 4};
        c.submajor_cell_ids.assign((size_t)c.M * c.n, 1);
        cases.push_back(c);
    }
    {
        SyntheticCase c;
        c.name = "sparse-buckets";
        c.n = 83;
        c.M = 4;
        c.assignments.resize((size_t)c.n);
        c.cells_per_subspace = {11, 13, 17, 19};
        c.submajor_cell_ids.resize((size_t)c.M * c.n);
        const uint32_t sparse_assignments[] = {0u, 5u, 11u};
        for (int i = 0; i < c.n; ++i) {
            c.assignments[(size_t)i] = sparse_assignments[(i * 5 + i / 7) % 3];
        }
        for (int m = 0; m < c.M; ++m) {
            for (int i = 0; i < c.n; ++i) {
                int bucket = ((i * (m + 1) + 2 * m) % 4) * 2;
                c.submajor_cell_ids[(size_t)m * c.n + i] =
                    bucket % (int)c.cells_per_subspace[(size_t)m];
            }
        }
        cases.push_back(c);
    }

    const std::vector<int> modes = {0, 1, 2};
    const std::vector<int> depths = {-1, 0, 1, 4, 5};
    const std::vector<std::vector<int>> subspace_orders = {
        {},
        {0, 1, 2, 3},
        {3, 2, 1, 0},
        {2, 0, 3, 1},
        {1, 1, 2, 0},
    };

    for (const auto& c : cases) {
        for (int mode : modes) {
            for (int depth : depths) {
                for (const auto& order : subspace_orders) {
                    IndexConfig config;
                    config.pid_reorder_key_mode = mode;
                    config.pid_reorder_cell_key_depth = depth;
                    config.pid_reorder_subspaces = order;

                    std::vector<uint32_t> legacy = impl::build_pid_reorder_perm_legacy(
                        c.assignments, c.submajor_cell_ids.data(), c.n, c.M, config);
                    std::vector<uint32_t> counting = impl::build_pid_reorder_perm_counting(
                        c.assignments, c.submajor_cell_ids.data(), c.n, c.M,
                        c.cells_per_subspace, config);

                    CHECK(legacy == counting,
                          "Counting reorder mismatch for " + c.name +
                          " mode=" + std::to_string(mode) +
                          " depth=" + std::to_string(depth));
                    CHECK(check_perm_valid(counting),
                          "Counting reorder is not a permutation for " + c.name);
                    if (c.name == "all-equal") {
                        for (size_t i = 0; i < counting.size(); ++i) {
                            CHECK(counting[i] == (uint32_t)i,
                                  "All-equal reorder must preserve original-pid tie order");
                        }
                    }
                    if (mode == 2 && depth <= 0) {
                        for (size_t i = 0; i < counting.size(); ++i) {
                            CHECK(counting[i] == (uint32_t)i,
                                  "Interleave reorder with depth<=0 must be identity");
                        }
                    }
                }
            }
        }
    }

    TEST("Counting permutation matches legacy comparator"); PASS();
}

void test_pid_reorder_artifact_contract_matches_legacy() {
    std::cout << "[PID reorder artifact contract]" << std::endl;

    const int n = 29;
    const int M = 3;
    const int verify_dim = 5;
    const int bin_bytes = 4;

    std::vector<uint32_t> assignments((size_t)n);
    for (int i = 0; i < n; ++i) {
        assignments[(size_t)i] = (uint32_t)((i * 5 + i / 4) % 6);
    }

    std::vector<size_t> cells_per_subspace = {5, 7, 4};
    std::vector<int> submajor_cell_ids((size_t)M * n);
    for (int m = 0; m < M; ++m) {
        for (int i = 0; i < n; ++i) {
            submajor_cell_ids[(size_t)m * n + i] =
                (i * (m + 2) + m + i / 3) % (int)cells_per_subspace[(size_t)m];
        }
    }

    IndexConfig config;
    config.pid_reorder_key_mode = 1;
    config.pid_reorder_cell_key_depth = 2;
    config.pid_reorder_subspaces = {2, 0};

    std::vector<uint32_t> legacy = impl::build_pid_reorder_perm_legacy(
        assignments, submajor_cell_ids.data(), n, M, config);
    std::vector<uint32_t> counting = impl::build_pid_reorder_perm_counting(
        assignments, submajor_cell_ids.data(), n, M, cells_per_subspace, config);
    CHECK(legacy == counting, "Artifact contract test requires identical permutations");

    std::vector<float> verify_orig((size_t)n * verify_dim);
    for (int i = 0; i < n; ++i) {
        for (int d = 0; d < verify_dim; ++d) {
            verify_orig[(size_t)i * verify_dim + d] = (float)(1000 * i + d);
        }
    }

    std::vector<uint8_t> bin_orig((size_t)n * bin_bytes);
    for (int i = 0; i < n; ++i) {
        for (int b = 0; b < bin_bytes; ++b) {
            bin_orig[(size_t)i * bin_bytes + b] = (uint8_t)((i * 17 + b * 3) & 0xff);
        }
    }

    auto reordered_assignments = [&](const std::vector<uint32_t>& perm) {
        std::vector<uint32_t> out((size_t)n);
        for (int new_pid = 0; new_pid < n; ++new_pid) {
            out[(size_t)new_pid] = assignments[(size_t)perm[(size_t)new_pid]];
        }
        return out;
    };

    auto reordered_verify = [&](const std::vector<uint32_t>& perm) {
        std::vector<float> out((size_t)n * verify_dim);
        for (int new_pid = 0; new_pid < n; ++new_pid) {
            uint32_t orig = perm[(size_t)new_pid];
            std::memcpy(out.data() + (size_t)new_pid * verify_dim,
                        verify_orig.data() + (size_t)orig * verify_dim,
                        (size_t)verify_dim * sizeof(float));
        }
        return out;
    };

    auto reordered_bin = [&](const std::vector<uint32_t>& perm) {
        std::vector<uint8_t> out((size_t)n * bin_bytes);
        for (int new_pid = 0; new_pid < n; ++new_pid) {
            uint32_t orig = perm[(size_t)new_pid];
            std::memcpy(out.data() + (size_t)new_pid * bin_bytes,
                        bin_orig.data() + (size_t)orig * bin_bytes,
                        (size_t)bin_bytes);
        }
        return out;
    };

    auto reordered_postings = [&](const std::vector<uint32_t>& perm) {
        std::vector<std::vector<int>> flat;
        for (int m = 0; m < M; ++m) {
            int K = (int)cells_per_subspace[(size_t)m];
            std::vector<std::vector<int>> cells((size_t)K);
            const int* col = submajor_cell_ids.data() + (size_t)m * n;
            for (int new_pid = 0; new_pid < n; ++new_pid) {
                uint32_t orig = perm[(size_t)new_pid];
                cells[(size_t)col[(size_t)orig]].push_back(new_pid);
            }
            for (auto& cell : cells) {
                flat.push_back(cell);
            }
        }
        return flat;
    };

    CHECK(reordered_assignments(legacy) == reordered_assignments(counting),
          "Reordered assignments differ between legacy and counting perms");
    CHECK(reordered_verify(legacy) == reordered_verify(counting),
          "Reordered verify rows differ between legacy and counting perms");
    CHECK(reordered_bin(legacy) == reordered_bin(counting),
          "Reordered bin rows differ between legacy and counting perms");
    CHECK(reordered_postings(legacy) == reordered_postings(counting),
          "Reordered posting IDs differ between legacy and counting perms");

    const auto postings = reordered_postings(counting);
    size_t flat_idx = 0;
    for (int m = 0; m < M; ++m) {
        const int* col = submajor_cell_ids.data() + (size_t)m * n;
        for (int cell = 0; cell < (int)cells_per_subspace[(size_t)m]; ++cell, ++flat_idx) {
            for (int new_pid : postings[flat_idx]) {
                uint32_t orig = counting[(size_t)new_pid];
                CHECK(col[(size_t)orig] == cell,
                      "Posting new_pid must map through perm_to_orig before cell lookup");
            }
        }
    }

    TEST("Counting permutation preserves reordered artifact contract"); PASS();
}

void test_pid_reorder_disk_staging_matches_counting() {
    std::cout << "[PID reorder disk staging]" << std::endl;

    auto write_assignments_file = [](const std::string& path, const std::vector<uint32_t>& assignments) {
        std::ofstream out(path, std::ios::binary);
        int n = (int)assignments.size();
        out.write(reinterpret_cast<const char*>(&n), sizeof(int));
        out.write(reinterpret_cast<const char*>(assignments.data()), assignments.size() * sizeof(uint32_t));
    };
    auto write_cell_file = [](const std::string& path, const std::vector<int>& cells) {
        std::ofstream out(path, std::ios::binary);
        std::vector<uint16_t> encoded(cells.size());
        for (size_t i = 0; i < cells.size(); ++i) encoded[i] = (uint16_t)cells[i];
        out.write(reinterpret_cast<const char*>(encoded.data()), encoded.size() * sizeof(uint16_t));
    };
    auto read_uint32_file = [](const std::string& path, int n) {
        std::ifstream in(path, std::ios::binary);
        std::vector<uint32_t> out((size_t)n);
        in.read(reinterpret_cast<char*>(out.data()), (size_t)n * sizeof(uint32_t));
        return out;
    };

    const int n = 73;
    const int M = 4;
    const std::vector<size_t> cells_per_subspace = {7, 5, 9, 11};
    std::vector<uint32_t> assignments((size_t)n);
    std::vector<int> submajor_cell_ids((size_t)M * n);
    for (int i = 0; i < n; ++i) {
        assignments[(size_t)i] = (uint32_t)((i * 11 + i / 6 + 3) % 8);
    }
    for (int m = 0; m < M; ++m) {
        for (int i = 0; i < n; ++i) {
            submajor_cell_ids[(size_t)m * n + i] =
                (i * (m + 4) + 2 * m + i / 4) % (int)cells_per_subspace[(size_t)m];
        }
    }

    const std::string prefix = "/tmp/score_pid_disk_" + std::to_string(getpid()) + "_";
    const std::string assignment_path = prefix + "assignments.bin";
    std::vector<std::string> cell_paths((size_t)M);
    write_assignments_file(assignment_path, assignments);
    for (int m = 0; m < M; ++m) {
        cell_paths[(size_t)m] = prefix + "cell_" + std::to_string(m) + ".tmp";
        std::vector<int> cells((size_t)n);
        for (int i = 0; i < n; ++i) cells[(size_t)i] = submajor_cell_ids[(size_t)m * n + i];
        write_cell_file(cell_paths[(size_t)m], cells);
    }

    const std::vector<int> modes = {0, 1, 2};
    const std::vector<int> depths = {-1, 0, 1, 4, 5};
    const std::vector<std::vector<int>> orders = {{}, {3, 2, 1, 0}, {1, 1, 2, 0}};
    int case_id = 0;
    for (int mode : modes) {
        for (int depth : depths) {
            for (const auto& order : orders) {
                IndexConfig config;
                config.pid_reorder_key_mode = mode;
                config.pid_reorder_cell_key_depth = depth;
                config.pid_reorder_subspaces = order;
                std::vector<uint32_t> expected = impl::build_pid_reorder_perm_counting(
                    assignments, submajor_cell_ids.data(), n, M, cells_per_subspace, config);

                const std::string perm_path = prefix + "perm_" + std::to_string(case_id) + ".tmp";
                const std::string scratch_path = prefix + "scratch_" + std::to_string(case_id) + ".tmp";
                const std::string inverse_path = prefix + "inverse_" + std::to_string(case_id) + ".tmp";
                std::remove(perm_path.c_str());
                std::remove(scratch_path.c_str());
                std::remove(inverse_path.c_str());

                impl::DiskPidReorderOptions options;
                options.n = n;
                options.M = M;
                options.assignment_key_range = 8;
                options.cells_per_subspace = cells_per_subspace;
                options.subspace_assignment_paths = cell_paths;
                options.rabitq_assignments_path = assignment_path;
                options.perm_to_orig_path = perm_path;
                options.scratch_perm_path = scratch_path;
                options.orig_to_new_path = inverse_path;
                options.config = config;
                options.io_rows = 13;
                options.bucket_buffer_rows = 3;
                impl::build_pid_reorder_perm_disk_staging(options);

                std::vector<uint32_t> actual = read_uint32_file(perm_path, n);
                CHECK(actual == expected,
                      "Disk reorder permutation mismatch mode=" + std::to_string(mode) +
                      " depth=" + std::to_string(depth));
                std::vector<uint32_t> inverse = read_uint32_file(inverse_path, n);
                for (int new_pid = 0; new_pid < n; ++new_pid) {
                    uint32_t orig = actual[(size_t)new_pid];
                    CHECK(inverse[(size_t)orig] == (uint32_t)new_pid,
                          "Disk reorder inverse map mismatch");
                }

                std::remove(perm_path.c_str());
                std::remove(scratch_path.c_str());
                std::remove(inverse_path.c_str());
                ++case_id;
            }
        }
    }

    for (const std::string& path : cell_paths) std::remove(path.c_str());
    std::remove(assignment_path.c_str());
    TEST("Disk staging permutation matches in-memory counting"); PASS();
}

void test_pid_reorder_disk_staging_rejects_bad_inputs() {
    std::cout << "[PID reorder disk staging errors]" << std::endl;

    auto write_assignments_file = [](const std::string& path, const std::vector<uint32_t>& assignments) {
        std::ofstream out(path, std::ios::binary);
        int n = (int)assignments.size();
        out.write(reinterpret_cast<const char*>(&n), sizeof(int));
        out.write(reinterpret_cast<const char*>(assignments.data()), assignments.size() * sizeof(uint32_t));
    };
    auto write_cell_file = [](const std::string& path, const std::vector<int>& cells) {
        std::ofstream out(path, std::ios::binary);
        std::vector<uint16_t> encoded(cells.size());
        for (size_t i = 0; i < cells.size(); ++i) encoded[i] = (uint16_t)cells[i];
        out.write(reinterpret_cast<const char*>(encoded.data()), encoded.size() * sizeof(uint16_t));
    };

    const int n = 6;
    const int M = 2;
    const std::string prefix = "/tmp/score_pid_disk_bad_" + std::to_string(getpid()) + "_";
    const std::string assignment_path = prefix + "assignments.bin";
    const std::string bad_assignment_path = prefix + "bad_assignments.bin";
    const std::string wrong_n_assignment_path = prefix + "wrong_n_assignments.bin";
    const std::string cell0_path = prefix + "cell0.tmp";
    const std::string cell1_path = prefix + "cell1.tmp";
    const std::string bad_cell_path = prefix + "bad_cell.tmp";
    write_assignments_file(assignment_path, {0, 1, 2, 0, 1, 2});
    write_assignments_file(bad_assignment_path, {0, 1, 7, 0, 1, 2});
    write_assignments_file(wrong_n_assignment_path, {0, 1, 2, 0, 1});
    write_cell_file(cell0_path, {0, 1, 2, 0, 1, 2});
    write_cell_file(cell1_path, {0, 1, 0, 1, 0, 1});
    write_cell_file(bad_cell_path, {0, 1, -1, 0, 1, 0});

    auto make_options = [&](const std::string& assignments) {
        impl::DiskPidReorderOptions options;
        options.n = n;
        options.M = M;
        options.assignment_key_range = 3;
        options.cells_per_subspace = {3, 2};
        options.subspace_assignment_paths = {cell0_path, cell1_path};
        options.rabitq_assignments_path = assignments;
        options.perm_to_orig_path = prefix + "perm.tmp";
        options.scratch_perm_path = prefix + "scratch.tmp";
        options.config.pid_reorder_key_mode = 0;
        options.config.pid_reorder_cell_key_depth = 1;
        options.config.pid_reorder_subspaces = {1};
        options.io_rows = 2;
        options.bucket_buffer_rows = 2;
        std::remove(options.perm_to_orig_path.c_str());
        std::remove(options.scratch_perm_path.c_str());
        return options;
    };

    bool rejected_assignment = false;
    try {
        impl::build_pid_reorder_perm_disk_staging(make_options(bad_assignment_path));
    } catch (const std::runtime_error&) {
        rejected_assignment = true;
    }
    CHECK(rejected_assignment, "Disk reorder must reject out-of-range assignments");
    CHECK(::access((prefix + "perm.tmp").c_str(), F_OK) != 0,
          "Disk reorder failure must remove partial perm output");
    CHECK(::access((prefix + "scratch.tmp").c_str(), F_OK) != 0,
          "Disk reorder failure must remove partial scratch output");

    bool rejected_header_n = false;
    try {
        impl::build_pid_reorder_perm_disk_staging(make_options(wrong_n_assignment_path));
    } catch (const std::runtime_error&) {
        rejected_header_n = true;
    }
    CHECK(rejected_header_n, "Disk reorder must reject wrong assignment header n");

    impl::DiskPidReorderOptions zero_range = make_options(assignment_path);
    zero_range.assignment_key_range = 0;
    bool rejected_zero_assignment_range = false;
    try {
        impl::build_pid_reorder_perm_disk_staging(zero_range);
    } catch (const std::runtime_error&) {
        rejected_zero_assignment_range = true;
    }
    CHECK(rejected_zero_assignment_range, "Disk reorder must reject zero assignment key range");

    impl::DiskPidReorderOptions invalid_subspace = make_options(assignment_path);
    invalid_subspace.config.pid_reorder_subspaces = {2};
    bool rejected_subspace = false;
    try {
        impl::build_pid_reorder_perm_disk_staging(invalid_subspace);
    } catch (const std::runtime_error&) {
        rejected_subspace = true;
    }
    CHECK(rejected_subspace, "Disk reorder must reject invalid custom subspace id");

    impl::DiskPidReorderOptions negative_cell = make_options(assignment_path);
    negative_cell.subspace_assignment_paths = {cell0_path, bad_cell_path};
    bool rejected_negative_cell = false;
    try {
        impl::build_pid_reorder_perm_disk_staging(negative_cell);
    } catch (const std::runtime_error&) {
        rejected_negative_cell = true;
    }
    CHECK(rejected_negative_cell, "Disk reorder must reject negative cell ids");

    impl::DiskPidReorderOptions preexisting_options = make_options(assignment_path);
    {
        std::ofstream preexisting(preexisting_options.perm_to_orig_path, std::ios::binary);
        preexisting.put('\0');
    }
    bool rejected_preexisting = false;
    try {
        impl::build_pid_reorder_perm_disk_staging(preexisting_options);
    } catch (const std::runtime_error&) {
        rejected_preexisting = true;
    }
    CHECK(rejected_preexisting, "Disk reorder must reject pre-existing output paths");
    std::remove((prefix + "perm.tmp").c_str());

    std::ofstream(cell1_path, std::ios::binary | std::ios::trunc).put('\0');
    bool rejected_size = false;
    try {
        impl::build_pid_reorder_perm_disk_staging(make_options(assignment_path));
    } catch (const std::runtime_error&) {
        rejected_size = true;
    }
    CHECK(rejected_size, "Disk reorder must reject malformed cell file size");

    std::remove((prefix + "perm.tmp").c_str());
    std::remove((prefix + "scratch.tmp").c_str());
    std::remove(assignment_path.c_str());
    std::remove(bad_assignment_path.c_str());
    std::remove(wrong_n_assignment_path.c_str());
    std::remove(cell0_path.c_str());
    std::remove(cell1_path.c_str());
    std::remove(bad_cell_path.c_str());
    TEST("Disk staging rejects malformed inputs"); PASS();
}

void test_reordered_artifact_finalize_from_staging() {
    std::cout << "[PID reorder artifact finalize]" << std::endl;

    auto write_raw_uint32 = [](const std::string& path, const std::vector<uint32_t>& values) {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(uint32_t));
    };
    auto write_uint32_header_file = [](const std::string& path, const std::vector<uint32_t>& values) {
        std::ofstream out(path, std::ios::binary);
        int n = (int)values.size();
        out.write(reinterpret_cast<const char*>(&n), sizeof(int));
        out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(uint32_t));
    };
    auto write_bin_file = [](const std::string& path, const std::vector<uint8_t>& values, int n, int record_bytes) {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&n), sizeof(int));
        out.write(reinterpret_cast<const char*>(&record_bytes), sizeof(int));
        out.write(reinterpret_cast<const char*>(values.data()), values.size());
    };
    auto write_cell_file = [](const std::string& path, const std::vector<int>& values) {
        std::ofstream out(path, std::ios::binary);
        std::vector<uint16_t> encoded(values.size());
        for (size_t i = 0; i < values.size(); ++i) encoded[i] = (uint16_t)values[i];
        out.write(reinterpret_cast<const char*>(encoded.data()), encoded.size() * sizeof(uint16_t));
    };

    const int n = 8;
    const int M = 2;
    const int verify_dim = 4;
    const int bin_bytes = 4;
    const std::vector<uint32_t> perm = {3, 0, 7, 2, 5, 1, 6, 4};
    const std::string prefix = "/tmp/score_reorder_finalize_" + std::to_string(getpid()) + "_";
    const std::string raw_perm_path = prefix + "raw_perm.tmp";
    write_raw_uint32(raw_perm_path, perm);

    std::vector<float> verify((size_t)n * verify_dim);
    for (int i = 0; i < n; ++i) {
        for (int d = 0; d < verify_dim; ++d) verify[(size_t)i * verify_dim + d] = (float)(100 * i + d);
    }
    const std::string verify_in = prefix + "verify_in.fbin";
    write_test_fbin(verify_in, verify, n, verify_dim);

    std::vector<uint32_t> assignments = {4, 3, 2, 1, 0, 4, 3, 2};
    const std::string assign_in = prefix + "assign_in.bin";
    write_uint32_header_file(assign_in, assignments);

    std::vector<uint8_t> bin((size_t)n * bin_bytes);
    for (int i = 0; i < n; ++i) {
        for (int b = 0; b < bin_bytes; ++b) bin[(size_t)i * bin_bytes + b] = (uint8_t)(10 * i + b);
    }
    const std::string bin_in = prefix + "bin_in.bin";
    write_bin_file(bin_in, bin, n, bin_bytes);

    const std::vector<size_t> cells_per_subspace = {3, 2};
    const std::vector<int> cell0 = {0, 1, 2, 0, 1, 2, 0, 1};
    const std::vector<int> cell1 = {1, 0, 1, 0, 1, 0, 1, 0};
    const std::string cell0_path = prefix + "cell0.tmp";
    const std::string cell1_path = prefix + "cell1.tmp";
    write_cell_file(cell0_path, cell0);
    write_cell_file(cell1_path, cell1);

    impl::ReorderedArtifactFinalizeOptions options;
    options.n = n;
    options.M = M;
    options.cells_per_subspace = cells_per_subspace;
    options.subspace_assignment_paths = {cell0_path, cell1_path};
    options.raw_perm_to_orig_path = raw_perm_path;
    options.verify_input_path = verify_in;
    options.assignments_input_path = assign_in;
    options.bin_data_input_path = bin_in;
    options.verify_output_path = prefix + "verify_out.fbin";
    options.assignments_output_path = prefix + "assign_out.bin";
    options.bin_data_output_path = prefix + "bin_out.bin";
    options.posting_output_paths = {prefix + "posting0.bin", prefix + "posting1.bin"};
    options.final_perm_to_orig_path = prefix + "perm_to_orig.bin";
    options.io_rows = 3;
    impl::finalize_reordered_artifacts_from_staging(options);

    int out_n = 0, out_dim = 0;
    std::vector<float> verify_out = load_fbin_payload(options.verify_output_path, out_n, out_dim);
    CHECK(out_n == n && out_dim == verify_dim, "Reordered verify header mismatch");
    for (int new_pid = 0; new_pid < n; ++new_pid) {
        uint32_t orig = perm[(size_t)new_pid];
        for (int d = 0; d < verify_dim; ++d) {
            CHECK(verify_out[(size_t)new_pid * verify_dim + d] == verify[(size_t)orig * verify_dim + d],
                  "Reordered verify payload mismatch");
        }
    }

    std::ifstream assign_out(options.assignments_output_path, std::ios::binary);
    int assign_n = 0;
    assign_out.read(reinterpret_cast<char*>(&assign_n), sizeof(int));
    std::vector<uint32_t> reordered_assignments((size_t)n);
    assign_out.read(reinterpret_cast<char*>(reordered_assignments.data()), (size_t)n * sizeof(uint32_t));
    CHECK(assign_n == n, "Reordered assignment header mismatch");
    for (int new_pid = 0; new_pid < n; ++new_pid) {
        CHECK(reordered_assignments[(size_t)new_pid] == assignments[(size_t)perm[(size_t)new_pid]],
              "Reordered assignment payload mismatch");
    }

    std::ifstream bin_out_file(options.bin_data_output_path, std::ios::binary);
    int bin_n = 0, out_bin_bytes = 0;
    bin_out_file.read(reinterpret_cast<char*>(&bin_n), sizeof(int));
    bin_out_file.read(reinterpret_cast<char*>(&out_bin_bytes), sizeof(int));
    std::vector<uint8_t> bin_out((size_t)n * bin_bytes);
    bin_out_file.read(reinterpret_cast<char*>(bin_out.data()), bin_out.size());
    CHECK(bin_n == n && out_bin_bytes == bin_bytes, "Reordered bin header mismatch");
    for (int new_pid = 0; new_pid < n; ++new_pid) {
        uint32_t orig = perm[(size_t)new_pid];
        for (int b = 0; b < bin_bytes; ++b) {
            CHECK(bin_out[(size_t)new_pid * bin_bytes + b] == bin[(size_t)orig * bin_bytes + b],
                  "Reordered bin payload mismatch");
        }
    }

    auto read_posting_ids = [](const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        int K = 0, total = 0;
        in.read(reinterpret_cast<char*>(&K), sizeof(int));
        in.read(reinterpret_cast<char*>(&total), sizeof(int));
        std::vector<int> offsets((size_t)K + 1);
        std::vector<uint32_t> ids((size_t)total);
        in.read(reinterpret_cast<char*>(offsets.data()), offsets.size() * sizeof(int));
        in.read(reinterpret_cast<char*>(ids.data()), ids.size() * sizeof(uint32_t));
        return std::make_pair(offsets, ids);
    };
    auto posting0 = read_posting_ids(options.posting_output_paths[0]);
    auto posting1 = read_posting_ids(options.posting_output_paths[1]);
    auto expected_posting = [&](const std::vector<int>& cells, int K) {
        std::vector<int> offsets((size_t)K + 1, 0);
        for (int new_pid = 0; new_pid < n; ++new_pid) offsets[(size_t)cells[(size_t)perm[(size_t)new_pid]] + 1]++;
        for (int k = 0; k < K; ++k) offsets[(size_t)k + 1] += offsets[(size_t)k];
        std::vector<uint32_t> ids((size_t)n);
        std::vector<int> cursors = offsets;
        for (int new_pid = 0; new_pid < n; ++new_pid) {
            int cell = cells[(size_t)perm[(size_t)new_pid]];
            ids[(size_t)cursors[(size_t)cell]++] = (uint32_t)new_pid;
        }
        return std::make_pair(offsets, ids);
    };
    CHECK(posting0 == expected_posting(cell0, 3), "Reordered posting0 mismatch");
    CHECK(posting1 == expected_posting(cell1, 2), "Reordered posting1 mismatch");

    std::ifstream final_perm(options.final_perm_to_orig_path, std::ios::binary);
    int perm_n = 0;
    final_perm.read(reinterpret_cast<char*>(&perm_n), sizeof(int));
    std::vector<uint32_t> final_perm_payload((size_t)n);
    final_perm.read(reinterpret_cast<char*>(final_perm_payload.data()), (size_t)n * sizeof(uint32_t));
    CHECK(perm_n == n && final_perm_payload == perm, "Final perm_to_orig header/payload mismatch");

    std::remove(raw_perm_path.c_str());
    std::remove(verify_in.c_str());
    std::remove(assign_in.c_str());
    std::remove(bin_in.c_str());
    std::remove(cell0_path.c_str());
    std::remove(cell1_path.c_str());
    std::remove(options.verify_output_path.c_str());
    std::remove(options.assignments_output_path.c_str());
    std::remove(options.bin_data_output_path.c_str());
    for (const std::string& path : options.posting_output_paths) std::remove(path.c_str());
    std::remove(options.final_perm_to_orig_path.c_str());
    TEST("Finalize rewrites reordered artifacts from staging"); PASS();
}

void test_reordered_artifact_finalize_failure_hides_final_perm() {
    std::cout << "[PID reorder artifact finalize errors]" << std::endl;

    const int n = 4;
    const std::string prefix = "/tmp/score_reorder_finalize_bad_" + std::to_string(getpid()) + "_";
    const std::string raw_perm = prefix + "raw_perm.tmp";
    {
        std::vector<uint32_t> perm = {0, 1, 2, 3};
        std::ofstream out(raw_perm, std::ios::binary);
        out.write(reinterpret_cast<const char*>(perm.data()), perm.size() * sizeof(uint32_t));
    }
    const std::string bad_verify = prefix + "bad_verify.fbin";
    {
        std::ofstream out(bad_verify, std::ios::binary);
        int bad_n = n + 1, dim = 2, reserved = 0;
        out.write(reinterpret_cast<const char*>(&bad_n), sizeof(int));
        out.write(reinterpret_cast<const char*>(&dim), sizeof(int));
        out.write(reinterpret_cast<const char*>(&reserved), sizeof(int));
    }
    const std::string assign_in = prefix + "assign.bin";
    {
        std::vector<uint32_t> assignments = {0, 1, 0, 1};
        std::ofstream out(assign_in, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&n), sizeof(int));
        out.write(reinterpret_cast<const char*>(assignments.data()), assignments.size() * sizeof(uint32_t));
    }
    const std::string bin_in = prefix + "bin.bin";
    {
        int bin_bytes = 1;
        std::vector<uint8_t> bin((size_t)n, 7);
        std::ofstream out(bin_in, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&n), sizeof(int));
        out.write(reinterpret_cast<const char*>(&bin_bytes), sizeof(int));
        out.write(reinterpret_cast<const char*>(bin.data()), bin.size());
    }
    const std::string cell_path = prefix + "cell.tmp";
    {
        std::vector<uint16_t> cells = {0, 0, 0, 0};
        std::ofstream out(cell_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(cells.data()), cells.size() * sizeof(uint16_t));
    }

    impl::ReorderedArtifactFinalizeOptions options;
    options.n = n;
    options.M = 1;
    options.cells_per_subspace = {1};
    options.subspace_assignment_paths = {cell_path};
    options.raw_perm_to_orig_path = raw_perm;
    options.verify_input_path = bad_verify;
    options.assignments_input_path = assign_in;
    options.bin_data_input_path = bin_in;
    options.verify_output_path = prefix + "verify_out.fbin";
    options.assignments_output_path = prefix + "assign_out.bin";
    options.bin_data_output_path = prefix + "bin_out.bin";
    options.posting_output_paths = {prefix + "posting.bin"};
    options.final_perm_to_orig_path = prefix + "perm_to_orig.bin";
    options.io_rows = 2;

    bool rejected = false;
    try {
        impl::finalize_reordered_artifacts_from_staging(options);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    CHECK(rejected, "Finalize must reject malformed verify header");
    CHECK(::access(options.final_perm_to_orig_path.c_str(), F_OK) != 0,
          "Finalize failure must not expose final perm_to_orig.bin");
    CHECK(::access(options.verify_output_path.c_str(), F_OK) != 0,
          "Finalize failure must clean partial verify output");

    std::remove(raw_perm.c_str());
    std::remove(bad_verify.c_str());
    std::remove(assign_in.c_str());
    std::remove(bin_in.c_str());
    std::remove(cell_path.c_str());
    TEST("Finalize failure hides final perm marker"); PASS();
}

void test_reordered_artifact_finalize_rejects_bad_perm() {
    std::cout << "[PID reorder artifact finalize bad perm]" << std::endl;

    auto write_raw_perm = [](const std::string& path, const std::vector<uint32_t>& perm) {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(perm.data()), perm.size() * sizeof(uint32_t));
    };
    auto write_assignments = [](const std::string& path, int n) {
        std::vector<uint32_t> assignments((size_t)n, 0);
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&n), sizeof(int));
        out.write(reinterpret_cast<const char*>(assignments.data()), assignments.size() * sizeof(uint32_t));
    };
    auto write_bin = [](const std::string& path, int n) {
        int bin_bytes = 1;
        std::vector<uint8_t> bin((size_t)n, 3);
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&n), sizeof(int));
        out.write(reinterpret_cast<const char*>(&bin_bytes), sizeof(int));
        out.write(reinterpret_cast<const char*>(bin.data()), bin.size());
    };
    auto write_cells = [](const std::string& path, int n) {
        std::vector<uint16_t> cells((size_t)n, 0);
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(cells.data()), cells.size() * sizeof(uint16_t));
    };

    const int n = 4;
    const int verify_dim = 4;
    const std::string prefix = "/tmp/score_reorder_finalize_perm_bad_" + std::to_string(getpid()) + "_";
    const std::string verify_in = prefix + "verify.fbin";
    const std::string assign_in = prefix + "assign.bin";
    const std::string bin_in = prefix + "bin.bin";
    const std::string cell_in = prefix + "cell.tmp";
    std::vector<float> verify((size_t)n * verify_dim, 1.0f);
    write_test_fbin(verify_in, verify, n, verify_dim);
    write_assignments(assign_in, n);
    write_bin(bin_in, n);
    write_cells(cell_in, n);

    auto make_options = [&](const std::string& raw_perm, const std::string& tag) {
        impl::ReorderedArtifactFinalizeOptions options;
        options.n = n;
        options.M = 1;
        options.cells_per_subspace = {1};
        options.subspace_assignment_paths = {cell_in};
        options.raw_perm_to_orig_path = raw_perm;
        options.verify_input_path = verify_in;
        options.assignments_input_path = assign_in;
        options.bin_data_input_path = bin_in;
        options.verify_output_path = prefix + tag + "_verify_out.fbin";
        options.assignments_output_path = prefix + tag + "_assign_out.bin";
        options.bin_data_output_path = prefix + tag + "_bin_out.bin";
        options.posting_output_paths = {prefix + tag + "_posting.bin"};
        options.final_perm_to_orig_path = prefix + tag + "_perm_to_orig.bin";
        options.io_rows = 2;
        return options;
    };

    const std::string duplicate_perm = prefix + "duplicate_perm.tmp";
    write_raw_perm(duplicate_perm, {0, 1, 1, 3});
    impl::ReorderedArtifactFinalizeOptions duplicate_options = make_options(duplicate_perm, "dup");
    bool rejected_duplicate = false;
    try {
        impl::finalize_reordered_artifacts_from_staging(duplicate_options);
    } catch (const std::runtime_error&) {
        rejected_duplicate = true;
    }
    CHECK(rejected_duplicate, "Finalize must reject duplicate/missing perm entries");
    CHECK(::access(duplicate_options.final_perm_to_orig_path.c_str(), F_OK) != 0,
          "Bad perm must not expose final perm marker");

    const std::string valid_perm = prefix + "valid_perm.tmp";
    write_raw_perm(valid_perm, {0, 1, 2, 3});
    impl::ReorderedArtifactFinalizeOptions no_inverse_options = make_options(valid_perm, "noinv");
    bool accepted_no_inverse = true;
    try {
        impl::finalize_reordered_artifacts_from_staging(no_inverse_options);
    } catch (const std::runtime_error&) {
        accepted_no_inverse = false;
    }
    CHECK(accepted_no_inverse, "Finalize should not require inverse scratch for default sorted rewrite");
    CHECK(::access(no_inverse_options.final_perm_to_orig_path.c_str(), F_OK) == 0,
          "Valid default finalize should expose final perm marker");

    std::remove(no_inverse_options.verify_output_path.c_str());
    std::remove(no_inverse_options.assignments_output_path.c_str());
    std::remove(no_inverse_options.bin_data_output_path.c_str());
    for (const std::string& path : no_inverse_options.posting_output_paths) std::remove(path.c_str());
    std::remove(no_inverse_options.final_perm_to_orig_path.c_str());

    std::remove(verify_in.c_str());
    std::remove(assign_in.c_str());
    std::remove(bin_in.c_str());
    std::remove(cell_in.c_str());
    std::remove(duplicate_perm.c_str());
    std::remove(valid_perm.c_str());
    TEST("Finalize rejects invalid raw permutations"); PASS();
}

void test_uint16_posting_assignment_boundaries() {
    std::cout << "[PID reorder uint16 assignment boundaries]" << std::endl;

    const std::string prefix = "/tmp/score_uint16_assign_boundary_" + std::to_string(getpid()) + "_";
    const std::string temp_assign = prefix + "cell.tmp";
    const std::string raw_perm = prefix + "perm.tmp";
    const std::string posting_out = prefix + "posting.bin";
    const int n = 3;
    {
        std::vector<uint16_t> cells = {0, 65534, 0};
        std::ofstream out(temp_assign, std::ios::binary);
        out.write(reinterpret_cast<const char*>(cells.data()), cells.size() * sizeof(uint16_t));
    }
    {
        std::vector<uint32_t> perm = {0, 1, 2};
        std::ofstream out(raw_perm, std::ios::binary);
        out.write(reinterpret_cast<const char*>(perm.data()), perm.size() * sizeof(uint32_t));
    }

    bool accepted_limit = true;
    try {
        impl::materialize_reordered_posting_from_temp(temp_assign, posting_out, raw_perm,
                                                        n, 65535, 2);
    } catch (const std::runtime_error&) {
        accepted_limit = false;
    }
    CHECK(accepted_limit, "K=65535 should be accepted for uint16 assignment materialize");
    {
        std::ifstream in(posting_out, std::ios::binary);
        int K = 0, total = 0;
        in.read(reinterpret_cast<char*>(&K), sizeof(int));
        in.read(reinterpret_cast<char*>(&total), sizeof(int));
        std::vector<int> offsets((size_t)K + 1);
        in.read(reinterpret_cast<char*>(offsets.data()), offsets.size() * sizeof(int));
        CHECK(K == 65535 && total == n, "uint16 boundary posting header mismatch");
        CHECK(offsets[0] == 0 && offsets[1] == 2, "cell 0 offset mismatch at uint16 boundary");
        CHECK(offsets[65534] == 2 && offsets[65535] == 3, "max cell offset mismatch at uint16 boundary");
    }
    std::remove(posting_out.c_str());

    bool rejected_overflow = false;
    try {
        impl::materialize_reordered_posting_from_temp(temp_assign, posting_out, raw_perm,
                                                        n, 65536, 2);
    } catch (const std::runtime_error&) {
        rejected_overflow = true;
    }
    CHECK(rejected_overflow, "K=65536 must be rejected for uint16 assignment materialize");

    std::remove(temp_assign.c_str());
    std::remove(raw_perm.c_str());
    std::remove(posting_out.c_str());
    TEST("Uint16 assignment boundary checks"); PASS();
}

void test_preserved_assignment_manifest_rejects_v1() {
    std::cout << "[PID reorder preserved assignment manifest version]" << std::endl;

    const std::string dir = "/tmp/score_manifest_v1_" + std::to_string(getpid());
    mkdir(dir.c_str(), 0755);
    {
        std::ofstream manifest(dir + "/posting_assignments.manifest");
        manifest << "SCORE_POSTING_ASSIGNMENTS_V1\n";
        manifest << "n 1\n";
        manifest << "M 1\n";
        manifest << "subspace 0 K 1 bytes " << sizeof(int) << "\n";
    }

    bool rejected = false;
    try {
        (void)impl::load_preserved_posting_assignments(dir, 1, 1);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    CHECK(rejected, "V1 preserved assignment manifest must be rejected after uint16 V2 switch");

    std::remove((dir + "/posting_assignments.manifest").c_str());
    rmdir(dir.c_str());
    TEST("Preserved assignment manifest rejects V1"); PASS();
}

void test_posting_uint32_ids_load_legacy_and_reject_corrupt() {
    std::cout << "[PostingLists uint32 ids]" << std::endl;

    const std::string dir = "/tmp/score_posting_uint32_ids_" + std::to_string(getpid());
    mkdir(dir.c_str(), 0755);
    const std::string path = dir + "/posting_subspace_0.bin";

    {
        std::ofstream out(path, std::ios::binary);
        int K = 2;
        int total = 3;
        int offsets[] = {0, 2, 3};
        int legacy_ids[] = {0, 2, 1};
        out.write(reinterpret_cast<const char*>(&K), sizeof(int));
        out.write(reinterpret_cast<const char*>(&total), sizeof(int));
        out.write(reinterpret_cast<const char*>(offsets), sizeof(offsets));
        out.write(reinterpret_cast<const char*>(legacy_ids), sizeof(legacy_ids));
    }
    {
        PostingLists postings;
        postings.load(dir, 1);
        postings.validate_point_ids(3);
        CHECK(postings.K(0) == 2, "legacy int32 posting K mismatch");
        CHECK(postings.cell_size(0, 0) == 2 && postings.cell_size(0, 1) == 1,
              "legacy int32 posting cell sizes mismatch");
        const uint32_t* cell0 = postings.cell_data(0, 0);
        const uint32_t* cell1 = postings.cell_data(0, 1);
        CHECK(cell0[0] == 0u && cell0[1] == 2u && cell1[0] == 1u,
              "legacy int32 posting IDs should decode as uint32 IDs");
    }

    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        int K = 1;
        int total = 2;
        int offsets[] = {0, 2};
        uint32_t ids[] = {0u, std::numeric_limits<uint32_t>::max()};
        out.write(reinterpret_cast<const char*>(&K), sizeof(int));
        out.write(reinterpret_cast<const char*>(&total), sizeof(int));
        out.write(reinterpret_cast<const char*>(offsets), sizeof(offsets));
        out.write(reinterpret_cast<const char*>(ids), sizeof(ids));
    }
    bool rejected = false;
    try {
        PostingLists postings;
        postings.load(dir, 1);
        postings.validate_point_ids(2);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    CHECK(rejected, "uint32 posting IDs >= n must be rejected");

    std::remove(path.c_str());
    rmdir(dir.c_str());
    TEST("Posting uint32 IDs load legacy and reject corrupt IDs"); PASS();
}

