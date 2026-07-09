#pragma once

// Shared test infrastructure for the score unit tests: the score includes, the
// pass/fail counters, the TEST/PASS/FAIL/CHECK macros, and `using namespace
// score`. Each per-feature test_*.cpp includes this; tests/test_main.cpp is the
// runner that invokes all test_*() functions.

#include "core/common.h"
#include "build/index_builder.h"
#include "query/query_engine.h"
#include "core/rotation.h"
#include "core/rabitq.h"
#include "query/sc_score.h"
#include "build/streaming_build.h"
#include "update/update_compaction.h"
#include "update/update_delta.h"
#include "update/update_index.h"
#include "update/update_wal.h"
#include <iostream>
#include <cmath>
#include <cassert>
#include <random>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <thread>
#include <sys/stat.h>
#include <unistd.h>

using namespace score;

extern int tests_passed;
extern int tests_failed;

#define TEST(name) std::cout << "  " << name << "... "
#define PASS() do { std::cout << "PASSED" << std::endl; tests_passed++; } while (0)
#define FAIL(msg) do { std::cout << "FAILED: " << msg << std::endl; tests_failed++; } while (0)
#define CHECK(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while (0)

// Shared test helpers (inline so each test_*.cpp can include this header).
inline void write_test_fvecs(const std::string& path, const std::vector<float>& rows, int n, int dim) {
    std::ofstream out(path, std::ios::binary);
    for (int i = 0; i < n; ++i) {
        out.write(reinterpret_cast<const char*>(&dim), sizeof(int));
        out.write(reinterpret_cast<const char*>(rows.data() + (size_t)i * dim),
                  (size_t)dim * sizeof(float));
    }
}

inline void write_test_fbin(const std::string& path, const std::vector<float>& rows, int n, int dim) {
    std::ofstream out(path, std::ios::binary);
    int reserved = 0;
    out.write(reinterpret_cast<const char*>(&n), sizeof(int));
    out.write(reinterpret_cast<const char*>(&dim), sizeof(int));
    out.write(reinterpret_cast<const char*>(&reserved), sizeof(int));
    out.write(reinterpret_cast<const char*>(rows.data()), rows.size() * sizeof(float));
}

inline void copy_test_file(const std::string& src, const std::string& dst) {
    std::ifstream in(src, std::ios::binary);
    std::ofstream out(dst, std::ios::binary);
    out << in.rdbuf();
}

inline void write_valid_update_manifest(const std::string& dir) {
    std::ofstream manifest(dir + "/update_manifest.json");
    manifest << "{\n";
    manifest << "  \"format\": \"score_update_manifest_v1\",\n";
    manifest << "  \"generation\": 17,\n";
    manifest << "  \"base\": {\"path\": \".\", \"base_n\": 100},\n";
    manifest << "  \"pid\": {\"base_n\": 100, \"max_pid\": 112, \"live_count\": 108},\n";
    manifest << "  \"sealed_deltas\": [\n";
    manifest << "    {\"segment_id\": \"00000015\", \"path\": \"deltas/00000015\",";
    manifest << " \"pid_begin\": 100, \"pid_count\": 10, \"live_count\": 8}\n";
    manifest << "  ],\n";
    manifest << "  \"growing\": {\"wal_path\": \"updates/l0.wal\",";
    manifest << " \"checkpoint_path\": \"updates/l0.checkpoint\"},\n";
    manifest << "  \"direct_map\": {\"mode\": \"dense\", \"path\": \"updates/direct_map.bin\",";
    manifest << " \"optional_for_search\": true},\n";
    manifest << "  \"durability\": {\"mode\": \"buffered\", \"last_committed_lsn\": 9182,";
    manifest << " \"last_fsynced_lsn\": 9182}\n";
    manifest << "}\n";
}
