#include "query/query_engine.h"
#include "core/common.h"
#include <iostream>
#include <algorithm>

using namespace score;

// Benchmark: sweep parameters and measure recall/QPS/latency
void run_benchmark(const std::string& index_dir, const std::string& query_path,
                   const std::string& gt_path, const IndexConfig& base_config,
                   const std::string& approach_name) {
    if (base_config.num_threads > 0) {
        omp_set_num_threads(base_config.num_threads);
    }

    // Load queries
    int nq, dim;
    std::vector<float> queries = load_fvecs(query_path, nq, dim);

    // Load ground truth
    int gt_n, gt_dim;
    std::vector<int> groundtruth = load_ivecs(gt_path, gt_n, gt_dim);

    // Load index
    IndexConfig config = base_config;
    config.index_path = index_dir;

    QueryEngine engine;
    engine.init(config);
    engine.load_index(index_dir);

    // Search
    Timer timer;
    timer.reset();
    auto results = engine.search_batch(queries.data(), nq);
    double total_ms = timer.elapsed_ms();
    double qps = nq / (total_ms / 1000.0);

    // Compute recall
    int topk = config.topk;
    int hits = 0, total = 0;
    for (int q = 0; q < nq; q++) {
        std::vector<int> ret;
        for (auto& [d, pid] : results[q].topk) ret.push_back(pid);
        int gt_topk = std::min(topk, gt_dim);
        for (int k = 0; k < gt_topk; k++) {
            int gt_id = groundtruth[(size_t)q * gt_dim + k];
            for (int r = 0; r < (int)ret.size(); r++) {
                if (ret[r] == gt_id) { hits++; break; }
            }
            total++;
        }
    }
    float recall = (total > 0) ? (float)hits / total : 0;

    double avg_latency_ms = total_ms / nq;

    // Output TSV line for results.tsv
    char date[32];
    time_t now = time(nullptr);
    strftime(date, sizeof(date), "%Y-%m-%d", localtime(&now));

    std::cout << date << "\t"
              << approach_name << "\t"
              << "SIFT1M\t"
              << engine.num_vectors() << "\t"
              << engine.dim() << "\t"
              << config.num_subspaces << "\t"
              << config.rabitq_k << "\t"
              << config.collision_ratio << "\t"
              << config.candidate_ratio << "\t"
              << recall << "\t"
              << qps << "\t"
              << avg_latency_ms << "\t"
              << "0\t" // RAM GB placeholder
              << "0\t" // build time placeholder
              << std::endl;

    // Human-readable summary
    std::cerr << "\n  [" << approach_name << "]"
              << "\n    Recall@10: " << recall
              << "\n    QPS:       " << qps
              << "\n    Avg query latency: " << avg_latency_ms << "ms"
              << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: score_bench <index_dir> <query.fvecs> <groundtruth.ivecs>" << std::endl;
        return 1;
    }

    std::string index_dir = argv[1];
    std::string query_path = argv[2];
    std::string gt_path = argv[3];

    std::cout << "date\tapproach\tdataset\tN\tD\tM\tK\talpha\tbeta\trecall@10\tqps\tavg_latency_ms\tram_gb\tbuild_s" << std::endl;

    // Sweep: IMI-based (Idea #1) with different candidate ratios
    {
        IndexConfig config;
        config.index_path = index_dir;
        config.rabitq_k = 200;

        std::vector<float> betas = {0.001f, 0.002f, 0.005f, 0.01f, 0.02f};
        for (float beta : betas) {
            config.candidate_ratio = beta;
            run_benchmark(index_dir, query_path, gt_path, config,
                          "IMI_beta" + std::to_string(beta));
        }
    }

    return 0;
}
