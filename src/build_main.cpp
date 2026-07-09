#include "build/index_builder.h"
#include "core/common.h"
#include <iostream>
#include <string>

using namespace score;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: score_build <data.fvecs> <index_dir> [options]" << std::endl;
        std::cerr << "Options:" << std::endl;
        std::cerr << "  -M <int>     Number of subspaces (default: 16)" << std::endl;
        std::cerr << "  -rabitq_k <int>  RaBitQ clusters (default: 200)" << std::endl;
        std::cerr << "  -ivf_k <int> K centroids per subspace for IVF (default: 2500)" << std::endl;
        std::cerr << "  -rotate_subspace  Apply rotation before subspace partitioning" << std::endl;
        std::cerr << "  -subspace_force_raw  Force raw vectors for subspace partitioning only" << std::endl;
        std::cerr << "  -pid_reorder   Reorder point IDs for better posting locality" << std::endl;
        std::cerr << "  -no_pid_reorder  Disable PID reordering for this build" << std::endl;
        std::cerr << "  -pid_reorder_cell_key_depth <int>  Number of subspace cell IDs in reorder key (default: 2)" << std::endl;
        std::cerr << "  -pid_reorder_key_mode <cluster-first|cell-first|interleave>  Reorder key mode (default: cluster-first)" << std::endl;
        std::cerr << "  -pid_reorder_subspaces <csv>  Comma-separated subspace order for reorder key" << std::endl;
        return 1;
    }

    std::string data_path = argv[1];
    std::string index_dir = argv[2];

    IndexConfig config;
    config.index_path = index_dir;

    // Parse options
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-M" && i + 1 < argc) config.num_subspaces = std::stoi(argv[++i]);
        else if (arg == "-rabitq_k" && i + 1 < argc) config.rabitq_k = std::stoi(argv[++i]);
        else if (arg == "-ivf_k" && i + 1 < argc) config.ivf_k = std::stoi(argv[++i]);
        else if (arg == "-rotate_subspace") config.rotate_subspace = true;
        else if (arg == "-subspace_force_raw") config.subspace_force_raw = true;
        else if (arg == "-rabitq_seed" && i + 1 < argc) config.rabitq_seed = std::stoi(argv[++i]);
        else if (arg == "-pid_reorder") config.pid_reorder = true;
        else if (arg == "-no_pid_reorder") config.pid_reorder = false;
        else if (arg == "-pid_reorder_cell_key_depth" && i + 1 < argc) config.pid_reorder_cell_key_depth = std::stoi(argv[++i]);
        else if (arg == "-pid_reorder_key_mode" && i + 1 < argc) {
            std::string mode = argv[++i];
            if (mode == "cluster-first") config.pid_reorder_key_mode = 0;
            else if (mode == "cell-first") config.pid_reorder_key_mode = 1;
            else if (mode == "interleave") config.pid_reorder_key_mode = 2;
        }
        else if (arg == "-pid_reorder_subspaces" && i + 1 < argc) {
            std::string csv = argv[++i];
            config.pid_reorder_subspaces.clear();
            size_t start = 0;
            while (start < csv.size()) {
                size_t comma = csv.find(',', start);
                std::string tok = csv.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
                if (!tok.empty()) config.pid_reorder_subspaces.push_back(std::stoi(tok));
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
        }
        else {
            std::cerr << "[Build] Error: unknown option: " << arg << std::endl;
            return 1;
        }
    }

    if (config.rotate_subspace && config.subspace_force_raw) {
        std::cerr << "[Build] Error: -rotate_subspace and -subspace_force_raw are mutually exclusive." << std::endl;
        return 1;
    }

    // Load data
    std::cout << "Loading data from " << data_path << "..." << std::endl;
    int n, dim;
    std::vector<float> data = load_fvecs(data_path, n, dim);
    std::cout << "Loaded " << n << " vectors, dim=" << dim << std::endl;

    // Build index
    IndexBuilder builder;
    builder.build(data, n, dim, config);

    std::cout << "Index built successfully at " << index_dir << std::endl;
    return 0;
}
