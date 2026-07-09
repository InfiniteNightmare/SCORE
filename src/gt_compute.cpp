#include "core/common.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstring>

#if HAS_AOCL_BLAS
#include <cblas.h>
#endif

using namespace score;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: gt_compute <base.fvecs> <query.fvecs> <output.ivecs> [topk]\n";
        std::cerr << "  Computes brute-force top-k nearest neighbors\n";
        std::cerr << "  Uses BLAS sgemm for fast distance computation\n";
        return 1;
    }

    std::string base_path = argv[1];
    std::string query_path = argv[2];
    std::string output_path = argv[3];
    int topk = 100;
    if (argc > 4) topk = std::stoi(argv[4]);

    // Load base vectors
    std::cout << "Loading base from " << base_path << "..." << std::endl;
    int nb, dim;
    std::vector<float> base = load_fvecs(base_path, nb, dim);
    std::cout << "  " << nb << " vectors, dim=" << dim << std::endl;

    // Load queries
    std::cout << "Loading queries from " << query_path << "..." << std::endl;
    int nq, qdim;
    std::vector<float> queries = load_fvecs(query_path, nq, qdim);
    std::cout << "  " << nq << " queries, dim=" << qdim << std::endl;

    if (dim != qdim) {
        std::cerr << "ERROR: dimension mismatch: base=" << dim << " query=" << qdim << std::endl;
        return 1;
    }

    Timer timer;
    timer.reset();

    // Compute ||q - b||² = ||q||² + ||b||² - 2·q·b^T
    // Using BLAS sgemm: C = Q * B^T (nq x nb)
    // Then: dist² = row-wise ||q||² + col-wise ||b||² - 2*C

    // Pre-compute ||b||² for all base vectors
    std::vector<float> base_norms_sq(nb);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nb; i++) {
        base_norms_sq[i] = dot_self(base.data() + (size_t)i * dim, dim);
    }

    // Pre-compute ||q||² for all queries
    std::vector<float> query_norms_sq(nq);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nq; i++) {
        query_norms_sq[i] = dot_self(queries.data() + (size_t)i * dim, dim);
    }

    // Chunked computation to avoid OOM on large datasets
    // Process base in chunks, maintain per-query top-k heap
    const int base_chunk = 500000;  // 500K base vectors per chunk
    int n_chunks = (nb + base_chunk - 1) / base_chunk;
    std::cout << "Computing distances in " << n_chunks << " chunks of "
              << base_chunk << " base vectors..." << std::endl;

    std::vector<int> gt((size_t)nq * topk);
    // Per-query top-k: (dist, id) max-heap
    std::vector<std::vector<std::pair<float, int>>> topk_heaps(nq);
    for (int q = 0; q < nq; q++) {
        topk_heaps[q].reserve(topk + 1);
    }

    for (int chunk = 0; chunk < n_chunks; chunk++) {
        int b_start = chunk * base_chunk;
        int b_end = std::min(b_start + base_chunk, nb);
        int b_count = b_end - b_start;

        // IP matrix for this chunk: nq x b_count
        std::vector<float> ip_chunk((size_t)nq * b_count);

#if HAS_AOCL_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    nq, b_count, dim,
                    1.0f, queries.data(), dim,
                    base.data() + (size_t)b_start * dim, dim,
                    0.0f, ip_chunk.data(), b_count);
#else
        #pragma omp parallel for schedule(dynamic, 4)
        for (int q = 0; q < nq; q++) {
            const float* qv = queries.data() + (size_t)q * dim;
            for (int b = 0; b < b_count; b++) {
                const float* bv = base.data() + (size_t)(b_start + b) * dim;
                float ip = 0;
                for (int d = 0; d < dim; d++) ip += qv[d] * bv[d];
                ip_chunk[(size_t)q * b_count + b] = ip;
            }
        }
#endif

        // Merge chunk results into per-query top-k
        #pragma omp parallel for schedule(dynamic, 4)
        for (int q = 0; q < nq; q++) {
            auto& heap = topk_heaps[q];
            const float* ip_row = ip_chunk.data() + (size_t)q * b_count;
            for (int b = 0; b < b_count; b++) {
                float d2 = query_norms_sq[q] + base_norms_sq[b_start + b] - 2.0f * ip_row[b];
                if ((int)heap.size() < topk) {
                    heap.push_back({d2, b_start + b});
                    if ((int)heap.size() == topk) {
                        std::make_heap(heap.begin(), heap.end());
                    }
                } else if (d2 < heap[0].first) {
                    std::pop_heap(heap.begin(), heap.end());
                    heap.back() = {d2, b_start + b};
                    std::push_heap(heap.begin(), heap.end());
                }
            }
        }

        std::cout << "  chunk " << (chunk+1) << "/" << n_chunks
                  << " (base " << b_start << "-" << b_end << ")" << std::endl;
    }

    double compute_ms = timer.elapsed_ms();
    std::cout << "  Total compute time: " << compute_ms << " ms" << std::endl;

    // Extract sorted top-k from heaps
    std::cout << "Extracting top-" << topk << " results..." << std::endl;
    timer.reset();

    #pragma omp parallel for schedule(static)
    for (int q = 0; q < nq; q++) {
        auto& heap = topk_heaps[q];
        std::sort(heap.begin(), heap.end());
        for (int k = 0; k < topk && k < (int)heap.size(); k++) {
            gt[(size_t)q * topk + k] = heap[k].second;
        }
    }

    double sort_ms = timer.elapsed_ms();
    std::cout << "  Top-k selection in " << sort_ms << " ms" << std::endl;

    // Save as ivecs
    std::ofstream out(output_path, std::ios::binary);
    for (int q = 0; q < nq; q++) {
        out.write(reinterpret_cast<const char*>(&topk), sizeof(int));
        out.write(reinterpret_cast<const char*>(gt.data() + (size_t)q * topk),
                  topk * sizeof(int));
    }

    double total_ms = compute_ms + sort_ms;
    std::cout << "\nDone! Total time: " << total_ms << " ms" << std::endl;
    std::cout << "  Compute: " << compute_ms << " ms (" << (double)nq * nb * dim * 2 / 1e9 / (compute_ms/1000) << " GFLOPS)" << std::endl;
    std::cout << "  Sorting: " << sort_ms << " ms" << std::endl;
    std::cout << "  Output: " << output_path << " (" << nq << " x " << topk << ")" << std::endl;

    // Verify
    std::cout << "\nVerification:" << std::endl;
    for (int q = 0; q < std::min(3, nq); q++) {
        int nn = gt[(size_t)q * topk];
        float dist = 0;
        for (int d = 0; d < dim; d++) {
            float diff = queries[(size_t)q * dim + d] - base[(size_t)nn * dim + d];
            dist += diff * diff;
        }
        std::cout << "  Query " << q << ": NN=" << nn << ", dist=" << dist << std::endl;
    }

    return 0;
}
