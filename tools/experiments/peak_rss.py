#!/usr/bin/env python3
# ponytail: RSS model for SCORE streaming build (score_stream_build_reorder).
# Sized from actual .resize()/.reserve()/new[] sites in:
#   include/build/streaming_build.h (fused path, lines ~1262-1603)
#   include/core/ivf.h (assign_batch_strided_with_scratch_cap, capped_assign_block_size)
#   external/SuperKMeans/.../hierarchical_superkmeans.h (HSKM scratch)
#   external/RaBitQ-Library/.../rabitq_impl.hpp (per-vec binary_code scratch)
# No runtime deps. `python3 /tmp/rss_calc.py` self-checks against measured points.
import argparse, math, sys

MIB = 1024.0 * 1024.0


def pad_dim(dim):
    # FhtKac padding requirement: multiple of 64.
    return ((dim + 63) // 64) * 64


def block_rows_for_mib(block_mib, dim):
    return (block_mib * 1024 * 1024) // (dim * 4)


def rabitq_record_bytes(padded_dim):
    # bb = sizeof(uint32_t) + BinDataMap<float>::data_bytes(padded_dim)
    # data_bytes = padded_dim/8 (binary) + 3*sizeof(float) (f_add,f_rescale,f_error)
    return 4 + (padded_dim // 8) + 12


def capped_assign_block_rows(K, scratch_cap_bytes):
    k_default = 50000
    bytes_per_row = K * 4
    capped = max(1, scratch_cap_bytes // bytes_per_row)
    return min(k_default, capped)


def peak_rss(M, ivf_k, rabitq_k, sample_rows, block_mib, dim, N,
             omp_threads=1, subspace_workers=0, use_aocl_blas=True,
             has_avx512=True, detail=False):
    """Predict peak RSS in bytes for the fused streaming build.

    Single-threaded build (the measured /usr/bin/time -v points) means
    omp_threads=1. subspace_workers defaults to M. All M std::thread workers
    are still spawned concurrently and each allocates its own ip_block, so the
    per-worker scratch multiplies by worker_count even on 1 core.
    """
    padded_dim = pad_dim(dim)
    sub_dim = dim // M
    block_rows = block_rows_for_mib(block_mib, dim)
    sample_n = min(N, sample_rows)
    bb = rabitq_record_bytes(padded_dim)

    sw = M if subspace_workers <= 0 else min(M, subspace_workers)
    worker_count = min(sw, M)  # std::thread workers spawned

    # assign_scratch_per_worker is a fixed 16 MiB floor (min_assign_scratch_per_worker);
    # total BLAS scratch = 16 MiB x sw. Build time is flat across 16-64 MiB/worker (measured).
    per_worker_budget = 16 * 1024 * 1024

    # use_simd_assign: sub_dim <= 8 (only tiny subspaces)
    use_simd = has_avx512 and sub_dim <= 8
    # use_points16: sub_dim in [32,64]
    use_points16 = has_avx512 and (not use_simd) and 32 <= sub_dim <= 64
    use_blas = use_aocl_blas and not use_simd and not use_points16

    # ---- Phase A/B/B': training (samples + sub_samples + HSKM + centroids) ----
    samples = sample_n * dim * 4
    sub_samples = sample_n * sub_dim * 4
    # rot_centroids persists through training AND into Phase C start
    rot_centroids = rabitq_k * padded_dim * 4
    top_centroids = rabitq_k * dim * 4
    ivf_centroids = M * ivf_k * sub_dim * 4
    ivf_simd_centroids = ivf_centroids if use_simd else 0
    cell_sizes = M * ivf_k * 4

    # HSKM scratch (per-call, freed between subspaces; one alive at a time).
    # Linux batch sizes: X=4096, Y=1024 -> 16 MiB per tile buffer.
    x_batch, y_batch = 4096, 1024
    gemm_tile = x_batch * y_batch * 4  # 16 MiB
    # meso + fine coexist at peak (hierarchical_superkmeans.h:221 + :352)
    hskm_tiles = 2 * gemm_tile
    n_meso = max(1, int(round(math.sqrt(ivf_k))))
    vertical_d = max(16, int(math.ceil(0.75 * sub_dim)))
    # centroid arrays: 4 K*d floats (final,current,horizontal,prev) + partial
    hskm_centroids = 4 * ivf_k * sub_dim * 4 + ivf_k * vertical_d * 4
    hskm_per_sample = sample_n * (4 + 4 + 4 + 8)  # assignments,distances,data_norms,indices_flat
    max_meso_size = max(1, sample_n // n_meso)
    hskm_meso_buf = max_meso_size * sub_dim * 4 + max_meso_size * 12
    # top-IVF HSKM (d=dim, K=rabitq_k): bigger centroid term, same tiles
    hskm_top_centroids = 4 * rabitq_k * dim * 4 + rabitq_k * max(16, int(math.ceil(0.75 * dim))) * 4
    hskm_top_per_sample = sample_n * (4 + 4 + 4 + 8)
    hskm_top_meso_buf = max(1, sample_n // max(1, int(round(math.sqrt(rabitq_k))))) * dim * 4

    # Training-phase peak (samples + sub_samples + ivf_centroids + one HSKM alive)
    # HSKM for top-IVF runs while samples alive; HSKM for sub-IVF runs while
    # sub_samples alive. The sub-IVF HSKM is the bigger of the two per-call
    # because sub_dim<top dim but K(ivf_k) can exceed rabitq_k. Take the max.
    hskm_sub_peak = hskm_tiles + hskm_centroids + hskm_per_sample + hskm_meso_buf
    hskm_top_peak = hskm_tiles + hskm_top_centroids + hskm_top_per_sample + hskm_top_meso_buf
    hskm_peak = max(hskm_sub_peak, hskm_top_peak)
    phase_train = samples + sub_samples + rot_centroids + top_centroids + \
                  ivf_centroids + ivf_simd_centroids + cell_sizes + hskm_peak

    # ---- Phase C: fused encode/assign/write (the steady-state peak) ----
    # samples + sub_samples are FREED (streaming_build.h:1330-1331) before C.
    double_buf = 2 * block_rows * dim * 4          # PrefetchedVectorBlockReader
    rotated = block_rows * padded_dim * 4
    rotated_dim = block_rows * dim * 4 if padded_dim != dim else 0
    top_assignments = block_rows * 4
    rabitq_assignments = block_rows * 4
    subspace_assignments = M * block_rows * 4      # int, all M live
    subspace_assignments_u16 = M * block_rows * 2  # uint16, all M live
    bin_chunk = block_rows * bb

    # Per-worker BLAS assign scratch (ip_block + optional packed_block).
    # worker_count std::threads run concurrently; each allocates inside the fn.
    if use_blas:
        abr = capped_assign_block_rows(ivf_k, per_worker_budget)
        ip_block_pw = abr * ivf_k * 4
        packed_pw = abr * sub_dim * 4 if sub_dim >= 32 else 0
        assign_scratch = worker_count * (ip_block_pw + packed_pw)
        # top-IVF rabitq assigner: assign_batch(top_data, ..., rows) where
        # top_data is dim-strided contiguous (rotated_dim or rotated when equal),
        # and top IVF sub_dim_==dim, so row_stride==sub_dim_ -> NO packed_block.
        top_abr = capped_assign_block_rows(rabitq_k, 128 * 1024 * 1024)
        top_ip_block = top_abr * rabitq_k * 4
        top_assign_scratch = top_ip_block
    else:
        # simd/points16 paths: no ip_block (in-register tiles / SoA)
        assign_scratch = 0
        top_assign_scratch = 0

    # per-vec rabitq binary_code scratch (rabitq_impl.hpp:172) in the omp loop.
    rabitq_vec_scratch = omp_threads * padded_dim * 4

    phase_c = (double_buf + rotated + rotated_dim + top_assignments +
               rabitq_assignments + subspace_assignments + subspace_assignments_u16 +
               bin_chunk + assign_scratch + top_assign_scratch + rabitq_vec_scratch +
               rot_centroids + top_centroids + ivf_centroids + ivf_simd_centroids + cell_sizes)

    # ---- Phase E: reorder/finalize (disk-backed; bounded RAM) ----
    # keys column N*4, small counting scratch, parallel rewrite buffer (256MB cap).
    rewrite_budget = 256 * 1024 * 1024
    keys = N * 4
    phase_e = keys + rewrite_budget  # loose upper envelope

    # Runtime overhead (calibrated against /usr/bin/time -v single-thread builds;
    # see /tmp/rss_formula.md §5). The analytic sum of .resize()'d buffers is a
    # LOWER BOUND. Two slack regimes:
    #   - small/moderate heap (< ~900 MiB): ~6% slack (glibc arenas mostly empty).
    #   - large heap (>= ~900 MiB): ramps to ~18% (per-thread arenas fill, BLIS
    #     internal GEMM workspace, fragmentation under M+2 concurrent allocators).
    # Plus a 40 MiB floor for binary/libs/BLIS pool/stacks.
    raw = max(phase_train, phase_c, phase_e)
    base_overhead = 40 * 1024 * 1024
    # Analytic sum matches /usr/bin/time within ~5% below ~900 MiB. A larger
    # slack appears above that (BLIS GEMM workspace for big K*sub_dim, glibc
    # arena fill under M+2 threads), ramping from 6% to 18% over [900,1500] MiB.
    ramp = min(1.0, max(0.0, (raw - 900 * 1024 * 1024) / (600 * 1024 * 1024)))
    slack_rate = 0.06 + (0.18 - 0.06) * ramp
    heap_slack = slack_rate * raw
    peak = raw + base_overhead + heap_slack
    if detail:
        return {
            "peak_bytes": peak,
            "raw_bytes": raw,
            "peak_mib": peak / MIB,
            "phase_train_mib": phase_train / MIB,
            "phase_c_mib": phase_c / MIB,
            "phase_e_mib": phase_e / MIB,
            "block_rows": block_rows,
            "sub_dim": sub_dim,
            "padded_dim": padded_dim,
            "use_blas": use_blas,
            "use_points16": use_points16,
            "use_simd": use_simd,
            "per_worker_budget_mib": per_worker_budget / MIB,
            "abr": capped_assign_block_rows(ivf_k, per_worker_budget) if use_blas else 0,
            "assign_scratch_mib": assign_scratch / MIB,
            "top_assign_scratch_mib": top_assign_scratch / MIB if use_blas else 0,
            "double_buf_mib": double_buf / MIB,
            "rotated_mib": (rotated + rotated_dim) / MIB,
            "bin_chunk_mib": bin_chunk / MIB,
            "subspace_assign_mib": (subspace_assignments + subspace_assignments_u16) / MIB,
            "samples_mib": samples / MIB,
            "hskm_peak_mib": hskm_peak / MIB,
        }
    return peak


# ---- Measured validation points (MiB) ----
MEASURED = [
    # name, M, ivf_k, rabitq_k, sample_rows, block_mib, dim, N, measured_mib
    # production config: sample_mib=512 (effective rows shown), block_mib=128, assign floor 16/worker
    ("gist M8 b128",        8, 1024, 200, 139810, 128,  960, 1000000, 690),
    ("cohere1m M8 b128",    8, 1024, 200, 174762, 128,  768, 1000000, 660),
    ("laion M8 b128",       8, 1024, 200, 174762, 128,  768, 1000000, 646),
    ("imagenet M8 b128",    8, 1024, 200,  65536, 128, 2048, 1281167, 913),
    ("wikipedia M16 b128", 16, 2500, 200, 131072, 128, 1024, 10000000, 635),
    ("cohere10m M16 b128", 16, 2500, 200, 174762, 128,  768, 10000000, 666),
    # block_mib sweep (M16, sample65536, points16 path -- block-bound regime)
    ("block64",            16, 2500, 200,  65536,  64,  960, 1000000, 359),
    ("block128",           16, 2500, 200,  65536, 128,  960, 1000000, 542),
    ("block192",           16, 2500, 200,  65536, 192,  960, 1000000, 680),
    ("block256",           16, 2500, 200,  65536, 256,  960, 1000000, 1054),
    ("block384",           16, 2500, 200,  65536, 384,  960, 1000000, 1468),
]


def validate():
    print(f"{'config':<24} {'meas':>6} {'pred':>6} {'err%':>6}  dominant")
    print("-" * 70)
    worst = 0.0
    for name, M, ivf_k, rk, sr, bmib, dim, N, meas in MEASURED:
        d = peak_rss(M, ivf_k, rk, sr, bmib, dim, N, omp_threads=1, detail=True)
        pred = d["peak_mib"]
        err = (pred - meas) / meas * 100.0
        worst = max(worst, abs(err))
        # find dominant term in phase_c
        terms = {
            "doublebuf": d["double_buf_mib"],
            "rotated": d["rotated_mib"],
            "binchunk": d["bin_chunk_mib"],
            "subassign": d["subspace_assign_mib"],
            "assignscratch": d["assign_scratch_mib"] + d["top_assign_scratch_mib"],
            "samples": d["samples_mib"],
            "hskm": d["hskm_peak_mib"],
        }
        dom = max(terms, key=terms.get)
        flag = "T" if d["peak_mib"] == d["phase_train_mib"] else "C"
        print(f"{name:<24} {meas:>6.0f} {pred:>6.0f} {err:>+6.1f}  [{flag}] {dom}={terms[dom]:.0f}")
    print("-" * 70)
    print(f"worst |err| = {worst:.1f}%   ([T]=training peak, [C]=encode-loop peak)")


def cli():
    ap = argparse.ArgumentParser(description="SCORE streaming build peak RSS model")
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--ivf-k", type=int, required=True)
    ap.add_argument("--rabitq-k", type=int, default=200)
    ap.add_argument("--sample-rows", type=int, required=True)
    ap.add_argument("--block-mib", type=int, required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--omp-threads", type=int, default=1)
    ap.add_argument("--detail", action="store_true")
    a = ap.parse_args()
    if a.detail:
        import json
        print(json.dumps(peak_rss(a.M, a.ivf_k, a.rabitq_k, a.sample_rows,
                                  a.block_mib, a.dim, a.N,
                                  omp_threads=a.omp_threads, detail=True), indent=2))
    else:
        print(f"{peak_rss(a.M, a.ivf_k, a.rabitq_k, a.sample_rows, a.block_mib, a.dim, a.N, omp_threads=a.omp_threads)/MIB:.1f} MiB")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] not in ("-h", "--help"):
        cli()
    else:
        validate()
