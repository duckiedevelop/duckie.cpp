// duckie: decode flash attention (one query token) for quantized KV caches, head dim 256.
//
// The generic decode kernels are compute/latency bound on quantized KV: the VEC kernel re-reads and re-dequantizes
// K/V once per query head and looks codebooks up in __constant__ memory with divergent indices (serialized), and the
// turbo MMA path dequantizes synchronously with no load/compute overlap. Here:
//   * one warp walks a contiguous slice of the KV sequence for ONE kv head and ALL of its G query heads,
//     so every K/V byte is read and dequantized exactly once;
//   * each lane owns 8 of the 256 dims; codebooks live in shared memory (16 entries -> conflict-free);
//   * KQ for 4 positions x G heads is reduced with a transpose-reduce (27 shuffles instead of 120 for G = 6);
//   * the softmax is online, in f32, in the log2 domain; per-warp partials are merged by a small combine kernel.
// Math is the same as the reference: softmax(scale * Q K^T + mask) V.

#include "fattn-duckie.cuh"
#include "turbo-quant.cuh"

#include <cfloat>

#define DFA_D  256  // head dim (K and V)
static constexpr float DFA_LOG2E = 1.4426950408889634f;
#define DFA_NW 4    // warps per block
#define DFA_P  4    // KV positions per KQ step
#define DFA_T  32   // KV positions per softmax tile (one per lane)
#define DFA_VPL (DFA_D / WARP_SIZE) // values per lane

static_assert(DFA_VPL == 8, "lane layout assumes 8 values per lane");

// Dequantize the 8 values [8*lane, 8*lane + 8) of one 256-value K/V row.
template <ggml_type type>
static __device__ __forceinline__ void dfa_dequant8(const char * __restrict__ row, const int lane, const float * __restrict__ lut, float * v) {
    const int e0 = DFA_VPL*lane;
    if constexpr (type == GGML_TYPE_TURBO4_0) {
        const block_turbo4_0 * b = (const block_turbo4_0 *) row + e0/QK_TURBO4;
        const int        j  = e0 % QK_TURBO4;
        const float      n  = __half2float(b->norm);
        const uint16_t * qs = (const uint16_t *) (b->qs + j/2);
        const uint32_t   q  = qs[0] | ((uint32_t) qs[1] << 16);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            v[k] = lut[(q >> (4*k)) & 0xF] * n;
        }
    } else if constexpr (type == GGML_TYPE_TURBO3_0) {
        const block_turbo3_0 * b = (const block_turbo3_0 *) row + e0/QK_TURBO3;
        const int      j  = e0 % QK_TURBO3;
        const float    n  = __half2float(b->norm);
        const uint32_t qs = *(const uint16_t *) (b->qs + j/4);
        const uint32_t sg = b->signs[j/8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            v[k] = lut[((qs >> (2*k)) & 0x3) | (((sg >> k) & 0x1) << 2)] * n;
        }
    } else if constexpr (type == GGML_TYPE_Q8_0) {
        const block_q8_0 * b = (const block_q8_0 *) row + e0/QK8_0;
        const int        j  = e0 % QK8_0;
        const float      d  = __half2float(b->d);
        const uint16_t * qs = (const uint16_t *) (b->qs + j);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint16_t w = qs[k];
            v[2*k + 0] = d * (float) (int8_t) (w & 0xFF);
            v[2*k + 1] = d * (float) (int8_t) (w >> 8);
        }
    } else {
        static_assert(type == GGML_TYPE_F16, "unsupported KV type");
        const uint4 raw = *(const uint4 *) ((const half *) row + e0);
        const half2 * h = (const half2 *) &raw;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 f = __half22float2(h[k]);
            v[2*k + 0] = f.x;
            v[2*k + 1] = f.y;
        }
    }
    GGML_UNUSED(lut);
}

template <ggml_type type>
static __device__ __forceinline__ void dfa_init_lut(float * lut) {
    if constexpr (type == GGML_TYPE_TURBO4_0) {
        if (threadIdx.x < 16) { lut[threadIdx.x] = TURBO_CENTROIDS_4BIT[threadIdx.x]; }
    } else if constexpr (type == GGML_TYPE_TURBO3_0) {
        if (threadIdx.x < 8)  { lut[threadIdx.x] = TURBO_CENTROIDS_3BIT[threadIdx.x]; }
    }
    GGML_UNUSED(lut);
}

// One halving round of the transpose-reduce: lanes with bit o set keep the upper half of their n values.
template <int n>
static __device__ __forceinline__ void dfa_halve(float * v, const bool upper, const int o) {
#pragma unroll
    for (int i = 0; i < n/2; ++i) {
        const float send = upper ? v[i]       : v[i + n/2];
        const float keep = upper ? v[i + n/2] : v[i];
        v[i] = keep + __shfl_xor_sync(0xFFFFFFFF, send, o, WARP_SIZE);
    }
}

// Sum N per-lane values over the warp. Afterwards v[0 .. N/8) hold full sums of the values with indices
// dfa_reduce_base(lane, N) + i (identical across each group of 4 lanes).
template <int N>
static __device__ __forceinline__ void dfa_reduce(float * v, const int lane) {
    static_assert(N % 8 == 0, "N must be a multiple of 8");
    dfa_halve<N    >(v, lane & 16, 16);
    dfa_halve<N / 2>(v, lane &  8,  8);
    dfa_halve<N / 4>(v, lane &  4,  4);
#pragma unroll
    for (int i = 0; i < N/8; ++i) {
        v[i] += __shfl_xor_sync(0xFFFFFFFF, v[i], 2, WARP_SIZE);
        v[i] += __shfl_xor_sync(0xFFFFFFFF, v[i], 1, WARP_SIZE);
    }
}

static __device__ __forceinline__ int dfa_reduce_base(const int lane, const int N) {
    return ((lane & 16) ? N/2 : 0) + ((lane & 8) ? N/4 : 0) + ((lane & 4) ? N/8 : 0);
}

template <int G, ggml_type type_K, ggml_type type_V>
__launch_bounds__(DFA_NW*WARP_SIZE, 3) // 168 regs, no spills: 3 blocks/SM instead of 2
static __global__ void flash_attn_duckie_decode(
        const char * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V, const half * __restrict__ mask,
        float2 * __restrict__ part_ml, float * __restrict__ part_acc,
        const float scale_log2, const int n_kv, const int pos_per_warp, const int n_splits,
        const int64_t nbq2, const int64_t nbk1, const int64_t nbk2, const int64_t nbv1, const int64_t nbv2) {
    constexpr int N = DFA_P*G; // KQ partials per step

    const int lane  = threadIdx.x;
    const int warp  = threadIdx.y;
    const int hkv   = blockIdx.y;
    const int split = blockIdx.x*DFA_NW + warp;

    __shared__ float lutK[16];
    __shared__ float lutV[16];
    __shared__ float sc[DFA_NW][DFA_T][G]; // scores, then probabilities

    if (warp == 0) {
        dfa_init_lut<type_K>(lutK);
        dfa_init_lut<type_V>(lutV);
    }
    __syncthreads();

    if (split >= n_splits) {
        return; // no block-level sync after this point
    }

    const int j0 = split*pos_per_warp;
    const int j1 = min(j0 + pos_per_warp, n_kv);

    float q[G][DFA_VPL];
#pragma unroll
    for (int h = 0; h < G; ++h) {
        const float4 * qh = (const float4 *) (Q + (int64_t) (hkv*G + h)*nbq2) + 2*lane;
        const float4 a = qh[0];
        const float4 b = qh[1];
        q[h][0] = a.x*scale_log2; q[h][1] = a.y*scale_log2; q[h][2] = a.z*scale_log2; q[h][3] = a.w*scale_log2;
        q[h][4] = b.x*scale_log2; q[h][5] = b.y*scale_log2; q[h][6] = b.z*scale_log2; q[h][7] = b.w*scale_log2;
    }

    float m[G];
    float l[G];
    float acc[G][DFA_VPL];
#pragma unroll
    for (int h = 0; h < G; ++h) {
        m[h] = -INFINITY;
        l[h] = 0.0f;
#pragma unroll
        for (int k = 0; k < DFA_VPL; ++k) {
            acc[h][k] = 0.0f;
        }
    }

    const char * Kh = K + hkv*nbk2;
    const char * Vh = V + hkv*nbv2;
    const int    rb = dfa_reduce_base(lane, N);

    for (int t0 = j0; t0 < j1; t0 += DFA_T) {
        // ---- scores for the tile: sc[warp][t][h] = log2(e) * (scale * q_h . k_t + mask_t) ----
#pragma unroll 1
        for (int g = 0; g < DFA_T; g += DFA_P) {
            float part[N];
#pragma unroll
            for (int p = 0; p < DFA_P; ++p) {
                const int j = min(t0 + g + p, j1 - 1); // clamped reads; masked out below
                float kf[DFA_VPL];
                dfa_dequant8<type_K>(Kh + (int64_t) j*nbk1, lane, lutK, kf);
#pragma unroll
                for (int h = 0; h < G; ++h) {
                    float s = 0.0f;
#pragma unroll
                    for (int k = 0; k < DFA_VPL; ++k) {
                        s += q[h][k]*kf[k];
                    }
                    part[p*G + h] = s;
                }
            }
            dfa_reduce<N>(part, lane);
            if ((lane & 3) == 0) {
#pragma unroll
                for (int i = 0; i < N/8; ++i) {
                    const int idx = rb + i;
                    const int p   = idx / G;
                    const int h   = idx % G;
                    const int j   = t0 + g + p;
                    float s = -INFINITY;
                    if (j < j1) {
                        s = part[i] + (mask ? DFA_LOG2E*__half2float(mask[j]) : 0.0f);
                    }
                    sc[warp][g + p][h] = s;
                }
            }
        }
        __syncwarp();

        // ---- online softmax over the tile (lane t owns position t0 + t) ----
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const float s     = sc[warp][lane][h];
            const float m_new = fmaxf(m[h], warp_reduce_max<WARP_SIZE>(s));
            const float m_use = m_new == -INFINITY ? 0.0f : m_new;
            const float alpha = exp2f(m[h] - m_use);
            const float p     = exp2f(s - m_use);
            l[h] = l[h]*alpha + warp_reduce_sum<WARP_SIZE>(p);
            m[h] = m_new;
#pragma unroll
            for (int k = 0; k < DFA_VPL; ++k) {
                acc[h][k] *= alpha;
            }
            sc[warp][lane][h] = p;
        }
        __syncwarp();

        // ---- acc += p_t * v_t ----
        const int tn = min(DFA_T, j1 - t0);
#pragma unroll 4
        for (int t = 0; t < tn; ++t) {
            float vf[DFA_VPL];
            dfa_dequant8<type_V>(Vh + (int64_t) (t0 + t)*nbv1, lane, lutV, vf);
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const float p = sc[warp][t][h];
#pragma unroll
                for (int k = 0; k < DFA_VPL; ++k) {
                    acc[h][k] += p*vf[k];
                }
            }
        }
        __syncwarp();
    }

#pragma unroll
    for (int h = 0; h < G; ++h) {
        const int64_t ip = (int64_t) (hkv*G + h)*n_splits + split;
        if (lane == 0) {
            part_ml[ip] = make_float2(m[h], l[h]);
        }
        float4 * dst4 = (float4 *) (part_acc + ip*DFA_D) + 2*lane;
        dst4[0] = make_float4(acc[h][0], acc[h][1], acc[h][2], acc[h][3]);
        dst4[1] = make_float4(acc[h][4], acc[h][5], acc[h][6], acc[h][7]);
    }
}

// One block per query head, one thread per output dim: merge the per-warp partials.
static __global__ void flash_attn_duckie_combine(
        const float2 * __restrict__ part_ml, const float * __restrict__ part_acc, char * __restrict__ dst,
        const int n_splits, const int64_t nb_dst1) {
    const int hq = blockIdx.x;
    const int d  = threadIdx.x;

    const float2 * ml = part_ml + (int64_t) hq*n_splits;

    float M = -INFINITY;
    for (int s = 0; s < n_splits; ++s) {
        M = fmaxf(M, ml[s].x);
    }
    const float M_use = M == -INFINITY ? 0.0f : M;

    float L = 0.0f;
    float o = 0.0f;
    const float * acc = part_acc + (int64_t) hq*n_splits*DFA_D + d;
    for (int s = 0; s < n_splits; ++s) {
        const float w = exp2f(ml[s].x - M_use);
        L += w*ml[s].y;
        o += w*acc[(int64_t) s*DFA_D];
    }
    ((float *) (dst + hq*nb_dst1))[d] = L > 0.0f ? o/L : 0.0f;
}

template <int G, ggml_type type_K, ggml_type type_V>
static void flash_attn_duckie_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int n_kv = K->ne[1];
    const int H_kv = K->ne[2];
    const int H_q  = Q->ne[2];
    const int nsm  = ggml_cuda_info().devices[ctx.device].nsm;

    // Every warp gets an equal slice, so size the grid to exactly one full wave of resident blocks
    // (a partial second wave would leave most SMs idle). Slices are whole 32-position tiles, at least two.
    static int blocks_per_sm[GGML_CUDA_MAX_DEVICES] = {0};
    if (blocks_per_sm[ctx.device] == 0) {
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm[ctx.device],
            flash_attn_duckie_decode<G, type_K, type_V>, DFA_NW*WARP_SIZE, 0));
        blocks_per_sm[ctx.device] = std::max(1, blocks_per_sm[ctx.device]);
    }
    const int target       = std::max(1, nsm*blocks_per_sm[ctx.device]*DFA_NW/H_kv); // warps per kv head
    const int pos_per_warp = std::max(2*DFA_T, GGML_PAD((n_kv + target - 1)/target, DFA_T));
    const int n_splits     = (n_kv + pos_per_warp - 1)/pos_per_warp;

    ggml_cuda_pool_alloc<float2> part_ml (ctx.pool(), (size_t) H_q*n_splits);
    ggml_cuda_pool_alloc<float>  part_acc(ctx.pool(), (size_t) H_q*n_splits*DFA_D);

    cudaStream_t stream = ctx.stream();

    const dim3 grid((n_splits + DFA_NW - 1)/DFA_NW, H_kv, 1);
    const dim3 block(WARP_SIZE, DFA_NW, 1);
    flash_attn_duckie_decode<G, type_K, type_V><<<grid, block, 0, stream>>>(
        (const char *) Q->data, (const char *) K->data, (const char *) V->data,
        mask ? (const half *) mask->data : nullptr, part_ml.get(), part_acc.get(),
        scale*DFA_LOG2E, n_kv, pos_per_warp, n_splits,
        Q->nb[2], K->nb[1], K->nb[2], V->nb[1], V->nb[2]);
    CUDA_CHECK(cudaGetLastError());

    flash_attn_duckie_combine<<<H_q, DFA_D, 0, stream>>>(part_ml.get(), part_acc.get(), (char *) dst->data, n_splits, dst->nb[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <int G, ggml_type type_K>
static bool flash_attn_duckie_switch_V(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (dst->src[2]->type) {
        case GGML_TYPE_TURBO4_0: flash_attn_duckie_launch<G, type_K, GGML_TYPE_TURBO4_0>(ctx, dst); return true;
        case GGML_TYPE_TURBO3_0: flash_attn_duckie_launch<G, type_K, GGML_TYPE_TURBO3_0>(ctx, dst); return true;
        case GGML_TYPE_Q8_0:     flash_attn_duckie_launch<G, type_K, GGML_TYPE_Q8_0    >(ctx, dst); return true;
        case GGML_TYPE_F16:      flash_attn_duckie_launch<G, type_K, GGML_TYPE_F16     >(ctx, dst); return true;
        default: return false;
    }
}

template <int G>
static bool flash_attn_duckie_switch_K(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (dst->src[1]->type) {
        case GGML_TYPE_TURBO4_0: return flash_attn_duckie_switch_V<G, GGML_TYPE_TURBO4_0>(ctx, dst);
        case GGML_TYPE_TURBO3_0: return flash_attn_duckie_switch_V<G, GGML_TYPE_TURBO3_0>(ctx, dst);
        case GGML_TYPE_Q8_0:     return flash_attn_duckie_switch_V<G, GGML_TYPE_Q8_0    >(ctx, dst);
        case GGML_TYPE_F16:      return flash_attn_duckie_switch_V<G, GGML_TYPE_F16     >(ctx, dst);
        default: return false;
    }
}

bool ggml_cuda_flash_attn_ext_duckie(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static const bool disabled = getenv("GGML_DUCKIE_FA") && atoi(getenv("GGML_DUCKIE_FA")) == 0;
    if (disabled) {
        return false;
    }

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    float max_bias;
    float logit_softcap;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const bool ok =
        GGML_CUDA_CC_IS_NVIDIA(ggml_cuda_info().devices[ctx.device].cc) &&
        Q->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        Q->ne[0] == DFA_D && K->ne[0] == DFA_D && V->ne[0] == DFA_D &&
        Q->ne[1] == 1 && Q->ne[3] == 1 && K->ne[3] == 1 && V->ne[3] == 1 &&
        K->ne[1] == V->ne[1] && K->ne[2] == V->ne[2] && Q->ne[2] % K->ne[2] == 0 &&
        K->nb[0] == ggml_type_size(K->type) && V->nb[0] == ggml_type_size(V->type) &&
        Q->nb[0] == sizeof(float) && Q->nb[2] % 16 == 0 &&
        (!mask || (mask->type == GGML_TYPE_F16 && mask->ne[0] >= K->ne[1])) &&
        !sinks && max_bias == 0.0f && logit_softcap == 0.0f &&
        !(K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16) && // the MMA kernel is better for f16 KV
        dst->nb[0] == sizeof(float) && dst->ne[1] == Q->ne[2] &&
        K->ne[1] >= 256; // short contexts: the existing kernels are already fine
    if (!ok) {
        return false;
    }

    switch (Q->ne[2] / K->ne[2]) {
        case 4: return flash_attn_duckie_switch_K<4>(ctx, dst);
        case 6: return flash_attn_duckie_switch_K<6>(ctx, dst);
        case 8: return flash_attn_duckie_switch_K<8>(ctx, dst);
        default: return false;
    }
}
