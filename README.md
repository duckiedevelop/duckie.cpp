# duckie.cpp

<div align="center">

<b>llama.cpp + TurboQuant, rebuilt around one model on one GPU:<br>
Bonsai-27B (Qwen3.6-27B, 1-bit <code>Q1_0</code>) on an RTX 4050 Laptop (6 GB, Ada <code>sm_89</code>).</b>

Faster decode, much faster attention at long context, same flags and same output.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

</div>

duckie.cpp is a fork of [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant), which is itself
a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). Everything those projects do still works here.
On top of that:

- **The TurboQuant KV cache types are renamed** `turbo2..turbo6` → **`duckie2..duckie6`**. Same bits, same quality.
- **A dedicated 1-bit matvec kernel** for `Q1_0` weights: token generation is **22-33% faster**.
- **A new decode flash-attention kernel** for quantized KV caches: attention at 32k-64k context runs
  **4-10x faster** than the stock kernels, and full-model decode at 32k context is **+52%** over stock attention.
  Speed holds up over long agentic sessions instead of falling off a cliff.
- **A build script tuned for the card**: `sm_89` only, native CPU, static binaries.

## Quick start

Build (Windows, Visual Studio 2022 + CUDA 12.x):

```bat
build-duckie.bat
```

Binaries land in `build\bin\`: `llama-server.exe`, `llama-cli.exe`, `llama-bench.exe`, `test-backend-ops.exe`.
Pass extra targets as arguments, e.g. `build-duckie.bat test-turbo-quant`.

Run the OpenAI-compatible server:

```bat
build\bin\llama-server.exe -m Bonsai-27B-Q1_0.gguf -ngl 99 -fa on -c 32768 ^
  --cache-type-k duckie4 --cache-type-v duckie4 -b 256 -ub 128
```

Heads-up: Bonsai shares each KV head across 6 query heads. For models like that, the inherited TurboQuant
*auto-asymmetric* rule upgrades a `duckieN` K cache to `q8_0` to protect quality, and logs a warning when it does.
So the command above actually runs `q8_0` K + `duckie4` V. Set `TURBO_AUTO_ASYMMETRIC=0` to keep K at `duckie4`
and halve the K cache. To fit more context in 6 GB, drop V to `duckie3`. The weights take 3.8 GB, so every saved
KV byte becomes context.

Benchmark it yourself:

```bat
build\bin\llama-bench.exe -m Bonsai-27B-Q1_0.gguf -ngl 99 -fa 1 -ctk duckie4 -ctv duckie4 -b 256 -ub 128 -p 0 -n 64 -d 0,16384,32768
```

## duckie KV cache types

| Type      | Was      | Bits/value | KV cache vs f16 | Use it for |
|-----------|----------|-----------:|----------------:|------------|
| `duckie2` | `turbo2` | 2.125      | 7.5x smaller    | maximum context, most quality loss |
| `duckie3` | `turbo3` | 3.125      | 5.1x smaller    | the capacity sweet spot |
| `duckie4` | `turbo4` | 4.125      | 3.9x smaller    | quality first; the default pick |
| `duckie5` | `turbo5` | 5.125      | 3.1x smaller    | between 4 and 6 (CUDA/CPU only) |
| `duckie6` | `turbo6` | 6.125      | 2.6x smaller    | highest fidelity (CUDA/CPU only) |

They work everywhere the old names did: `-ctk` / `-ctv` / `--cache-type-k` / `--cache-type-v` in `llama-server`,
`llama-cli`, `llama-bench` and `llama-perplexity`, mixed freely with `f16`, `q8_0` and friends. They need flash
attention, which is switched on automatically. The old `turbo*` names are gone, so old scripts need the new names.

[docs/KV-cache-quantization.md](docs/KV-cache-quantization.md) still describes the codec under its original `turbo*`
names (read `turbo4` as `duckie4`). The `TURBO_*` environment variables keep their names.

## Why it's fast

Bonsai-27B is a hybrid model: 64 layers, of which 48 are Gated DeltaNet (linear attention with a fixed-size state)
and 16 are full attention with 24 query heads sharing 4 KV heads (head dim 256). Every weight matrix is `Q1_0`:
one sign bit per weight plus an fp16 scale per 128 weights. That leaves two places where decode time goes:
streaming 3.8 GB of 1-bit weights per token, and, as the context grows, reading the KV cache in those 16 attention layers.
duckie.cpp attacks both.

### 1. A 1-bit matvec that runs at memory speed

The stock `Q1_0 × q8_1` matvec was a generic template sized for 4-bit and 8-bit weights. At decode time one
token touches every weight exactly once, so the only thing that matters is how close the kernel gets to the
card's ~192 GB/s. The duckie kernel (`mul_mat_vec_q1_0_rows` in
[ggml/src/ggml-cuda/mmvq.cu](ggml/src/ggml-cuda/mmvq.cu)):

- gives each warp 4 whole rows and 4 warps per block, so even the 5120-row projections launch enough warps to keep
  all 20 SMs streaming;
- has every lane stream 32 contiguous weights per step: 4 bytes of sign bits are expanded to ±1 bytes with
  `__byte_perm` and fed into 8 `dp4a` instructions against the 8-bit activations, and the fp16 block scale is
  applied once per block;
- fuses the SwiGLU/GeGLU gate into the same pass (gate and up projections read together, one write), matching
  the existing fusion path.

Result: the 248k-vocab output head went from 99 to 168 GB/s, and full-model token generation from 23.9 to
29-32 tokens/s (it varies with how hot the laptop is running).

### 2. Decode flash attention built for quantized KV

During decode there is one query token per sequence, and the stock kernels handled that case poorly for
TurboQuant caches. The vector kernel re-read and re-dequantized every K/V row once per query head (6x redundant
work at GQA 6) and looked codebook entries up in `__constant__` memory with divergent indices, which serializes.
The tensor-core path dequantized synchronously and could not overlap loads with math. Attention cost grew so fast
that at 32k context it was the biggest thing in the token.

The duckie kernel ([ggml/src/ggml-cuda/fattn-duckie.cu](ggml/src/ggml-cuda/fattn-duckie.cu)):

- **Packs the GQA group.** Each warp handles one KV head and all 6 query heads that share it, so every K/V byte is
  read and dequantized exactly once.
- **Splits the sequence (flash-decoding).** The KV range is cut into equal per-warp slices, sized from the measured
  occupancy so the grid is exactly one full wave of resident blocks. A small combine kernel merges the partial
  softmax results.
- **Keeps codebooks in shared memory.** The 16 `duckie4` / 8 `duckie3` centroids sit in shared memory, where
  divergent lookups don't bank-conflict, instead of serialized constant-cache reads.
- **Reduces with a transpose.** Each lane owns 8 of the 256 dims. Dot products for 4 positions × 6 heads are summed
  across the warp with a transpose-reduce: 27 shuffles instead of 120.
- **Runs an online softmax in the log2 domain** (scale and `log2(e)` folded into Q once), in fp32 throughout, so the
  numbers match the reference.
- **Tunes occupancy.** Register use is capped at 168 per thread with zero spills, which fits 3 blocks per SM instead
  of 2.

It handles `duckie3`, `duckie4`, `q8_0` and `f16` caches in any K/V mix at head dim 256 and GQA ratios 4, 6 and 8.
Anything else (prompt processing, other head sizes, f16+f16 caches, non-NVIDIA GPUs) goes to the stock kernels.

### 3. Long sessions

Long-horizon agent work means huge contexts that get reused turn after turn. With the duckie kernel, attention cost
per token grows far more slowly with context: at 32k tokens deep, decode still runs at ~88% of its short-context
speed (stock: ~56%; see below).
For a hybrid model the server also keeps context checkpoints of the DeltaNet state (up to 32 per slot, at least
8192 tokens apart), so a follow-up turn reuses the cached prefix instead of re-reading the whole conversation.
These defaults were already good for this model and are unchanged.

## Benchmarks

RTX 4050 Laptop GPU (6 GB), Bonsai-27B `Q1_0`, `-ngl 99 -fa 1 -b 256 -ub 128`.

**Full model, token generation (tokens/s), by context depth:**

| Context depth | Stock attention (`GGML_DUCKIE_FA=0`) | duckie attention | Gain |
|--------------:|-------------------------------------:|-----------------:|-----:|
| 0             | 30.2                                 | 29.2             | tie (within run-to-run noise) |
| 16k           | 21.7                                 | **28.2**         | +30% |
| 32k           | 16.8                                 | **25.6**         | +52% |

KV cache `q8_0` / `duckie4`; `duckie4` / `duckie4` gives the same numbers on this model (see the auto-asymmetric
note above). Both columns include the 1-bit matvec kernel. The TurboQuant fork as cloned ran at **23.9 tokens/s**
at depth 0 with the same settings. At 32k context, duckie.cpp keeps ~88% of its short-context speed. With the stock
attention kernels it keeps ~56%.

**One attention layer at decode** (24 query heads / 4 KV heads, head dim 256; `test-backend-ops perf`, µs per call, lower is better):

| KV cache (K / V)    | Context | Stock kernels | duckie kernel | Speedup |
|---------------------|--------:|--------------:|--------------:|--------:|
| `q8_0` / `duckie4`  | 32k     | 1811          | ~410-480      | ~4x     |
| `duckie4` / `duckie3` | 64k   | 8344          | ~820          | ~10x    |

Laptop GPUs boost and throttle, so expect ±10-20% run-to-run noise on any single number.

## Knobs

| Variable          | Default | Effect |
|-------------------|---------|--------|
| `GGML_DUCKIE_FA`  | on      | `0` turns off the duckie decode attention kernel (falls back to the stock kernels; handy for A/B checks) |
| `TURBO_AUTO_ASYMMETRIC` | `1` | `0` keeps a `duckieN` K cache as requested instead of upgrading it to `q8_0` on GQA ≥ 6 models |

Everything else inherited from TurboQuant still applies (`TURBO_LAYER_ADAPTIVE`, `TURBO_SPARSE_V`,
`GGML_CUDA_FUSE_CHAIN`, ...); see [docs/KV-cache-quantization.md](docs/KV-cache-quantization.md).

## Correctness

Every new kernel is checked against the CPU reference with `test-backend-ops`:

- Flash attention, head dim 256, all types, masks, GQA ratios and batch sizes: **796/796**
- `Q1_0` matmul: **51/51**; fused `Q1_0` gate/up + GLU: **32/32**
- `test-turbo-quant` and `test-quantize-fns`: clean

```bat
build\bin\test-backend-ops.exe -b CUDA0 -o FLASH_ATTN_EXT -p "hsk=256,hsv=256"
build\bin\test-backend-ops.exe -b CUDA0 -o MUL_MAT -p "type_a=q1_0"
```

## Scope and limits

- **Tuned for one setup.** `build-duckie.bat` compiles for `sm_89` (RTX 40-series) only. For another NVIDIA card,
  change `CMAKE_CUDA_ARCHITECTURES` in the script and delete `build\` to reconfigure. The kernels are plain CUDA and
  should run on any recent NVIDIA GPU, but they were only measured here.
- **NVIDIA only for the new kernels.** AMD, Metal, Vulkan, SYCL and CPU builds still compile and run, on the stock
  TurboQuant paths.
- **Decode is the focus.** Prompt processing uses the stock kernels and runs at the same speed as TurboQuant.
- Everything else from upstream TurboQuant (TQ weight types, ConvRot types, other backends) is untouched. See
  [docs/KV-cache-quantization.md](docs/KV-cache-quantization.md) and [docs/build.md](docs/build.md).

## Credits and license

MIT, like the projects it builds on. Built on the work of the
[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) contributors and
[TheTom's TurboQuant fork](https://github.com/TheTom/llama-cpp-turboquant), which implements
[TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874).
Fork development rules and test gates: [AGENTS.md](AGENTS.md).

Third-party single-header libraries:
[cpp-httplib](https://github.com/yhirose/cpp-httplib) (MIT),
[stb-image](https://github.com/nothings/stb) (public domain),
[nlohmann/json](https://github.com/nlohmann/json) (MIT),
[miniaudio.h](https://github.com/mackron/miniaudio) (public domain),
[subprocess.h](https://github.com/sheredom/subprocess.h) (public domain).
