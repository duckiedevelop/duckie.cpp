#pragma once

#include "common.cuh"

// duckie: GQA-packed split-KV decode flash attention for quantized KV caches.
// Returns false (and does nothing) when the op is outside the shapes/types it handles.
bool ggml_cuda_flash_attn_ext_duckie(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
