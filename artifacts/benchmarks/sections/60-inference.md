## Phase 6 - inference engine

Hardware: CPU-only | Intel64 Family 6 Model 186 Stepping 2, GenuineIntel | 2 torch thread(s) | Windows 10 | python 3.11.9 | torch 2.13.0+cpu. Seeds: [0, 1, 2]. CI: bootstrap 95%.
Config: `configs/model/12m_proxy.yaml` (LocalMind-12M-proxy).

### 0. CPU thread scaling of one decode step

| threads | decode step | decode tok/s |
|---|---|---|
| threads=1 | decode step 33.32 [22.45, 53.37] ms | 34.91 [18.74, 44.54] tok/s |
| threads=2 | decode step 62.51 [55.99, 72.93] ms | 16.21 [13.71, 17.86] tok/s |
| threads=4 | decode step 42.87 [29.09, 60.59] ms | 25.52 [16.50, 34.38] tok/s |
| threads=8 | decode step 48.48 [26.13, 80.80] ms | 25.54 [12.38, 38.28] tok/s |

### 1-3. KV cache: naive -> contiguous -> paged

| variant | prompt | new | tokens/s (95% CI) | TTFT ms | TPOT ms | speedup vs naive |
|---|---|---|---|---|---|---|
| naive_no_cache | 32 | 64 | 19.28 [13.46, 24.96] | 45.9 [33.6, 56.1] | 56.44 [40.99, 76.13] | - |
| naive_no_cache | 32 | 256 | 8.67 [7.11, 10.81] | 47.0 [17.2, 62.4] | 119.84 [94.51, 141.03] | - |
| naive_no_cache | 32 | 512 | 5.84 [4.14, 8.50] | 40.6 [14.3, 61.7] | 190.53 [118.58, 247.31] | - |
| contiguous_kv | 32 | 64 | 47.16 [31.58, 67.40] | 47.7 [20.7, 64.2] | 23.22 [15.06, 31.17] | 2.45x |
| contiguous_kv | 32 | 256 | 34.03 [23.89, 48.81] | 54.6 [43.7, 66.8] | 33.11 [22.95, 41.79] | 3.93x |
| contiguous_kv | 32 | 512 | 46.27 [27.83, 79.72] | 46.1 [17.3, 63.1] | 27.84 [12.60, 36.83] | 7.92x |
| dynamic_kv | 32 | 64 | 49.50 [33.85, 72.90] | 43.3 [19.1, 63.5] | 22.35 [13.61, 28.97] | 2.57x |
| dynamic_kv | 32 | 256 | 38.59 [25.98, 59.45] | 55.3 [38.4, 75.4] | 29.83 [17.72, 38.87] | 4.45x |
| dynamic_kv | 32 | 512 | 44.27 [32.69, 65.75] | 41.8 [20.5, 57.7] | 25.14 [15.26, 30.58] | 7.58x |
| paged_kv | 32 | 64 | 42.39 [31.81, 58.46] | 51.6 [26.8, 70.9] | 25.23 [17.27, 30.87] | 2.20x |
| paged_kv | 32 | 256 | 35.30 [25.66, 49.88] | 56.9 [38.1, 73.4] | 30.76 [20.53, 38.87] | 4.07x |
| paged_kv | 32 | 512 | 37.50 [28.26, 54.74] | 54.3 [38.1, 70.8] | 30.14 [20.32, 36.07] | 6.42x |

### 3. Fragmentation: contiguous vs paged

| mix | mean len | contiguous internal frag | paged internal frag | memory amplification |
|---|---|---|---|---|
| uniform_short(seed=0) | 41 | 96.0% | 14.0% | 21.56x |
| rag_mixed(seed=0) | 216 | 78.9% | 1.5% | 4.68x |
| long_tail(seed=0) | 151 | 85.3% | 3.4% | 6.56x |
| uniform_short(seed=1) | 39 | 96.2% | 15.1% | 22.51x |
| rag_mixed(seed=1) | 212 | 79.2% | 1.4% | 4.75x |
| long_tail(seed=1) | 345 | 66.3% | 2.0% | 2.91x |
| uniform_short(seed=2) | 40 | 96.1% | 16.7% | 21.56x |
| rag_mixed(seed=2) | 242 | 76.3% | 1.1% | 4.18x |
| long_tail(seed=2) | 235 | 77.1% | 2.2% | 4.27x |

| budget MB | seq len | contiguous max seqs | paged max seqs | gain |
|---|---|---|---|---|
| 64 | 64 | 21 | 341 | 16.2x |
| 64 | 128 | 21 | 170 | 8.1x |
| 64 | 256 | 21 | 85 | 4.0x |
| 64 | 512 | 21 | 42 | 2.0x |

### 4-5. Batching and chunked prefill

| scenario | mode | throughput tok/s | goodput req/s | p99 TTFT ms | p99 TPOT ms | p99 e2e ms | batch |
|---|---|---|---|---|---|---|---|
| poisson | static | 43.69 [40.76, 45.75] | 0.778 [0.572, 1.019] | 1634.7 [1390.4, 1791.4] | 119.7 [89.5, 179.3] | 2994.4 [2675.6, 3387.9] | 1.39 [1.15, 1.51] |
| poisson | continuous | 36.72 [28.27, 45.10] | 0.470 [0.307, 0.752] | 1169.8 [474.9, 1578.3] | 107.6 [80.5, 129.0] | 4193.8 [2932.4, 5117.5] | 1.02 [1.00, 1.03] |
| burst | static | 37.95 [34.43, 44.65] | 0.062 [0.000, 0.186] | 4867.8 [3861.1, 5500.7] | 179.5 [153.1, 202.3] | 6416.9 [5375.4, 6970.8] | 1.62 [1.62, 1.62] |
| burst | continuous | 50.15 [42.64, 55.35] | 0.209 [0.178, 0.231] | 2484.5 [1989.9, 3135.5] | 97.7 [69.0, 129.7] | 4797.2 [4275.7, 5568.2] | 1.13 [1.13, 1.13] |

| chunked prefill | p50 ITL ms | p99 ITL ms | max ITL ms | p99 TPOT ms | p99 ITL improvement |
|---|---|---|---|---|---|
| False | 41.8 [40.2, 44.9] | 430.9 [388.4, 484.9] | 451.7 [407.1, 503.7] | 67.6 [65.5, 69.6] | - |
| True | 80.7 [72.2, 88.5] | 219.9 [188.2, 251.3] | 253.2 [223.6, 284.3] | 87.5 [82.7, 90.1] | 49.0% |

### 6. Prefix caching (RAG-shaped: shared system prompt)

| enabled | mean TTFT ms | request hit rate | token hit rate | TTFT reduction |
|---|---|---|---|---|
| False | 1052.8 [966.7, 1139.6] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | - |
| True | 932.7 [812.6, 1082.6] | 0.875 [0.875, 0.875] | 0.737 [0.737, 0.737] | 11.4% |

### 7. Speculative decoding (UNTRAINED weights - a bracket, not a prediction)

| proposer | sampling | tokens/s | acceptance | tokens/iter | speedup |
|---|---|---|---|---|---|
| none_baseline | greedy | 25.14 [19.81, 28.18] | - | None | - |
| ngram | greedy | 42.35 [37.43, 49.09] | 1.000 [1.000, 1.000] | 1.88 [1.88, 1.88] | 1.68x |
| draft_model(L=2/6) | greedy | 33.58 [31.44, 35.26] | 1.000 [1.000, 1.000] | 4.57 [4.57, 4.57] | 1.34x |
| none_baseline | temp1.0 | 23.76 [22.18, 24.63] | - | None | - |
| ngram | temp1.0 | 27.51 [23.75, 30.03] | 0.000 [0.000, 0.000] | 1.00 [1.00, 1.00] | 1.16x |
| draft_model(L=2/6) | temp1.0 | 17.49 [14.73, 18.92] | 0.395 [0.358, 0.447] | 2.41 [2.29, 2.67] | 0.74x |

### 8. Constrained decoding

| constrained | invalid JSON rate | forced-close rate | tokens/s | tok/s cost |
|---|---|---|---|---|
| False | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 27.39 [16.17, 47.24] | - |
| True | 0.000 [0.000, 0.000] | 0.500 [0.400, 0.600] | 29.06 [23.26, 36.14] | -6.1% |

### 9. Quantization

| variant | weight MB | tokens/s | val BPB | KL vs fp32 | compression |
|---|---|---|---|---|---|
| fp32 | 45.83 [45.83, 45.83] | 55.88 [55.46, 56.61] | 4.742 [4.740, 4.744] | 0.00000 [0.00000, 0.00000] | 1.00x |
| int8_weight_only_g64 | 27.24 [27.24, 27.24] | 35.02 [29.62, 38.58] | 4.742 [4.741, 4.744] | 0.00002 [0.00001, 0.00002] | 1.68x |
| int4_weight_only_g64 | 24.08 [24.08, 24.08] | 16.87 [15.56, 18.01] | 4.741 [4.739, 4.744] | 0.00602 [0.00496, 0.00698] | 1.90x |
| torch_dynamic_int8 | 20.52 [20.52, 20.52] | 36.68 [33.48, 42.65] | 4.742 [4.741, 4.744] | 0.00032 [0.00030, 0.00033] | 2.23x |

| GGUF export | quant | MB | tensors | verified against llama.cpp |
|---|---|---|---|---|
| artifacts\gguf\localmind-12m-proxy-f32.gguf | f32 | 45.63 | 56 | **NO** |
| artifacts\gguf\localmind-12m-proxy-q8_0.gguf | q8_0 | 12.35 | 56 | **NO** |

### Honest baseline

vLLM on a T4: **NOT RUN** - no CUDA device and no vllm package in this environment (CPU-only laptop).
Expected outcome when run: LocalMind loses. The gap is kernel fusion, CUDA-graph capture, a real paged-attention kernel (no gather copy), and continuous batching with ragged attention instead of length bucketing.
