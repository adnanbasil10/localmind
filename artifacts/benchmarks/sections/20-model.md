## Phase 2 - model

Hardware: CPU | Intel64 Family 6 Model 186 Stepping 2, GenuineIntel | 8 torch threads | Windows | torch 2.13.0+cpu. Seeds: [0, 1, 2]. CI: bootstrap 95%.

### KV cache cost (fp16)

| variant | n_kv_heads | KB/token | MB @ 2048 ctx | vs MHA |
|---|---|---|---|---|
| MHA | 8 | 16.00 | 32.00 | 1.00x |
| GQA(8q/2kv) | 2 | 4.00 | 8.00 | 4.00x |
| MQA | 1 | 2.00 | 4.00 | 8.00x |

### Attention backends (one layer, batch 1)

| backend | kernel | seq | latency ms (95% CI) | peak MB | total alloc MB |
|---|---|---|---|---|---|
| naive | naive_reference | 512 | 36.04 [32.61, 42.67] | 19.5 | 48.4 |
| naive | naive_reference | 1024 | 87.74 [83.11, 95.09] | 72.0 | 161.9 |
| naive | naive_reference | 2048 | 423.78 [338.55, 468.43] | 276.0 | 583.8 |
| naive | naive_reference | 4096 | 1416.94 [1383.94, 1472.16] | 1080.0 | 2207.6 |
| sdpa_math | math | 512 | 40.99 [35.89, 49.22] | 5.0 | 37.4 |
| sdpa_math | math | 1024 | 88.64 [81.18, 97.62] | 10.0 | 113.9 |
| sdpa_math | math | 2048 | 259.66 [250.72, 270.25] | 20.0 | 383.8 |
| sdpa_math | math | 4096 | 1275.19 [1219.05, 1369.19] | 40.0 | 1391.5 |
| sdpa_efficient | flash_cpu | 512 | 38.61 [16.72, 49.86] | 5.0 | 16.1 |
| sdpa_efficient | flash_cpu | 1024 | 65.23 [43.23, 101.80] | 10.0 | 34.4 |
| sdpa_efficient | flash_cpu | 2048 | 99.09 [93.92, 106.06] | 20.0 | 64.3 |
| sdpa_efficient | flash_cpu | 4096 | 256.57 [250.71, 259.97] | 40.0 | 124.1 |

### Throughput

| mode | batch x seq | TFLOP/s (95% CI) | tokens/s | MFU vs measured CPU peak |
|---|---|---|---|---|
| forward | 2x512 | 0.0782 [0.0646, 0.0895] | 1113.5 | 25.2% |
| forward_backward | 2x512 | 0.0746 [0.0735, 0.0753] | 354.1 | 24.1% |
