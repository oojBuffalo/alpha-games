# M2 train-step benchmark — D5 batches 128/256/512

D5 pins "batch 256 (benchmark 128/256/512)" (§7) on the RTX 4060 Ti 16 GB; this
artifact is the measurement behind that pin. Produced by
`python3 scripts/bench_train_step.py --out docs/bench/m2-train-step.md`.

- **GPU:** NVIDIA GeForce RTX 4060 Ti, 16.0 GiB
- **torch:** 2.13.0+cu130 (CUDA 13.0)
- **Date:** 2026-08-10
- **Config:** seed 0; 10 warmup + 50 timed steps per batch size; one fixed synthetic batch per size reused every step; legal-set sizes from 215 plies of 8 seeded random playouts (min/median/max 1/195/828); visit counts 512-trial multinomial.

| batch | mean ms/step | median | min | max | positions/s | peak alloc (GiB) | peak reserved (GiB) |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 128 | 26.8 | 24.2 | 21.9 | 45.2 | 4,777 | 0.28 | 0.35 |
| 256 | 41.9 | 39.5 | 38.2 | 56.7 | 6,116 | 0.52 | 0.62 |
| 512 | 90.7 | 89.3 | 87.8 | 93.5 | 5,642 | 1.02 | 1.19 |

**Methodology.** Real `core.train.train_step` (AMP autocast + GradScaler on CUDA) on a fresh full-size D5 net and optimizer per batch size; per-step times from CUDA events; positions/s from a wall clock bracketed by `torch.cuda.synchronize`; peak VRAM from `torch.cuda.max_memory_allocated` after a post-warmup `reset_peak_memory_stats` (resident params/batch stay counted).

**Summary:** Batch 256 (the D5 pin) peak-allocated 0.52 of 16.0 GiB device memory (3%) and sustained 6,116 positions/s (100% of the best measured, 6,116 at batch 256). D5 stays pinned; M3 argues from these numbers (any re-pin is doc-first).
