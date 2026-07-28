# Complexity report

Threshold: **5** (tasks scoring ≥ 5 are flagged for expansion). Codebase-aware scoring against
the merged M1–M1.6 tree; tasks 12–13 scored on addition (review follow-up to PR #5). **4 of 13
tasks flagged.**

| ID | Title | Complexity | Recommended subtasks | Expand? |
|----|-------|------------|----------------------|---------|
| 7 | Build the D5 residual network | 6 | 4 | yes |
| 3 | Implement Blokus 46-plane encode_state | 5 | 3 | yes |
| 8 | Implement sparse policy and composite loss | 5 | 3 | yes |
| 9 | Implement the AMP train step | 5 | 3 | yes |
| 6 | Add the pipeline decode test | 4 | 0 | no |
| 10 | Add the overfit-one-batch test | 4 | 0 | no |
| 11 | Promote the encoding surface to abstract methods | 4 | 0 | no |
| 12 | Bridge the network into MCTS leaf evaluation | 4 | 0 | no |
| 2 | Add NumPy and PyTorch dependencies | 3 | 0 | no |
| 4 | Implement Blokus plane-side symmetry transform | 3 | 0 | no |
| 1 | Pin λ_aux doc-first and replace the NaN sentinel | 2 | 0 | no |
| 5 | Add training-side symmetry augmentation | 2 | 0 | no |
| 13 | Benchmark D5 batch sizes 128/256/512 on the 4060 Ti | 2 | 0 | no |

The flagged four are the production-code core of M2: the encoding projection (3), the network
(7), the loss (8), and the train step (9). The test-only tasks (6, 10), the config/doc tasks
(1, 2), and the review-added glue/benchmark tasks (12, 13) stay atomic — task 12 is ~30 lines
whose risk lives in sign/normalization conventions, not decomposable structure, and task 13 is
a single script plus its committed artifact. Expand task 7 first — it is both the
highest-scored and the one whose flatten-order contract everything downstream gathers against.
