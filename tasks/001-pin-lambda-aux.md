---
id: 1
title: Pin λ_aux doc-first and replace the NaN sentinel
status: pending
priority: high
dependencies: []
complexity: 2
recommended_subtasks: 0
---

## Description
Commit a concrete value for the aux-loss weight `λ_aux` in the design doc (§7), then replace the
NaN sentinel in `games/blokus_duo/targets.py` with the pinned value. The overfit-one-batch test
(task 10) cannot run without it.

## Details
Doc-first, per the project working principle — the doc amendment precedes the code change:

1. Amend `metadocs/blokus-duo-az-design-v0_5.md` §7: the current text reads "The aux-term weight
   `λ_aux` is an unpinned config scalar — to pin doc-first at M2". Replace with a committed value
   and one sentence of rationale. The doc constrains the choice: the KataGo lineage implies a
   *small* value, and it is explicitly *not* an implicit 1.0. A defensible default is
   `λ_aux = 0.25`; the human owner signs off on the number before it lands (this is a settled-
   decision amendment, not a code detail).
2. Update `games/blokus_duo/targets.py::value_target_spec()` — currently returns
   `aux_loss_weights=(float("nan"),)` as the deliberate sentinel. Substitute the pinned value.
3. Update `tests/test_blokus_targets.py` — it currently asserts
   `math.isnan(spec.aux_loss_weights[0])`; change to an equality golden against the pinned value.

## Test Strategy
`python3 -m pytest tests/test_blokus_targets.py` passes with the new golden; grep the design doc
for the pinned value to confirm doc and code agree; full battery stays green.

## Complexity Analysis
A one-line doc amendment, a one-tuple code change, and one test-golden update — all three sites
are already located (`§7`, `targets.py::value_target_spec`, `test_blokus_targets.py`). The only
non-mechanical element is the human sign-off on the value, which is a decision gate, not
implementation effort.

**Suggested expansion approach:** none — already atomic; splitting a three-line change would be
overhead.
