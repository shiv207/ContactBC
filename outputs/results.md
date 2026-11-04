Success over 50 held-out rollouts per seed, mean ± std over 3 training seeds.

| variant | training data | contact-gated loss | success | grasp held | placed | per-seed success |
|---|---|---|---|---|---|---|
| `plain` | 3 direct human replays | no | **0 ± 0%** | 0 ± 0% | 0 ± 0% | 0/50, 0/50, 0/50 |
| `plain+cg` | 3 direct human replays | yes (all dims) | **0 ± 0%** | 1 ± 1% | 0 ± 0% | 0/50, 0/50, 0/50 |
| `aug` | 500 augmented episodes | no | **28 ± 3%** | 83 ± 5% | 29 ± 2% | 13/50, 13/50, 16/50 |
| `aug+cg` | 500 augmented episodes | yes (all dims) | **14 ± 2%** | 75 ± 5% | 15 ± 1% | 7/50, 6/50, 8/50 |
| `aug+cg-hand` | 500 augmented episodes | yes (finger dims) | **22 ± 6%** | 71 ± 2% | 23 ± 7% | 10/50, 8/50, 15/50 |
