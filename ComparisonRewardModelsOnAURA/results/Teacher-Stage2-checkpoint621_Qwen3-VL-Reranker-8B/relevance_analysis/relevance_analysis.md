### Relevance of the column's own metric (threshold 0.5)

| Column | Pairs | Mean r_own | % r_own >= threshold | % own metric most relevant (ties count) |
|---|---|---|---|---|
| Aesthetic | 262 | 1.000 | 100.00 | 61.07 |
| Realism | 253 | 0.868 | 94.47 | 0.00 |
| Fidelity | 279 | 1.000 | 100.00 | 32.62 |
| BC | 254 | 0.797 | 86.61 | 0.00 |
| FC | 266 | 0.693 | 100.00 | 0.00 |
| SC | 247 | 0.349 | 32.39 | 0.00 |

### Head of the column's metric, pairs split by r_own (Acc \| Margin AUC on non-tie pairs, Tie Head AUC on all pairs; %)

| Column | Group | Pairs (non-tie / tie) | Acc | Margin AUC | Tie Head AUC |
|---|---|---|---|---|---|
| Aesthetic | all | 262 (209 / 53) | 54.55 | 59.46 | 72.37 |
| Aesthetic | >= 0.5 | 262 (209 / 53) | 54.55 | 59.46 | 72.37 |
| Aesthetic | < 0.5 | 0 (0 / 0) | - | - | - |
| Realism | all | 253 (185 / 68) | 63.24 | 67.35 | 56.30 |
| Realism | >= 0.5 | 239 (174 / 65) | 63.22 | 67.84 | 56.02 |
| Realism | < 0.5 | 14 (11 / 3) | 63.64 | 57.02 | 51.52 |
| Fidelity | all | 279 (239 / 40) | 60.25 | 65.71 | 76.61 |
| Fidelity | >= 0.5 | 279 (239 / 40) | 60.25 | 65.71 | 76.61 |
| Fidelity | < 0.5 | 0 (0 / 0) | - | - | - |
| BC | all | 254 (217 / 37) | 58.99 | 64.53 | 70.53 |
| BC | >= 0.5 | 220 (186 / 34) | 58.06 | 63.87 | 70.83 |
| BC | < 0.5 | 34 (31 / 3) | 64.52 | 66.39 | 63.44 |
| FC | all | 266 (235 / 31) | 62.98 | 71.03 | 63.27 |
| FC | >= 0.5 | 266 (235 / 31) | 62.98 | 71.03 | 63.27 |
| FC | < 0.5 | 0 (0 / 0) | - | - | - |
| SC | all | 247 (180 / 67) | 66.11 | 72.75 | 55.46 |
| SC | >= 0.5 | 80 (64 / 16) | 68.75 | 79.05 | 64.26 |
| SC | < 0.5 | 167 (116 / 51) | 64.66 | 69.31 | 53.08 |

### Mean relevance of every Student head per column

| Column | aesthetic | fidelity | realism | background_consistency | foreground_consistency | structure_consistency |
|---|---|---|---|---|---|---|
| Aesthetic | 1.000 | 1.000 | 0.811 | 0.708 | 0.691 | 0.264 |
| Realism | 1.000 | 1.000 | 0.868 | 0.764 | 0.684 | 0.215 |
| Fidelity | 1.000 | 1.000 | 0.809 | 0.706 | 0.691 | 0.265 |
| BC | 1.000 | 1.000 | 0.859 | 0.797 | 0.645 | 0.161 |
| FC | 1.000 | 1.000 | 0.863 | 0.772 | 0.693 | 0.211 |
| SC | 1.000 | 1.000 | 0.746 | 0.626 | 0.732 | 0.349 |
