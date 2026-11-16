### Relevance of the column's own metric (threshold 0.5)

| Column | Pairs | Mean r_own | % r_own >= threshold | % own metric most relevant (ties count) |
|---|---|---|---|---|
| Aesthetic | 262 | 1.000 | 100.00 | 44.66 |
| Realism | 253 | 0.865 | 95.26 | 0.00 |
| Fidelity | 279 | 1.000 | 100.00 | 46.24 |
| BC | 254 | 0.813 | 87.01 | 0.00 |
| FC | 266 | 0.706 | 100.00 | 0.00 |
| SC | 247 | 0.363 | 32.79 | 0.00 |

### Head of the column's metric, pairs split by r_own (Acc \| Margin AUC on non-tie pairs, Tie Head AUC on all pairs; %)

| Column | Group | Pairs (non-tie / tie) | Acc | Margin AUC | Tie Head AUC |
|---|---|---|---|---|---|
| Aesthetic | all | 262 (209 / 53) | 55.98 | 60.04 | 71.55 |
| Aesthetic | >= 0.5 | 262 (209 / 53) | 55.98 | 60.04 | 71.55 |
| Aesthetic | < 0.5 | 0 (0 / 0) | - | - | - |
| Realism | all | 253 (185 / 68) | 63.24 | 66.94 | 57.22 |
| Realism | >= 0.5 | 241 (176 / 65) | 63.07 | 67.24 | 57.69 |
| Realism | < 0.5 | 12 (9 / 3) | 66.67 | 59.26 | 40.74 |
| Fidelity | all | 279 (239 / 40) | 61.51 | 66.54 | 71.97 |
| Fidelity | >= 0.5 | 279 (239 / 40) | 61.51 | 66.54 | 71.97 |
| Fidelity | < 0.5 | 0 (0 / 0) | - | - | - |
| BC | all | 254 (217 / 37) | 61.75 | 65.72 | 68.02 |
| BC | >= 0.5 | 221 (186 / 35) | 60.22 | 64.48 | 67.14 |
| BC | < 0.5 | 33 (31 / 2) | 70.97 | 69.93 | 67.74 |
| FC | all | 266 (235 / 31) | 62.55 | 70.95 | 65.07 |
| FC | >= 0.5 | 266 (235 / 31) | 62.55 | 70.95 | 65.07 |
| FC | < 0.5 | 0 (0 / 0) | - | - | - |
| SC | all | 247 (180 / 67) | 65.00 | 71.24 | 52.31 |
| SC | >= 0.5 | 81 (65 / 16) | 70.77 | 78.11 | 56.15 |
| SC | < 0.5 | 166 (115 / 51) | 61.74 | 67.18 | 50.69 |

### Mean relevance of every Student head per column

| Column | aesthetic | fidelity | realism | background_consistency | foreground_consistency | structure_consistency |
|---|---|---|---|---|---|---|
| Aesthetic | 1.000 | 1.000 | 0.804 | 0.715 | 0.701 | 0.268 |
| Realism | 1.000 | 1.000 | 0.865 | 0.773 | 0.686 | 0.211 |
| Fidelity | 1.000 | 1.000 | 0.802 | 0.714 | 0.697 | 0.267 |
| BC | 1.000 | 1.000 | 0.853 | 0.813 | 0.636 | 0.146 |
| FC | 1.000 | 1.000 | 0.858 | 0.781 | 0.706 | 0.211 |
| SC | 1.000 | 1.000 | 0.732 | 0.629 | 0.754 | 0.363 |
