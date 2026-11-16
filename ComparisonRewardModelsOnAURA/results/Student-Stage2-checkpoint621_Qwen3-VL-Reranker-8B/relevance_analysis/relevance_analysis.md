### Relevance of the column's own metric (threshold 0.5)

| Column | Pairs | Mean r_own | % r_own >= threshold | % own metric most relevant (ties count) |
|---|---|---|---|---|
| Aesthetic | 262 | 1.000 | 100.00 | 30.53 |
| Realism | 253 | 0.862 | 96.05 | 0.00 |
| Fidelity | 279 | 1.000 | 100.00 | 63.08 |
| BC | 254 | 0.803 | 87.80 | 0.00 |
| FC | 266 | 0.795 | 100.00 | 0.00 |
| SC | 247 | 0.436 | 39.68 | 0.00 |

### Head of the column's metric, pairs split by r_own (Acc \| Margin AUC on non-tie pairs, Tie Head AUC on all pairs; %)

| Column | Group | Pairs (non-tie / tie) | Acc | Margin AUC | Tie Head AUC |
|---|---|---|---|---|---|
| Aesthetic | all | 262 (209 / 53) | 57.42 | 60.78 | 66.06 |
| Aesthetic | >= 0.5 | 262 (209 / 53) | 57.42 | 60.78 | 66.06 |
| Aesthetic | < 0.5 | 0 (0 / 0) | - | - | - |
| Realism | all | 253 (185 / 68) | 63.78 | 69.42 | 53.16 |
| Realism | >= 0.5 | 243 (177 / 66) | 63.84 | 70.09 | 53.83 |
| Realism | < 0.5 | 10 (8 / 2) | 62.50 | 60.94 | 31.25 |
| Fidelity | all | 279 (239 / 40) | 63.18 | 67.74 | 71.57 |
| Fidelity | >= 0.5 | 279 (239 / 40) | 63.18 | 67.74 | 71.57 |
| Fidelity | < 0.5 | 0 (0 / 0) | - | - | - |
| BC | all | 254 (217 / 37) | 62.21 | 64.71 | 71.20 |
| BC | >= 0.5 | 223 (188 / 35) | 60.64 | 64.01 | 69.62 |
| BC | < 0.5 | 31 (29 / 2) | 72.41 | 65.99 | 94.83 |
| FC | all | 266 (235 / 31) | 63.40 | 69.88 | 62.84 |
| FC | >= 0.5 | 266 (235 / 31) | 63.40 | 69.88 | 62.84 |
| FC | < 0.5 | 0 (0 / 0) | - | - | - |
| SC | all | 247 (180 / 67) | 58.33 | 66.87 | 53.96 |
| SC | >= 0.5 | 98 (77 / 21) | 62.34 | 72.36 | 54.24 |
| SC | < 0.5 | 149 (103 / 46) | 55.34 | 63.05 | 52.47 |

### Mean relevance of every Student head per column

| Column | aesthetic | fidelity | realism | background_consistency | foreground_consistency | structure_consistency |
|---|---|---|---|---|---|---|
| Aesthetic | 1.000 | 1.000 | 0.797 | 0.697 | 0.777 | 0.316 |
| Realism | 1.000 | 1.000 | 0.862 | 0.754 | 0.754 | 0.250 |
| Fidelity | 1.000 | 1.000 | 0.788 | 0.694 | 0.769 | 0.321 |
| BC | 1.000 | 1.000 | 0.832 | 0.803 | 0.684 | 0.155 |
| FC | 1.000 | 1.000 | 0.848 | 0.763 | 0.795 | 0.259 |
| SC | 1.000 | 1.000 | 0.718 | 0.608 | 0.849 | 0.436 |
